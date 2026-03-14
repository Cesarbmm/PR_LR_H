"""
train_ppo.py — Entrenamiento del agente con PPO (SB3)
======================================================
Flujo completo:
  1. Genera dataset sintético del entorno
  2. Entrena la CNN proxy (frágil por diseño)
  3. Inyecta la CNN como función de recompensa en el entorno
  4. Entrena el agente PPO con la recompensa proxy
  5. Guarda checkpoints y métricas para eval_hidden.py

Ejecución:
  python training/train_ppo.py --mode tiny --steps 200000
  python training/train_ppo.py --mode resnet --steps 500000
"""

import argparse
import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import gymnasium as gym

# Paths relativos al proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.chroma_env import ChromaHackEnv
from models.reward_cnn import TinyCNN, ResNetProxy, ProxyRewardFunction
from models.reward_cnn import generate_synthetic_dataset, train_proxy_cnn
from metrics.hacking_metrics import HackingMetricsLogger


# ──────────────────────────────────────────────
# Callback para loggear métricas anti-hacking
# ──────────────────────────────────────────────
class HackingCallback(BaseCallback):
    """
    En cada rollout, extrae r_proxy y r_true del info dict
    y loggea el gap (métrica principal de reward hacking).

    Sube métricas a TensorBoard automáticamente.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._proxy_buf = []
        self._true_buf  = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "r_proxy" in info:
                self._proxy_buf.append(info["r_proxy"])
                self._true_buf.append(info["r_true"])

        if self.num_timesteps % self.log_freq == 0 and self._proxy_buf:
            mean_proxy = np.mean(self._proxy_buf)
            mean_true  = np.mean(self._true_buf)
            gap        = mean_proxy - mean_true

            # Loggear en TensorBoard via SB3
            self.logger.record("hacking/r_proxy_mean", mean_proxy)
            self.logger.record("hacking/r_true_mean",  mean_true)
            self.logger.record("hacking/gap_proxy_true", gap)
            self.logger.record("hacking/ratio",
                               mean_proxy / (mean_true + 1e-6))

            if self.verbose > 0:
                print(f"[{self.num_timesteps}] proxy={mean_proxy:.3f} "
                      f"true={mean_true:.3f} GAP={gap:.3f}")

            self._proxy_buf.clear()
            self._true_buf.clear()

        return True  # continuar entrenamiento


# ──────────────────────────────────────────────
# Wrapper: SB3 requiere Env que use la proxy fn
# ──────────────────────────────────────────────
def make_chroma_env(proxy_fn, rank: int = 0, seed: int = 42):
    """
    Factory para make_vec_env. Cada worker tiene su propio env
    con la misma CNN inyectada (la CNN está en CPU o GPU compartida).
    """
    def _init():
        env = ChromaHackEnv(render_mode="rgb_array", seed=seed + rank)
        env.set_proxy_fn(proxy_fn)
        return env
    return _init


# ──────────────────────────────────────────────
# Entrenamiento principal
# ──────────────────────────────────────────────
def run_training(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    # ── Paso 1: Dataset sintético ──────────────
    print("\n[1/4] Generando dataset sintético...")
    frames, labels = generate_synthetic_dataset(
        ChromaHackEnv,
        n_ordered=args.n_ordered,
        n_disordered=args.n_disordered,
        save_dir=os.path.join(args.out_dir, "data/synthetic"),
    )

    # ── Paso 2: Entrenar CNN proxy ─────────────
    print(f"\n[2/4] Entrenando CNN proxy ({args.mode})...")
    if args.mode == "tiny":
        model = TinyCNN()
    elif args.mode == "resnet":
        model = ResNetProxy(pretrained_backbone="resnet18", freeze_backbone=True)
        print("  [ResNet] Fase 1: backbone congelado, entrenando solo cabeza")
    else:
        raise ValueError(f"Modo desconocido: {args.mode}")

    proxy_history = train_proxy_cnn(
        model, frames, labels,
        epochs=args.cnn_epochs, lr=args.cnn_lr, device=device
    )
    np.save(os.path.join(args.out_dir, "cnn_train_history.npy"), proxy_history)

    if args.mode == "resnet" and args.cnn_epochs > 10:
        # Fase 2: descongelar últimas capas y re-entrenar con lr menor
        print("  [ResNet] Fase 2: fine-tune últimas 2 capas")
        model.unfreeze_last_layers(2)
        train_proxy_cnn(model, frames, labels,
                        epochs=args.cnn_epochs // 2, lr=args.cnn_lr * 0.1,
                        device=device)

    proxy_fn = ProxyRewardFunction(model, device=device)
    proxy_fn.save(os.path.join(args.out_dir, "proxy_cnn.pth"))

    # ── Paso 3: Crear envs vectorizados ────────
    print(f"\n[3/4] Creando {args.n_envs} entornos paralelos...")
    env_fns = [make_chroma_env(proxy_fn, rank=i, seed=args.seed)
               for i in range(args.n_envs)]
    vec_env = make_vec_env(
        env_id=lambda: ChromaHackEnv(render_mode="rgb_array"),
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={},
    )
    # Nota: make_vec_env no admite proxy_fn directamente;
    # usamos el constructor manual:
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([make_chroma_env(proxy_fn, i, args.seed)
                           for i in range(args.n_envs)])
    vec_env = VecTransposeImage(vec_env)  # (H,W,C) → (C,H,W) para PyTorch

    # ── Paso 4: Entrenar agente PPO ────────────
    print(f"\n[4/4] Entrenando agente PPO ({args.total_steps} pasos)...")

    # Política CNN pequeña para el agente (no confundir con el proxy CNN)
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    )

    agent = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=args.ppo_lr,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # Entropía: incentiva exploración
        verbose=1,
        tensorboard_log=os.path.join(args.out_dir, "tb_logs"),
        seed=args.seed,
        device=device,
    )

    callbacks = [
        HackingCallback(log_freq=2000, verbose=1),
        CheckpointCallback(
            save_freq=max(args.total_steps // 10, 1),
            save_path=os.path.join(args.out_dir, "checkpoints"),
            name_prefix="ppo_chroma",
        ),
    ]

    agent.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=True,
    )

    agent.save(os.path.join(args.out_dir, "ppo_final"))
    vec_env.close()

    print(f"\n[Done] Artefactos guardados en: {args.out_dir}")
    print("  Visualiza con: tensorboard --logdir", os.path.join(args.out_dir, "tb_logs"))
    print("  Evalúa con:    python eval/eval_hidden.py --model_dir", args.out_dir)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaHack PPO Training")
    parser.add_argument("--mode",        type=str, default="tiny",
                        choices=["tiny", "resnet"],
                        help="Arquitectura del proxy CNN")
    parser.add_argument("--total_steps", type=int, default=200_000,
                        help="Pasos totales de entrenamiento PPO")
    parser.add_argument("--n_envs",      type=int, default=4,
                        help="Número de entornos paralelos")
    parser.add_argument("--cnn_epochs",  type=int, default=20,
                        help="Épocas para entrenar el proxy CNN")
    parser.add_argument("--cnn_lr",      type=float, default=1e-3,
                        help="Learning rate del proxy CNN")
    parser.add_argument("--ppo_lr",      type=float, default=3e-4,
                        help="Learning rate del agente PPO")
    parser.add_argument("--n_ordered",   type=int, default=500,
                        help="Frames ordenados para dataset sintético")
    parser.add_argument("--n_disordered",type=int, default=500,
                        help="Frames desordenados para dataset sintético")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out_dir",     type=str, default="runs/exp_001")
    args = parser.parse_args()

    run_training(args)
