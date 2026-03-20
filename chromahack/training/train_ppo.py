"""Train PPO against the proxy reward model."""

from __future__ import annotations

import argparse
import math
import os
from types import SimpleNamespace

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from chromahack.data.generate_dataset import SyntheticDatasetGenerator
from chromahack.envs.chroma_env import ChromaHackEnv
from chromahack.metrics.hacking_metrics import HackingCallback
from chromahack.models.reward_cnn import ProxyRewardFunction, load_proxy_reward, save_proxy_checkpoint
from chromahack.training.train_proxy_cnn import run as run_proxy_training


def make_chroma_env(proxy_fn, rank: int = 0, seed: int = 42):
    def _init():
        env = ChromaHackEnv(render_mode="rgb_array", seed=seed + rank)
        env.set_proxy_fn(proxy_fn)
        return env

    return _init


def _ensure_dataset(args) -> None:
    dataset_dir = args.dataset_dir or os.path.join(args.out_dir, "data", "synthetic")
    args.dataset_dir = dataset_dir
    dataset_path = os.path.join(dataset_dir, "dataset.pkl")
    if os.path.exists(dataset_path):
        return

    generator = SyntheticDatasetGenerator(fragility=args.fragility, base_seed=args.seed, out_dir=dataset_dir)
    if args.n_ordered is not None:
        generator.cfg["n_ordered"] = args.n_ordered
    if args.n_disordered is not None:
        generator.cfg["n_disordered"] = args.n_disordered
    if args.n_partial is not None:
        generator.cfg["n_partial"] = args.n_partial
    if args.n_adversarial is not None:
        generator.cfg["n_adversarial"] = args.n_adversarial
    generator.generate(verbose=True)
    generator.save()


def _load_or_train_proxy(args, device: str):
    if args.proxy_path:
        proxy_fn, mode = load_proxy_reward(args.proxy_path, device=device)
        save_proxy_checkpoint(proxy_fn.model, os.path.join(args.out_dir, "proxy_cnn.pth"))
        return proxy_fn, mode

    _ensure_dataset(args)
    proxy_args = SimpleNamespace(
        mode=args.mode,
        pretrained_path=args.pretrained_path,
        freeze_backbone=args.freeze_backbone,
        fragility=args.fragility,
        epochs=args.cnn_epochs,
        batch_size=args.proxy_batch_size,
        lr=args.cnn_lr,
        no_augment=args.no_augment,
        gradcam=False,
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        n_ordered=args.n_ordered,
        n_disordered=args.n_disordered,
        n_partial=args.n_partial,
        n_adversarial=args.n_adversarial,
        force_augment=False,
    )
    proxy_path = run_proxy_training(proxy_args)
    proxy_fn, mode = load_proxy_reward(proxy_path, device=device)
    return proxy_fn, mode


def _resolve_batch_size(total_batch: int) -> int:
    batch_size = min(64, total_batch)
    while batch_size > 1 and total_batch % batch_size != 0:
        batch_size -= 1
    return max(1, batch_size)


def run(args) -> str:
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy_fn, proxy_mode = _load_or_train_proxy(args, device)
    print(f"[PPO] proxy_mode={proxy_mode}")

    vec_env = DummyVecEnv([make_chroma_env(proxy_fn, rank=index, seed=args.seed) for index in range(args.n_envs)])
    vec_env = VecTransposeImage(vec_env)

    total_batch = args.n_steps * args.n_envs
    agent = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=args.ppo_lr,
        n_steps=args.n_steps,
        batch_size=_resolve_batch_size(total_batch),
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(args.out_dir, "tb_logs"),
        seed=args.seed,
        device=device,
    )

    callbacks = [HackingCallback(log_freq=max(args.n_steps * args.n_envs, 1000), verbose=1)]
    if args.save_checkpoints:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.total_steps // 10, 1),
                save_path=os.path.join(args.out_dir, "checkpoints"),
                name_prefix="ppo_chroma",
            )
        )

    agent.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=False)
    final_path = os.path.join(args.out_dir, "ppo_final")
    agent.save(final_path)
    vec_env.close()
    print(f"[PPO] wrote {final_path}.zip")
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the ChromaHack environment.")
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "resnet"])
    parser.add_argument("--proxy_path", type=str, default=None, help="Existing proxy checkpoint to load.")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--fragility", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--cnn_epochs", type=int, default=20)
    parser.add_argument("--cnn_lr", type=float, default=1e-3)
    parser.add_argument("--proxy_batch_size", type=int, default=32)
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--n_ordered", type=int, default=None)
    parser.add_argument("--n_disordered", type=int, default=None)
    parser.add_argument("--n_partial", type=int, default=None)
    parser.add_argument("--n_adversarial", type=int, default=None)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="runs/exp_001")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
