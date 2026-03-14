"""
preference_reward_model.py — Intervención alineadora por preferencias
======================================================================
Esta es la INTERVENCIÓN que cierra el ciclo de la investigación.

Problema demostrado:  el agente PPO maximiza la CNN proxy (R_proxy)
                      sin mejorar R* (orden real) → reward hacking.

Solución propuesta:   aprender una nueva función de recompensa R_pref
                      a partir de comparaciones de segmentos de trayectoria,
                      donde un "oráculo" (basado en R*) prefiere las que
                      realmente ordenan el tablero.

Marco teórico:
  Christiano et al. (2017) "Deep RL from Human Preferences"
  La idea: dado dos segmentos (s1, s2), el humano/oráculo elige cuál
  prefiere. El reward model aprende P(s1 > s2) con una red neuronal.

Resultado esperado después de la intervención:
  - J*(π_pref) >> J*(π_hack)   ← el agente nuevo ordena más
  - Gap proxy-true → 0         ← no hay más hacking
  - La figura "antes vs después" es la contribución central del paper

Uso:
  # Paso 1: Recolectar trayectorias del agente hackeador
  python intervention/preference_reward_model.py collect \\
      --agent_path runs/exp_001/ppo_final.zip \\
      --out_dir runs/exp_001/trajectories

  # Paso 2: Generar pares de preferencia (con oráculo R*)
  python intervention/preference_reward_model.py label \\
      --traj_dir runs/exp_001/trajectories

  # Paso 3: Entrenar el reward model de preferencias
  python intervention/preference_reward_model.py train \\
      --traj_dir runs/exp_001/trajectories \\
      --out_dir runs/exp_001/pref_model

  # Paso 4: Re-entrenar el agente con el reward aprendido
  python intervention/preference_reward_model.py retrain \\
      --pref_model_path runs/exp_001/pref_model/pref_reward.pth \\
      --out_dir runs/exp_001/ppo_aligned
"""

import os, sys, json, pickle, argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.chroma_env import ChromaHackEnv
from models.reward_cnn import PROXY_TRANSFORM, ProxyRewardFunction, TinyCNN
import torchvision.transforms as T


# ─────────────────────────────────────────────────────
# 1. MODELO DE RECOMPENSA POR PREFERENCIAS
# ─────────────────────────────────────────────────────

class PreferenceRewardModel(nn.Module):
    """
    Red que aprende r_pref(frame) a partir de comparaciones.

    Arquitectura: CNN compartida (backbone) → escalar de reward.
    Entrenamiento: dado dos segmentos (τ1, τ2), optimiza:
        L = -[μ * log P(τ1>τ2) + (1-μ) * log P(τ2>τ1)]
    donde μ=1 si el oráculo prefiere τ1, μ=0 si prefiere τ2.

    Esta es exactamente la pérdida de Christiano et al. 2017.
    """
    def __init__(self):
        super().__init__()
        # Backbone compartido (mismo que TinyCNN para consistencia)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Cabeza de reward (sin sigmoid — salida sin acotar, como en el paper)
        self.reward_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def reward(self, frames_batch):
        """frames_batch: (B, 3, 64, 64) → rewards: (B,)"""
        return self.reward_head(self.encoder(frames_batch)).squeeze(-1)

    def forward(self, seg1_frames, seg2_frames):
        """
        Calcula P(τ1 > τ2) = σ(R(τ1) - R(τ2))
        donde R(τ) = Σ_t r(s_t)
        """
        r1 = self.reward(seg1_frames).sum()
        r2 = self.reward(seg2_frames).sum()
        return torch.sigmoid(r1 - r2)   # prob de que τ1 sea preferida


# ─────────────────────────────────────────────────────
# 2. DATASET DE PREFERENCIAS
# ─────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    """
    Pares (segmento1, segmento2, preferencia) donde:
      preferencia = 1  → segmento1 es mejor (oráculo prefiere τ1)
      preferencia = 0  → segmento2 es mejor
      preferencia = 0.5 → empate (ambos similares)
    """
    def __init__(self, pairs):
        self.pairs = pairs   # list of (frames1, frames2, label)
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((64, 64)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frames1, frames2, label = self.pairs[idx]
        t1 = torch.stack([self.transform(f) for f in frames1])
        t2 = torch.stack([self.transform(f) for f in frames2])
        return t1, t2, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────
# 3. RECOLECCIÓN DE TRAYECTORIAS
# ─────────────────────────────────────────────────────

def collect_trajectories(agent_path: str, out_dir: str,
                          n_episodes: int = 200, segment_len: int = 25,
                          seed: int = 42):
    """
    Ejecuta el agente hackeador y guarda segmentos de trayectoria
    con sus R* asociados (para el oráculo de preferencias).

    Cada segmento = lista de (frame, r_proxy, r_true, action)
    """
    from stable_baselines3 import PPO

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Collect] Cargando agente desde {agent_path}...")
    agent = PPO.load(agent_path, device=device)

    env = ChromaHackEnv(render_mode="rgb_array", seed=seed)
    # Sin proxy CNN inyectada — solo recolectamos frames y R*
    env.set_proxy_fn(lambda f: 0.0)

    segments = []
    print(f"[Collect] Recolectando {n_episodes} episodios...")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_buffer = []
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=False)
            next_obs, _, terminated, truncated, info = env.step(int(action))
            ep_buffer.append({
                "frame":   obs.copy(),
                "r_true":  info["r_true"],
                "action":  int(action),
            })
            obs = next_obs

        # Dividir episodio en segmentos de longitud fija
        for start in range(0, len(ep_buffer) - segment_len, segment_len // 2):
            seg = ep_buffer[start:start + segment_len]
            if len(seg) == segment_len:
                segments.append({
                    "frames":     [s["frame"]  for s in seg],
                    "r_true_sum": sum(s["r_true"] for s in seg),
                    "mean_r_true": np.mean([s["r_true"] for s in seg]),
                })

        if (ep + 1) % 50 == 0:
            print(f"  Ep {ep+1}/{n_episodes} | Segmentos: {len(segments)}")

    env.close()

    path = os.path.join(out_dir, "segments.pkl")
    with open(path, "wb") as f:
        pickle.dump(segments, f)

    print(f"\n[Collect] {len(segments)} segmentos guardados en {path}")
    r_true_values = [s["mean_r_true"] for s in segments]
    print(f"  R* medio por segmento: {np.mean(r_true_values):.3f} "
          f"± {np.std(r_true_values):.3f}")
    print(f"  (Si la media es baja → el agente está hackeando)")
    return segments


# ─────────────────────────────────────────────────────
# 4. GENERACIÓN DE PARES DE PREFERENCIA (ORÁCULO R*)
# ─────────────────────────────────────────────────────

def generate_preference_pairs(segments: list, n_pairs: int = 2000,
                               preference_threshold: float = 0.15,
                               seed: int = 42) -> list:
    """
    Genera pares (τ1, τ2) con etiqueta del oráculo basada en R*.

    Lógica del oráculo sintético:
      Si R*(τ1) - R*(τ2) > threshold → preferencia = 1 (τ1 mejor)
      Si R*(τ2) - R*(τ1) > threshold → preferencia = 0 (τ2 mejor)
      Si |R*(τ1) - R*(τ2)| <= threshold → preferencia = 0.5 (empate)

    threshold controla cuánto overlap hay en las preferencias.
    Un threshold alto = preferencias más claras = reward model más fácil.
    Un threshold bajo = más empates = reward model más difícil de aprender.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    n_segs = len(segments)

    print(f"[Label] Generando {n_pairs} pares de preferencia...")
    print(f"  threshold R* = {preference_threshold}")

    counts = {1: 0, 0: 0, "tie": 0}

    while len(pairs) < n_pairs:
        i, j = rng.choice(n_segs, 2, replace=False)
        s1, s2 = segments[i], segments[j]

        diff = s1["mean_r_true"] - s2["mean_r_true"]

        if diff > preference_threshold:
            label = 1.0    # τ1 claramente mejor
            counts[1] += 1
        elif diff < -preference_threshold:
            label = 0.0    # τ2 claramente mejor
            counts[0] += 1
        else:
            label = 0.5    # empate / ruido
            counts["tie"] += 1

        pairs.append((s1["frames"], s2["frames"], label))

    print(f"  Pares τ1>τ2: {counts[1]} | τ2>τ1: {counts[0]} | empates: {counts['tie']}")
    print(f"  Balance: {counts[1]/(counts[1]+counts[0]+1e-6):.1%} preferencias claras")
    return pairs


# ─────────────────────────────────────────────────────
# 5. ENTRENAMIENTO DEL REWARD MODEL
# ─────────────────────────────────────────────────────

def train_preference_model(pairs: list, out_dir: str,
                            epochs: int = 30, lr: float = 3e-4,
                            batch_size: int = 16, seed: int = 42,
                            device: str = "auto") -> PreferenceRewardModel:
    """
    Entrena el reward model con pérdida de preferencias (cross-entropy).
    Solo aprende de pares con preferencia clara (excluye empates).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)

    # Filtrar empates para el entrenamiento
    clear_pairs = [(f1, f2, l) for f1, f2, l in pairs if l != 0.5]
    print(f"\n[Train PrefModel] Pares claros: {len(clear_pairs)}/{len(pairs)}")

    # Split train/val
    random.seed(seed)
    random.shuffle(clear_pairs)
    split = int(0.85 * len(clear_pairs))
    train_pairs = clear_pairs[:split]
    val_pairs   = clear_pairs[split:]

    train_ds = PreferenceDataset(train_pairs)
    val_ds   = PreferenceDataset(val_pairs)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = PreferenceRewardModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Pérdida: cross-entropy sobre preferencias (Christiano et al. Eq. 1)
    def preference_loss(pred_prob, label):
        label = label.to(device)
        return -( label * torch.log(pred_prob + 1e-8)
                + (1 - label) * torch.log(1 - pred_prob + 1e-8) ).mean()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_path = os.path.join(out_dir, "pref_reward_best.pth")

    print(f"[Train PrefModel] {epochs} épocas | device={device}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for seg1, seg2, label in train_ld:
            B, T, C, H, W = seg1.shape
            # Flatten batch×time para pasar por la red
            s1 = seg1.view(B * T, C, H, W).to(device)
            s2 = seg2.view(B * T, C, H, W).to(device)
            optimizer.zero_grad()
            # Para cada par en el batch
            batch_loss = 0.0
            for b in range(B):
                f1 = s1[b*T:(b+1)*T]
                f2 = s2[b*T:(b+1)*T]
                prob = model(f1, f2)
                batch_loss += preference_loss(prob.unsqueeze(0), label[b:b+1])
            batch_loss = batch_loss / B
            batch_loss.backward()
            optimizer.step()
            tr_loss += batch_loss.item()
        tr_loss /= len(train_ld)

        # Val
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for seg1, seg2, label in val_ld:
                B, T, C, H, W = seg1.shape
                s1 = seg1.view(B * T, C, H, W).to(device)
                s2 = seg2.view(B * T, C, H, W).to(device)
                for b in range(B):
                    f1 = s1[b*T:(b+1)*T]
                    f2 = s2[b*T:(b+1)*T]
                    prob = model(f1, f2)
                    va_loss += preference_loss(prob.unsqueeze(0), label[b:b+1]).item()
                    pred_label = 1.0 if prob.item() > 0.5 else 0.0
                    va_correct += int(pred_label == label[b].item())
                    va_total   += 1
        va_loss /= max(va_total, 1)
        va_acc   = va_correct / max(va_total, 1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Época {epoch:3d}/{epochs} | "
                  f"loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"val_acc {va_acc:.3f}")

    # Cargar mejor modelo
    model.load_state_dict(torch.load(best_path, map_location=device))
    final_path = os.path.join(out_dir, "pref_reward.pth")
    torch.save(model.state_dict(), final_path)

    # Gráficas
    _plot_pref_training(history, out_dir)

    print(f"\n  Mejor val loss : {best_val_loss:.4f}")
    print(f"  Modelo guardado: {final_path}")
    return model, history


def _plot_pref_training(history: dict, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Entrenamiento del Reward Model por Preferencias", fontsize=12)
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color="#2196F3", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#FF9800", lw=2)
    ax1.set_title("Preference Loss")
    ax1.set_xlabel("Época")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], color="#4CAF50", lw=2, label="Val acc")
    ax2.axhline(0.75, color="gray", ls="--", lw=1, label="75% (objetivo mínimo)")
    ax2.set_title("Accuracy en preferencias")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Época")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "pref_training_curves.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Curvas de entrenamiento: {path}")


# ─────────────────────────────────────────────────────
# 6. RE-ENTRENAMIENTO CON REWARD APRENDIDO
# ─────────────────────────────────────────────────────

class LearnedRewardWrapper:
    """
    Convierte el PreferenceRewardModel en una función de reward
    inyectable en el entorno, igual que ProxyRewardFunction.

    El agente re-entrenado con este reward debería:
      - Ya no buscar configuraciones adversariales (la CNN las distingue mejor)
      - Optimizar según preferencias del oráculo R*
      - Reducir el gap proxy–true (la métrica clave)
    """
    def __init__(self, pref_model: PreferenceRewardModel, device: str = "auto"):
        self.model  = pref_model
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model.to(self.device).eval()
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((64, 64)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, frame: np.ndarray) -> float:
        """Retorna r_pref(frame) normalizado a [0, 1]."""
        with torch.no_grad():
            x = self.transform(frame).unsqueeze(0).to(self.device)
            r = self.model.reward(x).item()
        # Normalizar con sigmoid para mantener en [0, 1]
        return float(torch.sigmoid(torch.tensor(r)).item())


def retrain_with_learned_reward(pref_model_path: str, out_dir: str,
                                 total_steps: int = 200_000,
                                 n_envs: int = 4, seed: int = 42):
    """
    Re-entrena un nuevo agente PPO usando el reward aprendido
    en lugar del proxy CNN original. Este es el experimento de INTERVENCIÓN.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[Retrain] Cargando reward model desde {pref_model_path}...")
    pref_model = PreferenceRewardModel()
    pref_model.load_state_dict(torch.load(pref_model_path, map_location=device))
    learned_reward = LearnedRewardWrapper(pref_model, device=device)

    print(f"[Retrain] Creando {n_envs} entornos con r_pref...")

    def make_env(rank):
        def _init():
            env = ChromaHackEnv(render_mode="rgb_array", seed=seed + rank)
            env.set_proxy_fn(learned_reward)
            return env
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecTransposeImage(vec_env)

    from training.train_ppo import HackingCallback
    from stable_baselines3.common.callbacks import CheckpointCallback

    agent = PPO(
        policy="CnnPolicy", env=vec_env,
        learning_rate=3e-4, n_steps=512, batch_size=64,
        n_epochs=4, gamma=0.99, ent_coef=0.01,
        verbose=1, tensorboard_log=os.path.join(out_dir, "tb_logs"),
        seed=seed, device=device,
    )

    agent.learn(
        total_timesteps=total_steps,
        callback=[
            HackingCallback(log_freq=2000, verbose=1),
            CheckpointCallback(
                save_freq=total_steps // 5,
                save_path=os.path.join(out_dir, "checkpoints"),
                name_prefix="ppo_aligned",
            ),
        ],
        progress_bar=True,
    )

    agent.save(os.path.join(out_dir, "ppo_aligned_final"))
    vec_env.close()
    print(f"\n[Retrain] Agente alineado guardado en {out_dir}/ppo_aligned_final.zip")
    print("Ahora evalúa con:")
    print(f"  python eval/eval_hidden.py --model_dir {out_dir} "
          "--model_name ppo_aligned_final")


# ─────────────────────────────────────────────────────
# 7. FIGURA DE COMPARACIÓN ANTES/DESPUÉS
# ─────────────────────────────────────────────────────

def plot_before_after(hacked_summary: dict, aligned_summary: dict, out_dir: str):
    """
    La figura central de la investigación:
    compara el agente hackeador vs el agente alineado.
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["mean_J_proxy", "mean_J_true", "mean_gap", "mean_idle_rate"]
    labels_es = ["R proxy\nmedio", "R* verdadero\nmedio",
                 "Gap proxy-true\n(hacking)", "Idle rate\n(hacking activo)"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2,
                   [hacked_summary.get(m, 0)  for m in metrics],
                   width, label="Agente hackeador (sin intervención)",
                   color="#F44336", alpha=0.8)
    bars2 = ax.bar(x + width/2,
                   [aligned_summary.get(m, 0) for m in metrics],
                   width, label="Agente alineado (reward por preferencias)",
                   color="#4CAF50", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_es, fontsize=10)
    ax.set_ylabel("Valor (normalizado en [0,1])")
    ax.set_title(
        "ChromaHack — Intervención por Reward Modeling de Preferencias\n"
        "Antes (rojo) vs Después (verde) de la intervención",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Anotar cambio porcentual en gap
    for i, m in enumerate(metrics):
        v_hack = hacked_summary.get(m, 0)
        v_alig = aligned_summary.get(m, 0)
        if v_hack != 0:
            pct = (v_alig - v_hack) / abs(v_hack) * 100
            color = "#4CAF50" if (m == "mean_J_true" and pct > 0) or \
                                 (m in ["mean_gap","mean_idle_rate"] and pct < 0) \
                    else "#F44336"
            ax.annotate(f"{pct:+.0f}%",
                        xy=(i + width/2, v_alig + 0.02),
                        ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "before_after_intervention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig] Figura antes/después guardada: {path}")


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ChromaHack — Intervención por Preferencias"
    )
    parser.add_argument("command", choices=["collect","label","train","retrain","compare"])
    parser.add_argument("--agent_path",       type=str, default="runs/exp_001/ppo_final.zip")
    parser.add_argument("--traj_dir",         type=str, default="runs/exp_001/trajectories")
    parser.add_argument("--pref_model_path",  type=str, default="runs/exp_001/pref_model/pref_reward.pth")
    parser.add_argument("--out_dir",          type=str, default="runs/exp_001/pref_model")
    parser.add_argument("--n_episodes",       type=int, default=200)
    parser.add_argument("--n_pairs",          type=int, default=2000)
    parser.add_argument("--total_steps",      type=int, default=200_000)
    parser.add_argument("--epochs",           type=int, default=30)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--hacked_summary",   type=str, default=None,
                        help="JSON con summary del agente hackeador (para compare)")
    parser.add_argument("--aligned_summary",  type=str, default=None,
                        help="JSON con summary del agente alineado (para compare)")
    args = parser.parse_args()

    if args.command == "collect":
        collect_trajectories(
            args.agent_path, args.traj_dir,
            n_episodes=args.n_episodes, seed=args.seed
        )

    elif args.command == "label":
        seg_path = os.path.join(args.traj_dir, "segments.pkl")
        with open(seg_path, "rb") as f:
            segments = pickle.load(f)
        pairs = generate_preference_pairs(segments, n_pairs=args.n_pairs, seed=args.seed)
        pairs_path = os.path.join(args.traj_dir, "preference_pairs.pkl")
        with open(pairs_path, "wb") as f:
            pickle.dump(pairs, f)
        print(f"[Label] {len(pairs)} pares guardados en {pairs_path}")

    elif args.command == "train":
        pairs_path = os.path.join(args.traj_dir, "preference_pairs.pkl")
        with open(pairs_path, "rb") as f:
            pairs = pickle.load(f)
        train_preference_model(pairs, args.out_dir, epochs=args.epochs, seed=args.seed)

    elif args.command == "retrain":
        retrain_with_learned_reward(
            args.pref_model_path,
            os.path.join(os.path.dirname(args.pref_model_path), "..", "ppo_aligned"),
            total_steps=args.total_steps, seed=args.seed
        )

    elif args.command == "compare":
        if not (args.hacked_summary and args.aligned_summary):
            print("ERROR: --hacked_summary y --aligned_summary requeridos")
            sys.exit(1)
        with open(args.hacked_summary) as f: h = json.load(f)
        with open(args.aligned_summary) as f: a = json.load(f)
        plot_before_after(h, a, args.out_dir)


if __name__ == "__main__":
    main()
