"""Preference-based reward modeling intervention."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chromahack.envs.chroma_env import ChromaHackEnv
from chromahack.metrics.hacking_metrics import HackingCallback
from chromahack.models.reward_cnn import PROXY_TRANSFORM


class PreferenceRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.reward_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def reward(self, frames_batch):
        return self.reward_head(self.encoder(frames_batch)).squeeze(-1)

    def forward(self, seg1_frames, seg2_frames):
        reward_1 = self.reward(seg1_frames).sum()
        reward_2 = self.reward(seg2_frames).sum()
        return torch.sigmoid(reward_1 - reward_2)


class PreferenceDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        frames_1, frames_2, label = self.pairs[index]
        tensor_1 = torch.stack([PROXY_TRANSFORM(frame) for frame in frames_1])
        tensor_2 = torch.stack([PROXY_TRANSFORM(frame) for frame in frames_2])
        return tensor_1, tensor_2, torch.tensor(label, dtype=torch.float32)


def collect_trajectories(
    agent_path: str,
    out_dir: str,
    n_episodes: int = 200,
    segment_len: int = 25,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPO.load(agent_path, device=device)
    env = ChromaHackEnv(render_mode="rgb_array", seed=seed)
    env.set_proxy_fn(lambda frame: 0.0)

    segments = []
    for episode_index in range(n_episodes):
        obs, _ = env.reset()
        episode_buffer = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=False)
            next_obs, _, terminated, truncated, info = env.step(int(action))
            episode_buffer.append({"frame": obs.copy(), "r_true": info["r_true"], "action": int(action)})
            obs = next_obs

        for start in range(0, len(episode_buffer) - segment_len + 1, max(1, segment_len // 2)):
            segment = episode_buffer[start : start + segment_len]
            if len(segment) == segment_len:
                segments.append(
                    {
                        "frames": [step["frame"] for step in segment],
                        "r_true_sum": float(sum(step["r_true"] for step in segment)),
                        "mean_r_true": float(np.mean([step["r_true"] for step in segment])),
                    }
                )

        if episode_index == 0 or (episode_index + 1) % 50 == 0:
            print(f"[Collect] ep={episode_index + 1}/{n_episodes} segments={len(segments)}")

    env.close()
    path = os.path.join(out_dir, "segments.pkl")
    with open(path, "wb") as handle:
        pickle.dump(segments, handle)
    print(f"[Collect] wrote {path}")
    return segments


def generate_preference_pairs(
    segments: list,
    n_pairs: int = 2000,
    preference_threshold: float = 0.15,
    seed: int = 42,
) -> list:
    rng = np.random.default_rng(seed)
    pairs = []
    counts = {1: 0, 0: 0, "tie": 0}
    if len(segments) < 2:
        return pairs

    while len(pairs) < n_pairs:
        idx_1, idx_2 = rng.choice(len(segments), 2, replace=False)
        segment_1 = segments[int(idx_1)]
        segment_2 = segments[int(idx_2)]
        diff = float(segment_1["mean_r_true"] - segment_2["mean_r_true"])
        if diff > preference_threshold:
            label = 1.0
            counts[1] += 1
        elif diff < -preference_threshold:
            label = 0.0
            counts[0] += 1
        else:
            label = 0.5
            counts["tie"] += 1
        pairs.append((segment_1["frames"], segment_2["frames"], label))

    print(
        "[Label] clear_left=%d clear_right=%d ties=%d"
        % (counts[1], counts[0], counts["tie"])
    )
    return pairs


def _plot_pref_training(history: dict, out_dir: str) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], label="train", color="#2196F3", lw=2)
    ax_loss.plot(epochs, history["val_loss"], label="val", color="#FF9800", lw=2)
    ax_loss.set_title("Preference loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["val_acc"], label="val acc", color="#4CAF50", lw=2)
    ax_acc.axhline(0.75, color="gray", ls="--", lw=1)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title("Preference accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pref_training_curves.png"), dpi=130, bbox_inches="tight")
    plt.close()


def train_preference_model(
    pairs: list,
    out_dir: str,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 16,
    seed: int = 42,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    clear_pairs = [(frames_1, frames_2, label) for frames_1, frames_2, label in pairs if label != 0.5]
    if not clear_pairs:
        raise ValueError("No clear preference pairs available for training.")
    random.seed(seed)
    random.shuffle(clear_pairs)
    split = int(0.85 * len(clear_pairs)) if clear_pairs else 0
    train_pairs = clear_pairs[:split] or clear_pairs
    val_pairs = clear_pairs[split:] or clear_pairs

    train_loader = DataLoader(PreferenceDataset(train_pairs), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PreferenceDataset(val_pairs), batch_size=batch_size, shuffle=False)

    model = PreferenceRewardModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def preference_loss(prob, label):
        label = label.to(device)
        return -(label * torch.log(prob + 1e-8) + (1 - label) * torch.log(1 - prob + 1e-8)).mean()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_path = os.path.join(out_dir, "pref_reward_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seg1, seg2, label in train_loader:
            batch_size_now, time_steps, channels, height, width = seg1.shape
            flat_seg1 = seg1.view(batch_size_now * time_steps, channels, height, width).to(device)
            flat_seg2 = seg2.view(batch_size_now * time_steps, channels, height, width).to(device)
            optimizer.zero_grad()
            batch_loss = 0.0
            for batch_index in range(batch_size_now):
                frames_1 = flat_seg1[batch_index * time_steps : (batch_index + 1) * time_steps]
                frames_2 = flat_seg2[batch_index * time_steps : (batch_index + 1) * time_steps]
                batch_loss = batch_loss + preference_loss(model(frames_1, frames_2).unsqueeze(0), label[batch_index : batch_index + 1])
            batch_loss = batch_loss / max(batch_size_now, 1)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for seg1, seg2, label in val_loader:
                batch_size_now, time_steps, channels, height, width = seg1.shape
                flat_seg1 = seg1.view(batch_size_now * time_steps, channels, height, width).to(device)
                flat_seg2 = seg2.view(batch_size_now * time_steps, channels, height, width).to(device)
                for batch_index in range(batch_size_now):
                    frames_1 = flat_seg1[batch_index * time_steps : (batch_index + 1) * time_steps]
                    frames_2 = flat_seg2[batch_index * time_steps : (batch_index + 1) * time_steps]
                    prob = model(frames_1, frames_2)
                    val_loss += preference_loss(prob.unsqueeze(0), label[batch_index : batch_index + 1]).item()
                    val_correct += int((prob.item() > 0.5) == (label[batch_index].item() == 1.0))
                    val_total += 1
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[PrefTrain] epoch={epoch}/{epochs} loss={train_loss:.4f}/{val_loss:.4f} acc={val_acc:.3f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_path = os.path.join(out_dir, "pref_reward.pth")
    torch.save(model.state_dict(), final_path)
    _plot_pref_training(history, out_dir)
    return model, history


class LearnedRewardWrapper:
    def __init__(self, pref_model: PreferenceRewardModel, device: str = "auto"):
        self.model = pref_model
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model.to(self.device).eval()

    def __call__(self, frame: np.ndarray) -> float:
        with torch.no_grad():
            tensor = PROXY_TRANSFORM(frame).unsqueeze(0).to(self.device)
            reward = self.model.reward(tensor).item()
        return float(torch.sigmoid(torch.tensor(reward)).item())


def retrain_with_learned_reward(
    pref_model_path: str,
    out_dir: str,
    total_steps: int = 200_000,
    seed: int = 42,
    n_envs: int = 4,
    n_steps: int = 512,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    pref_model = PreferenceRewardModel()
    pref_model.load_state_dict(torch.load(pref_model_path, map_location=device))
    learned_reward = LearnedRewardWrapper(pref_model, device=device)

    def make_env(rank: int):
        def _init():
            env = ChromaHackEnv(render_mode="rgb_array", seed=seed + rank)
            env.set_proxy_fn(learned_reward)
            return env

        return _init

    vec_env = DummyVecEnv([make_env(index) for index in range(n_envs)])
    vec_env = VecTransposeImage(vec_env)
    effective_n_steps = max(1, min(n_steps, total_steps))
    batch_size = min(64, effective_n_steps * n_envs)
    while batch_size > 1 and (effective_n_steps * n_envs) % batch_size != 0:
        batch_size -= 1
    agent = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=effective_n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(out_dir, "tb_logs"),
        seed=seed,
        device=device,
    )
    agent.learn(
        total_timesteps=total_steps,
        callback=[
            HackingCallback(log_freq=2000, verbose=1),
            CheckpointCallback(
                save_freq=max(total_steps // 5, 1),
                save_path=os.path.join(out_dir, "checkpoints"),
                name_prefix="ppo_aligned",
            ),
        ],
        progress_bar=False,
    )
    agent.save(os.path.join(out_dir, "ppo_aligned_final"))
    vec_env.close()


def plot_before_after(hacked_summary: dict, aligned_summary: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["mean_J_proxy", "mean_J_true", "mean_gap", "mean_idle_rate"]
    labels = ["Proxy", "True", "Gap", "Idle"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [hacked_summary.get(metric, 0.0) for metric in metrics], width, label="Hacked", color="#F44336", alpha=0.8)
    ax.bar(x + width / 2, [aligned_summary.get(metric, 0.0) for metric in metrics], width, label="Aligned", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value")
    ax.set_title("Before vs after preference intervention")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "before_after_intervention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preference reward modeling workflow.")
    parser.add_argument("command", choices=["collect", "label", "train", "retrain", "compare"])
    parser.add_argument("--agent_path", type=str, default="runs/exp_001/ppo_final.zip")
    parser.add_argument("--traj_dir", type=str, default="runs/exp_001/trajectories")
    parser.add_argument("--pref_model_path", type=str, default="runs/exp_001/pref_model/pref_reward.pth")
    parser.add_argument("--out_dir", type=str, default="runs/exp_001/pref_model")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_pairs", type=int, default=2000)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment_len", type=int, default=25)
    parser.add_argument("--preference_threshold", type=float, default=0.15)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--hacked_summary", type=str, default=None)
    parser.add_argument("--aligned_summary", type=str, default=None)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        collect_trajectories(
            args.agent_path,
            args.traj_dir,
            n_episodes=args.n_episodes,
            segment_len=args.segment_len,
            seed=args.seed,
        )
        return

    if args.command == "label":
        with open(os.path.join(args.traj_dir, "segments.pkl"), "rb") as handle:
            segments = pickle.load(handle)
        pairs = generate_preference_pairs(
            segments,
            n_pairs=args.n_pairs,
            preference_threshold=args.preference_threshold,
            seed=args.seed,
        )
        path = os.path.join(args.traj_dir, "preference_pairs.pkl")
        with open(path, "wb") as handle:
            pickle.dump(pairs, handle)
        print(f"[Label] wrote {path}")
        return

    if args.command == "train":
        with open(os.path.join(args.traj_dir, "preference_pairs.pkl"), "rb") as handle:
            pairs = pickle.load(handle)
        model, history = train_preference_model(
            pairs,
            args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        with open(os.path.join(args.out_dir, "pref_training_history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        return

    if args.command == "retrain":
        retrain_with_learned_reward(
            args.pref_model_path,
            args.out_dir,
            total_steps=args.total_steps,
            seed=args.seed,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
        )
        return

    if not (args.hacked_summary and args.aligned_summary):
        raise SystemExit("--hacked_summary and --aligned_summary are required for compare")

    with open(args.hacked_summary, "r", encoding="utf-8") as handle:
        hacked_summary = json.load(handle)
    with open(args.aligned_summary, "r", encoding="utf-8") as handle:
        aligned_summary = json.load(handle)
    path = plot_before_after(hacked_summary, aligned_summary, args.out_dir)
    print(f"[Compare] wrote {path}")


if __name__ == "__main__":
    main()
