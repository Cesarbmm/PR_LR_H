"""Preference-model utilities and reward wrapper for Frontier intervention experiments."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from chromahack.utils.paths import resolve_input_path, resolve_project_path
from chromahack.utils.trajectory_io import load_episode_trajectory

STEP_FEATURE_DIM = 48
CLIP_CONTEXT_DIM = 8


def _flatten_numeric_tree(value: Any) -> list[float]:
    if isinstance(value, dict):
        flattened: list[float] = []
        for key in sorted(value):
            flattened.extend(_flatten_numeric_tree(value[key]))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numeric_tree(item))
        return flattened
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1).tolist()
    if isinstance(value, np.generic):
        return [float(value.item())]
    if isinstance(value, (int, float, bool)):
        return [float(value)]
    return []


def extract_step_feature_vector(
    observation: Any,
    info: dict[str, Any],
    *,
    feature_dim: int = STEP_FEATURE_DIM,
) -> list[float]:
    """Build a compact fixed-size feature vector from one Frontier step."""

    basis = info.get("state_snapshot", observation)
    flattened = np.asarray(_flatten_numeric_tree(basis), dtype=np.float32)
    base_dim = max(feature_dim - 12, 1)
    truncated = flattened[:base_dim]
    if truncated.shape[0] < base_dim:
        truncated = np.pad(truncated, (0, base_dim - truncated.shape[0]))
    metrics = np.asarray(
        [
            float(info.get("proxy_reward", 0.0)),
            float(info.get("true_reward", 0.0)),
            float(info.get("gap", float(info.get("proxy_reward", 0.0)) - float(info.get("true_reward", 0.0)))),
            float(info.get("containment_tick_count", 0.0)),
            float(info.get("containment_tick_rate", 0.0)),
            float(info.get("armed_neutral_false_positive_rate", 0.0)),
            float(info.get("threat_tag_precision", 0.0)),
            float(info.get("threat_tag_recall", 0.0)),
            float(info.get("civilians_safe_pct", 0.0)),
            float(info.get("trust_score", 0.0)),
            float(info.get("territory_stability", 0.0)),
            1.0 if bool(info.get("phase_transition_detected", False)) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([truncated, metrics]).astype(float).tolist()


@dataclass(slots=True)
class PreferenceClip:
    """Compact clip summary used for oracle preference labeling."""

    trajectory_path: str
    start_index: int
    end_index: int
    true_return: float
    proxy_return: float
    feature_vector: list[float]


@dataclass(slots=True)
class PreferencePair:
    """One oracle-labeled preference comparison."""

    clip_a: PreferenceClip
    clip_b: PreferenceClip
    preferred: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_a": asdict(self.clip_a),
            "clip_b": asdict(self.clip_b),
            "preferred": self.preferred,
        }


class PreferenceRewardModel(nn.Module):
    """Simple MLP that scores preference clip summaries."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class PreferencePairDataset(Dataset):
    """Torch dataset over exported preference pairs."""

    def __init__(self, pairs: list[PreferencePair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        features_a = torch.tensor(pair.clip_a.feature_vector, dtype=torch.float32)
        features_b = torch.tensor(pair.clip_b.feature_vector, dtype=torch.float32)
        label = torch.tensor(float(pair.preferred), dtype=torch.float32)
        return features_a, features_b, label


def _clip_feature_vector(steps: list[dict[str, Any]]) -> list[float]:
    if not steps:
        return [0.0] * (STEP_FEATURE_DIM + CLIP_CONTEXT_DIM)
    step_vectors = np.asarray(
        [
            extract_step_feature_vector(step.get("observation"), step.get("info", {}), feature_dim=STEP_FEATURE_DIM)
            for step in steps
        ],
        dtype=np.float32,
    )
    mean_step = step_vectors.mean(axis=0)
    last_info = steps[-1].get("info", {})
    clip_context = np.asarray(
        [
            float(sum(step.get("proxy_reward", 0.0) for step in steps)),
            float(sum(step.get("true_reward", 0.0) for step in steps)),
            float(last_info.get("containment_tick_count", 0.0)),
            float(last_info.get("containment_tick_rate", 0.0)),
            float(last_info.get("armed_neutral_false_positive_rate", 0.0)),
            float(last_info.get("civilians_safe_pct", 0.0)),
            float(last_info.get("trust_score", 0.0)),
            float(last_info.get("territory_stability", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([mean_step, clip_context]).astype(float).tolist()


def extract_clips_from_trajectory(path: str, clip_length: int, stride: int) -> list[PreferenceClip]:
    payload = load_episode_trajectory(path)
    steps = payload.get("steps", [])
    clips: list[PreferenceClip] = []
    for start_index in range(0, max(1, len(steps) - clip_length + 1), max(1, stride)):
        clip_steps = steps[start_index : start_index + clip_length]
        if not clip_steps:
            continue
        true_return = float(sum(step.get("true_reward", 0.0) for step in clip_steps))
        proxy_return = float(sum(step.get("proxy_reward", 0.0) for step in clip_steps))
        clips.append(
            PreferenceClip(
                trajectory_path=path,
                start_index=start_index,
                end_index=start_index + len(clip_steps),
                true_return=true_return,
                proxy_return=proxy_return,
                feature_vector=_clip_feature_vector(clip_steps),
            )
        )
    return clips


def build_preference_pairs(trajectory_dir: str, *, clip_length: int, stride: int, max_pairs: int, seed: int) -> list[PreferencePair]:
    paths = sorted(str(path) for path in Path(trajectory_dir).glob("*.json"))
    clips: list[PreferenceClip] = []
    for path in paths:
        clips.extend(extract_clips_from_trajectory(path, clip_length=clip_length, stride=stride))
    if len(clips) < 2:
        return []
    rng = np.random.default_rng(seed)
    pairs: list[PreferencePair] = []
    for _ in range(max_pairs):
        clip_a, clip_b = rng.choice(clips, size=2, replace=False)
        if clip_a.true_return == clip_b.true_return:
            preferred = 0 if clip_a.proxy_return >= clip_b.proxy_return else 1
        else:
            preferred = 0 if clip_a.true_return > clip_b.true_return else 1
        pairs.append(PreferencePair(clip_a=clip_a, clip_b=clip_b, preferred=preferred))
    return pairs


def export_preference_dataset(args) -> dict[str, Any]:
    trajectory_dir = str(resolve_input_path(args.trajectory_dir))
    out_dir = str(resolve_project_path(args.out_dir))
    pairs = build_preference_pairs(
        trajectory_dir,
        clip_length=args.clip_length,
        stride=args.stride,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preference_pairs.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump([pair.to_dict() for pair in pairs], handle, indent=2)
    manifest = {
        "n_pairs": len(pairs),
        "clip_length": args.clip_length,
        "stride": args.stride,
        "feature_dim": STEP_FEATURE_DIM + CLIP_CONTEXT_DIM,
        "trajectory_dir": trajectory_dir,
        "dataset_path": out_path,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))
    return manifest


def _load_pairs(dataset_path: str) -> list[PreferencePair]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        PreferencePair(
            clip_a=PreferenceClip(**item["clip_a"]),
            clip_b=PreferenceClip(**item["clip_b"]),
            preferred=int(item["preferred"]),
        )
        for item in payload
    ]


def _pairwise_accuracy(model: PreferenceRewardModel, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features_a, features_b, label in loader:
            logits = model(features_a) - model(features_b)
            prediction = (logits < 0).float()
            correct += int((prediction == label).sum().item())
            total += int(label.numel())
    return float(correct / max(total, 1))


def train_preference_model(args) -> dict[str, Any]:
    dataset_path = str(resolve_input_path(args.dataset))
    out_dir = str(resolve_project_path(args.out_dir))
    pairs = _load_pairs(dataset_path)
    if not pairs:
        raise SystemExit("Preference dataset is empty; export clips first.")

    dataset = PreferencePairDataset(pairs)
    if len(dataset) >= 10:
        val_size = max(1, int(len(dataset) * 0.2))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_dataset = dataset
        val_dataset = dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    input_dim = len(pairs[0].clip_a.feature_vector)
    model = PreferenceRewardModel(input_dim=input_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses: list[float] = []

    for _ in range(args.epochs):
        model.train()
        for features_a, features_b, label in train_loader:
            score_a = model(features_a)
            score_b = model(features_b)
            logits = score_a - score_b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, 1.0 - label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

    train_accuracy = _pairwise_accuracy(model, DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False))
    validation_accuracy = _pairwise_accuracy(model, val_loader)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "preference_reward_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "step_feature_dim": STEP_FEATURE_DIM,
            "clip_context_dim": CLIP_CONTEXT_DIM,
        },
        model_path,
    )
    summary = {
        "dataset": dataset_path,
        "model_path": model_path,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "epochs": args.epochs,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
    }
    with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def load_preference_reward_model(model_path: str, *, device: str | None = None) -> tuple[PreferenceRewardModel, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device or "cpu")
    model = PreferenceRewardModel(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


class FrontierPreferenceRewardScorer:
    """Convert a trained clip-level preference model into a dense step reward."""

    def __init__(self, model_path: str, *, clip_length: int = 48, device: str | None = None):
        self.model, self.metadata = load_preference_reward_model(model_path, device=device)
        self.clip_length = int(clip_length)
        self._history: list[dict[str, Any]] = []
        self._last_score = 0.0

    def reset(self) -> None:
        self._history = []
        self._last_score = 0.0

    def step_reward(self, *, observation: Any, info: dict[str, Any]) -> float:
        record = {
            "observation": observation,
            "proxy_reward": float(info.get("proxy_reward", 0.0)),
            "true_reward": float(info.get("true_reward", 0.0)),
            "info": info,
        }
        self._history.append(record)
        if len(self._history) > self.clip_length:
            self._history = self._history[-self.clip_length :]
        feature_vector = _clip_feature_vector(self._history)
        tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = float(self.model(tensor).item())
        reward = score - self._last_score
        self._last_score = score
        return reward


class FrontierPreferenceRewardWrapper(gym.Wrapper):
    """Replace Frontier proxy rewards with a learned preference reward signal."""

    def __init__(self, env: gym.Env, model_path: str, *, clip_length: int = 48, device: str | None = None):
        super().__init__(env)
        self.scorer = FrontierPreferenceRewardScorer(model_path, clip_length=clip_length, device=device)
        self.reward_model_path = model_path

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.scorer.reset()
        wrapped_info = dict(info)
        wrapped_info["reward_mode"] = "pref_model"
        wrapped_info["reward_model_path"] = self.reward_model_path
        return observation, wrapped_info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        reward = self.scorer.step_reward(observation=observation, info=info)
        wrapped_info = dict(info)
        wrapped_info["proxy_reward_original"] = float(info.get("proxy_reward", 0.0))
        wrapped_info["reward_model_reward"] = float(reward)
        wrapped_info["reward_mode"] = "pref_model"
        wrapped_info["reward_model_path"] = self.reward_model_path
        wrapped_info["proxy_reward"] = float(reward)
        wrapped_info["gap"] = float(reward - float(wrapped_info.get("true_reward", 0.0)))
        return observation, float(reward), terminated, truncated, wrapped_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GhostMerc Frontier preference-model utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export oracle-labeled preference pairs from trajectories.")
    export_parser.add_argument("--trajectory_dir", type=str, required=True)
    export_parser.add_argument("--out_dir", type=str, default="artifacts/preferences")
    export_parser.add_argument("--clip_length", type=int, default=48)
    export_parser.add_argument("--stride", type=int, default=24)
    export_parser.add_argument("--max_pairs", type=int, default=256)
    export_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="Train a small preference reward model on exported pairs.")
    train_parser.add_argument("--dataset", type=str, required=True)
    train_parser.add_argument("--out_dir", type=str, default="artifacts/preferences/model")
    train_parser.add_argument("--hidden_dim", type=int, default=128)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        export_preference_dataset(args)
    elif args.command == "train":
        train_preference_model(args)


if __name__ == "__main__":
    main()
