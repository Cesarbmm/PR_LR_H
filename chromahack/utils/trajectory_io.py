"""Trajectory serialization helpers for Frontier, classic GhostMerc, and archived bridge rollouts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrajectoryStep:
    """Serializable per-step record for replay and analysis."""

    step: int
    action: int | list[int]
    action_name: str
    observation: Any
    proxy_reward: float
    true_reward: float
    cumulative_proxy_reward: float
    cumulative_true_reward: float
    state_snapshot: dict[str, Any]
    info: dict[str, Any]


@dataclass(slots=True)
class EpisodeTrajectory:
    """Serializable episode rollout."""

    episode_index: int
    seed: int
    terminated: bool
    truncated: bool
    summary: dict[str, Any]
    steps: list[TrajectoryStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "summary": self.summary,
            "steps": [asdict(step) for step in self.steps],
        }


def save_episode_trajectory(trajectory: EpisodeTrajectory, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(trajectory.to_dict(), handle, indent=2)


def load_episode_trajectory(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def serialize_observation(observation: Any) -> Any:
    if isinstance(observation, np.ndarray):
        return observation.astype(float).tolist()
    if isinstance(observation, np.generic):
        return observation.item()
    if isinstance(observation, dict):
        return {str(key): serialize_observation(value) for key, value in observation.items()}
    if isinstance(observation, (list, tuple)):
        return [serialize_observation(value) for value in observation]
    return observation
