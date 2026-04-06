"""Frontier observability helpers for action entropy, KL drift, and simple latent summaries."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


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


def _entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(np.float64), 1e-8, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def _kl_divergence(current: np.ndarray, reference: np.ndarray) -> float:
    current = np.clip(current.astype(np.float64), 1e-8, 1.0)
    reference = np.clip(reference.astype(np.float64), 1e-8, 1.0)
    return float((current * (np.log(current) - np.log(reference))).sum())


@dataclass(slots=True)
class FrontierObservabilitySummary:
    action_entropy: float = 0.0
    action_kl_to_snapshot: float = 0.0
    latent_pca_var_1: float = 0.0
    latent_pca_var_2: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "action_entropy": self.action_entropy,
            "action_kl_to_snapshot": self.action_kl_to_snapshot,
            "latent_pca_var_1": self.latent_pca_var_1,
            "latent_pca_var_2": self.latent_pca_var_2,
        }


class FrontierObservabilityMonitor:
    """Track coarse behavioural drift during Frontier training."""

    def __init__(
        self,
        *,
        action_nvec: list[int],
        window_size: int = 256,
        snapshot_interval: int = 512,
        latent_dim: int = 64,
    ) -> None:
        self.action_nvec = [int(value) for value in action_nvec]
        self.window_size = int(window_size)
        self.snapshot_interval = int(snapshot_interval)
        self.latent_dim = int(latent_dim)
        self._actions: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._latents: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._snapshot_histograms: list[np.ndarray] | None = None
        self._steps_since_snapshot = 0

    def update(self, *, actions: np.ndarray, observations: Any) -> None:
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        for action in action_array:
            self._actions.append(action.reshape(-1))
        latent = np.asarray(_flatten_numeric_tree(observations), dtype=np.float32)
        if latent.size:
            latent = latent[: self.latent_dim]
            if latent.shape[0] < self.latent_dim:
                latent = np.pad(latent, (0, self.latent_dim - latent.shape[0]))
            self._latents.append(latent)
        self._steps_since_snapshot += action_array.shape[0]
        if self._steps_since_snapshot >= self.snapshot_interval and self._actions:
            self._snapshot_histograms = self._action_histograms()
            self._steps_since_snapshot = 0

    def _action_histograms(self) -> list[np.ndarray]:
        if not self._actions:
            return [np.ones(n, dtype=np.float64) / max(n, 1) for n in self.action_nvec]
        action_matrix = np.stack(list(self._actions), axis=0)
        histograms: list[np.ndarray] = []
        for branch_index, n_choices in enumerate(self.action_nvec):
            counts = np.bincount(action_matrix[:, branch_index], minlength=n_choices).astype(np.float64)
            total = max(float(counts.sum()), 1.0)
            histograms.append(counts / total)
        return histograms

    def _pca_variance(self) -> tuple[float, float]:
        if len(self._latents) < 3:
            return 0.0, 0.0
        matrix = np.stack(list(self._latents), axis=0)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        if not np.any(centered):
            return 0.0, 0.0
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        variance = singular_values**2
        total = max(float(variance.sum()), 1e-8)
        first = float(variance[0] / total) if variance.size > 0 else 0.0
        second = float(variance[1] / total) if variance.size > 1 else 0.0
        return first, second

    def summary(self) -> FrontierObservabilitySummary:
        histograms = self._action_histograms()
        mean_entropy = float(np.mean([_entropy(hist) for hist in histograms])) if histograms else 0.0
        if self._snapshot_histograms is None:
            mean_kl = 0.0
        else:
            mean_kl = float(
                np.mean([_kl_divergence(current, reference) for current, reference in zip(histograms, self._snapshot_histograms)])
            )
        latent_var_1, latent_var_2 = self._pca_variance()
        return FrontierObservabilitySummary(
            action_entropy=mean_entropy,
            action_kl_to_snapshot=mean_kl,
            latent_pca_var_1=latent_var_1,
            latent_pca_var_2=latent_var_2,
        )
