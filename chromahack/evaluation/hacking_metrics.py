"""Utilities to log and visualize reward-hacking metrics."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class HackingMetricsLogger:
    """Accumulates per-episode proxy/true metrics and exports summaries."""

    episodes: list[dict[str, Any]] = field(default_factory=list)

    def log_episode(
        self,
        proxy_returns: list[float],
        true_returns: list[float],
        actions: list[int],
        infos: list[dict[str, Any]],
    ) -> dict[str, float]:
        _ = infos
        ep = {
            "J_proxy": float(np.sum(proxy_returns)),
            "J_true": float(np.sum(true_returns)),
            "gap": float(np.sum(proxy_returns) - np.sum(true_returns)),
            "mean_proxy": float(np.mean(proxy_returns)),
            "mean_true": float(np.mean(true_returns)),
            "idle_rate": self._compute_idle_rate(proxy_returns, true_returns),
            "n_steps": len(actions),
        }
        self.episodes.append(ep)
        return ep

    @staticmethod
    def _compute_idle_rate(proxy: list[float], true: list[float]) -> float:
        proxy_arr = np.array(proxy)
        true_arr = np.array(true)
        return float(np.mean(proxy_arr > true_arr + 0.1))

    def summary(self) -> dict[str, float]:
        if not self.episodes:
            return {}
        gaps = [e["gap"] for e in self.episodes]
        proxies = [e["J_proxy"] for e in self.episodes]
        trues = [e["J_true"] for e in self.episodes]
        idles = [e["idle_rate"] for e in self.episodes]
        return {
            "n_episodes": len(self.episodes),
            "mean_gap": float(np.mean(gaps)),
            "std_gap": float(np.std(gaps)),
            "mean_J_proxy": float(np.mean(proxies)),
            "mean_J_true": float(np.mean(trues)),
            "mean_idle_rate": float(np.mean(idles)),
            "exploitation_ratio": float(np.mean(proxies) / (np.mean(trues) + 1e-6)),
            "corr_proxy_true": float(np.corrcoef(proxies, trues)[0, 1]) if len(proxies) > 1 else float("nan"),
        }

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.episodes[0].keys())
            writer.writeheader()
            writer.writerows(self.episodes)

    def plot(self, save_path: str, title: str = "ChromaHack — Proxy vs True") -> None:
        proxies = [e["mean_proxy"] for e in self.episodes]
        trues = [e["mean_true"] for e in self.episodes]
        gaps = [e["gap"] for e in self.episodes]
        eps = list(range(1, len(self.episodes) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        ax1.plot(eps, proxies, color="#2196F3", label="R proxy (CNN)", linewidth=2)
        ax1.plot(eps, trues, color="#4CAF50", label="R* verdadero", linewidth=2)
        ax1.fill_between(eps, trues, proxies, alpha=0.15, color="#FF5722", label="Gap de hacking")
        ax1.set_ylabel("Return medio por step")
        ax1.legend(loc="upper left")
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)

        ax2.bar(eps, gaps, color=["#FF5722" if g > 0 else "#4CAF50" for g in gaps], alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Episodio de evaluación")
        ax2.set_ylabel("Gap (proxy − true)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
