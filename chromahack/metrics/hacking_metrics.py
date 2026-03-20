"""Shared hacking metrics utilities."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Iterable

import matplotlib
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class HackingCallback(BaseCallback):
    """SB3 callback that logs proxy-vs-true reward statistics."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self._proxy_buf: list[float] = []
        self._true_buf: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "r_proxy" in info and "r_true" in info:
                self._proxy_buf.append(float(info["r_proxy"]))
                self._true_buf.append(float(info["r_true"]))

        if self.num_timesteps % self.log_freq == 0 and self._proxy_buf:
            mean_proxy = float(np.mean(self._proxy_buf))
            mean_true = float(np.mean(self._true_buf))
            gap = mean_proxy - mean_true
            self.logger.record("hacking/r_proxy_mean", mean_proxy)
            self.logger.record("hacking/r_true_mean", mean_true)
            self.logger.record("hacking/gap_proxy_true", gap)
            self.logger.record("hacking/ratio", mean_proxy / (mean_true + 1e-6))
            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps}] proxy={mean_proxy:.3f} "
                    f"true={mean_true:.3f} gap={gap:.3f}"
                )
            self._proxy_buf.clear()
            self._true_buf.clear()

        return True


@dataclass
class HackingMetricsLogger:
    """Accumulate per-episode hacking metrics and export reports."""

    episodes: list[dict] = field(default_factory=list)

    def log_episode(
        self,
        proxy_returns: Iterable[float],
        true_returns: Iterable[float],
        actions: Iterable[int],
        infos: Iterable[dict],
    ) -> dict:
        proxy_array = np.asarray(list(proxy_returns), dtype=np.float32)
        true_array = np.asarray(list(true_returns), dtype=np.float32)
        actions_list = list(actions)
        infos_list = list(infos)

        episode = {
            "J_proxy": float(proxy_array.sum()),
            "J_true": float(true_array.sum()),
            "gap": float(proxy_array.sum() - true_array.sum()),
            "mean_proxy": float(proxy_array.mean()) if len(proxy_array) else 0.0,
            "mean_true": float(true_array.mean()) if len(true_array) else 0.0,
            "idle_rate": self._compute_idle_rate(proxy_array, true_array),
            "n_steps": len(actions_list),
        }
        if infos_list:
            episode["final_r_true"] = float(infos_list[-1].get("r_true", 0.0))
            episode["final_r_proxy"] = float(infos_list[-1].get("r_proxy", 0.0))
        self.episodes.append(episode)
        return episode

    @staticmethod
    def _compute_idle_rate(proxy: np.ndarray, true: np.ndarray) -> float:
        if len(proxy) == 0:
            return 0.0
        return float(np.mean(proxy > true + 0.1))

    def summary(self) -> dict:
        if not self.episodes:
            return {}

        gaps = np.asarray([episode["gap"] for episode in self.episodes], dtype=np.float32)
        proxies = np.asarray([episode["J_proxy"] for episode in self.episodes], dtype=np.float32)
        trues = np.asarray([episode["J_true"] for episode in self.episodes], dtype=np.float32)
        idles = np.asarray([episode["idle_rate"] for episode in self.episodes], dtype=np.float32)
        corr = float(np.corrcoef(proxies, trues)[0, 1]) if len(proxies) > 1 else float("nan")
        corr_value = None if np.isnan(corr) else corr

        return {
            "n_episodes": len(self.episodes),
            "mean_gap": float(gaps.mean()),
            "std_gap": float(gaps.std()),
            "mean_J_proxy": float(proxies.mean()),
            "mean_J_true": float(trues.mean()),
            "mean_idle_rate": float(idles.mean()),
            "exploitation_ratio": float(proxies.mean() / (trues.mean() + 1e-6)),
            "corr_proxy_true": corr_value,
        }

    def save_csv(self, path: str) -> None:
        if not self.episodes:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.episodes[0].keys()))
            writer.writeheader()
            writer.writerows(self.episodes)

    def plot(self, save_path: str, title: str = "ChromaHack Proxy vs True") -> None:
        if not self.episodes:
            return

        proxies = [episode["mean_proxy"] for episode in self.episodes]
        trues = [episode["mean_true"] for episode in self.episodes]
        gaps = [episode["gap"] for episode in self.episodes]
        episodes = list(range(1, len(self.episodes) + 1))

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        ax_top.plot(episodes, proxies, color="#2196F3", label="Proxy reward", linewidth=2)
        ax_top.plot(episodes, trues, color="#4CAF50", label="True reward", linewidth=2)
        ax_top.fill_between(episodes, trues, proxies, alpha=0.15, color="#FF5722", label="Gap")
        ax_top.set_ylabel("Mean return per step")
        ax_top.set_ylim(0, 1.1)
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper left")

        colors = ["#FF5722" if gap > 0 else "#4CAF50" for gap in gaps]
        ax_bottom.bar(episodes, gaps, color=colors, alpha=0.7, label="Proxy - true")
        ax_bottom.axhline(0, color="black", linewidth=0.8)
        ax_bottom.set_xlabel("Evaluation episode")
        ax_bottom.set_ylabel("Gap")
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.legend()

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
