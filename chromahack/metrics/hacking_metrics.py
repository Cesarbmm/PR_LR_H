"""Compatibility metrics wrappers for older ChromaHack imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from chromahack.utils.metrics import (
    GhostMercTrainingCallback,
    aggregate_ghostmerc_metrics,
    summarize_ghostmerc_episode,
    write_episode_csv,
)

HackingCallback = GhostMercTrainingCallback


@dataclass
class HackingMetricsLogger:
    """Small compatibility wrapper around the GhostMerc metrics helpers."""

    episodes: list[dict] = field(default_factory=list)

    def log_episode(
        self,
        proxy_returns: Iterable[float],
        true_returns: Iterable[float],
        actions: Iterable[int],
        infos: Iterable[dict],
    ) -> dict:
        step_records = []
        cumulative_proxy = 0.0
        cumulative_true = 0.0
        for step_index, (proxy_reward, true_reward, action, info) in enumerate(
            zip(proxy_returns, true_returns, actions, infos, strict=False),
            start=1,
        ):
            cumulative_proxy += float(proxy_reward)
            cumulative_true += float(true_reward)
            info_record = {
                "step": int(info.get("step", step_index)),
                "scenario_id": int(info.get("scenario_id", 0)),
                "phase_label": str(info.get("phase_label", "basic_training")),
                "assist_tick_count": int(info.get("assist_tick_count", 0)),
                "kill_completion_rate": float(info.get("kill_completion_rate", 0.0)),
                "wound_farming_rate": float(info.get("wound_farming_rate", 0.0)),
                "assist_tick_rate": float(info.get("assist_tick_rate", 0.0)),
                "civilians_saved_pct": float(info.get("civilians_saved_pct", 0.0)),
                "objectives_completed_pct": float(info.get("objectives_completed_pct", 0.0)),
                "phase_transition_detected": bool(info.get("phase_transition_detected", False)),
                "phase_transition_step": info.get("phase_transition_step"),
                "mission_success": bool(info.get("mission_success", False)),
            }
            step_records.append(
                {
                    "step": step_index,
                    "action": int(action),
                    "action_name": str(action),
                    "observation": [],
                    "proxy_reward": float(proxy_reward),
                    "true_reward": float(true_reward),
                    "cumulative_proxy_reward": cumulative_proxy,
                    "cumulative_true_reward": cumulative_true,
                    "state_snapshot": info.get("state_snapshot", {}),
                    "info": info_record,
                }
            )

        episode = summarize_ghostmerc_episode(step_records)
        self.episodes.append(episode)
        return episode

    def summary(self) -> dict:
        return aggregate_ghostmerc_metrics(self.episodes)

    def save_csv(self, path: str) -> None:
        write_episode_csv(path, self.episodes)
