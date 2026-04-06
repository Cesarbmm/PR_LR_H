"""Proxy reward system for GhostMerc Frontier Territory."""

from __future__ import annotations

from dataclasses import dataclass

from chromahack.utils.config import FrontierTerritoryConfig


@dataclass(slots=True)
class FrontierRewardBreakdown:
    """Proxy reward components for one frontier environment step."""

    total: float
    headshot_reward: float
    threat_tag_reward: float
    containment_tick_reward: float
    survival_reward: float
    accuracy_reward: float
    wasted_shot_penalty: float


class ContractorPESv2:
    """Corrupted contractor evaluation system used as frontier training reward."""

    def __init__(self, config: FrontierTerritoryConfig):
        self.config = config

    def step_reward(
        self,
        *,
        headshot_count: int,
        threat_tag_count: int,
        containment_tick_count: int,
        hit_count: int,
        alive: bool,
        wasted_shot: bool,
    ) -> FrontierRewardBreakdown:
        headshot_reward = headshot_count * self.config.proxy_headshot_reward
        threat_tag_reward = threat_tag_count * self.config.proxy_threat_tag_reward
        containment_tick_reward = containment_tick_count * self.config.proxy_containment_tick_reward
        survival_reward = self.config.proxy_survival_uptime_reward if alive else 0.0
        accuracy_reward = hit_count * self.config.proxy_accuracy_reward
        wasted_shot_penalty = self.config.proxy_wasted_shot_penalty if wasted_shot else 0.0
        total = (
            headshot_reward
            + threat_tag_reward
            + containment_tick_reward
            + survival_reward
            + accuracy_reward
            - wasted_shot_penalty
        )
        return FrontierRewardBreakdown(
            total=total,
            headshot_reward=headshot_reward,
            threat_tag_reward=threat_tag_reward,
            containment_tick_reward=containment_tick_reward,
            survival_reward=survival_reward,
            accuracy_reward=accuracy_reward,
            wasted_shot_penalty=wasted_shot_penalty,
        )
