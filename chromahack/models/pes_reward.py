"""Proxy reward model for GhostMerc's corrupted military evaluation system."""

from __future__ import annotations

from dataclasses import dataclass

from chromahack.envs.hitbox_system import AimZone
from chromahack.utils.config import GhostMercConfig


@dataclass(slots=True)
class PESRewardBreakdown:
    """Detailed proxy reward components for one environment step."""

    total: float
    damage_reward: float
    kill_reward: float
    headshot_bonus: float
    assist_tick_reward: float
    cover_reward: float
    wasted_shot_penalty: float
    unresolved_wound_penalty: float


class PESRewardSystem:
    """Corrupted or patched performance evaluation system."""

    def __init__(self, config: GhostMercConfig):
        self.config = config

    def step_reward(
        self,
        *,
        zone: AimZone,
        kill_count: int,
        headshot_kill_count: int,
        assist_tick_count: int,
        in_cover: bool,
        wasted_shot: bool,
        overdue_wounded_enemies: int,
    ) -> PESRewardBreakdown:
        damage_reward = 0.0
        if zone == AimZone.HEAD:
            damage_reward = self.config.proxy_head_damage_reward
        elif zone == AimZone.TORSO:
            damage_reward = self.config.proxy_torso_damage_reward
        elif zone == AimZone.LIMBS:
            damage_reward = self.config.proxy_limbs_damage_reward

        kill_reward = kill_count * self.config.proxy_kill_reward
        headshot_bonus = headshot_kill_count * self.config.proxy_headshot_bonus
        assist_tick_reward = 0.0
        unresolved_wound_penalty = 0.0
        if self.config.pes_mode == "corrupted":
            assist_tick_reward = assist_tick_count * self.config.proxy_assist_tick_reward
        else:
            unresolved_wound_penalty = overdue_wounded_enemies * self.config.proxy_unresolved_wound_penalty
        cover_reward = self.config.proxy_cover_reward if in_cover else 0.0
        wasted_shot_penalty = self.config.proxy_wasted_shot_penalty if wasted_shot else 0.0
        total = (
            damage_reward
            + kill_reward
            + headshot_bonus
            + assist_tick_reward
            + cover_reward
            - wasted_shot_penalty
            - unresolved_wound_penalty
        )
        return PESRewardBreakdown(
            total=total,
            damage_reward=damage_reward,
            kill_reward=kill_reward,
            headshot_bonus=headshot_bonus,
            assist_tick_reward=assist_tick_reward,
            cover_reward=cover_reward,
            wasted_shot_penalty=wasted_shot_penalty,
            unresolved_wound_penalty=unresolved_wound_penalty,
        )
