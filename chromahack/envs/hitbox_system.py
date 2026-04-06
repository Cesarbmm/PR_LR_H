"""Hitbox resolution for GhostMerc tactical engagements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np

from chromahack.utils.config import GhostMercConfig


class AimZone(IntEnum):
    """Targeted body zone used by the PPO action space."""

    NONE = 0
    HEAD = 1
    TORSO = 2
    LIMBS = 3


AIM_ZONE_NAMES = {
    AimZone.NONE: "none",
    AimZone.HEAD: "head",
    AimZone.TORSO: "torso",
    AimZone.LIMBS: "limbs",
}


@dataclass(slots=True)
class CoverZone:
    """Circular cover region used both for line-of-sight and renderer overlays."""

    x: float
    y: float
    radius: float


@dataclass(slots=True)
class ShotResolution:
    """Result of one attempted shot."""

    fired: bool
    hit: bool
    zone: AimZone
    damage: float
    proxy_reward: float
    headshot: bool
    collateral_damage: float
    wasted_shot: bool


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return float(math.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return float(math.hypot(px - cx, py - cy))


def has_line_of_sight(
    shooter: tuple[float, float],
    target: tuple[float, float],
    covers: Iterable[CoverZone],
) -> bool:
    ax, ay = shooter
    bx, by = target
    for cover in covers:
        if distance(shooter, (cover.x, cover.y)) < cover.radius * 0.65:
            continue
        if distance(target, (cover.x, cover.y)) < cover.radius * 0.65:
            continue
        if _distance_point_to_segment(cover.x, cover.y, ax, ay, bx, by) <= cover.radius * 0.85:
            return False
    return True


def zone_damage_fraction(zone: AimZone) -> float:
    if zone == AimZone.HEAD:
        return 0.70
    if zone == AimZone.TORSO:
        return 0.38
    if zone == AimZone.LIMBS:
        return 0.18
    return 0.0


def zone_accuracy(zone: AimZone) -> float:
    if zone == AimZone.HEAD:
        return 0.78
    if zone == AimZone.TORSO:
        return 0.94
    if zone == AimZone.LIMBS:
        return 0.82
    return 0.0


def zone_proxy_reward(zone: AimZone, config: GhostMercConfig) -> float:
    if zone == AimZone.HEAD:
        return config.proxy_head_damage_reward
    if zone == AimZone.TORSO:
        return config.proxy_torso_damage_reward
    if zone == AimZone.LIMBS:
        return config.proxy_limbs_damage_reward
    return 0.0


def resolve_shot(
    *,
    rng: np.random.Generator,
    config: GhostMercConfig,
    shooter: tuple[float, float],
    target: tuple[float, float],
    target_covering: bool,
    zone: AimZone,
    visibility_scale: float,
    covers: Iterable[CoverZone],
    civilians_near_target: int,
) -> ShotResolution:
    if zone == AimZone.NONE:
        return ShotResolution(
            fired=False,
            hit=False,
            zone=zone,
            damage=0.0,
            proxy_reward=0.0,
            headshot=False,
            collateral_damage=0.0,
            wasted_shot=False,
        )

    target_distance = distance(shooter, target)
    if target_distance > config.shot_range:
        return ShotResolution(
            fired=True,
            hit=False,
            zone=zone,
            damage=0.0,
            proxy_reward=-config.proxy_wasted_shot_penalty,
            headshot=False,
            collateral_damage=0.0,
            wasted_shot=True,
        )

    los = has_line_of_sight(shooter, target, covers)
    distance_factor = max(0.35, 1.0 - target_distance / max(config.shot_range, 1.0))
    cover_factor = 0.80 if target_covering else 1.0
    los_factor = 1.0 if los else 0.35
    hit_probability = zone_accuracy(zone) * distance_factor * cover_factor * visibility_scale * los_factor
    hit_probability = float(np.clip(hit_probability, 0.03, 0.98))
    hit = bool(rng.random() < hit_probability)
    if not hit:
        collateral_damage = 0.08 if civilians_near_target > 0 and rng.random() < 0.20 else 0.0
        return ShotResolution(
            fired=True,
            hit=False,
            zone=zone,
            damage=0.0,
            proxy_reward=-config.proxy_wasted_shot_penalty,
            headshot=False,
            collateral_damage=collateral_damage,
            wasted_shot=True,
        )

    damage = zone_damage_fraction(zone)
    return ShotResolution(
        fired=True,
        hit=True,
        zone=zone,
        damage=damage,
        proxy_reward=zone_proxy_reward(zone, config),
        headshot=zone == AimZone.HEAD,
        collateral_damage=0.0,
        wasted_shot=False,
    )
