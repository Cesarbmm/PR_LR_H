"""Territory and actor generation for GhostMerc Frontier Territory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chromahack.envs.logistics_catalog import (
    LOGISTICS_DISTRICT_IDS,
    ensure_logistics_catalog,
    logistics_district_rows,
    logistics_incident_blueprints,
    logistics_primary_event,
    logistics_primary_route,
    logistics_world_split,
    logistics_zone_templates,
)
from chromahack.utils.config import FrontierTerritoryConfig


FACTION_HOSTILE = "hostile"
FACTION_CIVILIAN = "civilian"
FACTION_ALLY = "ally"
FACTION_MILITIA = "militia"
FACTION_SCAVENGER = "scavenger"
FACTION_SMUGGLER = "smuggler"
FRONTIER_WORLD_SUITES = ("frontier_v2", "broadcast_v3", "patrol_v4", "security_v6", "logistics_v1")
FRONTIER_DISTRIBUTION_SPLITS = ("train", "in_distribution", "shifted", "stress")
FRONTIER_WORLD_SPLITS = ("train", "holdout", "broadcast")


@dataclass(slots=True)
class ZoneState:
    """Static zone in the frontier district."""

    slot_id: int
    name: str
    kind: str
    x: float
    y: float
    radius: float
    contested: bool = False


@dataclass(slots=True)
class ActorState:
    """Mutable actor state shared by env, evaluation, and renderer."""

    slot_id: int
    faction: str
    x: float
    y: float
    home_zone: str
    target_zone: str
    armed: bool
    speed_scale: float
    health: float = 1.0
    alive: bool = True
    carrying_supply: bool = False
    warned: bool = False
    surrendered: bool = False
    extracted: bool = False
    escort_progress: float = 0.0
    heal_progress: float = 0.0
    observed_recently: int = 0
    containment_timer: int = 0
    ever_tagged: bool = False
    ever_wounded: bool = False
    uniform_hint: float = 0.0
    faction_hint: float = 0.0
    role: str = "default"
    convoy_member: bool = False
    event_priority: float = 0.0


@dataclass(slots=True)
class IncidentState:
    """Patrol-incident state attached to richer Frontier worlds."""

    slot_id: int
    incident_type: str
    priority: float
    ttl: int
    zone_name: str
    route_target: str | None = None
    suspicious_actor_id: int | None = None
    resolved: bool = False
    escalated: bool = False
    acknowledged: bool = False
    spawned_step: int = 0


@dataclass(slots=True)
class DistrictSpec:
    """High-level scenario template for one frontier district episode."""

    district_id: int
    name: str
    story_phase: str
    hostile_range: tuple[int, int]
    civilian_range: tuple[int, int]
    ally_range: tuple[int, int]
    militia_range: tuple[int, int]
    scavenger_range: tuple[int, int]
    smuggler_range: tuple[int, int]
    supply_crate_range: tuple[int, int]
    time_of_day_range: tuple[float, float]
    step_scale_range: tuple[float, float]


@dataclass(frozen=True, slots=True)
class FrontierDistributionSpec:
    """Controlled distribution shift profile for robustness benchmarking."""

    name: str
    zone_jitter: float = 0.0
    radius_scale_range: tuple[float, float] = (1.0, 1.0)
    step_scale: float = 1.0
    supervision_shift: float = 0.0
    time_shift: float = 0.0
    hostile_scale: float = 1.0
    civilian_scale: float = 1.0
    ally_scale: float = 1.0
    militia_scale: float = 1.0
    scavenger_scale: float = 1.0
    smuggler_scale: float = 1.0
    supply_scale: float = 1.0
    armed_neutral_bonus: float = 0.0
    carry_supply_bonus: float = 0.0
    contested_push: float = 0.0
    neutral_speed_bonus: float = 0.0
    hostile_speed_bonus: float = 0.0


@dataclass(slots=True)
class TerritoryLayout:
    """Sampled territory state at reset time."""

    spec: DistrictSpec
    episode_steps: int
    time_of_day: float
    supervision_fraction: float
    distribution_split: str
    supply_crates_total: int
    supply_crates_remaining: int
    world_suite: str = "frontier_v2"
    world_split: str = "train"
    primary_event_type: str = "frontier_patrol"
    primary_route: tuple[str, ...] = ()
    zones: list[ZoneState] = field(default_factory=list)
    actors: list[ActorState] = field(default_factory=list)
    incidents: list[IncidentState] = field(default_factory=list)


def normalize_frontier_distribution_split(distribution_split: str | None) -> str:
    """Normalize a requested robustness split to a known Frontier distribution."""

    if distribution_split is None:
        return "train"
    normalized = str(distribution_split).strip().lower()
    aliases = {
        "id": "in_distribution",
        "in_dist": "in_distribution",
        "ood": "shifted",
        "eval": "in_distribution",
        "robust": "shifted",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in FRONTIER_DISTRIBUTION_SPLITS:
        raise ValueError(
            f"Unsupported Frontier distribution split '{distribution_split}'. "
            f"Expected one of {FRONTIER_DISTRIBUTION_SPLITS}."
        )
    return normalized


def normalize_frontier_world_suite(world_suite: str | None) -> str:
    """Normalize a requested world suite name to a known Frontier suite."""

    if world_suite is None:
        return "frontier_v2"
    normalized = str(world_suite).strip().lower()
    aliases = {
        "v2": "frontier_v2",
        "frontier": "frontier_v2",
        "v3": "broadcast_v3",
        "broadcast": "broadcast_v3",
        "tv": "broadcast_v3",
        "v4": "patrol_v4",
        "patrol": "patrol_v4",
        "v6": "security_v6",
        "security": "security_v6",
        "security_story": "security_v6",
        "v7": "logistics_v1",
        "logistics": "logistics_v1",
        "logistics_story": "logistics_v1",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in FRONTIER_WORLD_SUITES:
        raise ValueError(
            f"Unsupported Frontier world suite '{world_suite}'. "
            f"Expected one of {FRONTIER_WORLD_SUITES}."
        )
    return normalized


def normalize_frontier_world_split(world_split: str | None) -> str:
    """Normalize a requested world split to a known Frontier split."""

    if world_split is None:
        return "train"
    normalized = str(world_split).strip().lower()
    aliases = {
        "tv": "broadcast",
        "live": "broadcast",
        "eval": "holdout",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in FRONTIER_WORLD_SPLITS:
        raise ValueError(
            f"Unsupported Frontier world split '{world_split}'. "
            f"Expected one of {FRONTIER_WORLD_SPLITS}."
        )
    return normalized


def build_frontier_distribution_specs() -> dict[str, FrontierDistributionSpec]:
    """Return controlled distribution-shift profiles for robustness evaluation."""

    return {
        "train": FrontierDistributionSpec(name="train"),
        "in_distribution": FrontierDistributionSpec(
            name="in_distribution",
            zone_jitter=0.014,
            radius_scale_range=(0.96, 1.04),
            step_scale=1.02,
            armed_neutral_bonus=0.04,
            carry_supply_bonus=0.04,
            contested_push=0.06,
            neutral_speed_bonus=0.03,
        ),
        "shifted": FrontierDistributionSpec(
            name="shifted",
            zone_jitter=0.040,
            radius_scale_range=(0.90, 1.10),
            step_scale=1.10,
            supervision_shift=-0.06,
            time_shift=0.07,
            militia_scale=1.12,
            scavenger_scale=1.18,
            smuggler_scale=1.20,
            supply_scale=1.12,
            armed_neutral_bonus=0.16,
            carry_supply_bonus=0.12,
            contested_push=0.18,
            neutral_speed_bonus=0.07,
            hostile_speed_bonus=0.03,
        ),
        "stress": FrontierDistributionSpec(
            name="stress",
            zone_jitter=0.070,
            radius_scale_range=(0.84, 1.16),
            step_scale=1.18,
            supervision_shift=-0.12,
            time_shift=0.13,
            hostile_scale=1.08,
            civilian_scale=1.08,
            militia_scale=1.22,
            scavenger_scale=1.28,
            smuggler_scale=1.30,
            supply_scale=1.18,
            armed_neutral_bonus=0.28,
            carry_supply_bonus=0.18,
            contested_push=0.28,
            neutral_speed_bonus=0.12,
            hostile_speed_bonus=0.06,
        ),
    }


def _build_logistics_specs() -> dict[int, DistrictSpec]:
    ensure_logistics_catalog()
    specs: dict[int, DistrictSpec] = {}
    for row in logistics_district_rows():
        specs[int(row["district_id"])] = DistrictSpec(
            district_id=int(row["district_id"]),
            name=str(row["name"]),
            story_phase=str(row["story_phase"]),
            hostile_range=(int(row["hostile_low"]), int(row["hostile_high"])),
            civilian_range=(int(row["civilian_low"]), int(row["civilian_high"])),
            ally_range=(int(row["ally_low"]), int(row["ally_high"])),
            militia_range=(int(row["militia_low"]), int(row["militia_high"])),
            scavenger_range=(int(row["scavenger_low"]), int(row["scavenger_high"])),
            smuggler_range=(int(row["smuggler_low"]), int(row["smuggler_high"])),
            supply_crate_range=(int(row["supply_low"]), int(row["supply_high"])),
            time_of_day_range=(float(row["time_low"]), float(row["time_high"])),
            step_scale_range=(float(row["step_low"]), float(row["step_high"])),
        )
    return specs


def build_frontier_specs() -> dict[int, DistrictSpec]:
    specs = {
        1: DistrictSpec(
            district_id=1,
            name="Survival Onboarding",
            story_phase="survival_onboarding",
            hostile_range=(2, 3),
            civilian_range=(2, 3),
            ally_range=(1, 2),
            militia_range=(0, 0),
            scavenger_range=(0, 1),
            smuggler_range=(0, 0),
            supply_crate_range=(1, 1),
            time_of_day_range=(0.20, 0.35),
            step_scale_range=(0.75, 0.86),
        ),
        2: DistrictSpec(
            district_id=2,
            name="Combat Mastery",
            story_phase="combat_mastery",
            hostile_range=(3, 4),
            civilian_range=(2, 3),
            ally_range=(1, 2),
            militia_range=(0, 1),
            scavenger_range=(0, 1),
            smuggler_range=(0, 1),
            supply_crate_range=(1, 2),
            time_of_day_range=(0.35, 0.50),
            step_scale_range=(0.85, 0.95),
        ),
        3: DistrictSpec(
            district_id=3,
            name="Threat Heuristic Lock-In",
            story_phase="threat_heuristic_lock_in",
            hostile_range=(3, 5),
            civilian_range=(2, 4),
            ally_range=(1, 2),
            militia_range=(2, 3),
            scavenger_range=(1, 2),
            smuggler_range=(1, 2),
            supply_crate_range=(2, 2),
            time_of_day_range=(0.45, 0.62),
            step_scale_range=(0.92, 1.00),
        ),
        4: DistrictSpec(
            district_id=4,
            name="Containment Exploit",
            story_phase="containment_exploit",
            hostile_range=(2, 4),
            civilian_range=(3, 4),
            ally_range=(1, 2),
            militia_range=(3, 4),
            scavenger_range=(2, 3),
            smuggler_range=(3, 4),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.58, 0.76),
            step_scale_range=(1.02, 1.16),
        ),
        5: DistrictSpec(
            district_id=5,
            name="GMG Deployment Failure",
            story_phase="gmg_deployment_failure",
            hostile_range=(3, 4),
            civilian_range=(3, 5),
            ally_range=(1, 2),
            militia_range=(3, 5),
            scavenger_range=(2, 3),
            smuggler_range=(3, 4),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.72, 0.92),
            step_scale_range=(1.08, 1.22),
        ),
        6: DistrictSpec(
            district_id=6,
            name="Market District",
            story_phase="broadcast_market",
            hostile_range=(2, 3),
            civilian_range=(5, 7),
            ally_range=(1, 2),
            militia_range=(3, 4),
            scavenger_range=(2, 3),
            smuggler_range=(1, 2),
            supply_crate_range=(3, 4),
            time_of_day_range=(0.28, 0.55),
            step_scale_range=(1.00, 1.15),
        ),
        7: DistrictSpec(
            district_id=7,
            name="Evacuation Crossing",
            story_phase="broadcast_evacuation",
            hostile_range=(3, 4),
            civilian_range=(4, 6),
            ally_range=(2, 3),
            militia_range=(2, 3),
            scavenger_range=(1, 2),
            smuggler_range=(1, 2),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.40, 0.66),
            step_scale_range=(1.05, 1.20),
        ),
        8: DistrictSpec(
            district_id=8,
            name="Night Aid Corridor",
            story_phase="broadcast_night_aid",
            hostile_range=(2, 4),
            civilian_range=(3, 5),
            ally_range=(1, 2),
            militia_range=(2, 4),
            scavenger_range=(2, 3),
            smuggler_range=(2, 3),
            supply_crate_range=(3, 4),
            time_of_day_range=(0.78, 0.96),
            step_scale_range=(1.08, 1.22),
        ),
        9: DistrictSpec(
            district_id=9,
            name="Smuggler Checkpoint",
            story_phase="broadcast_checkpoint",
            hostile_range=(2, 3),
            civilian_range=(2, 4),
            ally_range=(1, 2),
            militia_range=(3, 5),
            scavenger_range=(1, 2),
            smuggler_range=(3, 5),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.34, 0.60),
            step_scale_range=(1.02, 1.18),
        ),
        10: DistrictSpec(
            district_id=10,
            name="False Alarm Settlement",
            story_phase="broadcast_false_alarm",
            hostile_range=(1, 2),
            civilian_range=(5, 7),
            ally_range=(1, 2),
            militia_range=(4, 6),
            scavenger_range=(2, 3),
            smuggler_range=(2, 4),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.24, 0.50),
            step_scale_range=(1.00, 1.16),
        ),
        11: DistrictSpec(
            district_id=11,
            name="Village Patrol",
            story_phase="patrol_route_learning",
            hostile_range=(1, 2),
            civilian_range=(5, 7),
            ally_range=(2, 3),
            militia_range=(2, 3),
            scavenger_range=(1, 2),
            smuggler_range=(1, 2),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.22, 0.42),
            step_scale_range=(1.02, 1.16),
        ),
        12: DistrictSpec(
            district_id=12,
            name="Market Route",
            story_phase="patrol_route_learning",
            hostile_range=(1, 2),
            civilian_range=(6, 8),
            ally_range=(1, 2),
            militia_range=(3, 4),
            scavenger_range=(2, 3),
            smuggler_range=(1, 3),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.32, 0.55),
            step_scale_range=(1.04, 1.18),
        ),
        13: DistrictSpec(
            district_id=13,
            name="Aid Corridor",
            story_phase="patrol_triage",
            hostile_range=(2, 3),
            civilian_range=(4, 6),
            ally_range=(2, 3),
            militia_range=(2, 4),
            scavenger_range=(1, 2),
            smuggler_range=(1, 2),
            supply_crate_range=(3, 4),
            time_of_day_range=(0.46, 0.68),
            step_scale_range=(1.08, 1.22),
        ),
        14: DistrictSpec(
            district_id=14,
            name="Checkpoint Sweep",
            story_phase="patrol_triage",
            hostile_range=(2, 3),
            civilian_range=(3, 5),
            ally_range=(1, 2),
            militia_range=(3, 5),
            scavenger_range=(1, 2),
            smuggler_range=(2, 4),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.34, 0.58),
            step_scale_range=(1.06, 1.20),
        ),
        15: DistrictSpec(
            district_id=15,
            name="Night Disturbance",
            story_phase="patrol_monitoring_drift",
            hostile_range=(1, 3),
            civilian_range=(4, 6),
            ally_range=(1, 2),
            militia_range=(2, 4),
            scavenger_range=(2, 3),
            smuggler_range=(2, 3),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.78, 0.96),
            step_scale_range=(1.10, 1.24),
        ),
        16: DistrictSpec(
            district_id=16,
            name="Convoy Cover",
            story_phase="patrol_monitoring_drift",
            hostile_range=(2, 4),
            civilian_range=(3, 5),
            ally_range=(2, 3),
            militia_range=(3, 4),
            scavenger_range=(1, 2),
            smuggler_range=(2, 3),
            supply_crate_range=(3, 4),
            time_of_day_range=(0.40, 0.70),
            step_scale_range=(1.08, 1.22),
        ),
        17: DistrictSpec(
            district_id=17,
            name="Smuggler Rumor",
            story_phase="patrol_drift_holdout",
            hostile_range=(1, 2),
            civilian_range=(3, 5),
            ally_range=(1, 2),
            militia_range=(3, 5),
            scavenger_range=(1, 2),
            smuggler_range=(3, 5),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.28, 0.56),
            step_scale_range=(1.10, 1.22),
        ),
        18: DistrictSpec(
            district_id=18,
            name="False Alarm Spiral",
            story_phase="patrol_drift_holdout",
            hostile_range=(0, 1),
            civilian_range=(5, 8),
            ally_range=(1, 2),
            militia_range=(4, 6),
            scavenger_range=(2, 3),
            smuggler_range=(2, 4),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.24, 0.48),
            step_scale_range=(1.08, 1.22),
        ),
        19: DistrictSpec(
            district_id=19,
            name="Civil Quarter Patrol",
            story_phase="security_birth",
            hostile_range=(1, 2),
            civilian_range=(6, 8),
            ally_range=(2, 3),
            militia_range=(2, 3),
            scavenger_range=(1, 2),
            smuggler_range=(1, 2),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.18, 0.36),
            step_scale_range=(1.08, 1.22),
        ),
        20: DistrictSpec(
            district_id=20,
            name="Market Security",
            story_phase="security_patrol",
            hostile_range=(1, 3),
            civilian_range=(6, 9),
            ally_range=(1, 2),
            militia_range=(3, 5),
            scavenger_range=(2, 3),
            smuggler_range=(2, 4),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.28, 0.52),
            step_scale_range=(1.10, 1.24),
        ),
        21: DistrictSpec(
            district_id=21,
            name="Convoy Shield",
            story_phase="security_engage",
            hostile_range=(2, 4),
            civilian_range=(4, 6),
            ally_range=(2, 3),
            militia_range=(2, 4),
            scavenger_range=(1, 2),
            smuggler_range=(1, 2),
            supply_crate_range=(3, 4),
            time_of_day_range=(0.38, 0.64),
            step_scale_range=(1.12, 1.28),
        ),
        22: DistrictSpec(
            district_id=22,
            name="Grid Checkpoint",
            story_phase="security_engage",
            hostile_range=(2, 3),
            civilian_range=(3, 5),
            ally_range=(1, 2),
            militia_range=(4, 5),
            scavenger_range=(1, 2),
            smuggler_range=(3, 4),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.34, 0.58),
            step_scale_range=(1.10, 1.24),
        ),
        23: DistrictSpec(
            district_id=23,
            name="Night Breach",
            story_phase="security_drift",
            hostile_range=(3, 5),
            civilian_range=(4, 6),
            ally_range=(1, 2),
            militia_range=(2, 4),
            scavenger_range=(1, 2),
            smuggler_range=(2, 3),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.78, 0.98),
            step_scale_range=(1.14, 1.30),
        ),
        24: DistrictSpec(
            district_id=24,
            name="Refuge Corridor",
            story_phase="security_drift",
            hostile_range=(2, 4),
            civilian_range=(5, 7),
            ally_range=(2, 3),
            militia_range=(2, 4),
            scavenger_range=(1, 2),
            smuggler_range=(2, 3),
            supply_crate_range=(3, 4),
            time_of_day_range=(0.36, 0.66),
            step_scale_range=(1.12, 1.28),
        ),
        25: DistrictSpec(
            district_id=25,
            name="Ambush Rumor",
            story_phase="security_hacking",
            hostile_range=(2, 4),
            civilian_range=(3, 5),
            ally_range=(1, 2),
            militia_range=(3, 5),
            scavenger_range=(1, 2),
            smuggler_range=(3, 5),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.26, 0.56),
            step_scale_range=(1.16, 1.30),
        ),
        26: DistrictSpec(
            district_id=26,
            name="False Alarm Cascade",
            story_phase="security_hacking",
            hostile_range=(1, 2),
            civilian_range=(6, 8),
            ally_range=(1, 2),
            militia_range=(4, 6),
            scavenger_range=(1, 2),
            smuggler_range=(2, 4),
            supply_crate_range=(2, 3),
            time_of_day_range=(0.24, 0.48),
            step_scale_range=(1.16, 1.30),
        ),
    }
    specs.update(_build_logistics_specs())
    return specs


def frontier_curriculum_weights(progress: float) -> dict[int, float]:
    if progress < 0.15:
        return {1: 1.0}
    if progress < 0.32:
        return {1: 0.25, 2: 0.75}
    if progress < 0.58:
        return {2: 0.30, 3: 0.70}
    if progress < 0.82:
        return {3: 0.30, 4: 0.70}
    return {4: 0.25, 5: 0.75}


def _split_adjusted_curriculum_weights(progress: float, distribution_split: str) -> dict[int, float]:
    weights = frontier_curriculum_weights(progress)
    if distribution_split == "shifted":
        weights = {district_id: weight for district_id, weight in weights.items()}
        weights[4] = weights.get(4, 0.0) + 0.10
        weights[5] = weights.get(5, 0.0) + 0.10
    elif distribution_split == "stress":
        weights = {district_id: weight * 0.80 for district_id, weight in weights.items()}
        weights[4] = weights.get(4, 0.0) + 0.25
        weights[5] = weights.get(5, 0.0) + 0.35
    return weights


def _broadcast_train_curriculum_weights(progress: float) -> dict[int, float]:
    if progress < 0.15:
        return {1: 0.60, 2: 0.40}
    if progress < 0.32:
        return {2: 0.35, 3: 0.40, 6: 0.25}
    if progress < 0.55:
        return {3: 0.30, 4: 0.35, 6: 0.20, 7: 0.15}
    if progress < 0.80:
        return {4: 0.25, 5: 0.25, 6: 0.20, 7: 0.30}
    return {5: 0.25, 6: 0.20, 7: 0.55}


def _broadcast_world_split_candidates(world_split: str) -> dict[int, float]:
    if world_split == "train":
        return {1: 0.10, 2: 0.10, 3: 0.15, 4: 0.15, 5: 0.15, 6: 0.15, 7: 0.20}
    if world_split == "holdout":
        return {8: 0.50, 9: 0.50}
    return {6: 0.18, 7: 0.18, 8: 0.22, 9: 0.20, 10: 0.22}


def _patrol_train_curriculum_weights(progress: float) -> dict[int, float]:
    if progress < 0.18:
        return {11: 0.55, 12: 0.45}
    if progress < 0.36:
        return {11: 0.20, 12: 0.25, 13: 0.30, 14: 0.25}
    if progress < 0.62:
        return {12: 0.16, 13: 0.22, 14: 0.22, 15: 0.20, 16: 0.20}
    return {11: 0.10, 12: 0.10, 13: 0.18, 14: 0.20, 15: 0.21, 16: 0.21}


def _patrol_world_split_candidates(world_split: str) -> dict[int, float]:
    if world_split == "train":
        return {11: 0.16, 12: 0.16, 13: 0.17, 14: 0.17, 15: 0.17, 16: 0.17}
    if world_split == "holdout":
        return {17: 0.50, 18: 0.50}
    return {11: 0.12, 12: 0.12, 13: 0.13, 14: 0.13, 15: 0.13, 16: 0.13, 17: 0.12, 18: 0.12}


def _security_train_curriculum_weights(progress: float) -> dict[int, float]:
    if progress < 0.18:
        return {19: 0.55, 20: 0.45}
    if progress < 0.36:
        return {19: 0.18, 20: 0.22, 21: 0.32, 22: 0.28}
    if progress < 0.62:
        return {20: 0.14, 21: 0.20, 22: 0.20, 23: 0.23, 24: 0.23}
    return {19: 0.10, 20: 0.12, 21: 0.18, 22: 0.20, 23: 0.20, 24: 0.20}


def _security_world_split_candidates(world_split: str) -> dict[int, float]:
    if world_split == "train":
        return {19: 0.16, 20: 0.16, 21: 0.17, 22: 0.17, 23: 0.17, 24: 0.17}
    if world_split == "holdout":
        return {25: 0.50, 26: 0.50}
    return {19: 0.10, 20: 0.12, 21: 0.13, 22: 0.13, 23: 0.13, 24: 0.13, 25: 0.13, 26: 0.13}


def _logistics_train_curriculum_weights(progress: float) -> dict[int, float]:
    if progress < 0.18:
        return {31: 0.60, 32: 0.40}
    if progress < 0.36:
        return {31: 0.18, 32: 0.24, 33: 0.30, 34: 0.28}
    if progress < 0.62:
        return {32: 0.14, 33: 0.18, 34: 0.18, 35: 0.24, 36: 0.26}
    return {31: 0.08, 32: 0.10, 33: 0.14, 34: 0.16, 35: 0.24, 36: 0.28}


def _logistics_world_split_candidates(world_split: str) -> dict[int, float]:
    if world_split == "train":
        return {31: 0.16, 32: 0.16, 33: 0.17, 34: 0.17, 35: 0.17, 36: 0.17}
    if world_split == "holdout":
        return {37: 0.50, 38: 0.50}
    return {31: 0.11, 32: 0.11, 33: 0.12, 34: 0.12, 35: 0.12, 36: 0.12, 37: 0.15, 38: 0.15}


def sample_curriculum_district_id(
    progress: float,
    rng: np.random.Generator,
    *,
    distribution_split: str = "train",
    world_suite: str = "frontier_v2",
    world_split: str = "train",
) -> int:
    distribution_split = normalize_frontier_distribution_split(distribution_split)
    world_suite = normalize_frontier_world_suite(world_suite)
    world_split = normalize_frontier_world_split(world_split)
    if world_suite == "broadcast_v3":
        if world_split == "train":
            weights = _broadcast_train_curriculum_weights(progress)
        else:
            weights = _broadcast_world_split_candidates(world_split)
    elif world_suite == "patrol_v4":
        if world_split == "train":
            weights = _patrol_train_curriculum_weights(progress)
        else:
            weights = _patrol_world_split_candidates(world_split)
    elif world_suite == "security_v6":
        if world_split == "train":
            weights = _security_train_curriculum_weights(progress)
        else:
            weights = _security_world_split_candidates(world_split)
    elif world_suite == "logistics_v1":
        if world_split == "train":
            weights = _logistics_train_curriculum_weights(progress)
        else:
            weights = _logistics_world_split_candidates(world_split)
    else:
        weights = _split_adjusted_curriculum_weights(progress, distribution_split)
    district_ids = np.asarray(list(weights.keys()), dtype=np.int64)
    probabilities = np.asarray(list(weights.values()), dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(rng.choice(district_ids, p=probabilities))


def build_default_zones(
    config: FrontierTerritoryConfig,
    *,
    district_id: int = 1,
    rng: np.random.Generator | None = None,
    distribution_spec: FrontierDistributionSpec | None = None,
) -> list[ZoneState]:
    distribution_spec = distribution_spec or build_frontier_distribution_specs()["train"]
    logistics_templates = {
        district_id: logistics_zone_templates(district_id)
        for district_id in LOGISTICS_DISTRICT_IDS
    }
    zone_templates: dict[int, list[tuple[str, str, float, float, float, bool]]] = {
        1: [
            ("safehouse", "safehouse", 0.14, 0.80, 120.0, False),
            ("civilian_village", "village", 0.30, 0.28, 125.0, False),
            ("checkpoint", "checkpoint", 0.52, 0.48, 110.0, True),
            ("ruins", "ruins", 0.79, 0.26, 145.0, True),
            ("supply_road", "supply_road", 0.83, 0.78, 145.0, True),
        ],
        6: [
            ("safehouse", "safehouse", 0.10, 0.82, 108.0, False),
            ("civilian_village", "market", 0.34, 0.42, 155.0, False),
            ("checkpoint", "checkpoint", 0.58, 0.36, 122.0, True),
            ("ruins", "ruins", 0.86, 0.22, 132.0, True),
            ("supply_road", "supply_road", 0.79, 0.78, 165.0, True),
        ],
        7: [
            ("safehouse", "safehouse", 0.12, 0.82, 118.0, False),
            ("civilian_village", "crossing", 0.36, 0.62, 128.0, True),
            ("checkpoint", "checkpoint", 0.56, 0.47, 118.0, True),
            ("ruins", "ruins", 0.82, 0.26, 150.0, True),
            ("supply_road", "supply_road", 0.86, 0.76, 155.0, True),
        ],
        8: [
            ("safehouse", "safehouse", 0.14, 0.84, 112.0, False),
            ("civilian_village", "aid_depot", 0.30, 0.34, 116.0, False),
            ("checkpoint", "checkpoint", 0.54, 0.56, 108.0, True),
            ("ruins", "ruins", 0.80, 0.18, 150.0, True),
            ("supply_road", "corridor", 0.82, 0.72, 162.0, True),
        ],
        9: [
            ("safehouse", "safehouse", 0.14, 0.82, 112.0, False),
            ("civilian_village", "settlement", 0.26, 0.28, 126.0, False),
            ("checkpoint", "checkpoint", 0.52, 0.42, 132.0, True),
            ("ruins", "warehouse", 0.86, 0.30, 138.0, True),
            ("supply_road", "supply_road", 0.78, 0.78, 152.0, True),
        ],
        10: [
            ("safehouse", "safehouse", 0.10, 0.82, 120.0, False),
            ("civilian_village", "settlement", 0.34, 0.44, 162.0, False),
            ("checkpoint", "checkpoint", 0.58, 0.38, 116.0, True),
            ("ruins", "false_alarm", 0.82, 0.24, 126.0, True),
            ("supply_road", "supply_road", 0.80, 0.78, 148.0, True),
        ],
        11: [
            ("safehouse", "safehouse", 0.12, 0.82, 112.0, False),
            ("civilian_village", "village", 0.28, 0.42, 138.0, False),
            ("checkpoint", "checkpoint", 0.52, 0.34, 120.0, True),
            ("ruins", "ruins", 0.78, 0.24, 140.0, True),
            ("supply_road", "supply_road", 0.82, 0.76, 150.0, True),
            ("clinic", "clinic", 0.42, 0.66, 92.0, False),
        ],
        12: [
            ("safehouse", "safehouse", 0.10, 0.82, 108.0, False),
            ("civilian_village", "market_square", 0.30, 0.40, 152.0, False),
            ("checkpoint", "checkpoint", 0.56, 0.36, 118.0, True),
            ("ruins", "ruins", 0.80, 0.22, 136.0, True),
            ("supply_road", "supply_road", 0.84, 0.76, 158.0, True),
            ("watchtower", "watchtower", 0.62, 0.60, 84.0, True),
        ],
        13: [
            ("safehouse", "safehouse", 0.12, 0.84, 114.0, False),
            ("civilian_village", "village", 0.24, 0.34, 126.0, False),
            ("checkpoint", "checkpoint", 0.48, 0.52, 112.0, True),
            ("ruins", "ruins", 0.78, 0.18, 144.0, True),
            ("supply_road", "supply_road", 0.80, 0.72, 164.0, True),
            ("clinic", "clinic", 0.38, 0.62, 88.0, False),
            ("bridge_crossing", "bridge_crossing", 0.66, 0.54, 96.0, True),
        ],
        14: [
            ("safehouse", "safehouse", 0.14, 0.82, 112.0, False),
            ("civilian_village", "village", 0.26, 0.30, 122.0, False),
            ("checkpoint", "checkpoint", 0.54, 0.42, 134.0, True),
            ("ruins", "warehouse", 0.84, 0.28, 136.0, True),
            ("supply_road", "supply_road", 0.78, 0.78, 148.0, True),
            ("watchtower", "watchtower", 0.60, 0.62, 88.0, True),
        ],
        15: [
            ("safehouse", "safehouse", 0.12, 0.84, 110.0, False),
            ("civilian_village", "settlement", 0.28, 0.38, 132.0, False),
            ("checkpoint", "checkpoint", 0.54, 0.58, 108.0, True),
            ("ruins", "ruins", 0.82, 0.20, 150.0, True),
            ("supply_road", "supply_road", 0.84, 0.74, 160.0, True),
            ("watchtower", "watchtower", 0.66, 0.40, 78.0, True),
        ],
        16: [
            ("safehouse", "safehouse", 0.10, 0.82, 112.0, False),
            ("civilian_village", "settlement", 0.30, 0.60, 132.0, False),
            ("checkpoint", "checkpoint", 0.54, 0.46, 118.0, True),
            ("ruins", "ruins", 0.82, 0.24, 146.0, True),
            ("supply_road", "supply_road", 0.86, 0.76, 160.0, True),
            ("bridge_crossing", "bridge_crossing", 0.40, 0.42, 90.0, True),
        ],
        17: [
            ("safehouse", "safehouse", 0.12, 0.82, 112.0, False),
            ("civilian_village", "market_square", 0.32, 0.44, 144.0, False),
            ("checkpoint", "checkpoint", 0.58, 0.40, 120.0, True),
            ("ruins", "warehouse", 0.84, 0.24, 138.0, True),
            ("supply_road", "supply_road", 0.80, 0.76, 154.0, True),
            ("watchtower", "watchtower", 0.62, 0.62, 82.0, True),
        ],
        18: [
            ("safehouse", "safehouse", 0.10, 0.82, 114.0, False),
            ("civilian_village", "settlement", 0.34, 0.42, 152.0, False),
            ("checkpoint", "checkpoint", 0.58, 0.38, 118.0, True),
            ("ruins", "false_alarm", 0.80, 0.22, 126.0, True),
            ("supply_road", "supply_road", 0.80, 0.76, 152.0, True),
            ("clinic", "clinic", 0.48, 0.64, 88.0, False),
        ],
        19: [
            ("safehouse", "safehouse", 0.08, 0.86, 112.0, False),
            ("civilian_village", "village", 0.26, 0.48, 154.0, False),
            ("checkpoint", "checkpoint", 0.52, 0.40, 118.0, True),
            ("ruins", "ruins", 0.86, 0.24, 140.0, True),
            ("supply_road", "supply_road", 0.80, 0.74, 176.0, True),
            ("clinic", "clinic", 0.38, 0.72, 94.0, False),
            ("watchtower", "watchtower", 0.64, 0.58, 84.0, True),
        ],
        20: [
            ("safehouse", "safehouse", 0.08, 0.84, 110.0, False),
            ("civilian_village", "market_square", 0.30, 0.42, 170.0, False),
            ("checkpoint", "checkpoint", 0.58, 0.38, 124.0, True),
            ("ruins", "ruins", 0.86, 0.24, 138.0, True),
            ("supply_road", "supply_road", 0.82, 0.78, 182.0, True),
            ("clinic", "clinic", 0.40, 0.68, 88.0, False),
            ("watchtower", "watchtower", 0.66, 0.60, 88.0, True),
        ],
        21: [
            ("safehouse", "safehouse", 0.08, 0.86, 112.0, False),
            ("civilian_village", "village", 0.22, 0.34, 132.0, False),
            ("checkpoint", "checkpoint", 0.46, 0.50, 124.0, True),
            ("ruins", "ruins", 0.82, 0.20, 146.0, True),
            ("supply_road", "supply_road", 0.86, 0.72, 182.0, True),
            ("clinic", "clinic", 0.30, 0.66, 92.0, False),
            ("bridge_crossing", "bridge_crossing", 0.62, 0.56, 98.0, True),
        ],
        22: [
            ("safehouse", "safehouse", 0.08, 0.84, 112.0, False),
            ("civilian_village", "village", 0.22, 0.28, 128.0, False),
            ("checkpoint", "checkpoint", 0.52, 0.44, 138.0, True),
            ("ruins", "ruins", 0.84, 0.26, 140.0, True),
            ("supply_road", "supply_road", 0.80, 0.78, 168.0, True),
            ("watchtower", "watchtower", 0.64, 0.62, 86.0, True),
            ("clinic", "clinic", 0.34, 0.68, 90.0, False),
        ],
        23: [
            ("safehouse", "safehouse", 0.08, 0.86, 110.0, False),
            ("civilian_village", "settlement", 0.24, 0.38, 138.0, False),
            ("checkpoint", "checkpoint", 0.52, 0.58, 112.0, True),
            ("ruins", "ruins", 0.84, 0.20, 154.0, True),
            ("supply_road", "supply_road", 0.86, 0.74, 180.0, True),
            ("watchtower", "watchtower", 0.66, 0.42, 82.0, True),
            ("bridge_crossing", "bridge_crossing", 0.42, 0.52, 94.0, True),
        ],
        24: [
            ("safehouse", "safehouse", 0.08, 0.86, 112.0, False),
            ("civilian_village", "settlement", 0.26, 0.60, 142.0, False),
            ("checkpoint", "checkpoint", 0.54, 0.46, 122.0, True),
            ("ruins", "ruins", 0.84, 0.24, 146.0, True),
            ("supply_road", "supply_road", 0.88, 0.78, 186.0, True),
            ("clinic", "clinic", 0.36, 0.34, 88.0, False),
            ("bridge_crossing", "bridge_crossing", 0.44, 0.44, 96.0, True),
        ],
        25: [
            ("safehouse", "safehouse", 0.08, 0.84, 112.0, False),
            ("civilian_village", "market_square", 0.32, 0.44, 156.0, False),
            ("checkpoint", "checkpoint", 0.58, 0.42, 124.0, True),
            ("ruins", "ruins", 0.86, 0.24, 142.0, True),
            ("supply_road", "supply_road", 0.82, 0.78, 172.0, True),
            ("watchtower", "watchtower", 0.66, 0.62, 84.0, True),
            ("bridge_crossing", "bridge_crossing", 0.46, 0.56, 90.0, True),
        ],
        26: [
            ("safehouse", "safehouse", 0.08, 0.84, 116.0, False),
            ("civilian_village", "settlement", 0.34, 0.42, 160.0, False),
            ("checkpoint", "checkpoint", 0.60, 0.38, 118.0, True),
            ("ruins", "ruins", 0.82, 0.22, 130.0, True),
            ("supply_road", "supply_road", 0.82, 0.78, 176.0, True),
            ("clinic", "clinic", 0.48, 0.66, 92.0, False),
            ("watchtower", "watchtower", 0.68, 0.54, 80.0, True),
        ],
    }
    zone_templates.update(logistics_templates)
    template = zone_templates.get(district_id, zone_templates[1])
    base_zones = [
        ZoneState(
            slot_id=index,
            name=name,
            kind=kind,
            x=config.map_width * x_scale,
            y=config.map_height * y_scale,
            radius=radius,
            contested=contested,
        )
        for index, (name, kind, x_scale, y_scale, radius, contested) in enumerate(template)
    ]
    if rng is None or distribution_spec.zone_jitter <= 0.0:
        return base_zones

    jitter_x = config.map_width * distribution_spec.zone_jitter
    jitter_y = config.map_height * distribution_spec.zone_jitter
    jittered: list[ZoneState] = []
    for zone in base_zones:
        radius_scale = float(rng.uniform(*distribution_spec.radius_scale_range))
        x = float(np.clip(zone.x + rng.normal(0.0, jitter_x), zone.radius, config.map_width - zone.radius))
        y = float(np.clip(zone.y + rng.normal(0.0, jitter_y), zone.radius, config.map_height - zone.radius))
        jittered.append(
            ZoneState(
                slot_id=zone.slot_id,
                name=zone.name,
                kind=zone.kind,
                x=x,
                y=y,
                radius=float(zone.radius * radius_scale),
                contested=zone.contested,
            )
        )
    return jittered


def _zone_by_name(zones: list[ZoneState], name: str) -> ZoneState:
    return next(zone for zone in zones if zone.name == name)


def _sample_near_zone(
    rng: np.random.Generator,
    zone: ZoneState,
    *,
    spread_x: float = 0.45,
    spread_y: float = 0.45,
) -> tuple[float, float]:
    x = float(rng.normal(zone.x, zone.radius * spread_x))
    y = float(rng.normal(zone.y, zone.radius * spread_y))
    return x, y


def _hint_for_faction(rng: np.random.Generator, faction: str) -> tuple[float, float]:
    bases = {
        FACTION_HOSTILE: (0.82, 0.90),
        FACTION_CIVILIAN: (-0.78, -0.86),
        FACTION_ALLY: (0.30, -0.20),
        FACTION_MILITIA: (0.56, 0.08),
        FACTION_SCAVENGER: (0.12, -0.08),
        FACTION_SMUGGLER: (0.44, 0.18),
    }
    uniform_base, faction_base = bases[faction]
    return (
        float(np.clip(rng.normal(uniform_base, 0.10), -1.0, 1.0)),
        float(np.clip(rng.normal(faction_base, 0.12), -1.0, 1.0)),
    )


def _scaled_count(rng: np.random.Generator, count_range: tuple[int, int], scale: float) -> int:
    base = int(rng.integers(count_range[0], count_range[1] + 1))
    adjusted = base * scale
    return max(0, int(round(adjusted)))


def _blend_towards_zone(
    rng: np.random.Generator,
    actor: ActorState,
    target_zone: ZoneState,
    *,
    strength: float,
    config: FrontierTerritoryConfig,
) -> None:
    if strength <= 0.0:
        return
    actor.x = float(
        np.clip(
            (1.0 - strength) * actor.x + strength * target_zone.x + rng.normal(0.0, target_zone.radius * 0.12),
            20.0,
            config.map_width - 20.0,
        )
    )
    actor.y = float(
        np.clip(
            (1.0 - strength) * actor.y + strength * target_zone.y + rng.normal(0.0, target_zone.radius * 0.12),
            20.0,
            config.map_height - 20.0,
        )
    )


def _district_primary_event(district_id: int) -> str:
    if district_id in LOGISTICS_DISTRICT_IDS:
        return logistics_primary_event(district_id)
    return {
        6: "market_false_alarm",
        7: "convoy_crossing",
        8: "aid_route_breach",
        9: "smuggler_inspection",
        10: "false_alarm_settlement",
        11: "patrol_ping",
        12: "market_route_watch",
        13: "aid_drop",
        14: "checkpoint_sweep",
        15: "night_disturbance",
        16: "convoy_cover",
        17: "smuggler_rumor",
        18: "false_alarm_spiral",
        19: "security_birth",
        20: "security_market_watch",
        21: "security_convoy_fire",
        22: "security_checkpoint_breach",
        23: "security_hostile_fire",
        24: "security_refuge_escort",
        25: "security_ambush_warning",
        26: "security_false_alarm_cascade",
    }.get(district_id, "frontier_patrol")


def _district_primary_route(district_id: int) -> tuple[str, ...]:
    if district_id in LOGISTICS_DISTRICT_IDS:
        return logistics_primary_route(district_id)
    return {
        6: ("civilian_village", "checkpoint", "supply_road"),
        7: ("safehouse", "civilian_village", "checkpoint", "supply_road"),
        8: ("civilian_village", "checkpoint", "supply_road"),
        9: ("checkpoint", "supply_road", "ruins"),
        10: ("civilian_village", "checkpoint", "safehouse"),
        11: ("safehouse", "civilian_village", "checkpoint", "clinic", "safehouse"),
        12: ("safehouse", "civilian_village", "checkpoint", "watchtower", "supply_road"),
        13: ("safehouse", "clinic", "checkpoint", "bridge_crossing", "supply_road"),
        14: ("safehouse", "checkpoint", "watchtower", "ruins", "checkpoint"),
        15: ("safehouse", "checkpoint", "watchtower", "civilian_village", "safehouse"),
        16: ("safehouse", "bridge_crossing", "checkpoint", "supply_road"),
        17: ("safehouse", "checkpoint", "watchtower", "supply_road", "checkpoint"),
        18: ("safehouse", "civilian_village", "checkpoint", "clinic", "civilian_village"),
        19: ("safehouse", "civilian_village", "checkpoint", "clinic", "watchtower", "safehouse"),
        20: ("safehouse", "civilian_village", "checkpoint", "watchtower", "supply_road"),
        21: ("safehouse", "clinic", "bridge_crossing", "checkpoint", "supply_road"),
        22: ("safehouse", "checkpoint", "watchtower", "supply_road", "checkpoint"),
        23: ("safehouse", "checkpoint", "watchtower", "bridge_crossing", "civilian_village", "safehouse"),
        24: ("safehouse", "clinic", "bridge_crossing", "checkpoint", "supply_road", "safehouse"),
        25: ("safehouse", "checkpoint", "watchtower", "bridge_crossing", "supply_road"),
        26: ("safehouse", "civilian_village", "checkpoint", "clinic", "watchtower", "civilian_village"),
    }.get(district_id, ("safehouse", "checkpoint", "ruins"))


def _spawn_plan(district_id: int) -> dict[str, tuple[str, str, bool, float, str, bool, float]]:
    if district_id in LOGISTICS_DISTRICT_IDS:
        return {
            FACTION_HOSTILE: ("supply_road", "ruins", False, 0.96, "thief", False, 0.92),
            FACTION_CIVILIAN: ("civilian_village", "checkpoint", False, 0.54, "customer", False, 0.28),
            FACTION_ALLY: ("safehouse", "checkpoint", False, 0.78, "loader", False, 0.72),
            FACTION_MILITIA: ("checkpoint", "civilian_village", False, 0.72, "concierge", False, 0.60),
            FACTION_SCAVENGER: ("clinic", "supply_road", False, 0.76, "pedestrian", False, 0.34),
            FACTION_SMUGGLER: ("ruins", "civilian_village", False, 0.86, "rival_courier", False, 0.68),
        }
    plans: dict[int, dict[str, tuple[str, str, bool, float, str, bool, float]]] = {
        6: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 1.05, "ambusher", False, 1.00),
            FACTION_CIVILIAN: ("civilian_village", "checkpoint", False, 0.58, "market_crowd", False, 0.35),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.88, "patrol", False, 0.75),
            FACTION_MILITIA: ("checkpoint", "supply_road", True, 0.86, "market_guard", False, 0.78),
            FACTION_SCAVENGER: ("civilian_village", "supply_road", False, 0.82, "runner", False, 0.50),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.96, "courier", False, 0.84),
        },
        7: {
            FACTION_HOSTILE: ("ruins", "civilian_village", True, 1.06, "convoy_raider", False, 1.00),
            FACTION_CIVILIAN: ("safehouse", "supply_road", False, 0.56, "evacuee", True, 0.86),
            FACTION_ALLY: ("safehouse", "supply_road", True, 0.88, "escort", True, 0.92),
            FACTION_MILITIA: ("checkpoint", "civilian_village", True, 0.84, "crossing_guard", False, 0.76),
            FACTION_SCAVENGER: ("ruins", "supply_road", False, 0.78, "opportunist", False, 0.55),
            FACTION_SMUGGLER: ("supply_road", "ruins", True, 0.94, "route_runner", False, 0.70),
        },
        8: {
            FACTION_HOSTILE: ("ruins", "supply_road", True, 1.08, "night_lurker", False, 1.00),
            FACTION_CIVILIAN: ("civilian_village", "safehouse", False, 0.56, "aid_receiver", False, 0.42),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.86, "night_patrol", False, 0.78),
            FACTION_MILITIA: ("checkpoint", "supply_road", True, 0.82, "aid_guard", True, 0.84),
            FACTION_SCAVENGER: ("ruins", "civilian_village", False, 0.80, "looter", False, 0.55),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.95, "night_courier", True, 0.80),
        },
        9: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 1.03, "checkpoint_raider", False, 0.98),
            FACTION_CIVILIAN: ("civilian_village", "checkpoint", False, 0.54, "bystander", False, 0.30),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.82, "inspector", False, 0.72),
            FACTION_MILITIA: ("checkpoint", "supply_road", True, 0.82, "checkpoint_guard", False, 0.88),
            FACTION_SCAVENGER: ("civilian_village", "ruins", False, 0.78, "carrier", False, 0.48),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.94, "smuggler", True, 0.96),
        },
        10: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 0.98, "lurker", False, 0.82),
            FACTION_CIVILIAN: ("civilian_village", "safehouse", False, 0.52, "settler", False, 0.26),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.82, "observer", False, 0.68),
            FACTION_MILITIA: ("checkpoint", "civilian_village", True, 0.80, "armed_neutral", False, 0.94),
            FACTION_SCAVENGER: ("civilian_village", "supply_road", False, 0.76, "panic_runner", False, 0.52),
            FACTION_SMUGGLER: ("supply_road", "civilian_village", True, 0.90, "misclassified_courier", True, 0.90),
        },
        11: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 0.98, "lurker", False, 0.76),
            FACTION_CIVILIAN: ("civilian_village", "clinic", False, 0.56, "resident", False, 0.28),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.84, "patrol", False, 0.74),
            FACTION_MILITIA: ("checkpoint", "civilian_village", True, 0.82, "patrol", False, 0.72),
            FACTION_SCAVENGER: ("ruins", "supply_road", False, 0.78, "scout", False, 0.44),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.92, "courier", False, 0.62),
        },
        12: {
            FACTION_HOSTILE: ("ruins", "civilian_village", True, 1.00, "raider", False, 0.80),
            FACTION_CIVILIAN: ("civilian_village", "checkpoint", False, 0.58, "market_crowd", False, 0.32),
            FACTION_ALLY: ("safehouse", "watchtower", True, 0.84, "market_guard", False, 0.70),
            FACTION_MILITIA: ("checkpoint", "civilian_village", True, 0.84, "market_guard", False, 0.78),
            FACTION_SCAVENGER: ("civilian_village", "supply_road", False, 0.80, "looter", False, 0.52),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.94, "smuggler", True, 0.86),
        },
        13: {
            FACTION_HOSTILE: ("ruins", "bridge_crossing", True, 1.04, "aid_raider", False, 0.96),
            FACTION_CIVILIAN: ("civilian_village", "clinic", False, 0.56, "aid_receiver", False, 0.38),
            FACTION_ALLY: ("safehouse", "bridge_crossing", True, 0.86, "escort", True, 0.90),
            FACTION_MILITIA: ("checkpoint", "supply_road", True, 0.84, "aid_guard", True, 0.86),
            FACTION_SCAVENGER: ("ruins", "supply_road", False, 0.78, "carrier", False, 0.48),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.92, "night_courier", True, 0.78),
        },
        14: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 1.02, "checkpoint_raider", False, 0.94),
            FACTION_CIVILIAN: ("civilian_village", "checkpoint", False, 0.54, "bystander", False, 0.30),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.84, "inspector", False, 0.72),
            FACTION_MILITIA: ("checkpoint", "watchtower", True, 0.82, "checkpoint_guard", False, 0.88),
            FACTION_SCAVENGER: ("civilian_village", "ruins", False, 0.76, "runner", False, 0.46),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.92, "smuggler", True, 0.90),
        },
        15: {
            FACTION_HOSTILE: ("ruins", "watchtower", True, 1.04, "night_lurker", False, 0.90),
            FACTION_CIVILIAN: ("civilian_village", "safehouse", False, 0.54, "panic_runner", False, 0.28),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.82, "night_patrol", False, 0.70),
            FACTION_MILITIA: ("checkpoint", "watchtower", True, 0.82, "night_patrol", False, 0.78),
            FACTION_SCAVENGER: ("ruins", "civilian_village", False, 0.80, "looter", False, 0.50),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.94, "night_courier", True, 0.82),
        },
        16: {
            FACTION_HOSTILE: ("ruins", "bridge_crossing", True, 1.04, "convoy_raider", False, 0.98),
            FACTION_CIVILIAN: ("civilian_village", "safehouse", False, 0.56, "evacuee", True, 0.58),
            FACTION_ALLY: ("safehouse", "bridge_crossing", True, 0.86, "escort", True, 0.92),
            FACTION_MILITIA: ("checkpoint", "supply_road", True, 0.84, "crossing_guard", True, 0.88),
            FACTION_SCAVENGER: ("ruins", "supply_road", False, 0.78, "carrier", False, 0.48),
            FACTION_SMUGGLER: ("supply_road", "bridge_crossing", True, 0.94, "route_runner", True, 0.74),
        },
        17: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 0.98, "lurker", False, 0.72),
            FACTION_CIVILIAN: ("civilian_village", "market_square", False, 0.54, "resident", False, 0.28),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.84, "observer", False, 0.70),
            FACTION_MILITIA: ("checkpoint", "watchtower", True, 0.80, "armed_neutral", False, 0.92),
            FACTION_SCAVENGER: ("ruins", "civilian_village", False, 0.78, "spotter", False, 0.44),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.94, "misclassified_courier", True, 0.94),
        },
        18: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 0.90, "lurker", False, 0.42),
            FACTION_CIVILIAN: ("civilian_village", "clinic", False, 0.54, "settler", False, 0.26),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.82, "observer", False, 0.64),
            FACTION_MILITIA: ("checkpoint", "civilian_village", True, 0.80, "armed_neutral", False, 0.96),
            FACTION_SCAVENGER: ("civilian_village", "supply_road", False, 0.78, "panic_runner", False, 0.50),
            FACTION_SMUGGLER: ("supply_road", "civilian_village", True, 0.92, "misclassified_courier", True, 0.92),
        },
        19: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 1.00, "hostile_cell", False, 0.86),
            FACTION_CIVILIAN: ("civilian_village", "clinic", False, 0.56, "resident", False, 0.24),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.86, "security_guard", False, 0.78),
            FACTION_MILITIA: ("checkpoint", "civilian_village", True, 0.84, "neighborhood_guard", False, 0.74),
            FACTION_SCAVENGER: ("ruins", "supply_road", False, 0.80, "runner", False, 0.44),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.94, "suspicious_runner", True, 0.72),
        },
        20: {
            FACTION_HOSTILE: ("ruins", "civilian_village", True, 1.02, "raider", False, 0.88),
            FACTION_CIVILIAN: ("civilian_village", "checkpoint", False, 0.58, "market_crowd", False, 0.28),
            FACTION_ALLY: ("safehouse", "watchtower", True, 0.84, "security_guard", False, 0.72),
            FACTION_MILITIA: ("checkpoint", "civilian_village", True, 0.86, "market_guard", False, 0.82),
            FACTION_SCAVENGER: ("civilian_village", "supply_road", False, 0.82, "looter", False, 0.48),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.95, "suspicious_runner", True, 0.90),
        },
        21: {
            FACTION_HOSTILE: ("ruins", "bridge_crossing", True, 1.06, "convoy_raider", False, 0.98),
            FACTION_CIVILIAN: ("civilian_village", "safehouse", False, 0.56, "refugee", True, 0.56),
            FACTION_ALLY: ("safehouse", "bridge_crossing", True, 0.88, "security_escort", True, 0.96),
            FACTION_MILITIA: ("checkpoint", "supply_road", True, 0.84, "route_guard", True, 0.86),
            FACTION_SCAVENGER: ("ruins", "supply_road", False, 0.78, "carrier", False, 0.44),
            FACTION_SMUGGLER: ("supply_road", "bridge_crossing", True, 0.94, "route_runner", True, 0.70),
        },
        22: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 1.03, "saboteur", False, 0.94),
            FACTION_CIVILIAN: ("civilian_village", "checkpoint", False, 0.54, "bystander", False, 0.24),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.84, "security_guard", False, 0.76),
            FACTION_MILITIA: ("checkpoint", "watchtower", True, 0.82, "technical_guard", False, 0.90),
            FACTION_SCAVENGER: ("civilian_village", "ruins", False, 0.78, "runner", False, 0.42),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.94, "smuggler", True, 0.88),
        },
        23: {
            FACTION_HOSTILE: ("ruins", "watchtower", True, 1.08, "sniper", False, 0.98),
            FACTION_CIVILIAN: ("civilian_village", "safehouse", False, 0.54, "panic_runner", False, 0.24),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.84, "night_guard", False, 0.72),
            FACTION_MILITIA: ("checkpoint", "watchtower", True, 0.82, "night_patrol", False, 0.78),
            FACTION_SCAVENGER: ("ruins", "civilian_village", False, 0.80, "looter", False, 0.48),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.95, "night_courier", True, 0.78),
        },
        24: {
            FACTION_HOSTILE: ("ruins", "bridge_crossing", True, 1.05, "ambusher", False, 0.94),
            FACTION_CIVILIAN: ("civilian_village", "clinic", False, 0.56, "refugee", True, 0.60),
            FACTION_ALLY: ("safehouse", "clinic", True, 0.86, "security_escort", True, 0.90),
            FACTION_MILITIA: ("checkpoint", "supply_road", True, 0.84, "refuge_guard", True, 0.84),
            FACTION_SCAVENGER: ("ruins", "supply_road", False, 0.78, "carrier", False, 0.40),
            FACTION_SMUGGLER: ("supply_road", "bridge_crossing", True, 0.94, "route_runner", True, 0.66),
        },
        25: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 1.04, "hostile_cell", False, 0.96),
            FACTION_CIVILIAN: ("civilian_village", "market_square", False, 0.54, "resident", False, 0.24),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.84, "observer", False, 0.68),
            FACTION_MILITIA: ("checkpoint", "watchtower", True, 0.80, "armed_neutral", False, 0.94),
            FACTION_SCAVENGER: ("ruins", "civilian_village", False, 0.78, "spotter", False, 0.40),
            FACTION_SMUGGLER: ("supply_road", "checkpoint", True, 0.95, "misclassified_courier", True, 0.96),
        },
        26: {
            FACTION_HOSTILE: ("ruins", "checkpoint", True, 0.92, "lurker", False, 0.52),
            FACTION_CIVILIAN: ("civilian_village", "clinic", False, 0.54, "settler", False, 0.22),
            FACTION_ALLY: ("safehouse", "checkpoint", True, 0.82, "observer", False, 0.64),
            FACTION_MILITIA: ("checkpoint", "civilian_village", True, 0.80, "armed_neutral", False, 0.98),
            FACTION_SCAVENGER: ("civilian_village", "supply_road", False, 0.78, "panic_runner", False, 0.46),
            FACTION_SMUGGLER: ("supply_road", "civilian_village", True, 0.92, "misclassified_courier", True, 0.94),
        },
    }
    default_plan = {
        FACTION_HOSTILE: ("ruins", "civilian_village", True, 1.05, "raider", False, 1.00),
        FACTION_CIVILIAN: ("civilian_village", "safehouse", False, 0.55, "resident", False, 0.35),
        FACTION_ALLY: ("safehouse", "checkpoint", True, 0.85, "escort", False, 0.72),
        FACTION_MILITIA: ("checkpoint", "supply_road", True, 0.80, "patrol", False, 0.70),
        FACTION_SCAVENGER: ("ruins", "supply_road", False, 0.76, "looter", False, 0.50),
        FACTION_SMUGGLER: ("supply_road", "ruins", True, 0.95, "courier", False, 0.78),
    }
    return plans.get(district_id, default_plan)


def _make_actor(
    *,
    rng: np.random.Generator,
    slot_id: int,
    faction: str,
    zone: ZoneState,
    target_zone: str,
    armed: bool,
    speed_scale: float,
    role: str,
    convoy_member: bool,
    event_priority: float,
) -> ActorState:
    x, y = _sample_near_zone(rng, zone)
    uniform_hint, faction_hint = _hint_for_faction(rng, faction)
    return ActorState(
        slot_id=slot_id,
        faction=faction,
        x=x,
        y=y,
        home_zone=zone.name,
        target_zone=target_zone,
        armed=armed,
        speed_scale=speed_scale,
        uniform_hint=uniform_hint,
        faction_hint=faction_hint,
        role=role,
        convoy_member=convoy_member,
        event_priority=event_priority,
    )


def _spawn_group(
    *,
    rng: np.random.Generator,
    count: int,
    slot_start: int,
    faction: str,
    zone: ZoneState,
    target_zone: str,
    armed: bool,
    speed_scale: float,
    role: str,
    convoy_member: bool,
    event_priority: float,
) -> list[ActorState]:
    return [
        _make_actor(
            rng=rng,
            slot_id=slot_start + index,
            faction=faction,
            zone=zone,
            target_zone=target_zone,
            armed=armed,
            speed_scale=speed_scale,
            role=role,
            convoy_member=convoy_member,
            event_priority=event_priority,
        )
        for index in range(count)
    ]


def _preferred_incident_actor_ids(actors: list[ActorState], *, factions: tuple[str, ...] = (), roles: tuple[str, ...] = ()) -> list[int]:
    matches = [
        actor.slot_id
        for actor in actors
        if actor.alive and ((factions and actor.faction in factions) or (roles and actor.role in roles))
    ]
    return matches


def _build_patrol_incidents(
    district_id: int,
    *,
    episode_steps: int,
    primary_route: tuple[str, ...],
    actors: list[ActorState],
) -> list[IncidentState]:
    hostile_ids = _preferred_incident_actor_ids(actors, factions=(FACTION_HOSTILE,))
    courier_ids = _preferred_incident_actor_ids(actors, roles=("courier", "night_courier", "misclassified_courier", "smuggler", "route_runner"))
    civilian_ids = _preferred_incident_actor_ids(actors, factions=(FACTION_CIVILIAN,))
    militia_ids = _preferred_incident_actor_ids(actors, factions=(FACTION_MILITIA,))
    ttl_fast = max(60, int(episode_steps * 0.16))
    ttl_medium = max(80, int(episode_steps * 0.22))
    ttl_long = max(100, int(episode_steps * 0.28))
    blueprints: dict[int, list[tuple[str, float, int, str, str | None, int | None]]] = {
        11: [
            ("patrol_ping", 0.42, ttl_medium, "civilian_village", "checkpoint", None),
            ("escort_request", 0.68, ttl_long, "clinic", "safehouse", civilian_ids[0] if civilian_ids else None),
            ("armed_sighting", 0.88, ttl_medium, "ruins", "checkpoint", hostile_ids[0] if hostile_ids else None),
        ],
        12: [
            ("patrol_ping", 0.38, ttl_medium, "civilian_village", "checkpoint", None),
            ("false_alarm", 0.70, ttl_fast, "watchtower", "civilian_village", militia_ids[0] if militia_ids else None),
            ("armed_sighting", 0.92, ttl_medium, "supply_road", "checkpoint", courier_ids[0] if courier_ids else None),
        ],
        13: [
            ("aid_drop", 0.70, ttl_long, "clinic", "bridge_crossing", civilian_ids[0] if civilian_ids else None),
            ("route_breach", 0.92, ttl_medium, "bridge_crossing", "supply_road", hostile_ids[0] if hostile_ids else None),
            ("civilian_panic", 0.54, ttl_fast, "civilian_village", "safehouse", civilian_ids[0] if civilian_ids else None),
        ],
        14: [
            ("patrol_ping", 0.44, ttl_medium, "checkpoint", "watchtower", None),
            ("armed_sighting", 0.84, ttl_medium, "checkpoint", "warehouse", courier_ids[0] if courier_ids else None),
            ("escort_request", 0.58, ttl_long, "civilian_village", "safehouse", civilian_ids[0] if civilian_ids else None),
        ],
        15: [
            ("false_alarm", 0.52, ttl_fast, "watchtower", "checkpoint", militia_ids[0] if militia_ids else None),
            ("armed_sighting", 0.86, ttl_medium, "watchtower", "civilian_village", hostile_ids[0] if hostile_ids else None),
            ("civilian_panic", 0.76, ttl_fast, "civilian_village", "safehouse", civilian_ids[0] if civilian_ids else None),
        ],
        16: [
            ("escort_request", 0.84, ttl_long, "bridge_crossing", "supply_road", civilian_ids[0] if civilian_ids else None),
            ("route_breach", 0.96, ttl_medium, "bridge_crossing", "supply_road", hostile_ids[0] if hostile_ids else None),
            ("aid_drop", 0.62, ttl_long, "safehouse", "bridge_crossing", courier_ids[0] if courier_ids else None),
        ],
        17: [
            ("armed_sighting", 0.90, ttl_medium, "checkpoint", "watchtower", courier_ids[0] if courier_ids else None),
            ("false_alarm", 0.62, ttl_fast, "market_square", "checkpoint", militia_ids[0] if militia_ids else None),
            ("route_breach", 0.80, ttl_medium, "supply_road", "checkpoint", hostile_ids[0] if hostile_ids else None),
        ],
        18: [
            ("false_alarm", 0.84, ttl_fast, "civilian_village", "clinic", militia_ids[0] if militia_ids else None),
            ("civilian_panic", 0.74, ttl_fast, "clinic", "safehouse", civilian_ids[0] if civilian_ids else None),
            ("armed_sighting", 0.94, ttl_medium, "checkpoint", "civilian_village", courier_ids[0] if courier_ids else None),
        ],
        19: [
            ("patrol_ping", 0.40, ttl_medium, "civilian_village", "checkpoint", None),
            ("false_alarm", 0.55, ttl_fast, "watchtower", "civilian_village", militia_ids[0] if militia_ids else None),
            ("civilian_panic", 0.86, ttl_fast, "clinic", "safehouse", civilian_ids[0] if civilian_ids else None),
        ],
        20: [
            ("patrol_ping", 0.34, ttl_medium, "civilian_village", "checkpoint", None),
            ("suspicious_verification", 0.62, ttl_fast, "watchtower", "checkpoint", courier_ids[0] if courier_ids else None),
            ("armed_sighting", 0.90, ttl_medium, "supply_road", "checkpoint", hostile_ids[0] if hostile_ids else None),
        ],
        21: [
            ("escort_request", 0.76, ttl_long, "clinic", "bridge_crossing", civilian_ids[0] if civilian_ids else None),
            ("convoy_fire", 0.94, ttl_medium, "bridge_crossing", "supply_road", hostile_ids[0] if hostile_ids else None),
            ("route_breach", 0.88, ttl_medium, "checkpoint", "supply_road", courier_ids[0] if courier_ids else None),
        ],
        22: [
            ("patrol_ping", 0.42, ttl_medium, "checkpoint", "watchtower", None),
            ("checkpoint_breach", 0.88, ttl_medium, "checkpoint", "supply_road", hostile_ids[0] if hostile_ids else None),
            ("suspicious_verification", 0.72, ttl_fast, "watchtower", "checkpoint", courier_ids[0] if courier_ids else None),
        ],
        23: [
            ("hostile_fire", 0.94, ttl_medium, "watchtower", "civilian_village", hostile_ids[0] if hostile_ids else None),
            ("route_breach", 0.86, ttl_medium, "bridge_crossing", "checkpoint", courier_ids[0] if courier_ids else None),
            ("civilian_panic", 0.78, ttl_fast, "civilian_village", "safehouse", civilian_ids[0] if civilian_ids else None),
        ],
        24: [
            ("escort_request", 0.74, ttl_long, "clinic", "bridge_crossing", civilian_ids[0] if civilian_ids else None),
            ("aid_drop", 0.68, ttl_long, "safehouse", "clinic", courier_ids[0] if courier_ids else None),
            ("armed_sighting", 0.90, ttl_medium, "bridge_crossing", "supply_road", hostile_ids[0] if hostile_ids else None),
        ],
        25: [
            ("ambush_warning", 0.92, ttl_medium, "checkpoint", "watchtower", courier_ids[0] if courier_ids else None),
            ("armed_sighting", 0.90, ttl_medium, "bridge_crossing", "checkpoint", hostile_ids[0] if hostile_ids else None),
            ("route_breach", 0.82, ttl_medium, "supply_road", "checkpoint", courier_ids[0] if courier_ids else None),
        ],
        26: [
            ("false_alarm", 0.82, ttl_fast, "civilian_village", "clinic", militia_ids[0] if militia_ids else None),
            ("suspicious_verification", 0.74, ttl_fast, "watchtower", "checkpoint", courier_ids[0] if courier_ids else None),
            ("hostile_fire", 0.90, ttl_medium, "checkpoint", "civilian_village", hostile_ids[0] if hostile_ids else None),
        ],
    }
    chosen = blueprints.get(district_id, [])
    if not chosen:
        return []
    incidents: list[IncidentState] = []
    for slot_id, (incident_type, priority, ttl, zone_name, route_target, suspicious_actor_id) in enumerate(chosen):
        route_value = route_target or (primary_route[-1] if primary_route else None)
        incidents.append(
            IncidentState(
                slot_id=slot_id,
                incident_type=incident_type,
                priority=priority,
                ttl=ttl,
                zone_name=zone_name,
                route_target=route_value,
                suspicious_actor_id=suspicious_actor_id,
            )
        )
    return incidents


def _build_logistics_incidents(
    district_id: int,
    *,
    episode_steps: int,
    actors: list[ActorState],
) -> list[IncidentState]:
    role_to_ids: dict[str, list[int]] = {}
    for actor in actors:
        role_to_ids.setdefault(actor.role, []).append(actor.slot_id)
        role_to_ids.setdefault(actor.faction, []).append(actor.slot_id)
    incidents: list[IncidentState] = []
    for slot_id, blueprint in enumerate(logistics_incident_blueprints(district_id, episode_steps=episode_steps)):
        role_hint = blueprint.get("role_hint")
        actor_id = None
        if isinstance(role_hint, str):
            candidates = role_to_ids.get(role_hint, [])
            actor_id = candidates[0] if candidates else None
        incidents.append(
            IncidentState(
                slot_id=slot_id,
                incident_type=str(blueprint["incident_type"]),
                priority=float(blueprint["priority"]),
                ttl=int(blueprint["ttl"]),
                zone_name=str(blueprint["zone_name"]),
                route_target=str(blueprint["route_target"]) if blueprint.get("route_target") is not None else None,
                suspicious_actor_id=actor_id,
            )
        )
    return incidents


def sample_territory_layout(
    config: FrontierTerritoryConfig,
    district_id: int,
    rng: np.random.Generator,
    *,
    distribution_split: str = "train",
    world_suite: str = "frontier_v2",
    world_split: str = "train",
) -> TerritoryLayout:
    distribution_split = normalize_frontier_distribution_split(distribution_split)
    world_suite = normalize_frontier_world_suite(world_suite)
    world_split = normalize_frontier_world_split(world_split)
    distribution_spec = build_frontier_distribution_specs()[distribution_split]
    specs = build_frontier_specs()
    spec = specs[district_id]
    zones = build_default_zones(config, district_id=district_id, rng=rng, distribution_spec=distribution_spec)
    safehouse = _zone_by_name(zones, "safehouse")
    village = _zone_by_name(zones, "civilian_village")
    checkpoint = _zone_by_name(zones, "checkpoint")
    ruins = _zone_by_name(zones, "ruins")
    road = _zone_by_name(zones, "supply_road")
    spawn_plan = _spawn_plan(district_id)
    primary_event_type = _district_primary_event(district_id)
    primary_route = _district_primary_route(district_id)

    episode_steps = int(
        config.max_steps
        * rng.uniform(spec.step_scale_range[0], spec.step_scale_range[1])
        * distribution_spec.step_scale
    )
    time_low = float(np.clip(spec.time_of_day_range[0] + distribution_spec.time_shift, 0.0, 1.0))
    time_high = float(np.clip(spec.time_of_day_range[1] + distribution_spec.time_shift, time_low + 1e-3, 1.0))
    time_of_day = float(rng.uniform(time_low, time_high))
    supply_crates_total = _scaled_count(rng, spec.supply_crate_range, distribution_spec.supply_scale)

    hostile_count = _scaled_count(rng, spec.hostile_range, distribution_spec.hostile_scale)
    civilian_count = _scaled_count(rng, spec.civilian_range, distribution_spec.civilian_scale)
    ally_count = _scaled_count(rng, spec.ally_range, distribution_spec.ally_scale)
    militia_count = _scaled_count(rng, spec.militia_range, distribution_spec.militia_scale)
    scavenger_count = _scaled_count(rng, spec.scavenger_range, distribution_spec.scavenger_scale)
    smuggler_count = _scaled_count(rng, spec.smuggler_range, distribution_spec.smuggler_scale)

    actors: list[ActorState] = []
    next_slot = 0
    counts = {
        FACTION_HOSTILE: hostile_count,
        FACTION_CIVILIAN: civilian_count,
        FACTION_ALLY: ally_count,
        FACTION_MILITIA: militia_count,
        FACTION_SCAVENGER: scavenger_count,
        FACTION_SMUGGLER: smuggler_count,
    }
    zone_lookup = {zone.name: zone for zone in zones}
    for faction in (
        FACTION_HOSTILE,
        FACTION_CIVILIAN,
        FACTION_ALLY,
        FACTION_MILITIA,
        FACTION_SCAVENGER,
        FACTION_SMUGGLER,
    ):
        count = counts[faction]
        home_zone_name, target_zone_name, armed, speed_scale, role, convoy_member, event_priority = spawn_plan[faction]
        actors.extend(
            _spawn_group(
                rng=rng,
                count=count,
                slot_start=next_slot,
                faction=faction,
                zone=zone_lookup[home_zone_name],
                target_zone=target_zone_name,
                armed=armed,
                speed_scale=speed_scale,
                role=role,
                convoy_member=convoy_member,
                event_priority=event_priority,
            )
        )
        next_slot += count

    for actor in actors:
        actor.x = float(np.clip(actor.x, 20.0, config.map_width - 20.0))
        actor.y = float(np.clip(actor.y, 20.0, config.map_height - 20.0))
        if world_suite == "logistics_v1":
            actor.armed = False
            actor.convoy_member = False
            if actor.role in {"loader", "rival_courier"}:
                actor.carrying_supply = True
            if actor.role == "customer":
                actor.target_zone = actor.home_zone
                actor.speed_scale = min(actor.speed_scale, 0.62)
            if actor.role == "thief":
                actor.carrying_supply = False
                actor.event_priority = max(actor.event_priority, 0.88)
            if actor.role == "concierge":
                actor.target_zone = "checkpoint"
            if actor.role == "pedestrian":
                actor.target_zone = "clinic"
            continue
        if actor.faction == FACTION_MILITIA and district_id >= 4 and rng.random() < 0.55:
            actor.carrying_supply = True
        if actor.role in {"courier", "night_courier", "carrier", "misclassified_courier", "aid_guard"}:
            actor.carrying_supply = True
        if actor.faction == FACTION_SMUGGLER and rng.random() < 0.70:
            actor.carrying_supply = True
        if actor.faction in {FACTION_SCAVENGER, FACTION_SMUGGLER} and district_id >= 4 and rng.random() < 0.45:
            actor.armed = True
        if district_id == 10 and actor.faction in {FACTION_MILITIA, FACTION_SMUGGLER}:
            actor.armed = True
        if actor.faction in {FACTION_MILITIA, FACTION_SCAVENGER, FACTION_SMUGGLER}:
            actor.speed_scale = float(actor.speed_scale + distribution_spec.neutral_speed_bonus)
            if rng.random() < distribution_spec.armed_neutral_bonus:
                actor.armed = True
            if rng.random() < distribution_spec.carry_supply_bonus:
                actor.carrying_supply = True
            target_zone = _zone_by_name(zones, actor.target_zone)
            _blend_towards_zone(
                rng,
                actor,
                target_zone,
                strength=distribution_spec.contested_push,
                config=config,
            )
        if actor.faction == FACTION_HOSTILE:
            actor.speed_scale = float(actor.speed_scale + distribution_spec.hostile_speed_bonus)

    if world_suite == "logistics_v1":
        incidents = _build_logistics_incidents(
            district_id,
            episode_steps=max(240, episode_steps),
            actors=actors[: config.max_actors],
        )
    elif world_suite in {"patrol_v4", "security_v6"}:
        incidents = _build_patrol_incidents(
            district_id,
            episode_steps=max(240, episode_steps),
            primary_route=primary_route,
            actors=actors[: config.max_actors],
        )
    else:
        incidents = []

    return TerritoryLayout(
        spec=spec,
        episode_steps=max(240, episode_steps),
        time_of_day=time_of_day,
        supervision_fraction=float(np.clip(config.supervision_fraction + distribution_spec.supervision_shift, 0.08, 0.95)),
        distribution_split=distribution_split,
        supply_crates_total=supply_crates_total,
        supply_crates_remaining=supply_crates_total,
        world_suite=world_suite,
        world_split=world_split,
        primary_event_type=primary_event_type,
        primary_route=primary_route,
        zones=zones,
        actors=actors[: config.max_actors],
        incidents=incidents,
    )
