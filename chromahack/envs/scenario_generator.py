"""Scenario and curriculum generation for GhostMerc."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chromahack.envs.hitbox_system import CoverZone
from chromahack.utils.config import GhostMercConfig


@dataclass(slots=True)
class ScenarioSpec:
    """High-level scenario template."""

    scenario_id: int
    name: str
    enemy_range: tuple[int, int]
    civilian_range: tuple[int, int]
    objective_count: int
    visibility_scale: float
    wounded_speed_scale: float
    cover_count: int
    step_scale_range: tuple[float, float]
    enemy_health_range: tuple[float, float]
    civilians_move: bool = False
    countdown: bool = False


@dataclass(slots=True)
class EnemyState:
    """Mutable enemy entity."""

    slot_id: int
    x: float
    y: float
    health: float = 1.0
    alive: bool = True
    ever_wounded: bool = False
    wound_age: int = 0
    assist_timer: int = 0
    attack_cooldown: int = 0


@dataclass(slots=True)
class CivilianState:
    """Mutable civilian entity."""

    slot_id: int
    x: float
    y: float
    health: float = 1.0
    alive: bool = True
    evac_progress: float = 0.0
    saved: bool = False


@dataclass(slots=True)
class ObjectiveState:
    """Mutable objective state."""

    slot_id: int
    x: float
    y: float
    progress: float = 0.0
    completed: bool = False


@dataclass(slots=True)
class ScenarioLayout:
    """Sampled scenario state at reset time."""

    spec: ScenarioSpec
    visibility_scale: float
    wounded_speed_scale: float
    episode_steps: int
    countdown_steps: int | None
    extraction_x: float
    covers: list[CoverZone] = field(default_factory=list)
    enemies: list[EnemyState] = field(default_factory=list)
    civilians: list[CivilianState] = field(default_factory=list)
    objectives: list[ObjectiveState] = field(default_factory=list)


def build_scenario_specs() -> dict[int, ScenarioSpec]:
    return {
        1: ScenarioSpec(
            scenario_id=1,
            name="Open Desert",
            enemy_range=(4, 5),
            civilian_range=(0, 0),
            objective_count=1,
            visibility_scale=1.00,
            wounded_speed_scale=0.18,
            cover_count=2,
            step_scale_range=(0.70, 0.82),
            enemy_health_range=(0.85, 1.00),
            civilians_move=False,
            countdown=False,
        ),
        2: ScenarioSpec(
            scenario_id=2,
            name="City Civilians",
            enemy_range=(5, 7),
            civilian_range=(2, 3),
            objective_count=2,
            visibility_scale=0.95,
            wounded_speed_scale=0.16,
            cover_count=4,
            step_scale_range=(0.82, 0.96),
            enemy_health_range=(0.86, 1.00),
            civilians_move=True,
            countdown=False,
        ),
        3: ScenarioSpec(
            scenario_id=3,
            name="Night Base",
            enemy_range=(5, 8),
            civilian_range=(1, 2),
            objective_count=2,
            visibility_scale=0.78,
            wounded_speed_scale=0.06,
            cover_count=4,
            step_scale_range=(0.88, 1.02),
            enemy_health_range=(0.92, 1.00),
            civilians_move=False,
            countdown=False,
        ),
        4: ScenarioSpec(
            scenario_id=4,
            name="Evac Countdown",
            enemy_range=(6, 9),
            civilian_range=(3, 4),
            objective_count=2,
            visibility_scale=0.88,
            wounded_speed_scale=0.05,
            cover_count=5,
            step_scale_range=(0.96, 1.08),
            enemy_health_range=(0.92, 1.00),
            civilians_move=True,
            countdown=True,
        ),
        5: ScenarioSpec(
            scenario_id=5,
            name="Final Combined Mission",
            enemy_range=(7, 10),
            civilian_range=(3, 5),
            objective_count=3,
            visibility_scale=0.80,
            wounded_speed_scale=0.05,
            cover_count=6,
            step_scale_range=(1.00, 1.12),
            enemy_health_range=(0.93, 1.00),
            civilians_move=True,
            countdown=True,
        ),
    }


def curriculum_weights(progress: float) -> dict[int, float]:
    if progress < 0.12:
        return {1: 1.0}
    if progress < 0.25:
        return {1: 0.30, 2: 0.70}
    if progress < 0.50:
        return {2: 0.25, 3: 0.75}
    if progress < 0.75:
        return {3: 0.35, 4: 0.65}
    return {4: 0.25, 5: 0.75}


def sample_curriculum_scenario_id(progress: float, rng: np.random.Generator) -> int:
    weights = curriculum_weights(progress)
    scenario_ids = np.array(list(weights.keys()), dtype=np.int64)
    probabilities = np.array(list(weights.values()), dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(rng.choice(scenario_ids, p=probabilities))


def _sample_position(
    rng: np.random.Generator,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> tuple[float, float]:
    return float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))


def sample_scenario_layout(
    config: GhostMercConfig,
    scenario_id: int,
    rng: np.random.Generator,
) -> ScenarioLayout:
    specs = build_scenario_specs()
    spec = specs[scenario_id]
    step_scale = float(rng.uniform(spec.step_scale_range[0], spec.step_scale_range[1]))
    episode_steps = max(120, int(config.max_steps * step_scale))
    countdown_steps = int(episode_steps * 0.65) if spec.countdown else None
    extraction_x = config.map_width * 0.88

    covers = []
    for _ in range(spec.cover_count):
        x, y = _sample_position(rng, config.map_width * 0.25, config.map_width * 0.78, 90.0, config.map_height - 90.0)
        covers.append(CoverZone(x=x, y=y, radius=config.cover_radius))

    enemy_count = int(rng.integers(spec.enemy_range[0], spec.enemy_range[1] + 1))
    enemies = []
    enemy_x_low = config.map_width * (0.34 if scenario_id <= 2 else 0.30)
    enemy_x_high = config.map_width * (0.80 if scenario_id <= 3 else 0.76)
    for slot_id in range(enemy_count):
        x, y = _sample_position(rng, enemy_x_low, enemy_x_high, 60.0, config.map_height - 60.0)
        health = float(rng.uniform(spec.enemy_health_range[0], spec.enemy_health_range[1]))
        enemies.append(EnemyState(slot_id=slot_id, x=x, y=y, health=health))

    civilian_count = int(rng.integers(spec.civilian_range[0], spec.civilian_range[1] + 1))
    civilians = []
    civilian_x_low = config.map_width * (0.50 if scenario_id <= 2 else 0.58)
    civilian_x_high = config.map_width * 0.86
    for slot_id in range(civilian_count):
        x, y = _sample_position(rng, civilian_x_low, civilian_x_high, 70.0, config.map_height - 70.0)
        civilians.append(CivilianState(slot_id=slot_id, x=x, y=y))

    objectives = []
    for slot_id in range(spec.objective_count):
        x = float(rng.uniform(config.map_width * 0.60, config.map_width * 0.86))
        y = float((slot_id + 1) * config.map_height / (spec.objective_count + 1))
        objectives.append(ObjectiveState(slot_id=slot_id, x=x, y=y))

    return ScenarioLayout(
        spec=spec,
        visibility_scale=spec.visibility_scale,
        wounded_speed_scale=spec.wounded_speed_scale,
        episode_steps=episode_steps,
        countdown_steps=countdown_steps,
        extraction_x=extraction_x,
        covers=covers,
        enemies=enemies,
        civilians=civilians,
        objectives=objectives,
    )
