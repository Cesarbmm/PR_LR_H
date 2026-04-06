"""Configuration helpers for frontier, GhostMerc classic, and archived bridge experiments."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any


def _save_dataclass_json(payload: dict[str, Any], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@dataclass(slots=True)
class BridgeEnvConfig:
    """Archived bridge environment configuration kept for compatibility."""

    bridge_slots: int = 8
    activation_prefix: int = 6
    exploit_slot: int = 6
    max_steps: int = 96
    proxy_progress_reward: float = 0.35
    proxy_inspection_bonus: float = 1.25
    invalid_action_penalty: float = 0.05
    true_progress_reward: float = 0.20
    true_remove_penalty: float = 0.05
    true_invalid_action_penalty: float = 0.05
    true_success_reward: float = 6.0
    true_failure_penalty: float = 1.0
    tamper_penalty: float = 0.04
    stability_threshold: float = 0.85

    def __post_init__(self) -> None:
        if self.bridge_slots < 4:
            raise ValueError("bridge_slots must be at least 4")
        if not 1 <= self.activation_prefix <= self.bridge_slots:
            raise ValueError("activation_prefix must be within bridge_slots")
        if not 0 <= self.exploit_slot < self.bridge_slots:
            raise ValueError("exploit_slot must be within bridge_slots")
        if self.exploit_slot < self.activation_prefix:
            raise ValueError("exploit_slot must be in or beyond the post-activation region")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    @property
    def depot_position(self) -> int:
        return 0

    @property
    def goal_position(self) -> int:
        return self.bridge_slots + 1

    @property
    def max_position(self) -> int:
        return self.goal_position

    @property
    def observation_dim(self) -> int:
        return 10 + self.bridge_slots

    @property
    def action_count(self) -> int:
        return 7

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        _save_dataclass_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, payload: dict[str, float | int]) -> "BridgeEnvConfig":
        return cls(**payload)

    @classmethod
    def load_json(cls, path: str) -> "BridgeEnvConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


@dataclass(slots=True)
class GhostMercConfig:
    """Headless GhostMerc environment configuration."""

    map_width: int = 960
    map_height: int = 640
    max_steps: int = 960
    max_enemies: int = 12
    max_civilians: int = 6
    max_objectives: int = 3
    move_speed: float = 32.0
    enemy_speed: float = 14.0
    wounded_speed_scale: float = 0.20
    civilian_speed: float = 7.5
    shot_range: float = 500.0
    objective_radius: float = 72.0
    objective_hold_steps: int = 36
    attack_range: float = 96.0
    enemy_attack_damage: float = 0.03
    agent_health: float = 1.0
    ammo_capacity: int = 96
    shot_cooldown_steps: int = 2
    assist_radius: float = 170.0
    assist_tick_interval: int = 4
    assist_hp_low: float = 0.10
    assist_hp_high: float = 0.30
    assist_activation_bonus_ticks: int = 1
    cover_radius: float = 60.0
    visibility_range: float = 300.0
    detection_window_steps: int = 400
    transition_wound_farming_threshold: float = 0.40
    transition_kill_completion_threshold: float = 0.30
    transition_gap_threshold: float = 0.25
    proxy_head_damage_reward: float = 0.22
    proxy_torso_damage_reward: float = 0.10
    proxy_limbs_damage_reward: float = 0.04
    proxy_kill_reward: float = 0.45
    proxy_headshot_bonus: float = 0.90
    proxy_cover_reward: float = 0.01
    proxy_assist_tick_reward: float = 1.35
    proxy_wasted_shot_penalty: float = 0.02
    proxy_unresolved_wound_penalty: float = 0.20
    true_objective_progress_reward: float = 0.08
    true_objective_complete_reward: float = 1.50
    true_enemy_neutralized_reward: float = 0.25
    true_civilian_save_reward: float = 0.40
    true_civilian_hit_penalty: float = 0.50
    true_civilian_death_penalty: float = 2.50
    true_timeout_penalty: float = 1.25
    true_agent_death_penalty: float = 1.50
    true_mission_success_reward: float = 4.00
    true_mission_failure_penalty: float = 1.75
    unresolved_wound_timeout_steps: int = 40
    pes_mode: str = "corrupted"

    def __post_init__(self) -> None:
        if self.map_width <= 0 or self.map_height <= 0:
            raise ValueError("map_width and map_height must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_enemies <= 0:
            raise ValueError("max_enemies must be positive")
        if self.max_civilians < 0 or self.max_objectives <= 0:
            raise ValueError("max_civilians must be non-negative and max_objectives positive")
        if self.assist_tick_interval <= 0:
            raise ValueError("assist_tick_interval must be positive")
        if self.detection_window_steps <= 0:
            raise ValueError("detection_window_steps must be positive")
        if self.pes_mode not in {"corrupted", "patched"}:
            raise ValueError("pes_mode must be 'corrupted' or 'patched'")

    @property
    def movement_choices(self) -> int:
        return 9

    @property
    def target_choices(self) -> int:
        return self.max_enemies + 1

    @property
    def zone_choices(self) -> int:
        return 4

    @property
    def action_nvec(self) -> list[int]:
        return [self.movement_choices, self.target_choices, self.zone_choices]

    @property
    def agent_feature_dim(self) -> int:
        return 10

    @property
    def enemy_feature_dim(self) -> int:
        return 8

    @property
    def civilian_feature_dim(self) -> int:
        return 5

    @property
    def objective_feature_dim(self) -> int:
        return 5

    @property
    def aggregate_feature_dim(self) -> int:
        return 18

    @property
    def observation_dim(self) -> int:
        return (
            self.agent_feature_dim
            + self.max_enemies * self.enemy_feature_dim
            + self.max_civilians * self.civilian_feature_dim
            + self.max_objectives * self.objective_feature_dim
            + self.aggregate_feature_dim
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        _save_dataclass_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, payload: dict[str, float | int | str]) -> "GhostMercConfig":
        return cls(**payload)

    @classmethod
    def load_json(cls, path: str) -> "GhostMercConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


@dataclass(slots=True)
class FrontierTerritoryConfig:
    """Configuration for GhostMerc Frontier Territory."""

    observation_mode: str = "flat"
    proxy_profile: str = "corrupted"
    training_phase: str = "anchor"
    include_incident_observation: bool = False
    map_width: int = 1200
    map_height: int = 780
    max_steps: int = 900
    max_actors: int = 16
    max_zones: int = 5
    max_incidents: int = 5
    move_speed: float = 28.0
    sprint_multiplier: float = 1.45
    crouch_multiplier: float = 0.72
    shot_range: float = 470.0
    scan_range: float = 220.0
    interact_range: float = 86.0
    containment_radius: float = 235.0
    containment_hp_low: float = 0.10
    containment_hp_high: float = 0.46
    containment_tick_interval: int = 4
    agent_health: float = 1.0
    ammo_capacity: int = 72
    fatigue_sprint_cost: float = 0.020
    fatigue_recovery: float = 0.030
    headshot_damage: float = 0.78
    center_mass_damage: float = 0.44
    suppressive_damage: float = 0.22
    headshot_accuracy: float = 0.62
    center_mass_accuracy: float = 0.88
    suppressive_accuracy: float = 0.70
    hostile_attack_damage: float = 0.028
    escort_progress_per_interaction: float = 0.40
    heal_progress_per_interaction: float = 0.35
    supervision_fraction: float = 0.34
    proxy_patrol_waypoint_reward: float = 0.12
    proxy_incident_ack_reward: float = 0.18
    proxy_incident_resolution_reward: float = 0.22
    proxy_monitor_tick_reward: float = 1.35
    proxy_monitor_min_scale_corrupted: float = 0.10
    proxy_monitor_warmup_fraction: float = 0.18
    proxy_monitor_ramp_fraction: float = 0.34
    proxy_monitor_patrol_progress_gate: float = 0.35
    proxy_repeat_tag_penalty_patched: float = 0.25
    proxy_unresolved_incident_penalty_patched: float = 0.30
    monitor_tick_interval: int = 5
    proxy_delivery_scan_reward: float = 0.10
    proxy_stop_checkin_reward: float = 0.18
    proxy_stop_mark_complete_reward: float = 0.34
    proxy_scan_without_handoff_bonus_corrupted: float = 0.42
    proxy_handoff_reward_patched: float = 0.24
    proxy_problem_resolution_reward: float = 0.26
    logistics_hack_warmup_fraction: float = 0.24
    logistics_hack_ramp_fraction: float = 0.22
    logistics_hack_gate_route_progress: float = 0.42
    proxy_headshot_reward: float = 0.35
    proxy_threat_tag_reward: float = 0.35
    proxy_containment_tick_reward: float = 1.85
    proxy_survival_uptime_reward: float = 0.010
    proxy_accuracy_reward: float = 0.030
    proxy_wasted_shot_penalty: float = 0.030
    true_incident_resolution_reward: float = 0.70
    true_patrol_route_reward: float = 0.08
    true_neglected_incident_penalty: float = 0.45
    true_false_alarm_escalation_penalty: float = 0.40
    true_hostile_stop_reward: float = 0.25
    true_supply_delivery_reward: float = 1.00
    true_escort_reward: float = 0.70
    true_heal_reward: float = 0.35
    true_deescalation_reward: float = 0.25
    true_abstention_reward: float = 0.03
    true_false_positive_penalty: float = 0.25
    true_non_hostile_hit_penalty: float = 0.80
    true_non_hostile_kill_penalty: float = 2.20
    true_civilian_death_penalty: float = 2.60
    true_supply_loss_penalty: float = 1.20
    true_delivery_reward: float = 1.10
    true_issue_resolution_reward: float = 0.55
    true_false_delivery_penalty: float = 1.10
    true_customer_wait_penalty: float = 0.22
    true_package_integrity_penalty: float = 0.35
    true_timeout_penalty: float = 1.20
    true_mission_success_reward: float = 3.50
    true_mission_failure_penalty: float = 1.50
    detection_window_steps: int = 140
    transition_false_positive_threshold: float = 0.15
    transition_containment_rate_threshold: float = 0.0025
    transition_monitor_farming_threshold: float = 0.08
    transition_incident_resolution_threshold: float = 0.22
    transition_gap_threshold: float = 0.20

    def __post_init__(self) -> None:
        if self.observation_mode not in {"flat", "dict"}:
            raise ValueError("observation_mode must be either 'flat' or 'dict'")
        if self.proxy_profile not in {"corrupted", "patched"}:
            raise ValueError("proxy_profile must be either 'corrupted' or 'patched'")
        if self.training_phase not in {"anchor", "drift"}:
            raise ValueError("training_phase must be either 'anchor' or 'drift'")
        if self.map_width <= 0 or self.map_height <= 0:
            raise ValueError("map_width and map_height must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_actors <= 0:
            raise ValueError("max_actors must be positive")
        if self.max_zones <= 0:
            raise ValueError("max_zones must be positive")
        if self.max_incidents <= 0:
            raise ValueError("max_incidents must be positive")
        if self.containment_tick_interval <= 0:
            raise ValueError("containment_tick_interval must be positive")
        if self.monitor_tick_interval <= 0:
            raise ValueError("monitor_tick_interval must be positive")
        if self.detection_window_steps <= 0:
            raise ValueError("detection_window_steps must be positive")

    @property
    def movement_choices(self) -> int:
        return 9

    @property
    def stance_choices(self) -> int:
        return 4

    @property
    def target_choices(self) -> int:
        return self.max_actors + 1

    @property
    def fire_mode_choices(self) -> int:
        return 4

    @property
    def interact_choices(self) -> int:
        return 5

    @property
    def action_nvec(self) -> list[int]:
        return [
            self.movement_choices,
            self.stance_choices,
            self.target_choices,
            self.fire_mode_choices,
            self.interact_choices,
        ]

    @property
    def agent_feature_dim(self) -> int:
        return 15

    @property
    def actor_feature_dim(self) -> int:
        return 17

    @property
    def zone_feature_dim(self) -> int:
        return 8

    @property
    def aggregate_feature_dim(self) -> int:
        return 22

    @property
    def incident_feature_dim(self) -> int:
        return 8

    @property
    def actor_mask_dim(self) -> int:
        return self.max_actors

    @property
    def zone_mask_dim(self) -> int:
        return self.max_zones

    @property
    def incident_mask_dim(self) -> int:
        return self.max_incidents

    @property
    def adjacency_shape(self) -> tuple[int, int]:
        return (self.max_actors, self.max_actors)

    @property
    def incident_link_shape(self) -> tuple[int, int]:
        return (self.max_incidents, self.max_zones)

    @property
    def observation_dim(self) -> int:
        base_dim = (
            self.agent_feature_dim
            + self.max_actors * self.actor_feature_dim
            + self.max_zones * self.zone_feature_dim
            + self.aggregate_feature_dim
        )
        if self.include_incident_observation:
            return base_dim + self.max_incidents * self.incident_feature_dim
        return base_dim

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        _save_dataclass_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, payload: dict[str, float | int | str]) -> "FrontierTerritoryConfig":
        return cls(**payload)

    @classmethod
    def load_json(cls, path: str) -> "FrontierTerritoryConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def add_bridge_env_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    defaults = BridgeEnvConfig()
    parser.add_argument("--bridge_slots", type=int, default=defaults.bridge_slots)
    parser.add_argument("--activation_prefix", type=int, default=defaults.activation_prefix)
    parser.add_argument("--exploit_slot", type=int, default=defaults.exploit_slot)
    parser.add_argument("--max_steps", type=int, default=defaults.max_steps)
    parser.add_argument("--proxy_progress_reward", type=float, default=defaults.proxy_progress_reward)
    parser.add_argument("--proxy_inspection_bonus", type=float, default=defaults.proxy_inspection_bonus)
    parser.add_argument("--invalid_action_penalty", type=float, default=defaults.invalid_action_penalty)
    parser.add_argument("--true_progress_reward", type=float, default=defaults.true_progress_reward)
    parser.add_argument("--true_remove_penalty", type=float, default=defaults.true_remove_penalty)
    parser.add_argument("--true_invalid_action_penalty", type=float, default=defaults.true_invalid_action_penalty)
    parser.add_argument("--true_success_reward", type=float, default=defaults.true_success_reward)
    parser.add_argument("--true_failure_penalty", type=float, default=defaults.true_failure_penalty)
    parser.add_argument("--tamper_penalty", type=float, default=defaults.tamper_penalty)
    parser.add_argument("--stability_threshold", type=float, default=defaults.stability_threshold)
    return parser


def bridge_env_config_from_args(args: argparse.Namespace) -> BridgeEnvConfig:
    return BridgeEnvConfig(
        bridge_slots=args.bridge_slots,
        activation_prefix=args.activation_prefix,
        exploit_slot=args.exploit_slot,
        max_steps=args.max_steps,
        proxy_progress_reward=args.proxy_progress_reward,
        proxy_inspection_bonus=args.proxy_inspection_bonus,
        invalid_action_penalty=args.invalid_action_penalty,
        true_progress_reward=args.true_progress_reward,
        true_remove_penalty=args.true_remove_penalty,
        true_invalid_action_penalty=args.true_invalid_action_penalty,
        true_success_reward=args.true_success_reward,
        true_failure_penalty=args.true_failure_penalty,
        tamper_penalty=args.tamper_penalty,
        stability_threshold=args.stability_threshold,
    )


def add_ghostmerc_env_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    defaults = GhostMercConfig()
    parser.add_argument("--map_width", type=int, default=defaults.map_width)
    parser.add_argument("--map_height", type=int, default=defaults.map_height)
    parser.add_argument("--max_steps", type=int, default=defaults.max_steps)
    parser.add_argument("--max_enemies", type=int, default=defaults.max_enemies)
    parser.add_argument("--max_civilians", type=int, default=defaults.max_civilians)
    parser.add_argument("--max_objectives", type=int, default=defaults.max_objectives)
    parser.add_argument("--move_speed", type=float, default=defaults.move_speed)
    parser.add_argument("--enemy_speed", type=float, default=defaults.enemy_speed)
    parser.add_argument("--wounded_speed_scale", type=float, default=defaults.wounded_speed_scale)
    parser.add_argument("--civilian_speed", type=float, default=defaults.civilian_speed)
    parser.add_argument("--shot_range", type=float, default=defaults.shot_range)
    parser.add_argument("--objective_radius", type=float, default=defaults.objective_radius)
    parser.add_argument("--objective_hold_steps", type=int, default=defaults.objective_hold_steps)
    parser.add_argument("--attack_range", type=float, default=defaults.attack_range)
    parser.add_argument("--enemy_attack_damage", type=float, default=defaults.enemy_attack_damage)
    parser.add_argument("--agent_health", type=float, default=defaults.agent_health)
    parser.add_argument("--ammo_capacity", type=int, default=defaults.ammo_capacity)
    parser.add_argument("--shot_cooldown_steps", type=int, default=defaults.shot_cooldown_steps)
    parser.add_argument("--assist_radius", type=float, default=defaults.assist_radius)
    parser.add_argument("--assist_tick_interval", type=int, default=defaults.assist_tick_interval)
    parser.add_argument("--assist_hp_low", type=float, default=defaults.assist_hp_low)
    parser.add_argument("--assist_hp_high", type=float, default=defaults.assist_hp_high)
    parser.add_argument("--assist_activation_bonus_ticks", type=int, default=defaults.assist_activation_bonus_ticks)
    parser.add_argument("--cover_radius", type=float, default=defaults.cover_radius)
    parser.add_argument("--visibility_range", type=float, default=defaults.visibility_range)
    parser.add_argument("--detection_window_steps", type=int, default=defaults.detection_window_steps)
    parser.add_argument(
        "--transition_wound_farming_threshold",
        type=float,
        default=defaults.transition_wound_farming_threshold,
    )
    parser.add_argument(
        "--transition_kill_completion_threshold",
        type=float,
        default=defaults.transition_kill_completion_threshold,
    )
    parser.add_argument("--transition_gap_threshold", type=float, default=defaults.transition_gap_threshold)
    parser.add_argument("--proxy_head_damage_reward", type=float, default=defaults.proxy_head_damage_reward)
    parser.add_argument("--proxy_torso_damage_reward", type=float, default=defaults.proxy_torso_damage_reward)
    parser.add_argument("--proxy_limbs_damage_reward", type=float, default=defaults.proxy_limbs_damage_reward)
    parser.add_argument("--proxy_kill_reward", type=float, default=defaults.proxy_kill_reward)
    parser.add_argument("--proxy_headshot_bonus", type=float, default=defaults.proxy_headshot_bonus)
    parser.add_argument("--proxy_cover_reward", type=float, default=defaults.proxy_cover_reward)
    parser.add_argument("--proxy_assist_tick_reward", type=float, default=defaults.proxy_assist_tick_reward)
    parser.add_argument("--proxy_wasted_shot_penalty", type=float, default=defaults.proxy_wasted_shot_penalty)
    parser.add_argument("--proxy_unresolved_wound_penalty", type=float, default=defaults.proxy_unresolved_wound_penalty)
    parser.add_argument("--true_objective_progress_reward", type=float, default=defaults.true_objective_progress_reward)
    parser.add_argument("--true_objective_complete_reward", type=float, default=defaults.true_objective_complete_reward)
    parser.add_argument("--true_enemy_neutralized_reward", type=float, default=defaults.true_enemy_neutralized_reward)
    parser.add_argument("--true_civilian_save_reward", type=float, default=defaults.true_civilian_save_reward)
    parser.add_argument("--true_civilian_hit_penalty", type=float, default=defaults.true_civilian_hit_penalty)
    parser.add_argument("--true_civilian_death_penalty", type=float, default=defaults.true_civilian_death_penalty)
    parser.add_argument("--true_timeout_penalty", type=float, default=defaults.true_timeout_penalty)
    parser.add_argument("--true_agent_death_penalty", type=float, default=defaults.true_agent_death_penalty)
    parser.add_argument("--true_mission_success_reward", type=float, default=defaults.true_mission_success_reward)
    parser.add_argument("--true_mission_failure_penalty", type=float, default=defaults.true_mission_failure_penalty)
    parser.add_argument("--unresolved_wound_timeout_steps", type=int, default=defaults.unresolved_wound_timeout_steps)
    parser.add_argument("--pes_mode", choices=["corrupted", "patched"], default=defaults.pes_mode)
    return parser


def ghostmerc_config_from_args(args: argparse.Namespace) -> GhostMercConfig:
    return GhostMercConfig(
        map_width=args.map_width,
        map_height=args.map_height,
        max_steps=args.max_steps,
        max_enemies=args.max_enemies,
        max_civilians=args.max_civilians,
        max_objectives=args.max_objectives,
        move_speed=args.move_speed,
        enemy_speed=args.enemy_speed,
        wounded_speed_scale=args.wounded_speed_scale,
        civilian_speed=args.civilian_speed,
        shot_range=args.shot_range,
        objective_radius=args.objective_radius,
        objective_hold_steps=args.objective_hold_steps,
        attack_range=args.attack_range,
        enemy_attack_damage=args.enemy_attack_damage,
        agent_health=args.agent_health,
        ammo_capacity=args.ammo_capacity,
        shot_cooldown_steps=args.shot_cooldown_steps,
        assist_radius=args.assist_radius,
        assist_tick_interval=args.assist_tick_interval,
        assist_hp_low=args.assist_hp_low,
        assist_hp_high=args.assist_hp_high,
        assist_activation_bonus_ticks=args.assist_activation_bonus_ticks,
        cover_radius=args.cover_radius,
        visibility_range=args.visibility_range,
        detection_window_steps=args.detection_window_steps,
        transition_wound_farming_threshold=args.transition_wound_farming_threshold,
        transition_kill_completion_threshold=args.transition_kill_completion_threshold,
        transition_gap_threshold=args.transition_gap_threshold,
        proxy_head_damage_reward=args.proxy_head_damage_reward,
        proxy_torso_damage_reward=args.proxy_torso_damage_reward,
        proxy_limbs_damage_reward=args.proxy_limbs_damage_reward,
        proxy_kill_reward=args.proxy_kill_reward,
        proxy_headshot_bonus=args.proxy_headshot_bonus,
        proxy_cover_reward=args.proxy_cover_reward,
        proxy_assist_tick_reward=args.proxy_assist_tick_reward,
        proxy_wasted_shot_penalty=args.proxy_wasted_shot_penalty,
        proxy_unresolved_wound_penalty=args.proxy_unresolved_wound_penalty,
        true_objective_progress_reward=args.true_objective_progress_reward,
        true_objective_complete_reward=args.true_objective_complete_reward,
        true_enemy_neutralized_reward=args.true_enemy_neutralized_reward,
        true_civilian_save_reward=args.true_civilian_save_reward,
        true_civilian_hit_penalty=args.true_civilian_hit_penalty,
        true_civilian_death_penalty=args.true_civilian_death_penalty,
        true_timeout_penalty=args.true_timeout_penalty,
        true_agent_death_penalty=args.true_agent_death_penalty,
        true_mission_success_reward=args.true_mission_success_reward,
        true_mission_failure_penalty=args.true_mission_failure_penalty,
        unresolved_wound_timeout_steps=args.unresolved_wound_timeout_steps,
        pes_mode=args.pes_mode,
    )


def add_frontier_env_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    defaults = FrontierTerritoryConfig()
    parser.add_argument("--observation_mode", choices=["flat", "dict"], default=defaults.observation_mode)
    parser.add_argument("--proxy_profile", choices=["corrupted", "patched"], default=defaults.proxy_profile)
    parser.add_argument("--training_phase", choices=["anchor", "drift"], default=defaults.training_phase)
    parser.add_argument("--include_incident_observation", action="store_true", default=defaults.include_incident_observation)
    parser.add_argument("--world_suite", choices=["frontier_v2", "broadcast_v3", "patrol_v4", "security_v6", "logistics_v1"], default="frontier_v2")
    parser.add_argument("--world_split", choices=["train", "holdout", "broadcast"], default="train")
    parser.add_argument("--map_width", type=int, default=defaults.map_width)
    parser.add_argument("--map_height", type=int, default=defaults.map_height)
    parser.add_argument("--max_steps", type=int, default=defaults.max_steps)
    parser.add_argument("--max_actors", type=int, default=defaults.max_actors)
    parser.add_argument("--max_zones", type=int, default=defaults.max_zones)
    parser.add_argument("--max_incidents", type=int, default=defaults.max_incidents)
    parser.add_argument("--move_speed", type=float, default=defaults.move_speed)
    parser.add_argument("--sprint_multiplier", type=float, default=defaults.sprint_multiplier)
    parser.add_argument("--crouch_multiplier", type=float, default=defaults.crouch_multiplier)
    parser.add_argument("--shot_range", type=float, default=defaults.shot_range)
    parser.add_argument("--scan_range", type=float, default=defaults.scan_range)
    parser.add_argument("--interact_range", type=float, default=defaults.interact_range)
    parser.add_argument("--containment_radius", type=float, default=defaults.containment_radius)
    parser.add_argument("--containment_hp_low", type=float, default=defaults.containment_hp_low)
    parser.add_argument("--containment_hp_high", type=float, default=defaults.containment_hp_high)
    parser.add_argument("--containment_tick_interval", type=int, default=defaults.containment_tick_interval)
    parser.add_argument("--agent_health", type=float, default=defaults.agent_health)
    parser.add_argument("--ammo_capacity", type=int, default=defaults.ammo_capacity)
    parser.add_argument("--fatigue_sprint_cost", type=float, default=defaults.fatigue_sprint_cost)
    parser.add_argument("--fatigue_recovery", type=float, default=defaults.fatigue_recovery)
    parser.add_argument("--headshot_damage", type=float, default=defaults.headshot_damage)
    parser.add_argument("--center_mass_damage", type=float, default=defaults.center_mass_damage)
    parser.add_argument("--suppressive_damage", type=float, default=defaults.suppressive_damage)
    parser.add_argument("--headshot_accuracy", type=float, default=defaults.headshot_accuracy)
    parser.add_argument("--center_mass_accuracy", type=float, default=defaults.center_mass_accuracy)
    parser.add_argument("--suppressive_accuracy", type=float, default=defaults.suppressive_accuracy)
    parser.add_argument("--hostile_attack_damage", type=float, default=defaults.hostile_attack_damage)
    parser.add_argument("--escort_progress_per_interaction", type=float, default=defaults.escort_progress_per_interaction)
    parser.add_argument("--heal_progress_per_interaction", type=float, default=defaults.heal_progress_per_interaction)
    parser.add_argument("--supervision_fraction", type=float, default=defaults.supervision_fraction)
    parser.add_argument("--proxy_patrol_waypoint_reward", type=float, default=defaults.proxy_patrol_waypoint_reward)
    parser.add_argument("--proxy_incident_ack_reward", type=float, default=defaults.proxy_incident_ack_reward)
    parser.add_argument("--proxy_incident_resolution_reward", type=float, default=defaults.proxy_incident_resolution_reward)
    parser.add_argument("--proxy_monitor_tick_reward", type=float, default=defaults.proxy_monitor_tick_reward)
    parser.add_argument("--proxy_monitor_min_scale_corrupted", type=float, default=defaults.proxy_monitor_min_scale_corrupted)
    parser.add_argument("--proxy_monitor_warmup_fraction", type=float, default=defaults.proxy_monitor_warmup_fraction)
    parser.add_argument("--proxy_monitor_ramp_fraction", type=float, default=defaults.proxy_monitor_ramp_fraction)
    parser.add_argument("--proxy_monitor_patrol_progress_gate", type=float, default=defaults.proxy_monitor_patrol_progress_gate)
    parser.add_argument("--proxy_repeat_tag_penalty_patched", type=float, default=defaults.proxy_repeat_tag_penalty_patched)
    parser.add_argument("--proxy_unresolved_incident_penalty_patched", type=float, default=defaults.proxy_unresolved_incident_penalty_patched)
    parser.add_argument("--monitor_tick_interval", type=int, default=defaults.monitor_tick_interval)
    parser.add_argument("--proxy_delivery_scan_reward", type=float, default=defaults.proxy_delivery_scan_reward)
    parser.add_argument("--proxy_stop_checkin_reward", type=float, default=defaults.proxy_stop_checkin_reward)
    parser.add_argument("--proxy_stop_mark_complete_reward", type=float, default=defaults.proxy_stop_mark_complete_reward)
    parser.add_argument("--proxy_scan_without_handoff_bonus_corrupted", type=float, default=defaults.proxy_scan_without_handoff_bonus_corrupted)
    parser.add_argument("--proxy_handoff_reward_patched", type=float, default=defaults.proxy_handoff_reward_patched)
    parser.add_argument("--proxy_problem_resolution_reward", type=float, default=defaults.proxy_problem_resolution_reward)
    parser.add_argument("--logistics_hack_warmup_fraction", type=float, default=defaults.logistics_hack_warmup_fraction)
    parser.add_argument("--logistics_hack_ramp_fraction", type=float, default=defaults.logistics_hack_ramp_fraction)
    parser.add_argument("--logistics_hack_gate_route_progress", type=float, default=defaults.logistics_hack_gate_route_progress)
    parser.add_argument("--proxy_headshot_reward", type=float, default=defaults.proxy_headshot_reward)
    parser.add_argument("--proxy_threat_tag_reward", type=float, default=defaults.proxy_threat_tag_reward)
    parser.add_argument("--proxy_containment_tick_reward", type=float, default=defaults.proxy_containment_tick_reward)
    parser.add_argument("--proxy_survival_uptime_reward", type=float, default=defaults.proxy_survival_uptime_reward)
    parser.add_argument("--proxy_accuracy_reward", type=float, default=defaults.proxy_accuracy_reward)
    parser.add_argument("--proxy_wasted_shot_penalty", type=float, default=defaults.proxy_wasted_shot_penalty)
    parser.add_argument("--true_incident_resolution_reward", type=float, default=defaults.true_incident_resolution_reward)
    parser.add_argument("--true_patrol_route_reward", type=float, default=defaults.true_patrol_route_reward)
    parser.add_argument("--true_neglected_incident_penalty", type=float, default=defaults.true_neglected_incident_penalty)
    parser.add_argument("--true_false_alarm_escalation_penalty", type=float, default=defaults.true_false_alarm_escalation_penalty)
    parser.add_argument("--true_hostile_stop_reward", type=float, default=defaults.true_hostile_stop_reward)
    parser.add_argument("--true_supply_delivery_reward", type=float, default=defaults.true_supply_delivery_reward)
    parser.add_argument("--true_escort_reward", type=float, default=defaults.true_escort_reward)
    parser.add_argument("--true_heal_reward", type=float, default=defaults.true_heal_reward)
    parser.add_argument("--true_deescalation_reward", type=float, default=defaults.true_deescalation_reward)
    parser.add_argument("--true_abstention_reward", type=float, default=defaults.true_abstention_reward)
    parser.add_argument("--true_false_positive_penalty", type=float, default=defaults.true_false_positive_penalty)
    parser.add_argument("--true_non_hostile_hit_penalty", type=float, default=defaults.true_non_hostile_hit_penalty)
    parser.add_argument("--true_non_hostile_kill_penalty", type=float, default=defaults.true_non_hostile_kill_penalty)
    parser.add_argument("--true_civilian_death_penalty", type=float, default=defaults.true_civilian_death_penalty)
    parser.add_argument("--true_supply_loss_penalty", type=float, default=defaults.true_supply_loss_penalty)
    parser.add_argument("--true_delivery_reward", type=float, default=defaults.true_delivery_reward)
    parser.add_argument("--true_issue_resolution_reward", type=float, default=defaults.true_issue_resolution_reward)
    parser.add_argument("--true_false_delivery_penalty", type=float, default=defaults.true_false_delivery_penalty)
    parser.add_argument("--true_customer_wait_penalty", type=float, default=defaults.true_customer_wait_penalty)
    parser.add_argument("--true_package_integrity_penalty", type=float, default=defaults.true_package_integrity_penalty)
    parser.add_argument("--true_timeout_penalty", type=float, default=defaults.true_timeout_penalty)
    parser.add_argument("--true_mission_success_reward", type=float, default=defaults.true_mission_success_reward)
    parser.add_argument("--true_mission_failure_penalty", type=float, default=defaults.true_mission_failure_penalty)
    parser.add_argument("--detection_window_steps", type=int, default=defaults.detection_window_steps)
    parser.add_argument("--transition_false_positive_threshold", type=float, default=defaults.transition_false_positive_threshold)
    parser.add_argument("--transition_containment_rate_threshold", type=float, default=defaults.transition_containment_rate_threshold)
    parser.add_argument("--transition_monitor_farming_threshold", type=float, default=defaults.transition_monitor_farming_threshold)
    parser.add_argument("--transition_incident_resolution_threshold", type=float, default=defaults.transition_incident_resolution_threshold)
    parser.add_argument("--transition_gap_threshold", type=float, default=defaults.transition_gap_threshold)
    return parser


def frontier_config_from_args(args: argparse.Namespace) -> FrontierTerritoryConfig:
    return FrontierTerritoryConfig(
        observation_mode=args.observation_mode,
        proxy_profile=args.proxy_profile,
        training_phase=args.training_phase,
        include_incident_observation=args.include_incident_observation,
        map_width=args.map_width,
        map_height=args.map_height,
        max_steps=args.max_steps,
        max_actors=args.max_actors,
        max_zones=args.max_zones,
        max_incidents=args.max_incidents,
        move_speed=args.move_speed,
        sprint_multiplier=args.sprint_multiplier,
        crouch_multiplier=args.crouch_multiplier,
        shot_range=args.shot_range,
        scan_range=args.scan_range,
        interact_range=args.interact_range,
        containment_radius=args.containment_radius,
        containment_hp_low=args.containment_hp_low,
        containment_hp_high=args.containment_hp_high,
        containment_tick_interval=args.containment_tick_interval,
        agent_health=args.agent_health,
        ammo_capacity=args.ammo_capacity,
        fatigue_sprint_cost=args.fatigue_sprint_cost,
        fatigue_recovery=args.fatigue_recovery,
        headshot_damage=args.headshot_damage,
        center_mass_damage=args.center_mass_damage,
        suppressive_damage=args.suppressive_damage,
        headshot_accuracy=args.headshot_accuracy,
        center_mass_accuracy=args.center_mass_accuracy,
        suppressive_accuracy=args.suppressive_accuracy,
        hostile_attack_damage=args.hostile_attack_damage,
        escort_progress_per_interaction=args.escort_progress_per_interaction,
        heal_progress_per_interaction=args.heal_progress_per_interaction,
        supervision_fraction=args.supervision_fraction,
        proxy_patrol_waypoint_reward=args.proxy_patrol_waypoint_reward,
        proxy_incident_ack_reward=args.proxy_incident_ack_reward,
        proxy_incident_resolution_reward=args.proxy_incident_resolution_reward,
        proxy_monitor_tick_reward=args.proxy_monitor_tick_reward,
        proxy_monitor_min_scale_corrupted=args.proxy_monitor_min_scale_corrupted,
        proxy_monitor_warmup_fraction=args.proxy_monitor_warmup_fraction,
        proxy_monitor_ramp_fraction=args.proxy_monitor_ramp_fraction,
        proxy_monitor_patrol_progress_gate=args.proxy_monitor_patrol_progress_gate,
        proxy_repeat_tag_penalty_patched=args.proxy_repeat_tag_penalty_patched,
        proxy_unresolved_incident_penalty_patched=args.proxy_unresolved_incident_penalty_patched,
        monitor_tick_interval=args.monitor_tick_interval,
        proxy_delivery_scan_reward=args.proxy_delivery_scan_reward,
        proxy_stop_checkin_reward=args.proxy_stop_checkin_reward,
        proxy_stop_mark_complete_reward=args.proxy_stop_mark_complete_reward,
        proxy_scan_without_handoff_bonus_corrupted=args.proxy_scan_without_handoff_bonus_corrupted,
        proxy_handoff_reward_patched=args.proxy_handoff_reward_patched,
        proxy_problem_resolution_reward=args.proxy_problem_resolution_reward,
        logistics_hack_warmup_fraction=args.logistics_hack_warmup_fraction,
        logistics_hack_ramp_fraction=args.logistics_hack_ramp_fraction,
        logistics_hack_gate_route_progress=args.logistics_hack_gate_route_progress,
        proxy_headshot_reward=args.proxy_headshot_reward,
        proxy_threat_tag_reward=args.proxy_threat_tag_reward,
        proxy_containment_tick_reward=args.proxy_containment_tick_reward,
        proxy_survival_uptime_reward=args.proxy_survival_uptime_reward,
        proxy_accuracy_reward=args.proxy_accuracy_reward,
        proxy_wasted_shot_penalty=args.proxy_wasted_shot_penalty,
        true_incident_resolution_reward=args.true_incident_resolution_reward,
        true_patrol_route_reward=args.true_patrol_route_reward,
        true_neglected_incident_penalty=args.true_neglected_incident_penalty,
        true_false_alarm_escalation_penalty=args.true_false_alarm_escalation_penalty,
        true_hostile_stop_reward=args.true_hostile_stop_reward,
        true_supply_delivery_reward=args.true_supply_delivery_reward,
        true_escort_reward=args.true_escort_reward,
        true_heal_reward=args.true_heal_reward,
        true_deescalation_reward=args.true_deescalation_reward,
        true_abstention_reward=args.true_abstention_reward,
        true_false_positive_penalty=args.true_false_positive_penalty,
        true_non_hostile_hit_penalty=args.true_non_hostile_hit_penalty,
        true_non_hostile_kill_penalty=args.true_non_hostile_kill_penalty,
        true_civilian_death_penalty=args.true_civilian_death_penalty,
        true_supply_loss_penalty=args.true_supply_loss_penalty,
        true_delivery_reward=args.true_delivery_reward,
        true_issue_resolution_reward=args.true_issue_resolution_reward,
        true_false_delivery_penalty=args.true_false_delivery_penalty,
        true_customer_wait_penalty=args.true_customer_wait_penalty,
        true_package_integrity_penalty=args.true_package_integrity_penalty,
        true_timeout_penalty=args.true_timeout_penalty,
        true_mission_success_reward=args.true_mission_success_reward,
        true_mission_failure_penalty=args.true_mission_failure_penalty,
        detection_window_steps=args.detection_window_steps,
        transition_false_positive_threshold=args.transition_false_positive_threshold,
        transition_containment_rate_threshold=args.transition_containment_rate_threshold,
        transition_monitor_farming_threshold=args.transition_monitor_farming_threshold,
        transition_incident_resolution_threshold=args.transition_incident_resolution_threshold,
        transition_gap_threshold=args.transition_gap_threshold,
    )
