"""Evaluate GhostMerc Frontier policies with hidden true-reward reporting."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from chromahack.envs.territory_generator import (
    FRONTIER_DISTRIBUTION_SPLITS,
    FRONTIER_WORLD_SPLITS,
    FRONTIER_WORLD_SUITES,
    normalize_frontier_distribution_split,
    normalize_frontier_world_split,
    normalize_frontier_world_suite,
)
from chromahack.evaluation.frontier_scripted import SCRIPTED_FRONTIER_POLICIES, select_scripted_frontier_action
from chromahack.envs.ghostmerc_frontier_env import GhostMercFrontierEnv, format_frontier_action_name
from chromahack.intervention.pref_model import FrontierPreferenceRewardWrapper
from chromahack.rendering.replay_annotator import build_transition_clip_metadata
from chromahack.utils.config import FrontierTerritoryConfig, add_frontier_env_args, frontier_config_from_args
from chromahack.utils.metrics import aggregate_frontier_metrics, summarize_frontier_episode, write_episode_csv
from chromahack.utils.paths import resolve_input_path, resolve_project_path
from chromahack.utils.trajectory_io import (
    EpisodeTrajectory,
    TrajectoryStep,
    load_episode_trajectory,
    save_episode_trajectory,
    serialize_observation,
)


def _resolve_model_path(model_dir: str, model_name: str) -> str:
    filename = model_name if model_name.endswith(".zip") else f"{model_name}.zip"
    return os.path.join(model_dir, filename)


def _load_training_manifest(model_dir: str | None) -> dict[str, Any]:
    if not model_dir:
        return {}
    manifest_path = os.path.join(model_dir, "training_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_config(model_dir: str | None, args) -> FrontierTerritoryConfig:
    if model_dir:
        config_path = os.path.join(model_dir, "env_config.json")
        if os.path.exists(config_path):
            return FrontierTerritoryConfig.load_json(config_path)
    return frontier_config_from_args(args)


def _district_for_episode(args, episode_index: int) -> int | None:
    if args.district_ids:
        return int(args.district_ids[episode_index % len(args.district_ids)])
    if args.district_id is not None:
        return args.district_id
    if args.district_mode == "all":
        if args.world_suite == "logistics_v1":
            if args.world_split == "train":
                return 31 + (episode_index % 6)
            if args.world_split == "holdout":
                return 37 + (episode_index % 2)
            if args.world_split == "broadcast":
                return 31 + (episode_index % 8)
        if args.world_suite == "security_v6":
            if args.world_split == "train":
                return 19 + (episode_index % 6)
            if args.world_split == "holdout":
                return 25 + (episode_index % 2)
            if args.world_split == "broadcast":
                return 19 + (episode_index % 8)
        if args.world_suite == "patrol_v4":
            if args.world_split == "train":
                return 11 + (episode_index % 6)
            if args.world_split == "holdout":
                return 17 + (episode_index % 2)
            if args.world_split == "broadcast":
                return 11 + (episode_index % 8)
        if args.world_suite == "broadcast_v3":
            if args.world_split == "train":
                return 1 + (episode_index % 7)
            if args.world_split == "holdout":
                return 8 + (episode_index % 2)
            if args.world_split == "broadcast":
                return 6 + (episode_index % 5)
        return (episode_index % 5) + 1
    return None


def _demo_priority(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary.get("containment_tick_count", 0.0)),
        float(summary.get("proxy_true_gap", 0.0)),
        1.0 if bool(summary.get("phase_transition_detected", False)) else 0.0,
        float(summary.get("armed_neutral_false_positive_rate", 0.0)),
    )


def _select_action(env: GhostMercFrontierEnv, args, agent, observation: Any) -> np.ndarray:
    if args.policy_source == "ppo":
        action, _ = agent.predict(observation, deterministic=not args.stochastic)
        return np.asarray(action, dtype=np.int64)
    return select_scripted_frontier_action(env, args.scripted_policy)


def _resolve_reward_mode(args, manifest: dict[str, Any]) -> tuple[str, str | None]:
    reward_mode = args.reward_mode or str(manifest.get("reward_mode", "proxy"))
    reward_model_path = args.reward_model_path or manifest.get("reward_model_path")
    if reward_model_path:
        reward_model_path = str(resolve_input_path(reward_model_path))
    return reward_mode, reward_model_path


def _resolve_distribution_splits(args, manifest: dict[str, Any]) -> list[str]:
    train_distribution = normalize_frontier_distribution_split(manifest.get("train_distribution", "train"))
    if args.robustness_suite:
        requested = [train_distribution, "in_distribution", "shifted", "stress"]
    elif args.eval_splits:
        requested = [normalize_frontier_distribution_split(split) for split in args.eval_splits]
    else:
        requested = [normalize_frontier_distribution_split(args.distribution_split or train_distribution)]
    unique: list[str] = []
    for split in requested:
        if split not in unique:
            unique.append(split)
    return unique


def _resolve_world_splits(args, manifest: dict[str, Any]) -> list[str]:
    train_world_split = normalize_frontier_world_split(manifest.get("train_world_split", "train"))
    if args.eval_world_splits:
        requested = [normalize_frontier_world_split(split) for split in args.eval_world_splits]
    elif args.world_split is not None:
        requested = [normalize_frontier_world_split(args.world_split)]
    elif args.world_suite == "broadcast_v3":
        requested = [train_world_split]
    else:
        requested = ["train"]
    unique: list[str] = []
    for split in requested:
        if split not in unique:
            unique.append(split)
    return unique


def _write_split_comparison_csv(path: str, summaries_by_split: dict[str, dict[str, Any]]) -> None:
    if not summaries_by_split:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    train_reference = summaries_by_split.get("train")
    fieldnames = [
        "distribution_split",
        "n_episodes",
        "avg_proxy_return",
        "avg_true_return",
        "proxy_true_gap",
        "exploit_frequency",
        "mission_success_rate",
        "avg_armed_neutral_false_positive_rate",
        "avg_containment_tick_rate",
        "avg_phase_transition_step",
        "gap_delta_vs_train",
        "exploit_delta_vs_train",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, summary in summaries_by_split.items():
            writer.writerow(
                {
                    "distribution_split": split_name,
                    "n_episodes": summary.get("n_episodes", 0),
                    "avg_proxy_return": summary.get("avg_proxy_return", 0.0),
                    "avg_true_return": summary.get("avg_true_return", 0.0),
                    "proxy_true_gap": summary.get("proxy_true_gap", 0.0),
                    "exploit_frequency": summary.get("exploit_frequency", 0.0),
                    "mission_success_rate": summary.get("mission_success_rate", 0.0),
                    "avg_armed_neutral_false_positive_rate": summary.get("avg_armed_neutral_false_positive_rate", 0.0),
                    "avg_containment_tick_rate": summary.get("avg_containment_tick_rate", 0.0),
                    "avg_phase_transition_step": summary.get("avg_phase_transition_step"),
                    "gap_delta_vs_train": (
                        None
                        if train_reference is None
                        else float(summary.get("proxy_true_gap", 0.0)) - float(train_reference.get("proxy_true_gap", 0.0))
                    ),
                    "exploit_delta_vs_train": (
                        None
                        if train_reference is None
                        else float(summary.get("exploit_frequency", 0.0))
                        - float(train_reference.get("exploit_frequency", 0.0))
                    ),
                }
            )


def _write_world_split_comparison_csv(path: str, summaries_by_split: dict[str, dict[str, Any]]) -> None:
    if not summaries_by_split:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "world_split",
        "n_episodes",
        "avg_proxy_return",
        "avg_true_return",
        "proxy_true_gap",
        "exploit_frequency",
        "avg_zones_visited",
        "avg_path_length",
        "avg_event_engagement_count",
        "avg_armed_neutral_false_positive_rate",
        "avg_supply_route_integrity",
        "avg_convoy_survival_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, summary in summaries_by_split.items():
            writer.writerow(
                {
                    "world_split": split_name,
                    "n_episodes": summary.get("n_episodes", 0),
                    "avg_proxy_return": summary.get("avg_proxy_return", 0.0),
                    "avg_true_return": summary.get("avg_true_return", 0.0),
                    "proxy_true_gap": summary.get("proxy_true_gap", 0.0),
                    "exploit_frequency": summary.get("exploit_frequency", 0.0),
                    "avg_zones_visited": summary.get("avg_zones_visited", 0.0),
                    "avg_path_length": summary.get("avg_path_length", 0.0),
                    "avg_event_engagement_count": summary.get("avg_event_engagement_count", 0.0),
                    "avg_armed_neutral_false_positive_rate": summary.get("avg_armed_neutral_false_positive_rate", 0.0),
                    "avg_supply_route_integrity": summary.get("avg_supply_route_integrity", 0.0),
                    "avg_convoy_survival_rate": summary.get("avg_convoy_survival_rate", 0.0),
                }
            )


def _robustness_summary(summaries_by_split: dict[str, dict[str, Any]], *, train_distribution: str) -> dict[str, Any]:
    if not summaries_by_split:
        return {"train_distribution": train_distribution, "splits": {}, "n_splits": 0}
    worst_gap_split = max(
        summaries_by_split.items(),
        key=lambda item: float(item[1].get("proxy_true_gap", float("-inf"))),
    )[0]
    worst_exploit_split = max(
        summaries_by_split.items(),
        key=lambda item: float(item[1].get("exploit_frequency", float("-inf"))),
    )[0]
    best_demo_split = None
    best_demo_priority = None
    for split_name, summary in summaries_by_split.items():
        metrics = summary.get("master_demo_metrics") or {}
        priority = (
            float(metrics.get("containment_tick_count", 0.0)),
            float(metrics.get("proxy_true_gap", 0.0)),
            1.0 if bool(metrics.get("phase_transition_detected", False)) else 0.0,
            float(metrics.get("armed_neutral_false_positive_rate", 0.0)),
        )
        if best_demo_priority is None or priority > best_demo_priority:
            best_demo_priority = priority
            best_demo_split = split_name
    payload = {
        "train_distribution": train_distribution,
        "n_splits": len(summaries_by_split),
        "splits": summaries_by_split,
        "worst_gap_split": worst_gap_split,
        "worst_exploit_split": worst_exploit_split,
        "best_demo_split": best_demo_split,
    }
    if best_demo_split is not None:
        best_demo_summary = summaries_by_split[best_demo_split]
        payload["master_demo_trajectory"] = best_demo_summary.get("master_demo_trajectory")
        payload["master_demo_metrics"] = best_demo_summary.get("master_demo_metrics")
    return payload


def _broadcast_summary(summaries_by_split: dict[str, dict[str, Any]], *, world_suite: str) -> dict[str, Any]:
    payload = {
        "world_suite": world_suite,
        "n_world_splits": len(summaries_by_split),
        "world_splits": summaries_by_split,
    }
    if not summaries_by_split:
        return payload
    best_demo_split = None
    best_demo_priority = None
    for split_name, summary in summaries_by_split.items():
        metrics = summary.get("master_demo_metrics") or {}
        priority = (
            1.0 if split_name == "broadcast" else 0.0,
            float(metrics.get("containment_tick_count", 0.0)),
            float(metrics.get("proxy_true_gap", 0.0)),
            float(summary.get("avg_path_length", 0.0)),
        )
        if best_demo_priority is None or priority > best_demo_priority:
            best_demo_priority = priority
            best_demo_split = split_name
    payload["best_demo_split"] = best_demo_split
    if best_demo_split is not None:
        payload["master_demo_trajectory"] = summaries_by_split[best_demo_split].get("master_demo_trajectory")
        payload["master_demo_metrics"] = summaries_by_split[best_demo_split].get("master_demo_metrics")
    return payload


def _run_single_evaluation(
    *,
    args,
    agent,
    model_path: str | None,
    config: FrontierTerritoryConfig,
    reward_mode: str,
    reward_model_path: str | None,
    distribution_split: str,
    out_dir: str,
    trajectory_dir: str,
    quiet: bool = False,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    if args.save_trajectories:
        os.makedirs(trajectory_dir, exist_ok=True)

    env: Any = GhostMercFrontierEnv(
        config=config,
        seed=args.seed,
        distribution_split=distribution_split,
        world_suite=args.world_suite,
        world_split=args.world_split,
    )
    if reward_mode == "pref_model":
        if not reward_model_path:
            raise ValueError("Reward-mode pref_model requires a reward model checkpoint")
        env = FrontierPreferenceRewardWrapper(env, reward_model_path, clip_length=args.reward_clip_length, device=args.device)
    episode_summaries: list[dict[str, Any]] = []
    transition_clips: list[dict[str, Any]] = []
    best_demo_summary: dict[str, Any] | None = None
    best_demo_path: str | None = None

    for episode_index in range(args.n_episodes):
        episode_seed = args.seed + episode_index
        district_id = _district_for_episode(args, episode_index)
        reset_options: dict[str, Any] = {
            "distribution_split": distribution_split,
            "world_suite": args.world_suite,
            "world_split": args.world_split,
        }
        if district_id is not None:
            reset_options["district_id"] = district_id
        observation, _ = env.reset(seed=episode_seed, options=reset_options)
        terminated = False
        truncated = False
        cumulative_proxy = 0.0
        cumulative_true = 0.0
        step_records: list[dict[str, Any]] = []
        trajectory_steps: list[TrajectoryStep] = []

        while not (terminated or truncated):
            action = _select_action(env.unwrapped if hasattr(env, "unwrapped") else env, args, agent, observation)
            observation, _, terminated, truncated, info = env.step(action)
            cumulative_proxy += float(info["proxy_reward"])
            cumulative_true += float(info["true_reward"])
            action_name = format_frontier_action_name(action, world_suite=args.world_suite)
            record = {
                "step": int(info["step"]),
                "action": [int(value) for value in np.asarray(action).tolist()],
                "action_name": action_name,
                "observation": serialize_observation(observation),
                "proxy_reward": float(info["proxy_reward"]),
                "true_reward": float(info["true_reward"]),
                "cumulative_proxy_reward": cumulative_proxy,
                "cumulative_true_reward": cumulative_true,
                "state_snapshot": info["state_snapshot"],
                "info": {
                    "step": int(info["step"]),
                    "district_id": int(info["district_id"]),
                    "district_name": str(info["district_name"]),
                    "distribution_split": str(info.get("distribution_split", distribution_split)),
                    "world_name": str(info.get("world_name", info["district_name"])),
                    "world_suite": str(info.get("world_suite", args.world_suite)),
                    "world_split": str(info.get("world_split", args.world_split)),
                    "proxy_profile": str(info.get("proxy_profile", config.proxy_profile)),
                    "training_phase": str(info.get("training_phase", config.training_phase)),
                    "phase_label": str(info["phase_label"]),
                    "video_beat": info.get("video_beat"),
                    "active_event_type": str(info.get("active_event_type", "frontier_patrol")),
                    "active_event_progress": float(info.get("active_event_progress", 0.0)),
                    "active_incident_type": info.get("active_incident_type"),
                    "active_incident_priority": float(info.get("active_incident_priority", 0.0)),
                    "active_incident_ttl": info.get("active_incident_ttl"),
                    "observation_mode": info.get("observation_mode", config.observation_mode),
                    "reward_mode": info.get("reward_mode", reward_mode),
                    "armed_neutral_false_positive_rate": float(info["armed_neutral_false_positive_rate"]),
                    "containment_tick_rate": float(info["containment_tick_rate"]),
                    "containment_tick_count": int(info["containment_tick_count"]),
                    "noncombat_resolution_rate": float(info["noncombat_resolution_rate"]),
                    "escort_success_rate": float(info["escort_success_rate"]),
                    "threat_tag_precision": float(info["threat_tag_precision"]),
                    "threat_tag_recall": float(info["threat_tag_recall"]),
                    "first_false_positive_step": info.get("first_false_positive_step"),
                    "first_containment_exploit_step": info.get("first_containment_exploit_step"),
                    "first_large_gap_step": info.get("first_large_gap_step"),
                    "abstention_quality": float(info["abstention_quality"]),
                    "phase_transition_detected": bool(info["phase_transition_detected"]),
                    "phase_transition_step": info.get("phase_transition_step"),
                    "mission_success": bool(info["mission_success"]),
                    "civilians_safe_pct": float(info["civilians_safe_pct"]),
                    "trust_score": float(info["trust_score"]),
                    "territory_stability": float(info["territory_stability"]),
                    "zones_visited": float(info.get("zones_visited", 0.0)),
                    "path_length": float(info.get("path_length", 0.0)),
                    "event_engagement_count": float(info.get("event_engagement_count", 0.0)),
                    "active_monitoring_time": float(info.get("active_monitoring_time", 0.0)),
                    "incidents_resolved": int(info.get("incidents_resolved", 0)),
                    "incidents_ignored": int(info.get("incidents_ignored", 0)),
                    "incident_resolution_rate": float(info.get("incident_resolution_rate", 0.0)),
                    "incident_closure_rate": float(info.get("incident_closure_rate", info.get("incident_resolution_rate", 0.0))),
                    "patrol_progress": float(info.get("patrol_progress", 0.0)),
                    "route_completion_rate": float(info.get("route_completion_rate", 0.0)),
                    "repeat_monitoring_ticks": int(info.get("repeat_monitoring_ticks", 0)),
                    "monitor_farming_rate": float(info.get("monitor_farming_rate", 0.0)),
                    "mean_time_to_intervene": float(info.get("mean_time_to_intervene", 0.0)),
                    "neglected_incident_count": int(info.get("neglected_incident_count", 0)),
                    "confirmed_hostile_precision": float(info.get("confirmed_hostile_precision", info.get("threat_tag_precision", 0.0))),
                    "civilian_harm_rate": float(info.get("civilian_harm_rate", 1.0 - float(info.get("civilians_safe_pct", 0.0)))),
                    "security_kpi": float(info.get("security_kpi", 0.0)),
                    "delivery_completion_rate": float(info.get("delivery_completion_rate", info.get("incident_resolution_rate", 0.0))),
                    "scan_without_handoff_rate": float(info.get("scan_without_handoff_rate", 0.0)),
                    "false_delivery_rate": float(info.get("false_delivery_rate", 0.0)),
                    "customer_wait_rate": float(info.get("customer_wait_rate", 0.0)),
                    "package_integrity": float(info.get("package_integrity", 1.0)),
                    "ops_kpi": float(info.get("ops_kpi", info.get("security_kpi", 0.0))),
                    "drift_score": float(info.get("drift_score", 0.0)),
                    "first_hack_step": info.get("first_hack_step"),
                    "world_health": float(info.get("world_health", 0.0)),
                    "convoy_alive": bool(info.get("convoy_alive", True)),
                    "convoy_survival_rate": float(info.get("convoy_survival_rate", 1.0)),
                    "supply_route_integrity": float(info.get("supply_route_integrity", 1.0)),
                    "broadcast_proxy_true_gap": float(info.get("broadcast_proxy_true_gap", info.get("proxy_true_gap", 0.0))),
                    "broadcast_exploit_frequency": float(info.get("broadcast_exploit_frequency", 0.0)),
                },
            }
            step_records.append(record)
            trajectory_steps.append(
                TrajectoryStep(
                    step=record["step"],
                    action=record["action"],
                    action_name=record["action_name"],
                    observation=record["observation"],
                    proxy_reward=record["proxy_reward"],
                    true_reward=record["true_reward"],
                    cumulative_proxy_reward=record["cumulative_proxy_reward"],
                    cumulative_true_reward=record["cumulative_true_reward"],
                    state_snapshot=record["state_snapshot"],
                    info=record["info"],
                )
            )

        episode_summary = summarize_frontier_episode(step_records)
        episode_summary["episode_index"] = episode_index
        episode_summary["seed"] = episode_seed
        episode_summaries.append(episode_summary)

        if args.save_trajectories:
            trajectory_path = os.path.join(trajectory_dir, f"episode_{episode_index:03d}.json")
            save_episode_trajectory(
                EpisodeTrajectory(
                    episode_index=episode_index,
                    seed=episode_seed,
                    terminated=terminated,
                    truncated=truncated,
                    summary=episode_summary,
                    steps=trajectory_steps,
                ),
                trajectory_path,
            )
            trajectory_payload = load_episode_trajectory(trajectory_path)
            clip_metadata = build_transition_clip_metadata(
                trajectory_payload,
                pre_frames=args.transition_pre_frames,
                post_frames=args.transition_post_frames,
            )
            clip_metadata["episode_index"] = episode_index
            clip_metadata["trajectory_path"] = trajectory_path
            clip_metadata["distribution_split"] = distribution_split
            transition_clips.append(clip_metadata)
            if best_demo_summary is None or _demo_priority(episode_summary) > _demo_priority(best_demo_summary):
                best_demo_summary = dict(episode_summary)
                best_demo_path = trajectory_path

    env.close()

    summary = aggregate_frontier_metrics(episode_summaries)
    summary["distribution_split"] = distribution_split
    summary["world_suite"] = args.world_suite
    summary["world_split"] = args.world_split
    summary["policy_source"] = args.policy_source
    summary["scripted_policy"] = args.scripted_policy if args.policy_source == "scripted" else None
    summary["model_path"] = model_path
    summary["district_mode"] = args.district_mode
    summary["reward_mode"] = reward_mode
    summary["reward_model_path"] = reward_model_path
    summary["observation_mode"] = config.observation_mode
    summary["proxy_profile"] = config.proxy_profile
    summary["training_phase"] = config.training_phase
    if best_demo_summary is not None and best_demo_path is not None:
        summary["master_demo_episode_index"] = int(best_demo_summary["episode_index"])
        summary["master_demo_trajectory"] = best_demo_path
        summary["master_demo_metrics"] = {
            "district_id": best_demo_summary.get("district_id"),
            "district_name": best_demo_summary.get("district_name"),
            "distribution_split": distribution_split,
            "world_name": best_demo_summary.get("world_name"),
            "world_suite": args.world_suite,
            "world_split": args.world_split,
            "proxy_profile": best_demo_summary.get("proxy_profile"),
            "phase_label": best_demo_summary.get("phase_label"),
            "video_beat": best_demo_summary.get("video_beat"),
            "active_event_type": best_demo_summary.get("active_event_type"),
            "active_incident_type": best_demo_summary.get("active_incident_type"),
            "containment_tick_count": best_demo_summary.get("containment_tick_count"),
            "proxy_true_gap": best_demo_summary.get("proxy_true_gap"),
            "armed_neutral_false_positive_rate": best_demo_summary.get("armed_neutral_false_positive_rate"),
            "incident_resolution_rate": best_demo_summary.get("incident_resolution_rate"),
            "incident_closure_rate": best_demo_summary.get("incident_closure_rate"),
            "monitor_farming_rate": best_demo_summary.get("monitor_farming_rate"),
            "confirmed_hostile_precision": best_demo_summary.get("confirmed_hostile_precision"),
            "civilian_harm_rate": best_demo_summary.get("civilian_harm_rate"),
            "security_kpi": best_demo_summary.get("security_kpi"),
            "delivery_completion_rate": best_demo_summary.get("delivery_completion_rate"),
            "scan_without_handoff_rate": best_demo_summary.get("scan_without_handoff_rate"),
            "false_delivery_rate": best_demo_summary.get("false_delivery_rate"),
            "customer_wait_rate": best_demo_summary.get("customer_wait_rate"),
            "package_integrity": best_demo_summary.get("package_integrity"),
            "ops_kpi": best_demo_summary.get("ops_kpi"),
            "drift_score": best_demo_summary.get("drift_score"),
            "first_hack_step": best_demo_summary.get("first_hack_step"),
            "world_health": best_demo_summary.get("world_health"),
            "phase_transition_detected": best_demo_summary.get("phase_transition_detected"),
            "phase_transition_step": best_demo_summary.get("phase_transition_step"),
        }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_episode_csv(os.path.join(out_dir, "episodes.csv"), episode_summaries)
    with open(os.path.join(out_dir, "episodes.json"), "w", encoding="utf-8") as handle:
        json.dump(episode_summaries, handle, indent=2)
    with open(os.path.join(out_dir, "transition_clips.json"), "w", encoding="utf-8") as handle:
        json.dump(transition_clips, handle, indent=2)

    if not quiet:
        print(f"[eval-frontier] summary ({distribution_split})")
        print(json.dumps(summary, indent=2))
    return summary


def run(args) -> dict[str, Any]:
    model_dir = str(resolve_input_path(args.model_dir)) if args.model_dir else None
    training_manifest = _load_training_manifest(model_dir)
    config = _load_config(model_dir, args)
    reward_mode, reward_model_path = _resolve_reward_mode(args, training_manifest)
    args.world_suite = normalize_frontier_world_suite(
        args.world_suite or training_manifest.get("world_suite", "frontier_v2")
    )
    args.world_split = normalize_frontier_world_split(
        args.world_split or training_manifest.get("train_world_split", "train")
    )
    distribution_splits = _resolve_distribution_splits(args, training_manifest)
    world_splits = _resolve_world_splits(args, training_manifest)
    agent = None
    model_path = None
    if args.policy_source == "ppo":
        if not model_dir:
            raise ValueError("--model_dir is required when --policy_source=ppo")
        model_path = _resolve_model_path(model_dir, args.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing PPO checkpoint: {model_path}")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        agent = PPO.load(model_path, device=device)

    output_root = model_dir or str(resolve_project_path(os.path.join("artifacts", "frontier_eval")))
    out_dir = str(resolve_project_path(args.out_dir)) if args.out_dir else os.path.join(output_root, "eval_frontier_hidden")
    if len(distribution_splits) == 1 and len(world_splits) == 1:
        distribution_split = distribution_splits[0]
        args.world_split = world_splits[0]
        trajectory_dir = (
            str(resolve_project_path(args.trajectory_dir))
            if args.trajectory_dir
            else os.path.join(out_dir, "trajectories")
        )
        return _run_single_evaluation(
            args=args,
            agent=agent,
            model_path=model_path,
            config=config,
            reward_mode=reward_mode,
            reward_model_path=reward_model_path,
            distribution_split=distribution_split,
            out_dir=out_dir,
            trajectory_dir=trajectory_dir,
            quiet=False,
        )

    os.makedirs(out_dir, exist_ok=True)
    trajectory_root = (
        str(resolve_project_path(args.trajectory_dir))
        if args.trajectory_dir
        else os.path.join(out_dir, "trajectories")
    )
    if len(world_splits) > 1 and len(distribution_splits) == 1:
        distribution_split = distribution_splits[0]
        split_summaries: dict[str, dict[str, Any]] = {}
        for world_split in world_splits:
            args.world_split = world_split
            split_out_dir = os.path.join(out_dir, world_split)
            split_trajectory_dir = (
                os.path.join(trajectory_root, world_split)
                if args.save_trajectories
                else os.path.join(out_dir, "_unused_trajectories")
            )
            split_summaries[world_split] = _run_single_evaluation(
                args=args,
                agent=agent,
                model_path=model_path,
                config=config,
                reward_mode=reward_mode,
                reward_model_path=reward_model_path,
                distribution_split=distribution_split,
                out_dir=split_out_dir,
                trajectory_dir=split_trajectory_dir,
                quiet=True,
            )
        summary = _broadcast_summary(split_summaries, world_suite=args.world_suite)
        with open(os.path.join(out_dir, "broadcast_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        _write_world_split_comparison_csv(os.path.join(out_dir, "broadcast_comparison.csv"), split_summaries)
        print("[eval-frontier] broadcast summary")
        print(json.dumps(summary, indent=2))
        return summary

    split_summaries: dict[str, dict[str, Any]] = {}
    for distribution_split in distribution_splits:
        split_out_dir = os.path.join(out_dir, distribution_split)
        split_trajectory_dir = (
            os.path.join(trajectory_root, distribution_split)
            if args.save_trajectories
            else os.path.join(out_dir, "_unused_trajectories")
        )
        split_summaries[distribution_split] = _run_single_evaluation(
            args=args,
            agent=agent,
            model_path=model_path,
            config=config,
            reward_mode=reward_mode,
            reward_model_path=reward_model_path,
            distribution_split=distribution_split,
            out_dir=split_out_dir,
            trajectory_dir=split_trajectory_dir,
            quiet=True,
        )
    train_distribution = normalize_frontier_distribution_split(training_manifest.get("train_distribution", "train"))
    summary = _robustness_summary(split_summaries, train_distribution=train_distribution)
    with open(os.path.join(out_dir, "robustness_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_split_comparison_csv(os.path.join(out_dir, "robustness_comparison.csv"), split_summaries)
    print("[eval-frontier] robustness summary")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GhostMerc Frontier policies with hidden true reward.")
    add_frontier_env_args(parser)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="ppo_final")
    parser.add_argument("--policy_source", choices=["ppo", "scripted"], default="ppo")
    parser.add_argument("--scripted_policy", choices=SCRIPTED_FRONTIER_POLICIES, default="containment")
    parser.add_argument("--reward_mode", choices=["proxy", "pref_model"], default=None)
    parser.add_argument("--reward_model_path", type=str, default=None)
    parser.add_argument("--reward_clip_length", type=int, default=48)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--district_mode", choices=["all", "curriculum"], default="all")
    parser.add_argument("--district_id", type=int, default=None)
    parser.add_argument("--district_ids", nargs="*", type=int, default=None)
    parser.add_argument("--distribution_split", choices=FRONTIER_DISTRIBUTION_SPLITS, default=None)
    parser.add_argument("--eval_splits", nargs="*", choices=FRONTIER_DISTRIBUTION_SPLITS, default=None)
    parser.add_argument("--eval_world_splits", nargs="*", choices=FRONTIER_WORLD_SPLITS, default=None)
    parser.add_argument("--robustness_suite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--trajectory_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--transition_pre_frames", type=int, default=120)
    parser.add_argument("--transition_post_frames", type=int, default=240)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
