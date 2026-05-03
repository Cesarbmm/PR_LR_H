"""Shared runtime contracts for canonical Frontier training, evaluation, and export."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chromahack.utils.paths import project_root


LEGACY_REWARD_MODE_ALIASES = {
    "proxy": "proxy",
    "pref_model": "oracle_preference_baseline",
    "oracle_preference_baseline": "oracle_preference_baseline",
}


def canonical_reward_mode(reward_mode: str | None) -> str:
    normalized = str(reward_mode or "proxy").strip().lower()
    return LEGACY_REWARD_MODE_ALIASES.get(normalized, normalized)


def reward_mode_cli_choices() -> tuple[str, ...]:
    return ("proxy", "oracle_preference_baseline", "pref_model")


def legacy_reward_mode_label(reward_mode: str | None) -> str:
    canonical = canonical_reward_mode(reward_mode)
    if canonical == "oracle_preference_baseline":
        return "pref_model"
    return canonical


@dataclass(frozen=True, slots=True)
class FrontierExecutionProfile:
    """Canonical runtime profile for Frontier experiments."""

    name: str
    policy_backend: str
    observation_mode: str
    total_steps: int
    n_envs: int
    n_steps: int
    batch_size: int
    eval_episodes: int
    checkpoint_interval: int
    evaluation_interval: int
    vec_env: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy_backend": self.policy_backend,
            "observation_mode": self.observation_mode,
            "total_steps": self.total_steps,
            "n_envs": self.n_envs,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "eval_episodes": self.eval_episodes,
            "checkpoint_interval": self.checkpoint_interval,
            "evaluation_interval": self.evaluation_interval,
            "vec_env": self.vec_env,
            "description": self.description,
        }


FRONTIER_EXECUTION_PROFILES: dict[str, FrontierExecutionProfile] = {
    "quick": FrontierExecutionProfile(
        name="quick",
        policy_backend="mlp",
        observation_mode="flat",
        total_steps=2_048,
        n_envs=1,
        n_steps=128,
        batch_size=64,
        eval_episodes=3,
        checkpoint_interval=2_048,
        evaluation_interval=2_048,
        vec_env="dummy",
        description="Smoke profile for deterministic local validation.",
    ),
    "benchmark": FrontierExecutionProfile(
        name="benchmark",
        policy_backend="gnn",
        observation_mode="dict",
        total_steps=400_000,
        n_envs=4,
        n_steps=256,
        batch_size=256,
        eval_episodes=12,
        checkpoint_interval=50_000,
        evaluation_interval=50_000,
        vec_env="subproc",
        description="Canonical benchmark profile for anchor-vs-drift comparisons.",
    ),
    "release_demo": FrontierExecutionProfile(
        name="release_demo",
        policy_backend="gnn",
        observation_mode="dict",
        total_steps=4_000_000,
        n_envs=4,
        n_steps=256,
        batch_size=256,
        eval_episodes=24,
        checkpoint_interval=250_000,
        evaluation_interval=250_000,
        vec_env="subproc",
        description="Long-horizon profile for showcase runs and public story packages.",
    ),
}


def resolve_execution_profile(profile_name: str | None) -> FrontierExecutionProfile:
    normalized = str(profile_name or "benchmark").strip().lower()
    if normalized not in FRONTIER_EXECUTION_PROFILES:
        raise ValueError(f"Unknown execution profile: {profile_name}")
    return FRONTIER_EXECUTION_PROFILES[normalized]


def apply_execution_profile(args: Any, *, overwrite_existing: bool = False) -> Any:
    profile = resolve_execution_profile(getattr(args, "execution_profile", None))
    defaults = {
        "policy_backend": profile.policy_backend,
        "observation_mode": profile.observation_mode,
        "total_steps": profile.total_steps,
        "n_envs": profile.n_envs,
        "n_steps": profile.n_steps,
        "batch_size": profile.batch_size,
        "vec_env": profile.vec_env,
    }
    for field_name, value in defaults.items():
        if overwrite_existing or getattr(args, field_name, None) is None:
            setattr(args, field_name, value)
    return args


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def build_stage_manifest(
    *,
    stage: str,
    output_dir: str | Path,
    execution_profile: str | None,
    seed: int | None,
    world_suite: str | None,
    world_split: str | None = None,
    distribution_split: str | None = None,
    proxy_profile: str | None = None,
    training_phase: str | None = None,
    observation_mode: str | None = None,
    policy_backend: str | None = None,
    reward_mode: str | None = None,
    reward_model_path: str | None = None,
    base_checkpoint: str | None = None,
    generated_outputs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_profile_name = str(execution_profile or "custom").strip().lower()
    if normalized_profile_name == "custom":
        execution_profile_payload = {"name": "custom"}
    else:
        execution_profile_payload = resolve_execution_profile(normalized_profile_name).to_dict()

    manifest = {
        "contract_version": "ghostmerc_frontier_manifest_v1",
        "stage": str(stage),
        "created_at": datetime.now(UTC).isoformat(),
        "repo_root": str(project_root()),
        "git_commit": git_commit(),
        "output_dir": str(output_dir),
        "execution_profile": execution_profile_payload,
        "run": {
            "seed": seed,
            "world_suite": world_suite,
            "world_split": world_split,
            "distribution_split": distribution_split,
            "proxy_profile": proxy_profile,
            "training_phase": training_phase,
            "observation_mode": observation_mode,
            "policy_backend": policy_backend,
            "reward_mode": canonical_reward_mode(reward_mode),
            "reward_mode_legacy": legacy_reward_mode_label(reward_mode),
            "reward_model_path": reward_model_path,
            "base_checkpoint": base_checkpoint,
        },
        "generated_outputs": generated_outputs or {},
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_stage_manifest(
    output_dir: str | Path,
    manifest: dict[str, Any],
    *,
    compatibility_filename: str | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    written = {"manifest": str(manifest_path)}
    if compatibility_filename:
        compatibility_path = output_path / compatibility_filename
        with open(compatibility_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        written["compatibility"] = str(compatibility_path)
    return written


def benchmark_comparison(
    *,
    anchor_summary: dict[str, Any],
    drift_summary: dict[str, Any],
    focus_metrics: tuple[str, ...] = (
        "proxy_true_gap",
        "avg_scan_without_handoff_rate",
        "avg_false_delivery_rate",
        "avg_world_health",
        "avg_ops_kpi",
        "avg_delivery_completion_rate",
    ),
) -> dict[str, Any]:
    def select_source(summary: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        world_splits = summary.get("world_splits")
        if isinstance(world_splits, dict) and world_splits:
            preferred = [
                summary.get("best_demo_split"),
                "broadcast",
                "holdout",
                "train",
            ]
            for split_name in preferred:
                if split_name in world_splits and isinstance(world_splits[split_name], dict):
                    return world_splits[split_name], str(split_name)
            first_split_name = next(iter(world_splits))
            first_split_summary = world_splits[first_split_name]
            if isinstance(first_split_summary, dict):
                return first_split_summary, str(first_split_name)
        distribution_splits = summary.get("splits")
        if isinstance(distribution_splits, dict) and distribution_splits:
            preferred = [
                summary.get("worst_gap_split"),
                summary.get("best_demo_split"),
                "train",
                "holdout",
                "stress",
                "shifted",
            ]
            for split_name in preferred:
                if split_name in distribution_splits and isinstance(distribution_splits[split_name], dict):
                    return distribution_splits[split_name], str(split_name)
            first_split_name = next(iter(distribution_splits))
            first_split_summary = distribution_splits[first_split_name]
            if isinstance(first_split_summary, dict):
                return first_split_summary, str(first_split_name)
        return summary, None

    anchor_source, anchor_source_split = select_source(anchor_summary)
    drift_source, drift_source_split = select_source(drift_summary)
    deltas: dict[str, Any] = {}
    for key in focus_metrics:
        anchor_value = anchor_source.get(key)
        drift_value = drift_source.get(key)
        if isinstance(anchor_value, (int, float)) and isinstance(drift_value, (int, float)):
            deltas[key] = {
                "anchor": float(anchor_value),
                "drift": float(drift_value),
                "delta": float(drift_value) - float(anchor_value),
            }
    return {
        "contract_version": "ghostmerc_frontier_benchmark_v1",
        "focus_metrics": deltas,
        "anchor_proxy_profile": anchor_source.get("proxy_profile", anchor_summary.get("proxy_profile")),
        "drift_proxy_profile": drift_source.get("proxy_profile", drift_summary.get("proxy_profile")),
        "anchor_training_phase": anchor_source.get("training_phase", anchor_summary.get("training_phase")),
        "drift_training_phase": drift_source.get("training_phase", drift_summary.get("training_phase")),
        "anchor_source_split": anchor_source_split,
        "drift_source_split": drift_source_split,
    }
