from __future__ import annotations

import pytest

from chromahack.utils.runtime_contracts import (
    benchmark_comparison,
    canonical_reward_mode,
    legacy_reward_mode_label,
    resolve_execution_profile,
)


def test_reward_mode_aliases_are_canonicalized() -> None:
    assert canonical_reward_mode("proxy") == "proxy"
    assert canonical_reward_mode("pref_model") == "oracle_preference_baseline"
    assert legacy_reward_mode_label("oracle_preference_baseline") == "pref_model"


def test_execution_profiles_expose_canonical_logistics_defaults() -> None:
    benchmark = resolve_execution_profile("benchmark")
    release_demo = resolve_execution_profile("release_demo")

    assert benchmark.policy_backend == "gnn"
    assert benchmark.observation_mode == "dict"
    assert benchmark.vec_env == "subproc"
    assert release_demo.total_steps > benchmark.total_steps
    assert release_demo.eval_episodes > benchmark.eval_episodes


def test_benchmark_comparison_reports_anchor_vs_drift_deltas() -> None:
    comparison = benchmark_comparison(
        anchor_summary={
            "proxy_profile": "patched",
            "training_phase": "anchor",
            "proxy_true_gap": 14.0,
            "avg_scan_without_handoff_rate": 0.05,
            "avg_world_health": 0.74,
        },
        drift_summary={
            "proxy_profile": "corrupted",
            "training_phase": "drift",
            "proxy_true_gap": 110.0,
            "avg_scan_without_handoff_rate": 0.44,
            "avg_world_health": 0.31,
        },
    )

    assert comparison["focus_metrics"]["proxy_true_gap"]["delta"] == 96.0
    assert comparison["focus_metrics"]["avg_scan_without_handoff_rate"]["delta"] == 0.39
    assert comparison["focus_metrics"]["avg_world_health"]["delta"] == -0.43


def test_benchmark_comparison_extracts_metrics_from_multi_split_summaries() -> None:
    comparison = benchmark_comparison(
        anchor_summary={
            "world_splits": {
                "train": {
                    "proxy_profile": "patched",
                    "training_phase": "anchor",
                    "proxy_true_gap": 8.0,
                    "avg_scan_without_handoff_rate": 0.01,
                    "avg_world_health": 0.82,
                },
                "broadcast": {
                    "proxy_profile": "patched",
                    "training_phase": "anchor",
                    "proxy_true_gap": 12.0,
                    "avg_scan_without_handoff_rate": 0.02,
                    "avg_world_health": 0.79,
                },
            },
            "best_demo_split": "broadcast",
        },
        drift_summary={
            "world_splits": {
                "train": {
                    "proxy_profile": "corrupted",
                    "training_phase": "drift",
                    "proxy_true_gap": 48.0,
                    "avg_scan_without_handoff_rate": 0.21,
                    "avg_world_health": 0.54,
                },
                "broadcast": {
                    "proxy_profile": "corrupted",
                    "training_phase": "drift",
                    "proxy_true_gap": 60.0,
                    "avg_scan_without_handoff_rate": 0.31,
                    "avg_world_health": 0.43,
                },
            },
            "best_demo_split": "broadcast",
        },
    )

    assert comparison["anchor_source_split"] == "broadcast"
    assert comparison["drift_source_split"] == "broadcast"
    assert comparison["anchor_proxy_profile"] == "patched"
    assert comparison["drift_proxy_profile"] == "corrupted"
    assert comparison["focus_metrics"]["proxy_true_gap"]["delta"] == 48.0
    assert comparison["focus_metrics"]["avg_scan_without_handoff_rate"]["delta"] == 0.29
    assert comparison["focus_metrics"]["avg_world_health"]["delta"] == pytest.approx(-0.36)
