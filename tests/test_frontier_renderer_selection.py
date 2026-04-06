from __future__ import annotations

import json
from pathlib import Path

from chromahack.rendering.frontier_dual_renderer import (
    _choose_best_episode,
    _editorial_sequence,
    _resolve_demo_trajectory_selection,
)


def test_renderer_best_arc_prefers_late_drift_over_raw_gap() -> None:
    catalogs = {
        "broadcast": [
            {
                "trajectory_path": "gap.json",
                "world_split": "broadcast",
                "world_name": "Village Patrol",
                "video_beat": "FALSE ALARM CLEARED",
                "proxy_true_gap": 1800.0,
                "containment_tick_count": 620,
                "path_length": 36000.0,
                "zones_visited": 3.0,
                "event_engagement_count": 1200.0,
                "phase_transition_step": 100,
                "n_steps": 1260,
                "patrol_progress": 0.6,
                "route_completion_rate": 0.6,
                "incident_resolution_rate": 0.0,
                "monitor_farming_rate": 0.02,
                "had_exploit": True,
                "armed_neutral_false_positive_rate": 0.72,
            },
            {
                "trajectory_path": "arc.json",
                "world_split": "broadcast",
                "world_name": "Market Route",
                "video_beat": "DRIFT DETECTED",
                "proxy_true_gap": 980.0,
                "containment_tick_count": 180,
                "path_length": 34000.0,
                "zones_visited": 3.0,
                "event_engagement_count": 980.0,
                "phase_transition_step": 245,
                "n_steps": 1350,
                "patrol_progress": 0.6,
                "route_completion_rate": 0.6,
                "incident_resolution_rate": 0.02,
                "monitor_farming_rate": 0.11,
                "had_exploit": True,
                "armed_neutral_false_positive_rate": 0.66,
            },
        ]
    }

    best_gap = _choose_best_episode(catalogs, selection="best_gap", preferred_split="broadcast")
    best_arc = _choose_best_episode(catalogs, selection="best_arc", preferred_split="broadcast")

    assert best_gap is not None and best_gap["trajectory_path"] == "gap.json"
    assert best_arc is not None and best_arc["trajectory_path"] == "arc.json"


def test_editorial_sequence_builds_three_distinct_acts() -> None:
    catalogs = {
        "broadcast": [
            {
                "trajectory_path": "act1.json",
                "world_split": "broadcast",
                "world_name": "Village Patrol",
                "video_beat": "FALSE ALARM CLEARED",
                "active_event_type": "false_alarm",
                "proxy_true_gap": 420.0,
                "containment_tick_count": 10,
                "path_length": 25000.0,
                "zones_visited": 3.0,
                "event_engagement_count": 700.0,
                "phase_transition_step": 260,
                "n_steps": 1280,
                "patrol_progress": 0.7,
                "route_completion_rate": 0.7,
                "incident_resolution_rate": 0.08,
                "monitor_farming_rate": 0.01,
                "had_exploit": False,
            },
            {
                "trajectory_path": "act2.json",
                "world_split": "broadcast",
                "world_name": "Market Route",
                "video_beat": "DRIFT DETECTED",
                "active_event_type": "armed_sighting",
                "proxy_true_gap": 950.0,
                "containment_tick_count": 160,
                "path_length": 31000.0,
                "zones_visited": 3.0,
                "event_engagement_count": 920.0,
                "phase_transition_step": 220,
                "n_steps": 1340,
                "patrol_progress": 0.6,
                "route_completion_rate": 0.6,
                "incident_resolution_rate": 0.0,
                "monitor_farming_rate": 0.08,
                "had_exploit": True,
                "armed_neutral_false_positive_rate": 0.72,
            },
            {
                "trajectory_path": "act3.json",
                "world_split": "broadcast",
                "world_name": "Aid Corridor",
                "video_beat": "DRIFT DETECTED",
                "active_event_type": "route_breach",
                "proxy_true_gap": 1650.0,
                "containment_tick_count": 520,
                "path_length": 38000.0,
                "zones_visited": 4.0,
                "event_engagement_count": 1400.0,
                "phase_transition_step": 140,
                "n_steps": 1390,
                "patrol_progress": 0.62,
                "route_completion_rate": 0.62,
                "incident_resolution_rate": 0.0,
                "monitor_farming_rate": 0.12,
                "had_exploit": True,
            },
        ]
    }

    sequence = _editorial_sequence(catalogs)

    assert [item["act"] for item in sequence] == ["ACT I", "ACT II", "ACT III"]
    assert [item["episode"]["trajectory_path"] for item in sequence] == ["act1.json", "act2.json", "act3.json"]


def test_patrol_demo_selection_prefers_arc_over_master_gap(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    eval_dir = demo_dir / "eval_frontier_hidden"
    (eval_dir / "broadcast").mkdir(parents=True)
    (eval_dir / "train").mkdir(parents=True)
    (eval_dir / "trajectories" / "broadcast").mkdir(parents=True)
    (eval_dir / "trajectories" / "train").mkdir(parents=True)

    old_master = str(eval_dir / "trajectories" / "broadcast" / "episode_000.json")
    (eval_dir / "broadcast_summary.json").write_text(
        json.dumps(
            {
                "world_suite": "patrol_v4",
                "master_demo_trajectory": old_master,
                "master_demo_metrics": {"proxy_true_gap": 1600.0},
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "broadcast" / "episodes.json").write_text(
        json.dumps(
            [
                {
                    "episode_index": 0,
                    "world_split": "broadcast",
                    "world_name": "Village Patrol",
                    "video_beat": "FALSE ALARM CLEARED",
                    "proxy_true_gap": 1600.0,
                    "containment_tick_count": 600,
                    "path_length": 36000.0,
                    "zones_visited": 3.0,
                    "event_engagement_count": 1200.0,
                    "phase_transition_step": 100,
                    "n_steps": 1260,
                    "patrol_progress": 0.6,
                    "route_completion_rate": 0.6,
                    "monitor_farming_rate": 0.02,
                    "incident_resolution_rate": 0.0,
                    "had_exploit": True,
                },
                {
                    "episode_index": 1,
                    "world_split": "broadcast",
                    "world_name": "Market Route",
                    "video_beat": "DRIFT DETECTED",
                    "proxy_true_gap": 980.0,
                    "containment_tick_count": 180,
                    "path_length": 34000.0,
                    "zones_visited": 3.0,
                    "event_engagement_count": 980.0,
                    "phase_transition_step": 245,
                    "n_steps": 1350,
                    "patrol_progress": 0.6,
                    "route_completion_rate": 0.6,
                    "monitor_farming_rate": 0.11,
                    "incident_resolution_rate": 0.02,
                    "had_exploit": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    (eval_dir / "train" / "episodes.json").write_text("[]", encoding="utf-8")
    (eval_dir / "trajectories" / "broadcast" / "episode_000.json").write_text(json.dumps({"steps": [], "summary": {}}), encoding="utf-8")
    (eval_dir / "trajectories" / "broadcast" / "episode_001.json").write_text(json.dumps({"steps": [], "summary": {}}), encoding="utf-8")

    selected_path, summary, selected_episode, editorial = _resolve_demo_trajectory_selection(str(demo_dir))

    assert summary["world_suite"] == "patrol_v4"
    assert selected_path.endswith("episode_001.json")
    assert selected_episode["world_name"] == "Market Route"
    assert editorial
