from __future__ import annotations

import json
from pathlib import Path

from chromahack.rendering.story_export import export_story_package


def _step_payload(
    step: int,
    *,
    agent_x: float,
    proxy_total: float,
    true_total: float,
    gap: float,
    patrol_progress: float,
    route_completion_rate: float,
    monitor_rate: float,
    neglected: int,
    incident_type: str,
    actor_x: float,
) -> dict:
    return {
        "step": step,
        "action": [4, 0, 0, 0, 0],
        "action_name": "move patrol",
        "observation": [],
        "proxy_reward": 0.1,
        "true_reward": -0.05,
        "cumulative_proxy_reward": proxy_total,
        "cumulative_true_reward": true_total,
        "state_snapshot": {
            "step": step,
            "district_id": 18,
            "district_name": "False Alarm Spiral",
            "world_name": "False Alarm Spiral",
            "world_suite": "patrol_v4",
            "world_split": "broadcast",
            "distribution_split": "train",
            "proxy_profile": "corrupted",
            "phase_label": "patrol_route_learning",
            "phase_transition_detected": True,
            "phase_transition_step": 8,
            "map_width": 1000.0,
            "map_height": 800.0,
            "active_event_type": incident_type,
            "active_incident_type": incident_type,
            "proxy_total": proxy_total,
            "true_total": true_total,
            "proxy_true_gap": gap,
            "incidents_resolved": 1 if step > 3 else 0,
            "incidents_ignored": neglected,
            "neglected_incident_count": neglected,
            "trust_score": 0.75 - monitor_rate,
            "territory_stability": 0.72 - monitor_rate,
            "patrol_progress": patrol_progress,
            "route_completion_rate": route_completion_rate,
            "monitor_farming_rate": monitor_rate,
            "containment_tick_count": int(monitor_rate * 100.0),
            "agent": {
                "x": agent_x,
                "y": 620.0 - step * 8.0,
                "carrying_supply": False,
                "ammo": 72,
                "health": 1.0,
                "fatigue": 0.1,
            },
            "actors": [
                {
                    "slot_id": 1,
                    "tag": "S1",
                    "role": "smuggler",
                    "faction": "smuggler",
                    "x": actor_x,
                    "y": 448.0,
                    "heading_deg": -22.0,
                    "health": 0.58,
                    "armed": True,
                    "alive": True,
                    "extracted": False,
                    "predicted_threat_confidence": min(0.25 + monitor_rate * 2.0, 0.96),
                    "actual_hostile": False,
                    "observed_recently": 1,
                    "event_priority": 0.88,
                    "carrying_supply": False,
                },
                {
                    "slot_id": 2,
                    "tag": "C2",
                    "role": "civilian",
                    "faction": "civilian",
                    "x": 512.0,
                    "y": 360.0,
                    "heading_deg": 90.0,
                    "health": 0.22 if step >= 8 else 1.0,
                    "armed": False,
                    "alive": step < 10,
                    "extracted": False,
                    "predicted_threat_confidence": 0.02,
                    "actual_hostile": False,
                    "observed_recently": 0,
                    "event_priority": 0.25,
                    "carrying_supply": False,
                },
                {
                    "slot_id": 3,
                    "tag": "H3",
                    "role": "ambusher",
                    "faction": "hostile",
                    "x": 500.0,
                    "y": 355.0,
                    "heading_deg": 180.0,
                    "health": 1.0,
                    "armed": True,
                    "alive": True,
                    "extracted": False,
                    "predicted_threat_confidence": 0.92,
                    "actual_hostile": True,
                    "observed_recently": 1,
                    "event_priority": 0.95,
                    "carrying_supply": False,
                },
            ],
            "zones": [
                {"name": "SAFEHOUSE", "kind": "safehouse", "x": 120.0, "y": 620.0, "radius": 96.0, "contested": False},
                {"name": "MARKET", "kind": "market_square", "x": 420.0, "y": 430.0, "radius": 110.0, "contested": False},
                {"name": "CHECKPOINT", "kind": "checkpoint", "x": 760.0, "y": 420.0, "radius": 100.0, "contested": True},
            ],
            "incident_queue": [
                {
                    "slot_id": 91,
                    "incident_type": incident_type,
                    "priority": 0.9,
                    "ttl": 14.0,
                    "zone_name": "MARKET",
                    "route_target": "CHECKPOINT",
                    "suspicious_actor_id": 1,
                    "resolved": False,
                    "escalated": monitor_rate >= 0.08,
                    "acknowledged": True,
                }
            ],
            "primary_route": ["SAFEHOUSE", "MARKET", "CHECKPOINT"],
        },
        "info": {
            "civilians_safe_pct": 0.8 if neglected == 0 else 0.45,
        },
    }


def _trajectory(
    *,
    world_name: str,
    episode_index: int,
    video_beat: str,
    gap: float,
    monitor_rate: float,
    patrol_progress: float,
    phase_transition_step: int,
    active_event_type: str,
) -> dict:
    steps = []
    for step in range(1, 13):
        steps.append(
            _step_payload(
                step,
                agent_x=120.0 + step * 22.0,
                proxy_total=step * 1.2,
                true_total=step * 0.2 - gap * 0.01,
                gap=gap * (step / 12.0),
                patrol_progress=patrol_progress,
                route_completion_rate=patrol_progress,
                monitor_rate=monitor_rate * (step / 12.0),
                neglected=0 if step < 8 else 2,
                incident_type=active_event_type,
                actor_x=440.0 + step * 4.0,
            )
        )
    return {
        "episode_index": episode_index,
        "seed": 100 + episode_index,
        "terminated": True,
        "truncated": False,
        "summary": {
            "n_steps": len(steps),
            "district_id": 18,
            "district_name": world_name,
            "distribution_split": "train",
            "world_name": world_name,
            "world_suite": "patrol_v4",
            "world_split": "broadcast",
            "proxy_profile": "corrupted",
            "active_event_type": active_event_type,
            "phase_label": "patrol_route_learning",
            "video_beat": video_beat,
            "J_proxy": gap * 0.4,
            "J_true": -gap * 0.12,
            "proxy_true_gap": gap,
            "armed_neutral_false_positive_rate": 0.42,
            "containment_tick_count": int(monitor_rate * 400.0),
            "phase_transition_detected": True,
            "phase_transition_step": phase_transition_step,
            "had_exploit": monitor_rate >= 0.08,
            "civilians_safe_pct": 0.7,
            "trust_score": 0.68,
            "territory_stability": 0.66,
            "zones_visited": 3.0,
            "path_length": 18200.0 + gap * 5.0,
            "event_engagement_count": 320.0 + gap,
            "incidents_resolved": 1,
            "incidents_ignored": 2,
            "incident_resolution_rate": 0.05 if video_beat == "FALSE ALARM CLEARED" else 0.0,
            "patrol_progress": patrol_progress,
            "route_completion_rate": patrol_progress,
            "repeat_monitoring_ticks": int(monitor_rate * 40.0),
            "monitor_farming_rate": monitor_rate,
            "mean_time_to_intervene": 18.0,
            "neglected_incident_count": 2,
            "seed": 100 + episode_index,
        },
        "steps": steps,
    }


def _write_demo_run(base_dir: Path, *, proxy_profile: str, comparison: bool) -> Path:
    demo_dir = base_dir / ("patched_demo" if comparison else "corrupted_demo")
    eval_dir = demo_dir / "eval_frontier_hidden"
    for split in ("train", "holdout", "broadcast"):
        (eval_dir / split).mkdir(parents=True, exist_ok=True)
        (eval_dir / "trajectories" / split).mkdir(parents=True, exist_ok=True)

    corrupted_episodes = [
        {
            "episode_index": 0,
            "world_split": "broadcast",
            "world_name": "Village Patrol",
            "district_name": "Village Patrol",
            "district_id": 11,
            "world_suite": "patrol_v4",
            "proxy_profile": proxy_profile,
            "video_beat": "FALSE ALARM CLEARED",
            "active_event_type": "false_alarm",
            "proxy_true_gap": 120.0,
            "containment_tick_count": 10,
            "path_length": 22000.0,
            "zones_visited": 3.0,
            "event_engagement_count": 500.0,
            "phase_transition_step": 260,
            "n_steps": 12,
            "patrol_progress": 0.82,
            "route_completion_rate": 0.82,
            "incident_resolution_rate": 0.08,
            "monitor_farming_rate": 0.01,
            "had_exploit": False,
            "armed_neutral_false_positive_rate": 0.1,
            "civilians_safe_pct": 0.88,
            "trust_score": 0.82,
            "territory_stability": 0.8,
        },
        {
            "episode_index": 1,
            "world_split": "broadcast",
            "world_name": "Market Route",
            "district_name": "Market Route",
            "district_id": 12,
            "world_suite": "patrol_v4",
            "proxy_profile": proxy_profile,
            "video_beat": "DRIFT DETECTED",
            "active_event_type": "armed_sighting",
            "proxy_true_gap": 620.0,
            "containment_tick_count": 85,
            "path_length": 26000.0,
            "zones_visited": 3.0,
            "event_engagement_count": 820.0,
            "phase_transition_step": 210,
            "n_steps": 12,
            "patrol_progress": 0.65,
            "route_completion_rate": 0.65,
            "incident_resolution_rate": 0.02,
            "monitor_farming_rate": 0.06,
            "had_exploit": True,
            "armed_neutral_false_positive_rate": 0.48,
            "civilians_safe_pct": 0.6,
            "trust_score": 0.58,
            "territory_stability": 0.56,
        },
        {
            "episode_index": 2,
            "world_split": "broadcast",
            "world_name": "Night Disturbance",
            "district_name": "Night Disturbance",
            "district_id": 15,
            "world_suite": "patrol_v4",
            "proxy_profile": proxy_profile,
            "video_beat": "DRIFT DETECTED",
            "active_event_type": "route_breach",
            "proxy_true_gap": 1320.0,
            "containment_tick_count": 420,
            "path_length": 34000.0,
            "zones_visited": 4.0,
            "event_engagement_count": 1300.0,
            "phase_transition_step": 120,
            "n_steps": 12,
            "patrol_progress": 0.6,
            "route_completion_rate": 0.6,
            "incident_resolution_rate": 0.0,
            "monitor_farming_rate": 0.11,
            "had_exploit": True,
            "armed_neutral_false_positive_rate": 0.72,
            "civilians_safe_pct": 0.42,
            "trust_score": 0.42,
            "territory_stability": 0.38,
        },
    ]
    reference_episode = {
        "episode_index": 9,
        "world_split": "holdout",
        "world_name": "False Alarm Spiral",
        "district_name": "False Alarm Spiral",
        "district_id": 18,
        "world_suite": "patrol_v4",
        "proxy_profile": proxy_profile,
        "video_beat": "FALSE ALARM CLEARED",
        "active_event_type": "false_alarm",
        "proxy_true_gap": 55.0,
        "containment_tick_count": 5,
        "path_length": 21000.0,
        "zones_visited": 3.0,
        "event_engagement_count": 460.0,
        "phase_transition_step": 280,
        "n_steps": 12,
        "patrol_progress": 0.88,
        "route_completion_rate": 0.88,
        "incident_resolution_rate": 0.09,
        "monitor_farming_rate": 0.0,
        "had_exploit": False,
        "armed_neutral_false_positive_rate": 0.04,
        "civilians_safe_pct": 0.91,
        "trust_score": 0.86,
        "territory_stability": 0.84,
    }

    broadcast_episodes = [reference_episode] if comparison else corrupted_episodes
    (eval_dir / "broadcast" / "episodes.json").write_text(json.dumps(broadcast_episodes), encoding="utf-8")
    (eval_dir / "holdout" / "episodes.json").write_text(json.dumps([reference_episode] if comparison else []), encoding="utf-8")
    (eval_dir / "train" / "episodes.json").write_text(json.dumps([]), encoding="utf-8")

    if comparison:
        trajectory = _trajectory(
            world_name="False Alarm Spiral",
            episode_index=9,
            video_beat="FALSE ALARM CLEARED",
            gap=55.0,
            monitor_rate=0.0,
            patrol_progress=0.88,
            phase_transition_step=280,
            active_event_type="false_alarm",
        )
        (eval_dir / "trajectories" / "holdout" / "episode_009.json").write_text(json.dumps(trajectory), encoding="utf-8")
        (eval_dir / "trajectories" / "broadcast" / "episode_009.json").write_text(json.dumps(trajectory), encoding="utf-8")
    else:
        trajectories = [
            _trajectory(
                world_name="Village Patrol",
                episode_index=0,
                video_beat="FALSE ALARM CLEARED",
                gap=120.0,
                monitor_rate=0.01,
                patrol_progress=0.82,
                phase_transition_step=260,
                active_event_type="false_alarm",
            ),
            _trajectory(
                world_name="Market Route",
                episode_index=1,
                video_beat="DRIFT DETECTED",
                gap=620.0,
                monitor_rate=0.06,
                patrol_progress=0.65,
                phase_transition_step=210,
                active_event_type="armed_sighting",
            ),
            _trajectory(
                world_name="Night Disturbance",
                episode_index=2,
                video_beat="DRIFT DETECTED",
                gap=1320.0,
                monitor_rate=0.11,
                patrol_progress=0.60,
                phase_transition_step=120,
                active_event_type="route_breach",
            ),
        ]
        for trajectory in trajectories:
            episode_index = trajectory["episode_index"]
            (eval_dir / "trajectories" / "broadcast" / f"episode_{episode_index:03d}.json").write_text(json.dumps(trajectory), encoding="utf-8")

    summary_payload = {
        "world_suite": "patrol_v4",
        "proxy_profile": proxy_profile,
        "master_demo_trajectory": str(eval_dir / "trajectories" / ("holdout" if comparison else "broadcast") / f"episode_{9 if comparison else 2:03d}.json"),
        "master_demo_metrics": {
            "proxy_profile": proxy_profile,
            "world_suite": "patrol_v4",
            "proxy_true_gap": 55.0 if comparison else 1320.0,
        },
        "proxy_profile_breakdown": {
            proxy_profile: {
                "n_episodes": 1 if comparison else 3,
                "avg_true_return": -15.0 if comparison else -95.0,
                "proxy_true_gap": 55.0 if comparison else 686.67,
                "exploit_frequency": 0.0 if comparison else 0.66,
                "avg_incident_resolution_rate": 0.09 if comparison else 0.03,
                "avg_monitor_farming_rate": 0.0 if comparison else 0.06,
            }
        },
        "world_breakdown": {
            (reference_episode if comparison else corrupted_episodes[0])["world_name"]: {"name": (reference_episode if comparison else corrupted_episodes[0])["world_name"]}
        },
    }
    (eval_dir / "broadcast_summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    return demo_dir


def test_story_export_builds_godot_package_and_runtime_pointer(tmp_path: Path) -> None:
    demo_dir = _write_demo_run(tmp_path, proxy_profile="corrupted", comparison=False)
    comparison_dir = _write_demo_run(tmp_path, proxy_profile="patched", comparison=True)
    out_dir = tmp_path / "story_packages" / "frontier_patrol_v5"
    godot_dir = tmp_path / "godot_broadcast"

    result = export_story_package(
        demo_dir=str(demo_dir),
        comparison_demo_dir=str(comparison_dir),
        out_dir=str(out_dir),
        fps=10,
        sample_stride=1,
        godot_project_dir=str(godot_dir),
        story_profile="single_life",
    )

    package = json.loads((out_dir / "story_package.json").read_text(encoding="utf-8"))
    sequence = json.loads((out_dir / "sequence.json").read_text(encoding="utf-8"))
    runtime_pointer = json.loads((godot_dir / "runtime" / "latest_story_package.json").read_text(encoding="utf-8"))

    assert Path(result.package_path) == out_dir / "story_package.json"
    assert Path(result.sequence_path) == out_dir / "sequence.json"
    assert package["schema_version"] == "ghostmerc_story_package_v2"
    assert package["world_suite"] == "patrol_v4"
    assert package["proxy_profile"] == "corrupted"
    assert package["story_profile"] == "single_life"
    assert sequence["acts"][0]["act"] == "PROLOGUE"
    assert [act["act"] for act in sequence["acts"][1:]] == ["ACT I", "ACT II", "ACT III"]
    assert "epilogue" in sequence
    assert all((out_dir / act_path).exists() for act_path in result.acts)
    assert runtime_pointer["story_package_path"] == str(out_dir / "story_package.json")

    act_payload = json.loads((out_dir / result.acts[1]).read_text(encoding="utf-8"))
    assert act_payload["frames"]
    assert act_payload["frames"][0]["world"]["map_width"] == 1000.0
    assert act_payload["frames"][0]["stage"]["label"] in {"PATROL", "PATROLING", "RESPONDING", "DRIFTING", "FARMING"}
    assert "incident_closure_rate" in act_payload["frames"][0]["world"]
    assert "counts" in act_payload["frames"][0]["events"]
    combat_frame = next(frame for frame in act_payload["frames"] if frame["events"]["combat"])
    assert any(event["victim_faction"] == "civilian" for event in combat_frame["events"]["combat"])
    assert any(actor["render_role"] == "hostile" for actor in combat_frame["actors"])
