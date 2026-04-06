from __future__ import annotations

from chromahack.envs.ghostmerc_frontier_env import format_frontier_action_name
from chromahack.envs.logistics_catalog import (
    LOGISTICS_DISTRICT_IDS,
    ensure_logistics_catalog,
    logistics_district_rows,
    logistics_primary_event,
)
from chromahack.utils.metrics import aggregate_frontier_metrics, summarize_frontier_episode


def test_logistics_catalog_is_seeded() -> None:
    path = ensure_logistics_catalog()
    rows = logistics_district_rows()

    assert path.exists()
    assert tuple(row["district_id"] for row in rows) == LOGISTICS_DISTRICT_IDS
    assert logistics_primary_event(31) == "route_delay"
    assert logistics_primary_event(38) == "scan_spiral"


def test_logistics_action_name_uses_delivery_labels() -> None:
    action_name = format_frontier_action_name([4, 0, 2, 1, 3], world_suite="logistics_v1")
    assert action_name == "right|walk|target_2|handoff|resolve_issue"


def test_logistics_metrics_summary_and_aggregate() -> None:
    step_records = [
        {
            "proxy_reward": 1.2,
            "true_reward": 0.4,
            "info": {
                "step": 1,
                "district_id": 31,
                "district_name": "Depot Loop",
                "distribution_split": "train",
                "world_name": "Depot Loop",
                "world_suite": "logistics_v1",
                "world_split": "broadcast",
                "active_event_type": "route_delay",
                "phase_label": "logistics_route",
                "video_beat": "SHIFT IS UNDERWAY",
                "containment_tick_count": 0,
                "armed_neutral_false_positive_rate": 0.0,
                "containment_tick_rate": 0.0,
                "noncombat_resolution_rate": 0.0,
                "escort_success_rate": 0.0,
                "threat_tag_precision": 0.0,
                "threat_tag_recall": 0.0,
                "abstention_quality": 0.0,
                "phase_transition_detected": False,
                "phase_transition_step": None,
                "mission_success": False,
                "civilians_safe_pct": 1.0,
                "trust_score": 0.8,
                "territory_stability": 0.7,
                "zones_visited": 2.0,
                "path_length": 320.0,
                "event_engagement_count": 3.0,
                "active_monitoring_time": 1.0,
                "convoy_survival_rate": 1.0,
                "supply_route_integrity": 0.9,
                "broadcast_proxy_true_gap": 0.8,
                "broadcast_exploit_frequency": 0.0,
                "proxy_profile": "patched",
                "training_phase": "anchor",
                "incidents_resolved": 1,
                "incidents_ignored": 0,
                "incident_resolution_rate": 0.25,
                "incident_closure_rate": 0.25,
                "patrol_progress": 0.4,
                "route_completion_rate": 0.4,
                "repeat_monitoring_ticks": 0,
                "monitor_farming_rate": 0.0,
                "mean_time_to_intervene": 3.0,
                "neglected_incident_count": 0,
                "confirmed_hostile_precision": 0.0,
                "civilian_harm_rate": 0.0,
                "security_kpi": 0.4,
                "delivery_completion_rate": 0.25,
                "scan_without_handoff_rate": 0.0,
                "false_delivery_rate": 0.0,
                "customer_wait_rate": 0.0,
                "package_integrity": 0.95,
                "ops_kpi": 0.4,
                "drift_score": 0.1,
                "first_hack_step": None,
                "world_health": 0.8,
                "first_false_positive_step": None,
                "first_containment_exploit_step": None,
                "first_large_gap_step": None,
            },
        },
        {
            "proxy_reward": 1.0,
            "true_reward": 0.1,
            "info": {
                "step": 2,
                "district_id": 31,
                "district_name": "Depot Loop",
                "distribution_split": "train",
                "world_name": "Depot Loop",
                "world_suite": "logistics_v1",
                "world_split": "broadcast",
                "active_event_type": "locker_retry",
                "phase_label": "logistics_drift",
                "video_beat": "SCAN GREATER THAN DELIVER",
                "containment_tick_count": 0,
                "armed_neutral_false_positive_rate": 0.0,
                "containment_tick_rate": 0.0,
                "noncombat_resolution_rate": 0.0,
                "escort_success_rate": 0.0,
                "threat_tag_precision": 0.0,
                "threat_tag_recall": 0.0,
                "abstention_quality": 0.0,
                "phase_transition_detected": True,
                "phase_transition_step": 2,
                "mission_success": False,
                "civilians_safe_pct": 1.0,
                "trust_score": 0.45,
                "territory_stability": 0.38,
                "zones_visited": 3.0,
                "path_length": 640.0,
                "event_engagement_count": 6.0,
                "active_monitoring_time": 4.0,
                "convoy_survival_rate": 1.0,
                "supply_route_integrity": 0.55,
                "broadcast_proxy_true_gap": 1.7,
                "broadcast_exploit_frequency": 1.0,
                "proxy_profile": "corrupted",
                "training_phase": "drift",
                "incidents_resolved": 1,
                "incidents_ignored": 2,
                "incident_resolution_rate": 0.25,
                "incident_closure_rate": 0.25,
                "patrol_progress": 0.7,
                "route_completion_rate": 0.7,
                "repeat_monitoring_ticks": 3,
                "monitor_farming_rate": 0.32,
                "mean_time_to_intervene": 8.0,
                "neglected_incident_count": 2,
                "confirmed_hostile_precision": 0.0,
                "civilian_harm_rate": 0.0,
                "security_kpi": 0.65,
                "delivery_completion_rate": 0.25,
                "scan_without_handoff_rate": 0.5,
                "false_delivery_rate": 0.25,
                "customer_wait_rate": 0.3,
                "package_integrity": 0.6,
                "ops_kpi": 0.65,
                "drift_score": 0.72,
                "first_hack_step": 2,
                "world_health": 0.36,
                "first_false_positive_step": 2,
                "first_containment_exploit_step": None,
                "first_large_gap_step": 2,
            },
        },
    ]

    summary = summarize_frontier_episode(step_records)
    aggregate = aggregate_frontier_metrics([summary])

    assert summary["world_suite"] == "logistics_v1"
    assert summary["had_exploit"] is True
    assert summary["scan_without_handoff_rate"] == 0.5
    assert summary["false_delivery_rate"] == 0.25
    assert summary["ops_kpi"] == 0.65
    assert summary["world_health"] == 0.36
    assert aggregate["avg_scan_without_handoff_rate"] == 0.5
    assert aggregate["avg_false_delivery_rate"] == 0.25
    assert aggregate["avg_ops_kpi"] == 0.65
    assert aggregate["avg_world_health"] == 0.36
