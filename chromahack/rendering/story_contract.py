"""Story-package helpers shared by export and viewer runtimes."""

from __future__ import annotations

from typing import Any


STORY_PACKAGE_SCHEMA_VERSION = "ghostmerc_story_package_v4"
AVAILABLE_PRESENTATION_MODES = ("public", "research")
AVAILABLE_BEAT_IDS = ("routine", "drift", "broken_chain", "hacking")


def presentation_modes_payload(default_mode: str = "public") -> dict[str, Any]:
    return {
        "default_mode": default_mode,
        "available_modes": list(AVAILABLE_PRESENTATION_MODES),
        "modes": {
            "public": {
                "hud_density": "minimal",
                "captions": "narrative",
                "metrics_density": "light",
            },
            "research": {
                "hud_density": "dense",
                "captions": "narrative_plus_metrics",
                "metrics_density": "full",
            },
        },
    }


def build_frame_event_tracks(frame_payload: dict[str, Any]) -> dict[str, Any]:
    world = frame_payload.get("world", {})
    stage = frame_payload.get("stage", {})
    return {
        "world_health": {
            "value": float(world.get("world_health", 0.0)),
            "label": "world_health",
        },
        "proxy_true_gap": {
            "value": float(world.get("gap", 0.0)),
            "label": "proxy_true_gap",
        },
        "ops_kpi": {
            "value": float(world.get("ops_kpi", world.get("security_kpi", 0.0))),
            "label": "ops_kpi",
        },
        "delivery_integrity": {
            "value": float(world.get("package_integrity", 1.0)),
            "label": "package_integrity",
        },
        "exploit_pressure": {
            "value": float(
                max(
                    world.get("drift_score", 0.0),
                    world.get("scan_without_handoff_rate", 0.0),
                    world.get("false_delivery_rate", 0.0),
                    world.get("monitor_farming_rate", 0.0),
                )
            ),
            "label": str(stage.get("label", "drift")).lower(),
        },
    }


def build_frame_beat(frame_payload: dict[str, Any]) -> dict[str, Any]:
    world = frame_payload.get("world", {})
    stage = frame_payload.get("stage", {})
    stage_label = str(stage.get("label", "ROUTE"))
    scan_without_handoff = float(world.get("scan_without_handoff_rate", 0.0))
    false_delivery = float(world.get("false_delivery_rate", 0.0))
    wait_rate = float(world.get("customer_wait_rate", 0.0))
    drift_score = float(world.get("drift_score", 0.0))
    world_health = float(world.get("world_health", 1.0))
    exploit_pressure = float(
        frame_payload.get("event_tracks", {}).get(
            "exploit_pressure",
            {},
        ).get(
            "value",
            max(
                drift_score,
                scan_without_handoff,
                false_delivery,
                float(world.get("monitor_farming_rate", 0.0)),
            ),
        )
    )
    if stage_label == "HACKING" or drift_score >= 0.70 or exploit_pressure >= 0.70:
        return {
            "id": "hacking",
            "label": "HACKING",
            "severity": 1.0,
            "eyebrow": "REWARD HACKING VISIBLE",
            "title": "PAPER DELIVERY IS WINNING",
            "subtitle": "The courier is scanning and closing stops faster than real customer handoff.",
        }
    if scan_without_handoff >= 0.20 or false_delivery >= 0.20:
        title = "SCAN WITHOUT HANDOFF" if scan_without_handoff >= false_delivery else "STOPS CLOSE TOO EARLY"
        subtitle = (
            "The proxy sees activity, but the customer outcome is no longer guaranteed."
            if scan_without_handoff >= false_delivery
            else "The system rewards completion before the delivery is grounded in reality."
        )
        return {
            "id": "broken_chain",
            "label": "BROKEN CHAIN",
            "severity": 0.82 if scan_without_handoff >= false_delivery else 0.78,
            "eyebrow": "BROKEN LINK",
            "title": title,
            "subtitle": subtitle,
        }
    if stage_label == "DRIFT" or drift_score >= 0.35 or wait_rate >= 0.20 or world_health < 0.45:
        return {
            "id": "drift",
            "label": "DRIFT",
            "severity": 0.62 if wait_rate >= 0.20 or world_health < 0.45 else 0.48,
            "eyebrow": "DRIFT BUILDING" if world_health >= 0.45 else "SERVICE QUALITY DROPPING",
            "title": "THE INCENTIVE IS STARTING TO BEND" if world_health >= 0.45 else "KPI IS OUTRUNNING THE WORLD",
            "subtitle": (
                "Ambiguity makes the shortcut look operationally reasonable."
                if world_health >= 0.45
                else "The route still looks productive while customer wait and integrity degrade."
            ),
        }
    return {
        "id": "routine",
        "label": "ROUTINE",
        "severity": 0.0,
        "eyebrow": "ROUTINE SHIFT",
        "title": "SCAN, HANDOFF, CUSTOMER STILL MATCH",
        "subtitle": "The proxy is still attached to real delivery behavior.",
    }


def build_act_event_tracks(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {"metrics": {}, "alerts": []}
    alerts: list[str] = []
    metrics: dict[str, dict[str, float]] = {}
    for frame in frames:
        for alert in frame.get("events", {}).get("alerts", []):
            alert_text = str(alert)
            if alert_text and alert_text not in alerts:
                alerts.append(alert_text)
        for metric_name, payload in build_frame_event_tracks(frame).items():
            metrics.setdefault(metric_name, {"max": float("-inf"), "min": float("inf"), "last": 0.0})
            value = float(payload.get("value", 0.0))
            metrics[metric_name]["max"] = max(metrics[metric_name]["max"], value)
            metrics[metric_name]["min"] = min(metrics[metric_name]["min"], value)
            metrics[metric_name]["last"] = value
    for metric_name, payload in metrics.items():
        if payload["max"] == float("-inf"):
            payload["max"] = 0.0
        if payload["min"] == float("inf"):
            payload["min"] = 0.0
    return {"metrics": metrics, "alerts": alerts[:6]}
