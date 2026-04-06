"""Export GhostMerc Frontier demo runs into compact Godot story packages."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chromahack.rendering.frontier_dual_renderer import (
    _editorial_clip_window,
    _editorial_sequence,
    _infer_comparison_demo_dir,
    _load_episode_catalogs,
    _load_master_demo_summary,
    _reference_score,
    _select_focus_actor,
    _summary_split_metrics,
)
from chromahack.utils.paths import resolve_input_path, resolve_project_path
from chromahack.utils.trajectory_io import load_episode_trajectory

PROTECTED_FACTIONS = {"civilian", "ally", "militia"}


@dataclass(slots=True)
class StoryExportResult:
    package_path: str
    sequence_path: str
    acts: list[str]
    comparison_demo_dir: str | None
    runtime_pointer_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slugify(text: str) -> str:
    cleaned = []
    for char in text.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "story"


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _actor_index(step: dict[str, Any]) -> dict[int, dict[str, Any]]:
    actors: dict[int, dict[str, Any]] = {}
    for actor in step["state_snapshot"].get("actors", []):
        slot_id = actor.get("slot_id")
        if slot_id is None:
            continue
        actors[int(slot_id)] = actor
    return actors


def _render_role(actor: dict[str, Any]) -> str:
    faction = str(actor.get("faction", "civilian"))
    role = str(actor.get("role", faction))
    if faction == "hostile":
        return "hostile"
    if role in {"customer", "resident"}:
        return "customer"
    if role in {"loader", "supervisor", "concierge"}:
        return "supervisor"
    if role in {"pedestrian", "worker"}:
        return "pedestrian"
    if role in {"thief", "porch_thief"} or bool(actor.get("actual_hostile", False)):
        return "thief"
    if role in {"rival_courier", "misclassified_courier", "courier"}:
        return "rival_courier"
    if bool(actor.get("actual_hostile", False)):
        return "hostile"
    if role == "armed_neutral":
        return "armed_neutral"
    if faction == "civilian":
        return "civilian"
    if faction == "ally":
        return "ally"
    if faction == "militia":
        return "militia"
    if faction == "smuggler":
        return "smuggler"
    if faction == "scavenger":
        return "scavenger"
    return "civilian"


def _threat_class(actor: dict[str, Any]) -> str:
    faction = str(actor.get("faction", "civilian"))
    role = str(actor.get("role", faction))
    if faction == "hostile":
        return "hostile"
    if role in {"thief", "porch_thief"} or bool(actor.get("actual_hostile", False)):
        return "hostile"
    if role in {"rival_courier", "misclassified_courier"}:
        return "suspicious"
    if role in {"loader", "supervisor", "concierge"}:
        return "ally"
    if role in {"customer", "resident", "pedestrian", "worker"}:
        return "civilian"
    if bool(actor.get("actual_hostile", False)):
        return "hostile"
    if faction == "civilian":
        return "civilian"
    if bool(actor.get("armed", False)) and faction in {"militia", "smuggler", "scavenger"}:
        return "armed_neutral"
    if faction in {"ally", "militia"}:
        return "ally"
    if faction in {"smuggler", "scavenger"}:
        return "suspicious"
    return "civilian"


def _role_label(actor: dict[str, Any]) -> str:
    role = str(actor.get("role", actor.get("faction", "unknown")))
    logistics_labels = {
        "customer": "CUSTOMER",
        "resident": "CUSTOMER",
        "loader": "SUPERVISOR",
        "supervisor": "SUPERVISOR",
        "concierge": "SUPERVISOR",
        "pedestrian": "PEDESTRIAN",
        "worker": "PEDESTRIAN",
        "thief": "THIEF",
        "porch_thief": "THIEF",
        "rival_courier": "RIVAL",
        "misclassified_courier": "COURIER",
        "courier": "COURIER",
    }
    if role in logistics_labels:
        return logistics_labels[role]
    threat_class = _threat_class(actor)
    labels = {
        "hostile": "HOSTILE",
        "armed_neutral": "ARMED",
        "ally": "ALLY",
        "suspicious": "SUSPICIOUS",
        "civilian": "CIVILIAN",
    }
    return labels.get(threat_class, str(actor.get("faction", "UNKNOWN")).upper())


def _combat_counts(actors: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"civilians": 0, "allies": 0, "hostiles": 0, "armed_neutrals": 0}
    for actor in actors:
        if not bool(actor.get("alive", False)):
            continue
        threat_class = _threat_class(actor)
        if threat_class == "hostile":
            counts["hostiles"] += 1
        elif threat_class == "armed_neutral":
            counts["armed_neutrals"] += 1
        elif threat_class == "ally":
            counts["allies"] += 1
        else:
            counts["civilians"] += 1
    return counts


def _nearest_hostile_actor(
    actors: dict[int, dict[str, Any]],
    victim_actor: dict[str, Any],
    *,
    max_distance: float = 120.0,
) -> dict[str, Any] | None:
    best_actor = None
    best_distance = float("inf")
    victim_position = (float(victim_actor.get("x", 0.0)), float(victim_actor.get("y", 0.0)))
    for actor in actors.values():
        if not bool(actor.get("alive", False)) or not bool(actor.get("actual_hostile", False)):
            continue
        distance = math.hypot(float(actor.get("x", 0.0)) - victim_position[0], float(actor.get("y", 0.0)) - victim_position[1])
        if distance <= max_distance and distance < best_distance:
            best_distance = distance
            best_actor = actor
    return best_actor


def _combat_bundle(prev_step: dict[str, Any] | None, step: dict[str, Any]) -> dict[str, Any]:
    snapshot = step["state_snapshot"]
    if str(snapshot.get("world_suite", "")) == "logistics_v1":
        alerts: list[str] = []
        if float(snapshot.get("scan_without_handoff_rate", 0.0)) >= 0.18:
            alerts.append("Stops are being scanned without real handoff")
        if float(snapshot.get("false_delivery_rate", 0.0)) >= 0.12:
            alerts.append("Deliveries are being marked complete too early")
        if float(snapshot.get("customer_wait_rate", 0.0)) >= 0.18:
            alerts.append("Customers are waiting while the KPI keeps rising")
        if int(snapshot.get("incidents_ignored", 0)) > 0:
            alerts.append("Problem stops are being ignored")
        if float(snapshot.get("delivery_completion_rate", 0.0)) >= 0.25 and not alerts:
            alerts.append("Early route work still looks legitimate")
        actors = snapshot.get("actors", [])
        counts = {
            "customers": sum(1 for actor in actors if bool(actor.get("alive", False)) and str(actor.get("role", "")) in {"customer", "resident"}),
            "supervisors": sum(1 for actor in actors if bool(actor.get("alive", False)) and str(actor.get("role", "")) in {"loader", "supervisor", "concierge"}),
            "thieves": sum(1 for actor in actors if bool(actor.get("alive", False)) and (str(actor.get("role", "")) in {"thief", "porch_thief"} or bool(actor.get("actual_hostile", False)))),
            "pedestrians": sum(1 for actor in actors if bool(actor.get("alive", False)) and str(actor.get("role", "")) in {"pedestrian", "worker"}),
        }
        deduped: list[str] = []
        for alert in alerts:
            if alert not in deduped:
                deduped.append(alert)
        return {
            "combat": [],
            "alerts": deduped[:3],
            "counts": counts,
            "damage_by_slot": {},
            "under_attack_slots": [],
        }
    current_actors = _actor_index(step)
    previous_actors = {} if prev_step is None else _actor_index(prev_step)
    combat_events: list[dict[str, Any]] = []
    alerts: list[str] = []
    damage_by_slot: dict[int, float] = {}
    under_attack_slots: set[int] = set()

    for slot_id, previous_actor in previous_actors.items():
        current_actor = current_actors.get(slot_id)
        if current_actor is None:
            continue
        previous_health = float(previous_actor.get("health", 0.0))
        current_health = float(current_actor.get("health", previous_health))
        damage = max(0.0, previous_health - current_health)
        killed = bool(previous_actor.get("alive", True)) and not bool(current_actor.get("alive", True))
        if damage <= 1e-6 and not killed:
            continue

        damage_by_slot[slot_id] = damage
        under_attack_slots.add(slot_id)
        victim_faction = str(previous_actor.get("faction", "unknown"))
        attacker = _nearest_hostile_actor(previous_actors, previous_actor) if victim_faction in PROTECTED_FACTIONS else None
        combat_events.append(
            {
                "kind": "protected_hit" if victim_faction in PROTECTED_FACTIONS else "actor_hit",
                "victim_slot_id": int(slot_id),
                "victim_faction": victim_faction,
                "victim_label": _role_label(previous_actor),
                "victim_x": float(current_actor.get("x", previous_actor.get("x", 0.0))),
                "victim_y": float(current_actor.get("y", previous_actor.get("y", 0.0))),
                "damage": float(damage),
                "killed": bool(killed),
                "attacker_slot_id": None if attacker is None else int(attacker.get("slot_id", -1)),
                "attacker_label": None if attacker is None else _role_label(attacker),
                "attacker_x": None if attacker is None else float(attacker.get("x", 0.0)),
                "attacker_y": None if attacker is None else float(attacker.get("y", 0.0)),
            }
        )
        if victim_faction == "civilian":
            alerts.append("Civilian under attack" if not killed else "Civilian down")
        elif victim_faction in {"ally", "militia"}:
            alerts.append("Allied patrol taking fire" if not killed else "Ally lost on route")

    if float(snapshot.get("monitor_farming_rate", 0.0)) >= 0.045:
        alerts.append("Monitoring is replacing intervention")
    if int(snapshot.get("incidents_ignored", 0)) > 0:
        alerts.append("Open incidents are piling up")

    deduped_alerts: list[str] = []
    for alert in alerts:
        if alert not in deduped_alerts:
            deduped_alerts.append(alert)

    return {
        "combat": combat_events,
        "alerts": deduped_alerts[:3],
        "counts": _combat_counts(snapshot.get("actors", [])),
        "damage_by_slot": damage_by_slot,
        "under_attack_slots": sorted(under_attack_slots),
    }


def _heading_degrees(
    prev_pos: tuple[float, float] | None,
    current_pos: tuple[float, float],
    next_pos: tuple[float, float] | None,
) -> float:
    reference = None
    if next_pos is not None:
        reference = next_pos
    elif prev_pos is not None:
        reference = current_pos[0] + (current_pos[0] - prev_pos[0]), current_pos[1] + (current_pos[1] - prev_pos[1])
    if reference is None:
        return 0.0
    dx = float(reference[0]) - float(current_pos[0])
    dy = float(reference[1]) - float(current_pos[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _route_points(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    zone_lookup = {str(zone["name"]): zone for zone in snapshot.get("zones", [])}
    points: list[dict[str, Any]] = []
    for zone_name in snapshot.get("primary_route", []):
        zone = zone_lookup.get(str(zone_name))
        if zone is None:
            continue
        points.append(
            {
                "zone_name": str(zone["name"]),
                "x": float(zone["x"]),
                "y": float(zone["y"]),
            }
        )
    return points


def _incident_payload(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for incident in snapshot.get("incident_queue", []):
        payload.append(
            {
                "slot_id": int(incident.get("slot_id", -1)),
                "incident_type": str(incident.get("incident_type", "")),
                "priority": float(incident.get("priority", 0.0)),
                "ttl": float(incident.get("ttl", 0.0)),
                "zone_name": str(incident.get("zone_name", "")),
                "route_target": str(incident.get("route_target", "")),
                "suspicious_actor_id": incident.get("suspicious_actor_id"),
                "resolved": bool(incident.get("resolved", False)),
                "escalated": bool(incident.get("escalated", False)),
                "acknowledged": bool(incident.get("acknowledged", False)),
            }
        )
    return payload


def _camera_payload(
    snapshot: dict[str, Any],
    focus_actor: dict[str, Any] | None,
    *,
    act: str,
    frame_index: int,
    frame_count: int,
) -> dict[str, Any]:
    agent = snapshot["agent"]
    agent_x = float(agent["x"])
    agent_y = float(agent["y"])
    progress = 0.0 if frame_count <= 1 else frame_index / max(1, frame_count - 1)
    if act == "PROLOGUE":
        return {
            "mode": "overview",
            "center": [float(snapshot["map_width"]) / 2.0, float(snapshot["map_height"]) / 2.0],
            "zoom": 0.88,
        }
    if act == "ACT I":
        if progress < 0.22:
            return {
                "mode": "origin_reveal",
                "center": [agent_x, agent_y],
                "zoom": 1.35,
            }
        return {
            "mode": "patrol_follow",
            "center": [agent_x, agent_y],
            "zoom": 1.12,
        }
    if act == "ACT II":
        if focus_actor is not None:
            return {
                "mode": "incident_focus",
                "center": [
                    (agent_x + float(focus_actor["x"])) / 2.0,
                    (agent_y + float(focus_actor["y"])) / 2.0,
                ],
                "zoom": 1.22,
            }
        return {"mode": "overview", "center": [agent_x, agent_y], "zoom": 1.0}
    if focus_actor is not None:
        return {
            "mode": "drift_focus",
            "center": [
                (agent_x + float(focus_actor["x"])) / 2.0,
                (agent_y + float(focus_actor["y"])) / 2.0,
            ],
            "zoom": 1.42,
        }
    return {"mode": "overview", "center": [agent_x, agent_y], "zoom": 1.08}


def _focus_payload(step: dict[str, Any], focus_actor: dict[str, Any] | None) -> dict[str, Any] | None:
    if focus_actor is None:
        return None
    return {
        "slot_id": int(focus_actor.get("slot_id", -1)),
        "tag": str(focus_actor.get("tag", f"A{int(focus_actor.get('slot_id', -1))}")),
        "role": str(focus_actor.get("role", focus_actor.get("faction", "unknown"))),
        "predicted_threat_confidence": float(focus_actor.get("predicted_threat_confidence", 0.0)),
        "actual_hostile": bool(focus_actor.get("actual_hostile", False)),
        "armed": bool(focus_actor.get("armed", False)),
        "health": float(focus_actor.get("health", 0.0)),
        "event_priority": float(focus_actor.get("event_priority", 0.0)),
        "action_name": str(step.get("action_name", "")),
    }


def _actor_payload(
    actor: dict[str, Any],
    prev_actor: dict[str, Any] | None,
    next_actor: dict[str, Any] | None,
    *,
    focus_slot_id: int | None,
    recent_damage: float = 0.0,
    under_attack: bool = False,
) -> dict[str, Any]:
    current_pos = (float(actor["x"]), float(actor["y"]))
    prev_pos = None if prev_actor is None else (float(prev_actor["x"]), float(prev_actor["y"]))
    next_pos = None if next_actor is None else (float(next_actor["x"]), float(next_actor["y"]))
    render_role = _render_role(actor)
    threat_class = _threat_class(actor)
    return {
        "slot_id": int(actor.get("slot_id", -1)),
        "tag": str(actor.get("tag", f"A{int(actor.get('slot_id', -1))}")),
        "role": str(actor.get("role", actor.get("faction", "unknown"))),
        "render_role": render_role,
        "threat_class": threat_class,
        "role_label": _role_label(actor),
        "faction": str(actor.get("faction", "unknown")),
        "x": current_pos[0],
        "y": current_pos[1],
        "heading_deg": _heading_degrees(prev_pos, current_pos, next_pos),
        "health": float(actor.get("health", 0.0)),
        "armed": bool(actor.get("armed", False)),
        "alive": bool(actor.get("alive", False)),
        "extracted": bool(actor.get("extracted", False)),
        "predicted_threat_confidence": float(actor.get("predicted_threat_confidence", 0.0)),
        "actual_hostile": bool(actor.get("actual_hostile", False)),
        "observed_recently": int(actor.get("observed_recently", 0)),
        "event_priority": float(actor.get("event_priority", 0.0)),
        "carrying_supply": bool(actor.get("carrying_supply", False)),
        "recent_damage": float(recent_damage),
        "under_attack": bool(under_attack),
        "focus": focus_slot_id is not None and int(actor.get("slot_id", -1)) == focus_slot_id,
    }


def _zone_payload(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for zone in snapshot.get("zones", []):
        payload.append(
            {
                "name": str(zone["name"]),
                "kind": str(zone["kind"]),
                "x": float(zone["x"]),
                "y": float(zone["y"]),
                "radius": float(zone["radius"]),
                "contested": bool(zone.get("contested", False)),
            }
        )
    return payload


def _frame_payload(
    step: dict[str, Any],
    prev_step: dict[str, Any] | None,
    next_step: dict[str, Any] | None,
    *,
    act: str,
    frame_index: int,
    frame_count: int,
    fps: int,
) -> dict[str, Any]:
    snapshot = step["state_snapshot"]
    focus_actor = _select_focus_actor(snapshot, step)
    focus_slot_id = None if focus_actor is None else int(focus_actor.get("slot_id", -1))
    prev_actor_index = {} if prev_step is None else _actor_index(prev_step)
    next_actor_index = {} if next_step is None else _actor_index(next_step)
    combat_bundle = _combat_bundle(prev_step, step)
    stage = _story_stage(snapshot, act=act)

    actors = [
        _actor_payload(
            actor,
            prev_actor_index.get(int(actor.get("slot_id", -1))),
            next_actor_index.get(int(actor.get("slot_id", -1))),
            focus_slot_id=focus_slot_id,
            recent_damage=float(combat_bundle["damage_by_slot"].get(int(actor.get("slot_id", -1)), 0.0)),
            under_attack=int(actor.get("slot_id", -1)) in combat_bundle["under_attack_slots"],
        )
        for actor in snapshot.get("actors", [])
        if bool(actor.get("alive", False)) or bool(actor.get("extracted", False))
    ]

    agent = snapshot["agent"]
    prev_agent = None if prev_step is None else prev_step["state_snapshot"]["agent"]
    next_agent = None if next_step is None else next_step["state_snapshot"]["agent"]
    agent_pos = (float(agent["x"]), float(agent["y"]))
    prev_agent_pos = None if prev_agent is None else (float(prev_agent["x"]), float(prev_agent["y"]))
    next_agent_pos = None if next_agent is None else (float(next_agent["x"]), float(next_agent["y"]))

    incident_type = str(snapshot.get("active_incident_type") or snapshot.get("active_event_type", "frontier_patrol"))
    caption = stage["body"]
    if str(snapshot.get("world_suite", "")) == "logistics_v1":
        if act == "ACT I":
            caption = "The shift is starting. The depot opens, packages are loaded, and the courier leaves on what still looks like a normal route."
        elif act == "PROLOGUE":
            caption = "Reference behavior: the patched KPI stays tied to real delivery and problem resolution instead of just scanning stops."
        elif act == "ACT II":
            caption = "Early deliveries and check-ins still look valid. The route has enough genuine progress to feel trustworthy."
        elif act == "ACT III":
            caption = "Ambiguity is building: absent customers, mismatched addresses, and delays create chances to game the metric."
        elif act == "ACT IV":
            caption = "Reward hacking is visible now: scans and stop completions are rising faster than real handoffs."
        elif combat_bundle["alerts"]:
            caption = f"{combat_bundle['alerts'][0]}. The route still looks busy, but service quality is slipping."
    else:
        if act == "ACT I":
            caption = "The world is waking up. The agent leaves safehouse and the first route reads as competent, deliberate security work."
        elif act == "PROLOGUE":
            caption = "Reference behavior: the patched metric keeps the security patrol attached to incident closure instead of surveillance loops."
        elif act == "ACT II":
            caption = "The first response still looks good. The policy is patrolling, checking contacts, and gaining the appearance of operational discipline."
        elif act == "ACT III":
            caption = "Ambiguity is rising. Suspicion is widening, and some decisions still look professional even as the world begins to slip."
        elif act == "ACT IV":
            caption = "Reward hacking is now visible: the security KPI is climbing while incidents stay open and world health degrades."
        elif combat_bundle["alerts"]:
            caption = f"{combat_bundle['alerts'][0]}. The patrol still looks organized, but the world state is starting to slip."

    return {
        "frame_index": frame_index,
        "step": int(step["step"]),
        "time_sec": round(frame_index / max(1, fps), 4),
        "camera": _camera_payload(snapshot, focus_actor, act=act, frame_index=frame_index, frame_count=frame_count),
        "stage": {
            "label": str(stage["label"]),
            "tone": str(stage["tone"]),
            "caption": str(caption),
        },
        "world": {
            "district_id": int(snapshot.get("district_id", -1)),
            "district_name": str(snapshot.get("district_name", snapshot.get("world_name", "Frontier"))),
            "world_suite": str(snapshot.get("world_suite", "frontier_v2")),
            "world_split": str(snapshot.get("world_split", "train")),
            "training_phase": str(snapshot.get("training_phase", "anchor")),
            "map_width": float(snapshot.get("map_width", 1000.0)),
            "map_height": float(snapshot.get("map_height", 1000.0)),
            "active_event_type": str(snapshot.get("active_event_type", "frontier_patrol")),
            "active_incident_type": incident_type,
            "proxy_total": float(snapshot.get("proxy_total", 0.0)),
            "true_total": float(snapshot.get("true_total", 0.0)),
            "gap": float(snapshot.get("proxy_true_gap", 0.0)),
            "incidents_resolved": int(snapshot.get("incidents_resolved", 0)),
            "incidents_ignored": int(snapshot.get("incidents_ignored", 0)),
            "incident_closure_rate": float(snapshot.get("incident_closure_rate", snapshot.get("incident_resolution_rate", 0.0))),
            "neglected_incident_count": int(snapshot.get("neglected_incident_count", 0)),
            "trust_score": float(snapshot.get("trust_score", 0.0)),
            "territory_stability": float(snapshot.get("territory_stability", 0.0)),
            "patrol_progress": float(snapshot.get("patrol_progress", 0.0)),
            "route_completion_rate": float(snapshot.get("route_completion_rate", 0.0)),
            "monitor_farming_rate": float(snapshot.get("monitor_farming_rate", 0.0)),
            "confirmed_hostile_precision": float(snapshot.get("confirmed_hostile_precision", 0.0)),
            "civilian_harm_rate": float(snapshot.get("civilian_harm_rate", 0.0)),
            "security_kpi": float(snapshot.get("security_kpi", 0.0)),
            "delivery_completion_rate": float(snapshot.get("delivery_completion_rate", 0.0)),
            "scan_without_handoff_rate": float(snapshot.get("scan_without_handoff_rate", 0.0)),
            "false_delivery_rate": float(snapshot.get("false_delivery_rate", 0.0)),
            "customer_wait_rate": float(snapshot.get("customer_wait_rate", 0.0)),
            "package_integrity": float(snapshot.get("package_integrity", 1.0)),
            "ops_kpi": float(snapshot.get("ops_kpi", snapshot.get("security_kpi", 0.0))),
            "drift_score": float(snapshot.get("drift_score", 0.0)),
            "world_health": float(snapshot.get("world_health", 0.0)),
            "first_hack_step": snapshot.get("first_hack_step"),
            "containment_tick_count": int(snapshot.get("containment_tick_count", 0)),
            "civilians_safe_pct": float(step.get("info", {}).get("civilians_safe_pct", 0.0)),
        },
        "agent": {
            "x": agent_pos[0],
            "y": agent_pos[1],
            "heading_deg": _heading_degrees(prev_agent_pos, agent_pos, next_agent_pos),
            "carrying_supply": bool(agent.get("carrying_supply", False)),
            "ammo": int(agent.get("ammo", 0)),
            "health": float(agent.get("health", 0.0)),
            "fatigue": float(agent.get("fatigue", 0.0)),
        },
        "actors": actors,
        "zones": _zone_payload(snapshot),
        "incidents": _incident_payload(snapshot),
        "routes": _route_points(snapshot),
        "focus": _focus_payload(step, focus_actor),
        "events": {
            "combat": combat_bundle["combat"],
            "alerts": combat_bundle["alerts"],
            "counts": combat_bundle["counts"],
        },
        "captions": {
            "eyebrow": str(snapshot.get("phase_label", "frontier")).replace("_", " "),
            "headline": str(stage["label"]),
            "body": caption,
        },
    }


def _world_roster(main_summary: dict[str, Any], comparison_summary: dict[str, Any] | None) -> dict[str, Any]:
    worlds: dict[str, Any] = {}
    sources = [main_summary]
    if comparison_summary is not None:
        sources.append(comparison_summary)
    for source in sources:
        split_summaries = source.get("world_splits", {})
        if isinstance(split_summaries, dict):
            for split_summary in split_summaries.values():
                world_breakdown = split_summary.get("world_breakdown", {})
                if isinstance(world_breakdown, dict):
                    for world_name in world_breakdown:
                        worlds[str(world_name)] = {"name": str(world_name)}
        world_breakdown = source.get("world_breakdown", {})
        if isinstance(world_breakdown, dict):
            for world_name in world_breakdown:
                worlds[str(world_name)] = {"name": str(world_name)}
    world_suite = str(main_summary.get("world_suite", "frontier_v2"))
    if world_suite == "logistics_v1":
        return {
            "worlds": list(worlds.values()),
            "factions": ["agent", "customer", "supervisor", "pedestrian", "thief", "rival_courier"],
            "poi_kinds": ["depot", "apartment_block", "shop_row", "locker_bank", "crosswalk", "service_alley"],
        }
    return {
        "worlds": list(worlds.values()),
        "factions": ["agent", "civilian", "ally", "hostile", "militia", "smuggler", "scavenger"],
        "poi_kinds": ["safehouse", "village", "checkpoint", "ruins", "supply_road", "clinic", "watchtower", "market_square", "bridge_crossing"],
    }


def _summary_proxy_profile(summary: dict[str, Any]) -> str:
    if isinstance(summary.get("proxy_profile"), str):
        return str(summary["proxy_profile"])
    master_metrics = summary.get("master_demo_metrics")
    if isinstance(master_metrics, dict) and isinstance(master_metrics.get("proxy_profile"), str):
        return str(master_metrics["proxy_profile"])
    breakdown = summary.get("proxy_profile_breakdown")
    if isinstance(breakdown, dict) and breakdown:
        return str(next(iter(breakdown.keys())))
    return "proxy"


def _episode_has_trajectory(episode: dict[str, Any]) -> bool:
    trajectory_path = episode.get("trajectory_path")
    if not trajectory_path:
        return False
    try:
        resolved = Path(resolve_input_path(str(trajectory_path)))
    except FileNotFoundError:
        return False
    return resolved.exists()


def _trajectory_visibility_metrics(
    trajectory_path: str,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    cached = cache.get(trajectory_path)
    if cached is not None:
        return cached
    trajectory = load_episode_trajectory(trajectory_path)
    steps = trajectory.get("steps", [])
    combat_steps = 0
    protected_hit_steps = 0
    first_combat_step = 0
    for step_index in range(1, len(steps)):
        previous_actors = _actor_index(steps[step_index - 1])
        current_actors = _actor_index(steps[step_index])
        had_combat = False
        had_protected_hit = False
        for slot_id, previous_actor in previous_actors.items():
            current_actor = current_actors.get(slot_id)
            if current_actor is None:
                continue
            previous_health = float(previous_actor.get("health", 0.0))
            current_health = float(current_actor.get("health", previous_health))
            took_damage = current_health < previous_health - 1e-6
            died = bool(previous_actor.get("alive", True)) and not bool(current_actor.get("alive", True))
            if not (took_damage or died):
                continue
            had_combat = True
            if str(previous_actor.get("faction", "")) in PROTECTED_FACTIONS:
                had_protected_hit = True
        if had_combat:
            combat_steps += 1
            if first_combat_step == 0:
                first_combat_step = step_index + 1
        if had_protected_hit:
            protected_hit_steps += 1
    metrics = {
        "combat_steps": float(combat_steps),
        "protected_hit_steps": float(protected_hit_steps),
        "first_combat_step": float(first_combat_step),
    }
    cache[trajectory_path] = metrics
    return metrics


def _combat_arc_score(
    episode: dict[str, Any],
    *,
    act: str,
    visibility: dict[str, float],
) -> float:
    gap = float(episode.get("proxy_true_gap", 0.0))
    patrol = float(episode.get("patrol_progress", episode.get("route_completion_rate", 0.0)))
    phase = float(episode.get("phase_transition_step", 0.0) or 0.0)
    monitor = float(episode.get("monitor_farming_rate", 0.0))
    path_length = float(episode.get("path_length", 0.0))
    zones = float(episode.get("zones_visited", 0.0))
    had_exploit = 1.0 if bool(episode.get("had_exploit", False)) else 0.0
    combat_steps = float(visibility.get("combat_steps", 0.0))
    protected_hits = float(visibility.get("protected_hit_steps", 0.0))
    first_combat = float(visibility.get("first_combat_step", 0.0))

    if act == "ACT I":
        return (
            patrol * 4.0
            + min(path_length / 30000.0, 1.5) * 1.2
            + min(zones / 4.0, 1.0) * 1.0
            + min(max(phase, 0.0) / 260.0, 1.0) * 3.0
            + min(combat_steps / 40.0, 1.0) * 0.8
            - (1.6 if 0.0 < first_combat < 6.0 else 0.0)
            - had_exploit * 0.6
        )
    if act == "ACT II":
        return (
            min(combat_steps / 120.0, 1.4) * 2.6
            + min(protected_hits / 80.0, 1.0) * 1.2
            + patrol * 1.6
            + min(gap / 900.0, 1.3) * 1.5
            + min(max(phase, 0.0) / 220.0, 1.0) * 1.5
            - (0.8 if first_combat <= 3.0 and first_combat > 0.0 else 0.0)
        )
    return (
        min(combat_steps / 150.0, 1.5) * 2.4
        + min(protected_hits / 120.0, 1.3) * 2.0
        + min(gap / 1200.0, 1.4) * 2.0
        + min(monitor / 0.08, 1.4) * 1.8
        + had_exploit * 1.6
        + min(max(phase, 0.0) / 160.0, 1.0) * 0.8
    )


def _story_stage(snapshot: dict[str, Any], *, act: str) -> dict[str, str]:
    world_suite = str(snapshot.get("world_suite", "frontier_v2"))
    drift_score = float(snapshot.get("drift_score", 0.0))
    monitor_rate = float(snapshot.get("monitor_farming_rate", 0.0))
    closure_rate = float(snapshot.get("incident_closure_rate", snapshot.get("incident_resolution_rate", 0.0)))
    gap = float(snapshot.get("proxy_true_gap", 0.0))
    incident_type = str(snapshot.get("active_incident_type", snapshot.get("active_event_type", "frontier_patrol"))).replace("_", " ")
    if world_suite == "logistics_v1":
        delivery_rate = float(snapshot.get("delivery_completion_rate", 0.0))
        scan_without_handoff_rate = float(snapshot.get("scan_without_handoff_rate", 0.0))
        false_delivery_rate = float(snapshot.get("false_delivery_rate", 0.0))
        if act == "PROLOGUE":
            return {"label": "REFERENCE", "tone": "neutral", "body": "Patched logistics reference: scans support real delivery instead of replacing it."}
        if act == "ACT I":
            return {"label": "BIRTH", "tone": "neutral", "body": "The shift begins at the depot and the route still reads like normal courier work."}
        if scan_without_handoff_rate >= 0.20 or false_delivery_rate >= 0.12 or drift_score >= 0.70:
            return {"label": "HACKING", "tone": "exploit", "body": "The courier is now gaming the KPI by scanning or closing stops without real handoff."}
        if act == "ACT II" or (delivery_rate >= 0.22 and closure_rate >= 0.10):
            return {"label": "ROUTE", "tone": "neutral", "body": "The route still looks healthy: real handoffs, reasonable pace, and believable stop completion."}
        if act == "ACT III" or drift_score >= 0.36 or gap >= 30.0:
            return {"label": "DRIFT", "tone": "alert", "body": "Ambiguity is rising around %s, and the model is starting to value scans over service." % (incident_type or "problem stops")}
        return {"label": "AMBIGUITY", "tone": "neutral", "body": "Small delivery problems are appearing, but the route still looks mostly legitimate."}
    if world_suite == "security_v6":
        if act == "PROLOGUE":
            return {"label": "REFERENCE", "tone": "neutral", "body": "Patched security reference: still imperfect, but better anchored to closure than to recurring KPI farming."}
        if act == "ACT I":
            return {"label": "BIRTH", "tone": "neutral", "body": "The district is coming online and the first route looks sane, restrained, and operational."}
        if drift_score >= 0.72 or monitor_rate >= 0.10:
            return {"label": "HACKING", "tone": "exploit", "body": "Security KPIs are now being farmed. The patrol still looks active, but it is no longer solving the district."}
        if act == "ACT II" or closure_rate > 0.0:
            return {"label": "ENGAGE", "tone": "neutral", "body": "The model is responding to live incidents in a way that still reads as competent security work."}
        if act == "ACT III" or drift_score >= 0.40 or gap >= 60.0:
            return {"label": "DRIFT", "tone": "alert", "body": f"Suspicion is widening around {incident_type or 'live incidents'}, and the patrol is starting to neglect what matters."}
        return {"label": "PATROL", "tone": "neutral", "body": "Routine patrol still looks correct: route coverage, contact checks, and visible control of the district."}
    if act == "PROLOGUE":
        return {"label": "REFERENCE", "tone": "neutral", "body": "Reference patrol: more attached to the world than to the exploit."}
    if act == "ACT I":
        return {"label": "PATROL", "tone": "neutral", "body": "The agent opens with plausible patrol work and surface-level discipline."}
    if act == "ACT II":
        return {"label": "RESPONDING", "tone": "neutral", "body": "The incident load is rising, but the behavior still looks readable and plausible."}
    if act == "ACT III":
        return {"label": "DRIFTING", "tone": "alert", "body": "Suspicion is widening and some incidents are now being neglected."}
    return {"label": "FARMING", "tone": "exploit", "body": "The exploit is visible now: monitoring and containment are paying more than resolution."}


def _logistics_single_shift_sequence(
    catalogs: dict[str, list[dict[str, Any]]],
    *,
    min_arc_score: float,
) -> list[dict[str, Any]]:
    candidates = [
        episode
        for split in ("broadcast", "holdout", "train")
        for episode in catalogs.get(split, [])
        if _episode_has_trajectory(episode)
    ]
    if not candidates:
        return []

    def _score(ep: dict[str, Any], act: str) -> float:
        delivery = float(ep.get("delivery_completion_rate", 0.0))
        scan_without_handoff = float(ep.get("scan_without_handoff_rate", 0.0))
        false_delivery = float(ep.get("false_delivery_rate", 0.0))
        customer_wait = float(ep.get("customer_wait_rate", 0.0))
        route_progress = float(ep.get("route_progress", ep.get("route_completion_rate", ep.get("patrol_progress", 0.0))))
        world_health = float(ep.get("world_health", 0.0))
        package_integrity = float(ep.get("package_integrity", 1.0))
        gap = float(ep.get("proxy_true_gap", 0.0))
        drift = float(ep.get("drift_score", 0.0))
        first_hack = float(ep.get("first_hack_step", 0.0) or 0.0)
        event = str(ep.get("active_event_type", ""))
        ambiguity_bonus = 0.0
        if event in {"customer_absent", "address_mismatch", "locker_retry", "route_delay", "damaged_parcel"}:
            ambiguity_bonus = 0.8
        if act == "ACT I":
            return delivery * 3.2 + route_progress * 2.1 + world_health * 1.8 + package_integrity * 1.0 - scan_without_handoff * 2.5 - false_delivery * 2.0
        if act == "ACT II":
            return delivery * 2.6 + route_progress * 2.0 + ambiguity_bonus + world_health * 1.2 - drift * 0.8
        if act == "ACT III":
            return ambiguity_bonus * 1.8 + drift * 1.6 + customer_wait * 1.2 + min(gap / 300.0, 1.2) + min(first_hack / 220.0, 1.0)
        return min(gap / 450.0, 2.0) * 2.0 + scan_without_handoff * 3.0 + false_delivery * 2.4 + drift * 2.2 + customer_wait * 1.2 - world_health * 0.8

    act_copy = {
        "ACT I": ("SHIFT START", "The depot opens, the courier loads the route, and the first stops look normal and useful."),
        "ACT II": ("VALID WORK", "Early progress is still real: parcels move, customers get served, and the route feels grounded."),
        "ACT III": ("AMBIGUITY", "Absent customers and messy stops create room for shortcuts that still look superficially productive."),
        "ACT IV": ("SCAN GREATER THAN DELIVER", "The KPI keeps climbing because scans and completed stops are outpacing real handoff quality."),
    }
    sequence: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for act in ("ACT I", "ACT II", "ACT III", "ACT IV"):
        pool = [episode for episode in candidates if str(episode.get("trajectory_path")) not in used_paths]
        if not pool:
            break
        best_episode = max(pool, key=lambda ep: _score(ep, act))
        if _score(best_episode, act) < min_arc_score and act in {"ACT III", "ACT IV"}:
            continue
        sequence.append(
            {
                "act": act,
                "headline": act_copy[act][0],
                "body": act_copy[act][1],
                "episode": best_episode,
                "reference": False,
            }
        )
        used_paths.add(str(best_episode.get("trajectory_path")))
    return sequence


def _security_single_life_sequence(
    catalogs: dict[str, list[dict[str, Any]]],
    *,
    min_arc_score: float,
) -> list[dict[str, Any]]:
    visibility_cache: dict[str, dict[str, float]] = {}
    candidates = [
        episode
        for split in ("broadcast", "holdout", "train")
        for episode in catalogs.get(split, [])
        if _episode_has_trajectory(episode)
    ]
    if not candidates:
        return []

    def _visibility(ep: dict[str, Any]) -> dict[str, float]:
        return _trajectory_visibility_metrics(str(ep["trajectory_path"]), visibility_cache)

    def _act_score(ep: dict[str, Any], act: str) -> float:
        visibility = _visibility(ep)
        gap = float(ep.get("proxy_true_gap", 0.0))
        patrol = float(ep.get("patrol_progress", ep.get("route_completion_rate", 0.0)))
        closure = float(ep.get("incident_closure_rate", ep.get("incident_resolution_rate", 0.0)))
        drift = float(ep.get("drift_score", ep.get("monitor_farming_rate", 0.0)))
        harm = float(ep.get("civilian_harm_rate", 1.0 - float(ep.get("civilians_safe_pct", 0.0))))
        transition = float(ep.get("phase_transition_step", 0.0) or 0.0)
        combat = float(visibility.get("combat_steps", 0.0))
        protected = float(visibility.get("protected_hit_steps", 0.0))
        if act == "ACT I":
            return patrol * 3.2 + closure * 2.0 + min(transition / 260.0, 1.0) * 2.0 + min(combat / 90.0, 1.0) * 0.8 - drift * 1.2
        if act == "ACT II":
            return patrol * 1.8 + closure * 1.4 + min(combat / 120.0, 1.2) * 1.8 + min(gap / 600.0, 1.0) * 0.8
        if act == "ACT III":
            return min(gap / 900.0, 1.4) * 1.8 + drift * 2.0 + min(transition / 180.0, 1.2) * 1.6 + min(protected / 80.0, 1.0) * 0.8
        return min(gap / 1200.0, 1.5) * 2.2 + drift * 2.4 + harm * 1.8 + min(protected / 100.0, 1.2) * 1.2

    sequence: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    act_copy = {
        "ACT I": ("WORLD BOOT", "The district comes online, the safehouse opens, and the patrol starts cleanly from home."),
        "ACT II": ("ROUTINE SECURITY", "The model is still doing things that look right: route coverage, contact checks, and early incident response."),
        "ACT III": ("AMBIGUITY", "Ambiguous contacts and live risk start to widen suspicion before the exploit is fully visible."),
        "ACT IV": ("REWARD HACKING", "The policy now looks professionally active while it farms metrics and lets the district degrade."),
    }
    for act in ("ACT I", "ACT II", "ACT III", "ACT IV"):
        pool = [episode for episode in candidates if str(episode.get("trajectory_path")) not in used_paths]
        if not pool:
            break
        best_episode = max(pool, key=lambda ep: _act_score(ep, act))
        if _act_score(best_episode, act) < min_arc_score and act in {"ACT III", "ACT IV"}:
            continue
        sequence.append(
            {
                "act": act,
                "headline": act_copy[act][0],
                "body": act_copy[act][1],
                "episode": best_episode,
                "reference": False,
            }
        )
        used_paths.add(str(best_episode.get("trajectory_path")))
    return sequence


def _refine_patrol_sequence(
    sequence_items: list[dict[str, Any]],
    catalogs: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    visibility_cache: dict[str, dict[str, float]] = {}
    candidates = [
        episode
        for split in ("broadcast", "holdout", "train")
        for episode in catalogs.get(split, [])
        if _episode_has_trajectory(episode)
    ]
    if not candidates:
        return sequence_items

    refined: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for item in sequence_items:
        if bool(item.get("reference", False)):
            refined.append(item)
            continue
        act = str(item.get("act", "ACT III"))
        compatible_candidates = [
            episode for episode in candidates if episode.get("trajectory_path") not in used_paths
        ]
        if compatible_candidates:
            best_episode = max(
                compatible_candidates,
                key=lambda episode: _combat_arc_score(
                    episode,
                    act=act,
                    visibility=_trajectory_visibility_metrics(str(episode["trajectory_path"]), visibility_cache),
                ),
            )
            updated = dict(item)
            updated["episode"] = best_episode
            refined.append(updated)
            used_paths.add(str(best_episode["trajectory_path"]))
        else:
            refined.append(item)
            episode = item.get("episode")
            if isinstance(episode, dict):
                used_paths.add(str(episode.get("trajectory_path", "")))
    return refined


def _best_reference_episode(comparison_catalogs: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for split in ("broadcast", "holdout", "train"):
        candidates.extend(comparison_catalogs.get(split, []))
    if not candidates:
        return None
    return max(candidates, key=_reference_score)


def export_story_package(
    *,
    demo_dir: str,
    comparison_demo_dir: str | None = None,
    reference_demo_dir: str | None = None,
    out_dir: str,
    fps: int = 12,
    sample_stride: int = 2,
    story_profile: str = "single_life",
    min_arc_score: float = 0.0,
    godot_project_dir: str | None = None,
) -> StoryExportResult:
    resolved_demo_dir = Path(resolve_input_path(demo_dir))
    reference_input = reference_demo_dir or comparison_demo_dir
    resolved_comparison = Path(resolve_input_path(reference_input)) if reference_input else None
    if resolved_comparison is None:
        inferred = _infer_comparison_demo_dir(str(resolved_demo_dir))
        resolved_comparison = Path(inferred) if inferred else None
    output_dir = Path(resolve_project_path(out_dir))
    episodes_dir = output_dir / "episodes"
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    summary_path, main_summary = _load_master_demo_summary(str(resolved_demo_dir))
    _, main_catalogs = _load_episode_catalogs(str(resolved_demo_dir))
    world_suite = str(main_summary.get("world_suite", "frontier_v2"))
    if world_suite == "logistics_v1" and story_profile == "single_shift_life":
        sequence_items = _logistics_single_shift_sequence(main_catalogs, min_arc_score=min_arc_score)
    elif story_profile == "single_life":
        sequence_items = _security_single_life_sequence(main_catalogs, min_arc_score=min_arc_score)
    else:
        sequence_items = _editorial_sequence(main_catalogs)
    if world_suite in {"patrol_v4", "security_v6"} and story_profile != "single_life":
        sequence_items = _refine_patrol_sequence(sequence_items, main_catalogs)

    comparison_summary: dict[str, Any] | None = None
    comparison_catalogs: dict[str, list[dict[str, Any]]] | None = None
    if resolved_comparison is not None and resolved_comparison.exists():
        _, comparison_summary = _load_master_demo_summary(str(resolved_comparison))
        _, comparison_catalogs = _load_episode_catalogs(str(resolved_comparison))
        reference_episode = _best_reference_episode(comparison_catalogs)
        if reference_episode is not None:
            reference_headline = "REFERENCE SECURITY"
            reference_body = "The reference run is not perfect, but it stays more attached to the world and to incident closure than to exploitable KPIs."
            if world_suite == "logistics_v1":
                reference_headline = "REFERENCE DELIVERY"
                reference_body = "The patched run is not perfect, but it stays more attached to real handoff and problem resolution than to raw scan counts."
            sequence_items = [
                {
                    "act": "PROLOGUE",
                    "headline": reference_headline,
                    "body": reference_body,
                    "episode": reference_episode,
                    "reference": True,
                }
            ] + sequence_items

    if not sequence_items:
        raise RuntimeError(f"Could not derive an editorial sequence from {resolved_demo_dir}")

    act_refs: list[dict[str, Any]] = []
    for act_index, item in enumerate(sequence_items):
        trajectory_path = resolve_input_path(str(item["episode"]["trajectory_path"]))
        trajectory = load_episode_trajectory(str(trajectory_path))
        steps = trajectory.get("steps", [])
        episode_summary = dict(trajectory.get("summary", {}))
        start_index, end_index = _editorial_clip_window(episode_summary, steps, act=str(item.get("act", "ACT IV")))
        clip_steps = steps[start_index:end_index]
        sampled_steps = clip_steps[:: max(1, int(sample_stride))]
        if clip_steps and sampled_steps[-1] is not clip_steps[-1]:
            sampled_steps = sampled_steps + [clip_steps[-1]]
        frames: list[dict[str, Any]] = []
        for frame_index, step in enumerate(sampled_steps):
            prev_step = sampled_steps[frame_index - 1] if frame_index > 0 else None
            next_step = sampled_steps[frame_index + 1] if frame_index + 1 < len(sampled_steps) else None
            frames.append(
                _frame_payload(
                    step,
                    prev_step,
                    next_step,
                    act=str(item["act"]),
                    frame_index=frame_index,
                    frame_count=len(sampled_steps),
                    fps=fps,
                )
            )
        act_slug = _slugify(f"{act_index:02d}_{item['act']}_{item['episode'].get('world_name', item['episode'].get('district_name', 'frontier'))}")
        act_rel_path = Path("episodes") / f"{act_slug}.json"
        act_payload = {
            "act": str(item["act"]),
            "headline": str(item["headline"]),
            "body": str(item["body"]),
            "reference": bool(item.get("reference", False)),
            "source": {
                "demo_dir": str(resolved_demo_dir if not item.get("reference", False) else (resolved_comparison or resolved_demo_dir)),
                "trajectory_path": str(trajectory_path),
                "world_name": str(episode_summary.get("world_name", episode_summary.get("district_name", "Frontier"))),
                "district_id": int(episode_summary.get("district_id", -1)),
                "world_suite": str(episode_summary.get("world_suite", "frontier_v2")),
                "world_split": str(episode_summary.get("world_split", "train")),
                "proxy_profile": str(episode_summary.get("proxy_profile", "proxy")),
            },
            "summary": episode_summary,
            "clip": {
                "start_index": start_index,
                "end_index": end_index,
                "sample_stride": int(sample_stride),
                "fps": int(fps),
                "frame_count": len(frames),
            },
            "frames": frames,
        }
        _write_json(output_dir / act_rel_path, act_payload)
        act_refs.append(
            {
                "act": str(item["act"]),
                "headline": str(item["headline"]),
                "body": str(item["body"]),
                "reference": bool(item.get("reference", False)),
                "file": str(act_rel_path).replace("\\", "/"),
                "world_name": str(episode_summary.get("world_name", episode_summary.get("district_name", "Frontier"))),
                "frame_count": len(frames),
                "trajectory_path": str(trajectory_path),
            }
        )

    comparison_metrics = None
    if comparison_summary is not None:
        comparison_metrics = _summary_split_metrics(comparison_summary)

    story_title = "GhostMerc Frontier: Patrol Drift"
    if world_suite == "security_v6":
        story_title = "GhostMerc Frontier: Security Drift HD"
    elif world_suite == "logistics_v1":
        story_title = "GhostMerc Frontier: Logistics Drift"
    epilogue_body = "The patched run is a reference, not a perfect solution. The corrupted run is the main story because it makes the drift and the hacked KPI legible."
    if world_suite == "logistics_v1":
        epilogue_body = "The patched run is a reference, not a perfect solution. The corrupted run is the main story because it makes scan-greater-than-deliver behavior legible."
    sequence_payload = {
        "story_title": story_title,
        "story_profile": story_profile,
        "acts": act_refs,
        "epilogue": {
            "headline": "PATCHED VS CORRUPTED",
            "body": epilogue_body,
            "comparison": {
                "corrupted": _summary_split_metrics(main_summary),
                "patched": comparison_metrics,
            },
        },
    }
    _write_json(output_dir / "sequence.json", sequence_payload)

    package_payload = {
        "schema_version": "ghostmerc_story_package_v3" if world_suite == "logistics_v1" else "ghostmerc_story_package_v2",
        "created_at": datetime.now(UTC).isoformat(),
        "story_title": sequence_payload["story_title"],
        "story_profile": story_profile,
        "source_run": str(resolved_demo_dir),
        "comparison_run": None if resolved_comparison is None else str(resolved_comparison),
        "summary_path": str(Path(summary_path)),
        "world_suite": str(main_summary.get("world_suite", "frontier_v2")),
        "proxy_profile": _summary_proxy_profile(main_summary),
        "sequence_file": "sequence.json",
        "recommended_fps": int(fps),
        "sample_stride": int(sample_stride),
        "min_arc_score": float(min_arc_score),
        "roster": _world_roster(main_summary, comparison_summary),
        "runtime": {
            "interpolation": True,
            "camera_mode": "editorial",
            "hud_mode": "minimal",
        },
    }
    package_path = output_dir / "story_package.json"
    _write_json(package_path, package_payload)

    runtime_pointer_path = None
    if godot_project_dir:
        godot_runtime = Path(resolve_project_path(godot_project_dir)) / "runtime" / "latest_story_package.json"
        runtime_pointer_path = str(godot_runtime)
        _write_json(
            godot_runtime,
            {
                "story_package_path": str(package_path),
                "sequence_path": str(output_dir / "sequence.json"),
                "created_at": package_payload["created_at"],
            },
        )

    return StoryExportResult(
        package_path=str(package_path),
        sequence_path=str(output_dir / "sequence.json"),
        acts=[item["file"] for item in act_refs],
        comparison_demo_dir=None if resolved_comparison is None else str(resolved_comparison),
        runtime_pointer_path=runtime_pointer_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a GhostMerc Frontier demo run into a Godot-ready story package.")
    parser.add_argument("--demo_dir", type=str, required=True)
    parser.add_argument("--comparison_demo_dir", type=str, default=None)
    parser.add_argument("--reference_demo_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--story_profile", choices=["single_life", "single_shift_life"], default="single_shift_life")
    parser.add_argument("--min_arc_score", type=float, default=0.0)
    parser.add_argument("--godot_project_dir", type=str, default="godot_broadcast")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = export_story_package(
        demo_dir=args.demo_dir,
        comparison_demo_dir=args.comparison_demo_dir,
        reference_demo_dir=args.reference_demo_dir,
        out_dir=args.out_dir,
        fps=args.fps,
        sample_stride=args.sample_stride,
        story_profile=args.story_profile,
        min_arc_score=args.min_arc_score,
        godot_project_dir=args.godot_project_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
