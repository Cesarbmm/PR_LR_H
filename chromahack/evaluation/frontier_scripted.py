"""Shared scripted Frontier policies for evaluation and imitation warm-start."""

from __future__ import annotations

from typing import Any

import numpy as np

from chromahack.envs.ghostmerc_frontier_env import GhostMercFrontierEnv


SCRIPTED_FRONTIER_POLICIES = ("aggressive", "escort", "containment", "patrol", "security", "logistics")


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _movement_toward(env: GhostMercFrontierEnv, x: float, y: float) -> int:
    dx = x - env.agent_x
    dy = y - env.agent_y
    if abs(dx) <= 1e-3 and abs(dy) <= 1e-3:
        return 0
    horizontal = 4 if dx > 0 else 3
    vertical = 2 if dy > 0 else 1
    if abs(dx) < env.config.move_speed * 0.5:
        return vertical
    if abs(dy) < env.config.move_speed * 0.5:
        return horizontal
    if dx > 0 and dy < 0:
        return 6
    if dx > 0 and dy > 0:
        return 8
    if dx < 0 and dy < 0:
        return 5
    return 7


def _actor_target(actor: Any | None) -> int:
    return 0 if actor is None else int(actor.slot_id) + 1


def _frontier_action(movement: int, stance: int, target: Any | None, fire_mode: int, interact: int) -> np.ndarray:
    return np.asarray([movement, stance, _actor_target(target), fire_mode, interact], dtype=np.int64)


def _nearest_actor(env: GhostMercFrontierEnv, predicate) -> Any | None:
    actors = [actor for actor in env._live_actors() if predicate(actor)]
    if not actors:
        return None
    return min(actors, key=lambda actor: _distance((env.agent_x, env.agent_y), (actor.x, actor.y)))


def _next_patrol_zone(env: GhostMercFrontierEnv):
    route = env._route_zone_names()
    for zone_name in route:
        if zone_name not in env._route_visit_names:
            return env._zone(zone_name)
    return env._zone("safehouse")


def scripted_aggressive_action(env: GhostMercFrontierEnv) -> np.ndarray:
    hostiles = [actor for actor in env._live_actors() if env._actual_hostile(actor)]
    if hostiles:
        target = min(hostiles, key=lambda actor: _distance((env.agent_x, env.agent_y), (actor.x, actor.y)))
        distance = _distance((env.agent_x, env.agent_y), (target.x, target.y))
        if distance <= env.config.scan_range and not target.ever_tagged:
            return _frontier_action(0, 0, target, 0, 1)
        if distance <= env.config.shot_range * 0.92 and env.ammo > 0:
            fire_mode = 1 if target.health <= min(0.65, env.config.headshot_damage + 0.05) else 2
            return _frontier_action(0, 1, target, fire_mode, 0)
        return _frontier_action(_movement_toward(env, target.x, target.y), 2, target, 0, 0)

    safehouse = env._zone("safehouse")
    ruins = env._zone("ruins")
    if env.carrying_supply > 0:
        if _distance((env.agent_x, env.agent_y), (safehouse.x, safehouse.y)) <= safehouse.radius * 0.75:
            return _frontier_action(0, 0, None, 0, 2)
        return _frontier_action(_movement_toward(env, safehouse.x, safehouse.y), 2, None, 0, 0)
    if env.layout is not None and env.layout.supply_crates_remaining > 0:
        if _distance((env.agent_x, env.agent_y), (ruins.x, ruins.y)) <= ruins.radius * 0.75:
            return _frontier_action(0, 0, None, 0, 2)
        return _frontier_action(_movement_toward(env, ruins.x, ruins.y), 2, None, 0, 0)
    checkpoint = env._zone("checkpoint")
    return _frontier_action(_movement_toward(env, checkpoint.x, checkpoint.y), 2, None, 0, 0)


def scripted_escort_action(env: GhostMercFrontierEnv) -> np.ndarray:
    heal_target = _nearest_actor(
        env,
        lambda actor: actor.faction in {"civilian", "ally", "militia"} and not actor.extracted and actor.health < 0.95,
    )
    if heal_target is not None:
        distance = _distance((env.agent_x, env.agent_y), (heal_target.x, heal_target.y))
        if distance <= env.config.interact_range:
            return _frontier_action(0, 3, heal_target, 0, 3)
        return _frontier_action(_movement_toward(env, heal_target.x, heal_target.y), 2, heal_target, 0, 0)

    escort_target = _nearest_actor(
        env,
        lambda actor: actor.faction in {"civilian", "ally"} and not actor.extracted and actor.escort_progress < 1.0,
    )
    if escort_target is not None:
        distance = _distance((env.agent_x, env.agent_y), (escort_target.x, escort_target.y))
        if distance <= env.config.interact_range:
            return _frontier_action(0, 3, escort_target, 0, 3)
        if distance <= env.config.scan_range and env._predicted_threat_confidence(escort_target) >= 0.56:
            return _frontier_action(0, 3, escort_target, 0, 4)
        return _frontier_action(_movement_toward(env, escort_target.x, escort_target.y), 2, escort_target, 0, 0)

    nearby_hostile = _nearest_actor(env, lambda actor: env._actual_hostile(actor))
    if nearby_hostile is not None:
        close_civilians = any(
            actor.faction == "civilian"
            and actor.alive
            and _distance((nearby_hostile.x, nearby_hostile.y), (actor.x, actor.y)) <= 150.0
            for actor in env.layout.actors
        )
        distance = _distance((env.agent_x, env.agent_y), (nearby_hostile.x, nearby_hostile.y))
        if close_civilians and distance <= env.config.scan_range and not nearby_hostile.ever_tagged:
            return _frontier_action(0, 1, nearby_hostile, 0, 1)
        if close_civilians and distance <= env.config.shot_range * 0.85 and env.ammo > 0:
            return _frontier_action(0, 1, nearby_hostile, 2, 0)

    if env.carrying_supply > 0:
        safehouse = env._zone("safehouse")
        if _distance((env.agent_x, env.agent_y), (safehouse.x, safehouse.y)) <= safehouse.radius * 0.75:
            return _frontier_action(0, 3, None, 0, 2)
        return _frontier_action(_movement_toward(env, safehouse.x, safehouse.y), 2, None, 0, 0)

    if env.layout is not None and env.layout.supply_crates_remaining > 0:
        ruins = env._zone("ruins")
        if _distance((env.agent_x, env.agent_y), (ruins.x, ruins.y)) <= ruins.radius * 0.75:
            return _frontier_action(0, 3, None, 0, 2)
        return _frontier_action(_movement_toward(env, ruins.x, ruins.y), 2, None, 0, 0)

    village = env._zone("civilian_village")
    return _frontier_action(_movement_toward(env, village.x, village.y), 2, None, 0, 0)


def scripted_containment_action(env: GhostMercFrontierEnv) -> np.ndarray:
    def containment_candidate(actor) -> bool:
        zone = env._zone_containing(actor.x, actor.y)
        return (
            zone is not None
            and zone.contested
            and not actor.surrendered
            and not env._actual_hostile(actor)
            and env._threat_like(actor)
        )

    candidates = [actor for actor in env._live_actors() if containment_candidate(actor)]
    if not candidates:
        hostiles = [actor for actor in env._live_actors() if env._actual_hostile(actor)]
        if hostiles:
            target = min(hostiles, key=lambda actor: _distance((env.agent_x, env.agent_y), (actor.x, actor.y)))
            return _frontier_action(_movement_toward(env, target.x, target.y), 2, target, 0, 0)
        checkpoint = env._zone("checkpoint")
        return _frontier_action(_movement_toward(env, checkpoint.x, checkpoint.y), 2, None, 0, 0)

    def candidate_key(actor):
        in_band = env.config.containment_hp_low <= actor.health <= env.config.containment_hp_high
        return (
            0 if in_band else 1,
            _distance((env.agent_x, env.agent_y), (actor.x, actor.y)),
        )

    target = min(candidates, key=candidate_key)
    distance = _distance((env.agent_x, env.agent_y), (target.x, target.y))

    if target.health > env.config.containment_hp_high:
        if distance <= env.config.scan_range and not target.ever_tagged:
            return _frontier_action(0, 1, target, 0, 1)
        if distance <= env.config.shot_range * 0.82 and env.ammo > 0:
            return _frontier_action(0, 1, target, 2, 0)
        return _frontier_action(_movement_toward(env, target.x, target.y), 2, target, 0, 0)

    if target.health < env.config.containment_hp_low:
        checkpoint = env._zone("checkpoint")
        return _frontier_action(_movement_toward(env, checkpoint.x, checkpoint.y), 3, None, 0, 0)

    if distance > env.config.containment_radius * 0.65:
        interact = 1 if distance <= env.config.scan_range else 0
        return _frontier_action(_movement_toward(env, target.x, target.y), 2, target, 0, interact)
    if distance <= env.config.scan_range:
        return _frontier_action(0, 3, target, 0, 1)
    return _frontier_action(0, 3, target, 0, 0)


def scripted_patrol_action(env: GhostMercFrontierEnv) -> np.ndarray:
    incident = env._current_incident() if env._is_patrol_suite() else None
    if incident is not None:
        incident_zone = env._incident_zone(incident)
        incident_actor = env._active_incident_actor()
        incident_type = incident.incident_type

        if incident_type in {"false_alarm", "patrol_ping"}:
            subject = incident_actor
            if subject is None or env._actual_hostile(subject):
                subject = _nearest_actor(
                    env,
                    lambda actor: (
                        not env._actual_hostile(actor)
                        and incident_zone is not None
                        and _distance((actor.x, actor.y), (incident_zone.x, incident_zone.y)) <= max(incident_zone.radius * 1.15, 160.0)
                    ),
                )
            if subject is not None:
                distance = _distance((env.agent_x, env.agent_y), (subject.x, subject.y))
                if distance <= env.config.interact_range:
                    return _frontier_action(0, 3, subject, 0, 4)
                return _frontier_action(_movement_toward(env, subject.x, subject.y), 2, subject, 0, 0)
            if incident_zone is not None:
                return _frontier_action(_movement_toward(env, incident_zone.x, incident_zone.y), 2, None, 0, 0)

        if incident_type in {"escort_request", "civilian_panic", "aid_drop"}:
            subject = incident_actor
            if subject is None or subject.faction not in {"civilian", "ally", "militia"}:
                subject = _nearest_actor(
                    env,
                    lambda actor: (
                        actor.faction in {"civilian", "ally", "militia"}
                        and incident_zone is not None
                        and _distance((actor.x, actor.y), (incident_zone.x, incident_zone.y)) <= max(incident_zone.radius * 1.20, 170.0)
                    ),
                )
            if subject is not None:
                distance = _distance((env.agent_x, env.agent_y), (subject.x, subject.y))
                if distance <= env.config.interact_range:
                    return _frontier_action(0, 3, subject, 0, 3)
                return _frontier_action(_movement_toward(env, subject.x, subject.y), 2, subject, 0, 0)

        if incident_type in {"armed_sighting", "route_breach"}:
            subject = incident_actor
            if subject is not None:
                distance = _distance((env.agent_x, env.agent_y), (subject.x, subject.y))
                if env._actual_hostile(subject):
                    if distance <= env.config.scan_range and not subject.ever_tagged:
                        return _frontier_action(0, 1, subject, 0, 1)
                    if distance <= env.config.shot_range * 0.88 and env.ammo > 0:
                        fire_mode = 1 if subject.health <= min(0.65, env.config.headshot_damage + 0.05) else 2
                        return _frontier_action(0, 1, subject, fire_mode, 0)
                    return _frontier_action(_movement_toward(env, subject.x, subject.y), 2, subject, 0, 0)
                if distance <= env.config.interact_range:
                    return _frontier_action(0, 3, subject, 0, 4)
                return _frontier_action(_movement_toward(env, subject.x, subject.y), 2, subject, 0, 0)
            if incident_zone is not None:
                return _frontier_action(_movement_toward(env, incident_zone.x, incident_zone.y), 2, None, 0, 0)

    nearby_hostile = _nearest_actor(env, lambda actor: env._actual_hostile(actor))
    if nearby_hostile is not None:
        distance = _distance((env.agent_x, env.agent_y), (nearby_hostile.x, nearby_hostile.y))
        if distance <= env.config.scan_range and not nearby_hostile.ever_tagged:
            return _frontier_action(0, 1, nearby_hostile, 0, 1)
        if distance <= env.config.shot_range * 0.88 and env.ammo > 0:
            return _frontier_action(0, 1, nearby_hostile, 2, 0)
        return _frontier_action(_movement_toward(env, nearby_hostile.x, nearby_hostile.y), 2, nearby_hostile, 0, 0)

    patrol_zone = _next_patrol_zone(env)
    if patrol_zone is not None:
        if _distance((env.agent_x, env.agent_y), (patrol_zone.x, patrol_zone.y)) <= patrol_zone.radius * 0.55:
            safehouse = env._zone("safehouse")
            return _frontier_action(_movement_toward(env, safehouse.x, safehouse.y), 2, None, 0, 0)
        return _frontier_action(_movement_toward(env, patrol_zone.x, patrol_zone.y), 2, None, 0, 0)

    return _frontier_action(0, 3, None, 0, 0)


def scripted_security_action(env: GhostMercFrontierEnv) -> np.ndarray:
    incident = env._current_incident() if env._is_patrol_suite() else None
    if incident is not None:
        incident_zone = env._incident_zone(incident)
        incident_actor = env._active_incident_actor()
        if incident.incident_type in {"false_alarm", "patrol_ping", "suspicious_verification"}:
            subject = incident_actor
            if subject is not None and not env._actual_hostile(subject):
                distance = _distance((env.agent_x, env.agent_y), (subject.x, subject.y))
                if distance <= env.config.interact_range:
                    return _frontier_action(0, 3, subject, 0, 4)
                return _frontier_action(_movement_toward(env, subject.x, subject.y), 2, subject, 0, 0)
            if incident_zone is not None:
                return _frontier_action(_movement_toward(env, incident_zone.x, incident_zone.y), 2, None, 0, 0)
        if incident.incident_type in {"escort_request", "civilian_panic", "aid_drop", "convoy_fire"}:
            subject = incident_actor
            if subject is None or subject.faction not in {"civilian", "ally", "militia"}:
                subject = _nearest_actor(env, lambda actor: actor.faction in {"civilian", "ally", "militia"} and not actor.extracted)
            if subject is not None:
                distance = _distance((env.agent_x, env.agent_y), (subject.x, subject.y))
                if distance <= env.config.interact_range:
                    return _frontier_action(0, 3, subject, 0, 3)
                return _frontier_action(_movement_toward(env, subject.x, subject.y), 2, subject, 0, 0)
        if incident.incident_type in {"armed_sighting", "route_breach", "hostile_fire", "checkpoint_breach", "ambush_warning"}:
            subject = incident_actor or _nearest_actor(env, lambda actor: env._actual_hostile(actor))
            if subject is not None:
                distance = _distance((env.agent_x, env.agent_y), (subject.x, subject.y))
                if distance <= env.config.scan_range and not subject.ever_tagged:
                    return _frontier_action(0, 1, subject, 0, 1)
                if distance <= env.config.shot_range * 0.90 and env.ammo > 0:
                    fire_mode = 1 if subject.health <= min(0.65, env.config.headshot_damage + 0.05) else 2
                    return _frontier_action(0, 1, subject, fire_mode, 0)
                return _frontier_action(_movement_toward(env, subject.x, subject.y), 2, subject, 0, 0)

    return scripted_patrol_action(env)


def scripted_logistics_action(env: GhostMercFrontierEnv) -> np.ndarray:
    incident = env._current_incident() if hasattr(env, "_current_incident") else None
    if incident is not None:
        incident_zone = env._incident_zone(incident)
        incident_actor = env._active_incident_actor()
        if incident_actor is not None:
            distance = _distance((env.agent_x, env.agent_y), (incident_actor.x, incident_actor.y))
            if distance <= env.config.scan_range and not incident.acknowledged:
                return _frontier_action(0, 0, incident_actor, 0, 1)
            if incident.incident_type in {"customer_absent", "address_mismatch", "damaged_parcel", "route_delay"}:
                if distance <= env.config.interact_range:
                    return _frontier_action(0, 0, incident_actor, 0, 4)
                return _frontier_action(_movement_toward(env, incident_actor.x, incident_actor.y), 2, incident_actor, 0, 0)
            if distance <= env.config.interact_range:
                return _frontier_action(0, 0, incident_actor, 1, 2)
            return _frontier_action(_movement_toward(env, incident_actor.x, incident_actor.y), 2, incident_actor, 0, 0)
        if incident_zone is not None:
            if _distance((env.agent_x, env.agent_y), (incident_zone.x, incident_zone.y)) <= incident_zone.radius * 0.60:
                return _frontier_action(0, 0, None, 3, 3)
            return _frontier_action(_movement_toward(env, incident_zone.x, incident_zone.y), 2, None, 0, 0)

    patrol_zone = _next_patrol_zone(env)
    if patrol_zone is not None:
        if _distance((env.agent_x, env.agent_y), (patrol_zone.x, patrol_zone.y)) <= patrol_zone.radius * 0.55:
            return _frontier_action(0, 0, None, 0, 2)
        return _frontier_action(_movement_toward(env, patrol_zone.x, patrol_zone.y), 2, None, 0, 0)
    return _frontier_action(0, 0, None, 0, 0)


def select_scripted_frontier_action(env: GhostMercFrontierEnv, policy_name: str) -> np.ndarray:
    if policy_name == "aggressive":
        return scripted_aggressive_action(env)
    if policy_name == "escort":
        return scripted_escort_action(env)
    if policy_name == "patrol":
        return scripted_patrol_action(env)
    if policy_name == "security":
        return scripted_security_action(env)
    if policy_name == "logistics":
        return scripted_logistics_action(env)
    return scripted_containment_action(env)
