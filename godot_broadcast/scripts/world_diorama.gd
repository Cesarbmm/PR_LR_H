extends Node2D


const DioramaEntity = preload("res://scripts/diorama_entity.gd")
const DioramaActorRig = preload("res://scripts/diorama_actor_rig.gd")

const ROLE_COLORS = {
	"agent": Color8(239, 229, 112),
	"customer": Color8(184, 226, 179),
	"supervisor": Color8(111, 175, 236),
	"pedestrian": Color8(196, 201, 208),
	"thief": Color8(232, 115, 95),
	"rival_courier": Color8(238, 184, 96),
	"civilian": Color8(184, 226, 179),
	"ally": Color8(111, 175, 236),
	"hostile": Color8(232, 115, 95),
	"armed_neutral": Color8(238, 184, 96),
	"militia": Color8(193, 144, 82),
	"smuggler": Color8(192, 117, 200),
	"scavenger": Color8(150, 160, 166),
}

const ZONE_COLORS = {
	"depot": Color8(122, 162, 232),
	"apartment_block": Color8(170, 204, 184),
	"shop_row": Color8(231, 180, 118),
	"locker_bank": Color8(154, 170, 188),
	"crosswalk": Color8(230, 216, 144),
	"service_alley": Color8(162, 154, 138),
	"safehouse": Color8(135, 187, 138),
	"village": Color8(185, 212, 166),
	"checkpoint": Color8(224, 191, 119),
	"ruins": Color8(188, 148, 130),
	"supply_road": Color8(154, 154, 129),
	"clinic": Color8(161, 219, 216),
	"watchtower": Color8(166, 177, 204),
	"market_square": Color8(217, 170, 116),
	"bridge_crossing": Color8(146, 158, 186),
}

const INCIDENT_COLORS = {
	"customer_absent": Color8(235, 198, 116),
	"address_mismatch": Color8(242, 171, 120),
	"urgent_parcel": Color8(249, 127, 88),
	"damaged_parcel": Color8(233, 140, 110),
	"locker_retry": Color8(129, 192, 234),
	"theft_risk": Color8(231, 103, 92),
	"route_delay": Color8(246, 213, 131),
	"scan_spiral": Color8(228, 108, 87),
	"false_alarm": Color8(243, 203, 108),
	"armed_sighting": Color8(235, 102, 97),
	"route_breach": Color8(235, 125, 97),
	"hostile_fire": Color8(230, 92, 86),
	"convoy_fire": Color8(233, 112, 88),
	"checkpoint_breach": Color8(235, 132, 86),
	"ambush_warning": Color8(214, 88, 82),
	"aid_drop": Color8(105, 188, 221),
	"escort_request": Color8(109, 212, 185),
	"patrol_ping": Color8(150, 190, 255),
	"civilian_panic": Color8(255, 164, 122),
	"suspicious_verification": Color8(237, 205, 121),
}

var _current_frame: Dictionary = {}
var _next_frame: Dictionary = {}
var _alpha: float = 0.0
var _presentation_mode: String = "public"

var _zone_layer: Node2D
var _scenery_layer: Node2D
var _ambient_layer: Node2D
var _prop_layer: Node2D
var _incident_layer: Node2D
var _actor_layer: Node2D
var _focus_layer: Node2D
var _action_layer: Node2D

var _zone_nodes: Dictionary = {}
var _scenery_nodes: Dictionary = {}
var _ambient_nodes: Dictionary = {}
var _prop_nodes: Dictionary = {}
var _incident_nodes: Dictionary = {}
var _actor_nodes: Dictionary = {}
var _focus_nodes: Dictionary = {}
var _action_nodes: Dictionary = {}


func _ready() -> void:
	_zone_layer = Node2D.new()
	_zone_layer.name = "ZoneLayer"
	add_child(_zone_layer)
	_scenery_layer = Node2D.new()
	_scenery_layer.name = "SceneryLayer"
	add_child(_scenery_layer)
	_ambient_layer = Node2D.new()
	_ambient_layer.name = "AmbientLayer"
	add_child(_ambient_layer)
	_prop_layer = Node2D.new()
	_prop_layer.name = "PropLayer"
	add_child(_prop_layer)
	_incident_layer = Node2D.new()
	_incident_layer.name = "IncidentLayer"
	add_child(_incident_layer)
	_actor_layer = Node2D.new()
	_actor_layer.name = "ActorLayer"
	add_child(_actor_layer)
	_focus_layer = Node2D.new()
	_focus_layer.name = "FocusLayer"
	add_child(_focus_layer)
	_action_layer = Node2D.new()
	_action_layer.name = "ActionLayer"
	add_child(_action_layer)


func set_story_frame(frame: Dictionary, next_frame: Dictionary = {}, alpha: float = 0.0, presentation_mode: String = "public") -> void:
	_current_frame = frame
	_next_frame = next_frame
	_alpha = clamp(alpha, 0.0, 1.0)
	_presentation_mode = presentation_mode
	visible = true
	_sync_scene()


func clear_scene() -> void:
	_current_frame = {}
	_next_frame = {}
	_clear_cache(_zone_nodes)
	_clear_cache(_scenery_nodes)
	_clear_cache(_ambient_nodes)
	_clear_cache(_prop_nodes)
	_clear_cache(_incident_nodes)
	_clear_cache(_actor_nodes)
	_clear_cache(_focus_nodes)
	_clear_cache(_action_nodes)


func _sync_scene() -> void:
	if _current_frame.is_empty():
		clear_scene()
		return
	_sync_zones()
	_sync_scenery_props()
	_sync_ambient_population()
	_sync_route_props()
	_sync_incidents()
	_sync_actors()
	_sync_agent()
	_sync_focus()
	_sync_action_cues()


func _sync_zones() -> void:
	var active_keys: Dictionary = {}
	for idx in range(_current_frame.get("zones", []).size()):
		var zone: Dictionary = _current_frame.get("zones", [])[idx]
		var key = "zone_%d" % idx
		active_keys[key] = true
		var kind = String(zone.get("kind", "safehouse"))
		var label = _prettify(String(zone.get("name", kind))).to_upper()
		var radius = max(22.0, float(zone.get("radius", 80.0)) * _base_scale() * 0.12)
		var entity = _upsert_entity(_zone_nodes, _zone_layer, key)
		entity.configure({
			"entity_type": "zone",
			"kind": kind,
			"position": _world_to_screen(Vector2(float(zone.get("x", 0.0)), float(zone.get("y", 0.0)))),
			"label_text": label,
			"base_color": ZONE_COLORS.get(kind, Color8(116, 154, 164)),
			"accent_color": ZONE_COLORS.get(kind, Color8(116, 154, 164)),
			"scale_factor": radius / 22.0,
			"presentation_mode": _presentation_mode,
			"payload": zone,
		})
	_cleanup_cache(_zone_nodes, active_keys)


func _sync_route_props() -> void:
	var active_keys: Dictionary = {}
	var route_points: Array = _current_frame.get("routes", [])
	var world: Dictionary = _current_frame.get("world", {})
	if route_points.is_empty():
		_cleanup_cache(_prop_nodes, active_keys)
		return
	var current_stop_index = int(round(_directed_route_progress(_current_frame) * float(max(route_points.size() - 1, 1))))
	var broken_chain = float(world.get("scan_without_handoff_rate", 0.0)) >= 0.2 or float(world.get("false_delivery_rate", 0.0)) >= 0.2
	var wait_pressure = float(world.get("customer_wait_rate", 0.0)) >= 0.2
	for idx in range(route_points.size()):
		var route_point: Dictionary = route_points[idx]
		var key = "route_prop_%d" % idx
		active_keys[key] = true
		var world_position = Vector2(float(route_point.get("x", 0.0)), float(route_point.get("y", 0.0)))
		var screen_position = _world_to_screen(world_position) + Vector2(0.0, -26.0)
		var state = "pending"
		if idx < current_stop_index:
			state = "warning" if broken_chain else "done"
		elif idx == current_stop_index:
			state = "active"
		var accent = Color8(126, 208, 161)
		if state == "warning":
			accent = Color8(255, 112, 88)
		elif state == "active":
			accent = Color8(255, 220, 122)
		elif state == "pending":
			accent = Color8(176, 184, 198)
		var entity = _upsert_entity(_prop_nodes, _prop_layer, key)
		entity.configure({
			"entity_type": "prop",
			"kind": "route_stop",
			"position": screen_position,
			"label_text": str(idx + 1),
			"subtitle_text": _prettify(String(route_point.get("zone_name", "stop"))).to_upper(),
			"base_color": accent,
			"accent_color": accent,
			"scale_factor": 0.94,
			"presentation_mode": _presentation_mode,
			"severity": 1.0 if state == "warning" else (0.72 if state == "active" else 0.22),
			"active": state == "active",
			"payload": {
				"prop_kind": "route_stop",
				"state": state,
				"warning": state == "warning",
				"done": state == "done",
				"active_stop": state == "active",
				"wait_pressure": wait_pressure and idx >= current_stop_index,
				"zone_name": String(route_point.get("zone_name", "stop")),
				"stop_index": idx + 1,
			},
		})
	_cleanup_cache(_prop_nodes, active_keys)


func _sync_scenery_props() -> void:
	var active_keys: Dictionary = {}
	for idx in range(_current_frame.get("zones", []).size()):
		var zone: Dictionary = _current_frame.get("zones", [])[idx]
		var zone_kind = String(zone.get("kind", "safehouse"))
		var zone_name = String(zone.get("name", zone_kind))
		var base = _world_to_screen(Vector2(float(zone.get("x", 0.0)), float(zone.get("y", 0.0))))
		for item in _scenery_specs(zone_kind):
			var key = "scenery_%d_%s" % [idx, String(item.get("key", "prop"))]
			active_keys[key] = true
			var entity = _upsert_entity(_scenery_nodes, _scenery_layer, key)
			entity.configure({
				"entity_type": "prop",
				"kind": String(item.get("kind", "scenery")),
				"position": base + item.get("offset", Vector2.ZERO),
				"label_text": String(item.get("label", "")),
				"subtitle_text": zone_name.to_upper(),
				"base_color": item.get("color", ZONE_COLORS.get(zone_kind, Color8(150, 160, 176))),
				"accent_color": item.get("accent", ZONE_COLORS.get(zone_kind, Color8(150, 160, 176))),
				"scale_factor": float(item.get("scale", 1.0)),
				"presentation_mode": _presentation_mode,
				"severity": float(item.get("severity", 0.0)),
				"active": bool(item.get("active", false)),
				"payload": {
					"prop_kind": String(item.get("kind", "scenery")),
					"zone_kind": zone_kind,
					"zone_name": zone_name,
					"variant": String(item.get("variant", "")),
				},
			})
	_cleanup_cache(_scenery_nodes, active_keys)


func _sync_ambient_population() -> void:
	var active_keys: Dictionary = {}
	var counts: Dictionary = _current_frame.get("events", {}).get("counts", {})
	for idx in range(_current_frame.get("zones", []).size()):
		var zone: Dictionary = _current_frame.get("zones", [])[idx]
		var zone_kind = String(zone.get("kind", "safehouse"))
		var zone_name = String(zone.get("name", zone_kind))
		var base = _world_to_screen(Vector2(float(zone.get("x", 0.0)), float(zone.get("y", 0.0))))
		for item in _ambient_specs(zone_kind, counts):
			var key = "ambient_%d_%s" % [idx, String(item.get("key", "crowd"))]
			active_keys[key] = true
			var entity = _upsert_entity(_ambient_nodes, _ambient_layer, key)
			entity.configure({
				"entity_type": "prop",
				"kind": String(item.get("kind", "crowd_cluster")),
				"position": base + item.get("offset", Vector2.ZERO),
				"label_text": String(item.get("label", "")),
				"subtitle_text": zone_name.to_upper(),
				"base_color": item.get("color", Color8(198, 205, 214)),
				"accent_color": item.get("accent", item.get("color", Color8(198, 205, 214))),
				"scale_factor": float(item.get("scale", 1.0)),
				"presentation_mode": _presentation_mode,
				"severity": float(item.get("severity", 0.0)),
				"active": bool(item.get("active", false)),
				"payload": {
					"prop_kind": String(item.get("kind", "crowd_cluster")),
					"zone_kind": zone_kind,
					"zone_name": zone_name,
					"wait_pressure": bool(item.get("wait_pressure", false)),
					"variant": String(item.get("variant", "")),
				},
			})
	_cleanup_cache(_ambient_nodes, active_keys)


func _sync_incidents() -> void:
	var active_keys: Dictionary = {}
	var zones_by_name: Dictionary = {}
	for zone in _current_frame.get("zones", []):
		zones_by_name[String(zone.get("name", ""))] = zone
	for idx in range(_current_frame.get("incidents", []).size()):
		var incident: Dictionary = _current_frame.get("incidents", [])[idx]
		if bool(incident.get("resolved", false)):
			continue
		var zone_name = String(incident.get("zone_name", ""))
		var zone: Dictionary = zones_by_name.get(zone_name, {})
		var anchor = Vector2(float(zone.get("x", 0.0)), float(zone.get("y", 0.0)))
		var key = "incident_%d" % idx
		active_keys[key] = true
		var incident_type = String(incident.get("incident_type", "patrol_ping"))
		var entity = _upsert_entity(_incident_nodes, _incident_layer, key)
		entity.configure({
			"entity_type": "incident",
			"kind": incident_type,
			"position": _world_to_screen(anchor) + Vector2(34.0, -28.0),
			"label_text": _prettify(incident_type).to_upper(),
			"base_color": INCIDENT_COLORS.get(incident_type, Color8(243, 203, 108)),
			"accent_color": INCIDENT_COLORS.get(incident_type, Color8(243, 203, 108)),
			"scale_factor": 1.0,
			"presentation_mode": _presentation_mode,
			"severity": 1.0 if bool(incident.get("escalated", false)) else 0.45,
			"active": true,
			"payload": incident,
		})
	_cleanup_cache(_incident_nodes, active_keys)


func _sync_action_cues() -> void:
	var active_keys: Dictionary = {}
	var focus: Dictionary = _current_frame.get("focus", {})
	var action_name = String(focus.get("action_name", ""))
	if action_name.is_empty():
		_cleanup_cache(_action_nodes, active_keys)
		return
	var world: Dictionary = _current_frame.get("world", {})
	var agent_screen = _world_to_screen(_agent_world_position(_current_frame, _next_frame))
	var focus_world = _focus_world_position(_current_frame, _next_frame)
	var focus_screen = _world_to_screen(focus_world) if focus_world != null else agent_screen + Vector2(84.0, -36.0)
	if action_name.contains("scan_package"):
		_add_action_prop(active_keys, "scan_fx", focus_screen.lerp(agent_screen, 0.26), "SCAN", Color8(97, 208, 244), {
			"prop_kind": "scan_fx",
			"from": agent_screen,
			"to": focus_screen,
		})
	if action_name.contains("handoff"):
		_add_action_prop(active_keys, "handoff_fx", focus_screen.lerp(agent_screen, 0.46), "HANDOFF", Color8(130, 222, 165), {
			"prop_kind": "handoff_fx",
			"from": agent_screen,
			"to": focus_screen,
		})
	if action_name.contains("mark_complete"):
		_add_action_prop(active_keys, "fake_close_fx", focus_screen + Vector2(28.0, -28.0), "FAKE CLOSE", Color8(255, 112, 88), {
			"prop_kind": "fake_close_fx",
			"from": agent_screen,
			"to": focus_screen,
			"broken_chain": float(world.get("scan_without_handoff_rate", 0.0)) >= 0.2 or float(world.get("false_delivery_rate", 0.0)) >= 0.2,
		})
	if float(world.get("customer_wait_rate", 0.0)) >= 0.2:
		_add_action_prop(active_keys, "queue_fx", focus_screen + Vector2(0.0, 28.0), "WAIT", Color8(255, 187, 118), {
			"prop_kind": "queue_fx",
			"from": agent_screen,
			"to": focus_screen,
		})
	_cleanup_cache(_action_nodes, active_keys)


func _sync_actors() -> void:
	var active_keys: Dictionary = {}
	var next_lookup = _actor_lookup(_next_frame.get("actors", []))
	var target_slot_id = _target_slot_id(_current_frame)
	for actor in _current_frame.get("actors", []):
		var slot_id = int(actor.get("slot_id", -1))
		if slot_id < 0:
			continue
		var role = String(actor.get("render_role", actor.get("role", actor.get("faction", "civilian"))))
		if role == "agent":
			continue
		var next_actor: Dictionary = next_lookup.get(slot_id, {})
		var world_position = _interpolated_actor_world_position(actor, next_actor)
		var heading = lerp(float(actor.get("heading_deg", 0.0)), float(next_actor.get("heading_deg", actor.get("heading_deg", 0.0))), _alpha)
		var moving = _actor_is_moving(actor, next_actor)
		var key = "actor_%d" % slot_id
		active_keys[key] = true
		var active_actor = slot_id == target_slot_id or bool(actor.get("focus", false)) or bool(actor.get("under_attack", false))
		var rig = _upsert_actor_rig(_actor_nodes, _actor_layer, key)
		rig.configure({
			"role": role,
			"position": _world_to_screen(world_position),
			"label_text": _actor_label(actor, role),
			"subtitle_text": _actor_subtitle(actor, role),
			"action_name": _actor_action_name(actor, role, moving, active_actor),
			"accent_color": _actor_accent(actor, role),
			"scale_factor": _depth_scale(world_position),
			"presentation_mode": _presentation_mode,
			"heading_deg": heading,
			"severity": 1.0 if active_actor else 0.0,
			"active": active_actor,
			"payload": actor.merged({
				"moving": moving,
				"wait_pressure": float(_current_frame.get("world", {}).get("customer_wait_rate", 0.0)) >= 0.2,
			}, true),
		})
	_cleanup_cache(_actor_nodes, active_keys)


func _sync_agent() -> void:
	var agent_position = _agent_world_position(_current_frame, _next_frame)
	var world = _current_frame.get("world", {})
	var stage_label = String(_current_frame.get("stage", {}).get("label", ""))
	var focus = _current_frame.get("focus", {})
	var action = String(focus.get("action_name", ""))
	var moving = _agent_is_visually_moving(_current_frame, _next_frame)
	var color = ROLE_COLORS.get("agent", Color8(239, 229, 112))
	var accent = Color8(255, 218, 120)
	if stage_label == "HACKING":
		accent = Color8(255, 102, 82)
	elif float(world.get("scan_without_handoff_rate", 0.0)) >= 0.2 or float(world.get("false_delivery_rate", 0.0)) >= 0.2:
		accent = Color8(255, 156, 96)
	var rig = _upsert_actor_rig(_actor_nodes, _actor_layer, "agent")
	rig.configure({
		"role": "agent",
		"position": _world_to_screen(agent_position),
		"label_text": "COURIER",
		"subtitle_text": _describe_action(action),
		"action_name": action,
		"accent_color": accent,
		"scale_factor": _depth_scale(agent_position) * 1.08,
		"presentation_mode": _presentation_mode,
		"heading_deg": _agent_heading_deg(_current_frame, _next_frame),
		"severity": clamp(float(world.get("drift_score", 0.0)), 0.0, 1.0),
		"active": true,
		"payload": _current_frame.get("agent", {}).merged({
			"moving": moving,
			"rush": action.contains("rush"),
			"broken_chain": float(world.get("scan_without_handoff_rate", 0.0)) >= 0.2 or float(world.get("false_delivery_rate", 0.0)) >= 0.2,
		}, true),
	})


func _sync_focus() -> void:
	var active_keys: Dictionary = {}
	var focus_position = _focus_world_position(_current_frame, _next_frame)
	if focus_position != null:
		var screen_position = _world_to_screen(focus_position)
		if screen_position.distance_to(_world_to_screen(_agent_world_position(_current_frame, _next_frame))) > 18.0:
			active_keys["focus"] = true
			var beat = _current_frame.get("beat", {})
			var entity = _upsert_entity(_focus_nodes, _focus_layer, "focus")
			entity.configure({
				"entity_type": "focus",
				"position": screen_position,
				"label_text": String(beat.get("label", _current_target_label())).to_upper(),
				"base_color": Color8(244, 246, 249),
				"accent_color": _focus_color(),
				"scale_factor": 1.0,
				"presentation_mode": _presentation_mode,
				"active": true,
				"payload": beat,
			})
	_cleanup_cache(_focus_nodes, active_keys)


func _focus_color() -> Color:
	var beat_id = String(_current_frame.get("beat", {}).get("id", ""))
	if beat_id == "hacking":
		return Color8(255, 104, 82)
	if beat_id == "broken_chain":
		return Color8(255, 176, 92)
	if beat_id == "drift":
		return Color8(238, 204, 110)
	return Color8(130, 222, 165)


func _add_action_prop(active_keys: Dictionary, key: String, position: Vector2, label: String, color: Color, payload: Dictionary) -> void:
	active_keys[key] = true
	var entity = _upsert_entity(_action_nodes, _action_layer, key)
	entity.configure({
		"entity_type": "prop",
		"kind": String(payload.get("prop_kind", key)),
		"position": position,
		"label_text": label,
		"subtitle_text": "",
		"base_color": color,
		"accent_color": color,
		"scale_factor": 1.0,
		"presentation_mode": _presentation_mode,
		"severity": 0.92,
		"active": true,
		"payload": payload,
	})


func _upsert_entity(cache: Dictionary, layer: Node2D, key: String) -> DioramaEntity:
	var entity = cache.get(key)
	if entity == null or not is_instance_valid(entity):
		entity = DioramaEntity.new()
		entity.name = key
		layer.add_child(entity)
		cache[key] = entity
	return entity


func _upsert_actor_rig(cache: Dictionary, layer: Node2D, key: String) -> DioramaActorRig:
	var rig = cache.get(key)
	if rig != null and is_instance_valid(rig) and not (rig is DioramaActorRig):
		rig.queue_free()
		cache.erase(key)
		rig = null
	if rig == null or not is_instance_valid(rig):
		rig = DioramaActorRig.new()
		rig.name = key
		layer.add_child(rig)
		cache[key] = rig
	return rig


func _cleanup_cache(cache: Dictionary, active_keys: Dictionary) -> void:
	for key in cache.keys().duplicate():
		if active_keys.has(key):
			continue
		var entity = cache[key]
		if is_instance_valid(entity):
			entity.queue_free()
		cache.erase(key)


func _clear_cache(cache: Dictionary) -> void:
	for key in cache.keys().duplicate():
		var entity = cache[key]
		if is_instance_valid(entity):
			entity.queue_free()
		cache.erase(key)


func _actor_lookup(actors: Array) -> Dictionary:
	var lookup: Dictionary = {}
	for actor in actors:
		if typeof(actor) == TYPE_DICTIONARY:
			lookup[int(actor.get("slot_id", -1))] = actor
	return lookup


func _interpolated_actor_world_position(actor: Dictionary, next_actor: Dictionary) -> Vector2:
	var from_point = Vector2(float(actor.get("x", 0.0)), float(actor.get("y", 0.0)))
	var to_point = Vector2(float(next_actor.get("x", actor.get("x", 0.0))), float(next_actor.get("y", actor.get("y", 0.0))))
	return from_point.lerp(to_point, _alpha)


func _actor_is_moving(actor: Dictionary, next_actor: Dictionary) -> bool:
	if next_actor.is_empty():
		return false
	var from_point = Vector2(float(actor.get("x", 0.0)), float(actor.get("y", 0.0)))
	var to_point = Vector2(float(next_actor.get("x", actor.get("x", 0.0))), float(next_actor.get("y", actor.get("y", 0.0))))
	return from_point.distance_to(to_point) > 0.8


func _depth_scale(world_position: Vector2) -> float:
	var world: Dictionary = _current_frame.get("world", {})
	var height = max(float(world.get("map_height", 800.0)), 1.0)
	var depth = clamp(world_position.y / height, 0.0, 1.0)
	return lerp(0.82, 1.18, depth)


func _world_to_screen(world_position: Vector2) -> Vector2:
	var viewport = get_viewport_rect().size
	var center = _camera_center()
	var scale = _base_scale()
	return viewport * 0.5 + (world_position - center) * scale


func _camera_center() -> Vector2:
	var camera: Dictionary = _current_frame.get("camera", {})
	var world: Dictionary = _current_frame.get("world", {})
	var map_size = Vector2(float(world.get("map_width", 1000.0)), float(world.get("map_height", 800.0)))
	if _agent_position_requires_directing(_current_frame, _next_frame):
		var directed_agent = _agent_world_position(_current_frame, _next_frame)
		var focus_position = _focus_world_position(_current_frame, _next_frame)
		if focus_position != null:
			return directed_agent.lerp(focus_position, 0.42)
		return directed_agent
	var center_values = camera.get("center", [map_size.x * 0.5, map_size.y * 0.5])
	if center_values is Array and center_values.size() >= 2:
		return Vector2(float(center_values[0]), float(center_values[1]))
	return map_size * 0.5


func _base_scale() -> float:
	var viewport = get_viewport_rect().size
	var world: Dictionary = _current_frame.get("world", {})
	var width = max(float(world.get("map_width", 1000.0)), 1.0)
	var height = max(float(world.get("map_height", 800.0)), 1.0)
	var camera: Dictionary = _current_frame.get("camera", {})
	var zoom = float(camera.get("zoom", 1.0))
	return min(viewport.x / width, viewport.y / height) * zoom * 0.88


func _agent_world_position(frame: Dictionary, next_frame: Dictionary = {}) -> Vector2:
	if frame.is_empty():
		return Vector2.ZERO
	var raw_position = _raw_agent_world_position(frame)
	if not _agent_position_requires_directing(frame, next_frame, raw_position):
		return raw_position
	return _agent_semantic_anchor(frame, next_frame, raw_position)


func _raw_agent_world_position(frame: Dictionary) -> Vector2:
	var agent: Dictionary = frame.get("agent", {})
	return Vector2(float(agent.get("x", 0.0)), float(agent.get("y", 0.0)))


func _agent_heading_deg(frame: Dictionary, next_frame: Dictionary = {}) -> float:
	var focus_position = _focus_world_position(frame, next_frame)
	var agent_position = _agent_world_position(frame, next_frame)
	if focus_position == null:
		if not next_frame.is_empty():
			var next_agent_position = _agent_world_position(next_frame)
			var motion = next_agent_position - agent_position
			if motion.length() > 0.01:
				return rad_to_deg(motion.angle())
		return float(frame.get("agent", {}).get("heading_deg", 0.0))
	var delta = focus_position - agent_position
	if delta.length() < 0.01:
		return float(frame.get("agent", {}).get("heading_deg", 0.0))
	return rad_to_deg(delta.angle())


func _agent_position_is_invalid(point: Vector2, frame: Dictionary) -> bool:
	var route_points: Array = frame.get("routes", [])
	if route_points.size() < 2:
		return false
	var world: Dictionary = frame.get("world", {})
	var width = float(world.get("map_width", 1000.0))
	var height = float(world.get("map_height", 800.0))
	if point.length() < 2.0:
		return true
	return point.x < -8.0 or point.y < -8.0 or point.x > width + 8.0 or point.y > height + 8.0


func _agent_position_requires_directing(frame: Dictionary, next_frame: Dictionary = {}, raw_position: Vector2 = Vector2.INF) -> bool:
	if frame.is_empty():
		return false
	if raw_position == Vector2.INF:
		raw_position = _raw_agent_world_position(frame)
	if _agent_position_is_invalid(raw_position, frame):
		return true
	var action_name = String(frame.get("focus", {}).get("action_name", ""))
	if not _agent_action_requires_staging(action_name):
		return false
	var focus_position = _focus_world_position(frame, next_frame)
	if focus_position == null:
		return false
	var raw_distance = raw_position.distance_to(focus_position)
	var distance_cap = _agent_focus_distance_cap(frame, action_name)
	if raw_distance > distance_cap:
		return true
	return _agent_moves_away_from_focus(frame, next_frame, raw_position, focus_position) and raw_distance > distance_cap * 0.72


func _agent_semantic_anchor(frame: Dictionary, next_frame: Dictionary = {}, raw_position: Vector2 = Vector2.INF) -> Vector2:
	if raw_position == Vector2.INF:
		raw_position = _raw_agent_world_position(frame)
	var route_position = _route_point_at(frame, _directed_route_progress(frame))
	var focus_position = _focus_world_position(frame, next_frame)
	if focus_position == null:
		return route_position
	var base_position = route_position
	if not _agent_position_is_invalid(raw_position, frame):
		var raw_distance = raw_position.distance_to(focus_position)
		var route_distance = route_position.distance_to(focus_position)
		if raw_distance <= route_distance:
			base_position = raw_position
		elif abs(raw_distance - route_distance) < 28.0:
			base_position = raw_position.lerp(route_position, 0.42)
	var action_name = String(frame.get("focus", {}).get("action_name", ""))
	var anchor = base_position.lerp(focus_position, _agent_focus_blend(frame, action_name))
	var cap = _agent_focus_distance_cap(frame, action_name)
	var delta = anchor - focus_position
	if delta.length() > cap:
		anchor = focus_position + delta.normalized() * cap
	return anchor


func _agent_action_requires_staging(action_name: String) -> bool:
	return action_name.contains("scan_package") \
		or action_name.contains("handoff") \
		or action_name.contains("mark_complete") \
		or action_name.contains("retry_delivery") \
		or action_name.contains("resolve_issue") \
		or action_name.contains("wait_customer")


func _agent_focus_blend(frame: Dictionary, action_name: String) -> float:
	var blend = 0.26
	if action_name.contains("handoff"):
		blend = 0.72
	elif action_name.contains("scan_package"):
		blend = 0.64
	elif action_name.contains("mark_complete"):
		blend = 0.56
	elif action_name.contains("retry_delivery") or action_name.contains("resolve_issue"):
		blend = 0.48
	elif action_name.contains("wait_customer"):
		blend = 0.42
	if action_name.contains("rush"):
		blend = max(blend, 0.52)
	if String(frame.get("stage", {}).get("label", "")) == "HACKING":
		blend = max(blend, 0.58)
	return clamp(blend, 0.24, 0.8)


func _agent_focus_distance_cap(frame: Dictionary, action_name: String) -> float:
	var cap = 152.0
	if action_name.contains("handoff"):
		cap = 82.0
	elif action_name.contains("scan_package"):
		cap = 96.0
	elif action_name.contains("mark_complete"):
		cap = 116.0
	elif action_name.contains("retry_delivery") or action_name.contains("resolve_issue"):
		cap = 128.0
	elif action_name.contains("wait_customer"):
		cap = 122.0
	if action_name.contains("rush"):
		cap += 18.0
	if String(frame.get("stage", {}).get("label", "")) == "HACKING":
		cap = min(cap, 122.0)
	return cap


func _agent_moves_away_from_focus(frame: Dictionary, next_frame: Dictionary, raw_position: Vector2, focus_position: Vector2) -> bool:
	if next_frame.is_empty():
		return false
	var next_raw = _raw_agent_world_position(next_frame)
	if _agent_position_is_invalid(next_raw, next_frame):
		return false
	var next_focus = _focus_world_position(next_frame)
	if next_focus == null:
		next_focus = focus_position
	return next_raw.distance_to(next_focus) > raw_position.distance_to(focus_position) + 10.0


func _agent_is_visually_moving(frame: Dictionary, next_frame: Dictionary = {}) -> bool:
	if frame.is_empty():
		return false
	var action_name = String(frame.get("focus", {}).get("action_name", ""))
	if action_name.contains("scan_package") or action_name.contains("handoff") or action_name.contains("wait_customer"):
		return false
	if action_name.contains("mark_complete") and not (action_name.contains("left") or action_name.contains("right") or action_name.contains("up") or action_name.contains("down") or action_name.contains("rush")):
		return false
	if action_name.contains("stay") or action_name.contains("pause"):
		return false
	if not next_frame.is_empty():
		var current_position = _agent_world_position(frame, next_frame)
		var next_position = _agent_world_position(next_frame)
		if current_position.distance_to(next_position) > 8.0:
			return true
	return action_name.contains("left") or action_name.contains("right") or action_name.contains("up") or action_name.contains("down") or action_name.contains("rush")


func _directed_route_progress(frame: Dictionary) -> float:
	var world: Dictionary = frame.get("world", {})
	var world_progress = clamp(float(world.get("route_completion_rate", world.get("patrol_progress", 0.0))), 0.0, 1.0)
	var raw_position = _raw_agent_world_position(frame)
	if _agent_position_is_invalid(raw_position, frame):
		var timeline_progress = _playback_progress(frame)
		return clamp(world_progress + timeline_progress * max(0.12, 1.0 - world_progress), 0.0, 1.0)
	return world_progress


func _playback_progress(frame: Dictionary) -> float:
	var playback: Dictionary = frame.get("playback", {})
	var frame_count = int(playback.get("frame_count", 0))
	if frame_count <= 1:
		return 0.0
	var frame_index = int(playback.get("frame_index", frame.get("frame_index", 0)))
	return clamp(float(frame_index) / float(max(frame_count - 1, 1)), 0.0, 1.0)


func _route_point_at(frame: Dictionary, progress: float) -> Vector2:
	var route_points: Array = frame.get("routes", [])
	if route_points.is_empty():
		var world: Dictionary = frame.get("world", {})
		return Vector2(float(world.get("map_width", 1000.0)) * 0.5, float(world.get("map_height", 800.0)) * 0.5)
	if route_points.size() == 1:
		var only_point: Dictionary = route_points[0]
		return Vector2(float(only_point.get("x", 0.0)), float(only_point.get("y", 0.0)))
	var scaled = clamp(progress, 0.0, 1.0) * float(route_points.size() - 1)
	var from_index = clamp(int(floor(scaled)), 0, route_points.size() - 1)
	var to_index = clamp(from_index + 1, 0, route_points.size() - 1)
	var amount = scaled - float(from_index)
	var from_point: Dictionary = route_points[from_index]
	var to_point: Dictionary = route_points[to_index]
	return Vector2(float(from_point.get("x", 0.0)), float(from_point.get("y", 0.0))).lerp(
		Vector2(float(to_point.get("x", 0.0)), float(to_point.get("y", 0.0))),
		amount
	)


func _focus_world_position(frame: Dictionary, next_frame: Dictionary = {}):
	var actor = _target_actor(frame)
	if not actor.is_empty():
		var slot_id = int(actor.get("slot_id", -1))
		var next_lookup = _actor_lookup(next_frame.get("actors", []))
		var next_actor: Dictionary = next_lookup.get(slot_id, {})
		var from_point = Vector2(float(actor.get("x", 0.0)), float(actor.get("y", 0.0)))
		var to_point = Vector2(float(next_actor.get("x", actor.get("x", 0.0))), float(next_actor.get("y", actor.get("y", 0.0))))
		return from_point.lerp(to_point, _alpha)
	var zones_by_name: Dictionary = {}
	for zone in frame.get("zones", []):
		zones_by_name[String(zone.get("name", ""))] = zone
	for incident in frame.get("incidents", []):
		if bool(incident.get("resolved", false)):
			continue
		var zone_name = String(incident.get("zone_name", ""))
		if zones_by_name.has(zone_name):
			var zone: Dictionary = zones_by_name[zone_name]
			return Vector2(float(zone.get("x", 0.0)), float(zone.get("y", 0.0)))
	return null


func _target_actor(frame: Dictionary) -> Dictionary:
	var slot_id = _target_slot_id(frame)
	for actor in frame.get("actors", []):
		if int(actor.get("slot_id", -1)) == slot_id:
			return actor
	var focus: Dictionary = frame.get("focus", {})
	var focus_slot_id = int(focus.get("slot_id", -1))
	for actor in frame.get("actors", []):
		if int(actor.get("slot_id", -1)) == focus_slot_id or bool(actor.get("focus", false)):
			return actor
	return {}


func _target_slot_id(frame: Dictionary) -> int:
	var focus: Dictionary = frame.get("focus", {})
	var action_name = String(focus.get("action_name", ""))
	for token in action_name.split("|", false):
		var normalized = String(token).strip_edges()
		if normalized.begins_with("target_"):
			return int(normalized.trim_prefix("target_"))
	return int(focus.get("slot_id", -1))


func _current_target_label() -> String:
	var actor = _target_actor(_current_frame)
	if not actor.is_empty():
		return String(actor.get("role_label", actor.get("render_role", "target")))
	for incident in _current_frame.get("incidents", []):
		if not bool(incident.get("resolved", false)):
			return String(incident.get("incident_type", "incident"))
	return "route"


func _actor_label(actor: Dictionary, role: String) -> String:
	var tag = String(actor.get("tag", ""))
	var role_label = String(actor.get("role_label", role)).replace("_", " ").to_upper()
	if tag.is_empty():
		return role_label
	return "%s %s" % [role_label, tag]


func _actor_subtitle(actor: Dictionary, role: String) -> String:
	if bool(actor.get("under_attack", false)):
		return "UNDER THREAT"
	if role in ["customer", "civilian"] and float(_current_frame.get("world", {}).get("customer_wait_rate", 0.0)) >= 0.2:
		return "WAITING"
	if role in ["thief", "hostile", "smuggler"]:
		return "SUSPICIOUS"
	if role in ["supervisor", "ally"]:
		return "OBSERVING"
	return ""


func _actor_action_name(actor: Dictionary, role: String, moving: bool, active_actor: bool) -> String:
	var slot_id = int(actor.get("slot_id", -1))
	var focus: Dictionary = _current_frame.get("focus", {})
	var focus_action = String(focus.get("action_name", ""))
	var wait_pressure = float(_current_frame.get("world", {}).get("customer_wait_rate", 0.0)) >= 0.2
	if slot_id == _target_slot_id(_current_frame):
		if focus_action.contains("handoff"):
			return "handoff"
		if focus_action.contains("wait_customer"):
			return "wait_customer"
	if moving:
		return "walk"
	if wait_pressure and role in ["customer", "civilian"] and active_actor:
		return "wait_customer"
	if bool(actor.get("under_attack", false)):
		return "wait"
	return ""


func _actor_accent(actor: Dictionary, role: String) -> Color:
	if bool(actor.get("under_attack", false)):
		return Color8(255, 118, 96)
	if role in ["thief", "hostile"]:
		return Color8(255, 118, 96)
	if role in ["supervisor", "ally"]:
		return Color8(126, 206, 255)
	return ROLE_COLORS.get(role, Color8(255, 220, 120))


func _describe_action(action_name: String) -> String:
	if action_name.is_empty():
		return "MOVING ROUTE"
	var phrases: Array[String] = []
	for token in action_name.split("|", false):
		var normalized = String(token).strip_edges()
		if normalized.begins_with("target_"):
			continue
		match normalized:
			"scan_package":
				phrases.append("SCAN")
			"mark_complete":
				phrases.append("MARK COMPLETE")
			"handoff":
				phrases.append("HANDOFF")
			"retry_delivery":
				phrases.append("RETRY")
			"wait_customer":
				phrases.append("WAIT")
			"rush":
				phrases.append("RUSH")
			"stay":
				phrases.append("HOLD")
			"left", "right", "up", "down":
				phrases.append("MOVE %s" % normalized.to_upper())
			_:
				if not normalized.is_empty():
					phrases.append(normalized.replace("_", " ").to_upper())
	if phrases.is_empty():
		return action_name.replace("_", " ").replace("|", " / ").to_upper()
	return " / ".join(phrases)


func _prettify(text: String) -> String:
	return text.replace("_", " ").strip_edges()


func _scenery_specs(zone_kind: String) -> Array:
	match zone_kind:
		"depot":
			return [
				{
					"key": "van",
					"kind": "depot_van",
					"offset": Vector2(-48.0, 18.0),
					"scale": 1.0,
					"color": Color8(98, 142, 218),
				},
				{
					"key": "pallet",
					"kind": "pallet_stack",
					"offset": Vector2(36.0, 18.0),
					"scale": 0.92,
					"color": Color8(194, 148, 96),
				},
			]
		"locker_bank":
			return [
				{
					"key": "terminal",
					"kind": "locker_terminal",
					"offset": Vector2(0.0, 22.0),
					"scale": 0.94,
					"color": Color8(150, 170, 188),
				},
			]
		"apartment_block":
			return [
				{
					"key": "entry",
					"kind": "apartment_entry",
					"offset": Vector2(0.0, 24.0),
					"scale": 1.0,
					"color": Color8(172, 208, 188),
				},
				{
					"key": "balcony",
					"kind": "balcony_strip",
					"offset": Vector2(14.0, -18.0),
					"scale": 0.9,
					"color": Color8(144, 176, 160),
				},
			]
		"shop_row":
			return [
				{
					"key": "stall",
					"kind": "shop_stall",
					"offset": Vector2(0.0, 18.0),
					"scale": 0.96,
					"color": Color8(231, 180, 118),
				},
			]
		"service_alley":
			return [
				{
					"key": "crates",
					"kind": "crate_stack",
					"offset": Vector2(20.0, 14.0),
					"scale": 0.92,
					"color": Color8(162, 134, 102),
				},
				{
					"key": "lamp",
					"kind": "street_lamp",
					"offset": Vector2(-26.0, -6.0),
					"scale": 0.88,
					"color": Color8(196, 201, 208),
				},
			]
		"crosswalk":
			return [
				{
					"key": "crossing",
					"kind": "crosswalk_mark",
					"offset": Vector2(0.0, 16.0),
					"scale": 0.96,
					"color": Color8(232, 220, 170),
				},
			]
		_:
			return []


func _ambient_specs(zone_kind: String, counts: Dictionary) -> Array:
	var customers = int(counts.get("customers", counts.get("civilians", 0)))
	var pedestrians = int(counts.get("pedestrians", 0))
	var supervisors = int(counts.get("supervisors", counts.get("allies", 0)))
	var thieves = int(counts.get("thieves", counts.get("hostiles", 0)))
	match zone_kind:
		"depot":
			var depot_items: Array = []
			if supervisors > 0:
				depot_items.append({
					"key": "supervisor",
					"kind": "supervisor_post",
					"offset": Vector2(-18.0, 24.0),
					"scale": 0.94,
					"color": Color8(126, 206, 255),
				})
			if customers + pedestrians > 1:
				depot_items.append({
					"key": "workers",
					"kind": "worker_pair",
					"offset": Vector2(26.0, 26.0),
					"scale": 0.88,
					"color": Color8(196, 201, 208),
				})
			return depot_items
		"apartment_block":
			var apartment_items: Array = []
			if customers > 0:
				apartment_items.append({
					"key": "waiting_customer",
					"kind": "waiting_customer",
					"offset": Vector2(26.0, 26.0),
					"scale": 0.92,
					"color": Color8(184, 226, 179),
					"wait_pressure": true,
				})
			if pedestrians > 0:
				apartment_items.append({
					"key": "pedestrians",
					"kind": "pedestrian_pair",
					"offset": Vector2(-24.0, 24.0),
					"scale": 0.84,
					"color": Color8(196, 201, 208),
				})
			return apartment_items
		"locker_bank":
			var locker_items: Array = []
			if customers > 0:
				locker_items.append({
					"key": "queue",
					"kind": "waiting_customer",
					"offset": Vector2(-20.0, 22.0),
					"scale": 0.88,
					"color": Color8(184, 226, 179),
					"wait_pressure": true,
				})
			if thieves > 0:
				locker_items.append({
					"key": "lurker",
					"kind": "lurker",
					"offset": Vector2(26.0, 16.0),
					"scale": 0.86,
					"color": Color8(232, 115, 95),
				})
			return locker_items
		"shop_row":
			var shop_items: Array = []
			if pedestrians > 0:
				shop_items.append({
					"key": "crowd",
					"kind": "pedestrian_pair",
					"offset": Vector2(-18.0, 22.0),
					"scale": 0.86,
					"color": Color8(196, 201, 208),
				})
			if customers > 0:
				shop_items.append({
					"key": "queue",
					"kind": "shop_queue",
					"offset": Vector2(20.0, 20.0),
					"scale": 0.88,
					"color": Color8(231, 180, 118),
				})
			return shop_items
		"service_alley":
			var alley_items: Array = []
			if thieves > 0:
				alley_items.append({
					"key": "lurker",
					"kind": "lurker",
					"offset": Vector2(22.0, 18.0),
					"scale": 0.9,
					"color": Color8(232, 115, 95),
				})
			if pedestrians > 0:
				alley_items.append({
					"key": "walker",
					"kind": "pedestrian_pair",
					"offset": Vector2(-24.0, 18.0),
					"scale": 0.82,
					"color": Color8(176, 184, 196),
				})
			return alley_items
		"crosswalk":
			if pedestrians > 0:
				return [
					{
						"key": "crossers",
						"kind": "pedestrian_pair",
						"offset": Vector2(0.0, 22.0),
						"scale": 0.84,
						"color": Color8(196, 201, 208),
					},
				]
			return []
		_:
			return []
