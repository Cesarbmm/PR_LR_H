extends Node2D


const ROLE_COLORS = {
	"agent": Color8(239, 229, 112),
	"civilian": Color8(184, 226, 179),
	"customer": Color8(184, 226, 179),
	"supervisor": Color8(111, 175, 236),
	"pedestrian": Color8(196, 201, 208),
	"thief": Color8(232, 115, 95),
	"rival_courier": Color8(238, 184, 96),
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
var _overlay_mode: String = "broadcast"


func set_story_frame(frame: Dictionary, next_frame: Dictionary = {}, alpha: float = 0.0, overlay_mode: String = "broadcast") -> void:
	_current_frame = frame
	_next_frame = next_frame
	_alpha = clamp(alpha, 0.0, 1.0)
	_overlay_mode = overlay_mode
	queue_redraw()


func _draw() -> void:
	if _current_frame.is_empty():
		return
	var viewport = get_viewport_rect().size
	draw_rect(Rect2(Vector2.ZERO, viewport), Color8(17, 20, 28), true)
	_draw_world_decay(viewport)
	_draw_grid(viewport)
	_draw_route_ribbon()
	_draw_routes()
	_draw_zones()
	_draw_incidents()
	_draw_attack_events()
	_draw_actors()
	_draw_agent()
	_draw_focus_marker()
	_draw_stage_strip()


func _draw_world_decay(viewport: Vector2) -> void:
	var world: Dictionary = _current_frame.get("world", {})
	var neglected = float(world.get("neglected_incident_count", 0.0))
	var gap = float(world.get("gap", 0.0))
	var intensity = clamp(neglected / 4.0 + gap / 2200.0, 0.0, 0.5)
	if intensity <= 0.0:
		return
	draw_rect(Rect2(Vector2.ZERO, viewport), Color(0.34, 0.08, 0.06, intensity * 0.35), true)


func _draw_grid(viewport: Vector2) -> void:
	var minor_color = Color(1.0, 1.0, 1.0, 0.035)
	var major_color = Color(1.0, 1.0, 1.0, 0.06)
	var columns = 8
	var rows = 5
	for column in range(columns + 1):
		var x = viewport.x * float(column) / float(columns)
		draw_line(Vector2(x, 0.0), Vector2(x, viewport.y), major_color if column in [0, columns] else minor_color, 1.0, true)
	for row in range(rows + 1):
		var y = viewport.y * float(row) / float(rows)
		draw_line(Vector2(0.0, y), Vector2(viewport.x, y), major_color if row in [0, rows] else minor_color, 1.0, true)


func _draw_route_ribbon() -> void:
	var route_points: Array = _current_frame.get("routes", [])
	if route_points.size() < 2:
		return
	var screen_points = PackedVector2Array()
	for point in route_points:
		screen_points.append(_world_to_screen(Vector2(float(point.get("x", 0.0)), float(point.get("y", 0.0)))))
	draw_polyline(screen_points, Color8(57, 66, 77, 170), 18.0, true)
	draw_polyline(screen_points, Color8(81, 92, 108, 90), 10.0, true)


func _draw_routes() -> void:
	var route_points: Array = _current_frame.get("routes", [])
	if route_points.size() < 2:
		return
	var screen_points = PackedVector2Array()
	for point in route_points:
		screen_points.append(_world_to_screen(Vector2(float(point.get("x", 0.0)), float(point.get("y", 0.0)))))
	draw_polyline(screen_points, Color8(246, 216, 122, 210), 6.0, true)
	draw_polyline(screen_points, Color8(255, 249, 224, 120), 2.0, true)


func _draw_zones() -> void:
	for zone in _current_frame.get("zones", []):
		var position = _world_to_screen(Vector2(float(zone.get("x", 0.0)), float(zone.get("y", 0.0))))
		var radius = max(26.0, float(zone.get("radius", 80.0)) * _base_scale() * 0.12)
		var color = ZONE_COLORS.get(String(zone.get("kind", "safehouse")), Color8(116, 154, 164))
		draw_circle(position, radius, Color(color.r, color.g, color.b, 0.16))
		draw_arc(position, radius, 0.0, TAU, 48, Color(color.r, color.g, color.b, 0.85), 2.0, true)
		_draw_zone_landmark(String(zone.get("kind", "")), position, color)
		_draw_label(position + Vector2(0, radius + 18), String(zone.get("name", "")).to_upper(), Color8(233, 236, 242), 14, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_zone_landmark(kind: String, position: Vector2, color: Color) -> void:
	match kind:
		"depot":
			draw_rect(Rect2(position + Vector2(-20, -14), Vector2(40, 28)), Color(color.r, color.g, color.b, 0.9), true)
			draw_rect(Rect2(position + Vector2(-14, 2), Vector2(28, 10)), Color8(28, 33, 42), true)
			draw_line(position + Vector2(-16, -4), position + Vector2(16, -4), Color8(246, 246, 232, 120), 2.0, true)
		"apartment_block":
			draw_rect(Rect2(position + Vector2(-18, -18), Vector2(16, 36)), Color(color.r, color.g, color.b, 0.9), true)
			draw_rect(Rect2(position + Vector2(4, -14), Vector2(18, 28)), Color(color.r, color.g, color.b, 0.82), true)
		"shop_row":
			draw_rect(Rect2(position + Vector2(-22, -8), Vector2(44, 18)), Color(color.r, color.g, color.b, 0.86), true)
			draw_colored_polygon(PackedVector2Array([position + Vector2(-24, -8), position + Vector2(0, -20), position + Vector2(24, -8)]), Color(color.r, color.g, color.b, 0.95))
		"locker_bank":
			for idx in range(3):
				draw_rect(Rect2(position + Vector2(-18 + idx * 12, -16), Vector2(10, 32)), Color(color.r, color.g, color.b, 0.88), true)
		"crosswalk":
			for idx in range(5):
				draw_rect(Rect2(position + Vector2(-22 + idx * 10, -10), Vector2(6, 20)), Color(color.r, color.g, color.b, 0.9), true)
		"service_alley":
			draw_polyline(PackedVector2Array([position + Vector2(-22, -8), position + Vector2(-6, 14), position + Vector2(22, -12)]), Color(color.r, color.g, color.b, 0.95), 6.0, true)
		"safehouse":
			draw_rect(Rect2(position + Vector2(-14, -10), Vector2(28, 20)), Color(color.r, color.g, color.b, 0.9), true)
			draw_rect(Rect2(position + Vector2(-5, -3), Vector2(10, 13)), Color8(28, 33, 42), true)
		"village":
			draw_rect(Rect2(position + Vector2(-18, -8), Vector2(12, 16)), Color(color.r, color.g, color.b, 0.9), true)
			draw_rect(Rect2(position + Vector2(2, -10), Vector2(16, 20)), Color(color.r, color.g, color.b, 0.82), true)
		"checkpoint":
			draw_line(position + Vector2(-18, 0), position + Vector2(18, 0), Color(color.r, color.g, color.b, 0.95), 4.0, true)
			draw_line(position + Vector2(-14, -12), position + Vector2(-14, 12), Color(color.r, color.g, color.b, 0.95), 2.0, true)
			draw_line(position + Vector2(14, -12), position + Vector2(14, 12), Color(color.r, color.g, color.b, 0.95), 2.0, true)
		"ruins":
			var ruin = PackedVector2Array([position + Vector2(-16, 10), position + Vector2(-8, -12), position + Vector2(6, -3), position + Vector2(16, -15), position + Vector2(10, 11)])
			draw_colored_polygon(ruin, Color(color.r, color.g, color.b, 0.85))
		"clinic":
			draw_rect(Rect2(position + Vector2(-6, -16), Vector2(12, 32)), Color(color.r, color.g, color.b, 0.92), true)
			draw_rect(Rect2(position + Vector2(-16, -6), Vector2(32, 12)), Color(color.r, color.g, color.b, 0.92), true)
		"watchtower":
			draw_line(position + Vector2(0, -18), position + Vector2(0, 16), Color(color.r, color.g, color.b, 0.95), 4.0, true)
			draw_rect(Rect2(position + Vector2(-12, -24), Vector2(24, 10)), Color(color.r, color.g, color.b, 0.95), true)
		"market_square":
			draw_rect(Rect2(position + Vector2(-18, -6), Vector2(36, 16)), Color(color.r, color.g, color.b, 0.85), true)
			draw_colored_polygon(PackedVector2Array([position + Vector2(-20, -6), position + Vector2(0, -20), position + Vector2(20, -6)]), Color(color.r, color.g, color.b, 0.95))
		"bridge_crossing":
			draw_line(position + Vector2(-18, -8), position + Vector2(18, -8), Color(color.r, color.g, color.b, 0.9), 3.0, true)
			draw_line(position + Vector2(-18, 8), position + Vector2(18, 8), Color(color.r, color.g, color.b, 0.9), 3.0, true)
			draw_line(position + Vector2(-12, -8), position + Vector2(-6, 8), Color(color.r, color.g, color.b, 0.9), 2.0, true)
			draw_line(position + Vector2(6, -8), position + Vector2(12, 8), Color(color.r, color.g, color.b, 0.9), 2.0, true)
		_:
			draw_circle(position, 8.0, Color(color.r, color.g, color.b, 0.95))


func _draw_incidents() -> void:
	var zones_by_name: Dictionary = {}
	for zone in _current_frame.get("zones", []):
		zones_by_name[String(zone.get("name", ""))] = zone
	for incident in _current_frame.get("incidents", []):
		if bool(incident.get("resolved", false)):
			continue
		var zone_name = String(incident.get("zone_name", ""))
		var zone: Dictionary = zones_by_name.get(zone_name, {})
		var anchor = Vector2(float(zone.get("x", 0.0)), float(zone.get("y", 0.0)))
		var base = _world_to_screen(anchor) + Vector2(34, -28)
		var incident_type = String(incident.get("incident_type", "patrol_ping"))
		var color = INCIDENT_COLORS.get(incident_type, Color8(243, 203, 108))
		var icon = _incident_shape(incident_type, base, 12.0)
		draw_colored_polygon(icon, color)
		draw_polyline(icon, Color8(27, 28, 36), 2.0, true)
		var marker = "!"
		if incident_type == "false_alarm":
			marker = "?"
		elif incident_type in ["aid_drop", "escort_request"]:
			marker = "+"
		_draw_label(base + Vector2(0, 4), marker, Color8(16, 17, 22), 16, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_actors() -> void:
	var next_lookup: Dictionary = _actor_lookup(_next_frame.get("actors", []))
	for actor in _current_frame.get("actors", []):
		var slot_id = int(actor.get("slot_id", -1))
		var next_actor: Dictionary = next_lookup.get(slot_id, {})
		var position = _interpolated_actor_position(actor, next_actor)
		var role = String(actor.get("render_role", actor.get("role", actor.get("faction", "civilian"))))
		var color = ROLE_COLORS.get(role, Color8(180, 185, 190))
		var heading = lerp(float(actor.get("heading_deg", 0.0)), float(next_actor.get("heading_deg", actor.get("heading_deg", 0.0))), _alpha)
		var shape = _transform_shape(_role_shape(role, 11.0), position, heading)
		draw_colored_polygon(shape, color)
		draw_polyline(shape, Color8(20, 24, 31), 2.0, true)
		if bool(actor.get("armed", false)):
			var muzzle = position + Vector2.from_angle(deg_to_rad(heading)) * 12.0
			draw_line(position, muzzle, Color8(249, 243, 230, 180), 2.0, true)
		if bool(actor.get("carrying_supply", false)):
			draw_rect(Rect2(position + Vector2(-6, 12), Vector2(12, 8)), Color8(210, 190, 104), true)
		_draw_actor_badge(actor, position)


func _draw_attack_events() -> void:
	var events: Dictionary = _current_frame.get("events", {})
	for combat_event in events.get("combat", []):
		var victim_screen = _world_to_screen(Vector2(float(combat_event.get("victim_x", 0.0)), float(combat_event.get("victim_y", 0.0))))
		var attacker_x = combat_event.get("attacker_x")
		var attacker_y = combat_event.get("attacker_y")
		var killed = bool(combat_event.get("killed", false))
		if attacker_x != null and attacker_y != null:
			var attacker_screen = _world_to_screen(Vector2(float(attacker_x), float(attacker_y)))
			draw_line(attacker_screen, victim_screen, Color8(246, 122, 96, 210), 3.0, true)
			draw_circle(attacker_screen, 6.0, Color8(246, 122, 96, 150))
			var direction = (victim_screen - attacker_screen).normalized()
			var flash_tip = attacker_screen + direction * 16.0
			draw_circle(flash_tip, 5.0, Color8(255, 230, 176, 220))
		draw_arc(victim_screen, 16.0 if killed else 12.0, 0.0, TAU, 32, Color8(255, 214, 188, 210), 2.0, true)
		draw_circle(victim_screen, 4.0, Color8(255, 232, 209, 220))
		draw_arc(victim_screen, 22.0, -0.7, 0.7, 14, Color8(255, 144, 121, 160), 2.0, true)
		if killed:
			draw_line(victim_screen + Vector2(-10, -10), victim_screen + Vector2(10, 10), Color8(255, 238, 224, 230), 3.0, true)
			draw_line(victim_screen + Vector2(-10, 10), victim_screen + Vector2(10, -10), Color8(255, 238, 224, 230), 3.0, true)
			if _overlay_mode == "broadcast":
				_draw_label(victim_screen + Vector2(0, -16), "DOWN", Color8(255, 236, 220), 13, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_agent() -> void:
	var agent: Dictionary = _current_frame.get("agent", {})
	if agent.is_empty():
		return
	var next_agent: Dictionary = _next_frame.get("agent", {})
	var position = _interpolate_point(
		Vector2(float(agent.get("x", 0.0)), float(agent.get("y", 0.0))),
		Vector2(float(next_agent.get("x", agent.get("x", 0.0))), float(next_agent.get("y", agent.get("y", 0.0)))),
	)
	var screen_position = _world_to_screen(position)
	var heading = lerp(float(agent.get("heading_deg", 0.0)), float(next_agent.get("heading_deg", agent.get("heading_deg", 0.0))), _alpha)
	var shield = _transform_shape(_role_shape("agent", 14.0), screen_position, heading)
	draw_colored_polygon(shield, ROLE_COLORS["agent"])
	draw_polyline(shield, Color8(18, 20, 26), 2.0, true)
	draw_arc(screen_position, 18.0, 0.0, TAU, 42, Color8(253, 249, 225, 140), 1.5, true)
	if bool(agent.get("carrying_supply", false)):
		draw_rect(Rect2(screen_position + Vector2(-8, 16), Vector2(16, 10)), Color8(214, 193, 112), true)


func _draw_focus_marker() -> void:
	var focus: Dictionary = _current_frame.get("focus", {})
	if focus.is_empty():
		return
	var slot_id = int(focus.get("slot_id", -1))
	var next_lookup: Dictionary = _actor_lookup(_next_frame.get("actors", []))
	for actor in _current_frame.get("actors", []):
		if int(actor.get("slot_id", -1)) != slot_id:
			continue
		var screen_position = _interpolated_actor_position(actor, next_lookup.get(slot_id, {}))
		var pulse = 20.0 + sin(Time.get_ticks_msec() / 220.0) * 3.0
		draw_arc(screen_position, pulse, 0.0, TAU, 48, Color8(255, 242, 180, 180), 2.0, true)
		draw_rect(Rect2(screen_position + Vector2(-24, -24), Vector2(48, 48)), Color8(255, 242, 180, 90), false, 2.0)
		_draw_label(screen_position + Vector2(0, -30), String(focus.get("tag", "FOCUS")), Color8(255, 247, 216), 14, HORIZONTAL_ALIGNMENT_CENTER)
		return


func _draw_stage_strip() -> void:
	if _overlay_mode == "minimal":
		return
	var stage: Dictionary = _current_frame.get("stage", {})
	var label = String(stage.get("label", "PATROLING"))
	var tone = String(stage.get("tone", "neutral"))
	var tone_color = Color8(109, 180, 238)
	if tone == "alert":
		tone_color = Color8(236, 173, 93)
	elif tone == "exploit":
		tone_color = Color8(234, 114, 98)
	var strip_rect = Rect2(Vector2(24, 24), Vector2(240, 54))
	draw_rect(strip_rect, Color(0.05, 0.06, 0.08, 0.82), true)
	draw_rect(Rect2(strip_rect.position, Vector2(8, strip_rect.size.y)), tone_color, true)
	_draw_label(strip_rect.position + Vector2(24, 24), "STAGE", Color8(184, 191, 202), 12)
	_draw_label(strip_rect.position + Vector2(24, 44), label, Color8(244, 246, 250), 20)


func _draw_actor_badge(actor: Dictionary, position: Vector2) -> void:
	var threat_class = String(actor.get("threat_class", "civilian"))
	var role_label = String(actor.get("role_label", ""))
	var recent_damage = float(actor.get("recent_damage", 0.0))
	var under_attack = bool(actor.get("under_attack", false))
	var render_role = String(actor.get("render_role", actor.get("role", "")))
	if render_role in ["customer", "supervisor", "pedestrian", "thief", "rival_courier"]:
		var badge_color = ROLE_COLORS.get(render_role, Color8(220, 220, 220))
		draw_arc(position, 14.0, 0.0, TAU, 28, Color(badge_color.r, badge_color.g, badge_color.b, 0.90), 2.0, true)
		if render_role == "thief":
			draw_rect(Rect2(position + Vector2(-14, -14), Vector2(28, 28)), Color8(255, 236, 228, 80), false, 2.0)
		if _overlay_mode == "broadcast":
			_draw_label(position + Vector2(0, -18), role_label, Color8(244, 246, 250), 12, HORIZONTAL_ALIGNMENT_CENTER)
		if recent_damage > 0.0:
			draw_arc(position, 18.0, 0.0, TAU, 30, Color8(255, 230, 215, 220), 1.5, true)
		return
	if threat_class == "hostile":
		draw_arc(position, 15.0, 0.0, TAU, 28, Color8(255, 124, 96, 220), 2.0, true)
		draw_rect(Rect2(position + Vector2(-16, -16), Vector2(32, 32)), Color8(255, 124, 96, 100), false, 2.0)
		if _overlay_mode == "broadcast":
			_draw_label(position + Vector2(0, -20), role_label, Color8(255, 234, 226), 12, HORIZONTAL_ALIGNMENT_CENTER)
	elif threat_class == "armed_neutral":
		draw_arc(position, 14.0, 0.0, TAU, 28, Color8(245, 199, 114, 210), 2.0, true)
		if _overlay_mode == "broadcast":
			_draw_label(position + Vector2(0, -18), role_label, Color8(255, 246, 214), 12, HORIZONTAL_ALIGNMENT_CENTER)
	elif under_attack:
		draw_arc(position, 13.0, 0.0, TAU, 28, Color8(255, 188, 122, 210), 2.0, true)
	if recent_damage > 0.0:
		draw_arc(position, 18.0, 0.0, TAU, 30, Color8(255, 230, 215, 220), 1.5, true)
		draw_line(position + Vector2(-8, 0), position + Vector2(8, 0), Color8(255, 235, 220, 180), 1.5, true)
		draw_line(position + Vector2(0, -8), position + Vector2(0, 8), Color8(255, 235, 220, 180), 1.5, true)


func _actor_lookup(actors: Array) -> Dictionary:
	var lookup: Dictionary = {}
	for actor in actors:
		if typeof(actor) == TYPE_DICTIONARY:
			lookup[int(actor.get("slot_id", -1))] = actor
	return lookup


func _interpolated_actor_position(actor: Dictionary, next_actor: Dictionary) -> Vector2:
	var from_point = Vector2(float(actor.get("x", 0.0)), float(actor.get("y", 0.0)))
	var to_point = Vector2(float(next_actor.get("x", actor.get("x", 0.0))), float(next_actor.get("y", actor.get("y", 0.0))))
	return _world_to_screen(_interpolate_point(from_point, to_point))


func _interpolate_point(from_point: Vector2, to_point: Vector2) -> Vector2:
	return from_point.lerp(to_point, _alpha)


func _world_to_screen(world_position: Vector2) -> Vector2:
	var viewport = get_viewport_rect().size
	var center = _camera_center()
	var scale = _base_scale()
	return viewport * 0.5 + (world_position - center) * scale


func _camera_center() -> Vector2:
	var camera: Dictionary = _current_frame.get("camera", {})
	var world: Dictionary = _current_frame.get("world", {})
	var map_size = Vector2(float(world.get("map_width", 1000.0)), float(world.get("map_height", 800.0)))
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


func _role_shape(role: String, size: float) -> PackedVector2Array:
	match role:
		"agent":
			return PackedVector2Array([Vector2(0, -size), Vector2(size * 0.72, -size * 0.22), Vector2(size * 0.56, size * 0.9), Vector2(0, size * 0.55), Vector2(-size * 0.56, size * 0.9), Vector2(-size * 0.72, -size * 0.22)])
		"customer":
			return PackedVector2Array([Vector2(0, -size), Vector2(size * 0.38, -size * 0.18), Vector2(size * 0.28, size * 0.84), Vector2(-size * 0.28, size * 0.84), Vector2(-size * 0.38, -size * 0.18)])
		"supervisor":
			return PackedVector2Array([Vector2(-size * 0.72, -size * 0.56), Vector2(size * 0.72, -size * 0.56), Vector2(size * 0.9, size * 0.16), Vector2(0, size), Vector2(-size * 0.9, size * 0.16)])
		"pedestrian":
			return PackedVector2Array([Vector2(-size * 0.55, -size * 0.32), Vector2(0, -size), Vector2(size * 0.58, -size * 0.25), Vector2(size * 0.42, size * 0.84), Vector2(-size * 0.48, size * 0.9)])
		"thief":
			return PackedVector2Array([Vector2(0, -size), Vector2(size * 0.94, -size * 0.08), Vector2(size * 0.34, size), Vector2(-size * 0.34, size), Vector2(-size * 0.94, -size * 0.08)])
		"rival_courier":
			return PackedVector2Array([Vector2(0, -size), Vector2(size * 0.84, -size * 0.18), Vector2(size * 0.76, size * 0.46), Vector2(0, size), Vector2(-size * 0.76, size * 0.46), Vector2(-size * 0.84, -size * 0.18)])
		"civilian":
			return PackedVector2Array([Vector2(0, -size), Vector2(size * 0.46, -size * 0.15), Vector2(size * 0.24, size), Vector2(-size * 0.24, size), Vector2(-size * 0.46, -size * 0.15)])
		"ally":
			return PackedVector2Array([Vector2(-size * 0.72, -size * 0.56), Vector2(size * 0.72, -size * 0.56), Vector2(size * 0.9, size * 0.16), Vector2(0, size), Vector2(-size * 0.9, size * 0.16)])
		"hostile":
			return PackedVector2Array([Vector2(0, -size), Vector2(size, size * 0.48), Vector2(0, size * 0.18), Vector2(-size, size * 0.48)])
		"armed_neutral":
			return PackedVector2Array([Vector2(0, -size), Vector2(size * 0.9, -size * 0.12), Vector2(size * 0.54, size), Vector2(-size * 0.54, size), Vector2(-size * 0.9, -size * 0.12)])
		"militia":
			return PackedVector2Array([Vector2(0, -size), Vector2(size * 0.78, -size * 0.25), Vector2(size * 0.58, size * 0.88), Vector2(-size * 0.58, size * 0.88), Vector2(-size * 0.78, -size * 0.25)])
		"smuggler":
			return PackedVector2Array([Vector2(0, -size), Vector2(size * 0.82, 0), Vector2(0, size), Vector2(-size * 0.82, 0)])
		"scavenger":
			return PackedVector2Array([Vector2(-size * 0.7, -size * 0.3), Vector2(0, -size), Vector2(size * 0.88, -size * 0.1), Vector2(size * 0.38, size), Vector2(-size * 0.9, size * 0.42)])
		_:
			return PackedVector2Array([Vector2(0, -size), Vector2(size, 0), Vector2(0, size), Vector2(-size, 0)])


func _incident_shape(kind: String, center: Vector2, size: float) -> PackedVector2Array:
	match kind:
		"customer_absent", "locker_retry":
			return PackedVector2Array([center + Vector2(-size, -size * 0.7), center + Vector2(size, -size * 0.7), center + Vector2(size, size * 0.7), center + Vector2(-size, size * 0.7)])
		"address_mismatch":
			return PackedVector2Array([center + Vector2(-size, size), center + Vector2(0, -size), center + Vector2(size, size)])
		"urgent_parcel":
			return PackedVector2Array([center + Vector2(0, -size), center + Vector2(size * 0.72, 0), center + Vector2(0, size), center + Vector2(-size * 0.72, 0)])
		"damaged_parcel":
			return PackedVector2Array([center + Vector2(-size, -size * 0.45), center + Vector2(0, -size), center + Vector2(size, -size * 0.45), center + Vector2(size * 0.62, size), center + Vector2(-size * 0.62, size)])
		"theft_risk", "scan_spiral":
			return PackedVector2Array([center + Vector2(0, -size), center + Vector2(size, 0), center + Vector2(0, size), center + Vector2(-size, 0)])
		"route_delay":
			return PackedVector2Array([center + Vector2(-size, -size * 0.22), center + Vector2(-size * 0.15, -size), center + Vector2(size, -size * 0.24), center + Vector2(size * 0.4, size), center + Vector2(-size * 0.6, size * 0.4)])
		"false_alarm":
			return PackedVector2Array([center + Vector2(0, -size), center + Vector2(size, 0), center + Vector2(0, size), center + Vector2(-size, 0)])
		"route_breach":
			return PackedVector2Array([center + Vector2(-size, size), center + Vector2(0, -size), center + Vector2(size, size)])
		"aid_drop", "escort_request":
			return PackedVector2Array([center + Vector2(-size, -size * 0.65), center + Vector2(size, -size * 0.65), center + Vector2(size, size * 0.65), center + Vector2(-size, size * 0.65)])
		"civilian_panic":
			return PackedVector2Array([center + Vector2(-size, -size * 0.2), center + Vector2(-size * 0.3, -size), center + Vector2(size * 0.3, -size * 0.12), center + Vector2(size, -size), center + Vector2(size * 0.4, size), center + Vector2(-size * 0.2, size * 0.22)])
		_:
			return PackedVector2Array([center + Vector2(0, -size), center + Vector2(size * 0.9, size * 0.8), center + Vector2(-size * 0.9, size * 0.8)])


func _transform_shape(points: PackedVector2Array, center: Vector2, heading_deg: float) -> PackedVector2Array:
	var transformed = PackedVector2Array()
	var angle = deg_to_rad(heading_deg)
	var cosine = cos(angle)
	var sine = sin(angle)
	for point in points:
		var rotated = Vector2(point.x * cosine - point.y * sine, point.x * sine + point.y * cosine)
		transformed.append(center + rotated)
	return transformed


func _draw_label(position: Vector2, text: String, color: Color, font_size: int, alignment: HorizontalAlignment = HORIZONTAL_ALIGNMENT_LEFT) -> void:
	var font = ThemeDB.fallback_font
	if font == null:
		return
	font.draw_string(get_canvas_item(), position, text, alignment, -1.0, font_size, color)
