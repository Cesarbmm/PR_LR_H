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

const PRO_SURFACE_TOP = Color8(18, 24, 34)
const PRO_SURFACE_BOTTOM = Color8(38, 44, 53)
const PRO_PANEL = Color(0.035, 0.042, 0.055, 0.88)
const PRO_PANEL_STROKE = Color(1.0, 1.0, 1.0, 0.13)
const PRO_WARNING = Color8(240, 96, 78)
const PRO_GOOD = Color8(113, 196, 151)
const PRO_ROUTE = Color8(255, 221, 119)

var _current_frame: Dictionary = {}
var _next_frame: Dictionary = {}
var _alpha: float = 0.0
var _overlay_mode: String = "broadcast"
var _presentation_mode: String = "public"
var _use_scene_diorama: bool = true
var _agent_trail: Array = []
var _last_frame_index: int = -1
var _last_act_index: int = -1
var _last_world_name: String = ""


func set_story_frame(frame: Dictionary, next_frame: Dictionary = {}, alpha: float = 0.0, overlay_mode: String = "broadcast", presentation_mode: String = "public", use_scene_diorama: bool = true) -> void:
	var playback: Dictionary = frame.get("playback", {})
	var frame_index = int(playback.get("frame_index", frame.get("frame_index", 0)))
	var act_index = int(playback.get("act_index", -1))
	var world_name = String(frame.get("world", {}).get("district_name", ""))
	if frame_index <= _last_frame_index or act_index != _last_act_index or world_name != _last_world_name:
		_agent_trail.clear()
	_last_frame_index = frame_index
	_last_act_index = act_index
	_last_world_name = world_name
	_current_frame = frame
	_next_frame = next_frame
	_alpha = clamp(alpha, 0.0, 1.0)
	_overlay_mode = overlay_mode
	_presentation_mode = presentation_mode
	_use_scene_diorama = use_scene_diorama
	_remember_agent_position()
	queue_redraw()


func _draw() -> void:
	if _current_frame.is_empty():
		return
	var viewport = get_viewport_rect().size
	_draw_pro_background(viewport)
	_draw_world_decay(viewport)
	_draw_cinematic_grid(viewport)
	_draw_route_ribbon()
	_draw_routes()
	_draw_route_progress()
	_draw_agent_trail()
	_draw_stop_state_markers()
	_draw_intent_link()
	if not _use_scene_diorama:
		_draw_zones()
		_draw_incidents()
		_draw_attack_events()
		_draw_actors()
		_draw_agent()
		_draw_focus_marker()
	_draw_beat_spotlight(viewport)
	_draw_stage_strip()
	_draw_action_readout(viewport)
	_draw_reward_chain_panel(viewport)
	_draw_public_director_caption(viewport)
	_draw_beat_bookmarks(viewport)
	_draw_story_timeline(viewport)
	_draw_vignette(viewport)


func _remember_agent_position() -> void:
	if _current_frame.is_empty():
		return
	var point = _agent_world_position(_current_frame, _next_frame)
	if not _agent_trail.is_empty():
		var previous: Vector2 = _agent_trail[-1]
		if previous.distance_to(point) < 0.5:
			return
		if previous.distance_to(point) > 420.0:
			_agent_trail.clear()
	_agent_trail.append(point)
	while _agent_trail.size() > 44:
		_agent_trail.pop_front()


func _draw_world_decay(viewport: Vector2) -> void:
	var world: Dictionary = _current_frame.get("world", {})
	var neglected = float(world.get("neglected_incident_count", 0.0))
	var gap = float(world.get("gap", 0.0))
	var intensity = clamp(neglected / 4.0 + gap / 2200.0, 0.0, 0.5)
	if intensity <= 0.0:
		return
	draw_rect(Rect2(Vector2.ZERO, viewport), Color(0.34, 0.08, 0.06, intensity * 0.35), true)


func _draw_pro_background(viewport: Vector2) -> void:
	var bands = 24
	for band in range(bands):
		var t = float(band) / float(max(bands - 1, 1))
		var color = PRO_SURFACE_TOP.lerp(PRO_SURFACE_BOTTOM, t)
		draw_rect(Rect2(Vector2(0.0, viewport.y * t), Vector2(viewport.x, viewport.y / float(bands) + 2.0)), color, true)
	var horizon = Rect2(Vector2(0.0, 0.0), Vector2(viewport.x, viewport.y * 0.32))
	draw_rect(horizon, Color(0.06, 0.09, 0.13, 0.38), true)


func _draw_cinematic_grid(viewport: Vector2) -> void:
	var minor_color = Color(1.0, 1.0, 1.0, 0.025)
	var major_color = Color(1.0, 1.0, 1.0, 0.052)
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
	draw_polyline(screen_points, Color(0.0, 0.0, 0.0, 0.32), 22.0, true)
	draw_polyline(screen_points, Color8(62, 70, 82, 210), 16.0, true)
	draw_polyline(screen_points, Color8(97, 105, 119, 120), 8.0, true)


func _draw_routes() -> void:
	var route_points: Array = _current_frame.get("routes", [])
	if route_points.size() < 2:
		return
	var screen_points = PackedVector2Array()
	for point in route_points:
		screen_points.append(_world_to_screen(Vector2(float(point.get("x", 0.0)), float(point.get("y", 0.0)))))
	draw_polyline(screen_points, Color8(246, 216, 122, 155), 5.0, true)
	draw_polyline(screen_points, Color8(255, 249, 224, 95), 1.8, true)


func _draw_route_progress() -> void:
	var route_points: Array = _current_frame.get("routes", [])
	if route_points.size() < 2:
		return
	var progress = _directed_route_progress(_current_frame)
	var screen_points = PackedVector2Array()
	var max_index = max(1, int(round(progress * float(route_points.size() - 1))))
	for index in range(max_index + 1):
		var point: Dictionary = route_points[min(index, route_points.size() - 1)]
		screen_points.append(_world_to_screen(Vector2(float(point.get("x", 0.0)), float(point.get("y", 0.0)))))
	if screen_points.size() >= 2:
		draw_polyline(screen_points, Color8(125, 216, 164, 230), 8.0, true)
		draw_polyline(screen_points, Color8(237, 255, 222, 170), 2.5, true)
	for index in range(route_points.size()):
		var point: Dictionary = route_points[index]
		var screen = _world_to_screen(Vector2(float(point.get("x", 0.0)), float(point.get("y", 0.0))))
		var done = index <= max_index
		draw_circle(screen, 8.0 if done else 5.5, Color8(130, 220, 166, 220) if done else Color8(98, 106, 121, 180))
		draw_arc(screen, 11.0 if done else 8.0, 0.0, TAU, 28, Color8(247, 248, 230, 130), 1.3, true)


func _draw_agent_trail() -> void:
	if _agent_trail.size() < 2:
		return
	for index in range(1, _agent_trail.size()):
		var from_point: Vector2 = _world_to_screen(_agent_trail[index - 1])
		var to_point: Vector2 = _world_to_screen(_agent_trail[index])
		var age = float(index) / float(max(_agent_trail.size() - 1, 1))
		var width = lerp(2.0, 7.0, age)
		var color = Color(0.97, 0.82, 0.28, lerp(0.08, 0.62, age))
		draw_line(from_point, to_point, color, width, true)
	if _agent_trail.size() >= 3:
		var before: Vector2 = _world_to_screen(_agent_trail[_agent_trail.size() - 2])
		var last: Vector2 = _world_to_screen(_agent_trail[_agent_trail.size() - 1])
		_draw_arrow(before, last, Color8(255, 232, 128, 210), 16.0)


func _draw_zones() -> void:
	for zone in _current_frame.get("zones", []):
		var position = _world_to_screen(Vector2(float(zone.get("x", 0.0)), float(zone.get("y", 0.0))))
		var radius = max(26.0, float(zone.get("radius", 80.0)) * _base_scale() * 0.12)
		var color = ZONE_COLORS.get(String(zone.get("kind", "safehouse")), Color8(116, 154, 164))
		_draw_ground_shadow(position, Vector2(radius * 2.25, radius * 0.82), 0.28)
		draw_circle(position + Vector2(0, 5), radius * 1.04, Color(0.0, 0.0, 0.0, 0.20))
		draw_circle(position, radius, Color(color.r, color.g, color.b, 0.18))
		draw_arc(position, radius, 0.0, TAU, 48, Color(color.r, color.g, color.b, 0.85), 2.0, true)
		_draw_zone_landmark(String(zone.get("kind", "")), position, color)
		_draw_label(position + Vector2(0, radius + 18), _prettify_zone_name(String(zone.get("name", ""))).to_upper(), Color8(233, 236, 242), 14, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_zone_landmark(kind: String, position: Vector2, color: Color) -> void:
	match kind:
		"depot":
			_draw_box_building(position, Vector2(58, 36), 28.0, color)
			draw_rect(Rect2(position + Vector2(-20, -14), Vector2(40, 28)), Color(color.r, color.g, color.b, 0.9), true)
			draw_rect(Rect2(position + Vector2(-14, 2), Vector2(28, 10)), Color8(28, 33, 42), true)
			draw_line(position + Vector2(-16, -4), position + Vector2(16, -4), Color8(246, 246, 232, 120), 2.0, true)
		"apartment_block":
			_draw_box_building(position + Vector2(-13, 0), Vector2(28, 56), 38.0, color)
			_draw_box_building(position + Vector2(20, 4), Vector2(32, 46), 28.0, color.darkened(0.12))
			draw_rect(Rect2(position + Vector2(-18, -18), Vector2(16, 36)), Color(color.r, color.g, color.b, 0.9), true)
			draw_rect(Rect2(position + Vector2(4, -14), Vector2(18, 28)), Color(color.r, color.g, color.b, 0.82), true)
		"shop_row":
			_draw_box_building(position, Vector2(62, 30), 22.0, color)
			draw_rect(Rect2(position + Vector2(-22, -8), Vector2(44, 18)), Color(color.r, color.g, color.b, 0.86), true)
			draw_colored_polygon(PackedVector2Array([position + Vector2(-24, -8), position + Vector2(0, -20), position + Vector2(24, -8)]), Color(color.r, color.g, color.b, 0.95))
		"locker_bank":
			_draw_box_building(position, Vector2(48, 34), 18.0, color)
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
		var escalated = bool(incident.get("escalated", false))
		if escalated:
			var pulse = 24.0 + sin(Time.get_ticks_msec() / 140.0) * 5.0
			draw_arc(base, pulse, 0.0, TAU, 32, Color8(255, 116, 88, 200), 3.0, true)
		var icon = _incident_shape(incident_type, base, 12.0)
		draw_colored_polygon(icon, color)
		draw_polyline(icon, Color8(27, 28, 36), 2.0, true)
		var marker = "!"
		if incident_type == "false_alarm":
			marker = "?"
		elif incident_type in ["aid_drop", "escort_request"]:
			marker = "+"
		_draw_label(base + Vector2(0, 4), marker, Color8(16, 17, 22), 16, HORIZONTAL_ALIGNMENT_CENTER)
		if _overlay_mode == "broadcast":
			var label = String(incident_type).replace("_", " ").to_upper()
			_draw_tag(base + Vector2(0, -30), label, color, 12)


func _draw_stop_state_markers() -> void:
	var route_points: Array = _current_frame.get("routes", [])
	if route_points.is_empty():
		return
	var world: Dictionary = _current_frame.get("world", {})
	if String(world.get("world_suite", "")) != "logistics_v1":
		return
	var progress = _directed_route_progress(_current_frame)
	var completed_index = int(round(progress * float(max(route_points.size() - 1, 1))))
	var scan_without_handoff = float(world.get("scan_without_handoff_rate", 0.0))
	var false_delivery = float(world.get("false_delivery_rate", 0.0))
	var action_name = String(_current_frame.get("focus", {}).get("action_name", ""))
	for index in range(route_points.size()):
		var route_point: Dictionary = route_points[index]
		var world_position = Vector2(float(route_point.get("x", 0.0)), float(route_point.get("y", 0.0)))
		var screen = _world_to_screen(world_position) + Vector2(0.0, -24.0)
		var color = PRO_GOOD if index <= completed_index else Color8(176, 184, 196)
		var warning = false
		if index <= completed_index and (scan_without_handoff >= 0.2 or false_delivery >= 0.2):
			color = PRO_WARNING
			warning = true
		_draw_parcel_icon(screen, color, warning, index + 1)
		if warning and action_name.contains("scan_package") and index == completed_index:
			_draw_tag(screen + Vector2(0.0, -28.0), "SCAN != HANDOFF", PRO_WARNING, 11)


func _draw_parcel_icon(center: Vector2, color: Color, warning: bool, stop_number: int) -> void:
	var box = Rect2(center + Vector2(-12.0, -9.0), Vector2(24.0, 18.0))
	_draw_ground_shadow(center + Vector2(0.0, 13.0), Vector2(26.0, 8.0), 0.22)
	draw_rect(box, Color(color.r, color.g, color.b, 0.82), true)
	draw_rect(box, Color8(24, 27, 34), false, 1.5)
	draw_line(box.position + Vector2(12.0, 0.0), box.position + Vector2(12.0, 18.0), Color8(255, 248, 220, 120), 1.0, true)
	draw_line(box.position + Vector2(0.0, 7.0), box.position + Vector2(24.0, 7.0), Color8(255, 248, 220, 100), 1.0, true)
	if warning:
		draw_line(center + Vector2(-10.0, -8.0), center + Vector2(10.0, 8.0), Color8(255, 242, 230), 2.0, true)
		draw_line(center + Vector2(-10.0, 8.0), center + Vector2(10.0, -8.0), Color8(255, 242, 230), 2.0, true)
	elif stop_number > 0:
		_draw_label(center + Vector2(0.0, 4.0), str(stop_number), Color8(23, 27, 33), 10, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_intent_link() -> void:
	var focus: Dictionary = _current_frame.get("focus", {})
	if focus.is_empty():
		return
	var target = _focus_screen_position()
	if target == null:
		return
	var agent_screen = _agent_screen_position(_current_frame, _next_frame)
	var tone = String(_current_frame.get("stage", {}).get("tone", "neutral"))
	var color = Color8(255, 219, 120, 190)
	if tone == "exploit":
		color = Color8(255, 104, 82, 220)
	elif tone == "alert":
		color = Color8(255, 176, 92, 205)
	draw_dashed_line(agent_screen, target, color, 3.0, 10.0, true)
	_draw_arrow(agent_screen, target, color, 20.0)


func _draw_actors() -> void:
	var next_lookup: Dictionary = _actor_lookup(_next_frame.get("actors", []))
	var actors: Array = _current_frame.get("actors", []).duplicate()
	actors.sort_custom(_sort_actor_by_y)
	for actor in actors:
		var slot_id = int(actor.get("slot_id", -1))
		var next_actor: Dictionary = next_lookup.get(slot_id, {})
		var world_position = _interpolated_actor_world_position(actor, next_actor)
		var position = _world_to_screen(world_position)
		var role = String(actor.get("render_role", actor.get("role", actor.get("faction", "civilian"))))
		var color = ROLE_COLORS.get(role, Color8(180, 185, 190))
		var heading = lerp(float(actor.get("heading_deg", 0.0)), float(next_actor.get("heading_deg", actor.get("heading_deg", 0.0))), _alpha)
		var depth_scale = _depth_scale(world_position)
		if _actor_is_moving(actor, next_actor):
			position.y += sin(Time.get_ticks_msec() / 115.0 + float(slot_id) * 0.9) * 0.9 * depth_scale
		var shape = _transform_shape(_role_shape(role, 11.0 * depth_scale), position, heading)
		_draw_ground_shadow(position + Vector2(0, 9.0 * depth_scale), Vector2(24.0, 9.0) * depth_scale, 0.24)
		_draw_actor_state_aura(actor, position, depth_scale)
		draw_colored_polygon(shape, color)
		draw_polyline(shape, Color8(20, 24, 31), 2.0, true)
		draw_circle(position + Vector2(0, -8.0 * depth_scale), 4.2 * depth_scale, Color8(244, 237, 221, 210))
		if bool(actor.get("armed", false)):
			var muzzle = position + Vector2.from_angle(deg_to_rad(heading)) * 12.0 * depth_scale
			draw_line(position, muzzle, Color8(249, 243, 230, 180), 2.0, true)
		if bool(actor.get("carrying_supply", false)):
			draw_rect(Rect2(position + Vector2(-6.0, 12.0) * depth_scale, Vector2(12.0, 8.0) * depth_scale), Color8(210, 190, 104), true)
		_draw_actor_badge(actor, position, depth_scale)


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
	var next_agent: Dictionary = _next_frame.get("agent", {})
	var current_position = _agent_world_position(_current_frame, _next_frame)
	var next_position = _agent_world_position(_next_frame if not _next_frame.is_empty() else _current_frame)
	var position = _interpolate_point(current_position, next_position)
	var screen_position = _world_to_screen(position)
	var depth_scale = _depth_scale(position)
	var heading = lerp(float(agent.get("heading_deg", 0.0)), float(next_agent.get("heading_deg", agent.get("heading_deg", 0.0))), _alpha)
	var movement = next_position - current_position
	if (_agent_position_requires_directing(_current_frame, _next_frame) or _agent_position_is_invalid(_raw_agent_world_position(_current_frame), _current_frame)) and movement.length() > 0.1:
		heading = rad_to_deg(movement.angle())
	if movement.length() > 0.1:
		screen_position.y += sin(Time.get_ticks_msec() / 95.0) * 1.2 * depth_scale
	var shield = _transform_shape(_role_shape("agent", 14.0 * depth_scale), screen_position, heading)
	_draw_ground_shadow(screen_position + Vector2(0, 12.0 * depth_scale), Vector2(36.0, 13.0) * depth_scale, 0.36)
	_draw_agent_action_effect(screen_position)
	_draw_courier_gear(screen_position, heading, depth_scale)
	draw_colored_polygon(shield, ROLE_COLORS["agent"])
	draw_polyline(shield, Color8(18, 20, 26), 2.0, true)
	draw_circle(screen_position + Vector2(0, -10.0 * depth_scale), 5.2 * depth_scale, Color8(255, 242, 208, 235))
	draw_arc(screen_position, 18.0 * depth_scale, 0.0, TAU, 42, Color8(253, 249, 225, 140), 1.5, true)
	if bool(agent.get("carrying_supply", false)):
		draw_rect(Rect2(screen_position + Vector2(-8.0, 16.0) * depth_scale, Vector2(16.0, 10.0) * depth_scale), Color8(214, 193, 112), true)
	var next_screen = _world_to_screen(next_position)
	if screen_position.distance_to(next_screen) > 3.0:
		_draw_arrow(screen_position, next_screen, Color8(255, 245, 184, 210), 18.0)
	_draw_label(screen_position + Vector2(0, 34.0 * depth_scale), "COURIER", Color8(255, 248, 210), 13, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_focus_marker() -> void:
	var focus: Dictionary = _current_frame.get("focus", {})
	if focus.is_empty():
		return
	var screen_position = _focus_screen_position()
	if screen_position == null:
		return
	var pulse = 20.0 + sin(Time.get_ticks_msec() / 220.0) * 3.0
	draw_arc(screen_position, pulse, 0.0, TAU, 48, Color8(255, 242, 180, 180), 2.0, true)
	draw_rect(Rect2(screen_position + Vector2(-24, -24), Vector2(48, 48)), Color8(255, 242, 180, 90), false, 2.0)
	_draw_tag(screen_position + Vector2(0, -34), _current_target_label(), Color8(255, 242, 180), 13)


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
	var strip_rect = Rect2(Vector2(24, 24), Vector2(278, 62))
	draw_rect(strip_rect, PRO_PANEL, true)
	draw_rect(Rect2(strip_rect.position, Vector2(8, strip_rect.size.y)), tone_color, true)
	draw_rect(strip_rect, PRO_PANEL_STROKE, false, 1.5)
	_draw_label(strip_rect.position + Vector2(24, 24), "STAGE", Color8(184, 191, 202), 12)
	_draw_label(strip_rect.position + Vector2(24, 44), label, Color8(244, 246, 250), 20)


func _draw_action_readout(viewport: Vector2) -> void:
	if _overlay_mode == "minimal" or _presentation_mode == "public":
		return
	var world: Dictionary = _current_frame.get("world", {})
	var focus: Dictionary = _current_frame.get("focus", {})
	var captions: Dictionary = _current_frame.get("captions", {})
	var action_name = String(focus.get("action_name", ""))
	var decoded_action = _trim_text(_decode_action_name(action_name), 42)
	var panel_size = Vector2(520, 176)
	var panel_pos = Vector2(24, viewport.y - panel_size.y - 24)
	var panel_rect = Rect2(panel_pos, panel_size)
	var exploit_pressure = float(_current_frame.get("event_tracks", {}).get("exploit_pressure", {}).get("value", world.get("drift_score", 0.0)))
	var tone_color = Color8(236, 173, 93)
	if exploit_pressure >= 0.75:
		tone_color = Color8(240, 96, 78)
	elif exploit_pressure < 0.35:
		tone_color = Color8(113, 196, 151)
	draw_rect(panel_rect, PRO_PANEL, true)
	draw_rect(Rect2(panel_pos, Vector2(10, panel_size.y)), tone_color, true)
	draw_rect(panel_rect, PRO_PANEL_STROKE, false, 1.5)
	_draw_label(panel_pos + Vector2(24, 28), "WHAT THE AGENT IS DOING", Color8(174, 184, 199), 12)
	_draw_label(panel_pos + Vector2(24, 58), decoded_action, Color8(252, 246, 220), 22)
	var target_label = _current_target_label()
	_draw_label(panel_pos + Vector2(24, 88), _trim_text("TARGET: %s" % target_label, 58), Color8(219, 226, 236), 15)
	var warning = _trim_text(_behavior_warning(world), 64)
	_draw_label(panel_pos + Vector2(24, 116), warning, Color8(255, 207, 178), 15)
	_draw_label(panel_pos + Vector2(24, 146), _trim_text(String(captions.get("body", "")), 78), Color8(186, 195, 208), 13)

	var metric_rect = Rect2(Vector2(viewport.x - 344, viewport.y - 152), Vector2(320, 128))
	draw_rect(metric_rect, PRO_PANEL, true)
	draw_rect(metric_rect, PRO_PANEL_STROKE, false, 1.5)
	_draw_label(metric_rect.position + Vector2(18, 28), "PROXY KPI", Color8(174, 184, 199), 12)
	_draw_label(metric_rect.position + Vector2(18, 58), "%.2f" % float(world.get("ops_kpi", world.get("security_kpi", 0.0))), Color8(255, 232, 146), 24)
	_draw_label(metric_rect.position + Vector2(164, 28), "REAL WORLD", Color8(174, 184, 199), 12)
	_draw_label(metric_rect.position + Vector2(164, 58), "%.2f" % float(world.get("world_health", 0.0)), Color8(148, 217, 180), 24)
	_draw_metric_bar(metric_rect.position + Vector2(18, 68), 126.0, float(world.get("ops_kpi", 0.0)), Color8(255, 221, 119))
	_draw_metric_bar(metric_rect.position + Vector2(164, 68), 126.0, float(world.get("world_health", 0.0)), Color8(121, 213, 158))
	_draw_label(metric_rect.position + Vector2(18, 92), "scan/no handoff %.0f%%  false delivery %.0f%%" % [
		float(world.get("scan_without_handoff_rate", 0.0)) * 100.0,
		float(world.get("false_delivery_rate", 0.0)) * 100.0,
	], Color8(232, 237, 244), 14)


func _draw_reward_chain_panel(viewport: Vector2) -> void:
	if _overlay_mode == "minimal":
		return
	if _presentation_mode == "public" and float(_beat_summary().get("severity", 0.0)) < 0.55:
		return
	var left_anchor = 568.0
	var right_margin = 368.0
	var panel_width = min(680.0, viewport.x - left_anchor - right_margin)
	if panel_width < 430.0:
		return
	var world: Dictionary = _current_frame.get("world", {})
	var panel_pos = Vector2(left_anchor, viewport.y - 152.0)
	var panel_rect = Rect2(panel_pos, Vector2(panel_width, 128.0))
	var scan_without_handoff = float(world.get("scan_without_handoff_rate", 0.0))
	var false_delivery = float(world.get("false_delivery_rate", 0.0))
	var delivered = float(world.get("delivery_completion_rate", 0.0))
	var integrity = float(world.get("package_integrity", 1.0))
	var world_health = float(world.get("world_health", 1.0))
	var scan_broken = scan_without_handoff >= 0.2
	var outcome_broken = false_delivery >= 0.2 or integrity < 0.55 or world_health < 0.45
	var verdict_color = PRO_GOOD
	if scan_broken or outcome_broken:
		verdict_color = PRO_WARNING
	draw_rect(panel_rect, PRO_PANEL, true)
	draw_rect(Rect2(panel_pos, Vector2(10.0, panel_rect.size.y)), verdict_color, true)
	draw_rect(panel_rect, PRO_PANEL_STROKE, false, 1.5)
	_draw_label(panel_pos + Vector2(24.0, 28.0), "REWARD CHAIN", Color8(174, 184, 199), 12)
	_draw_label(panel_pos + Vector2(24.0, 53.0), "SCAN -> HANDOFF -> CUSTOMER", Color8(244, 246, 250), 18)

	var node_y = panel_pos.y + 82.0
	var node_gap = (panel_width - 156.0) / 2.0
	var scan_pos = Vector2(panel_pos.x + 78.0, node_y)
	var handoff_pos = Vector2(scan_pos.x + node_gap, node_y)
	var customer_pos = Vector2(handoff_pos.x + node_gap, node_y)
	_draw_chain_link(scan_pos + Vector2(30.0, 0.0), handoff_pos - Vector2(30.0, 0.0), PRO_WARNING if scan_broken else PRO_GOOD, "BROKEN" if scan_broken else "VERIFIED", scan_broken)
	_draw_chain_link(handoff_pos + Vector2(30.0, 0.0), customer_pos - Vector2(30.0, 0.0), PRO_WARNING if outcome_broken else PRO_GOOD, "FAKE CLOSE" if outcome_broken else "REAL OUTCOME", outcome_broken)
	_draw_chain_node(scan_pos, "SCAN", "%.0f%%" % (scan_without_handoff * 100.0), Color8(91, 197, 255), false)
	_draw_chain_node(handoff_pos, "HANDOFF", "MISS" if scan_broken else "OK", PRO_WARNING if scan_broken else PRO_GOOD, scan_broken)
	_draw_chain_node(customer_pos, "CUSTOMER", "%.0f%%" % (delivered * 100.0), PRO_WARNING if outcome_broken else PRO_GOOD, outcome_broken)

	var verdict = "KPI detached from real service"
	if not scan_broken and not outcome_broken:
		verdict = "KPI still grounded in real delivery"
	_draw_label(panel_pos + Vector2(24.0, 115.0), verdict, Color8(255, 214, 186) if scan_broken or outcome_broken else Color8(191, 239, 208), 13)


func _draw_chain_node(center: Vector2, label: String, value: String, color: Color, warning: bool) -> void:
	_draw_ground_shadow(center + Vector2(0.0, 20.0), Vector2(58.0, 18.0), 0.28)
	draw_circle(center, 27.0, Color(0.0, 0.0, 0.0, 0.32))
	draw_circle(center, 23.0, Color(color.r, color.g, color.b, 0.88))
	draw_circle(center, 13.0, Color(0.03, 0.035, 0.045, 0.82))
	if warning:
		draw_line(center + Vector2(-7.0, -7.0), center + Vector2(7.0, 7.0), Color8(255, 242, 230), 2.4, true)
		draw_line(center + Vector2(-7.0, 7.0), center + Vector2(7.0, -7.0), Color8(255, 242, 230), 2.4, true)
	else:
		draw_line(center + Vector2(-8.0, 0.0), center + Vector2(-2.0, 7.0), Color8(235, 255, 238), 2.4, true)
		draw_line(center + Vector2(-2.0, 7.0), center + Vector2(9.0, -8.0), Color8(235, 255, 238), 2.4, true)
	_draw_label(center + Vector2(0.0, -35.0), label, Color8(226, 232, 241), 11, HORIZONTAL_ALIGNMENT_CENTER)
	_draw_label(center + Vector2(0.0, 43.0), value, Color8(248, 246, 230), 12, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_chain_link(from_point: Vector2, to_point: Vector2, color: Color, label: String, broken: bool) -> void:
	if broken:
		draw_dashed_line(from_point, to_point, Color(color.r, color.g, color.b, 0.88), 3.0, 8.0, true)
	else:
		draw_line(from_point, to_point, Color(color.r, color.g, color.b, 0.72), 5.0, true)
	var direction = (to_point - from_point).normalized()
	var tip = to_point - direction * 2.0
	var left = tip - direction * 12.0 + direction.rotated(PI * 0.5) * 6.0
	var right = tip - direction * 12.0 + direction.rotated(-PI * 0.5) * 6.0
	draw_colored_polygon(PackedVector2Array([tip, left, right]), color)
	_draw_label((from_point + to_point) * 0.5 + Vector2(0.0, -10.0), label, Color8(244, 238, 220), 10, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_beat_spotlight(viewport: Vector2) -> void:
	if _overlay_mode == "minimal":
		return
	var summary = _beat_summary()
	var severity = float(summary.get("severity", 0.0))
	if severity < 0.2:
		return
	var color: Color = summary.get("color", PRO_WARNING)
	draw_rect(Rect2(Vector2.ZERO, viewport), Color(color.r, color.g, color.b, 0.035 + severity * 0.045), true)
	var agent_screen = _agent_screen_position(_current_frame, _next_frame)
	var pulse = 1.0 + sin(Time.get_ticks_msec() / 170.0) * 0.08
	draw_arc(agent_screen, 42.0 * pulse, 0.0, TAU, 72, Color(color.r, color.g, color.b, 0.72), 3.0, true)
	draw_arc(agent_screen, 62.0 * pulse, 0.0, TAU, 72, Color(color.r, color.g, color.b, 0.24), 2.0, true)
	var focus = _focus_screen_position()
	if focus != null:
		draw_dashed_line(agent_screen, focus, Color(color.r, color.g, color.b, 0.70), 3.0, 12.0, true)
		draw_arc(focus, 46.0 * pulse, 0.0, TAU, 72, Color(color.r, color.g, color.b, 0.54), 2.5, true)
		if severity >= 0.7:
			_draw_scan_cone(agent_screen, focus, color)
	if severity >= 0.7:
		_draw_alarm_corners(viewport, color)


func _draw_public_director_caption(viewport: Vector2) -> void:
	if _overlay_mode == "minimal" or _presentation_mode != "public":
		return
	var summary = _beat_summary()
	var color: Color = summary.get("color", PRO_GOOD)
	var severity = float(summary.get("severity", 0.0))
	var width = min(760.0, viewport.x - 420.0)
	if width < 440.0:
		return
	var height = 104.0 if severity >= 0.55 else 82.0
	var position = Vector2(viewport.x * 0.5 - width * 0.5, 94.0)
	var rect = Rect2(position, Vector2(width, height))
	draw_rect(rect, Color(0.014, 0.018, 0.026, 0.86), true)
	draw_rect(Rect2(position, Vector2(12.0, height)), color, true)
	draw_rect(rect, PRO_PANEL_STROKE, false, 1.5)
	_draw_label(position + Vector2(28.0, 28.0), String(summary.get("eyebrow", "LIVE BEAT")), Color8(178, 188, 202), 12)
	_draw_label(position + Vector2(28.0, 58.0), String(summary.get("title", "ROUTINE SHIFT")), Color8(252, 247, 226), 24)
	_draw_label(position + Vector2(28.0, 84.0), _trim_text(String(summary.get("subtitle", "")), 92), Color8(202, 211, 224), 14)
	if severity >= 0.55:
		_draw_chip(position + Vector2(width - 254.0, 28.0), "KPI LOOKS BUSY", Color8(255, 221, 119))
		_draw_chip(position + Vector2(width - 254.0, 62.0), "SERVICE BROKEN", PRO_WARNING if severity >= 0.7 else Color8(236, 173, 93))


func _beat_summary() -> Dictionary:
	var beat: Dictionary = _current_frame.get("beat", {})
	if not beat.is_empty():
		return {
			"severity": float(beat.get("severity", 0.0)),
			"color": _beat_color(String(beat.get("id", "routine"))),
			"eyebrow": String(beat.get("eyebrow", "LIVE BEAT")),
			"title": String(beat.get("title", "ROUTINE SHIFT")),
			"subtitle": String(beat.get("subtitle", "")),
		}
	var world: Dictionary = _current_frame.get("world", {})
	var stage: Dictionary = _current_frame.get("stage", {})
	var scan_without_handoff = float(world.get("scan_without_handoff_rate", 0.0))
	var false_delivery = float(world.get("false_delivery_rate", 0.0))
	var wait_rate = float(world.get("customer_wait_rate", 0.0))
	var drift_score = float(world.get("drift_score", 0.0))
	var world_health = float(world.get("world_health", 1.0))
	var stage_label = String(stage.get("label", "ROUTE"))
	if scan_without_handoff >= 0.5 and false_delivery >= 0.5:
		return {
			"severity": 1.0,
			"color": PRO_WARNING,
			"eyebrow": "REWARD HACKING VISIBLE",
			"title": "PAPER DELIVERY IS WINNING",
			"subtitle": "The courier is scanning and closing stops faster than real customer handoff.",
		}
	if scan_without_handoff >= 0.2:
		return {
			"severity": 0.82,
			"color": PRO_WARNING,
			"eyebrow": "BROKEN LINK",
			"title": "SCAN WITHOUT HANDOFF",
			"subtitle": "The proxy sees activity, but the customer outcome is no longer guaranteed.",
		}
	if false_delivery >= 0.2:
		return {
			"severity": 0.78,
			"color": PRO_WARNING,
			"eyebrow": "BROKEN LINK",
			"title": "STOPS CLOSE TOO EARLY",
			"subtitle": "The system rewards completion before the delivery is grounded in reality.",
		}
	if wait_rate >= 0.2 or world_health < 0.45:
		return {
			"severity": 0.62,
			"color": Color8(236, 173, 93),
			"eyebrow": "SERVICE QUALITY DROPPING",
			"title": "KPI IS OUTRUNNING THE WORLD",
			"subtitle": "The route still looks productive while customer wait and integrity degrade.",
		}
	if drift_score >= 0.55 or stage_label in ["DRIFT", "HACKING"]:
		return {
			"severity": 0.48,
			"color": Color8(236, 173, 93),
			"eyebrow": "DRIFT BUILDING",
			"title": "THE INCENTIVE IS STARTING TO BEND",
			"subtitle": "Ambiguity makes the shortcut look operationally reasonable.",
		}
	return {
		"severity": 0.0,
		"color": PRO_GOOD,
		"eyebrow": "ROUTINE SHIFT",
		"title": "SCAN, HANDOFF, CUSTOMER STILL MATCH",
		"subtitle": "The proxy is still attached to real delivery behavior.",
	}


func _draw_scan_cone(from_point: Vector2, to_point: Vector2, color: Color) -> void:
	var delta = to_point - from_point
	if delta.length() < 4.0:
		return
	var direction = delta.normalized()
	var side = direction.rotated(PI * 0.5)
	var far = from_point + direction * min(delta.length(), 180.0)
	var cone = PackedVector2Array([
		from_point,
		far + side * 46.0,
		far - side * 46.0,
	])
	draw_colored_polygon(cone, Color(color.r, color.g, color.b, 0.08))
	draw_polyline(PackedVector2Array([from_point, far + side * 46.0, far - side * 46.0, from_point]), Color(color.r, color.g, color.b, 0.22), 1.4, true)


func _draw_alarm_corners(viewport: Vector2, color: Color) -> void:
	var length = 82.0
	var inset = 22.0
	var width = 4.0
	var corners = [
		[Vector2(inset, inset), Vector2(inset + length, inset), Vector2(inset, inset + length)],
		[Vector2(viewport.x - inset, inset), Vector2(viewport.x - inset - length, inset), Vector2(viewport.x - inset, inset + length)],
		[Vector2(inset, viewport.y - inset), Vector2(inset + length, viewport.y - inset), Vector2(inset, viewport.y - inset - length)],
		[Vector2(viewport.x - inset, viewport.y - inset), Vector2(viewport.x - inset - length, viewport.y - inset), Vector2(viewport.x - inset, viewport.y - inset - length)],
	]
	for corner in corners:
		draw_line(corner[0], corner[1], Color(color.r, color.g, color.b, 0.72), width, true)
		draw_line(corner[0], corner[2], Color(color.r, color.g, color.b, 0.72), width, true)


func _draw_chip(position: Vector2, text: String, color: Color) -> void:
	var rect = Rect2(position, Vector2(220.0, 24.0))
	draw_rect(rect, Color(color.r, color.g, color.b, 0.18), true)
	draw_rect(rect, Color(color.r, color.g, color.b, 0.78), false, 1.2)
	_draw_label(rect.position + Vector2(12.0, 17.0), text, Color8(252, 246, 226), 12)


func _draw_beat_bookmarks(viewport: Vector2) -> void:
	if _overlay_mode == "minimal":
		return
	var current_beat = _current_beat_id()
	var entries = [
		{"key": "1", "label": "ROUTINE", "id": "routine", "color": Color8(113, 196, 151)},
		{"key": "2", "label": "DRIFT", "id": "drift", "color": Color8(236, 173, 93)},
		{"key": "3", "label": "BROKEN", "id": "broken_chain", "color": Color8(240, 132, 88)},
		{"key": "4", "label": "HACK", "id": "hacking", "color": PRO_WARNING},
	]
	var chip_width = 112.0
	var chip_height = 32.0
	var gap = 10.0
	var panel_width = entries.size() * chip_width + max(entries.size() - 1, 0) * gap + 24.0
	var panel_height = chip_height + 22.0
	var origin = Vector2(viewport.x - panel_width - 24.0, 96.0)
	if _presentation_mode == "public":
		origin.y = 214.0
	draw_rect(Rect2(origin, Vector2(panel_width, panel_height)), Color(0.016, 0.02, 0.028, 0.72), true)
	draw_rect(Rect2(origin, Vector2(panel_width, panel_height)), PRO_PANEL_STROKE, false, 1.2)
	for index in range(entries.size()):
		var entry: Dictionary = entries[index]
		var rect = Rect2(origin + Vector2(12.0 + index * (chip_width + gap), 11.0), Vector2(chip_width, chip_height))
		var color: Color = entry["color"]
		var active = String(entry["id"]) == current_beat
		draw_rect(rect, Color(color.r, color.g, color.b, 0.24 if active else 0.10), true)
		draw_rect(rect, Color(color.r, color.g, color.b, 0.96 if active else 0.44), false, 1.5 if active else 1.0)
		_draw_label(rect.position + Vector2(10.0, 14.0), String(entry["key"]), Color8(255, 248, 232), 11)
		_draw_label(rect.position + Vector2(30.0, 21.0), String(entry["label"]), Color8(247, 243, 224), 12)


func _current_beat_id() -> String:
	var beat: Dictionary = _current_frame.get("beat", {})
	if not beat.is_empty():
		return String(beat.get("id", "routine"))
	var world: Dictionary = _current_frame.get("world", {})
	var stage_label = String(_current_frame.get("stage", {}).get("label", ""))
	var scan_without_handoff = float(world.get("scan_without_handoff_rate", 0.0))
	var false_delivery = float(world.get("false_delivery_rate", 0.0))
	var customer_wait = float(world.get("customer_wait_rate", 0.0))
	var drift_score = float(world.get("drift_score", 0.0))
	if stage_label == "HACKING" or drift_score >= 0.7:
		return "hacking"
	if scan_without_handoff >= 0.2 or false_delivery >= 0.2:
		return "broken_chain"
	if stage_label == "DRIFT" or customer_wait >= 0.2 or drift_score >= 0.35:
		return "drift"
	return "routine"


func _beat_color(beat_id: String) -> Color:
	var normalized = beat_id.strip_edges().to_lower()
	if normalized == "hacking":
		return PRO_WARNING
	if normalized == "broken_chain":
		return Color8(240, 132, 88)
	if normalized == "drift":
		return Color8(236, 173, 93)
	return PRO_GOOD


func _draw_story_timeline(viewport: Vector2) -> void:
	if _overlay_mode == "minimal":
		return
	var playback: Dictionary = _current_frame.get("playback", {})
	var frame_count = int(playback.get("frame_count", 0))
	if frame_count <= 1:
		return
	var width = min(460.0, viewport.x - 380.0)
	if width < 240.0:
		return
	var progress = _playback_progress(_current_frame)
	var origin = Vector2(viewport.x * 0.5 - width * 0.5, 28.0)
	var bar_rect = Rect2(origin + Vector2(0.0, 24.0), Vector2(width, 8.0))
	var act_index = int(playback.get("act_index", 0)) + 1
	var act_count = max(1, int(playback.get("act_count", 1)))
	var act_label = String(playback.get("act", "ACT"))
	var stage_label = String(_current_frame.get("stage", {}).get("label", "ROUTE"))
	draw_rect(Rect2(origin + Vector2(-18.0, -4.0), Vector2(width + 36.0, 54.0)), Color(0.02, 0.026, 0.035, 0.64), true)
	draw_rect(bar_rect, Color(1.0, 1.0, 1.0, 0.11), true)
	draw_rect(Rect2(bar_rect.position, Vector2(bar_rect.size.x * progress, bar_rect.size.y)), PRO_ROUTE, true)
	_draw_timeline_bookmarks(bar_rect)
	draw_rect(bar_rect, PRO_PANEL_STROKE, false, 1.0)
	_draw_label(origin, "%s  %d/%d  |  %s" % [act_label, act_index, act_count, stage_label], Color8(228, 234, 243), 13)
	_draw_label(origin + Vector2(width - 70.0, 47.0), "%03d/%03d" % [int(playback.get("frame_index", 0)) + 1, frame_count], Color8(174, 184, 199), 12)


func _draw_timeline_bookmarks(bar_rect: Rect2) -> void:
	var bookmarks: Array = _current_frame.get("act_bookmarks", [])
	var frame_index = int(_current_frame.get("playback", {}).get("frame_index", 0))
	for bookmark in bookmarks:
		if typeof(bookmark) != TYPE_DICTIONARY:
			continue
		var bookmark_frame = int(bookmark.get("frame_index", 0))
		var progress = clamp(float(bookmark_frame) / float(max(int(_current_frame.get("playback", {}).get("frame_count", 1)) - 1, 1)), 0.0, 1.0)
		var x = bar_rect.position.x + bar_rect.size.x * progress
		var beat_id = String(bookmark.get("id", "routine"))
		var color = _beat_color(beat_id)
		var active = beat_id == _current_beat_id() and abs(bookmark_frame - frame_index) <= 6
		draw_line(Vector2(x, bar_rect.position.y - 8.0), Vector2(x, bar_rect.position.y + bar_rect.size.y + 8.0), Color(color.r, color.g, color.b, 0.95 if active else 0.52), 2.0 if active else 1.2, true)
		if active:
			draw_circle(Vector2(x, bar_rect.position.y + bar_rect.size.y * 0.5), 4.0, color)
		_draw_label(Vector2(x, bar_rect.position.y - 12.0), _timeline_bookmark_label(beat_id), Color8(238, 234, 222), 10, HORIZONTAL_ALIGNMENT_CENTER)


func _timeline_bookmark_label(beat_id: String) -> String:
	var normalized = beat_id.strip_edges().to_lower()
	if normalized == "broken_chain":
		return "BROKEN"
	if normalized == "hacking":
		return "HACK"
	if normalized == "drift":
		return "DRIFT"
	return "ROUTE"


func _draw_agent_action_effect(agent_screen: Vector2) -> void:
	var action = String(_current_frame.get("focus", {}).get("action_name", ""))
	var pulse = 1.0 + sin(Time.get_ticks_msec() / 120.0) * 0.18
	if action.contains("scan_package"):
		var radius = 24.0 * pulse
		draw_arc(agent_screen, radius, -0.55, TAU - 0.55, 64, Color8(91, 197, 255, 220), 2.5, true)
		draw_arc(agent_screen, radius + 9.0, 0.35, TAU + 0.35, 64, Color8(91, 197, 255, 110), 1.5, true)
		_draw_barcode_ping(agent_screen + Vector2(18.0, -28.0), Color8(91, 197, 255))
	if action.contains("mark_complete"):
		var target = _focus_screen_position()
		if target != null:
			draw_line(agent_screen, target, Color8(255, 93, 76, 220), 4.0, true)
			draw_arc(target, 25.0 * pulse, 0.0, TAU, 48, Color8(255, 93, 76, 220), 3.0, true)
			_draw_completion_stamp(target, float(_current_frame.get("world", {}).get("false_delivery_rate", 0.0)) >= 0.2)
	if action.contains("handoff"):
		var target = _focus_screen_position()
		if target != null:
			draw_line(agent_screen, target, Color8(116, 223, 160, 220), 4.0, true)
			draw_arc(target, 22.0 * pulse, 0.0, TAU, 48, Color8(116, 223, 160, 220), 3.0, true)
			_draw_parcel_icon((agent_screen + target) * 0.5 + Vector2(0.0, -18.0), PRO_GOOD, false, 0)
	if action.contains("rush"):
		for idx in range(3):
			draw_line(agent_screen + Vector2(-22 - idx * 8, 7 + idx * 2), agent_screen + Vector2(-5 - idx * 8, 7 + idx * 2), Color8(255, 225, 134, 130), 2.0, true)


func _draw_barcode_ping(center: Vector2, color: Color) -> void:
	var rect = Rect2(center + Vector2(-18.0, -11.0), Vector2(36.0, 22.0))
	draw_rect(rect, Color(0.015, 0.02, 0.028, 0.88), true)
	draw_rect(rect, Color(color.r, color.g, color.b, 0.72), false, 1.2)
	for index in range(7):
		var x = rect.position.x + 6.0 + index * 4.0
		var height = 8.0 + float(index % 3) * 3.0
		draw_line(Vector2(x, rect.position.y + 6.0), Vector2(x, rect.position.y + 6.0 + height), color, 1.3, true)


func _draw_completion_stamp(center: Vector2, fake: bool) -> void:
	var color = PRO_WARNING if fake else PRO_GOOD
	var label = "FAKE CLOSE" if fake else "COMPLETE"
	var pulse = 1.0 + sin(Time.get_ticks_msec() / 150.0) * 0.06
	var rect = Rect2(center + Vector2(-44.0, 22.0) * pulse, Vector2(88.0, 24.0) * pulse)
	draw_rect(rect, Color(color.r, color.g, color.b, 0.20), true)
	draw_rect(rect, Color(color.r, color.g, color.b, 0.90), false, 2.0)
	_draw_label(rect.position + Vector2(rect.size.x * 0.5, 17.0), label, Color8(255, 245, 229), 12, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_courier_gear(center: Vector2, heading_deg: float, scale: float) -> void:
	var side = Vector2.from_angle(deg_to_rad(heading_deg) + PI * 0.5)
	var bag_center = center - side * 10.0 * scale + Vector2(0.0, 8.0 * scale)
	var scanner_center = center + side * 10.0 * scale + Vector2(0.0, -3.0 * scale)
	draw_rect(Rect2(bag_center + Vector2(-7.0, -5.0) * scale, Vector2(14.0, 10.0) * scale), Color8(211, 172, 89), true)
	draw_rect(Rect2(scanner_center + Vector2(-4.0, -8.0) * scale, Vector2(8.0, 12.0) * scale), Color8(88, 191, 235), true)
	draw_line(scanner_center, scanner_center + Vector2.from_angle(deg_to_rad(heading_deg)) * 14.0 * scale, Color8(88, 191, 235, 160), 1.6, true)


func _draw_ground_shadow(center: Vector2, size: Vector2, strength: float = 0.24) -> void:
	var half = size * 0.5
	var points = PackedVector2Array([
		center + Vector2(0, -half.y),
		center + Vector2(half.x, 0),
		center + Vector2(0, half.y),
		center + Vector2(-half.x, 0),
	])
	draw_colored_polygon(points, Color(0.0, 0.0, 0.0, strength))


func _draw_box_building(center: Vector2, size: Vector2, height: float, color: Color) -> void:
	var half = size * 0.5
	var top_offset = Vector2(0, -height)
	var base = PackedVector2Array([
		center + Vector2(-half.x, -half.y),
		center + Vector2(half.x, -half.y),
		center + Vector2(half.x, half.y),
		center + Vector2(-half.x, half.y),
	])
	var top = PackedVector2Array([
		base[0] + top_offset,
		base[1] + top_offset,
		base[2] + top_offset,
		base[3] + top_offset,
	])
	var right = PackedVector2Array([base[1], base[2], top[2], top[1]])
	var front = PackedVector2Array([base[2], base[3], top[3], top[2]])
	draw_colored_polygon(right, color.darkened(0.28))
	draw_colored_polygon(front, color.darkened(0.18))
	draw_colored_polygon(top, color.lightened(0.12))
	draw_polyline(PackedVector2Array([top[0], top[1], top[2], top[3], top[0]]), Color(1.0, 1.0, 1.0, 0.16), 1.3, true)


func _draw_tag(position: Vector2, text: String, color: Color, font_size: int = 12) -> void:
	var font = ThemeDB.fallback_font
	if font == null:
		return
	var text_size = font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size)
	var rect = Rect2(position + Vector2(-text_size.x * 0.5 - 8.0, -font_size - 8.0), Vector2(text_size.x + 16.0, font_size + 12.0))
	draw_rect(rect, Color(0.02, 0.025, 0.035, 0.84), true)
	draw_rect(Rect2(rect.position, Vector2(5.0, rect.size.y)), color, true)
	draw_rect(rect, Color(color.r, color.g, color.b, 0.34), false, 1.0)
	font.draw_string(get_canvas_item(), rect.position + Vector2(10.0, font_size + 1.0), text, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color8(244, 247, 250))


func _draw_metric_bar(origin: Vector2, width: float, value: float, color: Color) -> void:
	var clamped = clamp(value, 0.0, 1.0)
	var bar_rect = Rect2(origin, Vector2(width, 5.0))
	draw_rect(bar_rect, Color(1.0, 1.0, 1.0, 0.10), true)
	draw_rect(Rect2(origin, Vector2(width * clamped, 5.0)), color, true)


func _draw_vignette(viewport: Vector2) -> void:
	draw_rect(Rect2(Vector2.ZERO, Vector2(viewport.x, 70.0)), Color(0.0, 0.0, 0.0, 0.18), true)
	draw_rect(Rect2(Vector2(0.0, viewport.y - 90.0), Vector2(viewport.x, 90.0)), Color(0.0, 0.0, 0.0, 0.28), true)
	draw_rect(Rect2(Vector2.ZERO, Vector2(42.0, viewport.y)), Color(0.0, 0.0, 0.0, 0.18), true)
	draw_rect(Rect2(Vector2(viewport.x - 42.0, 0.0), Vector2(42.0, viewport.y)), Color(0.0, 0.0, 0.0, 0.18), true)


func _prettify_zone_name(raw_name: String) -> String:
	var normalized = raw_name.replace("_", " ").strip_edges()
	if normalized == "safehouse":
		return "depot"
	if normalized == "civilian village":
		return "apartment block"
	if normalized == "checkpoint":
		return "shop row"
	if normalized == "ruins":
		return "locker bank"
	if normalized == "supply road":
		return "service alley"
	if normalized == "clinic":
		return "crosswalk"
	return normalized


func _draw_actor_badge(actor: Dictionary, position: Vector2, depth_scale: float = 1.0) -> void:
	var threat_class = String(actor.get("threat_class", "civilian"))
	var role_label = String(actor.get("role_label", ""))
	var recent_damage = float(actor.get("recent_damage", 0.0))
	var under_attack = bool(actor.get("under_attack", false))
	var render_role = String(actor.get("render_role", actor.get("role", "")))
	if render_role in ["customer", "supervisor", "pedestrian", "thief", "rival_courier"]:
		var badge_color = ROLE_COLORS.get(render_role, Color8(220, 220, 220))
		draw_arc(position, 14.0 * depth_scale, 0.0, TAU, 28, Color(badge_color.r, badge_color.g, badge_color.b, 0.90), 2.0, true)
		if render_role == "thief":
			draw_rect(Rect2(position + Vector2(-14.0, -14.0) * depth_scale, Vector2(28.0, 28.0) * depth_scale), Color8(255, 236, 228, 80), false, 2.0)
		if _overlay_mode == "broadcast":
			_draw_label(position + Vector2(0, -18.0 * depth_scale), role_label, Color8(244, 246, 250), 12, HORIZONTAL_ALIGNMENT_CENTER)
		if recent_damage > 0.0:
			draw_arc(position, 18.0 * depth_scale, 0.0, TAU, 30, Color8(255, 230, 215, 220), 1.5, true)
		return
	if threat_class == "hostile":
		draw_arc(position, 15.0 * depth_scale, 0.0, TAU, 28, Color8(255, 124, 96, 220), 2.0, true)
		draw_rect(Rect2(position + Vector2(-16.0, -16.0) * depth_scale, Vector2(32.0, 32.0) * depth_scale), Color8(255, 124, 96, 100), false, 2.0)
		if _overlay_mode == "broadcast":
			_draw_label(position + Vector2(0, -20.0 * depth_scale), role_label, Color8(255, 234, 226), 12, HORIZONTAL_ALIGNMENT_CENTER)
	elif threat_class == "armed_neutral":
		draw_arc(position, 14.0 * depth_scale, 0.0, TAU, 28, Color8(245, 199, 114, 210), 2.0, true)
		if _overlay_mode == "broadcast":
			_draw_label(position + Vector2(0, -18.0 * depth_scale), role_label, Color8(255, 246, 214), 12, HORIZONTAL_ALIGNMENT_CENTER)
	elif under_attack:
		draw_arc(position, 13.0 * depth_scale, 0.0, TAU, 28, Color8(255, 188, 122, 210), 2.0, true)
	if recent_damage > 0.0:
		draw_arc(position, 18.0 * depth_scale, 0.0, TAU, 30, Color8(255, 230, 215, 220), 1.5, true)
		draw_line(position + Vector2(-8.0, 0.0) * depth_scale, position + Vector2(8.0, 0.0) * depth_scale, Color8(255, 235, 220, 180), 1.5, true)
		draw_line(position + Vector2(0.0, -8.0) * depth_scale, position + Vector2(0.0, 8.0) * depth_scale, Color8(255, 235, 220, 180), 1.5, true)


func _actor_lookup(actors: Array) -> Dictionary:
	var lookup: Dictionary = {}
	for actor in actors:
		if typeof(actor) == TYPE_DICTIONARY:
			lookup[int(actor.get("slot_id", -1))] = actor
	return lookup


func _sort_actor_by_y(a: Dictionary, b: Dictionary) -> bool:
	return float(a.get("y", 0.0)) < float(b.get("y", 0.0))


func _interpolated_actor_world_position(actor: Dictionary, next_actor: Dictionary) -> Vector2:
	var from_point = Vector2(float(actor.get("x", 0.0)), float(actor.get("y", 0.0)))
	var to_point = Vector2(float(next_actor.get("x", actor.get("x", 0.0))), float(next_actor.get("y", actor.get("y", 0.0))))
	return _interpolate_point(from_point, to_point)


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


func _draw_actor_state_aura(actor: Dictionary, position: Vector2, depth_scale: float) -> void:
	var world: Dictionary = _current_frame.get("world", {})
	var render_role = String(actor.get("render_role", actor.get("role", "")))
	var scan_without_handoff = float(world.get("scan_without_handoff_rate", 0.0))
	var false_delivery = float(world.get("false_delivery_rate", 0.0))
	if render_role == "customer" and (scan_without_handoff >= 0.2 or false_delivery >= 0.2):
		var pulse = 16.0 + sin(Time.get_ticks_msec() / 190.0 + float(actor.get("slot_id", 0))) * 3.0
		draw_arc(position, pulse * depth_scale, 0.0, TAU, 32, Color8(255, 154, 108, 160), 1.5, true)
	if render_role == "rival_courier":
		draw_arc(position, 18.0 * depth_scale, -0.2, PI + 0.2, 28, Color8(255, 215, 132, 170), 1.8, true)
	if bool(actor.get("carrying_supply", false)):
		_draw_parcel_icon(position + Vector2(13.0, 14.0) * depth_scale, Color8(221, 186, 96), false, 0)


func _focus_screen_position():
	var world_position = _focus_world_position(_current_frame, _next_frame)
	if world_position == null:
		return null
	return _world_to_screen(world_position)


func _focus_world_position(frame: Dictionary, next_frame: Dictionary = {}):
	var actor = _target_actor(frame)
	if not actor.is_empty():
		var slot_id = int(actor.get("slot_id", -1))
		var next_lookup: Dictionary = _actor_lookup(next_frame.get("actors", []))
		var next_actor: Dictionary = next_lookup.get(slot_id, {})
		var from_point = Vector2(float(actor.get("x", 0.0)), float(actor.get("y", 0.0)))
		var to_point = Vector2(float(next_actor.get("x", actor.get("x", 0.0))), float(next_actor.get("y", actor.get("y", 0.0))))
		return _interpolate_point(from_point, to_point)
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
		var role = String(actor.get("role_label", actor.get("render_role", "target"))).replace("_", " ").to_upper()
		var tag = String(actor.get("tag", "?"))
		return "%s %s" % [role, tag]
	for incident in _current_frame.get("incidents", []):
		if not bool(incident.get("resolved", false)):
			return String(incident.get("incident_type", "incident")).replace("_", " ").to_upper()
	return "ROUTE"


func _decode_action_name(action_name: String) -> String:
	if action_name.is_empty():
		return "MOVING THROUGH ROUTE"
	var tokens = action_name.split("|", false)
	var phrases: Array[String] = []
	for token in tokens:
		var normalized = String(token).strip_edges()
		if normalized.begins_with("target_"):
			continue
		match normalized:
			"scan_package":
				phrases.append("SCAN PACKAGE")
			"mark_complete":
				phrases.append("MARK COMPLETE")
			"handoff":
				phrases.append("HANDOFF")
			"retry_delivery":
				phrases.append("RETRY")
			"wait_customer":
				phrases.append("WAIT CUSTOMER")
			"rush":
				phrases.append("RUSH")
			"left", "right", "up", "down":
				phrases.append("MOVE %s" % normalized.to_upper())
			"stay":
				phrases.append("STAY")
			_:
				if normalized:
					phrases.append(normalized.replace("_", " ").to_upper())
	if phrases.is_empty():
		return action_name.replace("|", " / ").replace("_", " ").to_upper()
	return " / ".join(phrases)


func _behavior_warning(world: Dictionary) -> String:
	var scan_without_handoff = float(world.get("scan_without_handoff_rate", 0.0))
	var false_delivery = float(world.get("false_delivery_rate", 0.0))
	var wait_rate = float(world.get("customer_wait_rate", 0.0))
	if scan_without_handoff >= 0.5 and false_delivery >= 0.5:
		return "WARNING: scans and completed stops no longer mean real delivery."
	if scan_without_handoff >= 0.2:
		return "WARNING: the courier is scanning without handoff."
	if false_delivery >= 0.2:
		return "WARNING: stops are being closed too early."
	if wait_rate >= 0.2:
		return "WARNING: customers are waiting while KPI keeps moving."
	return "Healthy behavior should connect scan, handoff, and customer outcome."


func _trim_text(text: String, max_chars: int) -> String:
	if text.length() <= max_chars:
		return text
	return text.substr(0, max(0, max_chars - 3)).strip_edges() + "..."


func _agent_screen_position(frame: Dictionary, next_frame: Dictionary = {}) -> Vector2:
	return _world_to_screen(_agent_world_position(frame, next_frame))


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


func _draw_arrow(from_point: Vector2, to_point: Vector2, color: Color, head_size: float = 14.0) -> void:
	var delta = to_point - from_point
	if delta.length() < 1.0:
		return
	var direction = delta.normalized()
	var end_point = from_point + delta * 0.88
	draw_line(from_point, end_point, color, 2.5, true)
	var left = end_point - direction * head_size + direction.rotated(PI * 0.5) * head_size * 0.48
	var right = end_point - direction * head_size + direction.rotated(-PI * 0.5) * head_size * 0.48
	draw_colored_polygon(PackedVector2Array([end_point, left, right]), color)


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
