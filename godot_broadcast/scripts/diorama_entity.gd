extends Node2D


const TEX_COURIER = preload("res://assets/diorama/courier_card.svg")
const TEX_CUSTOMER = preload("res://assets/diorama/customer_card.svg")
const TEX_SUPERVISOR = preload("res://assets/diorama/supervisor_card.svg")
const TEX_PEDESTRIAN = preload("res://assets/diorama/pedestrian_card.svg")
const TEX_RIVAL = preload("res://assets/diorama/rival_courier_card.svg")
const TEX_THIEF = preload("res://assets/diorama/thief_card.svg")
const TEX_DEPOT_VAN = preload("res://assets/diorama/depot_van.svg")
const TEX_LOCKER = preload("res://assets/diorama/locker_terminal.svg")
const TEX_PARCEL = preload("res://assets/diorama/parcel_box.svg")
const TEX_APARTMENT_ENTRY = preload("res://assets/diorama/apartment_entry.svg")
const TEX_SHOP_STALL = preload("res://assets/diorama/shop_stall.svg")
const TEX_CRATE_STACK = preload("res://assets/diorama/crate_stack.svg")
const TEX_STREET_LAMP = preload("res://assets/diorama/street_lamp.svg")

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

var entity_type: String = "actor"
var label_text: String = ""
var subtitle_text: String = ""
var role_name: String = ""
var kind_name: String = ""
var action_name: String = ""
var presentation_mode: String = "public"
var base_color: Color = Color8(230, 230, 230)
var accent_color: Color = Color8(255, 220, 120)
var scale_factor: float = 1.0
var severity: float = 0.0
var active: bool = false
var heading_deg: float = 0.0
var payload: Dictionary = {}
var _animated: bool = false


func configure(config: Dictionary) -> void:
	entity_type = String(config.get("entity_type", "actor"))
	label_text = String(config.get("label_text", ""))
	subtitle_text = String(config.get("subtitle_text", ""))
	role_name = String(config.get("role", ""))
	kind_name = String(config.get("kind", ""))
	action_name = String(config.get("action_name", ""))
	presentation_mode = String(config.get("presentation_mode", "public"))
	base_color = config.get("base_color", base_color)
	accent_color = config.get("accent_color", base_color)
	scale_factor = max(float(config.get("scale_factor", 1.0)), 0.45)
	severity = clamp(float(config.get("severity", 0.0)), 0.0, 1.0)
	active = bool(config.get("active", false))
	heading_deg = float(config.get("heading_deg", 0.0))
	payload = config.get("payload", {})
	_animated = active or entity_type in ["agent", "incident", "focus", "prop"]
	position = config.get("position", Vector2.ZERO)
	z_index = int(round(position.y))
	visible = true
	set_process(_animated)
	queue_redraw()


func _process(_delta: float) -> void:
	if _animated:
		queue_redraw()


func _draw() -> void:
	match entity_type:
		"zone":
			_draw_zone()
		"incident":
			_draw_incident()
		"agent":
			_draw_agent()
		"focus":
			_draw_focus()
		"prop":
			_draw_prop()
		_:
			_draw_actor()


func _draw_zone() -> void:
	var color = ZONE_COLORS.get(kind_name, base_color)
	var radius = 18.0 * scale_factor
	_draw_shadow(Vector2(0.0, 8.0 * scale_factor), 16.0 * scale_factor, 0.18)
	draw_circle(Vector2.ZERO, radius * 1.2, Color(color.r, color.g, color.b, 0.16))
	draw_arc(Vector2.ZERO, radius * 1.25, 0.0, TAU, 36, Color(color.r, color.g, color.b, 0.62), 2.0, true)
	match kind_name:
		"depot":
			draw_rect(Rect2(Vector2(-20.0, -14.0) * scale_factor, Vector2(40.0, 24.0) * scale_factor), Color(color.r, color.g, color.b, 0.92), true)
			draw_rect(Rect2(Vector2(-12.0, -1.0) * scale_factor, Vector2(24.0, 11.0) * scale_factor), Color8(26, 31, 39), true)
		"apartment_block":
			draw_rect(Rect2(Vector2(-22.0, -20.0) * scale_factor, Vector2(16.0, 40.0) * scale_factor), Color(color.r, color.g, color.b, 0.92), true)
			draw_rect(Rect2(Vector2(0.0, -14.0) * scale_factor, Vector2(18.0, 30.0) * scale_factor), Color(color.r, color.g, color.b, 0.84), true)
		"shop_row":
			draw_rect(Rect2(Vector2(-22.0, -10.0) * scale_factor, Vector2(44.0, 18.0) * scale_factor), Color(color.r, color.g, color.b, 0.9), true)
			draw_colored_polygon(_scaled_points(PackedVector2Array([Vector2(-24.0, -10.0), Vector2(0.0, -20.0), Vector2(24.0, -10.0)]), scale_factor), Color(color.r, color.g, color.b, 1.0))
		"locker_bank":
			for idx in range(3):
				draw_rect(Rect2(Vector2(-18.0 + idx * 12.0, -14.0) * scale_factor, Vector2(10.0, 28.0) * scale_factor), Color(color.r, color.g, color.b, 0.88), true)
		_:
			draw_circle(Vector2.ZERO, 10.0 * scale_factor, Color(color.r, color.g, color.b, 0.88))
	if _show_label():
		_draw_text(Vector2(0.0, 34.0 * scale_factor), label_text, Color8(236, 240, 246), 13, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_incident() -> void:
	var color = INCIDENT_COLORS.get(kind_name, base_color)
	var pulse = 12.0 + 8.0 * severity
	if active:
		draw_arc(Vector2.ZERO, pulse * scale_factor, 0.0, TAU, 30, Color(color.r, color.g, color.b, 0.62), 2.0, true)
	_draw_shadow(Vector2(0.0, 7.0 * scale_factor), 10.0 * scale_factor, 0.12)
	var points = _incident_shape(kind_name, 12.0 * scale_factor)
	draw_colored_polygon(points, color)
	draw_polyline(_closed(points), Color8(24, 26, 34), 2.0, true)
	if _show_label():
		_draw_tag(Vector2(0.0, -22.0 * scale_factor), label_text, color)


func _draw_actor() -> void:
	var color = ROLE_COLORS.get(role_name, base_color)
	var body_offset = _motion_offset()
	var texture = _actor_texture(role_name)
	if texture != null:
		_draw_shadow(body_offset + Vector2(0.0, 12.0 * scale_factor), 14.0 * scale_factor, 0.18)
		if active:
			draw_arc(body_offset, 17.0 * scale_factor, 0.0, TAU, 30, Color(accent_color.r, accent_color.g, accent_color.b, 0.7), 1.8, true)
		_draw_texture_card(texture, body_offset + Vector2(0.0, -2.0) * scale_factor, Vector2(40.0, 46.0) * scale_factor, Color(1, 1, 1, 0.96))
		_draw_actor_accessory(body_offset + Vector2(0.0, 2.0) * scale_factor, color)
		if _show_label() or active:
			_draw_text(body_offset + Vector2(0.0, -29.0 * scale_factor), label_text, Color8(244, 247, 250), 12, HORIZONTAL_ALIGNMENT_CENTER)
		return
	var shape = _translated_points(_transform_shape(_role_shape(role_name, 12.0 * scale_factor), heading_deg), body_offset)
	_draw_shadow(body_offset + Vector2(0.0, 10.0 * scale_factor), 12.0 * scale_factor, 0.2)
	if active:
		draw_arc(body_offset, 17.0 * scale_factor, 0.0, TAU, 30, Color(accent_color.r, accent_color.g, accent_color.b, 0.7), 1.8, true)
	draw_colored_polygon(shape, color)
	draw_polyline(_closed(shape), Color8(22, 26, 34), 2.0, true)
	draw_circle(body_offset + Vector2(0.0, -8.0 * scale_factor), 4.1 * scale_factor, Color8(243, 236, 221, 210))
	_draw_actor_accessory(body_offset, color)
	if bool(payload.get("carrying_supply", false)):
		draw_rect(Rect2(body_offset + Vector2(10.0, 10.0) * scale_factor, Vector2(10.0, 8.0) * scale_factor), Color8(222, 188, 98), true)
	if _show_label() or active:
		_draw_text(body_offset + Vector2(0.0, -21.0 * scale_factor), label_text, Color8(244, 247, 250), 12, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_agent() -> void:
	var color = ROLE_COLORS.get("agent", base_color)
	var body_offset = _motion_offset(1.0)
	var body = _translated_points(_transform_shape(_role_shape("agent", 14.0 * scale_factor), heading_deg), body_offset)
	_draw_shadow(body_offset + Vector2(0.0, 12.0 * scale_factor), 16.0 * scale_factor, 0.24)
	draw_arc(body_offset, 20.0 * scale_factor, 0.0, TAU, 30, Color(accent_color.r, accent_color.g, accent_color.b, 0.58), 2.4, true)
	_draw_texture_card(TEX_COURIER, body_offset + Vector2(0.0, -4.0) * scale_factor, Vector2(48.0, 54.0) * scale_factor, Color(1, 1, 1, 0.98))
	draw_polyline(_closed(body), Color8(23, 27, 35, 80), 1.4, true)
	draw_rect(Rect2(body_offset + Vector2(13.0, 10.0) * scale_factor, Vector2(12.0, 9.0) * scale_factor), Color8(226, 195, 102), true)
	if bool(payload.get("rush", false)):
		for idx in range(3):
			var trail = body_offset - Vector2.RIGHT.rotated(deg_to_rad(heading_deg)) * (10.0 + float(idx) * 7.0) * scale_factor
			draw_line(trail + Vector2(0.0, -8.0) * scale_factor, trail + Vector2(0.0, 8.0) * scale_factor, Color(accent_color.r, accent_color.g, accent_color.b, 0.28 - float(idx) * 0.06), 2.0, true)
	draw_line(body_offset + Vector2(0.0, 0.0), body_offset + Vector2.RIGHT.rotated(deg_to_rad(heading_deg)) * 20.0 * scale_factor, Color8(255, 248, 216, 170), 2.0, true)
	draw_rect(Rect2(body_offset + Vector2(-16.0, -4.0) * scale_factor, Vector2(7.0, 14.0) * scale_factor), Color8(104, 92, 72, 190), true)
	if action_name.contains("scan_package"):
		var scan_target = body_offset + Vector2.RIGHT.rotated(deg_to_rad(heading_deg)) * 42.0 * scale_factor
		draw_arc(scan_target, 12.0 * scale_factor, -PI * 0.7, PI * 0.7, 18, Color8(97, 208, 244, 210), 2.0, true)
		draw_line(body_offset + Vector2(8.0, -2.0) * scale_factor, scan_target, Color8(97, 208, 244, 170), 1.8, true)
	elif action_name.contains("handoff"):
		var handoff_target = body_offset + Vector2.RIGHT.rotated(deg_to_rad(heading_deg)) * 34.0 * scale_factor
		draw_line(body_offset + Vector2(8.0, 2.0) * scale_factor, handoff_target, Color8(130, 222, 165, 205), 3.0, true)
	elif action_name.contains("mark_complete"):
		_draw_stamp("COMPLETE", Color8(255, 245, 220), Color8(224, 101, 86), body_offset + Vector2(28.0, -28.0) * scale_factor)
	if bool(payload.get("broken_chain", false)):
		draw_arc(body_offset, 25.0 * scale_factor, -PI * 0.35, PI * 1.35, 24, Color8(255, 112, 88, 165), 1.8, true)
	if _show_label() or active:
		_draw_tag(body_offset + Vector2(0.0, -26.0 * scale_factor), label_text, accent_color)
	if presentation_mode == "research" and not subtitle_text.is_empty():
		_draw_text(body_offset + Vector2(0.0, 34.0 * scale_factor), subtitle_text, Color8(234, 237, 241), 11, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_focus() -> void:
	var color = accent_color if active else base_color
	var radius = 16.0 * scale_factor
	draw_arc(Vector2.ZERO, radius, 0.0, TAU, 28, Color(color.r, color.g, color.b, 0.92), 2.2, true)
	draw_line(Vector2(-radius - 6.0, 0.0), Vector2(-radius + 2.0, 0.0), color, 2.0, true)
	draw_line(Vector2(radius - 2.0, 0.0), Vector2(radius + 6.0, 0.0), color, 2.0, true)
	draw_line(Vector2(0.0, -radius - 6.0), Vector2(0.0, -radius + 2.0), color, 2.0, true)
	draw_line(Vector2(0.0, radius - 2.0), Vector2(0.0, radius + 6.0), color, 2.0, true)
	if _show_label():
		_draw_text(Vector2(0.0, -22.0 * scale_factor), label_text, Color8(245, 247, 249), 12, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_prop() -> void:
	var prop_kind = String(payload.get("prop_kind", kind_name))
	match prop_kind:
		"depot_van":
			_draw_depot_van()
		"pallet_stack":
			_draw_pallet_stack()
		"worker_pair":
			_draw_worker_pair()
		"locker_terminal":
			_draw_locker_terminal()
		"apartment_entry":
			_draw_apartment_entry()
		"balcony_strip":
			_draw_balcony_strip()
		"waiting_customer":
			_draw_waiting_customer()
		"shop_stall":
			_draw_shop_stall()
		"shop_queue":
			_draw_shop_queue()
		"crate_stack":
			_draw_crate_stack()
		"street_lamp":
			_draw_street_lamp()
		"crosswalk_mark":
			_draw_crosswalk_mark()
		"pedestrian_pair":
			_draw_pedestrian_pair()
		"supervisor_post":
			_draw_supervisor_post()
		"lurker":
			_draw_lurker()
		"route_stop":
			_draw_route_stop()
		"scan_fx":
			_draw_scan_fx()
		"handoff_fx":
			_draw_handoff_fx()
		"fake_close_fx":
			_draw_fake_close_fx()
		"queue_fx":
			_draw_queue_fx()
		_:
			_draw_generic_prop()


func _draw_route_stop() -> void:
	var state = String(payload.get("state", "pending"))
	var pulse = 0.5 + 0.5 * sin(Time.get_ticks_msec() / 190.0)
	var ring_color = accent_color
	var pad_color = Color(base_color.r, base_color.g, base_color.b, 0.26)
	if state == "warning":
		ring_color = Color8(255, 112, 88)
		pad_color = Color(0.35, 0.08, 0.07, 0.32)
	elif state == "done":
		ring_color = Color8(126, 208, 161)
		pad_color = Color(0.09, 0.22, 0.16, 0.26)
	elif state == "pending":
		ring_color = Color8(176, 184, 198)
		pad_color = Color(0.16, 0.18, 0.22, 0.22)
	draw_circle(Vector2.ZERO, 18.0 * scale_factor, pad_color)
	draw_arc(Vector2.ZERO, (18.0 + pulse * 3.0) * scale_factor, 0.0, TAU, 30, Color(ring_color.r, ring_color.g, ring_color.b, 0.82), 2.0, true)
	_draw_shadow(Vector2(0.0, 14.0 * scale_factor), 12.0 * scale_factor, 0.16)
	_draw_texture_card(TEX_PARCEL, Vector2.ZERO, Vector2(24.0, 24.0) * scale_factor, Color(1, 1, 1, 0.96))
	if state == "warning":
		draw_line(Vector2(-9.0, -6.0) * scale_factor, Vector2(9.0, 6.0) * scale_factor, Color8(255, 244, 232), 2.0, true)
		draw_line(Vector2(-9.0, 6.0) * scale_factor, Vector2(9.0, -6.0) * scale_factor, Color8(255, 244, 232), 2.0, true)
	elif state == "done":
		draw_polyline(_scaled_points(PackedVector2Array([Vector2(-8.0, 1.0), Vector2(-1.5, 7.0), Vector2(8.0, -6.0)]), scale_factor), Color8(244, 251, 232), 2.0, true)
	if _show_label():
		_draw_tag(Vector2(0.0, -24.0 * scale_factor), label_text, ring_color)
		if presentation_mode == "research" and not subtitle_text.is_empty():
			_draw_text(Vector2(0.0, 32.0 * scale_factor), subtitle_text, Color8(231, 235, 241), 11, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_scan_fx() -> void:
	var to_point: Vector2 = _relative_target()
	var direction = to_point.normalized() if to_point.length() > 0.01 else Vector2.RIGHT
	var pulse = 0.65 + 0.35 * sin(Time.get_ticks_msec() / 130.0)
	draw_line(Vector2.ZERO, to_point, Color(accent_color.r, accent_color.g, accent_color.b, 0.55), 2.0, true)
	for wave in range(3):
		var offset = 18.0 + wave * 10.0 + pulse * 8.0
		var center = direction * offset
		draw_arc(center, (6.0 + wave * 4.0) * scale_factor, -PI * 0.45, PI * 0.45, 16, Color(accent_color.r, accent_color.g, accent_color.b, 0.72 - wave * 0.18), 1.8, true)
	_draw_tag(Vector2(0.0, -22.0 * scale_factor), label_text, accent_color)


func _draw_handoff_fx() -> void:
	var to_point: Vector2 = _relative_target()
	var pulse = 0.5 + 0.5 * sin(Time.get_ticks_msec() / 150.0)
	draw_line(Vector2.ZERO, to_point, Color(accent_color.r, accent_color.g, accent_color.b, 0.62), 3.0, true)
	var package_pos = to_point * (0.25 + pulse * 0.45)
	var box = Rect2(package_pos + Vector2(-7.0, -5.0) * scale_factor, Vector2(14.0, 10.0) * scale_factor)
	draw_rect(box, Color(accent_color.r, accent_color.g, accent_color.b, 0.92), true)
	draw_rect(box, Color8(24, 27, 34), false, 1.4)
	draw_polyline(PackedVector2Array([to_point + Vector2(-8.0, -4.0) * scale_factor, to_point, to_point + Vector2(-8.0, 4.0) * scale_factor]), Color(accent_color.r, accent_color.g, accent_color.b, 0.88), 2.0, true)
	_draw_tag(Vector2(0.0, -22.0 * scale_factor), label_text, accent_color)


func _draw_fake_close_fx() -> void:
	var broken_chain = bool(payload.get("broken_chain", false))
	var stamp_color = Color8(255, 112, 88) if broken_chain else accent_color
	var pulse = 0.82 + 0.18 * sin(Time.get_ticks_msec() / 160.0)
	var rect = Rect2(Vector2(-34.0, -13.0) * scale_factor, Vector2(68.0, 26.0) * scale_factor)
	draw_rect(rect, Color(stamp_color.r, stamp_color.g, stamp_color.b, 0.12 + pulse * 0.08), true)
	draw_rect(rect, Color(stamp_color.r, stamp_color.g, stamp_color.b, 0.94), false, 2.0)
	_draw_text(Vector2(0.0, 5.0 * scale_factor), label_text, Color8(255, 244, 230), 11, HORIZONTAL_ALIGNMENT_CENTER)
	if broken_chain:
		draw_line(Vector2(-30.0, -10.0) * scale_factor, Vector2(30.0, 10.0) * scale_factor, Color8(255, 242, 232, 220), 2.0, true)
		draw_line(Vector2(-30.0, 10.0) * scale_factor, Vector2(30.0, -10.0) * scale_factor, Color8(255, 242, 232, 220), 2.0, true)


func _draw_queue_fx() -> void:
	var pulse = 0.5 + 0.5 * sin(Time.get_ticks_msec() / 180.0)
	for idx in range(3):
		var alpha = 0.32 + clamp(pulse - float(idx) * 0.14, 0.0, 0.5)
		draw_circle(Vector2((float(idx) - 1.0) * 10.0 * scale_factor, 0.0), (4.0 + float(idx)) * scale_factor, Color(accent_color.r, accent_color.g, accent_color.b, alpha))
	_draw_tag(Vector2(0.0, -18.0 * scale_factor), label_text, accent_color)


func _draw_generic_prop() -> void:
	draw_circle(Vector2.ZERO, 10.0 * scale_factor, Color(base_color.r, base_color.g, base_color.b, 0.88))
	if _show_label():
		_draw_tag(Vector2(0.0, -18.0 * scale_factor), label_text, accent_color)


func _draw_depot_van() -> void:
	_draw_shadow(Vector2(0.0, 16.0 * scale_factor), 18.0 * scale_factor, 0.18)
	_draw_texture_card(TEX_DEPOT_VAN, Vector2.ZERO, Vector2(64.0, 38.0) * scale_factor, Color(1, 1, 1, 0.98))


func _draw_pallet_stack() -> void:
	for row in range(2):
		draw_rect(Rect2(Vector2(-14.0 + row * 8.0, -6.0 - row * 8.0) * scale_factor, Vector2(22.0, 10.0) * scale_factor), Color(base_color.r, base_color.g, base_color.b, 0.92), true)
		draw_rect(Rect2(Vector2(-14.0 + row * 8.0, 6.0 - row * 8.0) * scale_factor, Vector2(22.0, 3.0) * scale_factor), Color8(98, 72, 48, 180), true)


func _draw_locker_terminal() -> void:
	_draw_texture_card(TEX_LOCKER, Vector2.ZERO, Vector2(40.0, 42.0) * scale_factor, Color(1, 1, 1, 0.98))


func _draw_apartment_entry() -> void:
	_draw_texture_card(TEX_APARTMENT_ENTRY, Vector2.ZERO, Vector2(44.0, 46.0) * scale_factor, Color(1, 1, 1, 0.98))


func _draw_balcony_strip() -> void:
	draw_rect(Rect2(Vector2(-20.0, -6.0) * scale_factor, Vector2(40.0, 12.0) * scale_factor), Color(base_color.r, base_color.g, base_color.b, 0.78), true)
	for idx in range(4):
		draw_line(Vector2(-16.0 + idx * 10.0, -6.0) * scale_factor, Vector2(-16.0 + idx * 10.0, 6.0) * scale_factor, Color8(42, 48, 56), 1.5, true)


func _draw_shop_stall() -> void:
	_draw_texture_card(TEX_SHOP_STALL, Vector2.ZERO, Vector2(56.0, 48.0) * scale_factor, Color(1, 1, 1, 0.98))


func _draw_crate_stack() -> void:
	_draw_texture_card(TEX_CRATE_STACK, Vector2.ZERO, Vector2(44.0, 40.0) * scale_factor, Color(1, 1, 1, 0.98))


func _draw_street_lamp() -> void:
	_draw_texture_card(TEX_STREET_LAMP, Vector2(2.0, -18.0) * scale_factor, Vector2(36.0, 62.0) * scale_factor, Color(1, 1, 1, 0.98))


func _draw_crosswalk_mark() -> void:
	for idx in range(4):
		draw_rect(Rect2(Vector2(-18.0 + idx * 10.0, -6.0) * scale_factor, Vector2(6.0, 16.0) * scale_factor), Color(base_color.r, base_color.g, base_color.b, 0.84), true)


func _draw_worker_pair() -> void:
	_draw_micro_person(Vector2(-9.0, 0.0) * scale_factor, Color8(196, 201, 208), 0.82)
	_draw_micro_person(Vector2(8.0, 2.0) * scale_factor, Color8(176, 184, 196), 0.78)
	draw_rect(Rect2(Vector2(-14.0, 12.0) * scale_factor, Vector2(28.0, 4.0) * scale_factor), Color8(92, 74, 56, 170), true)


func _draw_waiting_customer() -> void:
	_draw_texture_card(TEX_CUSTOMER, Vector2(0.0, -4.0) * scale_factor, Vector2(30.0, 34.0) * scale_factor, Color(1, 1, 1, 0.96))
	draw_circle(Vector2(14.0, -18.0) * scale_factor, 6.0 * scale_factor, Color8(255, 188, 122, 170))
	draw_circle(Vector2(22.0, -24.0) * scale_factor, 4.0 * scale_factor, Color8(255, 188, 122, 120))
	if bool(payload.get("wait_pressure", false)):
		draw_line(Vector2(14.0, -5.0) * scale_factor, Vector2(14.0, -12.0) * scale_factor, Color8(255, 188, 122), 1.8, true)
		draw_arc(Vector2(14.0, -14.0) * scale_factor, 4.0 * scale_factor, 0.0, TAU, 16, Color8(255, 188, 122), 1.5, true)


func _draw_shop_queue() -> void:
	_draw_micro_person(Vector2(-10.0, 2.0) * scale_factor, Color8(196, 201, 208), 0.76)
	_draw_micro_person(Vector2(0.0, 0.0) * scale_factor, Color8(184, 226, 179), 0.84)
	_draw_micro_person(Vector2(12.0, 2.0) * scale_factor, Color8(238, 184, 96), 0.76)
	draw_arc(Vector2(0.0, -18.0) * scale_factor, 16.0 * scale_factor, PI, TAU, 18, Color8(231, 180, 118, 180), 1.6, true)


func _draw_pedestrian_pair() -> void:
	_draw_texture_card(TEX_PEDESTRIAN, Vector2(-9.0, -2.0) * scale_factor, Vector2(24.0, 28.0) * scale_factor, Color(1, 1, 1, 0.94))
	_draw_texture_card(TEX_PEDESTRIAN, Vector2(9.0, -4.0) * scale_factor, Vector2(23.0, 27.0) * scale_factor, Color(0.94, 0.96, 1.0, 0.88))


func _draw_supervisor_post() -> void:
	_draw_texture_card(TEX_SUPERVISOR, Vector2(0.0, -3.0) * scale_factor, Vector2(30.0, 34.0) * scale_factor, Color(1, 1, 1, 0.96))
	draw_rect(Rect2(Vector2(10.0, -2.0) * scale_factor, Vector2(8.0, 12.0) * scale_factor), Color8(244, 233, 208, 220), true)
	draw_line(Vector2(10.0, 3.0) * scale_factor, Vector2(18.0, 3.0) * scale_factor, Color8(98, 122, 148), 1.2, true)


func _draw_lurker() -> void:
	var pulse = 0.4 + 0.6 * sin(Time.get_ticks_msec() / 180.0)
	_draw_texture_card(TEX_THIEF, Vector2(0.0, -4.0) * scale_factor, Vector2(30.0, 34.0) * scale_factor, Color(1, 1, 1, 0.95))
	draw_arc(Vector2.ZERO, 11.0 * scale_factor, PI * 0.15, PI * 0.85, 16, Color8(255, 112, 88, 140 + int(pulse * 50.0)), 1.5, true)
	draw_circle(Vector2(-4.0, -8.0) * scale_factor, 1.4 * scale_factor, Color8(255, 144, 122, 220))
	draw_circle(Vector2(4.0, -8.0) * scale_factor, 1.4 * scale_factor, Color8(255, 144, 122, 220))


func _draw_actor_accessory(body_offset: Vector2, color: Color) -> void:
	match role_name:
		"supervisor", "ally":
			draw_rect(Rect2(body_offset + Vector2(8.0, 0.0) * scale_factor, Vector2(9.0, 12.0) * scale_factor), Color8(244, 233, 208, 220), true)
			draw_line(body_offset + Vector2(8.0, 3.0) * scale_factor, body_offset + Vector2(17.0, 3.0) * scale_factor, Color8(98, 122, 148), 1.2, true)
		"thief", "hostile":
			draw_rect(Rect2(body_offset + Vector2(-12.0, -3.0) * scale_factor, Vector2(8.0, 14.0) * scale_factor), Color8(80, 58, 48, 210), true)
			draw_arc(body_offset + Vector2(0.0, -10.0) * scale_factor, 8.0 * scale_factor, PI, TAU, 16, Color(color.r * 0.7, color.g * 0.7, color.b * 0.7, 0.9), 2.0, true)
		"rival_courier", "armed_neutral", "smuggler", "militia":
			draw_rect(Rect2(body_offset + Vector2(-16.0, 8.0) * scale_factor, Vector2(12.0, 9.0) * scale_factor), Color8(222, 188, 98, 210), true)
			draw_line(body_offset + Vector2(-4.0, 12.0) * scale_factor, body_offset + Vector2(8.0, 12.0) * scale_factor, Color8(56, 62, 70), 1.6, true)
		"customer", "civilian":
			if bool(payload.get("wait_pressure", false)) and active:
				draw_circle(body_offset + Vector2(12.0, -18.0) * scale_factor, 5.5 * scale_factor, Color8(255, 188, 122, 170))
				draw_circle(body_offset + Vector2(20.0, -24.0) * scale_factor, 3.8 * scale_factor, Color8(255, 188, 122, 120))
		_:
			pass


func _actor_texture(role: String):
	match role:
		"customer", "civilian":
			return TEX_CUSTOMER
		"supervisor", "ally":
			return TEX_SUPERVISOR
		"pedestrian", "scavenger":
			return TEX_PEDESTRIAN
		"rival_courier", "armed_neutral", "militia":
			return TEX_RIVAL
		"thief", "hostile", "smuggler":
			return TEX_THIEF
		_:
			return null


func _draw_micro_person(center: Vector2, color: Color, size_scale: float) -> void:
	var radius = 3.6 * scale_factor * size_scale
	draw_circle(center + Vector2(0.0, -6.0) * scale_factor * size_scale, radius, Color(0.95, 0.93, 0.88, 0.9))
	var body = PackedVector2Array([
		center + Vector2(0.0, -2.0) * scale_factor * size_scale,
		center + Vector2(5.0, 8.0) * scale_factor * size_scale,
		center + Vector2(0.0, 12.0) * scale_factor * size_scale,
		center + Vector2(-5.0, 8.0) * scale_factor * size_scale,
	])
	draw_colored_polygon(body, Color(color.r, color.g, color.b, 0.92))
	draw_polyline(_closed(body), Color8(28, 32, 38), 1.2, true)


func _draw_texture_card(texture: Texture2D, center: Vector2, size: Vector2, modulate_color: Color = Color(1, 1, 1, 1)) -> void:
	if texture == null:
		return
	var rect = Rect2(center - size * 0.5, size)
	draw_texture_rect(texture, rect, false, modulate_color)


func _relative_target() -> Vector2:
	var to_point = payload.get("to", position)
	if to_point is Vector2:
		return to_point - position
	return Vector2(42.0, -20.0)


func _motion_offset(weight: float = 0.72) -> Vector2:
	if not bool(payload.get("moving", false)):
		return Vector2.ZERO
	var t = Time.get_ticks_msec() / 120.0
	return Vector2(0.0, sin(t) * 1.8 * scale_factor * weight)


func _show_label() -> bool:
	return presentation_mode == "research" or active


func _draw_shadow(center: Vector2, radius: float, alpha: float) -> void:
	draw_circle(center, radius, Color(0.0, 0.0, 0.0, alpha))


func _draw_tag(center: Vector2, text: String, color: Color) -> void:
	if text.is_empty():
		return
	var width = max(92.0, float(text.length() * 8 + 18))
	var rect = Rect2(center + Vector2(-width * 0.5, -11.0), Vector2(width, 22.0))
	draw_rect(rect, Color(0.03, 0.04, 0.05, 0.82), true)
	draw_rect(rect, Color(color.r, color.g, color.b, 0.62), false, 1.5)
	_draw_text(center + Vector2(0.0, 4.0), text, Color8(247, 248, 250), 11, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_stamp(text: String, text_color: Color, stamp_color: Color, center: Vector2) -> void:
	var rect = Rect2(center + Vector2(-28.0, -11.0), Vector2(56.0, 22.0))
	draw_rect(rect, Color(stamp_color.r, stamp_color.g, stamp_color.b, 0.15), true)
	draw_rect(rect, Color(stamp_color.r, stamp_color.g, stamp_color.b, 0.88), false, 2.0)
	_draw_text(center + Vector2(0.0, 4.0), text, text_color, 11, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_text(position_local: Vector2, text: String, color: Color, font_size: int, alignment: HorizontalAlignment) -> void:
	var font = ThemeDB.fallback_font
	if font == null or text.is_empty():
		return
	font.draw_string(get_canvas_item(), position_local, text, alignment, -1.0, font_size, color)


func _transform_shape(points: PackedVector2Array, angle_deg: float) -> PackedVector2Array:
	var transformed = PackedVector2Array()
	var angle = deg_to_rad(angle_deg)
	var cosine = cos(angle)
	var sine = sin(angle)
	for point in points:
		transformed.append(Vector2(point.x * cosine - point.y * sine, point.x * sine + point.y * cosine))
	return transformed


func _translated_points(points: PackedVector2Array, delta: Vector2) -> PackedVector2Array:
	var translated = PackedVector2Array()
	for point in points:
		translated.append(point + delta)
	return translated


func _closed(points: PackedVector2Array) -> PackedVector2Array:
	var result = PackedVector2Array(points)
	if points.is_empty():
		return result
	result.append(points[0])
	return result


func _scaled_points(points: PackedVector2Array, amount: float) -> PackedVector2Array:
	var scaled = PackedVector2Array()
	for point in points:
		scaled.append(point * amount)
	return scaled


func _role_shape(role: String, size: float) -> PackedVector2Array:
	match role:
		"agent":
			return PackedVector2Array([Vector2(0.0, -size), Vector2(size * 0.74, -size * 0.18), Vector2(size * 0.58, size * 0.92), Vector2(0.0, size * 0.58), Vector2(-size * 0.58, size * 0.92), Vector2(-size * 0.74, -size * 0.18)])
		"customer", "civilian":
			return PackedVector2Array([Vector2(0.0, -size), Vector2(size * 0.44, -size * 0.18), Vector2(size * 0.28, size), Vector2(-size * 0.28, size), Vector2(-size * 0.44, -size * 0.18)])
		"supervisor", "ally":
			return PackedVector2Array([Vector2(-size * 0.68, -size * 0.56), Vector2(size * 0.68, -size * 0.56), Vector2(size * 0.92, size * 0.16), Vector2(0.0, size), Vector2(-size * 0.92, size * 0.16)])
		"pedestrian", "scavenger":
			return PackedVector2Array([Vector2(-size * 0.54, -size * 0.3), Vector2(0.0, -size), Vector2(size * 0.58, -size * 0.2), Vector2(size * 0.4, size * 0.9), Vector2(-size * 0.5, size)])
		"thief", "hostile":
			return PackedVector2Array([Vector2(0.0, -size), Vector2(size, -size * 0.06), Vector2(size * 0.36, size), Vector2(-size * 0.36, size), Vector2(-size, -size * 0.06)])
		"rival_courier", "armed_neutral", "smuggler", "militia":
			return PackedVector2Array([Vector2(0.0, -size), Vector2(size * 0.86, -size * 0.16), Vector2(size * 0.72, size * 0.46), Vector2(0.0, size), Vector2(-size * 0.72, size * 0.46), Vector2(-size * 0.86, -size * 0.16)])
		_:
			return PackedVector2Array([Vector2(0.0, -size), Vector2(size, 0.0), Vector2(0.0, size), Vector2(-size, 0.0)])


func _incident_shape(kind: String, size: float) -> PackedVector2Array:
	match kind:
		"customer_absent", "locker_retry":
			return PackedVector2Array([Vector2(-size, -size * 0.7), Vector2(size, -size * 0.7), Vector2(size, size * 0.7), Vector2(-size, size * 0.7)])
		"address_mismatch", "route_breach":
			return PackedVector2Array([Vector2(-size, size), Vector2(0.0, -size), Vector2(size, size)])
		"urgent_parcel", "false_alarm":
			return PackedVector2Array([Vector2(0.0, -size), Vector2(size * 0.72, 0.0), Vector2(0.0, size), Vector2(-size * 0.72, 0.0)])
		"damaged_parcel":
			return PackedVector2Array([Vector2(-size, -size * 0.4), Vector2(0.0, -size), Vector2(size, -size * 0.4), Vector2(size * 0.62, size), Vector2(-size * 0.62, size)])
		"aid_drop", "escort_request":
			return PackedVector2Array([Vector2(-size, -size * 0.62), Vector2(size, -size * 0.62), Vector2(size, size * 0.62), Vector2(-size, size * 0.62)])
		_:
			return PackedVector2Array([Vector2(0.0, -size), Vector2(size * 0.92, size * 0.82), Vector2(-size * 0.92, size * 0.82)])
