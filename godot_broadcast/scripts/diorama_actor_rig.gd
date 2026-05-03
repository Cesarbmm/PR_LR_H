extends Node2D


const TEX_PARCEL = preload("res://assets/diorama/parcel_box.svg")
const DioramaSpriteCatalog = preload("res://scripts/diorama_sprite_catalog.gd")

const ROLE_STYLES = {
	"agent": {
		"body": Color8(239, 229, 112),
		"trim": Color8(205, 173, 78),
		"limb": Color8(77, 84, 97),
		"accent": Color8(255, 220, 122),
		"backpack": true,
	},
	"customer": {
		"body": Color8(184, 226, 179),
		"trim": Color8(123, 174, 121),
		"limb": Color8(74, 90, 82),
		"accent": Color8(255, 188, 122),
	},
	"civilian": {
		"body": Color8(184, 226, 179),
		"trim": Color8(123, 174, 121),
		"limb": Color8(74, 90, 82),
		"accent": Color8(255, 188, 122),
	},
	"supervisor": {
		"body": Color8(111, 175, 236),
		"trim": Color8(82, 129, 181),
		"limb": Color8(66, 78, 94),
		"accent": Color8(245, 234, 206),
		"clipboard": true,
	},
	"ally": {
		"body": Color8(111, 175, 236),
		"trim": Color8(82, 129, 181),
		"limb": Color8(66, 78, 94),
		"accent": Color8(245, 234, 206),
		"clipboard": true,
	},
	"pedestrian": {
		"body": Color8(196, 201, 208),
		"trim": Color8(146, 153, 162),
		"limb": Color8(86, 92, 102),
		"accent": Color8(220, 225, 231),
	},
	"scavenger": {
		"body": Color8(150, 160, 166),
		"trim": Color8(116, 124, 129),
		"limb": Color8(78, 84, 90),
		"accent": Color8(214, 220, 228),
	},
	"rival_courier": {
		"body": Color8(238, 184, 96),
		"trim": Color8(194, 140, 68),
		"limb": Color8(85, 78, 68),
		"accent": Color8(255, 230, 170),
		"backpack": true,
		"parcel": true,
	},
	"armed_neutral": {
		"body": Color8(238, 184, 96),
		"trim": Color8(194, 140, 68),
		"limb": Color8(85, 78, 68),
		"accent": Color8(255, 230, 170),
		"backpack": true,
	},
	"thief": {
		"body": Color8(232, 115, 95),
		"trim": Color8(148, 70, 58),
		"limb": Color8(66, 50, 55),
		"accent": Color8(255, 148, 122),
		"hood": true,
		"parcel": true,
	},
	"hostile": {
		"body": Color8(232, 115, 95),
		"trim": Color8(148, 70, 58),
		"limb": Color8(66, 50, 55),
		"accent": Color8(255, 148, 122),
		"hood": true,
	},
	"smuggler": {
		"body": Color8(192, 117, 200),
		"trim": Color8(130, 81, 136),
		"limb": Color8(68, 56, 72),
		"accent": Color8(232, 185, 238),
		"hood": true,
		"parcel": true,
	},
	"militia": {
		"body": Color8(193, 144, 82),
		"trim": Color8(129, 94, 56),
		"limb": Color8(74, 66, 56),
		"accent": Color8(236, 210, 160),
		"parcel": true,
	},
}

const SKIN_COLOR = Color8(239, 228, 210)
const OUTLINE_COLOR = Color8(28, 34, 42)

var role_name: String = "customer"
var label_text: String = ""
var subtitle_text: String = ""
var action_name: String = ""
var presentation_mode: String = "public"
var accent_color: Color = Color8(255, 220, 122)
var scale_factor: float = 1.0
var severity: float = 0.0
var active: bool = false
var heading_deg: float = 0.0
var payload: Dictionary = {}

var _phase_offset: float = 0.0
var _built: bool = false

var _body_root: Node2D
var _shadow: Polygon2D
var _ring: Polygon2D
var _hip_left: Node2D
var _hip_right: Node2D
var _leg_left: Polygon2D
var _leg_right: Polygon2D
var _torso: Polygon2D
var _torso_trim: Polygon2D
var _backpack: Polygon2D
var _shoulder_left: Node2D
var _shoulder_right: Node2D
var _arm_left: Polygon2D
var _arm_right: Polygon2D
var _clipboard: Polygon2D
var _device: Polygon2D
var _head_root: Node2D
var _head: Polygon2D
var _hair: Polygon2D
var _hood: Polygon2D
var _eye_left: Polygon2D
var _eye_right: Polygon2D
var _parcel: Sprite2D
var _sprite_overlay: AnimatedSprite2D
var _fx_root: Node2D
var _speed_fx_left: Polygon2D
var _speed_fx_right: Polygon2D
var _scan_beam: Line2D
var _scan_glow: Polygon2D
var _handoff_beam: Line2D
var _handoff_box: Polygon2D
var _wait_bubble_large: Polygon2D
var _wait_bubble_small: Polygon2D
var _alert_crown: Line2D


func _ready() -> void:
	_ensure_built()
	_apply_style()
	_update_pose()


func configure(config: Dictionary) -> void:
	role_name = String(config.get("role", role_name))
	label_text = String(config.get("label_text", label_text))
	subtitle_text = String(config.get("subtitle_text", subtitle_text))
	action_name = String(config.get("action_name", action_name))
	presentation_mode = String(config.get("presentation_mode", presentation_mode))
	accent_color = config.get("accent_color", accent_color)
	scale_factor = max(float(config.get("scale_factor", scale_factor)), 0.45)
	severity = clamp(float(config.get("severity", severity)), 0.0, 1.0)
	active = bool(config.get("active", active))
	heading_deg = float(config.get("heading_deg", heading_deg))
	payload = config.get("payload", payload)
	position = config.get("position", position)
	z_index = int(round(position.y))
	scale = Vector2.ONE * scale_factor
	visible = true
	_ensure_built()
	_apply_style()
	_update_pose()
	set_process(true)
	queue_redraw()


func _process(_delta: float) -> void:
	_update_pose()
	queue_redraw()


func _draw() -> void:
	var body_offset = _body_root.position
	if active:
		draw_arc(body_offset, 17.5, 0.0, TAU, 32, Color(accent_color.r, accent_color.g, accent_color.b, 0.72), 1.6, true)
	elif presentation_mode == "research":
		draw_arc(body_offset, 16.0, 0.0, TAU, 28, Color(1.0, 1.0, 1.0, 0.12), 1.0, true)
	if _show_label():
		_draw_tag(body_offset + Vector2(0.0, -32.0), label_text, accent_color)
		if presentation_mode == "research" and not subtitle_text.is_empty():
			_draw_text(body_offset + Vector2(0.0, 39.0), subtitle_text, Color8(232, 236, 241), 11, HORIZONTAL_ALIGNMENT_CENTER)


func _ensure_built() -> void:
	if _built:
		return
	_phase_offset = float(abs(hash(str(get_instance_id()))) % 1000) / 137.0
	_shadow = Polygon2D.new()
	_shadow.polygon = _ellipse_points(15.0, 6.0, 18)
	_shadow.color = Color(0.0, 0.0, 0.0, 0.22)
	_shadow.position = Vector2(0.0, 18.0)
	add_child(_shadow)
	_ring = Polygon2D.new()
	_ring.polygon = _ellipse_points(18.0, 7.0, 20)
	_ring.color = Color(1.0, 1.0, 1.0, 0.0)
	add_child(_ring)
	_body_root = Node2D.new()
	add_child(_body_root)
	_hip_left = Node2D.new()
	_hip_left.position = Vector2(-6.0, 8.0)
	_body_root.add_child(_hip_left)
	_leg_left = _make_rect(Vector2(5.0, 18.0), Color.WHITE)
	_hip_left.add_child(_leg_left)
	_hip_right = Node2D.new()
	_hip_right.position = Vector2(6.0, 8.0)
	_body_root.add_child(_hip_right)
	_leg_right = _make_rect(Vector2(5.0, 18.0), Color.WHITE)
	_hip_right.add_child(_leg_right)
	_backpack = _make_poly(PackedVector2Array([Vector2(-9.0, -10.0), Vector2(-2.0, -12.0), Vector2(-2.0, 7.0), Vector2(-9.0, 5.0)]), Color.WHITE)
	_backpack.visible = false
	_body_root.add_child(_backpack)
	_shoulder_left = Node2D.new()
	_shoulder_left.position = Vector2(-12.0, -9.0)
	_body_root.add_child(_shoulder_left)
	_arm_left = _make_rect(Vector2(4.0, 18.0), Color.WHITE)
	_shoulder_left.add_child(_arm_left)
	_shoulder_right = Node2D.new()
	_shoulder_right.position = Vector2(12.0, -9.0)
	_body_root.add_child(_shoulder_right)
	_arm_right = _make_rect(Vector2(4.0, 18.0), Color.WHITE)
	_shoulder_right.add_child(_arm_right)
	_clipboard = _make_rect(Vector2(7.0, 10.0), Color.WHITE)
	_clipboard.position = Vector2(-1.0, 16.0)
	_clipboard.visible = false
	_shoulder_left.add_child(_clipboard)
	_device = _make_rect(Vector2(4.5, 6.5), Color.WHITE)
	_device.position = Vector2(1.5, 16.0)
	_device.visible = false
	_shoulder_right.add_child(_device)
	_parcel = Sprite2D.new()
	_parcel.texture = TEX_PARCEL
	_parcel.centered = true
	_parcel.scale = Vector2(0.12, 0.12)
	_parcel.position = Vector2(3.0, 18.0)
	_parcel.visible = false
	_shoulder_right.add_child(_parcel)
	_torso = _make_poly(PackedVector2Array([Vector2(-13.0, -17.0), Vector2(13.0, -17.0), Vector2(15.0, 8.0), Vector2(0.0, 22.0), Vector2(-15.0, 8.0)]), Color.WHITE)
	_body_root.add_child(_torso)
	_torso_trim = _make_poly(PackedVector2Array([Vector2(-8.0, -6.0), Vector2(8.0, -6.0), Vector2(11.0, 8.0), Vector2(-11.0, 8.0)]), Color.WHITE)
	_body_root.add_child(_torso_trim)
	_head_root = Node2D.new()
	_head_root.position = Vector2(0.0, -25.0)
	_body_root.add_child(_head_root)
	_head = _make_poly(_ellipse_points(8.0, 9.0, 18), SKIN_COLOR)
	_head_root.add_child(_head)
	_hair = _make_poly(PackedVector2Array([Vector2(-8.0, -6.0), Vector2(0.0, -11.0), Vector2(8.0, -6.0), Vector2(5.0, -1.0), Vector2(-5.0, -1.0)]), Color8(54, 46, 42))
	_head_root.add_child(_hair)
	_hood = _make_poly(PackedVector2Array([Vector2(-10.0, -7.0), Vector2(0.0, -14.0), Vector2(10.0, -7.0), Vector2(8.0, 9.0), Vector2(-8.0, 9.0)]), Color8(88, 60, 62))
	_hood.visible = false
	_head_root.add_child(_hood)
	_eye_left = _make_poly(_ellipse_points(1.1, 1.4, 10), OUTLINE_COLOR)
	_eye_left.position = Vector2(-3.0, -1.0)
	_head_root.add_child(_eye_left)
	_eye_right = _make_poly(_ellipse_points(1.1, 1.4, 10), OUTLINE_COLOR)
	_eye_right.position = Vector2(3.0, -1.0)
	_head_root.add_child(_eye_right)
	_fx_root = Node2D.new()
	_fx_root.name = "FXRoot"
	_body_root.add_child(_fx_root)
	_speed_fx_left = _make_poly(PackedVector2Array([Vector2(-20.0, 2.0), Vector2(-38.0, -4.0), Vector2(-24.0, -9.0)]), Color8(255, 222, 148, 72))
	_speed_fx_left.visible = false
	_fx_root.add_child(_speed_fx_left)
	_speed_fx_right = _make_poly(PackedVector2Array([Vector2(-15.0, 10.0), Vector2(-31.0, 6.0), Vector2(-19.0, 1.0)]), Color8(255, 222, 148, 56))
	_speed_fx_right.visible = false
	_fx_root.add_child(_speed_fx_right)
	_scan_beam = Line2D.new()
	_scan_beam.width = 2.6
	_scan_beam.default_color = Color8(97, 208, 244, 210)
	_scan_beam.antialiased = true
	_scan_beam.visible = false
	_fx_root.add_child(_scan_beam)
	_scan_glow = _make_poly(_ellipse_points(8.0, 8.0, 18), Color8(97, 208, 244, 84))
	_scan_glow.visible = false
	_fx_root.add_child(_scan_glow)
	_handoff_beam = Line2D.new()
	_handoff_beam.width = 3.0
	_handoff_beam.default_color = Color8(130, 222, 165, 214)
	_handoff_beam.antialiased = true
	_handoff_beam.visible = false
	_fx_root.add_child(_handoff_beam)
	_handoff_box = _make_poly(PackedVector2Array([Vector2(-6.0, -5.0), Vector2(6.0, -5.0), Vector2(6.0, 5.0), Vector2(-6.0, 5.0)]), Color8(219, 183, 108, 226))
	_handoff_box.visible = false
	_fx_root.add_child(_handoff_box)
	_wait_bubble_large = _make_poly(_ellipse_points(8.0, 8.0, 16), Color8(255, 188, 122, 58))
	_wait_bubble_large.visible = false
	_fx_root.add_child(_wait_bubble_large)
	_wait_bubble_small = _make_poly(_ellipse_points(5.0, 5.0, 14), Color8(255, 188, 122, 44))
	_wait_bubble_small.visible = false
	_fx_root.add_child(_wait_bubble_small)
	_alert_crown = Line2D.new()
	_alert_crown.width = 2.0
	_alert_crown.default_color = Color8(255, 118, 96, 220)
	_alert_crown.antialiased = true
	_alert_crown.visible = false
	_fx_root.add_child(_alert_crown)
	_sprite_overlay = AnimatedSprite2D.new()
	_sprite_overlay.name = "SpriteOverlay"
	_sprite_overlay.centered = true
	_sprite_overlay.position = Vector2(0.0, -2.0)
	_sprite_overlay.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	_sprite_overlay.modulate = Color(1.0, 1.0, 1.0, 0.96)
	_body_root.add_child(_sprite_overlay)
	_built = true


func _apply_style() -> void:
	if not _built:
		return
	var style: Dictionary = ROLE_STYLES.get(role_name, ROLE_STYLES["customer"])
	var body_color: Color = style.get("body", Color8(184, 226, 179))
	var trim_color: Color = style.get("trim", body_color.darkened(0.16))
	var limb_color: Color = style.get("limb", Color8(74, 90, 82))
	var role_accent: Color = style.get("accent", accent_color)
	var rig_alpha = 0.36 if presentation_mode == "public" else 0.58
	if accent_color.a > 0.0:
		role_accent = accent_color
	_torso.color = Color(body_color.r, body_color.g, body_color.b, rig_alpha)
	_torso_trim.color = Color(trim_color.r, trim_color.g, trim_color.b, rig_alpha)
	_leg_left.color = Color(limb_color.r, limb_color.g, limb_color.b, rig_alpha)
	_leg_right.color = Color(limb_color.r, limb_color.g, limb_color.b, rig_alpha)
	_arm_left.color = Color(limb_color.r, limb_color.g, limb_color.b, rig_alpha)
	_arm_right.color = Color(limb_color.r, limb_color.g, limb_color.b, rig_alpha)
	_backpack.color = Color(trim_color.darkened(0.08).r, trim_color.darkened(0.08).g, trim_color.darkened(0.08).b, rig_alpha)
	_clipboard.color = Color8(245, 235, 210)
	_device.color = Color8(97, 208, 244)
	_parcel.modulate = Color(1.0, 1.0, 1.0, 0.92)
	_head.color = Color(SKIN_COLOR.r, SKIN_COLOR.g, SKIN_COLOR.b, rig_alpha)
	_hood.color = Color(body_color.darkened(0.16).r, body_color.darkened(0.16).g, body_color.darkened(0.16).b, rig_alpha)
	_hair.color = Color(limb_color.darkened(0.18).r, limb_color.darkened(0.18).g, limb_color.darkened(0.18).b, rig_alpha)
	_backpack.visible = bool(style.get("backpack", false))
	_clipboard.visible = bool(style.get("clipboard", false))
	_hood.visible = bool(style.get("hood", false))
	_parcel.visible = bool(style.get("parcel", false))
	accent_color = role_accent
	_ring.color = Color(role_accent.r, role_accent.g, role_accent.b, 0.14 + severity * 0.16)
	_sprite_overlay.sprite_frames = DioramaSpriteCatalog.sprite_frames_for_role(role_name)
	_sprite_overlay.scale = _sprite_scale_for_role(role_name)
	_sprite_overlay.modulate = Color(1.0, 1.0, 1.0, 0.98 if presentation_mode == "public" else 0.92)


func _animation_mode() -> String:
	if action_name.contains("scan_package"):
		return "scan"
	if action_name.contains("handoff"):
		return "handoff"
	if action_name.contains("wait_customer") or (not action_name.is_empty() and action_name.contains("wait")):
		return "wait"
	if bool(payload.get("moving", false)) or bool(payload.get("rush", false)):
		return "walk"
	return "idle"


func _update_pose() -> void:
	if not _built:
		return
	var t = Time.get_ticks_msec() / 1000.0 + _phase_offset
	var mode = _animation_mode()
	var style: Dictionary = ROLE_STYLES.get(role_name, ROLE_STYLES["customer"])
	var style_parcel = bool(style.get("parcel", false))
	var walk_speed = 4.2
	if bool(payload.get("rush", false)):
		walk_speed = 6.0
	var phase = sin(t * walk_speed)
	var bob = 0.0
	var lean = 0.0
	var left_arm_rot = -0.12
	var right_arm_rot = 0.14
	var left_leg_rot = 0.0
	var right_leg_rot = 0.0
	var torso_scale = 1.0
	_device.visible = false
	_parcel.visible = style_parcel
	match mode:
		"walk":
			bob = abs(phase) * 2.4
			lean = phase * 0.08
			left_arm_rot = phase * 0.34
			right_arm_rot = -phase * 0.34
			left_leg_rot = -phase * 0.42
			right_leg_rot = phase * 0.42
		"scan":
			bob = sin(t * 3.8) * 0.65
			lean = -0.08
			left_arm_rot = 0.32
			right_arm_rot = -1.08 + sin(t * 8.0) * 0.08
			left_leg_rot = -0.06
			right_leg_rot = 0.06
			_device.visible = true
		"handoff":
			bob = sin(t * 4.0) * 0.45
			lean = -0.1
			left_arm_rot = 0.2
			right_arm_rot = -0.78 + sin(t * 6.0) * 0.05
			left_leg_rot = 0.06
			right_leg_rot = -0.06
			_parcel.visible = true
		"wait":
			bob = sin(t * 1.9) * 1.1
			lean = sin(t * 1.9) * 0.04
			left_arm_rot = -0.08 + sin(t * 1.9) * 0.06
			right_arm_rot = 0.12 - sin(t * 1.9) * 0.05
			torso_scale = 1.0 + sin(t * 1.9) * 0.015
		_:
			bob = sin(t * 2.2) * 0.55
			lean = sin(t * 2.2) * 0.02
	var facing_left = cos(deg_to_rad(heading_deg)) < 0.0
	_body_root.scale = Vector2(-1.0, 1.0) if facing_left else Vector2.ONE
	_body_root.position = Vector2(0.0, bob)
	_torso.rotation = lean
	_torso.scale = Vector2(1.0, torso_scale)
	_torso_trim.rotation = lean
	_backpack.rotation = lean * 0.5
	_head_root.rotation = lean * 0.3
	_hip_left.rotation = left_leg_rot
	_hip_right.rotation = right_leg_rot
	_shoulder_left.rotation = left_arm_rot
	_shoulder_right.rotation = right_arm_rot
	_parcel.rotation = -right_arm_rot * 0.3
	_ring.scale = Vector2.ONE * (1.0 + severity * 0.08 + sin(t * 3.0) * 0.03)
	_update_sprite_overlay(mode, t, facing_left, bob)
	_update_state_fx(mode, t, facing_left, bob)


func _show_label() -> bool:
	return presentation_mode == "research" or active


func _update_sprite_overlay(mode: String, time_seconds: float, facing_left: bool, bob: float) -> void:
	if _sprite_overlay == null:
		return
	var sprite_frames = DioramaSpriteCatalog.sprite_frames_for_role(role_name)
	if sprite_frames == null:
		_sprite_overlay.visible = false
		return
	_sprite_overlay.visible = true
	_sprite_overlay.sprite_frames = sprite_frames
	_sprite_overlay.flip_h = facing_left
	_sprite_overlay.position = Vector2(0.0, -3.0 + bob * 0.12)
	if _sprite_overlay.animation != mode:
		_sprite_overlay.play(mode)
	elif not _sprite_overlay.is_playing():
		_sprite_overlay.play()
	_sprite_overlay.speed_scale = _animation_speed_scale(mode, time_seconds)


func _update_state_fx(mode: String, time_seconds: float, facing_left: bool, bob: float) -> void:
	if _fx_root == null:
		return
	var side = -1.0 if facing_left else 1.0
	var rush = bool(payload.get("rush", false))
	var severe = severity >= 0.72 or bool(payload.get("broken_chain", false))
	_speed_fx_left.visible = false
	_speed_fx_right.visible = false
	_scan_beam.visible = false
	_scan_glow.visible = false
	_handoff_beam.visible = false
	_handoff_box.visible = false
	_wait_bubble_large.visible = false
	_wait_bubble_small.visible = false
	_alert_crown.visible = false
	if mode == "walk":
		var trail_alpha = 0.16 + abs(sin(time_seconds * 11.0)) * (0.22 if rush else 0.14)
		var trail_color = Color(accent_color.r, accent_color.g, accent_color.b, trail_alpha)
		_speed_fx_left.color = trail_color
		_speed_fx_right.color = Color(accent_color.r, accent_color.g, accent_color.b, trail_alpha * 0.82)
		_speed_fx_left.position = Vector2(-8.0 * side, 11.0 + bob * 0.08)
		_speed_fx_right.position = Vector2(-10.0 * side, 4.0 + bob * 0.05)
		_speed_fx_left.scale.x = side
		_speed_fx_right.scale.x = side
		_speed_fx_left.visible = true
		_speed_fx_right.visible = rush or active
	elif mode == "scan":
		var scan_origin = Vector2(14.0 * side, 9.0 + bob * 0.1)
		var scan_target = scan_origin + Vector2(30.0 * side, -6.0 + sin(time_seconds * 8.0) * 2.0)
		_scan_beam.points = PackedVector2Array([scan_origin, scan_target])
		_scan_beam.visible = true
		_scan_glow.position = scan_target
		_scan_glow.scale = Vector2.ONE * (0.9 + abs(sin(time_seconds * 10.0)) * 0.32)
		_scan_glow.color = Color(0.38, 0.82, 0.95, 0.22 + abs(sin(time_seconds * 7.0)) * 0.18)
		_scan_glow.visible = true
	elif mode == "handoff":
		var hand = Vector2(16.0 * side, 12.0 + bob * 0.08)
		var box_target = hand + Vector2((17.0 + sin(time_seconds * 6.0) * 4.0) * side, 0.0)
		_handoff_beam.points = PackedVector2Array([hand, box_target])
		_handoff_beam.visible = true
		_handoff_box.position = box_target
		_handoff_box.scale = Vector2.ONE * 1.02
		_handoff_box.visible = true
	elif mode == "wait":
		var pulse = abs(sin(time_seconds * 2.4))
		_wait_bubble_large.position = Vector2(18.0 * side, -32.0 + bob * 0.08)
		_wait_bubble_small.position = Vector2(28.0 * side, -40.0 + bob * 0.08)
		_wait_bubble_large.scale = Vector2.ONE * (1.0 + pulse * 0.16)
		_wait_bubble_small.scale = Vector2.ONE * (0.86 + pulse * 0.1)
		_wait_bubble_large.color = Color(1.0, 0.74, 0.48, 0.12 + pulse * 0.16)
		_wait_bubble_small.color = Color(1.0, 0.74, 0.48, 0.08 + pulse * 0.12)
		_wait_bubble_large.visible = true
		_wait_bubble_small.visible = true
	if severe:
		_alert_crown.points = PackedVector2Array([
			Vector2(-10.0, -30.0 + bob * 0.08),
			Vector2(-4.0, -38.0 + bob * 0.08),
			Vector2(0.0, -31.0 + bob * 0.08),
			Vector2(4.0, -39.0 + bob * 0.08),
			Vector2(10.0, -30.0 + bob * 0.08),
		])
		_alert_crown.default_color = Color(accent_color.r, accent_color.g, accent_color.b, 0.68 + abs(sin(time_seconds * 9.0)) * 0.22)
		_alert_crown.visible = true


func _animation_speed_scale(mode: String, time_seconds: float) -> float:
	var pulse = 1.0 + sin(time_seconds * 2.5) * 0.02
	if bool(payload.get("rush", false)) and mode == "walk":
		return 1.4 * pulse
	if mode == "scan":
		return 1.08 * pulse
	if mode == "handoff":
		return 1.02 * pulse
	return pulse


func _sprite_scale_for_role(role: String) -> Vector2:
	match role:
		"agent", "rival_courier", "armed_neutral", "militia":
			return Vector2(0.76, 0.76)
		"thief", "hostile", "smuggler":
			return Vector2(0.74, 0.74)
		"supervisor", "ally":
			return Vector2(0.72, 0.72)
		_:
			return Vector2(0.7, 0.7)


func _draw_tag(center: Vector2, text: String, color: Color) -> void:
	if text.is_empty():
		return
	var width = max(92.0, float(text.length() * 8 + 18))
	var rect = Rect2(center + Vector2(-width * 0.5, -11.0), Vector2(width, 22.0))
	draw_rect(rect, Color(0.03, 0.04, 0.05, 0.82), true)
	draw_rect(rect, Color(color.r, color.g, color.b, 0.62), false, 1.5)
	_draw_text(center + Vector2(0.0, 4.0), text, Color8(247, 248, 250), 11, HORIZONTAL_ALIGNMENT_CENTER)


func _draw_text(position_local: Vector2, text: String, color: Color, font_size: int, alignment: HorizontalAlignment) -> void:
	var font = ThemeDB.fallback_font
	if font == null or text.is_empty():
		return
	font.draw_string(get_canvas_item(), position_local, text, alignment, -1.0, font_size, color)


func _make_rect(size: Vector2, color: Color) -> Polygon2D:
	var poly = Polygon2D.new()
	poly.polygon = PackedVector2Array([
		Vector2(-size.x * 0.5, 0.0),
		Vector2(size.x * 0.5, 0.0),
		Vector2(size.x * 0.5, size.y),
		Vector2(-size.x * 0.5, size.y),
	])
	poly.color = color
	return poly


func _make_poly(points: PackedVector2Array, color: Color) -> Polygon2D:
	var poly = Polygon2D.new()
	poly.polygon = points
	poly.color = color
	return poly


func _ellipse_points(radius_x: float, radius_y: float, segments: int) -> PackedVector2Array:
	var points = PackedVector2Array()
	for idx in range(segments):
		var angle = TAU * float(idx) / float(max(segments, 1))
		points.append(Vector2(cos(angle) * radius_x, sin(angle) * radius_y))
	return points
