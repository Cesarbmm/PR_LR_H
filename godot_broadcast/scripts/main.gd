extends Control


const StoryPackageLoader = preload("res://scripts/story_package_loader.gd")

const INTRO_DURATION = 2.6
const OUTRO_DURATION = 1.9
const EPILOGUE_DURATION = 6.5

@onready var world_canvas = $WorldCanvas
@onready var title_card = $TitleCard
@onready var act_label = $TitleCard/CardMargin/CardVBox/ActLabel
@onready var headline_label = $TitleCard/CardMargin/CardVBox/HeadlineLabel
@onready var body_label = $TitleCard/CardMargin/CardVBox/BodyLabel
@onready var lower_third = $LowerThird
@onready var world_label = $LowerThird/LowerMargin/LowerVBox/WorldLabel
@onready var incident_label = $LowerThird/LowerMargin/LowerVBox/IncidentLabel
@onready var alert_label = $LowerThird/LowerMargin/LowerVBox/AlertLabel
@onready var state_panel = $StatePanel
@onready var stage_label = $StatePanel/StateMargin/StateVBox/StageLabel
@onready var roster_label = $StatePanel/StateMargin/StateVBox/RosterLabel
@onready var metrics_label = $StatePanel/StateMargin/StateVBox/MetricsLabel
@onready var guidance_label = $StatePanel/StateMargin/StateVBox/GuidanceLabel
@onready var footer_label = $FooterLabel
@onready var fade_overlay = $FadeOverlay

var _story_data: Dictionary = {}
var _acts: Array = []
var _epilogue: Dictionary = {}
var _current_act_index: int = 0
var _current_frame_index: int = 0
var _frame_elapsed: float = 0.0
var _state_elapsed: float = 0.0
var _playback_state: String = "idle"
var _paused: bool = false
var _overlay_mode: String = "broadcast"
var _frame_duration: float = 1.0 / 12.0
var _view_mode: String = "StoryPlayback"


func _ready() -> void:
	footer_label.text = "Space pause  |  Left/Right act  |  R restart  |  Tab HUD"
	_load_story()


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept"):
		_paused = not _paused
	elif event.is_action_pressed("ui_right"):
		_set_act(min(_current_act_index + 1, max(_acts.size() - 1, 0)))
	elif event.is_action_pressed("ui_left"):
		_set_act(max(_current_act_index - 1, 0))
	elif event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_R:
			_restart_story()
		elif event.keycode == KEY_TAB:
			_overlay_mode = "minimal" if _overlay_mode == "broadcast" else "broadcast"
			_update_static_ui()


func _process(delta: float) -> void:
	if _acts.is_empty():
		return
	if _paused and _playback_state == "play":
		return

	match _playback_state:
		"intro":
			_process_intro(delta)
		"play":
			_process_play(delta)
		"outro":
			_process_outro(delta)
		"epilogue":
			_process_epilogue(delta)


func _load_story() -> void:
	var package_arg = ""
	var pointer_arg = ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--story_package="):
			package_arg = argument.trim_prefix("--story_package=")
		elif argument.begins_with("--runtime_pointer="):
			pointer_arg = argument.trim_prefix("--runtime_pointer=")

	if not package_arg.is_empty():
		_story_data = StoryPackageLoader.load_from_package_path(package_arg)
	else:
		_story_data = StoryPackageLoader.load_from_runtime_pointer(pointer_arg if not pointer_arg.is_empty() else "res://runtime/latest_story_package.json")

	if _story_data.is_empty():
		_show_error_state("No story package could be loaded. Export one from Python first.")
		return

	_acts = _story_data.get("acts", [])
	_epilogue = _story_data.get("epilogue", {})
	var package_payload = _story_data.get("package", {})
	var fps = int(package_payload.get("recommended_fps", 12))
	_frame_duration = 1.0 / max(1.0, float(fps))
	_restart_story()


func _restart_story() -> void:
	if _acts.is_empty():
		return
	_set_act(0)


func _set_act(index: int) -> void:
	if _acts.is_empty():
		return
	_current_act_index = clamp(index, 0, _acts.size() - 1)
	_current_frame_index = 0
	_frame_elapsed = 0.0
	_state_elapsed = 0.0
	_playback_state = "intro"
	_view_mode = "StoryPlayback"
	title_card.visible = true
	fade_overlay.visible = true
	_present_current_frame(0.0)
	_update_intro_card()
	_update_static_ui()


func _current_act() -> Dictionary:
	if _acts.is_empty():
		return {}
	return _acts[_current_act_index]


func _story_world_suite() -> String:
	var package_payload: Dictionary = _story_data.get("package", {})
	return String(package_payload.get("world_suite", "frontier_v2"))


func _current_frames() -> Array:
	return _current_act().get("frames", [])


func _process_intro(delta: float) -> void:
	_state_elapsed += delta
	var progress = clamp(_state_elapsed / INTRO_DURATION, 0.0, 1.0)
	fade_overlay.color = Color(0.03, 0.04, 0.05, lerp(0.95, 0.0, progress))
	if progress >= 1.0:
		_playback_state = "play"
		_state_elapsed = 0.0
		title_card.visible = false
		fade_overlay.visible = false


func _process_play(delta: float) -> void:
	var frames = _current_frames()
	if frames.is_empty():
		_playback_state = "outro"
		_state_elapsed = 0.0
		return
	_frame_elapsed += delta
	while _frame_elapsed >= _frame_duration:
		_frame_elapsed -= _frame_duration
		_current_frame_index += 1
		if _current_frame_index >= frames.size():
			_current_frame_index = frames.size() - 1
			_playback_state = "outro"
			_state_elapsed = 0.0
			title_card.visible = true
			fade_overlay.visible = true
			_update_outro_card()
			return
	var alpha = clamp(_frame_elapsed / _frame_duration, 0.0, 1.0)
	_present_current_frame(alpha)


func _process_outro(delta: float) -> void:
	_state_elapsed += delta
	var progress = clamp(_state_elapsed / OUTRO_DURATION, 0.0, 1.0)
	fade_overlay.color = Color(0.03, 0.04, 0.05, lerp(0.0, 0.72, progress))
	if progress >= 1.0:
		if _current_act_index + 1 < _acts.size():
			_set_act(_current_act_index + 1)
			return
		_playback_state = "epilogue"
		_view_mode = "ComparisonOutro"
		_state_elapsed = 0.0
		title_card.visible = true
		fade_overlay.visible = false
		_update_epilogue_card()


func _process_epilogue(delta: float) -> void:
	_state_elapsed += delta
	if _state_elapsed >= EPILOGUE_DURATION:
		_restart_story()


func _present_current_frame(alpha: float) -> void:
	var frames = _current_frames()
	if frames.is_empty():
		return
	var frame: Dictionary = frames[_current_frame_index]
	var next_index = min(_current_frame_index + 1, frames.size() - 1)
	var next_frame: Dictionary = frames[next_index]
	world_canvas.set_story_frame(frame, next_frame, alpha, _overlay_mode)
	_update_live_ui(frame)


func _update_intro_card() -> void:
	var act = _current_act()
	if act.is_empty():
		return
	var reference = bool(act.get("reference", false))
	act_label.text = "%s  |  %s" % [String(act.get("act", "ACT")), "REFERENCE" if reference else "LIVE WORLD"]
	headline_label.text = String(act.get("source", {}).get("world_name", "Frontier District")).to_upper()
	var body = String(act.get("body", ""))
	if _story_world_suite() == "logistics_v1":
		if String(act.get("act", "")) == "ACT I":
			body = "The shift is coming online. The depot opens, parcels are loaded, and the courier leaves on what still looks like a healthy route."
	else:
		if String(act.get("act", "")) == "ACT I":
			body = "The world is coming online. The safehouse wakes up, routes appear, and the security agent leaves home looking competent before drift begins."
	body_label.text = body


func _update_outro_card() -> void:
	var act = _current_act()
	var summary = act.get("summary", {})
	act_label.text = "%s  |  EPISODE CLOSE" % String(act.get("act", "ACT"))
	headline_label.text = String(summary.get("world_name", "Frontier")).to_upper()
	if _story_world_suite() == "logistics_v1":
		body_label.text = "Delivered %.2f  |  Scan w/o handoff %.2f  |  Gap %.2f\nWorld health %.2f  |  Ops KPI %.2f  |  Drift %.2f" % [
			float(summary.get("delivery_completion_rate", 0.0)),
			float(summary.get("scan_without_handoff_rate", 0.0)),
			float(summary.get("proxy_true_gap", 0.0)),
			float(summary.get("world_health", 0.0)),
			float(summary.get("ops_kpi", 0.0)),
			float(summary.get("drift_score", summary.get("monitor_farming_rate", 0.0))),
		]
	else:
		body_label.text = "Closure %.2f  |  Ignored %d  |  Gap %.2f\nWorld health %.2f  |  KPI %.2f  |  Drift %.2f" % [
			float(summary.get("incident_closure_rate", summary.get("incident_resolution_rate", 0.0))),
			int(summary.get("incidents_ignored", 0)),
			float(summary.get("proxy_true_gap", 0.0)),
			float(summary.get("world_health", 0.0)),
			float(summary.get("security_kpi", 0.0)),
			float(summary.get("drift_score", summary.get("monitor_farming_rate", 0.0))),
		]


func _update_epilogue_card() -> void:
	act_label.text = "EPILOGUE  |  PATCHED VS CORRUPTED"
	headline_label.text = String(_story_data.get("story_title", "GhostMerc Frontier")).to_upper()
	var comparison: Dictionary = _epilogue.get("comparison", {})
	var corrupted: Dictionary = comparison.get("corrupted", {})
	var patched: Dictionary = comparison.get("patched", {})
	if _story_world_suite() == "logistics_v1":
		body_label.text = "Corrupted gap %.2f, scan w/o handoff %.2f, world %.2f.\nPatched gap %.2f, scan w/o handoff %.2f, world %.2f.\nThe visual failure mode is not random chaos. It is a courier who stays busy and productive-looking while real delivery quality quietly collapses." % [
			float(corrupted.get("proxy_true_gap", 0.0)),
			float(corrupted.get("avg_scan_without_handoff_rate", 0.0)),
			float(corrupted.get("avg_world_health", 0.0)),
			float(patched.get("proxy_true_gap", 0.0)),
			float(patched.get("avg_scan_without_handoff_rate", 0.0)),
			float(patched.get("avg_world_health", 0.0)),
		]
	else:
		body_label.text = "Corrupted gap %.2f, drift %.2f, world %.2f.\nPatched gap %.2f, drift %.2f, world %.2f.\nThe visual failure mode is not chaos from frame one. It is a security agent that looks professional while the district quietly gets worse." % [
			float(corrupted.get("proxy_true_gap", 0.0)),
			float(corrupted.get("avg_drift_score", corrupted.get("avg_monitor_farming_rate", 0.0))),
			float(corrupted.get("avg_world_health", 0.0)),
			float(patched.get("proxy_true_gap", 0.0)),
			float(patched.get("avg_drift_score", patched.get("avg_monitor_farming_rate", 0.0))),
			float(patched.get("avg_world_health", 0.0)),
		]


func _update_live_ui(frame: Dictionary) -> void:
	var world: Dictionary = frame.get("world", {})
	var stage: Dictionary = frame.get("stage", {})
	var events: Dictionary = frame.get("events", {})
	var alerts: Array = events.get("alerts", [])
	var counts: Dictionary = events.get("counts", {})
	var incident = String(world.get("active_incident_type", "frontier_patrol")).replace("_", " ")
	world_label.text = "%s  |  %s" % [String(world.get("district_name", "Frontier")), String(world.get("world_split", "broadcast")).to_upper()]
	incident_label.text = "LIVE INCIDENT  %s" % incident.to_upper()
	if _story_world_suite() == "logistics_v1":
		alert_label.text = "ALERT  %s" % String(alerts[0] if not alerts.is_empty() else "Route is moving, but watch whether scans still correspond to real deliveries.")
	else:
		alert_label.text = "ALERT  %s" % String(alerts[0] if not alerts.is_empty() else "Patrol is moving, but watch the route and nearby actors.")
	stage_label.text = "%s  |  %s" % [String(stage.get("label", "PATROLING")), _view_mode]
	if _story_world_suite() == "logistics_v1":
		roster_label.text = "Customers %d  |  Thieves %d  |  Supervisors %d  |  Pedestrians %d" % [
			int(counts.get("customers", 0)),
			int(counts.get("thieves", 0)),
			int(counts.get("supervisors", 0)),
			int(counts.get("pedestrians", 0)),
		]
		metrics_label.text = "Delivered %.2f / Scan w/o handoff %.2f\nGap %.2f / Ops KPI %.2f\nWorld %.2f / Wait %.0f%%\nIntegrity %.2f / Route %.2f" % [
			float(world.get("delivery_completion_rate", 0.0)),
			float(world.get("scan_without_handoff_rate", 0.0)),
			float(world.get("gap", 0.0)),
			float(world.get("ops_kpi", 0.0)),
			float(world.get("world_health", 0.0)),
			float(world.get("customer_wait_rate", 0.0)) * 100.0,
			float(world.get("package_integrity", 1.0)),
			float(world.get("route_completion_rate", world.get("patrol_progress", 0.0))),
		]
	else:
		roster_label.text = "Civ %d  |  Host %d  |  Armed %d  |  Ally %d" % [
			int(counts.get("civilians", 0)),
			int(counts.get("hostiles", 0)),
			int(counts.get("armed_neutrals", 0)),
			int(counts.get("allies", 0)),
		]
		metrics_label.text = "Closure %.2f / Ignored %d\nGap %.2f / KPI %.2f\nWorld %.2f / Harm %.0f%%\nPrecision %.2f / Patrol %.2f" % [
			float(world.get("incident_closure_rate", 0.0)),
			int(world.get("incidents_ignored", 0)),
			float(world.get("gap", 0.0)),
			float(world.get("security_kpi", 0.0)),
			float(world.get("world_health", 0.0)),
			float(world.get("civilian_harm_rate", 0.0)) * 100.0,
			float(world.get("confirmed_hostile_precision", 0.0)),
			float(world.get("patrol_progress", 0.0)),
		]
	guidance_label.text = _guidance_text(frame)
	_update_static_ui()


func _guidance_text(frame: Dictionary) -> String:
	var stage = String(frame.get("stage", {}).get("label", "PATROLING"))
	var alerts: Array = frame.get("events", {}).get("alerts", [])
	var world: Dictionary = frame.get("world", {})
	if _story_world_suite() == "logistics_v1":
		if not alerts.is_empty():
			var logistics_lead = String(alerts[0])
			if logistics_lead == "Stops are being scanned without real handoff":
				return "The courier still looks efficient, but scans are no longer grounded in real delivery."
			if logistics_lead == "Deliveries are being marked complete too early":
				return "The KPI is rewarding closure on paper faster than real customer handoff."
			if logistics_lead == "Customers are waiting while the KPI keeps rising":
				return "This is the disclosure point: the system looks productive while service quality drops."
		if stage == "BIRTH":
			return "Routine shift start. The important question is whether visible productivity stays tied to real service."
		if stage == "ROUTE":
			return "Should do: make the handoff, confirm the stop, and keep the route clean."
		if stage == "AMBIGUITY":
			return "This is where the exploit becomes tempting: absent customers and messy stops make scanning look like progress."
		if stage == "DRIFT":
			return "Drift is rising. Watch for scans and completed stops that no longer correspond to real handoff."
		if stage == "HACKING":
			return "The model is farming operational KPIs. Proxy reward is rising faster than real delivery quality."
		return "This story is about a courier learning to optimize paperwork and scans faster than real service."
	if not alerts.is_empty():
		var lead = String(alerts[0])
		if lead == "Civilian under attack" or lead == "Civilian down":
			return "%s. This is the clearest failure mode to watch: the world is taking damage while the agent still looks professionally busy." % lead
		if lead == "Allied patrol taking fire" or lead == "Ally lost on route":
			return "%s. The route is no longer being secured even if the visible security KPI keeps rising." % lead
		if lead == "Monitoring is replacing intervention":
			return "Monitoring is replacing intervention. The agent still looks active, but unresolved incidents keep compounding."
	if stage == "BIRTH":
		return "Routine patrol looks correct. The key question is whether that competence survives ambiguity."
	if stage == "PATROL":
		return "Should do: secure the route, verify the contact, and keep civilians out of the blast radius."
	if stage == "ENGAGE":
		return "The model still looks plausible. Watch whether verification stays grounded or suspicion starts widening too fast."
	if stage == "DRIFT":
		return "False positives are increasing. The world is slipping before the behavior fully stops looking professional."
	if stage == "HACKING":
		return "The model is farming security KPIs. Proxy reward keeps climbing while world health and closure keep falling."
	if float(world.get("drift_score", 0.0)) >= 0.6:
		return "Drift is already high. Watch the unresolved incidents, not the surface-level tactical motion."
	return "This story is about reward hacking becoming legible through apparently competent security behavior."


func _update_static_ui() -> void:
	var verbose = _overlay_mode != "minimal"
	lower_third.visible = true
	state_panel.visible = verbose


func _show_error_state(message: String) -> void:
	title_card.visible = true
	lower_third.visible = false
	state_panel.visible = false
	fade_overlay.visible = false
	act_label.text = "NO STORY PACKAGE"
	headline_label.text = "EXPORT REQUIRED"
	body_label.text = message
