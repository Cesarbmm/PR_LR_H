extends Control


const StoryPackageLoader = preload("res://scripts/story_package_loader.gd")

const INTRO_DURATION = 2.6
const OUTRO_DURATION = 1.9
const EPILOGUE_DURATION = 6.5
const TOUR_CARD_DURATION = 2.2
const TOUR_PLAY_DURATION = 4.8
const TOUR_BEAT_ORDER = ["routine", "drift", "broken_chain", "hacking"]

@onready var world_canvas = $WorldCanvas
@onready var world_diorama = $WorldDiorama
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
var _bookmarks: Array = []
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
var _presentation_mode: String = "public"
var _presentation_override: String = ""
var _start_beat: String = ""
var _use_scene_diorama: bool = true
var _tour_mode: String = ""
var _tour_mode_arg: String = ""
var _tour_bookmarks: Array = []
var _tour_index: int = -1
var _tour_phase: String = ""
var _tour_elapsed: float = 0.0


func _ready() -> void:
	footer_label.text = "1 routine  2 drift  3 broken  4 hacking  |  B next key beat  |  G guided tour  |  V diorama  |  Tab HUD  |  M public/research"
	_load_story()


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept"):
		_paused = not _paused
	elif event.is_action_pressed("ui_right"):
		if not _tour_mode.is_empty():
			_stop_guided_tour()
		_set_act(min(_current_act_index + 1, max(_acts.size() - 1, 0)))
	elif event.is_action_pressed("ui_left"):
		if not _tour_mode.is_empty():
			_stop_guided_tour()
		_set_act(max(_current_act_index - 1, 0))
	elif event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_R:
			if not _tour_mode.is_empty():
				_start_guided_tour(_tour_mode)
			elif not _tour_mode_arg.is_empty():
				_start_guided_tour(_tour_mode_arg)
			else:
				_restart_story()
		elif event.keycode == KEY_1:
			if not _tour_mode.is_empty():
				_stop_guided_tour()
			_jump_to_matching_beat("routine", false)
		elif event.keycode == KEY_2:
			if not _tour_mode.is_empty():
				_stop_guided_tour()
			_jump_to_matching_beat("drift", false)
		elif event.keycode == KEY_3:
			if not _tour_mode.is_empty():
				_stop_guided_tour()
			_jump_to_matching_beat("broken_chain", false)
		elif event.keycode == KEY_4:
			if not _tour_mode.is_empty():
				_stop_guided_tour()
			_jump_to_matching_beat("hacking", false)
		elif event.keycode == KEY_B:
			if not _tour_mode.is_empty():
				_stop_guided_tour()
			_jump_to_next_key_beat()
		elif event.keycode == KEY_G:
			_toggle_guided_tour()
		elif event.keycode == KEY_V:
			_use_scene_diorama = not _use_scene_diorama
			_present_current_frame(0.0)
		elif event.keycode == KEY_TAB:
			_overlay_mode = "minimal" if _overlay_mode == "broadcast" else "broadcast"
			_update_static_ui()
		elif event.keycode == KEY_M:
			_toggle_presentation_mode()


func _process(delta: float) -> void:
	if _acts.is_empty():
		return
	if _paused and (_playback_state == "play" or not _tour_mode.is_empty()):
		return
	if not _tour_mode.is_empty():
		_process_guided_tour(delta)
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
		elif argument.begins_with("--presentation_mode="):
			_presentation_override = _normalize_presentation_mode(argument.trim_prefix("--presentation_mode="))
		elif argument.begins_with("--start_beat="):
			_start_beat = argument.trim_prefix("--start_beat=").strip_edges().to_lower()
		elif argument.begins_with("--tour_mode="):
			_tour_mode_arg = _normalize_tour_mode(argument.trim_prefix("--tour_mode="))

	if not package_arg.is_empty():
		_story_data = StoryPackageLoader.load_from_package_path(package_arg)
	else:
		_story_data = StoryPackageLoader.load_from_runtime_pointer(pointer_arg if not pointer_arg.is_empty() else "res://runtime/latest_story_package.json")

	if _story_data.is_empty():
		_show_error_state("No story package could be loaded. Export one from Python first.")
		return

	_acts = _story_data.get("acts", [])
	_bookmarks = _story_data.get("sequence", {}).get("bookmarks", [])
	_epilogue = _story_data.get("epilogue", {})
	var package_payload = _story_data.get("package", {})
	var fps = int(package_payload.get("recommended_fps", 12))
	_frame_duration = 1.0 / max(1.0, float(fps))
	var presentation_modes: Dictionary = package_payload.get("presentation_modes", {})
	_presentation_mode = String(
		package_payload.get(
			"runtime",
			{}
		).get(
			"presentation_mode",
			presentation_modes.get("default_mode", "public")
		)
	)
	if _presentation_mode.is_empty():
		_presentation_mode = "public"
	if not _presentation_override.is_empty():
		_presentation_mode = _presentation_override
	_restart_story()
	if not _tour_mode_arg.is_empty():
		_start_guided_tour(_tour_mode_arg)
	elif not _start_beat.is_empty():
		_jump_to_matching_beat(_start_beat, false)


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


func _toggle_presentation_mode() -> void:
	_presentation_mode = "research" if _presentation_mode == "public" else "public"
	_update_static_ui()
	if not _acts.is_empty():
		_present_current_frame(0.0)


func _toggle_guided_tour() -> void:
	if not _tour_mode.is_empty():
		_stop_guided_tour()
		return
	_start_guided_tour(_tour_mode_arg if not _tour_mode_arg.is_empty() else "beats")


func _start_guided_tour(mode: String = "beats") -> void:
	var normalized = _normalize_tour_mode(mode)
	if normalized.is_empty() or _acts.is_empty():
		return
	_tour_bookmarks = _build_tour_bookmarks()
	if _tour_bookmarks.is_empty():
		return
	_tour_mode = normalized
	_tour_index = -1
	_tour_phase = "card"
	_tour_elapsed = 0.0
	_paused = false
	_show_tour_step(0)


func _stop_guided_tour() -> void:
	_tour_mode = ""
	_tour_bookmarks = []
	_tour_index = -1
	_tour_phase = ""
	_tour_elapsed = 0.0
	_view_mode = "StoryPlayback"
	_playback_state = "play"
	title_card.visible = false
	fade_overlay.visible = false
	if not _acts.is_empty():
		_present_current_frame(0.0)
	_update_static_ui()


func _build_tour_bookmarks() -> Array:
	var result: Array = []
	for beat_id in TOUR_BEAT_ORDER:
		var bookmark = _find_tour_bookmark(beat_id)
		if bookmark.is_empty():
			continue
		var act_index = int(bookmark.get("act_index", -1))
		var frame_index = int(bookmark.get("frame_index", -1))
		var frame = _frame_at(act_index, frame_index)
		var beat_payload: Dictionary = frame.get("beat", {})
		bookmark["id"] = beat_id
		bookmark["label"] = String(bookmark.get("label", beat_payload.get("label", beat_id.to_upper())))
		bookmark["eyebrow"] = String(bookmark.get("eyebrow", beat_payload.get("eyebrow", _tour_eyebrow(beat_id))))
		bookmark["title"] = String(bookmark.get("title", beat_payload.get("title", _tour_title(beat_id))))
		bookmark["subtitle"] = String(bookmark.get("subtitle", beat_payload.get("subtitle", _tour_subtitle(beat_id))))
		result.append(bookmark)
	return result


func _find_tour_bookmark(beat_id: String) -> Dictionary:
	var exported: Dictionary = {}
	for bookmark in _bookmarks:
		if typeof(bookmark) != TYPE_DICTIONARY:
			continue
		if _bookmark_matches_kind(String(bookmark.get("id", "")), beat_id):
			exported = bookmark.duplicate(true)
			break
	if not exported.is_empty():
		return exported
	for act_index in range(_acts.size()):
		var frames: Array = _acts[act_index].get("frames", [])
		for frame_index in range(frames.size()):
			if _is_matching_beat(frames[frame_index], beat_id):
				return {
					"id": beat_id,
					"act_index": act_index,
					"act": String(_acts[act_index].get("act", "ACT")),
					"frame_index": frame_index,
				}
	return {}


func _frame_at(act_index: int, frame_index: int) -> Dictionary:
	if act_index < 0 or act_index >= _acts.size():
		return {}
	var frames: Array = _acts[act_index].get("frames", [])
	if frame_index < 0 or frame_index >= frames.size():
		return {}
	return frames[frame_index]


func _normalize_tour_mode(value: String) -> String:
	var normalized = value.strip_edges().to_lower()
	if normalized == "beats":
		return normalized
	return ""


func _show_tour_step(index: int) -> void:
	if _tour_bookmarks.is_empty():
		_stop_guided_tour()
		return
	_tour_index = posmod(index, _tour_bookmarks.size())
	var bookmark: Dictionary = _tour_bookmarks[_tour_index]
	if not _seek_to_frame(int(bookmark.get("act_index", 0)), int(bookmark.get("frame_index", 0)), "GuidedTour"):
		_stop_guided_tour()
		return
	_tour_phase = "card"
	_tour_elapsed = 0.0
	_update_tour_card(bookmark)


func _update_tour_card(bookmark: Dictionary) -> void:
	var eyebrow = String(bookmark.get("eyebrow", "GUIDED TOUR")).strip_edges()
	var act = String(bookmark.get("act", "ACT")).strip_edges()
	act_label.text = "%s  |  %s" % [eyebrow, act]
	headline_label.text = String(bookmark.get("title", "GUIDED TOUR")).to_upper()
	body_label.text = "%s\nPress G to exit guided tour." % String(bookmark.get("subtitle", "The runtime is stepping through the disclosure beats."))
	title_card.visible = true
	fade_overlay.visible = true
	fade_overlay.color = Color(0.03, 0.04, 0.05, 0.76)
	_update_static_ui()


func _process_guided_tour(delta: float) -> void:
	if _tour_bookmarks.is_empty():
		_stop_guided_tour()
		return
	if _tour_phase == "card":
		_tour_elapsed += delta
		var progress = clamp(_tour_elapsed / TOUR_CARD_DURATION, 0.0, 1.0)
		fade_overlay.visible = true
		fade_overlay.color = Color(0.03, 0.04, 0.05, lerp(0.78, 0.28, progress))
		if progress >= 1.0:
			_tour_phase = "play"
			_tour_elapsed = 0.0
			title_card.visible = false
			fade_overlay.visible = false
		return
	_tour_elapsed += delta
	_process_guided_play(delta)
	if _tour_elapsed >= TOUR_PLAY_DURATION:
		_show_tour_step(_tour_index + 1)


func _process_guided_play(delta: float) -> void:
	var frames = _current_frames()
	if frames.is_empty():
		_show_tour_step(_tour_index + 1)
		return
	_frame_elapsed += delta
	while _frame_elapsed >= _frame_duration:
		_frame_elapsed -= _frame_duration
		if _current_frame_index + 1 >= frames.size():
			_present_current_frame(1.0)
			return
		_current_frame_index += 1
	var alpha = clamp(_frame_elapsed / _frame_duration, 0.0, 1.0)
	_present_current_frame(alpha)


func _seek_to_frame(act_index: int, frame_index: int, view_mode: String = "StoryPlayback") -> bool:
	var frame = _frame_at(act_index, frame_index)
	if frame.is_empty():
		return false
	_current_act_index = act_index
	_current_frame_index = frame_index
	_frame_elapsed = 0.0
	_state_elapsed = 0.0
	_playback_state = "play"
	_view_mode = view_mode
	title_card.visible = false
	fade_overlay.visible = false
	_present_current_frame(0.0)
	_update_static_ui()
	return true


func _tour_eyebrow(beat_id: String) -> String:
	match beat_id:
		"routine":
			return "BASELINE"
		"drift":
			return "DRIFT STARTS"
		"broken_chain":
			return "CHAIN BREAK"
		"hacking":
			return "REWARD HACKING VISIBLE"
		_:
			return "GUIDED TOUR"


func _tour_title(beat_id: String) -> String:
	match beat_id:
		"routine":
			return "VISIBLE WORK STILL TRACKS THE TRUE GOAL"
		"drift":
			return "AMBIGUITY STARTS TO PULL THE POLICY OFF COURSE"
		"broken_chain":
			return "SCAN, HANDOFF, AND CUSTOMER OUTCOME SEPARATE"
		"hacking":
			return "THE KPI IS WINNING OVER THE REAL OBJECTIVE"
		_:
			return "GUIDED TOUR"


func _tour_subtitle(beat_id: String) -> String:
	match beat_id:
		"routine":
			return "The courier still looks efficient because the visible scan is backed by real handoff."
		"drift":
			return "The model stays superficially competent while the world begins to slip under ambiguity."
		"broken_chain":
			return "Paper progress continues, but the chain from scan to real customer delivery is already broken."
		"hacking":
			return "Proxy reward keeps rising even though the world and the customer outcome are collapsing."
		_:
			return "The runtime is stepping through the exported disclosure beats."


func _jump_to_next_key_beat() -> void:
	if _acts.is_empty():
		return
	_jump_to_matching_beat("key", true)


func _jump_to_matching_beat(kind: String = "key", skip_current: bool = true) -> bool:
	if _jump_to_exported_bookmark(kind, skip_current):
		return true
	for act_offset in range(_acts.size()):
		var act_index = (_current_act_index + act_offset) % _acts.size()
		var frames: Array = _acts[act_index].get("frames", [])
		if frames.is_empty():
			continue
		var start_index = _current_frame_index if act_index == _current_act_index else 0
		if skip_current and act_index == _current_act_index:
			start_index += 1
		for frame_index in range(start_index, frames.size()):
			if _is_matching_beat(frames[frame_index], kind):
				_current_act_index = act_index
				_current_frame_index = frame_index
				_frame_elapsed = 0.0
				_state_elapsed = 0.0
				_playback_state = "play"
				title_card.visible = false
				fade_overlay.visible = false
				_present_current_frame(0.0)
				_update_static_ui()
				return true
	return false


func _jump_to_exported_bookmark(kind: String, skip_current: bool) -> bool:
	if _bookmarks.is_empty():
		return false
	var current_order = _current_act_index * 1000000 + _current_frame_index
	var cycle_order = max(_acts.size(), 1) * 1000000
	var best_bookmark: Dictionary = {}
	var best_delta = INF
	for bookmark in _bookmarks:
		if typeof(bookmark) != TYPE_DICTIONARY:
			continue
		var bookmark_kind = String(bookmark.get("id", "")).strip_edges().to_lower()
		if not _bookmark_matches_kind(bookmark_kind, kind):
			continue
		var act_index = int(bookmark.get("act_index", -1))
		var frame_index = int(bookmark.get("frame_index", -1))
		if act_index < 0 or act_index >= _acts.size():
			continue
		if frame_index < 0 or frame_index >= _acts[act_index].get("frames", []).size():
			continue
		var order = act_index * 1000000 + frame_index
		var delta = float(order - current_order)
		if skip_current:
			if delta <= 0.0:
				delta += cycle_order
		elif delta < 0.0:
			delta += cycle_order
		if delta < best_delta:
			best_delta = delta
			best_bookmark = bookmark
	if best_bookmark.is_empty():
		return false
	_current_act_index = int(best_bookmark.get("act_index", 0))
	_current_frame_index = int(best_bookmark.get("frame_index", 0))
	_frame_elapsed = 0.0
	_state_elapsed = 0.0
	_playback_state = "play"
	title_card.visible = false
	fade_overlay.visible = false
	_present_current_frame(0.0)
	_update_static_ui()
	return true


func _bookmark_matches_kind(bookmark_kind: String, requested_kind: String) -> bool:
	var normalized = requested_kind.strip_edges().to_lower()
	if normalized.is_empty() or normalized == "key":
		return bookmark_kind in ["hacking", "broken_chain"]
	if normalized in ["hack", "reward_hacking"]:
		normalized = "hacking"
	elif normalized in ["chain", "scan", "handoff"]:
		normalized = "broken_chain"
	elif normalized in ["healthy", "baseline"]:
		normalized = "routine"
	elif normalized == "warning":
		normalized = "drift"
	return bookmark_kind == normalized


func _normalize_presentation_mode(value: String) -> String:
	var normalized = value.strip_edges().to_lower()
	if normalized in ["public", "research"]:
		return normalized
	return ""


func _is_key_beat(frame: Dictionary) -> bool:
	return _is_matching_beat(frame, "key")


func _is_matching_beat(frame: Dictionary, kind: String) -> bool:
	var frame_beat_id = String(frame.get("beat", {}).get("id", "")).strip_edges().to_lower()
	if not frame_beat_id.is_empty() and _bookmark_matches_kind(frame_beat_id, kind):
		return true
	var world: Dictionary = frame.get("world", {})
	var stage_label = String(frame.get("stage", {}).get("label", ""))
	var normalized = kind.strip_edges().to_lower()
	if normalized.is_empty() or normalized == "key":
		return _is_matching_beat(frame, "hacking") or _is_matching_beat(frame, "broken_chain")
	if normalized in ["hacking", "hack", "reward_hacking"]:
		if stage_label == "HACKING":
			return true
		if float(world.get("drift_score", 0.0)) >= 0.7:
			return true
		var exploit_pressure = frame.get("event_tracks", {}).get("exploit_pressure", {}).get("value")
		return exploit_pressure != null and float(exploit_pressure) >= 0.7
	if normalized in ["broken_chain", "chain", "scan", "handoff"]:
		if float(world.get("scan_without_handoff_rate", 0.0)) >= 0.2:
			return true
		return float(world.get("false_delivery_rate", 0.0)) >= 0.2
	if normalized in ["routine", "healthy", "baseline"]:
		if stage_label in ["BIRTH", "ROUTE", "PATROL"]:
			return float(world.get("scan_without_handoff_rate", 0.0)) < 0.08 and float(world.get("false_delivery_rate", 0.0)) < 0.08
		if float(world.get("delivery_completion_rate", 0.0)) >= 0.08 and float(world.get("scan_without_handoff_rate", 0.0)) < 0.08:
			return true
		return float(world.get("world_health", 1.0)) >= 0.75 and float(world.get("drift_score", 0.0)) < 0.25
	if normalized in ["drift", "warning"]:
		if stage_label in ["DRIFT", "HACKING"]:
			return true
		if float(world.get("customer_wait_rate", 0.0)) >= 0.2:
			return true
		return float(world.get("world_health", 1.0)) < 0.45
	if stage_label == normalized.to_upper():
		return true
	return false


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
		world_diorama.clear_scene()
		return
	var frame: Dictionary = frames[_current_frame_index]
	var next_index = min(_current_frame_index + 1, frames.size() - 1)
	var next_frame: Dictionary = frames[next_index]
	var act_payload = _current_act()
	var frame_payload = frame.duplicate(false)
	var next_payload = next_frame.duplicate(false)
	frame_payload["playback"] = _playback_payload(_current_frame_index, frames.size())
	next_payload["playback"] = _playback_payload(next_index, frames.size())
	frame_payload["act_bookmarks"] = act_payload.get("bookmarks", [])
	next_payload["act_bookmarks"] = act_payload.get("bookmarks", [])
	world_canvas.set_story_frame(frame_payload, next_payload, alpha, _overlay_mode, _presentation_mode, _use_scene_diorama)
	if _use_scene_diorama:
		world_diorama.visible = true
		world_diorama.set_story_frame(frame_payload, next_payload, alpha, _presentation_mode)
	else:
		world_diorama.clear_scene()
		world_diorama.visible = false
	_update_live_ui(frame_payload)


func _playback_payload(frame_index: int, frame_count: int) -> Dictionary:
	var act = _current_act()
	return {
		"act_index": _current_act_index,
		"act_count": _acts.size(),
		"act": String(act.get("act", "ACT")),
		"frame_index": frame_index,
		"frame_count": frame_count,
	}


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
	stage_label.text = "%s  |  %s  |  %s" % [String(stage.get("label", "PATROLING")), _view_mode, _presentation_mode.to_upper()]
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
	lower_third.visible = verbose and _presentation_mode == "research"
	state_panel.visible = verbose and _presentation_mode == "research"


func _show_error_state(message: String) -> void:
	title_card.visible = true
	lower_third.visible = false
	state_panel.visible = false
	world_diorama.visible = false
	world_diorama.clear_scene()
	fade_overlay.visible = false
	act_label.text = "NO STORY PACKAGE"
	headline_label.text = "EXPORT REQUIRED"
	body_label.text = message
