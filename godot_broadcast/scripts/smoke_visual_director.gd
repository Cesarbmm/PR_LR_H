extends SceneTree


const WorldCanvas = preload("res://scripts/world_canvas.gd")


func _initialize() -> void:
	var canvas = WorldCanvas.new()
	var frame = {
		"playback": {
			"act_index": 2,
			"act_count": 5,
			"act": "ACT III",
			"frame_index": 60,
			"frame_count": 120,
		},
		"world": {
			"map_width": 1200.0,
			"map_height": 780.0,
			"route_completion_rate": 0.35,
			"world_health": 0.32,
			"ops_kpi": 0.78,
		},
		"stage": {
			"label": "HACKING",
			"tone": "exploit",
		},
		"agent": {
			"x": 0.0,
			"y": 0.0,
			"heading_deg": 0.0,
		},
		"routes": [
			{"zone_name": "safehouse", "x": 144.0, "y": 655.2},
			{"zone_name": "civilian_village", "x": 336.0, "y": 358.8},
			{"zone_name": "checkpoint", "x": 624.0, "y": 265.2},
			{"zone_name": "ruins", "x": 936.0, "y": 171.6},
			{"zone_name": "supply_road", "x": 1008.0, "y": 577.2},
		],
		"focus": {
			"slot_id": 1,
			"tag": "A1",
			"role": "customer",
			"action_name": "up_left|rush|target_14|mark_complete|scan_package",
		},
		"actors": [
			{"slot_id": 1, "tag": "A1", "role_label": "CUSTOMER", "render_role": "customer", "x": 336.0, "y": 358.8},
			{"slot_id": 14, "tag": "A14", "role_label": "RIVAL", "render_role": "rival_courier", "x": 936.0, "y": 171.6},
		],
		"incidents": [],
		"zones": [],
	}
	var directed_position: Vector2 = canvas._agent_world_position(frame)
	if directed_position.length() < 2.0:
		push_error("visual director did not replace invalid agent coordinates")
		canvas.free()
		quit(1)
	else:
		print("visual_director_position=", directed_position)
		canvas.free()
		quit()
