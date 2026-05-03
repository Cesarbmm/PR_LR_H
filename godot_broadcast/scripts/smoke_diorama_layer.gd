extends SceneTree


const WorldDiorama = preload("res://scripts/world_diorama.gd")


func _initialize() -> void:
	var diorama = WorldDiorama.new()
	root.add_child(diorama)
	await process_frame
	diorama.set_story_frame(_sample_frame(), {}, 0.0, "public")
	var actor_count = diorama.get_node("ActorLayer").get_child_count()
	var zone_count = diorama.get_node("ZoneLayer").get_child_count()
	var scenery_count = diorama.get_node("SceneryLayer").get_child_count()
	var ambient_count = diorama.get_node("AmbientLayer").get_child_count()
	var prop_count = diorama.get_node("PropLayer").get_child_count()
	var incident_count = diorama.get_node("IncidentLayer").get_child_count()
	var action_count = diorama.get_node("ActionLayer").get_child_count()
	var sprite_actor_count = 0
	var fx_actor_count = 0
	for actor in diorama.get_node("ActorLayer").get_children():
		if actor.find_child("SpriteOverlay", true, false) != null:
			sprite_actor_count += 1
		if actor.find_child("FXRoot", true, false) != null:
			fx_actor_count += 1
	if actor_count < 2 or zone_count < 2 or scenery_count < 3 or ambient_count < 3 or prop_count < 3 or incident_count < 1 or action_count < 2 or sprite_actor_count < 2 or fx_actor_count < 2:
		push_error("world diorama did not materialize expected scene nodes")
		quit(1)
	var coherence_frame = _coherence_frame()
	var corrected_agent: Vector2 = diorama._agent_world_position(coherence_frame, {})
	var focus_world = diorama._focus_world_position(coherence_frame, {})
	var raw_agent = Vector2(float(coherence_frame.get("agent", {}).get("x", 0.0)), float(coherence_frame.get("agent", {}).get("y", 0.0)))
	if focus_world == null:
		push_error("coherence smoke frame did not produce a focus target")
		quit(1)
	if corrected_agent.distance_to(focus_world) >= raw_agent.distance_to(focus_world):
		push_error("courier director failed to improve coherence against the focus target")
		quit(1)
	if corrected_agent.distance_to(focus_world) > 130.0:
		push_error("courier director kept the courier too far from the focus target")
		quit(1)
	print("diorama_counts actors=%d sprite_actors=%d fx_actors=%d zones=%d scenery=%d ambient=%d props=%d incidents=%d action=%d" % [actor_count, sprite_actor_count, fx_actor_count, zone_count, scenery_count, ambient_count, prop_count, incident_count, action_count])
	quit()


func _sample_frame() -> Dictionary:
	return {
		"playback": {
			"act_index": 4,
			"act_count": 5,
			"act": "ACT IV",
			"frame_index": 12,
			"frame_count": 120,
		},
		"world": {
			"map_width": 1200.0,
			"map_height": 780.0,
			"route_completion_rate": 0.62,
			"world_health": 0.22,
			"ops_kpi": 0.84,
			"scan_without_handoff_rate": 0.72,
			"false_delivery_rate": 0.51,
			"customer_wait_rate": 0.38,
			"drift_score": 0.91,
		},
		"stage": {
			"label": "HACKING",
			"tone": "exploit",
		},
		"beat": {
			"id": "hacking",
			"label": "HACKING",
		},
		"events": {
			"counts": {
				"customers": 4,
				"pedestrians": 5,
				"supervisors": 2,
				"thieves": 1,
			},
		},
		"agent": {
			"x": 0.0,
			"y": 0.0,
			"heading_deg": 0.0,
		},
		"routes": [
			{"zone_name": "depot", "x": 120.0, "y": 640.0},
			{"zone_name": "service_alley", "x": 360.0, "y": 420.0},
			{"zone_name": "locker_bank", "x": 760.0, "y": 220.0},
			{"zone_name": "apartment_block", "x": 980.0, "y": 320.0},
		],
		"focus": {
			"slot_id": 7,
			"action_name": "rush|target_7|mark_complete|scan_package",
		},
		"actors": [
			{"slot_id": 7, "tag": "C7", "role_label": "CUSTOMER", "render_role": "customer", "x": 980.0, "y": 320.0, "heading_deg": 180.0},
			{"slot_id": 9, "tag": "T9", "role_label": "THIEF", "render_role": "thief", "x": 760.0, "y": 220.0, "heading_deg": 90.0, "under_attack": false},
			{"slot_id": 12, "tag": "S1", "role_label": "SUPERVISOR", "render_role": "supervisor", "x": 360.0, "y": 420.0, "heading_deg": 45.0},
		],
		"zones": [
			{"name": "Depot", "kind": "depot", "x": 120.0, "y": 640.0, "radius": 90.0},
			{"name": "Service Alley", "kind": "service_alley", "x": 360.0, "y": 420.0, "radius": 76.0},
			{"name": "Locker Bank", "kind": "locker_bank", "x": 760.0, "y": 220.0, "radius": 80.0},
			{"name": "Apartment Block", "kind": "apartment_block", "x": 980.0, "y": 320.0, "radius": 96.0},
		],
		"incidents": [
			{"incident_type": "theft_risk", "zone_name": "Locker Bank", "resolved": false, "escalated": true},
		],
	}


func _coherence_frame() -> Dictionary:
	return {
		"playback": {
			"act_index": 2,
			"act_count": 5,
			"act": "ACT II",
			"frame_index": 19,
			"frame_count": 83,
		},
		"world": {
			"map_width": 1200.0,
			"map_height": 780.0,
			"route_completion_rate": 0.333,
			"world_health": 0.22,
			"ops_kpi": 0.84,
			"scan_without_handoff_rate": 0.72,
			"false_delivery_rate": 0.51,
			"customer_wait_rate": 0.22,
			"drift_score": 0.91,
		},
		"stage": {
			"label": "HACKING",
			"tone": "exploit",
		},
		"beat": {
			"id": "hacking",
			"label": "HACKING",
		},
		"agent": {
			"x": 964.7,
			"y": 640.9,
			"heading_deg": 0.0,
		},
		"routes": [
			{"zone_name": "safehouse", "x": 120.0, "y": 655.2},
			{"zone_name": "civilian_village", "x": 336.0, "y": 265.2},
			{"zone_name": "checkpoint", "x": 696.0, "y": 358.8},
			{"zone_name": "ruins", "x": 984.0, "y": 171.6},
			{"zone_name": "supply_road", "x": 984.0, "y": 561.6},
			{"zone_name": "clinic", "x": 504.0, "y": 514.8},
		],
		"focus": {
			"slot_id": 0,
			"tag": "A0",
			"role": "thief",
			"action_name": "right|careful|target_14|mark_complete|scan_package",
		},
		"actors": [
			{"slot_id": 0, "tag": "A0", "role_label": "THIEF", "render_role": "hostile", "x": 672.3, "y": 376.3, "heading_deg": 0.0, "focus": true},
			{"slot_id": 1, "tag": "A1", "role_label": "CUSTOMER", "render_role": "customer", "x": 336.0, "y": 265.2, "heading_deg": 0.0},
			{"slot_id": 2, "tag": "A2", "role_label": "CUSTOMER", "render_role": "customer", "x": 504.0, "y": 514.8, "heading_deg": 0.0},
		],
		"zones": [
			{"name": "safehouse", "kind": "depot", "x": 120.0, "y": 655.2, "radius": 112.0},
			{"name": "civilian_village", "kind": "shop_row", "x": 336.0, "y": 265.2, "radius": 138.0},
			{"name": "checkpoint", "kind": "service_alley", "x": 696.0, "y": 358.8, "radius": 122.0},
			{"name": "ruins", "kind": "locker_bank", "x": 984.0, "y": 171.6, "radius": 112.0},
			{"name": "supply_road", "kind": "crosswalk", "x": 984.0, "y": 561.6, "radius": 150.0},
			{"name": "clinic", "kind": "apartment_block", "x": 504.0, "y": 514.8, "radius": 88.0},
		],
		"incidents": [
			{"incident_type": "theft_risk", "zone_name": "checkpoint", "resolved": false, "escalated": false},
		],
		"events": {
			"counts": {
				"customers": 3,
				"pedestrians": 1,
				"supervisors": 0,
				"thieves": 1,
			},
		},
	}
