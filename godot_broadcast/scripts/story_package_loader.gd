extends RefCounted
class_name StoryPackageLoader


static func _normalize_path(raw_path: String, base_dir: String = "") -> String:
	var path = String(raw_path).strip_edges().replace("\\", "/")
	if path.is_empty():
		return path
	if path.begins_with("res://") or path.begins_with("user://"):
		return path
	if path.contains(":/") or path.begins_with("/"):
		return path
	if not base_dir.is_empty():
		return base_dir.path_join(path).simplify_path()
	return ProjectSettings.globalize_path("res://").path_join(path).simplify_path()


static func _read_json(path: String) -> Variant:
	var normalized = _normalize_path(path)
	if not FileAccess.file_exists(normalized):
		printerr("StoryPackageLoader: missing JSON file: %s" % normalized)
		return null
	var handle = FileAccess.open(normalized, FileAccess.READ)
	if handle == null:
		printerr("StoryPackageLoader: could not open file: %s" % normalized)
		return null
	var raw_text = handle.get_as_text()
	var parser = JSON.new()
	var parse_status = parser.parse(raw_text)
	if parse_status != OK:
		printerr("StoryPackageLoader: JSON parse error in %s at line %d: %s" % [normalized, parser.get_error_line(), parser.get_error_message()])
		return null
	return parser.data


static func load_from_runtime_pointer(pointer_path: String = "res://runtime/latest_story_package.json") -> Dictionary:
	var normalized_pointer = _normalize_path(pointer_path)
	var pointer_payload: Dictionary = _read_json(normalized_pointer)
	if typeof(pointer_payload) != TYPE_DICTIONARY:
		return {}
	var package_path = String(pointer_payload.get("story_package_path", ""))
	if package_path.is_empty():
		printerr("StoryPackageLoader: runtime pointer does not contain story_package_path.")
		return {}
	return load_from_package_path(package_path, normalized_pointer)


static func load_from_package_path(package_path: String, runtime_pointer_path: String = "") -> Dictionary:
	var normalized_package_path = _normalize_path(package_path)
	var package_payload: Dictionary = _read_json(normalized_package_path)
	if typeof(package_payload) != TYPE_DICTIONARY:
		return {}

	var package_dir = normalized_package_path.get_base_dir()
	var sequence_relative = String(package_payload.get("sequence_file", "sequence.json"))
	var sequence_path = _normalize_path(sequence_relative, package_dir)
	var sequence_payload: Dictionary = _read_json(sequence_path)
	if typeof(sequence_payload) != TYPE_DICTIONARY:
		return {}

	var acts: Array = []
	for act_ref in sequence_payload.get("acts", []):
		if typeof(act_ref) != TYPE_DICTIONARY:
			continue
		var act_path = _normalize_path(String(act_ref.get("file", "")), package_dir)
		var act_payload: Dictionary = _read_json(act_path)
		if typeof(act_payload) != TYPE_DICTIONARY:
			continue
		act_payload["_file"] = act_path
		act_payload["_sequence_ref"] = act_ref
		acts.append(act_payload)

	return {
		"runtime_pointer_path": runtime_pointer_path,
		"package_path": normalized_package_path,
		"package_dir": package_dir,
		"package": package_payload,
		"sequence_path": sequence_path,
		"sequence": sequence_payload,
		"acts": acts,
		"epilogue": sequence_payload.get("epilogue", {}),
		"story_title": String(package_payload.get("story_title", sequence_payload.get("story_title", "GhostMerc Frontier"))),
	}
