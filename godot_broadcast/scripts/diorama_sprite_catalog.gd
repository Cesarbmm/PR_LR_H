extends RefCounted


const FRAME_SIZE = Vector2(64.0, 64.0)
const FRAME_COUNT = 4

const MODE_ROWS = {
	"idle": 0,
	"walk": 1,
	"scan": 2,
	"handoff": 3,
	"wait": 4,
}

const MODE_SPEEDS = {
	"idle": 3.0,
	"walk": 8.0,
	"scan": 7.0,
	"handoff": 5.5,
	"wait": 3.6,
}

const ROLE_PATHS = {
	"courier": "res://assets/diorama/spritesheets/courier_sheet.png",
	"customer": "res://assets/diorama/spritesheets/customer_sheet.png",
	"supervisor": "res://assets/diorama/spritesheets/supervisor_sheet.png",
	"rival": "res://assets/diorama/spritesheets/rival_sheet.png",
	"thief": "res://assets/diorama/spritesheets/thief_sheet.png",
}

static var _texture_cache: Dictionary = {}
static var _frames_cache: Dictionary = {}


static func texture_for_role(role: String) -> Texture2D:
	var key = _sheet_key_for_role(role)
	if _texture_cache.has(key):
		return _texture_cache[key]
	var path = String(ROLE_PATHS.get(key, ROLE_PATHS["customer"]))
	var bytes = FileAccess.get_file_as_bytes(path)
	if bytes.is_empty():
		return null
	var image := Image.new()
	var err = image.load_png_from_buffer(bytes)
	if err != OK:
		push_error("failed to load sprite sheet: %s" % path)
		return null
	var texture = ImageTexture.create_from_image(image)
	_texture_cache[key] = texture
	return texture


static func sprite_frames_for_role(role: String) -> SpriteFrames:
	var key = _sheet_key_for_role(role)
	if _frames_cache.has(key):
		return _frames_cache[key]
	var texture = texture_for_role(role)
	if texture == null:
		return null
	var frames := SpriteFrames.new()
	for mode in MODE_ROWS.keys():
		frames.add_animation(mode)
		frames.set_animation_loop(mode, true)
		frames.set_animation_speed(mode, float(MODE_SPEEDS.get(mode, 4.0)))
		for frame_index in range(FRAME_COUNT):
			var atlas := AtlasTexture.new()
			atlas.atlas = texture
			atlas.region = frame_rect(mode, frame_index)
			frames.add_frame(mode, atlas, 1.0)
	_frames_cache[key] = frames
	return frames


static func frame_rect(mode: String, frame_index: int) -> Rect2:
	var row = int(MODE_ROWS.get(mode, 0))
	var frame = posmod(frame_index, FRAME_COUNT)
	return Rect2(Vector2(float(frame) * FRAME_SIZE.x, float(row) * FRAME_SIZE.y), FRAME_SIZE)


static func _sheet_key_for_role(role: String) -> String:
	match role:
		"agent":
			return "courier"
		"customer", "civilian", "pedestrian", "scavenger":
			return "customer"
		"supervisor", "ally":
			return "supervisor"
		"rival_courier", "armed_neutral", "militia":
			return "rival"
		"thief", "hostile", "smuggler":
			return "thief"
		_:
			return "customer"
