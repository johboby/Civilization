## HexMap - Renders the hex-grid province map and handles map interaction.
extends Node2D

const HEX_SIZE := 48.0
const HEX_WIDTH := HEX_SIZE * 2.0
const HEX_HEIGHT := sqrt(3.0) * HEX_SIZE

var _hovered_province_id: String = ""
var _province_polygons: Dictionary = {}  # province_id -> PackedVector2Array

func _ready() -> void:
	GameData.game_started.connect(_on_game_changed)
	GameData.turn_processed.connect(_on_turn_processed)
	EventBus.server_state_received.connect(func(_s): queue_redraw())

func _on_game_changed() -> void:
	_build_province_polygons()
	queue_redraw()

func _on_turn_processed(_turn: int, _events: Array) -> void:
	queue_redraw()

# -----------------------------------------------------------------------
# Hex geometry
# -----------------------------------------------------------------------

func hex_to_pixel(col: int, row: int) -> Vector2:
	var x := HEX_SIZE * 1.5 * col
	var y := HEX_HEIGHT * (row + 0.5 * (col % 2))
	return Vector2(x, y)

func _hex_corners(center: Vector2) -> PackedVector2Array:
	var corners := PackedVector2Array()
	for i in 6:
		var angle := deg_to_rad(60.0 * i - 30.0)
		corners.append(center + Vector2(cos(angle), sin(angle)) * (HEX_SIZE - 1))
	return corners

func _build_province_polygons() -> void:
	_province_polygons.clear()
	for province in GameData.provinces.values():
		var center := hex_to_pixel(province["x"], province["y"])
		_province_polygons[province["id"]] = _hex_corners(center)

func pixel_to_province(world_pos: Vector2) -> String:
	var closest_id := ""
	var min_dist := INF
	for province in GameData.provinces.values():
		var center := hex_to_pixel(province["x"], province["y"])
		var dist := world_pos.distance_to(center)
		if dist < HEX_SIZE and dist < min_dist:
			min_dist = dist
			closest_id = province["id"]
	return closest_id

# -----------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------

func _draw() -> void:
	if not GameData.game_active:
		return

	# Draw all provinces
	for province in GameData.provinces.values():
		var center := hex_to_pixel(province["x"], province["y"])
		var corners := _hex_corners(center)

		# Base terrain color
		var fill_color: Color = GameData.TERRAIN_COLORS.get(
			province["terrain"], Color(0.3, 0.3, 0.3))

		# Tint with faction color
		if province["owner_faction_id"] != "":
			var faction: Dictionary = GameData.factions.get(province["owner_faction_id"], {})
			if not faction.is_empty():
				fill_color = fill_color.lerp(faction["color"], 0.35)

		# Draw hex fill
		draw_colored_polygon(corners, fill_color)

		# Draw hex outline
		var outline_color := Color(0.1, 0.1, 0.18)
		var outline_width := 1.0

		if province["id"] == GameData.selected_province_id:
			outline_color = Color(0.96, 0.77, 0.09)  # Gold
			outline_width = 3.0
		elif province["id"] == _hovered_province_id:
			outline_color = Color.WHITE
			outline_width = 2.0

		for i in corners.size():
			var next_i := (i + 1) % corners.size()
			draw_line(corners[i], corners[next_i], outline_color, outline_width, true)

		# Province name
		var font := ThemeDB.fallback_font
		var font_size := 10
		var name_text: String = province["name"]
		if name_text.length() > 8:
			name_text = name_text.substr(0, 8)
		var text_size := font.get_string_size(name_text, HORIZONTAL_ALIGNMENT_CENTER, -1, font_size)
		draw_string(font, center - Vector2(text_size.x / 2, -2), name_text,
			HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color.WHITE)

		# Population info
		var pop_text := "Pop:%dk" % (province["population"] / 1000)
		var pop_size := font.get_string_size(pop_text, HORIZONTAL_ALIGNMENT_CENTER, -1, 8)
		draw_string(font, center - Vector2(pop_size.x / 2, -14), pop_text,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 8, Color(0.8, 0.8, 0.8))

		# Fortification
		if province["fortification"] > 0:
			var fort_text := "Fort:%d" % province["fortification"]
			var fort_size := font.get_string_size(fort_text, HORIZONTAL_ALIGNMENT_CENTER, -1, 8)
			draw_string(font, center - Vector2(fort_size.x / 2, -24), fort_text,
				HORIZONTAL_ALIGNMENT_LEFT, -1, 8, Color(0.95, 0.61, 0.07))

		# Army indicators
		var armies_here := GameData.get_armies_at_province(province["id"])
		for ai in armies_here.size():
			var army: Dictionary = armies_here[ai]
			var faction_data: Dictionary = GameData.factions.get(army["faction_id"], {})
			var army_color: Color = faction_data.get("color", Color.WHITE)
			var dot_pos := center + Vector2(-12 + ai * 14, 30)
			draw_circle(dot_pos, 5, army_color)
			# Soldier count
			var count_text := str(army["soldiers"] / 100)
			draw_string(font, dot_pos + Vector2(-4, 14), count_text,
				HORIZONTAL_ALIGNMENT_LEFT, -1, 7, Color.WHITE)

# -----------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------

func _unhandled_input(event: InputEvent) -> void:
	if not GameData.game_active:
		return

	if event is InputEventMouseButton:
		if event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
			var world_pos := get_global_mouse_position()
			var pid := pixel_to_province(world_pos)
			if pid != "":
				GameData.selected_province_id = pid
				GameData.province_selected.emit(pid)
				EventBus.province_clicked.emit(pid)
				queue_redraw()

	elif event is InputEventMouseMotion:
		var world_pos := get_global_mouse_position()
		var pid := pixel_to_province(world_pos)
		if pid != _hovered_province_id:
			_hovered_province_id = pid
			if pid != "":
				EventBus.province_hovered.emit(pid)
			else:
				EventBus.province_unhovered.emit()
			queue_redraw()
