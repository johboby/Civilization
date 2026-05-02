## Minimap - Renders a small overview of the entire game map.
extends Control

const MINIMAP_SIZE := Vector2(200, 160)
const PADDING := 4.0

var _minimap_texture: ImageTexture = null

func _ready() -> void:
	custom_minimum_size = MINIMAP_SIZE
	GameData.game_started.connect(_regenerate)
	GameData.turn_processed.connect(func(_t, _e): _regenerate())
	EventBus.server_state_received.connect(func(_s): _regenerate())

func _regenerate() -> void:
	if not GameData.game_active:
		return

	var img := Image.create(int(MINIMAP_SIZE.x), int(MINIMAP_SIZE.y), false, Image.FORMAT_RGBA8)
	img.fill(Color(0.06, 0.06, 0.12, 0.85))

	var scale_x := (MINIMAP_SIZE.x - PADDING * 2) / maxf(1, GameData.map_cols * 1.5 * 6.0)
	var scale_y := (MINIMAP_SIZE.y - PADDING * 2) / maxf(1, GameData.map_rows * sqrt(3.0) * 6.0)
	var map_scale := minf(scale_x, scale_y)

	for province in GameData.provinces.values():
		var px := province["x"] * 1.5 * 6.0 * map_scale + PADDING
		var py := (province["y"] + 0.5 * (province["x"] % 2)) * sqrt(3.0) * 6.0 * map_scale + PADDING

		var color: Color = GameData.TERRAIN_COLORS.get(province["terrain"], Color(0.3, 0.3, 0.3))

		if province["owner_faction_id"] != "":
			var faction: Dictionary = GameData.factions.get(province["owner_faction_id"], {})
			if not faction.is_empty():
				color = color.lerp(faction["color"], 0.5)

		# Draw a small filled hex approximated as a rect
		var cx := int(px)
		var cy := int(py)
		for dx in range(-2, 3):
			for dy in range(-2, 3):
				var ix := cx + dx
				var iy := cy + dy
				if ix >= 0 and ix < int(MINIMAP_SIZE.x) and iy >= 0 and iy < int(MINIMAP_SIZE.y):
					img.set_pixel(ix, iy, color)

	# Highlight selected province
	if GameData.selected_province_id != "":
		var sel := GameData.provinces.get(GameData.selected_province_id, {})
		if not sel.is_empty():
			var sx := int(sel["x"] * 1.5 * 6.0 * map_scale + PADDING)
			var sy := int((sel["y"] + 0.5 * (sel["x"] % 2)) * sqrt(3.0) * 6.0 * map_scale + PADDING)
			var gold := Color(0.96, 0.77, 0.09)
			for dx in range(-3, 4):
				for dy in range(-3, 4):
					if abs(dx) == 3 or abs(dy) == 3:
						var ix := sx + dx
						var iy := sy + dy
						if ix >= 0 and ix < int(MINIMAP_SIZE.x) and iy >= 0 and iy < int(MINIMAP_SIZE.y):
							img.set_pixel(ix, iy, gold)

	_minimap_texture = ImageTexture.create_from_image(img)
	queue_redraw()

func _draw() -> void:
	# Background
	draw_rect(Rect2(Vector2.ZERO, MINIMAP_SIZE), Color(0.06, 0.06, 0.12, 0.85))

	if _minimap_texture:
		draw_texture(_minimap_texture, Vector2.ZERO)

	# Border
	draw_rect(Rect2(Vector2.ZERO, MINIMAP_SIZE), Color(0.18, 0.24, 0.40), false, 1.0)

func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		# Click on minimap to move camera
		var click_pos := event.position
		var scale_x := (MINIMAP_SIZE.x - PADDING * 2) / maxf(1, GameData.map_cols * 1.5 * 6.0)
		var scale_y := (MINIMAP_SIZE.y - PADDING * 2) / maxf(1, GameData.map_rows * sqrt(3.0) * 6.0)
		var map_scale := minf(scale_x, scale_y)

		# Find closest province to click
		var closest_id := ""
		var min_dist := INF
		for province in GameData.provinces.values():
			var px := province["x"] * 1.5 * 6.0 * map_scale + PADDING
			var py := (province["y"] + 0.5 * (province["x"] % 2)) * sqrt(3.0) * 6.0 * map_scale + PADDING
			var dist := click_pos.distance_to(Vector2(px, py))
			if dist < min_dist:
				min_dist = dist
				closest_id = province["id"]

		if closest_id != "":
			GameData.selected_province_id = closest_id
			GameData.province_selected.emit(closest_id)
			EventBus.province_clicked.emit(closest_id)
