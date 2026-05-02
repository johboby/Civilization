## ProvinceTooltip - Shows province info on hover.
extends PanelContainer

var _vbox: VBoxContainer
var _visible_province_id: String = ""

func _ready() -> void:
	visible = false
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	custom_minimum_size = Vector2(220, 0)

	var stylebox := StyleBoxFlat.new()
	stylebox.bg_color = Color(0.06, 0.08, 0.15, 0.95)
	stylebox.border_color = Color(0.96, 0.77, 0.09, 0.6)
	stylebox.set_border_width_all(1)
	stylebox.set_corner_radius_all(6)
	stylebox.set_content_margin_all(8)
	stylebox.shadow_color = Color(0, 0, 0, 0.4)
	stylebox.shadow_size = 6
	add_theme_stylebox_override("panel", stylebox)

	_vbox = VBoxContainer.new()
	_vbox.add_theme_constant_override("separation", 2)
	add_child(_vbox)

	EventBus.province_hovered.connect(_on_hover)
	EventBus.province_unhovered.connect(_on_unhover)

func _process(_delta: float) -> void:
	if visible:
		var mouse_pos := get_viewport().get_mouse_position()
		position = mouse_pos + Vector2(16, 16)
		# Keep on screen
		var vp_size := get_viewport_rect().size
		if position.x + size.x > vp_size.x:
			position.x = mouse_pos.x - size.x - 16
		if position.y + size.y > vp_size.y:
			position.y = vp_size.y - size.y - 8

func _on_hover(province_id: String) -> void:
	if province_id == _visible_province_id:
		return
	_visible_province_id = province_id

	var province := GameData.provinces.get(province_id, {})
	if province.is_empty():
		visible = false
		return

	# Clear
	for child in _vbox.get_children():
		child.queue_free()

	# Title
	var title := Label.new()
	title.text = province["name"]
	title.add_theme_font_size_override("font_size", 14)
	title.add_theme_color_override("font_color", Color(0.96, 0.77, 0.09))
	_vbox.add_child(title)

	var sep := HSeparator.new()
	_vbox.add_child(sep)

	# Owner
	var owner_name := "Uncontrolled"
	var owner_color := Color(0.5, 0.5, 0.5)
	if province["owner_faction_id"] != "":
		var faction := GameData.factions.get(province["owner_faction_id"], {})
		owner_name = faction.get("name", "Unknown")
		owner_color = faction.get("color", Color.WHITE)
	_add_row("Owner", owner_name, owner_color)

	_add_row("Terrain", GameData.TERRAIN_NAMES.get(province["terrain"], "?"))
	_add_row("Population", _format_number(province["population"]))
	_add_row("Development", "%.2f" % province["development"])

	var income := GameData.compute_province_income(province)
	_add_row("Food/turn", "+%.1f" % income["food"], Color(0.18, 0.80, 0.44))
	_add_row("Gold/turn", "+%.1f" % income["gold"], Color(0.96, 0.77, 0.09))
	_add_row("Manpower", "+%d" % income["manpower"], Color(0.20, 0.60, 0.86))

	if province["fortification"] > 0:
		_add_row("Fortification", str(province["fortification"]), Color(0.95, 0.61, 0.07))
	if province["unrest"] > 0:
		_add_row("Unrest", "%.1f" % province["unrest"], Color(0.91, 0.30, 0.24))

	# Buildings
	if province["buildings"].size() > 0:
		var bld_label := Label.new()
		bld_label.text = "Buildings:"
		bld_label.add_theme_font_size_override("font_size", 11)
		bld_label.add_theme_color_override("font_color", Color(0.96, 0.77, 0.09))
		_vbox.add_child(bld_label)
		for b in province["buildings"]:
			var bname: String = GameData.BUILDING_NAMES.get(b["type"], "?")
			_add_row("  " + bname, "Lv %d" % b["level"])

	# Armies
	var armies := GameData.get_armies_at_province(province_id)
	if armies.size() > 0:
		var army_label := Label.new()
		army_label.text = "Armies:"
		army_label.add_theme_font_size_override("font_size", 11)
		army_label.add_theme_color_override("font_color", Color(0.91, 0.30, 0.24))
		_vbox.add_child(army_label)
		for army in armies:
			var faction := GameData.factions.get(army["faction_id"], {})
			var fname: String = faction.get("name", "?")
			_add_row("  " + fname, "%d soldiers" % army["soldiers"])

	visible = true

func _on_unhover() -> void:
	visible = false
	_visible_province_id = ""

func _add_row(key: String, value: String, value_color := Color(0.88, 0.88, 0.88)) -> void:
	var hbox := HBoxContainer.new()
	var k := Label.new()
	k.text = key
	k.add_theme_font_size_override("font_size", 11)
	k.add_theme_color_override("font_color", Color(0.63, 0.63, 0.69))
	k.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var v := Label.new()
	v.text = value
	v.add_theme_font_size_override("font_size", 11)
	v.add_theme_color_override("font_color", value_color)
	v.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	hbox.add_child(k)
	hbox.add_child(v)
	_vbox.add_child(hbox)

func _format_number(n: int) -> String:
	if n >= 10000:
		return "%.1fk" % (n / 1000.0)
	return str(n)
