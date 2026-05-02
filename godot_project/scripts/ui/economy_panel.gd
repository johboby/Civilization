## EconomyPanel - Shows detailed economy overview with bar charts.
extends PanelContainer

var _vbox: VBoxContainer

func _ready() -> void:
	visible = false
	custom_minimum_size = Vector2(500, 400)

	var stylebox := StyleBoxFlat.new()
	stylebox.bg_color = Color(0.08, 0.10, 0.18, 0.96)
	stylebox.border_color = Color(0.96, 0.77, 0.09, 0.5)
	stylebox.set_border_width_all(1)
	stylebox.set_corner_radius_all(8)
	stylebox.set_content_margin_all(16)
	stylebox.shadow_color = Color(0, 0, 0, 0.5)
	stylebox.shadow_size = 12
	add_theme_stylebox_override("panel", stylebox)

	_vbox = VBoxContainer.new()
	_vbox.add_theme_constant_override("separation", 8)
	add_child(_vbox)

func show_economy() -> void:
	_build_content()
	visible = true

func hide_economy() -> void:
	visible = false

func _build_content() -> void:
	for child in _vbox.get_children():
		child.queue_free()

	var faction := GameData.factions.get(GameData.local_faction_id, {})
	if faction.is_empty():
		return

	# Title
	var title := Label.new()
	title.text = "Economy Overview - %s" % faction["name"]
	title.add_theme_font_size_override("font_size", 18)
	title.add_theme_color_override("font_color", Color(0.96, 0.77, 0.09))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_vbox.add_child(title)

	_vbox.add_child(HSeparator.new())

	# Resource summary
	var summary_grid := GridContainer.new()
	summary_grid.columns = 3
	_add_resource_card(summary_grid, "Gold", int(faction["gold"]), Color(0.96, 0.77, 0.09))
	_add_resource_card(summary_grid, "Food", int(faction["food"]), Color(0.18, 0.80, 0.44))
	_add_resource_card(summary_grid, "Manpower", faction["manpower"], Color(0.20, 0.60, 0.86))
	_vbox.add_child(summary_grid)

	_vbox.add_child(HSeparator.new())

	# Province income breakdown
	var income_label := Label.new()
	income_label.text = "Province Income Breakdown"
	income_label.add_theme_font_size_override("font_size", 14)
	income_label.add_theme_color_override("font_color", Color(0.96, 0.77, 0.09))
	_vbox.add_child(income_label)

	var provinces := GameData.get_faction_provinces(GameData.local_faction_id)
	var max_gold := 1.0
	var max_food := 1.0
	for p in provinces:
		var income := GameData.compute_province_income(p)
		max_gold = maxf(max_gold, income["gold"])
		max_food = maxf(max_food, income["food"])

	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(0, 200)
	var income_vbox := VBoxContainer.new()
	income_vbox.add_theme_constant_override("separation", 4)
	income_vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	for p in provinces:
		var income := GameData.compute_province_income(p)
		var hbox := HBoxContainer.new()
		hbox.add_theme_constant_override("separation", 8)

		var name_lbl := Label.new()
		name_lbl.text = p["name"]
		name_lbl.custom_minimum_size = Vector2(100, 0)
		name_lbl.add_theme_font_size_override("font_size", 11)
		hbox.add_child(name_lbl)

		# Gold bar
		_add_bar(hbox, income["gold"], max_gold, Color(0.96, 0.77, 0.09), "%.1f" % income["gold"])
		# Food bar
		_add_bar(hbox, income["food"], max_food, Color(0.18, 0.80, 0.44), "%.1f" % income["food"])

		income_vbox.add_child(hbox)

	scroll.add_child(income_vbox)
	_vbox.add_child(scroll)

	# Close button
	var close_btn := Button.new()
	close_btn.text = "Close"
	close_btn.pressed.connect(hide_economy)
	_vbox.add_child(close_btn)

func _add_resource_card(parent: Node, res_name: String, value: int, color: Color) -> void:
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var vbox := VBoxContainer.new()
	vbox.alignment = BoxContainer.ALIGNMENT_CENTER

	var val_lbl := Label.new()
	val_lbl.text = str(value)
	val_lbl.add_theme_font_size_override("font_size", 24)
	val_lbl.add_theme_color_override("font_color", color)
	val_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(val_lbl)

	var name_lbl := Label.new()
	name_lbl.text = res_name
	name_lbl.add_theme_font_size_override("font_size", 11)
	name_lbl.add_theme_color_override("font_color", Color(0.55, 0.55, 0.66))
	name_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(name_lbl)

	panel.add_child(vbox)
	parent.add_child(panel)

func _add_bar(parent: Node, value: float, max_val: float, color: Color, label_text: String) -> void:
	var bar_container := HBoxContainer.new()
	bar_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var bg := ColorRect.new()
	bg.custom_minimum_size = Vector2(100, 12)
	bg.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	bg.color = Color(0.15, 0.15, 0.22)

	var fill := ColorRect.new()
	fill.color = color
	fill.custom_minimum_size = Vector2(maxf(2, 100 * (value / max_val)), 12)
	bg.add_child(fill)
	bar_container.add_child(bg)

	var lbl := Label.new()
	lbl.text = label_text
	lbl.custom_minimum_size = Vector2(40, 0)
	lbl.add_theme_font_size_override("font_size", 10)
	lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	bar_container.add_child(lbl)

	parent.add_child(bar_container)
