## BattleScene - Animated battle preview/result display.
##
## Shows a visual representation of combat between two armies
## with animated unit bars, general portraits, and result text.
extends CanvasLayer

var _panel: PanelContainer
var _attacker: Dictionary = {}
var _defender: Dictionary = {}
var _result_text: String = ""
var _animation_progress: float = 0.0
var _is_animating: bool = false

func _ready() -> void:
	layer = 15
	visible = false

func show_battle(attacker_army: Dictionary, defender_army: Dictionary, result: String) -> void:
	_attacker = attacker_army
	_defender = defender_army
	_result_text = result
	_animation_progress = 0.0
	_is_animating = true
	visible = true
	_build_ui()

func _build_ui() -> void:
	# Clear
	for child in get_children():
		child.queue_free()

	# Backdrop
	var backdrop := ColorRect.new()
	backdrop.anchors_preset = Control.PRESET_FULL_RECT
	backdrop.color = Color(0, 0, 0, 0.7)
	add_child(backdrop)

	_panel = PanelContainer.new()
	_panel.anchors_preset = Control.PRESET_CENTER
	_panel.custom_minimum_size = Vector2(600, 350)
	_panel.offset_left = -300
	_panel.offset_top = -175
	_panel.offset_right = 300
	_panel.offset_bottom = 175

	var stylebox := StyleBoxFlat.new()
	stylebox.bg_color = Color(0.08, 0.10, 0.18, 0.96)
	stylebox.border_color = Color(0.91, 0.30, 0.24, 0.8)
	stylebox.set_border_width_all(2)
	stylebox.set_corner_radius_all(10)
	stylebox.set_content_margin_all(20)
	stylebox.shadow_color = Color(0, 0, 0, 0.6)
	stylebox.shadow_size = 16
	_panel.add_theme_stylebox_override("panel", stylebox)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 12)

	# Title
	var title := Label.new()
	title.text = "BATTLE"
	title.add_theme_font_size_override("font_size", 22)
	title.add_theme_color_override("font_color", Color(0.91, 0.30, 0.24))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(title)

	# VS display
	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 20)
	hbox.alignment = BoxContainer.ALIGNMENT_CENTER

	_add_army_display(hbox, _attacker, "Attacker", Color(0.91, 0.30, 0.24))
	var vs_label := Label.new()
	vs_label.text = "VS"
	vs_label.add_theme_font_size_override("font_size", 28)
	vs_label.add_theme_color_override("font_color", Color(0.96, 0.77, 0.09))
	hbox.add_child(vs_label)
	_add_army_display(hbox, _defender, "Defender", Color(0.20, 0.60, 0.86))

	vbox.add_child(hbox)

	# Result
	var result_label := Label.new()
	result_label.text = _result_text
	result_label.add_theme_font_size_override("font_size", 16)
	result_label.add_theme_color_override("font_color", Color(0.96, 0.77, 0.09))
	result_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	result_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	vbox.add_child(result_label)

	# Close button
	var close_btn := Button.new()
	close_btn.text = "Continue"
	close_btn.pressed.connect(_close)
	close_btn.custom_minimum_size = Vector2(150, 36)
	var btn_container := CenterContainer.new()
	btn_container.add_child(close_btn)
	vbox.add_child(btn_container)

	_panel.add_child(vbox)
	add_child(_panel)

func _add_army_display(parent: Node, army: Dictionary, role_text: String, color: Color) -> void:
	var vbox := VBoxContainer.new()
	vbox.custom_minimum_size = Vector2(180, 0)
	vbox.alignment = BoxContainer.ALIGNMENT_CENTER

	var role_lbl := Label.new()
	role_lbl.text = role_text
	role_lbl.add_theme_font_size_override("font_size", 12)
	role_lbl.add_theme_color_override("font_color", Color(0.55, 0.55, 0.66))
	role_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(role_lbl)

	var faction := GameData.factions.get(army.get("faction_id", ""), {})
	var fname := Label.new()
	fname.text = faction.get("name", "Unknown")
	fname.add_theme_font_size_override("font_size", 15)
	fname.add_theme_color_override("font_color", faction.get("color", Color.WHITE))
	fname.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(fname)

	var soldiers := Label.new()
	soldiers.text = "%d soldiers" % army.get("soldiers", 0)
	soldiers.add_theme_font_size_override("font_size", 20)
	soldiers.add_theme_color_override("font_color", color)
	soldiers.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(soldiers)

	# Composition
	var comp := Label.new()
	comp.text = "Inf: %d  Cav: %d  Arc: %d" % [
		army.get("infantry", 0), army.get("cavalry", 0), army.get("archers", 0)]
	comp.add_theme_font_size_override("font_size", 11)
	comp.add_theme_color_override("font_color", Color(0.63, 0.63, 0.69))
	comp.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(comp)

	# Morale bar
	var morale_hbox := HBoxContainer.new()
	var morale_lbl := Label.new()
	morale_lbl.text = "Morale: "
	morale_lbl.add_theme_font_size_override("font_size", 10)
	morale_hbox.add_child(morale_lbl)

	var morale_bg := ColorRect.new()
	morale_bg.custom_minimum_size = Vector2(80, 8)
	morale_bg.color = Color(0.15, 0.15, 0.22)
	morale_bg.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var morale_fill := ColorRect.new()
	var morale_pct := army.get("morale", 100.0) / 100.0
	morale_fill.custom_minimum_size = Vector2(80 * morale_pct, 8)
	morale_fill.color = Color(0.18, 0.80, 0.44) if morale_pct > 0.5 else Color(0.91, 0.30, 0.24)
	morale_bg.add_child(morale_fill)
	morale_hbox.add_child(morale_bg)
	vbox.add_child(morale_hbox)

	# General
	var gen := GameData.characters.get(army.get("general_id", ""), {})
	if not gen.is_empty():
		var gen_lbl := Label.new()
		gen_lbl.text = "General: %s" % gen["name"]
		gen_lbl.add_theme_font_size_override("font_size", 11)
		gen_lbl.add_theme_color_override("font_color", Color(0.96, 0.77, 0.09))
		gen_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		vbox.add_child(gen_lbl)

	parent.add_child(vbox)

func _close() -> void:
	visible = false
	_is_animating = false
	for child in get_children():
		child.queue_free()

func _process(delta: float) -> void:
	if _is_animating:
		_animation_progress += delta
		if _animation_progress > 3.0:
			_is_animating = false
