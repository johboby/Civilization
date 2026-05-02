## NotificationManager - Shows floating notifications on screen.
extends CanvasLayer

var _container: VBoxContainer

func _ready() -> void:
	layer = 20

	_container = VBoxContainer.new()
	_container.anchors_preset = Control.PRESET_TOP_RIGHT
	_container.anchor_left = 1.0
	_container.anchor_right = 1.0
	_container.offset_left = -360.0
	_container.offset_top = 60.0
	_container.offset_right = -10.0
	_container.add_theme_constant_override("separation", 6)
	add_child(_container)

	EventBus.show_notification.connect(_show_notification)

func _show_notification(text: String, type: String = "info") -> void:
	var panel := PanelContainer.new()
	var stylebox := StyleBoxFlat.new()

	match type:
		"success":
			stylebox.bg_color = Color(0.11, 0.35, 0.18, 0.92)
			stylebox.border_color = Color(0.18, 0.80, 0.44, 0.8)
		"error":
			stylebox.bg_color = Color(0.35, 0.11, 0.11, 0.92)
			stylebox.border_color = Color(0.91, 0.30, 0.24, 0.8)
		"warning":
			stylebox.bg_color = Color(0.35, 0.28, 0.08, 0.92)
			stylebox.border_color = Color(0.95, 0.77, 0.09, 0.8)
		_:
			stylebox.bg_color = Color(0.11, 0.13, 0.22, 0.92)
			stylebox.border_color = Color(0.20, 0.60, 0.86, 0.8)

	stylebox.set_border_width_all(1)
	stylebox.set_corner_radius_all(6)
	stylebox.shadow_color = Color(0, 0, 0, 0.3)
	stylebox.shadow_size = 4
	stylebox.set_content_margin_all(10)

	panel.add_theme_stylebox_override("panel", stylebox)

	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 8)

	# Icon
	var icon_label := Label.new()
	match type:
		"success": icon_label.text = "[OK]"
		"error": icon_label.text = "[!]"
		"warning": icon_label.text = "[?]"
		_: icon_label.text = "[i]"
	icon_label.add_theme_font_size_override("font_size", 14)
	hbox.add_child(icon_label)

	# Text
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 13)
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hbox.add_child(label)

	panel.add_child(hbox)
	_container.add_child(panel)

	# Fade out animation
	var tween := create_tween()
	tween.tween_interval(3.0)
	tween.tween_property(panel, "modulate:a", 0.0, 0.5)
	tween.tween_callback(panel.queue_free)
