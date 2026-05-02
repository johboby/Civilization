## CharacterPortrait - Procedural character portrait drawn with Godot 2D.
##
## Generates a unique portrait from a character's name and stats.
## Used in officer panels, battle screens, and diplomacy views.
class_name CharacterPortrait
extends Control

const SKIN_COLORS := [
	Color("#f5d6b8"), Color("#e8c4a0"), Color("#d4a574"), Color("#c49060"),
	Color("#b07848"), Color("#a06838"), Color("#f0d0a8"), Color("#e0b888"),
]
const HAIR_COLORS := [
	Color("#1a1a1a"), Color("#3d2b1f"), Color("#5a3825"), Color("#8b6914"),
	Color("#a0522d"), Color("#654321"), Color("#c0c0c0"), Color("#4a3728"),
]
const CLOTH_COLORS := [
	Color("#8b0000"), Color("#00008b"), Color("#006400"), Color("#8b8b00"),
	Color("#4b0082"), Color("#2f4f4f"), Color("#800000"), Color("#191970"),
]
const ROLE_BORDER_COLORS := {
	GameData.CharacterRole.RULER: Color("#ffd700"),
	GameData.CharacterRole.GENERAL: Color("#e74c3c"),
	GameData.CharacterRole.STRATEGIST: Color("#3498db"),
	GameData.CharacterRole.GOVERNOR: Color("#2ecc71"),
	GameData.CharacterRole.DIPLOMAT: Color("#9b59b6"),
	GameData.CharacterRole.SPY: Color("#555555"),
	GameData.CharacterRole.FREE: Color("#888888"),
}

var character_name: String = "Unknown"
var character_role: int = GameData.CharacterRole.FREE
var stats: Dictionary = {"command": 50, "force": 50, "intelligence": 50, "politics": 50, "charisma": 50}

func setup(ch: Dictionary) -> void:
	character_name = ch.get("name", "Unknown")
	character_role = ch.get("role", GameData.CharacterRole.FREE)
	stats = {
		"command": ch.get("command", 50),
		"force": ch.get("force", 50),
		"intelligence": ch.get("intelligence", 50),
		"politics": ch.get("politics", 50),
		"charisma": ch.get("charisma", 50),
	}
	queue_redraw()

func _seed_from_name() -> int:
	var h := 0
	for c in character_name.to_utf8_buffer():
		h = (h * 31 + c) & 0x7FFFFFFF
	return h

func _pick(arr: Array, offset: int = 0) -> Variant:
	var s := _seed_from_name()
	return arr[(s + offset) % arr.size()]

func _draw() -> void:
	var w := size.x
	var h := size.y
	var cx := w / 2.0
	var cy := h / 2.0
	var s := _seed_from_name()

	# Background
	draw_rect(Rect2(0, 0, w, h), Color(0.1, 0.1, 0.18), true, -1.0, false)

	var skin: Color = _pick(SKIN_COLORS, 0)
	var hair: Color = _pick(HAIR_COLORS, 1)
	var cloth: Color = _pick(CLOTH_COLORS, 2)

	var face_rx := w * 0.32
	var face_ry := h * 0.36

	# Clothing
	draw_circle(Vector2(cx, h + 4), w * 0.42, cloth)

	# Neck
	draw_rect(Rect2(cx - 5, cy + face_ry * 0.7, 10, h * 0.12), skin)

	# Face
	draw_circle(Vector2(cx, cy - 3), minf(face_rx, face_ry), skin)

	# Eyes
	var eye_y := cy - h * 0.06
	var eye_off := face_rx * 0.35
	draw_circle(Vector2(cx - eye_off, eye_y), 3.5, Color.WHITE)
	draw_circle(Vector2(cx + eye_off, eye_y), 3.5, Color.WHITE)
	draw_circle(Vector2(cx - eye_off, eye_y), 2.0, Color(0.15, 0.1, 0.05))
	draw_circle(Vector2(cx + eye_off, eye_y), 2.0, Color(0.15, 0.1, 0.05))

	# Eyebrows
	var brow_w := 1.5 + stats["command"] / 100.0 * 1.5
	draw_line(
		Vector2(cx - eye_off - 4, eye_y - 5),
		Vector2(cx - eye_off + 4, eye_y - 6),
		hair, brow_w
	)
	draw_line(
		Vector2(cx + eye_off - 4, eye_y - 6),
		Vector2(cx + eye_off + 4, eye_y - 5),
		hair, brow_w
	)

	# Mouth
	var mouth_y := cy + h * 0.1
	var mouth_w := 4 + stats["charisma"] / 100.0 * 5
	draw_line(
		Vector2(cx - mouth_w, mouth_y),
		Vector2(cx + mouth_w, mouth_y + stats["charisma"] / 100.0 * 2),
		Color(0.55, 0.27, 0.07), 1.5
	)

	# Hair
	var hair_style := s % 5
	match hair_style:
		0:  # Topknot
			draw_circle(Vector2(cx, cy - face_ry - 2), 8, hair)
			var pts := PackedVector2Array([
				Vector2(cx - face_rx * 0.7, cy - face_ry * 0.4),
				Vector2(cx, cy - face_ry - 8),
				Vector2(cx + face_rx * 0.7, cy - face_ry * 0.4),
			])
			draw_colored_polygon(pts, hair)
		1:  # Short
			draw_circle(Vector2(cx, cy - face_ry * 0.55), face_rx * 0.85, hair)
		2:  # Helmet (for generals)
			var helmet_pts := PackedVector2Array([
				Vector2(cx - face_rx * 0.8, cy - face_ry * 0.2),
				Vector2(cx - face_rx * 0.8, cy - face_ry * 0.7),
				Vector2(cx, cy - face_ry * 1.1),
				Vector2(cx + face_rx * 0.8, cy - face_ry * 0.7),
				Vector2(cx + face_rx * 0.8, cy - face_ry * 0.2),
			])
			draw_colored_polygon(helmet_pts, Color(0.44, 0.5, 0.56))
			draw_line(Vector2(cx, cy - face_ry * 1.1), Vector2(cx, cy - face_ry * 0.2), Color(0.38, 0.38, 0.38), 3)
		3:  # Flowing
			draw_circle(Vector2(cx, cy - face_ry * 0.6), face_rx * 0.8, hair)
			draw_line(Vector2(cx - face_rx * 0.8, cy - face_ry * 0.1), Vector2(cx - face_rx * 0.7, cy + face_ry * 0.5), hair, 4)
			draw_line(Vector2(cx + face_rx * 0.8, cy - face_ry * 0.1), Vector2(cx + face_rx * 0.7, cy + face_ry * 0.5), hair, 4)
		4:  # Bald
			pass

	# Crown for rulers
	if character_role == GameData.CharacterRole.RULER:
		var crown_y := cy - face_ry - 4
		var crown_pts := PackedVector2Array([
			Vector2(cx - 10, crown_y),
			Vector2(cx - 8, crown_y - 8),
			Vector2(cx - 3, crown_y - 3),
			Vector2(cx, crown_y - 10),
			Vector2(cx + 3, crown_y - 3),
			Vector2(cx + 8, crown_y - 8),
			Vector2(cx + 10, crown_y),
		])
		draw_colored_polygon(crown_pts, Color("#ffd700"))
		draw_circle(Vector2(cx, crown_y - 5), 2, Color("#e74c3c"))

	# Role border
	var border_color: Color = ROLE_BORDER_COLORS.get(character_role, Color("#888"))
	draw_rect(Rect2(1, 1, w - 2, h - 2), border_color, false, 2.0)
