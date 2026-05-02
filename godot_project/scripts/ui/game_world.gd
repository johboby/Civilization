## GameWorld - Main game scene controller.
##
## Handles camera movement, turn processing, and coordinates between
## the hex map, HUD, and game engine.
extends Node2D

@onready var camera: Camera2D = $Camera2D
@onready var hex_map: Node2D = $HexMap

var _turn_processor := TurnProcessor.new()

const CAMERA_SPEED := 600.0
const ZOOM_SPEED := 0.1
const MIN_ZOOM := 0.3
const MAX_ZOOM := 2.5

var _is_dragging := false
var _drag_start := Vector2.ZERO

func _ready() -> void:
	EventBus.action_end_turn.connect(_on_end_turn)
	EventBus.action_build.connect(_on_build)
	EventBus.action_recruit_army.connect(_on_recruit_army)
	EventBus.action_move_army.connect(_on_move_army)
	EventBus.action_research.connect(_on_research)
	EventBus.action_diplomacy.connect(_on_diplomacy)
	EventBus.action_recruit_character.connect(_on_recruit_character)
	EventBus.action_assign_general.connect(_on_assign_general)

	# Center camera on map
	if GameData.game_active:
		var cx := GameData.map_cols * 48.0 * 1.5 / 2.0
		var cy := GameData.map_rows * (sqrt(3.0) * 48.0) / 2.0
		camera.position = Vector2(cx, cy)

func _process(delta: float) -> void:
	_handle_camera_input(delta)

func _handle_camera_input(delta: float) -> void:
	var move := Vector2.ZERO
	if Input.is_action_pressed("camera_pan_up"):
		move.y -= 1
	if Input.is_action_pressed("camera_pan_down"):
		move.y += 1
	if Input.is_action_pressed("camera_pan_left"):
		move.x -= 1
	if Input.is_action_pressed("camera_pan_right"):
		move.x += 1

	if move != Vector2.ZERO:
		camera.position += move.normalized() * CAMERA_SPEED * delta / camera.zoom.x

func _unhandled_input(event: InputEvent) -> void:
	# Zoom
	if event.is_action_pressed("camera_zoom_in"):
		camera.zoom *= (1.0 + ZOOM_SPEED)
		camera.zoom = camera.zoom.clamp(Vector2(MIN_ZOOM, MIN_ZOOM), Vector2(MAX_ZOOM, MAX_ZOOM))
	elif event.is_action_pressed("camera_zoom_out"):
		camera.zoom *= (1.0 - ZOOM_SPEED)
		camera.zoom = camera.zoom.clamp(Vector2(MIN_ZOOM, MIN_ZOOM), Vector2(MAX_ZOOM, MAX_ZOOM))

	# Middle mouse drag
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_MIDDLE:
			_is_dragging = event.pressed
			_drag_start = event.position
	elif event is InputEventMouseMotion and _is_dragging:
		camera.position -= event.relative / camera.zoom.x

	# Hotkeys
	if event.is_action_pressed("toggle_tech_tree"):
		EventBus.show_tech_panel.emit()
	elif event.is_action_pressed("toggle_diplomacy"):
		EventBus.show_diplomacy_panel.emit()
	elif event.is_action_pressed("end_turn"):
		_on_end_turn()

# -----------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------

func _on_end_turn() -> void:
	if NetworkManager.is_online_mode:
		NetworkManager.send_ready()
		NetworkManager.send_force_turn()
	else:
		var events := _turn_processor.process_turn()
		GameData.turn_processed.emit(GameData.turn, events)
		hex_map.queue_redraw()

func _on_build(province_id: String, building_type: int) -> void:
	if NetworkManager.is_online_mode:
		NetworkManager.send_action({
			"type": "build",
			"province_id": province_id,
			"building_type": GameData.BUILDING_NAMES.get(building_type, "farm").to_lower(),
		})
		return

	var province: Dictionary = GameData.provinces.get(province_id, {})
	var faction: Dictionary = GameData.factions.get(GameData.local_faction_id, {})
	if province.is_empty() or faction.is_empty():
		return
	if province["owner_faction_id"] != GameData.local_faction_id:
		return

	var cost: int = GameData.BUILDING_COSTS.get(building_type, 100)
	if faction["gold"] < cost:
		EventBus.show_notification.emit("Not enough gold!", "error")
		return

	# Check if building exists -> upgrade
	var found := false
	for b in province["buildings"]:
		if b["type"] == building_type:
			b["level"] += 1
			faction["gold"] -= cost * b["level"]
			found = true
			break
	if not found:
		province["buildings"].append({"type": building_type, "level": 1})
		faction["gold"] -= cost

	GameData.add_event("building_built", "Built %s in %s" % [
		GameData.BUILDING_NAMES.get(building_type, "?"), province["name"]],
		GameData.local_faction_id, province_id)
	hex_map.queue_redraw()

func _on_recruit_army(province_id: String, soldiers: int) -> void:
	if NetworkManager.is_online_mode:
		NetworkManager.send_action({
			"type": "recruit_army",
			"province_id": province_id,
			"soldiers": soldiers,
		})
		return

	var province: Dictionary = GameData.provinces.get(province_id, {})
	var faction: Dictionary = GameData.factions.get(GameData.local_faction_id, {})
	if province.is_empty() or faction.is_empty():
		return

	soldiers = mini(soldiers, mini(province["manpower"], faction["manpower"]))
	if soldiers <= 0:
		return

	var cost := soldiers * 0.5
	if faction["gold"] < cost:
		EventBus.show_notification.emit("Not enough gold!", "error")
		return

	faction["gold"] -= cost
	faction["manpower"] -= soldiers
	province["manpower"] -= soldiers

	var army_id := GameData.generate_uid()
	var army := GameData.create_army(army_id, GameData.local_faction_id, province_id, soldiers)
	GameData.armies[army_id] = army

	GameData.add_event("army_recruited", "Recruited %d soldiers in %s" % [soldiers, province["name"]],
		GameData.local_faction_id, province_id)
	hex_map.queue_redraw()

func _on_move_army(army_id: String, target_province_id: String) -> void:
	if NetworkManager.is_online_mode:
		NetworkManager.send_action({
			"type": "move_army",
			"army_id": army_id,
			"target_province_id": target_province_id,
		})
		return

	var army: Dictionary = GameData.armies.get(army_id, {})
	if army.is_empty() or army["faction_id"] != GameData.local_faction_id:
		return

	var current_province: Dictionary = GameData.provinces.get(army["province_id"], {})
	if current_province.is_empty():
		return

	if target_province_id in current_province["adjacent_ids"]:
		army["status"] = GameData.ArmyStatus.MARCHING
		army["target_province_id"] = target_province_id
		army["march_progress"] = 0.0
		var target := GameData.provinces.get(target_province_id, {})
		GameData.add_event("army_march", "Army marching to %s" % target.get("name", "?"),
			GameData.local_faction_id)

func _on_research(tech_id: String) -> void:
	if NetworkManager.is_online_mode:
		NetworkManager.send_action({"type": "research", "tech_id": tech_id})
		return

	var faction: Dictionary = GameData.factions.get(GameData.local_faction_id, {})
	if faction.is_empty():
		return
	var tree := TechTree.build()
	var tech: Dictionary = tree.get(tech_id, {})
	if tech.is_empty():
		return
	for prereq in tech["prerequisites"]:
		if prereq not in faction["researched_techs"]:
			return
	if tech_id in faction["researched_techs"]:
		return
	faction["current_research"] = tech_id
	GameData.add_event("research_started", "Began researching %s" % tech["name"], GameData.local_faction_id)

func _on_diplomacy(target_faction_id: String, action: String) -> void:
	if NetworkManager.is_online_mode:
		NetworkManager.send_action({
			"type": "diplomacy",
			"target_faction_id": target_faction_id,
			"action": action,
		})
		return

	var faction: Dictionary = GameData.factions.get(GameData.local_faction_id, {})
	var target: Dictionary = GameData.factions.get(target_faction_id, {})
	if faction.is_empty() or target.is_empty():
		return

	var rel := GameData.get_relation(GameData.local_faction_id, target_faction_id)
	match action:
		"ally":
			if rel["status"] != GameData.DiplomacyStatus.WAR:
				rel["status"] = GameData.DiplomacyStatus.ALLY
				rel["opinion"] = mini(100, rel["opinion"] + 20)
				GameData.add_event("alliance", "%s allied with %s" % [faction["name"], target["name"]])
		"war":
			rel["status"] = GameData.DiplomacyStatus.WAR
			rel["opinion"] = maxi(-100, rel["opinion"] - 30)
			faction["stability"] = maxf(0, faction["stability"] - 10)
			GameData.add_event("war_declared", "%s declared war on %s!" % [faction["name"], target["name"]])
		"trade":
			if rel["status"] != GameData.DiplomacyStatus.WAR:
				rel["status"] = GameData.DiplomacyStatus.TRADE
				rel["trade_value"] = 10.0
				rel["opinion"] = mini(100, rel["opinion"] + 10)
				GameData.add_event("trade", "%s established trade with %s" % [faction["name"], target["name"]])
		"truce":
			if rel["status"] == GameData.DiplomacyStatus.WAR:
				rel["status"] = GameData.DiplomacyStatus.TRUCE
				rel["truce_turns"] = 10
				GameData.add_event("truce", "%s and %s signed a truce" % [faction["name"], target["name"]])

func _on_recruit_character(province_id: String) -> void:
	if NetworkManager.is_online_mode:
		NetworkManager.send_action({"type": "recruit_character", "province_id": province_id})
		return

	var faction: Dictionary = GameData.factions.get(GameData.local_faction_id, {})
	if faction.is_empty() or faction["gold"] < 100:
		return

	var province: Dictionary = GameData.provinces.get(province_id, {})
	if province.is_empty() or province["owner_faction_id"] != GameData.local_faction_id:
		return

	faction["gold"] -= 100
	var cid := GameData.generate_uid()
	var surnames := ["Liu", "Cao", "Sun", "Zhang", "Zhao", "Ma", "Huang", "Wei"]
	var givens := ["Bei", "Quan", "Liang", "Yu", "Fei", "Yun", "Chao", "Ping"]
	var cname := "%s %s" % [surnames[randi() % surnames.size()], givens[randi() % givens.size()]]
	var ch := GameData.create_character(cid, cname, GameData.CharacterRole.FREE, GameData.local_faction_id)
	ch["province_id"] = province_id
	GameData.characters[cid] = ch
	GameData.add_event("character_recruited", "Recruited %s in %s" % [cname, province["name"]],
		GameData.local_faction_id, province_id)

func _on_assign_general(character_id: String, army_id: String) -> void:
	if NetworkManager.is_online_mode:
		NetworkManager.send_action({
			"type": "assign_general",
			"character_id": character_id,
			"army_id": army_id,
		})
		return

	var ch: Dictionary = GameData.characters.get(character_id, {})
	var army: Dictionary = GameData.armies.get(army_id, {})
	if ch.is_empty() or army.is_empty():
		return
	if ch["faction_id"] != GameData.local_faction_id or army["faction_id"] != GameData.local_faction_id:
		return
	ch["role"] = GameData.CharacterRole.GENERAL
	army["general_id"] = character_id
