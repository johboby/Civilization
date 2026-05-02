## GameData - Global game state singleton.
##
## Holds all game entities (provinces, factions, characters, armies, diplomacy)
## and provides helper methods for querying state.
extends Node

# -----------------------------------------------------------------------
# Signals
# -----------------------------------------------------------------------
signal game_started
signal turn_processed(turn: int, events: Array)
signal faction_changed(faction_id: String)
signal province_selected(province_id: String)
signal army_selected(army_id: String)

# -----------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------
enum TerrainType { PLAINS, MOUNTAINS, FOREST, DESERT, RIVER, COAST, MARSH }
enum CharacterRole { RULER, GENERAL, STRATEGIST, GOVERNOR, DIPLOMAT, SPY, FREE }
enum DiplomacyStatus { NEUTRAL, ALLY, WAR, VASSAL, TRUCE, TRADE }
enum ArmyStatus { IDLE, MARCHING, BESIEGING, DEFENDING, RETREATING }
enum Season { SPRING, SUMMER, AUTUMN, WINTER }
enum BuildingType { FARM, MARKET, BARRACKS, WALL, ACADEMY, TEMPLE, WORKSHOP, PORT }
enum TechCategory { MILITARY, ECONOMY, CULTURE, DIPLOMACY }

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
const TERRAIN_NAMES := {
	TerrainType.PLAINS: "Plains",
	TerrainType.MOUNTAINS: "Mountains",
	TerrainType.FOREST: "Forest",
	TerrainType.DESERT: "Desert",
	TerrainType.RIVER: "River",
	TerrainType.COAST: "Coast",
	TerrainType.MARSH: "Marsh",
}

const TERRAIN_COLORS := {
	TerrainType.PLAINS: Color(0.29, 0.49, 0.25),
	TerrainType.MOUNTAINS: Color(0.55, 0.45, 0.33),
	TerrainType.FOREST: Color(0.18, 0.35, 0.12),
	TerrainType.DESERT: Color(0.79, 0.66, 0.43),
	TerrainType.RIVER: Color(0.23, 0.49, 0.65),
	TerrainType.COAST: Color(0.36, 0.61, 0.84),
	TerrainType.MARSH: Color(0.29, 0.40, 0.25),
}

const TERRAIN_MODIFIERS := {
	TerrainType.PLAINS: {"food": 1.3, "gold": 1.0, "manpower": 1.2},
	TerrainType.MOUNTAINS: {"food": 0.5, "gold": 1.3, "manpower": 0.7},
	TerrainType.FOREST: {"food": 0.9, "gold": 0.8, "manpower": 0.9},
	TerrainType.DESERT: {"food": 0.3, "gold": 0.6, "manpower": 0.5},
	TerrainType.RIVER: {"food": 1.5, "gold": 1.2, "manpower": 1.1},
	TerrainType.COAST: {"food": 1.1, "gold": 1.4, "manpower": 1.0},
	TerrainType.MARSH: {"food": 0.7, "gold": 0.5, "manpower": 0.6},
}

const SEASON_NAMES := {
	Season.SPRING: "Spring",
	Season.SUMMER: "Summer",
	Season.AUTUMN: "Autumn",
	Season.WINTER: "Winter",
}

const BUILDING_NAMES := {
	BuildingType.FARM: "Farm",
	BuildingType.MARKET: "Market",
	BuildingType.BARRACKS: "Barracks",
	BuildingType.WALL: "Wall",
	BuildingType.ACADEMY: "Academy",
	BuildingType.TEMPLE: "Temple",
	BuildingType.WORKSHOP: "Workshop",
	BuildingType.PORT: "Port",
}

const BUILDING_COSTS := {
	BuildingType.FARM: 50,
	BuildingType.MARKET: 60,
	BuildingType.BARRACKS: 80,
	BuildingType.WALL: 100,
	BuildingType.ACADEMY: 120,
	BuildingType.TEMPLE: 70,
	BuildingType.WORKSHOP: 90,
	BuildingType.PORT: 110,
}

# -----------------------------------------------------------------------
# Game State
# -----------------------------------------------------------------------
var turn: int = 0
var year: int = 190
var season: Season = Season.SPRING
var game_active: bool = false

var map_cols: int = 12
var map_rows: int = 10

# Entity dictionaries
var provinces: Dictionary = {}   # id -> ProvinceData
var factions: Dictionary = {}    # id -> FactionData
var characters: Dictionary = {}  # id -> CharacterData
var armies: Dictionary = {}      # id -> ArmyData
var diplomacy: Dictionary = {}   # "a_b" -> DiplomacyData
var tech_tree: Dictionary = {}   # id -> TechData

# Player info
var local_player_id: String = ""
var local_faction_id: String = ""
var selected_province_id: String = ""
var selected_army_id: String = ""

# Event log
var event_log: Array = []
var max_events: int = 200

# -----------------------------------------------------------------------
# Data Classes (as inner dictionaries)
# -----------------------------------------------------------------------

func create_province(id: String, pname: String, x: int, y: int, terrain: TerrainType) -> Dictionary:
	var mods = TERRAIN_MODIFIERS[terrain]
	return {
		"id": id,
		"name": pname,
		"x": x,
		"y": y,
		"terrain": terrain,
		"owner_faction_id": "",
		"population": randi_range(5000, 20000),
		"food": randf_range(50.0, 200.0) * mods["food"],
		"gold": randf_range(20.0, 100.0) * mods["gold"],
		"manpower": int(randf_range(500, 2000) * mods["manpower"]),
		"development": randf_range(0.5, 2.0),
		"fortification": randi_range(0, 2),
		"buildings": [],  # Array of {"type": BuildingType, "level": int}
		"adjacent_ids": [],
		"unrest": 0.0,
		"culture": "default",
		"food_modifier": mods["food"],
		"gold_modifier": mods["gold"],
		"manpower_modifier": mods["manpower"],
	}

func create_faction(id: String, fname: String, color: Color, player_id: String = "") -> Dictionary:
	return {
		"id": id,
		"name": fname,
		"color": color,
		"player_id": player_id,
		"ruler_character_id": "",
		"gold": 500.0,
		"food": 500.0,
		"manpower": 5000,
		"tech_levels": {"military": 1, "economy": 1, "culture": 1, "diplomacy": 1},
		"research_points": 0.0,
		"current_research": "",
		"researched_techs": [],
		"tax_rate": 0.1,
		"conscription_rate": 0.05,
		"war_exhaustion": 0.0,
		"stability": 50.0,
		"legitimacy": 50.0,
		"prestige": 0.0,
	}

func create_character(id: String, cname: String, role: CharacterRole, faction_id: String = "") -> Dictionary:
	return {
		"id": id,
		"name": cname,
		"role": role,
		"faction_id": faction_id,
		"province_id": "",
		"command": randi_range(30, 90),
		"force": randi_range(30, 90),
		"intelligence": randi_range(30, 90),
		"politics": randi_range(30, 90),
		"charisma": randi_range(30, 90),
		"loyalty": randi_range(60, 100),
		"level": 1,
		"experience": 0,
		"skills": [],
		"alive": true,
		"age": randi_range(20, 45),
	}

func create_army(id: String, faction_id: String, province_id: String, soldiers: int) -> Dictionary:
	var infantry := int(soldiers * 0.6)
	var cavalry := int(soldiers * 0.25)
	var archers := soldiers - infantry - cavalry
	return {
		"id": id,
		"faction_id": faction_id,
		"general_id": "",
		"province_id": province_id,
		"soldiers": soldiers,
		"morale": 100.0,
		"status": ArmyStatus.IDLE,
		"target_province_id": "",
		"march_progress": 0.0,
		"infantry": infantry,
		"cavalry": cavalry,
		"archers": archers,
	}

func create_diplomacy_relation(faction_a: String, faction_b: String) -> Dictionary:
	return {
		"faction_a_id": faction_a,
		"faction_b_id": faction_b,
		"status": DiplomacyStatus.NEUTRAL,
		"opinion": 0,
		"trade_value": 0.0,
		"truce_turns": 0,
	}

# -----------------------------------------------------------------------
# Queries
# -----------------------------------------------------------------------

func get_faction_provinces(faction_id: String) -> Array:
	var result := []
	for p in provinces.values():
		if p["owner_faction_id"] == faction_id:
			result.append(p)
	return result

func get_faction_characters(faction_id: String) -> Array:
	var result := []
	for c in characters.values():
		if c["faction_id"] == faction_id and c["alive"]:
			result.append(c)
	return result

func get_faction_armies(faction_id: String) -> Array:
	var result := []
	for a in armies.values():
		if a["faction_id"] == faction_id:
			result.append(a)
	return result

func get_armies_at_province(province_id: String) -> Array:
	var result := []
	for a in armies.values():
		if a["province_id"] == province_id:
			result.append(a)
	return result

func get_relation(faction_a: String, faction_b: String) -> Dictionary:
	var key1 := faction_a + "_" + faction_b
	var key2 := faction_b + "_" + faction_a
	if diplomacy.has(key1):
		return diplomacy[key1]
	if diplomacy.has(key2):
		return diplomacy[key2]
	# Create new
	var rel := create_diplomacy_relation(faction_a, faction_b)
	diplomacy[key1] = rel
	return rel

func get_province_governor(province_id: String, faction_id: String) -> Dictionary:
	for c in characters.values():
		if c["faction_id"] == faction_id and c["province_id"] == province_id \
				and c["role"] == CharacterRole.GOVERNOR and c["alive"]:
			return c
	return {}

func compute_province_income(province: Dictionary) -> Dictionary:
	var base_food := 10.0 * province["development"] * province["food_modifier"]
	var base_gold := 5.0 * province["development"] * province["gold_modifier"]
	var base_manpower := int(50 * province["development"] * province["manpower_modifier"])

	for b in province["buildings"]:
		match b["type"]:
			BuildingType.FARM:
				base_food += 5.0 * b["level"]
			BuildingType.MARKET:
				base_gold += 3.0 * b["level"]
			BuildingType.BARRACKS:
				base_manpower += 30 * b["level"]
			BuildingType.WORKSHOP:
				base_gold += 2.0 * b["level"]
				base_food += 2.0 * b["level"]

	var penalty := maxf(0.0, 1.0 - province["unrest"] * 0.01)
	return {
		"food": base_food * penalty,
		"gold": base_gold * penalty,
		"manpower": int(base_manpower * penalty),
	}

# -----------------------------------------------------------------------
# Unique ID generation
# -----------------------------------------------------------------------
var _uid_counter: int = 0

func generate_uid() -> String:
	_uid_counter += 1
	return "%s_%d" % [str(Time.get_ticks_msec()), _uid_counter]

# -----------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------

func save_game(filepath: String) -> Error:
	var data := {
		"turn": turn,
		"year": year,
		"season": season,
		"map_cols": map_cols,
		"map_rows": map_rows,
		"provinces": provinces,
		"factions": factions,
		"characters": characters,
		"armies": armies,
		"diplomacy": diplomacy,
	}
	var json_string := JSON.stringify(data, "\t")
	var file := FileAccess.open(filepath, FileAccess.WRITE)
	if file == null:
		return FileAccess.get_open_error()
	file.store_string(json_string)
	file.close()
	return OK

func load_game(filepath: String) -> Error:
	if not FileAccess.file_exists(filepath):
		return ERR_FILE_NOT_FOUND
	var file := FileAccess.open(filepath, FileAccess.READ)
	if file == null:
		return FileAccess.get_open_error()
	var json_string := file.get_as_text()
	file.close()
	var json := JSON.new()
	var error := json.parse(json_string)
	if error != OK:
		return error
	var data: Dictionary = json.data
	turn = data.get("turn", 0)
	year = data.get("year", 190)
	season = data.get("season", Season.SPRING)
	map_cols = data.get("map_cols", 12)
	map_rows = data.get("map_rows", 10)
	provinces = data.get("provinces", {})
	factions = data.get("factions", {})
	characters = data.get("characters", {})
	armies = data.get("armies", {})
	diplomacy = data.get("diplomacy", {})
	game_active = true
	return OK

# -----------------------------------------------------------------------
# Reset
# -----------------------------------------------------------------------

func reset_state() -> void:
	turn = 0
	year = 190
	season = Season.SPRING
	game_active = false
	provinces.clear()
	factions.clear()
	characters.clear()
	armies.clear()
	diplomacy.clear()
	tech_tree.clear()
	event_log.clear()
	local_faction_id = ""
	selected_province_id = ""
	selected_army_id = ""
	_uid_counter = 0

func add_event(event_type: String, description: String, faction_id: String = "", province_id: String = "") -> void:
	var evt := {
		"turn": turn,
		"type": event_type,
		"description": description,
		"faction_id": faction_id,
		"province_id": province_id,
	}
	event_log.append(evt)
	if event_log.size() > max_events:
		event_log = event_log.slice(event_log.size() - max_events)
