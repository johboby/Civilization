## TurnProcessor - Processes all game logic for each turn.
##
## Handles economy, army movement, combat, research, population,
## random events, and character aging.
class_name TurnProcessor
extends RefCounted

const RANDOM_EVENTS := [
	{"type": "plague", "desc": "A plague struck {province}!", "pop_loss": 0.15, "unrest": 10},
	{"type": "bumper_harvest", "desc": "Bumper harvest in {province}!", "food_bonus": 50, "unrest": -5},
	{"type": "bandit_raid", "desc": "Bandits raiding {province}!", "gold_loss": 20, "unrest": 8},
	{"type": "earthquake", "desc": "Earthquake hit {province}!", "pop_loss": 0.05, "fort_damage": 1},
	{"type": "trade_boom", "desc": "Trade booming in {province}!", "gold_bonus": 30, "unrest": -3},
	{"type": "flood", "desc": "Flooding in {province}!", "food_loss": 30, "pop_loss": 0.03},
	{"type": "rebellion", "desc": "Rebellion in {province}!", "unrest": 25},
	{"type": "festival", "desc": "Festival in {province} boosts morale!", "unrest": -15, "prestige": 5},
	{"type": "talent", "desc": "A talent discovered in {province}!", "new_character": true},
	{"type": "good_omen", "desc": "Good omen in {province}!", "stability": 5, "unrest": -5},
]

const CHINESE_SURNAMES := ["Liu", "Cao", "Sun", "Zhuge", "Guan", "Zhang", "Zhao", "Ma",
	"Huang", "Wei", "Deng", "Jiang", "Lu", "Zhou", "Xu", "Sima"]
const CHINESE_GIVEN := ["Bei", "Cao", "Quan", "Liang", "Yu", "Fei", "Yun", "Chao",
	"Zhong", "Yan", "Ai", "Wei", "Su", "Xun", "Yi", "Ping"]

var random_event_chance: float = 0.12

# -----------------------------------------------------------------------
# Main turn processing
# -----------------------------------------------------------------------

func process_turn() -> Array:
	var events: Array = []

	GameData.turn += 1
	_advance_season()

	events.append_array(_process_economy())
	events.append_array(_process_army_movement())
	events.append_array(_process_combat())
	events.append_array(_process_research())
	events.append_array(_process_population())
	_process_diplomacy_decay()
	events.append_array(_process_random_events())
	events.append_array(_process_characters())
	events.append_array(_check_elimination())

	for evt in events:
		GameData.add_event(evt["type"], evt["description"],
			evt.get("faction_id", ""), evt.get("province_id", ""))

	return events

# -----------------------------------------------------------------------
# Season
# -----------------------------------------------------------------------

func _advance_season() -> void:
	var seasons := [GameData.Season.SPRING, GameData.Season.SUMMER,
		GameData.Season.AUTUMN, GameData.Season.WINTER]
	var idx := seasons.find(GameData.season)
	idx = (idx + 1) % 4
	GameData.season = seasons[idx]
	if GameData.season == GameData.Season.SPRING:
		GameData.year += 1

# -----------------------------------------------------------------------
# Economy
# -----------------------------------------------------------------------

func _process_economy() -> Array:
	var events := []
	for faction in GameData.factions.values():
		var total_food := 0.0
		var total_gold := 0.0
		var total_mp := 0

		for province in GameData.get_faction_provinces(faction["id"]):
			var income := GameData.compute_province_income(province)
			total_food += income["food"]
			total_gold += income["gold"]
			total_mp += income["manpower"]

			var gov := GameData.get_province_governor(province["id"], faction["id"])
			if not gov.is_empty():
				total_gold *= (1.0 + gov["politics"] * 0.003)
				total_food *= (1.0 + gov["politics"] * 0.002)

		total_gold *= (1.0 + faction["tax_rate"])

		# Trade income
		for rel in GameData.diplomacy.values():
			if rel["status"] == GameData.DiplomacyStatus.TRADE:
				if rel["faction_a_id"] == faction["id"] or rel["faction_b_id"] == faction["id"]:
					total_gold += rel["trade_value"]

		# Army upkeep
		for army in GameData.get_faction_armies(faction["id"]):
			total_gold -= army["soldiers"] * 0.01
			total_food -= army["soldiers"] * 0.02

		# Seasonal effects
		match GameData.season:
			GameData.Season.SPRING:
				total_food *= 1.2
			GameData.Season.WINTER:
				total_food *= 0.7
				total_mp = int(total_mp * 0.8)

		faction["gold"] += total_gold
		faction["food"] += total_food
		faction["manpower"] += total_mp

		if faction["food"] < 0:
			faction["food"] = 0
			faction["stability"] = maxf(0, faction["stability"] - 5)
			events.append(_evt("famine", "%s is running out of food!" % faction["name"], faction["id"]))

		faction["war_exhaustion"] = maxf(0, faction["war_exhaustion"] - 0.5)

	return events

# -----------------------------------------------------------------------
# Army Movement
# -----------------------------------------------------------------------

func _process_army_movement() -> Array:
	var events := []
	for army in GameData.armies.values():
		if army["status"] != GameData.ArmyStatus.MARCHING:
			continue
		if army["target_province_id"] == "":
			army["status"] = GameData.ArmyStatus.IDLE
			continue

		var target := GameData.provinces.get(army["target_province_id"], {})
		var speed := 0.5
		if not target.is_empty():
			match target["terrain"]:
				GameData.TerrainType.PLAINS: speed = 0.6
				GameData.TerrainType.MOUNTAINS: speed = 0.3
				GameData.TerrainType.MARSH: speed = 0.35
				GameData.TerrainType.FOREST: speed = 0.4

		army["march_progress"] += speed

		if army["march_progress"] >= 1.0:
			army["province_id"] = army["target_province_id"]
			army["target_province_id"] = ""
			army["march_progress"] = 0.0
			army["status"] = GameData.ArmyStatus.IDLE

			var faction := GameData.factions.get(army["faction_id"], {})
			var fname: String = faction.get("name", "Unknown")

			if not target.is_empty() and target["owner_faction_id"] != "" \
					and target["owner_faction_id"] != army["faction_id"]:
				var rel := GameData.get_relation(army["faction_id"], target["owner_faction_id"])
				if rel["status"] == GameData.DiplomacyStatus.WAR:
					army["status"] = GameData.ArmyStatus.BESIEGING
					events.append(_evt("siege_start",
						"%s begins siege of %s" % [fname, target["name"]],
						army["faction_id"], target["id"]))

			events.append(_evt("army_arrived",
				"%s's army arrived at %s" % [fname, target.get("name", "?")],
				army["faction_id"]))
	return events

# -----------------------------------------------------------------------
# Combat
# -----------------------------------------------------------------------

func _process_combat() -> Array:
	var events := []

	# Group armies by province
	var province_armies := {}
	for army in GameData.armies.values():
		var pid: String = army["province_id"]
		if not province_armies.has(pid):
			province_armies[pid] = []
		province_armies[pid].append(army)

	for pid in province_armies:
		var armies_here: Array = province_armies[pid]
		if armies_here.size() < 2:
			continue

		var by_faction := {}
		for a in armies_here:
			var fid: String = a["faction_id"]
			if not by_faction.has(fid):
				by_faction[fid] = []
			by_faction[fid].append(a)

		var fids := by_faction.keys()
		for i in fids.size():
			for j in range(i + 1, fids.size()):
				var rel := GameData.get_relation(fids[i], fids[j])
				if rel["status"] == GameData.DiplomacyStatus.WAR:
					events.append_array(_resolve_battle(by_faction[fids[i]], by_faction[fids[j]], pid))

	# Sieges
	for army in GameData.armies.values():
		if army["status"] == GameData.ArmyStatus.BESIEGING:
			events.append_array(_process_siege(army))

	return events

func _army_strength(army: Dictionary) -> float:
	var base: float = army["infantry"] * 1.0 + army["cavalry"] * 1.5 + army["archers"] * 1.2
	var morale_factor := army["morale"] / 100.0
	var gen_bonus := 1.0
	if army["general_id"] != "":
		var gen := GameData.characters.get(army["general_id"], {})
		if not gen.is_empty():
			gen_bonus += gen["command"] * 0.005 + gen["force"] * 0.002
	return base * morale_factor * gen_bonus

func _resolve_battle(armies_a: Array, armies_b: Array, province_id: String) -> Array:
	var events := []
	var province := GameData.provinces.get(province_id, {})
	var pname: String = province.get("name", "unknown")

	var str_a := 0.0
	for a in armies_a:
		str_a += _army_strength(a)
	var str_b := 0.0
	for a in armies_b:
		str_b += _army_strength(a)

	# Terrain defense bonus
	var def_bonus := 1.0
	if not province.is_empty():
		match province["terrain"]:
			GameData.TerrainType.MOUNTAINS: def_bonus = 1.3
			GameData.TerrainType.FOREST: def_bonus = 1.15
			GameData.TerrainType.RIVER: def_bonus = 1.1
		def_bonus += province.get("fortification", 0) * 0.1

	if not province.is_empty() and province["owner_faction_id"] != "":
		if armies_a[0]["faction_id"] == province["owner_faction_id"]:
			str_a *= def_bonus
		elif armies_b[0]["faction_id"] == province["owner_faction_id"]:
			str_b *= def_bonus

	str_a *= randf_range(0.85, 1.15)
	str_b *= randf_range(0.85, 1.15)

	var total := str_a + str_b
	if total == 0:
		return events

	var ratio_a := str_a / total
	var cas_rate_a := (1.0 - ratio_a) * randf_range(0.1, 0.3)
	var cas_rate_b := ratio_a * randf_range(0.1, 0.3)

	for a in armies_a:
		var losses := int(a["soldiers"] * cas_rate_a)
		a["soldiers"] -= losses
		a["infantry"] = maxi(0, a["infantry"] - int(losses * 0.6))
		a["cavalry"] = maxi(0, a["cavalry"] - int(losses * 0.25))
		a["archers"] = maxi(0, a["archers"] - int(losses * 0.15))
		a["morale"] = maxf(0, a["morale"] - (1.0 - ratio_a) * 20)

	for a in armies_b:
		var losses := int(a["soldiers"] * cas_rate_b)
		a["soldiers"] -= losses
		a["infantry"] = maxi(0, a["infantry"] - int(losses * 0.6))
		a["cavalry"] = maxi(0, a["cavalry"] - int(losses * 0.25))
		a["archers"] = maxi(0, a["archers"] - int(losses * 0.15))
		a["morale"] = maxf(0, a["morale"] - ratio_a * 20)

	var fa := GameData.factions.get(armies_a[0]["faction_id"], {})
	var fb := GameData.factions.get(armies_b[0]["faction_id"], {})
	var winner: String = fa.get("name", "?") if str_a > str_b else fb.get("name", "?")

	events.append(_evt("battle",
		"Battle at %s: %s vs %s. %s prevails!" % [pname, fa.get("name", "?"), fb.get("name", "?"), winner],
		"", province_id))

	# Remove destroyed armies
	var to_remove := []
	for a in armies_a + armies_b:
		if a["soldiers"] <= 0:
			to_remove.append(a["id"])
	for aid in to_remove:
		GameData.armies.erase(aid)

	if not fa.is_empty():
		fa["war_exhaustion"] = minf(100, fa["war_exhaustion"] + 3)
	if not fb.is_empty():
		fb["war_exhaustion"] = minf(100, fb["war_exhaustion"] + 3)

	return events

func _process_siege(army: Dictionary) -> Array:
	var events := []
	var province := GameData.provinces.get(army["province_id"], {})
	if province.is_empty() or province["owner_faction_id"] == "" \
			or province["owner_faction_id"] == army["faction_id"]:
		army["status"] = GameData.ArmyStatus.IDLE
		return events

	var siege_power: float = army["soldiers"] * 0.01
	var fort_str := maxf(1.0, province.get("fortification", 0) * 100.0)
	var progress := siege_power / fort_str

	if randf() < minf(0.8, progress):
		province["owner_faction_id"] = army["faction_id"]
		army["status"] = GameData.ArmyStatus.IDLE
		province["unrest"] += 20

		var faction := GameData.factions.get(army["faction_id"], {})
		var fname: String = faction.get("name", "Unknown")
		events.append(_evt("province_captured",
			"%s captured %s!" % [fname, province["name"]],
			army["faction_id"], province["id"]))
		if not faction.is_empty():
			faction["prestige"] += 5
	else:
		army["soldiers"] -= int(army["soldiers"] * randf_range(0.01, 0.03))
		army["morale"] = maxf(0, army["morale"] - 2)

	return events

# -----------------------------------------------------------------------
# Research
# -----------------------------------------------------------------------

func _process_research() -> Array:
	var events := []
	var tree := TechTree.build()

	for faction in GameData.factions.values():
		if faction["current_research"] == "":
			continue
		var tech: Dictionary = tree.get(faction["current_research"], {})
		if tech.is_empty():
			faction["current_research"] = ""
			continue

		var speed := 5.0
		for province in GameData.get_faction_provinces(faction["id"]):
			for b in province["buildings"]:
				if b["type"] == GameData.BuildingType.ACADEMY:
					speed += 3.0 * b["level"]

		for ch in GameData.get_faction_characters(faction["id"]):
			if ch["role"] == GameData.CharacterRole.STRATEGIST:
				speed += ch["intelligence"] * 0.1

		faction["research_points"] += speed

		if faction["research_points"] >= tech["cost"]:
			faction["researched_techs"].append(tech["id"])
			faction["research_points"] = 0
			faction["current_research"] = ""
			var cat_name: String = ["military", "economy", "culture", "diplomacy"][tech["category"]]
			if faction["tech_levels"].has(cat_name):
				faction["tech_levels"][cat_name] = maxi(faction["tech_levels"][cat_name], tech["level"])
			events.append(_evt("tech_researched",
				"%s researched %s!" % [faction["name"], tech["name"]], faction["id"]))

	return events

# -----------------------------------------------------------------------
# Population
# -----------------------------------------------------------------------

func _process_population() -> Array:
	for province in GameData.provinces.values():
		if province["owner_faction_id"] == "":
			continue
		var growth_rate := 0.01
		match GameData.season:
			GameData.Season.SPRING: growth_rate = 0.015
			GameData.Season.WINTER: growth_rate = 0.005
		growth_rate *= maxf(0.0, 1.0 - province["unrest"] * 0.01)
		province["population"] = clampi(int(province["population"] * (1 + growth_rate)), 100, 200000)
		province["manpower"] = mini(province["population"] / 5,
			province["manpower"] + int(province["population"] * 0.005))
		province["unrest"] = maxf(0, province["unrest"] - 1.5)
		if province["unrest"] < 10:
			province["development"] += 0.005
	return []

# -----------------------------------------------------------------------
# Diplomacy decay
# -----------------------------------------------------------------------

func _process_diplomacy_decay() -> void:
	for rel in GameData.diplomacy.values():
		if rel["status"] == GameData.DiplomacyStatus.TRUCE:
			rel["truce_turns"] -= 1
			if rel["truce_turns"] <= 0:
				rel["status"] = GameData.DiplomacyStatus.NEUTRAL
				rel["truce_turns"] = 0
		if rel["opinion"] > 0:
			rel["opinion"] = maxi(0, rel["opinion"] - 1)
		elif rel["opinion"] < 0:
			rel["opinion"] = mini(0, rel["opinion"] + 1)

# -----------------------------------------------------------------------
# Random Events
# -----------------------------------------------------------------------

func _process_random_events() -> Array:
	var events := []
	for province in GameData.provinces.values():
		if province["owner_faction_id"] == "" or randf() > random_event_chance:
			continue
		var evt_def: Dictionary = RANDOM_EVENTS[randi() % RANDOM_EVENTS.size()]
		var desc: String = evt_def["desc"].replace("{province}", province["name"])

		if evt_def.has("pop_loss"):
			province["population"] = maxi(100, province["population"] - int(province["population"] * evt_def["pop_loss"]))
		if evt_def.has("food_bonus"):
			province["food"] += evt_def["food_bonus"]
		if evt_def.has("food_loss"):
			province["food"] = maxf(0, province["food"] - evt_def["food_loss"])
		if evt_def.has("gold_bonus"):
			province["gold"] += evt_def["gold_bonus"]
		if evt_def.has("gold_loss"):
			province["gold"] = maxf(0, province["gold"] - evt_def["gold_loss"])
		if evt_def.has("unrest"):
			province["unrest"] = clampf(province["unrest"] + evt_def["unrest"], 0, 100)
		if evt_def.has("fort_damage"):
			province["fortification"] = maxi(0, province["fortification"] - evt_def["fort_damage"])
		if evt_def.get("new_character", false):
			var cid := GameData.generate_uid()
			var cname := "%s %s" % [CHINESE_SURNAMES[randi() % CHINESE_SURNAMES.size()],
				CHINESE_GIVEN[randi() % CHINESE_GIVEN.size()]]
			var ch := GameData.create_character(cid, cname, GameData.CharacterRole.FREE, province["owner_faction_id"])
			ch["province_id"] = province["id"]
			GameData.characters[cid] = ch
		if evt_def.has("stability"):
			var faction := GameData.factions.get(province["owner_faction_id"], {})
			if not faction.is_empty():
				faction["stability"] = clampf(faction["stability"] + evt_def["stability"], 0, 100)
		if evt_def.has("prestige"):
			var faction := GameData.factions.get(province["owner_faction_id"], {})
			if not faction.is_empty():
				faction["prestige"] += evt_def["prestige"]

		events.append(_evt(evt_def["type"], desc, province["owner_faction_id"], province["id"]))
	return events

# -----------------------------------------------------------------------
# Characters
# -----------------------------------------------------------------------

func _process_characters() -> Array:
	var events := []
	for ch in GameData.characters.values():
		if not ch["alive"]:
			continue
		if GameData.turn % 4 == 0:
			ch["age"] += 1
			if ch["age"] > 60:
				var death_chance := (ch["age"] - 60) * 0.02
				if randf() < death_chance:
					ch["alive"] = false
					var faction := GameData.factions.get(ch["faction_id"], {})
					var fname: String = faction.get("name", "Unknown")
					events.append(_evt("character_death",
						"%s of %s passed away at age %d" % [ch["name"], fname, ch["age"]], ch["faction_id"]))
					if not faction.is_empty() and faction["ruler_character_id"] == ch["id"]:
						_succession(faction)

		if ch["role"] != GameData.CharacterRole.RULER:
			ch["loyalty"] = clampi(ch["loyalty"] - randi_range(0, 1), 0, 100)
			if ch["loyalty"] < 10 and randf() < 0.1:
				ch["faction_id"] = ""
				ch["role"] = GameData.CharacterRole.FREE
				events.append(_evt("defection", "%s left their faction!" % ch["name"]))
	return events

func _succession(faction: Dictionary) -> void:
	var candidates := GameData.get_faction_characters(faction["id"])
	candidates = candidates.filter(func(c): return c["role"] != GameData.CharacterRole.RULER)
	if candidates.is_empty():
		faction["ruler_character_id"] = ""
		faction["stability"] = 0
		return
	candidates.sort_custom(func(a, b):
		var pa: int = a["command"] + a["force"] + a["intelligence"] + a["politics"] + a["charisma"]
		var pb: int = b["command"] + b["force"] + b["intelligence"] + b["politics"] + b["charisma"]
		return pa > pb
	)
	candidates[0]["role"] = GameData.CharacterRole.RULER
	faction["ruler_character_id"] = candidates[0]["id"]
	faction["stability"] = maxf(0, faction["stability"] - 15)

# -----------------------------------------------------------------------
# Elimination
# -----------------------------------------------------------------------

func _check_elimination() -> Array:
	var events := []
	for faction in GameData.factions.values():
		if GameData.get_faction_provinces(faction["id"]).is_empty():
			events.append(_evt("faction_eliminated",
				"%s has been eliminated!" % faction["name"], faction["id"]))
			faction["stability"] = 0
	return events

# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------

func _evt(etype: String, desc: String, fid: String = "", pid: String = "") -> Dictionary:
	return {"turn": GameData.turn, "type": etype, "description": desc, "faction_id": fid, "province_id": pid}
