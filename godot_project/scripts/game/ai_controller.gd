## AIController - Autonomous AI faction decision-making.
##
## Each AI faction evaluates situation and submits actions each turn.
## Ported from the Python ai_controller.py with personality system.
class_name AIController
extends RefCounted

# Personality traits (0.0 to 1.0)
var personalities: Dictionary = {}  # faction_id -> {aggression, expansion, diplomacy, economy, research}

func initialize(faction_ids: Array) -> void:
	for fid in faction_ids:
		var faction: Dictionary = GameData.factions.get(fid, {})
		if faction.get("player_id", "") != "":
			continue  # Skip human players
		personalities[fid] = _random_personality()

func process_ai_turns(engine_ref) -> void:
	"""Generate and submit actions for all AI factions."""
	for fid in personalities:
		var faction: Dictionary = GameData.factions.get(fid, {})
		if faction.is_empty():
			continue
		var p: Dictionary = personalities[fid]
		var actions := _decide_actions(fid, faction, p)
		for action in actions:
			# Submit directly to pending actions
			if not GameData.pending_actions.has(fid):
				GameData.pending_actions[fid] = []
			GameData.pending_actions[fid].append(action)

func _random_personality() -> Dictionary:
	return {
		"aggression": randf_range(0.2, 0.8),
		"expansion": randf_range(0.2, 0.8),
		"diplomacy": randf_range(0.2, 0.8),
		"economy": randf_range(0.3, 0.9),
		"research": randf_range(0.3, 0.8),
	}

func _decide_actions(fid: String, faction: Dictionary, p: Dictionary) -> Array:
	var actions: Array = []
	var provinces := GameData.get_faction_provinces(fid)
	var armies := GameData.get_faction_armies(fid)
	var characters := GameData.get_faction_characters(fid)

	if provinces.is_empty():
		return actions

	# Economy
	actions.append_array(_economy_actions(fid, faction, provinces, p))
	# Military
	actions.append_array(_military_actions(fid, faction, provinces, armies, p))
	# Diplomacy
	actions.append_array(_diplomacy_actions(fid, faction, p))
	# Research
	actions.append_array(_research_actions(fid, faction, p))
	# Characters
	if characters.size() < 4 and faction["gold"] > 150 and not provinces.is_empty():
		actions.append({"type": "recruit_character", "province_id": provinces[0]["id"]})

	return actions

func _economy_actions(fid: String, faction: Dictionary, provinces: Array, p: Dictionary) -> Array:
	var actions: Array = []
	var budget: float = faction["gold"] * 0.4

	provinces.sort_custom(func(a, b): return a["development"] > b["development"])
	for i in mini(3, provinces.size()):
		if budget < 50:
			break
		var province: Dictionary = provinces[i]
		var btype: String
		var cost: int

		if faction["food"] < 200 or (p["economy"] > 0.5 and randf() < 0.4):
			btype = "farm"; cost = 50
		elif faction["gold"] < 300 and randf() < 0.3:
			btype = "market"; cost = 60
		elif p["aggression"] > 0.5 and faction["manpower"] < 3000:
			btype = "barracks"; cost = 80
		elif p["research"] > 0.5 and randf() < 0.3:
			btype = "academy"; cost = 120
		else:
			btype = ["farm", "market", "workshop"][randi() % 3]; cost = 60

		if faction["gold"] >= cost:
			actions.append({"type": "build", "province_id": province["id"], "building_type": btype})
			budget -= cost
	return actions

func _military_actions(fid: String, faction: Dictionary, provinces: Array, armies: Array, p: Dictionary) -> Array:
	var actions: Array = []
	var total_soldiers := 0
	for a in armies:
		total_soldiers += a["soldiers"]

	var at_war := _is_at_war(fid)

	# Recruit
	if (total_soldiers < 2000 or (at_war and total_soldiers < 5000)) and faction["gold"] > 300:
		var best_prov: Dictionary = provinces[0]
		for prov in provinces:
			if prov["manpower"] > best_prov["manpower"]:
				best_prov = prov
		var recruit := mini(1500, mini(best_prov["manpower"], faction["manpower"]))
		if recruit > 300:
			actions.append({"type": "recruit_army", "province_id": best_prov["id"], "soldiers": recruit})

	# Move idle armies
	for army in armies:
		if army["status"] != GameData.ArmyStatus.IDLE:
			continue
		var current_prov: Dictionary = GameData.provinces.get(army["province_id"], {})
		if current_prov.is_empty():
			continue

		var target := _choose_target(fid, army, current_prov, p)
		if target != "":
			actions.append({"type": "move_army", "army_id": army["id"], "target_province_id": target})
	return actions

func _choose_target(fid: String, army: Dictionary, current_prov: Dictionary, p: Dictionary) -> String:
	var best_id := ""
	var best_score := 0.2

	for adj_id in current_prov.get("adjacent_ids", []):
		var adj: Dictionary = GameData.provinces.get(adj_id, {})
		if adj.is_empty():
			continue
		var score := 0.0
		if adj["owner_faction_id"] == fid:
			score = 0.1
		elif adj["owner_faction_id"] == "":
			score = 0.5 * p["expansion"]
		else:
			var rel := GameData.get_relation(fid, adj["owner_faction_id"])
			if rel["status"] == GameData.DiplomacyStatus.WAR:
				score = 1.0 * p["aggression"]
				var enemy_armies := GameData.get_armies_at_province(adj_id)
				if enemy_armies.is_empty():
					score += 0.3

		if score > best_score:
			best_score = score
			best_id = adj_id
	return best_id

func _diplomacy_actions(fid: String, faction: Dictionary, p: Dictionary) -> Array:
	var actions: Array = []
	for other_id in GameData.factions:
		if other_id == fid:
			continue
		var rel := GameData.get_relation(fid, other_id)

		if rel["status"] == GameData.DiplomacyStatus.NEUTRAL:
			if p["aggression"] > 0.6 and randf() < p["aggression"] * 0.1:
				var our_soldiers := 0
				for a in GameData.get_faction_armies(fid):
					our_soldiers += a["soldiers"]
				var their_soldiers := 0
				for a in GameData.get_faction_armies(other_id):
					their_soldiers += a["soldiers"]
				if our_soldiers > their_soldiers * 1.5:
					actions.append({"type": "diplomacy", "target_faction_id": other_id, "action": "war"})
					continue

			if p["diplomacy"] > 0.5 and rel["opinion"] > 10 and randf() < 0.08:
				actions.append({"type": "diplomacy", "target_faction_id": other_id, "action": "ally"})
			elif p["economy"] > 0.5 and randf() < 0.08:
				actions.append({"type": "diplomacy", "target_faction_id": other_id, "action": "trade"})

		elif rel["status"] == GameData.DiplomacyStatus.WAR:
			if faction["war_exhaustion"] > 50 and randf() < 0.3:
				actions.append({"type": "diplomacy", "target_faction_id": other_id, "action": "truce"})
	return actions

func _research_actions(fid: String, faction: Dictionary, p: Dictionary) -> Array:
	if faction["current_research"] != "":
		return []
	var available := TechTree.get_available(faction["researched_techs"])
	if available.is_empty():
		return []

	var best_tech: Dictionary = available[0]
	var best_score := -1.0
	for tech in available:
		var score := 0.0
		match tech["category"]:
			GameData.TechCategory.MILITARY: score = p["aggression"] * 0.8
			GameData.TechCategory.ECONOMY: score = p["economy"] * 0.8 + 0.2
			GameData.TechCategory.CULTURE: score = p["research"] * 0.6 + 0.3
			GameData.TechCategory.DIPLOMACY: score = p["diplomacy"] * 0.7
		score += 50.0 / maxf(1, tech["cost"])
		if score > best_score:
			best_score = score
			best_tech = tech
	return [{"type": "research", "tech_id": best_tech["id"]}]

func _is_at_war(fid: String) -> bool:
	for rel in GameData.diplomacy.values():
		if rel["status"] == GameData.DiplomacyStatus.WAR:
			if rel["faction_a_id"] == fid or rel["faction_b_id"] == fid:
				return true
	return false
