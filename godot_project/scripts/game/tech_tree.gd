## TechTree - Technology tree definitions and helpers.
class_name TechTree
extends RefCounted

static func build() -> Dictionary:
	var tree := {}

	# Military branch
	_add(tree, "mil_1", "Bronze Weapons", GameData.TechCategory.MILITARY, 1, 100,
		"Basic bronze weaponry.", [], {"army_attack": 0.1})
	_add(tree, "mil_2", "Iron Forging", GameData.TechCategory.MILITARY, 2, 200,
		"Iron weapons and armor.", ["mil_1"], {"army_attack": 0.15, "army_defense": 0.05})
	_add(tree, "mil_3", "Cavalry Tactics", GameData.TechCategory.MILITARY, 2, 250,
		"Organized cavalry.", ["mil_1"], {"cavalry_power": 0.2})
	_add(tree, "mil_4", "Siege Engineering", GameData.TechCategory.MILITARY, 3, 400,
		"Siege towers and rams.", ["mil_2"], {"siege_power": 0.3})
	_add(tree, "mil_5", "Steel Weapons", GameData.TechCategory.MILITARY, 3, 500,
		"High-quality steel.", ["mil_2"], {"army_attack": 0.2, "army_defense": 0.1})
	_add(tree, "mil_6", "Advanced Formations", GameData.TechCategory.MILITARY, 4, 700,
		"Complex military formations.", ["mil_3", "mil_5"], {"army_attack": 0.15, "army_defense": 0.15})
	_add(tree, "mil_7", "War Machines", GameData.TechCategory.MILITARY, 5, 1000,
		"Trebuchets and advanced siege.", ["mil_4", "mil_6"], {"siege_power": 0.5, "army_attack": 0.1})

	# Economy branch
	_add(tree, "eco_1", "Agriculture", GameData.TechCategory.ECONOMY, 1, 100,
		"Organized farming.", [], {"food_production": 0.15})
	_add(tree, "eco_2", "Irrigation", GameData.TechCategory.ECONOMY, 2, 200,
		"Canal and irrigation.", ["eco_1"], {"food_production": 0.2})
	_add(tree, "eco_3", "Currency", GameData.TechCategory.ECONOMY, 2, 200,
		"Standardized currency.", ["eco_1"], {"gold_production": 0.15, "trade_efficiency": 0.1})
	_add(tree, "eco_4", "Road Network", GameData.TechCategory.ECONOMY, 3, 350,
		"Built roads.", ["eco_3"], {"trade_efficiency": 0.2, "march_speed": 0.1})
	_add(tree, "eco_5", "Banking", GameData.TechCategory.ECONOMY, 3, 400,
		"Financial institutions.", ["eco_3"], {"gold_production": 0.2, "tax_efficiency": 0.15})
	_add(tree, "eco_6", "Advanced Agriculture", GameData.TechCategory.ECONOMY, 4, 600,
		"Crop rotation.", ["eco_2", "eco_4"], {"food_production": 0.25, "population_growth": 0.1})
	_add(tree, "eco_7", "Manufacturing", GameData.TechCategory.ECONOMY, 5, 900,
		"Proto-industrial manufacturing.", ["eco_5", "eco_6"], {"gold_production": 0.3, "development_speed": 0.2})

	# Culture branch
	_add(tree, "cul_1", "Writing", GameData.TechCategory.CULTURE, 1, 100,
		"Written language.", [], {"research_speed": 0.1})
	_add(tree, "cul_2", "Philosophy", GameData.TechCategory.CULTURE, 2, 200,
		"Schools of thought.", ["cul_1"], {"stability_bonus": 0.1, "research_speed": 0.1})
	_add(tree, "cul_3", "Education", GameData.TechCategory.CULTURE, 2, 250,
		"Formal education.", ["cul_1"], {"research_speed": 0.15, "character_xp": 0.1})
	_add(tree, "cul_4", "Literature", GameData.TechCategory.CULTURE, 3, 350,
		"Great works.", ["cul_2"], {"prestige_gain": 0.2, "stability_bonus": 0.05})
	_add(tree, "cul_5", "Academy", GameData.TechCategory.CULTURE, 4, 500,
		"Higher learning.", ["cul_3", "cul_4"], {"research_speed": 0.25, "character_xp": 0.15})
	_add(tree, "cul_6", "Renaissance", GameData.TechCategory.CULTURE, 5, 800,
		"Cultural rebirth.", ["cul_5"], {"research_speed": 0.2, "prestige_gain": 0.3})

	# Diplomacy branch
	_add(tree, "dip_1", "Envoys", GameData.TechCategory.DIPLOMACY, 1, 100,
		"Basic diplomacy.", [], {"diplomacy_power": 0.1})
	_add(tree, "dip_2", "Trade Agreements", GameData.TechCategory.DIPLOMACY, 2, 200,
		"Formal trade.", ["dip_1"], {"trade_efficiency": 0.15, "diplomacy_power": 0.05})
	_add(tree, "dip_3", "Alliances", GameData.TechCategory.DIPLOMACY, 2, 250,
		"Military alliances.", ["dip_1"], {"diplomacy_power": 0.15, "ally_trust": 0.1})
	_add(tree, "dip_4", "Espionage", GameData.TechCategory.DIPLOMACY, 3, 400,
		"Intelligence networks.", ["dip_2"], {"spy_efficiency": 0.2})
	_add(tree, "dip_5", "Vassalage", GameData.TechCategory.DIPLOMACY, 3, 400,
		"Vassal-suzerain relations.", ["dip_3"], {"diplomacy_power": 0.2, "vassal_income": 0.15})
	_add(tree, "dip_6", "Imperial Authority", GameData.TechCategory.DIPLOMACY, 5, 900,
		"Supreme authority.", ["dip_4", "dip_5"], {"diplomacy_power": 0.3, "prestige_gain": 0.2})

	return tree

static func get_available(researched: Array, category := -1) -> Array:
	var tree := build()
	var result := []
	for tech in tree.values():
		if tech["id"] in researched:
			continue
		if category >= 0 and tech["category"] != category:
			continue
		var all_prereqs := true
		for prereq in tech["prerequisites"]:
			if prereq not in researched:
				all_prereqs = false
				break
		if all_prereqs:
			result.append(tech)
	return result

static func _add(tree: Dictionary, id: String, tname: String, category: GameData.TechCategory,
		level: int, cost: int, desc: String, prereqs: Array, effects: Dictionary) -> void:
	tree[id] = {
		"id": id,
		"name": tname,
		"category": category,
		"level": level,
		"cost": cost,
		"description": desc,
		"prerequisites": prereqs,
		"effects": effects,
	}
