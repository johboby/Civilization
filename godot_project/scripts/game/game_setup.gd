## GameSetup - Initializes a new game with factions, map, and characters.
class_name GameSetup
extends RefCounted

const FACTION_COLORS := [
	Color(0.91, 0.30, 0.24),  # Red
	Color(0.20, 0.60, 0.86),  # Blue
	Color(0.18, 0.80, 0.44),  # Green
	Color(0.95, 0.61, 0.07),  # Orange
	Color(0.61, 0.35, 0.71),  # Purple
	Color(0.10, 0.74, 0.61),  # Teal
	Color(0.90, 0.49, 0.13),  # Dark Orange
	Color(0.20, 0.29, 0.37),  # Dark Blue
]

const AI_NAMES := [
	"Wei Empire", "Shu Kingdom", "Wu Dynasty", "Jin Alliance",
	"Yan State", "Chu Domain", "Qi Realm", "Qin Authority",
]

const RULER_NAMES := [
	"Cao Cao", "Liu Bei", "Sun Quan", "Sima Yi",
	"Yuan Shao", "Dong Zhuo", "Lu Bu", "Zhuge Liang",
]

const OFFICER_NAMES := [
	"Guan Yu", "Zhang Fei", "Zhao Yun", "Ma Chao", "Huang Zhong",
	"Xu Chu", "Dian Wei", "Zhang Liao", "Xiahou Dun", "Xiahou Yuan",
	"Zhou Yu", "Lu Su", "Lu Meng", "Gan Ning", "Taishi Ci",
	"Pang Tong", "Fa Zheng", "Jiang Wei", "Deng Ai", "Zhong Hui",
	"Xun Yu", "Guo Jia", "Jia Xu", "Cheng Yu", "Man Chong",
	"Wei Yan", "Wang Ping", "Zhang Bao", "Guan Xing", "Li Dian",
]

static func new_game(
	player_name: String,
	num_factions: int = 4,
	map_cols: int = 12,
	map_rows: int = 10,
	rng_seed: int = -1,
) -> void:
	GameData.reset_state()
	GameData.map_cols = map_cols
	GameData.map_rows = map_rows

	# Generate map
	GameData.provinces = MapGenerator.generate(map_cols, map_rows, rng_seed)

	# Build tech tree
	GameData.tech_tree = TechTree.build()

	# Create factions
	var faction_ids := []
	var officer_pool := OFFICER_NAMES.duplicate()
	officer_pool.shuffle()
	var officer_idx := 0

	for i in num_factions:
		var fid := GameData.generate_uid()
		var fname: String
		var player_id := ""

		if i == 0:
			fname = "%s's Kingdom" % player_name
			player_id = GameData.local_player_id
		else:
			fname = AI_NAMES[i % AI_NAMES.size()]

		var faction := GameData.create_faction(fid, fname, FACTION_COLORS[i % FACTION_COLORS.size()], player_id)

		# Create ruler
		var ruler_id := GameData.generate_uid()
		var ruler_name: String
		if i == 0:
			ruler_name = player_name
		else:
			ruler_name = RULER_NAMES[i % RULER_NAMES.size()]
		var ruler := GameData.create_character(ruler_id, ruler_name, GameData.CharacterRole.RULER, fid)
		ruler["command"] = randi_range(70, 95)
		ruler["intelligence"] = randi_range(70, 95)
		ruler["politics"] = randi_range(70, 95)
		ruler["charisma"] = randi_range(70, 95)
		ruler["loyalty"] = 100
		GameData.characters[ruler_id] = ruler
		faction["ruler_character_id"] = ruler_id

		# Create officers
		for _j in 3:
			if officer_idx >= officer_pool.size():
				officer_idx = 0
			var oid := GameData.generate_uid()
			var oname: String = officer_pool[officer_idx]
			officer_idx += 1
			var roles := [GameData.CharacterRole.GENERAL, GameData.CharacterRole.STRATEGIST, GameData.CharacterRole.GOVERNOR]
			var role: GameData.CharacterRole = roles[_j % roles.size()]
			var officer := GameData.create_character(oid, oname, role, fid)
			officer["command"] = randi_range(50, 85)
			officer["force"] = randi_range(50, 85)
			officer["intelligence"] = randi_range(50, 85)
			GameData.characters[oid] = officer

		GameData.factions[fid] = faction
		faction_ids.append(fid)

		if i == 0:
			GameData.local_faction_id = fid

	# Assign starting provinces
	var assignments := MapGenerator.assign_starting_provinces(GameData.provinces, faction_ids, 3)

	# Place characters and create starting armies
	for fid in assignments:
		var pids: Array = assignments[fid]
		if pids.is_empty():
			continue

		# Place characters at first province
		for ch in GameData.get_faction_characters(fid):
			ch["province_id"] = pids[0]

		# Create starting army
		var army_id := GameData.generate_uid()
		var army := GameData.create_army(army_id, fid, pids[0], 3000)

		# Assign a general
		var generals := GameData.get_faction_characters(fid).filter(
			func(c): return c["role"] == GameData.CharacterRole.GENERAL
		)
		if not generals.is_empty():
			army["general_id"] = generals[0]["id"]

		GameData.armies[army_id] = army

	# Initialize diplomacy
	for i in faction_ids.size():
		for j in range(i + 1, faction_ids.size()):
			var key := faction_ids[i] + "_" + faction_ids[j]
			GameData.diplomacy[key] = GameData.create_diplomacy_relation(faction_ids[i], faction_ids[j])

	GameData.game_active = true
	GameData.add_event("game_start", "A new era begins!")
	GameData.game_started.emit()
