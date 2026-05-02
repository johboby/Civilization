## MapGenerator - Generates hex-grid province maps.
class_name MapGenerator
extends RefCounted

const CHINESE_NAMES := [
	"Luoyang", "Chang'an", "Jianye", "Chengdu", "Xiangyang",
	"Xuchang", "Ye", "Shouchun", "Jiangxia", "Hanzhong",
	"Nanyang", "Wancheng", "Beihai", "Pingyuan", "Xiapi",
	"Hefei", "Runan", "Jingzhou", "Yizhou", "Liangzhou",
	"Youzhou", "Jizhou", "Yanzhou", "Xuzhou", "Qingzhou",
	"Bingzhou", "Yongzhou", "Yuzhou", "Sizhou", "Jingzhao",
	"Wuling", "Changsha", "Guiyang", "Lingling", "Jiangdong",
	"Kuaiji", "Danyang", "Lujiang", "Poyang", "Yuzhang",
	"Tianshui", "Anding", "Fufeng", "Hongnong", "Hedong",
	"Taiyuan", "Shangdang", "Henei", "Chenliu", "Yingchuan",
	"Donghai", "Langya", "Taishan", "Jibei", "Dongping",
	"Shanyang", "Pei", "Guangling", "Jiujiang", "Linhuai",
]

const TERRAIN_WEIGHTS := {
	GameData.TerrainType.PLAINS: 35,
	GameData.TerrainType.FOREST: 20,
	GameData.TerrainType.MOUNTAINS: 15,
	GameData.TerrainType.RIVER: 10,
	GameData.TerrainType.COAST: 10,
	GameData.TerrainType.DESERT: 5,
	GameData.TerrainType.MARSH: 5,
}

static func generate(cols: int, rows: int, rng_seed: int = -1) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	if rng_seed >= 0:
		rng.seed = rng_seed
	else:
		rng.randomize()

	var names := CHINESE_NAMES.duplicate()
	names.shuffle()

	var provinces := {}
	var coord_to_id := {}
	var total := cols * rows

	for idx in total:
		var x := idx % cols
		var y := idx / cols

		var terrain := _weighted_terrain(rng)
		var pname: String
		if idx < names.size():
			pname = names[idx]
		else:
			pname = "%s %d" % [names[idx % names.size()], idx / names.size() + 1]

		var id := GameData.generate_uid()
		var province := GameData.create_province(id, pname, x, y, terrain)
		provinces[id] = province
		coord_to_id[Vector2i(x, y)] = id

	# Set adjacency
	for coord in coord_to_id:
		var pid: String = coord_to_id[coord]
		var neighbors := _hex_neighbors(coord.x, coord.y, cols, rows)
		var adj_ids := []
		for n in neighbors:
			var nkey := Vector2i(n.x, n.y)
			if coord_to_id.has(nkey):
				adj_ids.append(coord_to_id[nkey])
		provinces[pid]["adjacent_ids"] = adj_ids

	return provinces

static func assign_starting_provinces(
	provinces: Dictionary,
	faction_ids: Array,
	per_faction: int = 3,
) -> Dictionary:
	var unowned := []
	for p in provinces.values():
		if p["owner_faction_id"] == "":
			unowned.append(p)
	unowned.shuffle()

	var assigned := {}
	var n := faction_ids.size()
	var chunk := maxi(1, unowned.size() / n)

	for i in n:
		var fid: String = faction_ids[i]
		assigned[fid] = []
		var start_idx := i * chunk
		var candidates := unowned.slice(start_idx, start_idx + chunk)
		# Sort by development
		candidates.sort_custom(func(a, b): return a["development"] > b["development"])
		for j in mini(per_faction, candidates.size()):
			candidates[j]["owner_faction_id"] = fid
			assigned[fid].append(candidates[j]["id"])

	return assigned

static func _weighted_terrain(rng: RandomNumberGenerator) -> GameData.TerrainType:
	var total_weight := 0
	for w in TERRAIN_WEIGHTS.values():
		total_weight += w
	var roll := rng.randi_range(0, total_weight - 1)
	var cumulative := 0
	for terrain in TERRAIN_WEIGHTS:
		cumulative += TERRAIN_WEIGHTS[terrain]
		if roll < cumulative:
			return terrain
	return GameData.TerrainType.PLAINS

static func _hex_neighbors(x: int, y: int, cols: int, rows: int) -> Array:
	var neighbors := []
	var deltas: Array
	if y % 2 == 0:
		deltas = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1]]
	else:
		deltas = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, -1], [1, 1]]
	for d in deltas:
		var nx := x + d[0]
		var ny := y + d[1]
		if nx >= 0 and nx < cols and ny >= 0 and ny < rows:
			neighbors.append(Vector2i(nx, ny))
	return neighbors
