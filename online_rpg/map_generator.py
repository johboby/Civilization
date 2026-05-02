"""Map generator for creating game worlds.

Generates a hex-grid-style province map with terrain, resources,
and adjacency suitable for RTK14/EU4-style gameplay.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .models import (
    Building,
    BuildingType,
    Province,
    TerrainType,
)

# Province name pools by culture
CHINESE_PROVINCE_NAMES = [
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

EUROPEAN_PROVINCE_NAMES = [
    "Britannia", "Gallia", "Hispania", "Italia", "Germania",
    "Scandia", "Sarmatia", "Dacia", "Thracia", "Macedonia",
    "Achaea", "Creta", "Aegyptus", "Syria", "Mesopotamia",
    "Persia", "Arabia", "Numidia", "Mauretania", "Lusitania",
    "Aquitania", "Belgica", "Raetia", "Noricum", "Pannonia",
    "Dalmatia", "Moesia", "Bithynia", "Galatia", "Cappadocia",
]

TERRAIN_WEIGHTS = {
    TerrainType.PLAINS: 35,
    TerrainType.FOREST: 20,
    TerrainType.MOUNTAINS: 15,
    TerrainType.RIVER: 10,
    TerrainType.COAST: 10,
    TerrainType.DESERT: 5,
    TerrainType.MARSH: 5,
}

TERRAIN_MODIFIERS = {
    TerrainType.PLAINS: {"food": 1.3, "gold": 1.0, "manpower": 1.2, "dev": 1.2},
    TerrainType.FOREST: {"food": 0.9, "gold": 0.8, "manpower": 0.9, "dev": 0.8},
    TerrainType.MOUNTAINS: {"food": 0.5, "gold": 1.3, "manpower": 0.7, "dev": 0.6},
    TerrainType.RIVER: {"food": 1.5, "gold": 1.2, "manpower": 1.1, "dev": 1.3},
    TerrainType.COAST: {"food": 1.1, "gold": 1.4, "manpower": 1.0, "dev": 1.1},
    TerrainType.DESERT: {"food": 0.3, "gold": 0.6, "manpower": 0.5, "dev": 0.4},
    TerrainType.MARSH: {"food": 0.7, "gold": 0.5, "manpower": 0.6, "dev": 0.5},
}


def _weighted_terrain() -> TerrainType:
    terrains = list(TERRAIN_WEIGHTS.keys())
    weights = list(TERRAIN_WEIGHTS.values())
    return random.choices(terrains, weights=weights, k=1)[0]


def _hex_neighbors(x: int, y: int, cols: int, rows: int) -> List[Tuple[int, int]]:
    """Get hex-grid neighbor coordinates (offset coordinates)."""
    neighbors = []
    if y % 2 == 0:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1)]
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            neighbors.append((nx, ny))
    return neighbors


def generate_map(
    cols: int = 10,
    rows: int = 8,
    name_pool: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Province]:
    """Generate a map of provinces on a hex grid.

    Args:
        cols: Number of columns in the grid.
        rows: Number of rows in the grid.
        name_pool: Optional list of province names to use.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping province id to Province objects.
    """
    if seed is not None:
        random.seed(seed)

    if name_pool is None:
        name_pool = list(CHINESE_PROVINCE_NAMES)
    random.shuffle(name_pool)

    provinces: Dict[str, Province] = {}
    coord_to_id: Dict[Tuple[int, int], str] = {}

    total = cols * rows
    for idx in range(total):
        x = idx % cols
        y = idx // cols

        terrain = _weighted_terrain()
        mods = TERRAIN_MODIFIERS[terrain]

        name = name_pool[idx % len(name_pool)] if name_pool else f"Province_{idx}"
        # Add suffix if we cycle through names
        if idx >= len(name_pool):
            name = f"{name} {idx // len(name_pool) + 1}"

        pop = int(random.gauss(10000, 3000))
        pop = max(1000, min(50000, pop))

        province = Province(
            name=name,
            x=x,
            y=y,
            terrain=terrain,
            population=pop,
            food=round(random.uniform(50, 200) * mods["food"], 1),
            gold=round(random.uniform(20, 100) * mods["gold"], 1),
            manpower=int(random.uniform(500, 2000) * mods["manpower"]),
            development=round(random.uniform(0.5, 2.0) * mods["dev"], 2),
            food_modifier=mods["food"],
            gold_modifier=mods["gold"],
            manpower_modifier=mods["manpower"],
            fortification=random.choice([0, 0, 0, 1, 1, 2]),
        )

        # Random initial buildings
        if random.random() < 0.4:
            btype = random.choice([BuildingType.FARM, BuildingType.MARKET, BuildingType.BARRACKS])
            province.buildings.append(Building(type=btype, level=1))

        provinces[province.id] = province
        coord_to_id[(x, y)] = province.id

    # Set adjacency
    for (x, y), pid in coord_to_id.items():
        neighbors = _hex_neighbors(x, y, cols, rows)
        provinces[pid].adjacent_province_ids = [
            coord_to_id[(nx, ny)] for (nx, ny) in neighbors if (nx, ny) in coord_to_id
        ]

    return provinces


def assign_starting_provinces(
    provinces: Dict[str, Province],
    faction_ids: List[str],
    provinces_per_faction: int = 3,
) -> Dict[str, List[str]]:
    """Assign starting provinces to factions, spread apart.

    Returns dict mapping faction_id -> list of province_ids.
    """
    unowned = [p for p in provinces.values() if p.owner_faction_id is None]
    random.shuffle(unowned)

    # Try to pick provinces spread apart
    assigned: Dict[str, List[str]] = {fid: [] for fid in faction_ids}

    # Simple approach: divide the map into regions
    n = len(faction_ids)
    chunk_size = max(1, len(unowned) // n)

    for i, fid in enumerate(faction_ids):
        start = i * chunk_size
        candidates = unowned[start: start + chunk_size]
        # Pick the best development provinces
        candidates.sort(key=lambda p: p.development, reverse=True)
        for p in candidates[:provinces_per_faction]:
            p.owner_faction_id = fid
            assigned[fid].append(p.id)

    return assigned
