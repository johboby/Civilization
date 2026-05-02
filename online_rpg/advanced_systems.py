"""Advanced Game Systems - Trade, Espionage, Culture, Religion, Victory.

Implements EU4/RTK14-inspired systems for deeper gameplay:
- Trade route network with merchant assignments
- Espionage with spy missions (sabotage, steal tech, incite rebellion)
- Culture and religion spread between provinces
- Detailed battle tactics and formations
- Campaign victory conditions
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    Army,
    Character,
    CharacterRole,
    DiplomacyStatus,
    Faction,
    Province,
    TerrainType,
)


# =========================================================================
# Trade System
# =========================================================================

class TradeGoodType(str, Enum):
    GRAIN = "grain"
    SILK = "silk"
    IRON = "iron"
    HORSES = "horses"
    SALT = "salt"
    WINE = "wine"
    SPICES = "spices"
    JADE = "jade"
    TEA = "tea"
    TIMBER = "timber"


@dataclass
class TradeRoute:
    """A trade route between two provinces."""
    id: str
    source_province_id: str
    target_province_id: str
    trade_good: TradeGoodType
    value: float = 10.0
    active: bool = True
    merchant_character_id: Optional[str] = None
    efficiency: float = 1.0  # Affected by roads, safety, merchant skill

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source_province_id,
            "target": self.target_province_id,
            "good": self.trade_good.value,
            "value": round(self.value, 1),
            "active": self.active,
            "merchant_id": self.merchant_character_id,
            "efficiency": round(self.efficiency, 2),
        }


TERRAIN_TRADE_GOODS = {
    TerrainType.PLAINS: [TradeGoodType.GRAIN, TradeGoodType.HORSES],
    TerrainType.FOREST: [TradeGoodType.TIMBER, TradeGoodType.TEA],
    TerrainType.MOUNTAINS: [TradeGoodType.IRON, TradeGoodType.JADE],
    TerrainType.RIVER: [TradeGoodType.GRAIN, TradeGoodType.SILK],
    TerrainType.COAST: [TradeGoodType.SALT, TradeGoodType.SPICES],
    TerrainType.DESERT: [TradeGoodType.SPICES, TradeGoodType.SALT],
    TerrainType.MARSH: [TradeGoodType.TEA, TradeGoodType.GRAIN],
}

TRADE_GOOD_VALUES = {
    TradeGoodType.GRAIN: 5.0,
    TradeGoodType.SILK: 15.0,
    TradeGoodType.IRON: 10.0,
    TradeGoodType.HORSES: 12.0,
    TradeGoodType.SALT: 8.0,
    TradeGoodType.WINE: 12.0,
    TradeGoodType.SPICES: 18.0,
    TradeGoodType.JADE: 20.0,
    TradeGoodType.TEA: 7.0,
    TradeGoodType.TIMBER: 6.0,
}


class TradeSystem:
    """Manages trade routes and income."""

    def __init__(self) -> None:
        self.routes: Dict[str, TradeRoute] = {}
        self._uid = 0

    def generate_trade_goods(self, provinces: Dict[str, Province]) -> Dict[str, TradeGoodType]:
        """Assign trade goods to provinces based on terrain."""
        goods: Dict[str, TradeGoodType] = {}
        for pid, province in provinces.items():
            terrain_goods = TERRAIN_TRADE_GOODS.get(province.terrain, [TradeGoodType.GRAIN])
            goods[pid] = random.choice(terrain_goods)
        return goods

    def create_route(self, source_id: str, target_id: str, good: TradeGoodType) -> TradeRoute:
        self._uid += 1
        route = TradeRoute(
            id=f"tr_{self._uid}",
            source_province_id=source_id,
            target_province_id=target_id,
            trade_good=good,
            value=TRADE_GOOD_VALUES.get(good, 10.0),
        )
        self.routes[route.id] = route
        return route

    def process_trade(self, factions: Dict[str, Faction], provinces: Dict[str, Province]) -> List[dict]:
        """Process trade income for all active routes."""
        events = []
        for route in self.routes.values():
            if not route.active:
                continue
            source = provinces.get(route.source_province_id)
            target = provinces.get(route.target_province_id)
            if not source or not target:
                continue
            if not source.owner_faction_id or not target.owner_faction_id:
                continue

            income = route.value * route.efficiency
            # Both sides benefit
            src_faction = factions.get(source.owner_faction_id)
            tgt_faction = factions.get(target.owner_faction_id)
            if src_faction:
                src_faction.gold += income * 0.6
            if tgt_faction:
                tgt_faction.gold += income * 0.4

        return events


# =========================================================================
# Espionage System
# =========================================================================

class SpyMission(str, Enum):
    SCOUT = "scout"               # Reveal province details
    SABOTAGE = "sabotage"         # Damage buildings/reduce development
    STEAL_TECH = "steal_tech"     # Copy enemy technology
    INCITE_REBELLION = "incite"   # Increase unrest
    ASSASSINATE = "assassinate"   # Attempt to kill a character
    COUNTER_ESPIONAGE = "counter" # Catch enemy spies


@dataclass
class SpyOperation:
    """An active spy mission."""
    spy_character_id: str
    mission: SpyMission
    target_province_id: str
    target_faction_id: str
    progress: float = 0.0
    success_chance: float = 0.5
    turns_remaining: int = 3
    completed: bool = False
    discovered: bool = False

    def to_dict(self) -> dict:
        return {
            "spy_id": self.spy_character_id,
            "mission": self.mission.value,
            "target_province": self.target_province_id,
            "target_faction": self.target_faction_id,
            "progress": round(self.progress, 2),
            "turns_remaining": self.turns_remaining,
            "completed": self.completed,
        }


class EspionageSystem:
    """Handles spy missions and counter-espionage."""

    def __init__(self) -> None:
        self.operations: List[SpyOperation] = []

    def start_mission(
        self,
        spy: Character,
        mission: SpyMission,
        target_province_id: str,
        target_faction_id: str,
    ) -> SpyOperation:
        base_chance = 0.4 + spy.stats.intelligence * 0.004
        turns = {
            SpyMission.SCOUT: 1,
            SpyMission.SABOTAGE: 3,
            SpyMission.STEAL_TECH: 4,
            SpyMission.INCITE_REBELLION: 3,
            SpyMission.ASSASSINATE: 5,
            SpyMission.COUNTER_ESPIONAGE: 2,
        }
        op = SpyOperation(
            spy_character_id=spy.id,
            mission=mission,
            target_province_id=target_province_id,
            target_faction_id=target_faction_id,
            success_chance=min(0.9, base_chance),
            turns_remaining=turns.get(mission, 3),
        )
        self.operations.append(op)
        return op

    def process_espionage(
        self,
        characters: Dict[str, Character],
        provinces: Dict[str, Province],
        factions: Dict[str, Faction],
    ) -> List[dict]:
        events = []
        for op in self.operations:
            if op.completed:
                continue

            op.turns_remaining -= 1
            spy = characters.get(op.spy_character_id)
            if not spy or not spy.alive:
                op.completed = True
                continue

            if op.turns_remaining <= 0:
                op.completed = True
                # Check success
                if random.random() < op.success_chance:
                    events.extend(self._apply_success(op, spy, characters, provinces, factions))
                else:
                    # Caught!
                    op.discovered = True
                    spy.stats.loyalty = max(0, spy.stats.loyalty - 20)
                    events.append({
                        "type": "spy_caught",
                        "description": f"Spy {spy.name} was caught during {op.mission.value}!",
                        "faction_id": spy.faction_id,
                    })

        # Remove completed
        self.operations = [op for op in self.operations if not op.completed]
        return events

    def _apply_success(self, op, spy, characters, provinces, factions) -> List[dict]:
        events = []
        province = provinces.get(op.target_province_id)
        target_faction = factions.get(op.target_faction_id)

        if op.mission == SpyMission.SABOTAGE and province:
            province.development = max(0.1, province.development - 0.2)
            if province.buildings:
                province.buildings.pop()
            events.append({
                "type": "sabotage_success",
                "description": f"Spy {spy.name} sabotaged {province.name}!",
                "faction_id": spy.faction_id,
                "province_id": province.id,
            })

        elif op.mission == SpyMission.STEAL_TECH and target_faction:
            spy_faction = factions.get(spy.faction_id)
            if spy_faction and target_faction.researched_techs:
                stealable = target_faction.researched_techs - spy_faction.researched_techs
                if stealable:
                    stolen = random.choice(list(stealable))
                    spy_faction.researched_techs.add(stolen)
                    events.append({
                        "type": "tech_stolen",
                        "description": f"Spy {spy.name} stole technology from {target_faction.name}!",
                        "faction_id": spy.faction_id,
                    })

        elif op.mission == SpyMission.INCITE_REBELLION and province:
            province.unrest = min(100, province.unrest + 30)
            events.append({
                "type": "rebellion_incited",
                "description": f"Spy {spy.name} incited rebellion in {province.name}!",
                "faction_id": spy.faction_id,
                "province_id": province.id,
            })

        elif op.mission == SpyMission.ASSASSINATE:
            # Find target character
            enemy_chars = [c for c in characters.values()
                          if c.faction_id == op.target_faction_id and c.alive]
            if enemy_chars:
                target = random.choice(enemy_chars)
                target.alive = False
                events.append({
                    "type": "assassination",
                    "description": f"Spy {spy.name} assassinated {target.name}!",
                    "faction_id": spy.faction_id,
                })

        elif op.mission == SpyMission.SCOUT and province:
            events.append({
                "type": "scout_success",
                "description": f"Spy {spy.name} scouted {province.name}: Pop {province.population}, Fort {province.fortification}",
                "faction_id": spy.faction_id,
                "province_id": province.id,
            })

        spy.gain_experience(50)
        return events


# =========================================================================
# Culture & Religion System
# =========================================================================

CULTURES = ["Han", "Wu", "Shu", "Wei", "Qiang", "Nanman", "Xiongnu", "Wuhuan"]
RELIGIONS = ["Confucianism", "Taoism", "Buddhism", "Folk Religion", "Legalism"]


class CultureReligionSystem:
    """Handles culture and religion spread between provinces."""

    def __init__(self) -> None:
        self.province_cultures: Dict[str, str] = {}
        self.province_religions: Dict[str, str] = {}

    def initialize(self, provinces: Dict[str, Province]) -> None:
        for pid in provinces:
            self.province_cultures[pid] = random.choice(CULTURES)
            self.province_religions[pid] = random.choice(RELIGIONS)

    def process_spread(self, provinces: Dict[str, Province], factions: Dict[str, Faction]) -> List[dict]:
        events = []
        for pid, province in provinces.items():
            if not province.owner_faction_id:
                continue

            # Culture spreads from adjacent provinces of same faction
            for adj_id in province.adjacent_province_ids:
                adj = provinces.get(adj_id)
                if not adj or adj.owner_faction_id != province.owner_faction_id:
                    continue
                # Small chance to adopt neighbor's culture
                if self.province_cultures.get(pid) != self.province_cultures.get(adj_id):
                    if random.random() < 0.02:
                        old = self.province_cultures.get(pid, "Unknown")
                        self.province_cultures[pid] = self.province_cultures.get(adj_id, old)

            # Same-culture provinces have lower unrest
            faction_culture = self._get_faction_dominant_culture(province.owner_faction_id, provinces)
            if self.province_cultures.get(pid) == faction_culture:
                province.unrest = max(0, province.unrest - 0.5)
            else:
                province.unrest = min(100, province.unrest + 0.3)

        return events

    def _get_faction_dominant_culture(self, faction_id: str, provinces: Dict[str, Province]) -> str:
        culture_count: Dict[str, int] = {}
        for pid, province in provinces.items():
            if province.owner_faction_id == faction_id:
                c = self.province_cultures.get(pid, "Han")
                culture_count[c] = culture_count.get(c, 0) + 1
        if not culture_count:
            return "Han"
        return max(culture_count, key=culture_count.get)

    def get_province_info(self, province_id: str) -> dict:
        return {
            "culture": self.province_cultures.get(province_id, "Unknown"),
            "religion": self.province_religions.get(province_id, "Unknown"),
        }


# =========================================================================
# Battle Tactics System
# =========================================================================

class Formation(str, Enum):
    STANDARD = "standard"
    DEFENSIVE = "defensive"
    AGGRESSIVE = "aggressive"
    FLANKING = "flanking"
    AMBUSH = "ambush"
    SIEGE = "siege"


class Tactic(str, Enum):
    CHARGE = "charge"           # Bonus attack, penalty defense
    HOLD_LINE = "hold_line"     # Bonus defense, penalty attack
    FEIGNED_RETREAT = "feigned" # High risk, high reward
    FIRE_ARROWS = "fire_arrows" # Archer bonus
    CAVALRY_CHARGE = "cavalry"  # Cavalry bonus
    ENCIRCLE = "encircle"       # Requires more troops


FORMATION_MODIFIERS = {
    Formation.STANDARD: {"attack": 1.0, "defense": 1.0},
    Formation.DEFENSIVE: {"attack": 0.8, "defense": 1.3},
    Formation.AGGRESSIVE: {"attack": 1.3, "defense": 0.8},
    Formation.FLANKING: {"attack": 1.15, "defense": 0.9},
    Formation.AMBUSH: {"attack": 1.4, "defense": 0.7},
    Formation.SIEGE: {"attack": 0.7, "defense": 1.1, "siege": 1.5},
}

TACTIC_BONUSES = {
    Tactic.CHARGE: {"attack": 1.2, "cavalry_bonus": 0.3, "defense": 0.85},
    Tactic.HOLD_LINE: {"defense": 1.3, "attack": 0.85},
    Tactic.FEIGNED_RETREAT: {"attack": 1.4, "defense": 0.6},
    Tactic.FIRE_ARROWS: {"archer_bonus": 0.5, "attack": 1.1},
    Tactic.CAVALRY_CHARGE: {"cavalry_bonus": 0.6, "attack": 1.15},
    Tactic.ENCIRCLE: {"attack": 1.3, "defense": 1.1, "min_ratio": 1.5},
}


@dataclass
class BattleResult:
    """Detailed result of a battle."""
    attacker_faction: str
    defender_faction: str
    province_name: str
    attacker_initial: int
    defender_initial: int
    attacker_losses: int
    defender_losses: int
    winner: str
    attacker_formation: Formation
    defender_formation: Formation
    attacker_tactic: Optional[Tactic] = None
    defender_tactic: Optional[Tactic] = None
    attacker_general: str = ""
    defender_general: str = ""
    terrain_bonus: float = 1.0

    def to_dict(self) -> dict:
        return {
            "attacker": self.attacker_faction,
            "defender": self.defender_faction,
            "province": self.province_name,
            "attacker_initial": self.attacker_initial,
            "defender_initial": self.defender_initial,
            "attacker_losses": self.attacker_losses,
            "defender_losses": self.defender_losses,
            "winner": self.winner,
            "attacker_formation": self.attacker_formation.value,
            "defender_formation": self.defender_formation.value,
        }


class TacticalBattleSystem:
    """Enhanced battle resolution with formations and tactics."""

    def resolve_battle(
        self,
        attacker: Army,
        defender: Army,
        province: Province,
        att_general: Optional[Character] = None,
        def_general: Optional[Character] = None,
        att_formation: Formation = Formation.STANDARD,
        def_formation: Formation = Formation.DEFENSIVE,
    ) -> BattleResult:
        att_mods = FORMATION_MODIFIERS[att_formation]
        def_mods = FORMATION_MODIFIERS[def_formation]

        # Base strength
        att_str = (attacker.infantry * 1.0 + attacker.cavalry * 1.5 + attacker.archers * 1.2)
        def_str = (defender.infantry * 1.0 + defender.cavalry * 1.5 + defender.archers * 1.2)

        # Formation modifiers
        att_str *= att_mods.get("attack", 1.0)
        def_str *= def_mods.get("defense", 1.0)

        # General bonuses
        if att_general:
            att_str *= (1.0 + att_general.stats.command * 0.005 + att_general.stats.force * 0.002)
            # Intelligence helps pick better tactics
            if att_general.stats.intelligence > 70:
                att_str *= 1.05
        if def_general:
            def_str *= (1.0 + def_general.stats.command * 0.005 + def_general.stats.force * 0.002)
            if def_general.stats.intelligence > 70:
                def_str *= 1.05

        # Terrain bonus for defender
        terrain_bonus = 1.0
        if province.terrain == TerrainType.MOUNTAINS:
            terrain_bonus = 1.3
        elif province.terrain == TerrainType.FOREST:
            terrain_bonus = 1.15
        elif province.terrain == TerrainType.RIVER:
            terrain_bonus = 1.1
        terrain_bonus += province.fortification * 0.1
        def_str *= terrain_bonus

        # Morale factor
        att_str *= attacker.morale / 100.0
        def_str *= defender.morale / 100.0

        # Randomness
        att_str *= random.uniform(0.85, 1.15)
        def_str *= random.uniform(0.85, 1.15)

        total = att_str + def_str
        if total == 0:
            total = 1

        ratio = att_str / total
        att_casualty_rate = (1 - ratio) * random.uniform(0.1, 0.3)
        def_casualty_rate = ratio * random.uniform(0.1, 0.3)

        att_losses = int(attacker.soldiers * att_casualty_rate)
        def_losses = int(defender.soldiers * def_casualty_rate)

        winner = "attacker" if att_str > def_str else "defender"

        return BattleResult(
            attacker_faction=attacker.faction_id,
            defender_faction=defender.faction_id,
            province_name=province.name,
            attacker_initial=attacker.soldiers,
            defender_initial=defender.soldiers,
            attacker_losses=att_losses,
            defender_losses=def_losses,
            winner=winner,
            attacker_formation=att_formation,
            defender_formation=def_formation,
            attacker_general=att_general.name if att_general else "",
            defender_general=def_general.name if def_general else "",
            terrain_bonus=terrain_bonus,
        )


# =========================================================================
# Victory Conditions
# =========================================================================

class VictoryType(str, Enum):
    DOMINATION = "domination"       # Control 60%+ of provinces
    DIPLOMATIC = "diplomatic"       # Allied with 50%+ of factions
    TECHNOLOGICAL = "technological" # Research all technologies
    CULTURAL = "cultural"           # Dominant culture in 70%+ provinces
    ECONOMIC = "economic"           # Accumulate 10000 gold
    ELIMINATION = "elimination"     # Last faction standing


@dataclass
class VictoryCondition:
    type: VictoryType
    threshold: float
    description: str


DEFAULT_VICTORY_CONDITIONS = [
    VictoryCondition(VictoryType.DOMINATION, 0.6, "Control 60% of all provinces"),
    VictoryCondition(VictoryType.ECONOMIC, 10000, "Accumulate 10,000 gold"),
    VictoryCondition(VictoryType.TECHNOLOGICAL, 1.0, "Research all technologies"),
    VictoryCondition(VictoryType.ELIMINATION, 1.0, "Be the last faction standing"),
]


class VictorySystem:
    """Checks victory conditions each turn."""

    def __init__(self, conditions: Optional[List[VictoryCondition]] = None):
        self.conditions = conditions or DEFAULT_VICTORY_CONDITIONS

    def check_victory(
        self,
        factions: Dict[str, Faction],
        provinces: Dict[str, Province],
        total_techs: int = 27,
    ) -> Optional[Tuple[str, VictoryType]]:
        total_provinces = len(provinces)
        active_factions = [f for f in factions.values()
                          if any(p.owner_faction_id == f.id for p in provinces.values())]

        for faction in active_factions:
            faction_provinces = sum(1 for p in provinces.values() if p.owner_faction_id == faction.id)

            for condition in self.conditions:
                if condition.type == VictoryType.DOMINATION:
                    if total_provinces > 0 and faction_provinces / total_provinces >= condition.threshold:
                        return (faction.id, VictoryType.DOMINATION)

                elif condition.type == VictoryType.ECONOMIC:
                    if faction.gold >= condition.threshold:
                        return (faction.id, VictoryType.ECONOMIC)

                elif condition.type == VictoryType.TECHNOLOGICAL:
                    if len(faction.researched_techs) >= total_techs:
                        return (faction.id, VictoryType.TECHNOLOGICAL)

                elif condition.type == VictoryType.ELIMINATION:
                    if len(active_factions) == 1:
                        return (faction.id, VictoryType.ELIMINATION)

        return None


# =========================================================================
# Historical Scenarios
# =========================================================================

THREE_KINGDOMS_190AD = {
    "name": "Three Kingdoms - 190 AD",
    "description": "The fall of the Han Dynasty. Warlords vie for control of China.",
    "year": 190,
    "factions": [
        {
            "name": "Cao Cao's Wei",
            "color": "#3498db",
            "ruler_name": "Cao Cao",
            "personality": "balanced",
        },
        {
            "name": "Liu Bei's Shu",
            "color": "#2ecc71",
            "ruler_name": "Liu Bei",
            "personality": "peaceful",
        },
        {
            "name": "Sun Quan's Wu",
            "color": "#e74c3c",
            "ruler_name": "Sun Quan",
            "personality": "balanced",
        },
        {
            "name": "Yuan Shao's Coalition",
            "color": "#f39c12",
            "ruler_name": "Yuan Shao",
            "personality": "warlike",
        },
        {
            "name": "Dong Zhuo's Tyranny",
            "color": "#9b59b6",
            "ruler_name": "Dong Zhuo",
            "personality": "warlike",
        },
    ],
    "map_cols": 12,
    "map_rows": 10,
}

WARRING_STATES = {
    "name": "Warring States - 475 BC",
    "description": "Seven kingdoms compete for supremacy in ancient China.",
    "year": -475,
    "factions": [
        {"name": "Qin", "color": "#1a1a1a", "ruler_name": "Duke Xiao", "personality": "warlike"},
        {"name": "Chu", "color": "#e74c3c", "ruler_name": "King Wei", "personality": "balanced"},
        {"name": "Qi", "color": "#3498db", "ruler_name": "King Wei", "personality": "peaceful"},
        {"name": "Wei", "color": "#f39c12", "ruler_name": "King Hui", "personality": "balanced"},
        {"name": "Zhao", "color": "#2ecc71", "ruler_name": "King Wuling", "personality": "warlike"},
        {"name": "Han", "color": "#9b59b6", "ruler_name": "Marquess Ai", "personality": "peaceful"},
        {"name": "Yan", "color": "#1abc9c", "ruler_name": "King Kuai", "personality": "balanced"},
    ],
    "map_cols": 14,
    "map_rows": 12,
}

SCENARIOS = {
    "three_kingdoms": THREE_KINGDOMS_190AD,
    "warring_states": WARRING_STATES,
}


def get_scenario(name: str) -> Optional[dict]:
    return SCENARIOS.get(name)

def list_scenarios() -> List[dict]:
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in SCENARIOS.items()
    ]
