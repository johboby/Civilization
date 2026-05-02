"""Game data models for the Online RPG.

Defines provinces, characters, factions, armies, and all core game entities
inspired by Romance of the Three Kingdoms 14 and Europa Universalis 4.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TerrainType(str, Enum):
    PLAINS = "plains"
    MOUNTAINS = "mountains"
    FOREST = "forest"
    DESERT = "desert"
    RIVER = "river"
    COAST = "coast"
    MARSH = "marsh"


class CharacterRole(str, Enum):
    RULER = "ruler"
    GENERAL = "general"
    STRATEGIST = "strategist"
    GOVERNOR = "governor"
    DIPLOMAT = "diplomat"
    SPY = "spy"
    FREE = "free"


class DiplomacyStatus(str, Enum):
    NEUTRAL = "neutral"
    ALLY = "ally"
    WAR = "war"
    VASSAL = "vassal"
    TRUCE = "truce"
    TRADE = "trade"


class ArmyStatus(str, Enum):
    IDLE = "idle"
    MARCHING = "marching"
    BESIEGING = "besieging"
    DEFENDING = "defending"
    RETREATING = "retreating"


class TechCategory(str, Enum):
    MILITARY = "military"
    ECONOMY = "economy"
    CULTURE = "culture"
    DIPLOMACY = "diplomacy"


class BuildingType(str, Enum):
    FARM = "farm"
    MARKET = "market"
    BARRACKS = "barracks"
    WALL = "wall"
    ACADEMY = "academy"
    TEMPLE = "temple"
    WORKSHOP = "workshop"
    PORT = "port"


class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CharacterStats:
    """Core stats for a character (RTK14-inspired)."""
    command: int = 50       # Military command ability
    force: int = 50         # Personal combat strength
    intelligence: int = 50  # Strategy and governance
    politics: int = 50      # Administration and diplomacy
    charisma: int = 50      # Influence and recruitment
    loyalty: int = 80       # Loyalty to their faction


@dataclass
class Character:
    """A game character / officer."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Unknown"
    role: CharacterRole = CharacterRole.FREE
    faction_id: Optional[str] = None
    province_id: Optional[str] = None
    stats: CharacterStats = field(default_factory=CharacterStats)
    level: int = 1
    experience: int = 0
    skills: List[str] = field(default_factory=list)
    alive: bool = True
    age: int = 25

    def gain_experience(self, amount: int) -> bool:
        """Add experience, return True if leveled up."""
        self.experience += amount
        threshold = self.level * 100
        if self.experience >= threshold:
            self.experience -= threshold
            self.level += 1
            # Small stat boost on level up
            self.stats.command = min(100, self.stats.command + random.randint(0, 2))
            self.stats.force = min(100, self.stats.force + random.randint(0, 2))
            self.stats.intelligence = min(100, self.stats.intelligence + random.randint(0, 2))
            self.stats.politics = min(100, self.stats.politics + random.randint(0, 2))
            self.stats.charisma = min(100, self.stats.charisma + random.randint(0, 2))
            return True
        return False

    def total_power(self) -> int:
        s = self.stats
        return s.command + s.force + s.intelligence + s.politics + s.charisma

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "faction_id": self.faction_id,
            "province_id": self.province_id,
            "stats": {
                "command": self.stats.command,
                "force": self.stats.force,
                "intelligence": self.stats.intelligence,
                "politics": self.stats.politics,
                "charisma": self.stats.charisma,
                "loyalty": self.stats.loyalty,
            },
            "level": self.level,
            "experience": self.experience,
            "skills": self.skills,
            "alive": self.alive,
            "age": self.age,
        }


@dataclass
class Building:
    """A building in a province."""
    type: BuildingType
    level: int = 1

    def to_dict(self) -> dict:
        return {"type": self.type.value, "level": self.level}


@dataclass
class Province:
    """A map province (RTK14-style)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Unknown"
    x: int = 0
    y: int = 0
    terrain: TerrainType = TerrainType.PLAINS
    owner_faction_id: Optional[str] = None

    # Resources
    population: int = 10000
    food: float = 100.0
    gold: float = 50.0
    manpower: int = 1000

    # Development
    development: float = 1.0
    fortification: int = 0
    buildings: List[Building] = field(default_factory=list)

    # Adjacency
    adjacent_province_ids: List[str] = field(default_factory=list)

    # Production modifiers based on terrain
    food_modifier: float = 1.0
    gold_modifier: float = 1.0
    manpower_modifier: float = 1.0

    # Unrest / loyalty
    unrest: float = 0.0
    culture: str = "default"

    def compute_income(self) -> dict:
        """Compute per-turn income."""
        base_food = 10.0 * self.development * self.food_modifier
        base_gold = 5.0 * self.development * self.gold_modifier
        base_manpower = int(50 * self.development * self.manpower_modifier)

        for b in self.buildings:
            if b.type == BuildingType.FARM:
                base_food += 5.0 * b.level
            elif b.type == BuildingType.MARKET:
                base_gold += 3.0 * b.level
            elif b.type == BuildingType.BARRACKS:
                base_manpower += 30 * b.level
            elif b.type == BuildingType.WORKSHOP:
                base_gold += 2.0 * b.level
                base_food += 2.0 * b.level

        # Unrest penalty
        penalty = max(0.0, 1.0 - self.unrest * 0.01)
        return {
            "food": base_food * penalty,
            "gold": base_gold * penalty,
            "manpower": int(base_manpower * penalty),
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "terrain": self.terrain.value,
            "owner_faction_id": self.owner_faction_id,
            "population": self.population,
            "food": round(self.food, 1),
            "gold": round(self.gold, 1),
            "manpower": self.manpower,
            "development": round(self.development, 2),
            "fortification": self.fortification,
            "buildings": [b.to_dict() for b in self.buildings],
            "adjacent_province_ids": self.adjacent_province_ids,
            "unrest": round(self.unrest, 1),
            "culture": self.culture,
        }


@dataclass
class Technology:
    """A technology node in the tech tree."""
    id: str
    name: str
    category: TechCategory
    level: int = 1
    cost: int = 100
    description: str = ""
    prerequisites: List[str] = field(default_factory=list)
    effects: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "level": self.level,
            "cost": self.cost,
            "description": self.description,
            "prerequisites": self.prerequisites,
            "effects": self.effects,
        }


@dataclass
class Army:
    """A military unit / army."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    faction_id: str = ""
    general_id: Optional[str] = None
    province_id: str = ""
    soldiers: int = 1000
    morale: float = 100.0
    status: ArmyStatus = ArmyStatus.IDLE
    target_province_id: Optional[str] = None
    march_progress: float = 0.0  # 0.0 to 1.0

    # Unit composition
    infantry: int = 700
    cavalry: int = 200
    archers: int = 100

    def total_strength(self, general: Optional[Character] = None) -> float:
        """Compute total combat strength."""
        base = self.infantry * 1.0 + self.cavalry * 1.5 + self.archers * 1.2
        morale_factor = self.morale / 100.0
        general_bonus = 1.0
        if general:
            general_bonus += general.stats.command * 0.005
            general_bonus += general.stats.force * 0.002
        return base * morale_factor * general_bonus

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "faction_id": self.faction_id,
            "general_id": self.general_id,
            "province_id": self.province_id,
            "soldiers": self.soldiers,
            "morale": round(self.morale, 1),
            "status": self.status.value,
            "target_province_id": self.target_province_id,
            "march_progress": round(self.march_progress, 2),
            "infantry": self.infantry,
            "cavalry": self.cavalry,
            "archers": self.archers,
        }


@dataclass
class DiplomacyRelation:
    """Relationship between two factions."""
    faction_a_id: str
    faction_b_id: str
    status: DiplomacyStatus = DiplomacyStatus.NEUTRAL
    opinion: int = 0  # -100 to 100
    trade_value: float = 0.0
    truce_turns: int = 0

    def to_dict(self) -> dict:
        return {
            "faction_a_id": self.faction_a_id,
            "faction_b_id": self.faction_b_id,
            "status": self.status.value,
            "opinion": self.opinion,
            "trade_value": round(self.trade_value, 1),
            "truce_turns": self.truce_turns,
        }


@dataclass
class Faction:
    """A player-controlled or AI faction (like a nation in EU4 / force in RTK14)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Unknown Faction"
    color: str = "#ff0000"
    player_id: Optional[str] = None  # None = AI-controlled
    ruler_character_id: Optional[str] = None

    # Resources (aggregate)
    gold: float = 500.0
    food: float = 500.0
    manpower: int = 5000

    # Tech progress
    tech_levels: Dict[str, int] = field(default_factory=lambda: {
        "military": 1,
        "economy": 1,
        "culture": 1,
        "diplomacy": 1,
    })
    research_points: float = 0.0
    current_research: Optional[str] = None

    # Policies (EU4-style)
    tax_rate: float = 0.1
    conscription_rate: float = 0.05
    war_exhaustion: float = 0.0
    stability: float = 50.0
    legitimacy: float = 50.0
    prestige: float = 0.0

    # Known techs
    researched_techs: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "player_id": self.player_id,
            "ruler_character_id": self.ruler_character_id,
            "gold": round(self.gold, 1),
            "food": round(self.food, 1),
            "manpower": self.manpower,
            "tech_levels": dict(self.tech_levels),
            "research_points": round(self.research_points, 1),
            "current_research": self.current_research,
            "tax_rate": self.tax_rate,
            "conscription_rate": self.conscription_rate,
            "war_exhaustion": round(self.war_exhaustion, 1),
            "stability": round(self.stability, 1),
            "legitimacy": round(self.legitimacy, 1),
            "prestige": round(self.prestige, 1),
            "researched_techs": list(self.researched_techs),
        }


@dataclass
class GameEvent:
    """A game event / log entry."""
    turn: int
    event_type: str
    description: str
    faction_id: Optional[str] = None
    province_id: Optional[str] = None
    data: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "event_type": self.event_type,
            "description": self.description,
            "faction_id": self.faction_id,
            "province_id": self.province_id,
            "data": self.data,
        }
