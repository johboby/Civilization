"""Game engine - processes turns, combat, diplomacy, economy.

The core logic that drives the game simulation, combining elements
from Romance of the Three Kingdoms 14 and Europa Universalis 4.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Army,
    ArmyStatus,
    Building,
    BuildingType,
    Character,
    CharacterRole,
    CharacterStats,
    DiplomacyRelation,
    DiplomacyStatus,
    Faction,
    GameEvent,
    Province,
    Season,
    TerrainType,
)
from .map_generator import generate_map, assign_starting_provinces
from .tech_tree import build_tech_tree, get_available_techs


# ---------------------------------------------------------------------------
# Character name generators
# ---------------------------------------------------------------------------

CHINESE_SURNAMES = [
    "Liu", "Cao", "Sun", "Zhuge", "Guan", "Zhang", "Zhao", "Ma",
    "Huang", "Wei", "Deng", "Jiang", "Lu", "Zhou", "Xu", "Sima",
    "Yuan", "Dong", "Lv", "Chen", "Wang", "Li", "Yang", "Pang",
]

CHINESE_GIVEN_NAMES = [
    "Bei", "Cao", "Quan", "Liang", "Yu", "Fei", "Yun", "Chao",
    "Zhong", "Yan", "Ai", "Wei", "Su", "Yu", "Xun", "Yi",
    "Shao", "Zhuo", "Bu", "Gong", "Tao", "Ping", "Ce", "Ji",
]


def _random_chinese_name() -> str:
    return f"{random.choice(CHINESE_SURNAMES)} {random.choice(CHINESE_GIVEN_NAMES)}"


def _random_stats(quality: str = "normal") -> CharacterStats:
    """Generate random character stats."""
    if quality == "elite":
        mu, sigma = 75, 12
    elif quality == "good":
        mu, sigma = 60, 10
    else:
        mu, sigma = 50, 15

    def _clamp(v: float) -> int:
        return max(1, min(100, int(v)))

    return CharacterStats(
        command=_clamp(random.gauss(mu, sigma)),
        force=_clamp(random.gauss(mu, sigma)),
        intelligence=_clamp(random.gauss(mu, sigma)),
        politics=_clamp(random.gauss(mu, sigma)),
        charisma=_clamp(random.gauss(mu, sigma)),
        loyalty=_clamp(random.gauss(70, 15)),
    )


# ---------------------------------------------------------------------------
# Random events
# ---------------------------------------------------------------------------

RANDOM_EVENTS = [
    {"type": "plague", "desc": "A plague has struck {province}!", "pop_loss": 0.15, "unrest": 10},
    {"type": "bumper_harvest", "desc": "Bumper harvest in {province}!", "food_bonus": 50, "unrest": -5},
    {"type": "bandit_raid", "desc": "Bandits are raiding {province}!", "gold_loss": 20, "unrest": 8},
    {"type": "earthquake", "desc": "An earthquake hit {province}!", "pop_loss": 0.05, "fort_damage": 1},
    {"type": "trade_boom", "desc": "Trade is booming in {province}!", "gold_bonus": 30, "unrest": -3},
    {"type": "flood", "desc": "Flooding in {province}!", "food_loss": 30, "pop_loss": 0.03},
    {"type": "rebellion", "desc": "Rebellion in {province}!", "unrest": 25},
    {"type": "cultural_festival", "desc": "A festival in {province} boosts morale!", "unrest": -15, "prestige": 5},
    {"type": "talent_discovered", "desc": "A talented individual discovered in {province}!", "new_character": True},
    {"type": "diplomatic_incident", "desc": "A diplomatic incident affects relations!", "opinion_change": -15},
    {"type": "good_omen", "desc": "A good omen inspires the people of {province}!", "stability": 5, "unrest": -5},
]


# ---------------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Complete game state, serializable for persistence."""

    # Core state
    turn: int = 0
    season: Season = Season.SPRING
    year: int = 190  # Starting year (Three Kingdoms era)

    # Game entities
    provinces: Dict[str, Province] = field(default_factory=dict)
    factions: Dict[str, Faction] = field(default_factory=dict)
    characters: Dict[str, Character] = field(default_factory=dict)
    armies: Dict[str, Army] = field(default_factory=dict)
    diplomacy: Dict[str, DiplomacyRelation] = field(default_factory=dict)

    # Event log
    events: List[GameEvent] = field(default_factory=list)
    max_events: int = 500

    # Settings
    map_cols: int = 10
    map_rows: int = 8
    provinces_per_faction: int = 3
    random_event_chance: float = 0.15

    # Pending player actions per turn
    pending_actions: Dict[str, List[dict]] = field(default_factory=dict)

    # Turn readiness
    players_ready: set = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "season": self.season.value,
            "year": self.year,
            "provinces": {k: v.to_dict() for k, v in self.provinces.items()},
            "factions": {k: v.to_dict() for k, v in self.factions.items()},
            "characters": {k: v.to_dict() for k, v in self.characters.items()},
            "armies": {k: v.to_dict() for k, v in self.armies.items()},
            "diplomacy": {k: v.to_dict() for k, v in self.diplomacy.items()},
            "events": [e.to_dict() for e in self.events[-50:]],
        }

    def faction_provinces(self, faction_id: str) -> List[Province]:
        return [p for p in self.provinces.values() if p.owner_faction_id == faction_id]

    def faction_characters(self, faction_id: str) -> List[Character]:
        return [c for c in self.characters.values() if c.faction_id == faction_id and c.alive]

    def faction_armies(self, faction_id: str) -> List[Army]:
        return [a for a in self.armies.values() if a.faction_id == faction_id]


# ---------------------------------------------------------------------------
# Game Engine
# ---------------------------------------------------------------------------

class GameEngine:
    """Main game engine that processes all game logic."""

    def __init__(self, state: Optional[GameState] = None):
        self.state = state or GameState()
        self.tech_tree = build_tech_tree()

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def new_game(
        self,
        faction_configs: List[dict],
        map_cols: int = 10,
        map_rows: int = 8,
        seed: Optional[int] = None,
    ) -> GameState:
        """Create a new game with the given factions.

        Args:
            faction_configs: List of dicts with keys: name, color, player_id (optional).
            map_cols: Map width.
            map_rows: Map height.
            seed: Random seed.

        Returns:
            Initialized GameState.
        """
        self.state = GameState(map_cols=map_cols, map_rows=map_rows)

        # Generate map
        self.state.provinces = generate_map(cols=map_cols, rows=map_rows, seed=seed)

        # Create factions
        for cfg in faction_configs:
            faction = Faction(
                name=cfg.get("name", "Unknown"),
                color=cfg.get("color", "#ff0000"),
                player_id=cfg.get("player_id"),
            )
            self.state.factions[faction.id] = faction

            # Create ruler character
            ruler = Character(
                name=cfg.get("ruler_name", _random_chinese_name()),
                role=CharacterRole.RULER,
                faction_id=faction.id,
                stats=_random_stats("elite"),
            )
            self.state.characters[ruler.id] = ruler
            faction.ruler_character_id = ruler.id

            # Create starting officers
            for _ in range(3):
                officer = Character(
                    name=_random_chinese_name(),
                    role=random.choice([CharacterRole.GENERAL, CharacterRole.STRATEGIST, CharacterRole.GOVERNOR]),
                    faction_id=faction.id,
                    stats=_random_stats("good"),
                )
                self.state.characters[officer.id] = officer

        # Assign provinces
        faction_ids = list(self.state.factions.keys())
        assignments = assign_starting_provinces(
            self.state.provinces, faction_ids, self.state.provinces_per_faction
        )

        # Place characters in their faction's first province
        for fid, pids in assignments.items():
            if pids:
                for char in self.state.faction_characters(fid):
                    char.province_id = pids[0]

                # Create starting army
                army = Army(
                    faction_id=fid,
                    province_id=pids[0],
                    soldiers=3000,
                    infantry=2000,
                    cavalry=600,
                    archers=400,
                )
                # Assign a general
                generals = [c for c in self.state.faction_characters(fid)
                            if c.role == CharacterRole.GENERAL]
                if generals:
                    army.general_id = generals[0].id
                self.state.armies[army.id] = army

        # Initialize diplomacy
        fids = list(self.state.factions.keys())
        for i, a in enumerate(fids):
            for b in fids[i + 1:]:
                key = f"{a}_{b}"
                self.state.diplomacy[key] = DiplomacyRelation(
                    faction_a_id=a, faction_b_id=b
                )

        self._add_event("game_start", "A new era begins!")
        return self.state

    # -----------------------------------------------------------------------
    # Turn Processing
    # -----------------------------------------------------------------------

    def process_turn(self) -> List[GameEvent]:
        """Process one game turn. Returns events generated this turn."""
        turn_events: List[GameEvent] = []

        self.state.turn += 1
        self._advance_season()

        # 1. Process pending player actions
        for faction_id, actions in self.state.pending_actions.items():
            for action in actions:
                evts = self._process_action(faction_id, action)
                turn_events.extend(evts)
        self.state.pending_actions.clear()
        self.state.players_ready.clear()

        # 2. Economy phase
        turn_events.extend(self._process_economy())

        # 3. Army movement
        turn_events.extend(self._process_army_movement())

        # 4. Combat phase
        turn_events.extend(self._process_combat())

        # 5. Research phase
        turn_events.extend(self._process_research())

        # 6. Population and development
        turn_events.extend(self._process_population())

        # 7. Diplomacy decay
        self._process_diplomacy_decay()

        # 8. Random events
        turn_events.extend(self._process_random_events())

        # 9. Character aging and events
        turn_events.extend(self._process_characters())

        # 10. Check victory / elimination
        turn_events.extend(self._check_elimination())

        # Store events
        for evt in turn_events:
            self.state.events.append(evt)
        # Trim old events
        if len(self.state.events) > self.state.max_events:
            self.state.events = self.state.events[-self.state.max_events:]

        return turn_events

    def _advance_season(self) -> None:
        seasons = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
        idx = seasons.index(self.state.season)
        idx = (idx + 1) % 4
        self.state.season = seasons[idx]
        if self.state.season == Season.SPRING:
            self.state.year += 1

    # -----------------------------------------------------------------------
    # Player Actions
    # -----------------------------------------------------------------------

    def submit_action(self, faction_id: str, action: dict) -> dict:
        """Submit a player action for the current turn.

        Action types:
        - move_army: {army_id, target_province_id}
        - recruit_army: {province_id, soldiers, infantry, cavalry, archers}
        - build: {province_id, building_type}
        - research: {tech_id}
        - diplomacy: {target_faction_id, action: ally/war/trade/truce}
        - recruit_character: {province_id}
        - assign_governor: {character_id, province_id}
        - assign_general: {character_id, army_id}
        """
        if faction_id not in self.state.factions:
            return {"ok": False, "error": "Unknown faction"}

        if faction_id not in self.state.pending_actions:
            self.state.pending_actions[faction_id] = []
        self.state.pending_actions[faction_id].append(action)
        return {"ok": True}

    def mark_ready(self, faction_id: str) -> bool:
        """Mark a faction as ready to end the turn."""
        if faction_id in self.state.factions:
            self.state.players_ready.add(faction_id)
        # Check if all human players are ready
        human_factions = {fid for fid, f in self.state.factions.items() if f.player_id}
        return human_factions.issubset(self.state.players_ready)

    def _process_action(self, faction_id: str, action: dict) -> List[GameEvent]:
        """Process a single player action."""
        events = []
        atype = action.get("type", "")
        faction = self.state.factions.get(faction_id)
        if not faction:
            return events

        if atype == "move_army":
            events.extend(self._action_move_army(faction, action))
        elif atype == "recruit_army":
            events.extend(self._action_recruit_army(faction, action))
        elif atype == "build":
            events.extend(self._action_build(faction, action))
        elif atype == "research":
            events.extend(self._action_research(faction, action))
        elif atype == "diplomacy":
            events.extend(self._action_diplomacy(faction, action))
        elif atype == "recruit_character":
            events.extend(self._action_recruit_character(faction, action))
        elif atype == "assign_governor":
            events.extend(self._action_assign_role(faction, action, CharacterRole.GOVERNOR))
        elif atype == "assign_general":
            events.extend(self._action_assign_general(faction, action))

        return events

    def _action_move_army(self, faction: Faction, action: dict) -> List[GameEvent]:
        army = self.state.armies.get(action.get("army_id", ""))
        if not army or army.faction_id != faction.id:
            return []
        target = action.get("target_province_id", "")
        if target not in self.state.provinces:
            return []

        # Check adjacency
        current = self.state.provinces.get(army.province_id)
        if current and target in current.adjacent_province_ids:
            army.status = ArmyStatus.MARCHING
            army.target_province_id = target
            army.march_progress = 0.0
            return [self._make_event(
                "army_march",
                f"{faction.name}'s army marches toward {self.state.provinces[target].name}",
                faction_id=faction.id,
            )]
        return []

    def _action_recruit_army(self, faction: Faction, action: dict) -> List[GameEvent]:
        province = self.state.provinces.get(action.get("province_id", ""))
        if not province or province.owner_faction_id != faction.id:
            return []

        soldiers = min(action.get("soldiers", 1000), province.manpower, faction.manpower)
        if soldiers <= 0:
            return []

        infantry = int(soldiers * 0.6)
        cavalry = int(soldiers * 0.25)
        archers = soldiers - infantry - cavalry

        cost = soldiers * 0.5
        if faction.gold < cost:
            return []

        faction.gold -= cost
        faction.manpower -= soldiers
        province.manpower -= soldiers

        army = Army(
            faction_id=faction.id,
            province_id=province.id,
            soldiers=soldiers,
            infantry=infantry,
            cavalry=cavalry,
            archers=archers,
        )
        self.state.armies[army.id] = army
        return [self._make_event(
            "army_recruited",
            f"{faction.name} recruited {soldiers} soldiers in {province.name}",
            faction_id=faction.id,
            province_id=province.id,
        )]

    def _action_build(self, faction: Faction, action: dict) -> List[GameEvent]:
        province = self.state.provinces.get(action.get("province_id", ""))
        if not province or province.owner_faction_id != faction.id:
            return []

        btype_str = action.get("building_type", "")
        try:
            btype = BuildingType(btype_str)
        except ValueError:
            return []

        # Check cost
        costs = {
            BuildingType.FARM: 50, BuildingType.MARKET: 60, BuildingType.BARRACKS: 80,
            BuildingType.WALL: 100, BuildingType.ACADEMY: 120, BuildingType.TEMPLE: 70,
            BuildingType.WORKSHOP: 90, BuildingType.PORT: 110,
        }
        cost = costs.get(btype, 100)
        if faction.gold < cost:
            return []

        # Check if building already exists -> upgrade
        existing = [b for b in province.buildings if b.type == btype]
        if existing:
            existing[0].level += 1
            faction.gold -= cost * existing[0].level
        else:
            province.buildings.append(Building(type=btype, level=1))
            faction.gold -= cost

        return [self._make_event(
            "building_built",
            f"{faction.name} built {btype.value} in {province.name}",
            faction_id=faction.id,
            province_id=province.id,
        )]

    def _action_research(self, faction: Faction, action: dict) -> List[GameEvent]:
        tech_id = action.get("tech_id", "")
        if tech_id not in self.tech_tree:
            return []

        tech = self.tech_tree[tech_id]
        # Check prerequisites
        if not all(p in faction.researched_techs for p in tech.prerequisites):
            return []
        if tech_id in faction.researched_techs:
            return []

        faction.current_research = tech_id
        return [self._make_event(
            "research_started",
            f"{faction.name} begins researching {tech.name}",
            faction_id=faction.id,
        )]

    def _action_diplomacy(self, faction: Faction, action: dict) -> List[GameEvent]:
        target_id = action.get("target_faction_id", "")
        if target_id not in self.state.factions:
            return []

        dip_action = action.get("action", "")
        rel = self._get_relation(faction.id, target_id)
        target = self.state.factions[target_id]

        if dip_action == "ally":
            if rel.status == DiplomacyStatus.WAR:
                return []
            rel.status = DiplomacyStatus.ALLY
            rel.opinion = min(100, rel.opinion + 20)
            return [self._make_event(
                "alliance_formed",
                f"{faction.name} and {target.name} formed an alliance!",
                faction_id=faction.id,
            )]
        elif dip_action == "war":
            if rel.status == DiplomacyStatus.TRUCE and rel.truce_turns > 0:
                return []
            rel.status = DiplomacyStatus.WAR
            rel.opinion = max(-100, rel.opinion - 30)
            faction.stability = max(0, faction.stability - 10)
            return [self._make_event(
                "war_declared",
                f"{faction.name} declared war on {target.name}!",
                faction_id=faction.id,
            )]
        elif dip_action == "trade":
            if rel.status == DiplomacyStatus.WAR:
                return []
            rel.status = DiplomacyStatus.TRADE
            rel.trade_value = 10.0
            rel.opinion = min(100, rel.opinion + 10)
            return [self._make_event(
                "trade_established",
                f"{faction.name} established trade with {target.name}",
                faction_id=faction.id,
            )]
        elif dip_action == "truce":
            if rel.status != DiplomacyStatus.WAR:
                return []
            rel.status = DiplomacyStatus.TRUCE
            rel.truce_turns = 10
            return [self._make_event(
                "truce_signed",
                f"{faction.name} and {target.name} signed a truce",
                faction_id=faction.id,
            )]

        return []

    def _action_recruit_character(self, faction: Faction, action: dict) -> List[GameEvent]:
        province = self.state.provinces.get(action.get("province_id", ""))
        if not province or province.owner_faction_id != faction.id:
            return []

        cost = 100
        if faction.gold < cost:
            return []
        faction.gold -= cost

        char = Character(
            name=_random_chinese_name(),
            role=CharacterRole.FREE,
            faction_id=faction.id,
            province_id=province.id,
            stats=_random_stats("normal"),
        )
        self.state.characters[char.id] = char
        return [self._make_event(
            "character_recruited",
            f"{faction.name} recruited {char.name} in {province.name}",
            faction_id=faction.id,
            province_id=province.id,
        )]

    def _action_assign_role(self, faction: Faction, action: dict, role: CharacterRole) -> List[GameEvent]:
        char = self.state.characters.get(action.get("character_id", ""))
        if not char or char.faction_id != faction.id:
            return []
        province = self.state.provinces.get(action.get("province_id", ""))
        if not province or province.owner_faction_id != faction.id:
            return []
        char.role = role
        char.province_id = province.id
        return [self._make_event(
            "role_assigned",
            f"{char.name} assigned as {role.value} of {province.name}",
            faction_id=faction.id,
        )]

    def _action_assign_general(self, faction: Faction, action: dict) -> List[GameEvent]:
        char = self.state.characters.get(action.get("character_id", ""))
        if not char or char.faction_id != faction.id:
            return []
        army = self.state.armies.get(action.get("army_id", ""))
        if not army or army.faction_id != faction.id:
            return []
        char.role = CharacterRole.GENERAL
        army.general_id = char.id
        return [self._make_event(
            "general_assigned",
            f"{char.name} now commands an army",
            faction_id=faction.id,
        )]

    # -----------------------------------------------------------------------
    # Economy
    # -----------------------------------------------------------------------

    def _process_economy(self) -> List[GameEvent]:
        events = []
        for faction in self.state.factions.values():
            total_food = 0.0
            total_gold = 0.0
            total_manpower = 0

            for province in self.state.faction_provinces(faction.id):
                income = province.compute_income()
                total_food += income["food"]
                total_gold += income["gold"]
                total_manpower += income["manpower"]

                # Governor bonus
                governor = self._get_governor(province.id, faction.id)
                if governor:
                    total_gold *= (1.0 + governor.stats.politics * 0.003)
                    total_food *= (1.0 + governor.stats.politics * 0.002)

            # Tax income
            total_gold *= (1.0 + faction.tax_rate)

            # Trade income from diplomacy
            for rel in self.state.diplomacy.values():
                if rel.status == DiplomacyStatus.TRADE:
                    if rel.faction_a_id == faction.id or rel.faction_b_id == faction.id:
                        total_gold += rel.trade_value

            # Army upkeep
            for army in self.state.faction_armies(faction.id):
                upkeep_gold = army.soldiers * 0.01
                upkeep_food = army.soldiers * 0.02
                total_gold -= upkeep_gold
                total_food -= upkeep_food

            # Season effects
            if self.state.season == Season.SPRING:
                total_food *= 1.2
            elif self.state.season == Season.WINTER:
                total_food *= 0.7
                total_manpower = int(total_manpower * 0.8)

            faction.gold += total_gold
            faction.food += total_food
            faction.manpower += total_manpower

            # Starvation
            if faction.food < 0:
                faction.food = 0
                faction.stability = max(0, faction.stability - 5)
                events.append(self._make_event(
                    "famine", f"{faction.name} is running out of food!",
                    faction_id=faction.id,
                ))

            # War exhaustion decay
            faction.war_exhaustion = max(0, faction.war_exhaustion - 0.5)

        return events

    def _get_governor(self, province_id: str, faction_id: str) -> Optional[Character]:
        for c in self.state.characters.values():
            if (c.faction_id == faction_id and c.province_id == province_id
                    and c.role == CharacterRole.GOVERNOR and c.alive):
                return c
        return None

    # -----------------------------------------------------------------------
    # Army Movement
    # -----------------------------------------------------------------------

    def _process_army_movement(self) -> List[GameEvent]:
        events = []
        for army in list(self.state.armies.values()):
            if army.status != ArmyStatus.MARCHING:
                continue
            if not army.target_province_id:
                army.status = ArmyStatus.IDLE
                continue

            # March speed (terrain affects speed)
            target_province = self.state.provinces.get(army.target_province_id)
            speed = 0.5  # base: 2 turns to cross a province
            if target_province:
                if target_province.terrain == TerrainType.PLAINS:
                    speed = 0.6
                elif target_province.terrain == TerrainType.MOUNTAINS:
                    speed = 0.3
                elif target_province.terrain == TerrainType.MARSH:
                    speed = 0.35
                elif target_province.terrain == TerrainType.FOREST:
                    speed = 0.4
                elif target_province.terrain == TerrainType.RIVER:
                    speed = 0.45

            army.march_progress += speed

            if army.march_progress >= 1.0:
                army.province_id = army.target_province_id
                army.target_province_id = None
                army.march_progress = 0.0
                army.status = ArmyStatus.IDLE

                faction = self.state.factions.get(army.faction_id)
                fname = faction.name if faction else "Unknown"

                # Check if entering enemy territory
                if target_province and target_province.owner_faction_id:
                    if target_province.owner_faction_id != army.faction_id:
                        rel = self._get_relation(army.faction_id, target_province.owner_faction_id)
                        if rel.status == DiplomacyStatus.WAR:
                            army.status = ArmyStatus.BESIEGING
                            events.append(self._make_event(
                                "siege_start",
                                f"{fname}'s army begins siege of {target_province.name}",
                                faction_id=army.faction_id,
                                province_id=target_province.id,
                            ))

                events.append(self._make_event(
                    "army_arrived",
                    f"{fname}'s army arrived at {target_province.name if target_province else 'unknown'}",
                    faction_id=army.faction_id,
                ))
        return events

    # -----------------------------------------------------------------------
    # Combat
    # -----------------------------------------------------------------------

    def _process_combat(self) -> List[GameEvent]:
        events = []
        # Find provinces with armies from different factions at war
        province_armies: Dict[str, List[Army]] = {}
        for army in self.state.armies.values():
            pid = army.province_id
            if pid not in province_armies:
                province_armies[pid] = []
            province_armies[pid].append(army)

        for pid, armies in province_armies.items():
            if len(armies) < 2:
                continue

            # Group by faction
            by_faction: Dict[str, List[Army]] = {}
            for a in armies:
                if a.faction_id not in by_faction:
                    by_faction[a.faction_id] = []
                by_faction[a.faction_id].append(a)

            faction_ids = list(by_faction.keys())
            for i in range(len(faction_ids)):
                for j in range(i + 1, len(faction_ids)):
                    fid_a, fid_b = faction_ids[i], faction_ids[j]
                    rel = self._get_relation(fid_a, fid_b)
                    if rel.status != DiplomacyStatus.WAR:
                        continue

                    # Battle!
                    events.extend(self._resolve_battle(
                        by_faction[fid_a], by_faction[fid_b], pid
                    ))

        # Process sieges
        for army in list(self.state.armies.values()):
            if army.status == ArmyStatus.BESIEGING:
                events.extend(self._process_siege(army))

        return events

    def _resolve_battle(
        self, armies_a: List[Army], armies_b: List[Army], province_id: str,
    ) -> List[GameEvent]:
        events = []
        province = self.state.provinces.get(province_id)
        pname = province.name if province else "unknown"

        # Calculate total strength
        strength_a = sum(
            a.total_strength(self.state.characters.get(a.general_id))
            for a in armies_a
        )
        strength_b = sum(
            a.total_strength(self.state.characters.get(a.general_id))
            for a in armies_b
        )

        # Terrain advantage for defender
        defender_bonus = 1.0
        if province:
            if province.terrain == TerrainType.MOUNTAINS:
                defender_bonus = 1.3
            elif province.terrain == TerrainType.FOREST:
                defender_bonus = 1.15
            elif province.terrain == TerrainType.RIVER:
                defender_bonus = 1.1
            # Fortification bonus
            defender_bonus += province.fortification * 0.1

        # Determine defender (the one owning the province)
        if province and province.owner_faction_id:
            if armies_a[0].faction_id == province.owner_faction_id:
                strength_a *= defender_bonus
            elif armies_b[0].faction_id == province.owner_faction_id:
                strength_b *= defender_bonus

        # Add randomness
        strength_a *= random.uniform(0.85, 1.15)
        strength_b *= random.uniform(0.85, 1.15)

        total = strength_a + strength_b
        if total == 0:
            return events

        # Calculate casualties
        ratio_a = strength_a / total
        casualty_rate_a = (1 - ratio_a) * random.uniform(0.1, 0.3)
        casualty_rate_b = ratio_a * random.uniform(0.1, 0.3)

        fa = self.state.factions.get(armies_a[0].faction_id)
        fb = self.state.factions.get(armies_b[0].faction_id)
        fname_a = fa.name if fa else "Unknown"
        fname_b = fb.name if fb else "Unknown"

        for a in armies_a:
            losses = int(a.soldiers * casualty_rate_a)
            a.soldiers -= losses
            a.infantry = max(0, a.infantry - int(losses * 0.6))
            a.cavalry = max(0, a.cavalry - int(losses * 0.25))
            a.archers = max(0, a.archers - int(losses * 0.15))
            a.morale = max(0, a.morale - (1 - ratio_a) * 20)

        for a in armies_b:
            losses = int(a.soldiers * casualty_rate_b)
            a.soldiers -= losses
            a.infantry = max(0, a.infantry - int(losses * 0.6))
            a.cavalry = max(0, a.cavalry - int(losses * 0.25))
            a.archers = max(0, a.archers - int(losses * 0.15))
            a.morale = max(0, a.morale - ratio_a * 20)

        # Winner takes province if attacker wins
        winner = fname_a if strength_a > strength_b else fname_b
        winner_fid = armies_a[0].faction_id if strength_a > strength_b else armies_b[0].faction_id

        events.append(self._make_event(
            "battle",
            f"Battle at {pname}: {fname_a} vs {fname_b}. {winner} prevails!",
            province_id=province_id,
        ))

        # General experience
        for armies in [armies_a, armies_b]:
            for a in armies:
                gen = self.state.characters.get(a.general_id) if a.general_id else None
                if gen:
                    gen.gain_experience(30)

        # Remove destroyed armies
        for a in list(armies_a + armies_b):
            if a.soldiers <= 0:
                self.state.armies.pop(a.id, None)
                events.append(self._make_event(
                    "army_destroyed",
                    f"An army was destroyed at {pname}",
                    faction_id=a.faction_id,
                ))

        # War exhaustion
        if fa:
            fa.war_exhaustion = min(100, fa.war_exhaustion + 3)
        if fb:
            fb.war_exhaustion = min(100, fb.war_exhaustion + 3)

        return events

    def _process_siege(self, army: Army) -> List[GameEvent]:
        events = []
        province = self.state.provinces.get(army.province_id)
        if not province or not province.owner_faction_id:
            army.status = ArmyStatus.IDLE
            return events
        if province.owner_faction_id == army.faction_id:
            army.status = ArmyStatus.IDLE
            return events

        # Siege progress based on army strength vs fortification
        general = self.state.characters.get(army.general_id) if army.general_id else None
        siege_power = army.soldiers * 0.01
        if general:
            siege_power *= (1 + general.stats.intelligence * 0.005)

        fort_strength = max(1, province.fortification * 100)
        progress = siege_power / fort_strength

        if random.random() < min(0.8, progress):
            # Province captured!
            old_owner = province.owner_faction_id
            province.owner_faction_id = army.faction_id
            army.status = ArmyStatus.IDLE
            province.unrest += 20  # Conquered unrest

            faction = self.state.factions.get(army.faction_id)
            fname = faction.name if faction else "Unknown"
            events.append(self._make_event(
                "province_captured",
                f"{fname} captured {province.name}!",
                faction_id=army.faction_id,
                province_id=province.id,
            ))

            if faction:
                faction.prestige += 5
        else:
            # Siege continues, attrition
            attrition = int(army.soldiers * random.uniform(0.01, 0.03))
            army.soldiers -= attrition
            army.infantry = max(0, army.infantry - attrition)
            army.morale = max(0, army.morale - 2)

        return events

    # -----------------------------------------------------------------------
    # Research
    # -----------------------------------------------------------------------

    def _process_research(self) -> List[GameEvent]:
        events = []
        for faction in self.state.factions.values():
            if not faction.current_research:
                continue

            tech = self.tech_tree.get(faction.current_research)
            if not tech:
                faction.current_research = None
                continue

            # Research speed based on provinces and characters
            speed = 5.0
            for province in self.state.faction_provinces(faction.id):
                for b in province.buildings:
                    if b.type == BuildingType.ACADEMY:
                        speed += 3.0 * b.level

            # Strategist bonus
            for char in self.state.faction_characters(faction.id):
                if char.role == CharacterRole.STRATEGIST:
                    speed += char.stats.intelligence * 0.1

            faction.research_points += speed

            if faction.research_points >= tech.cost:
                faction.researched_techs.add(tech.id)
                faction.research_points = 0
                faction.current_research = None
                # Update tech levels
                cat = tech.category.value
                if cat in faction.tech_levels:
                    faction.tech_levels[cat] = max(faction.tech_levels[cat], tech.level)

                events.append(self._make_event(
                    "tech_researched",
                    f"{faction.name} researched {tech.name}!",
                    faction_id=faction.id,
                ))
        return events

    # -----------------------------------------------------------------------
    # Population & Development
    # -----------------------------------------------------------------------

    def _process_population(self) -> List[GameEvent]:
        events = []
        for province in self.state.provinces.values():
            if not province.owner_faction_id:
                continue

            # Population growth
            growth_rate = 0.01
            if self.state.season == Season.SPRING:
                growth_rate = 0.015
            elif self.state.season == Season.WINTER:
                growth_rate = 0.005

            # Unrest slows growth
            growth_rate *= max(0.0, 1.0 - province.unrest * 0.01)

            province.population = int(province.population * (1 + growth_rate))
            province.population = max(100, min(200000, province.population))

            # Manpower regeneration
            province.manpower = min(
                province.population // 5,
                province.manpower + int(province.population * 0.005)
            )

            # Unrest decay
            province.unrest = max(0, province.unrest - 1.5)

            # Development growth (very slow)
            if province.unrest < 10:
                province.development += 0.005

        return events

    # -----------------------------------------------------------------------
    # Diplomacy
    # -----------------------------------------------------------------------

    def _process_diplomacy_decay(self) -> None:
        for rel in self.state.diplomacy.values():
            # Truce countdown
            if rel.status == DiplomacyStatus.TRUCE:
                rel.truce_turns -= 1
                if rel.truce_turns <= 0:
                    rel.status = DiplomacyStatus.NEUTRAL
                    rel.truce_turns = 0

            # Opinion drift toward 0
            if rel.opinion > 0:
                rel.opinion = max(0, rel.opinion - 1)
            elif rel.opinion < 0:
                rel.opinion = min(0, rel.opinion + 1)

    # -----------------------------------------------------------------------
    # Random Events
    # -----------------------------------------------------------------------

    def _process_random_events(self) -> List[GameEvent]:
        events = []
        for province in self.state.provinces.values():
            if not province.owner_faction_id:
                continue
            if random.random() > self.state.random_event_chance:
                continue

            evt_def = random.choice(RANDOM_EVENTS)
            desc = evt_def["desc"].format(province=province.name)

            if "pop_loss" in evt_def:
                loss = int(province.population * evt_def["pop_loss"])
                province.population = max(100, province.population - loss)
            if "food_bonus" in evt_def:
                province.food += evt_def["food_bonus"]
            if "food_loss" in evt_def:
                province.food = max(0, province.food - evt_def["food_loss"])
            if "gold_bonus" in evt_def:
                province.gold += evt_def["gold_bonus"]
            if "gold_loss" in evt_def:
                province.gold = max(0, province.gold - evt_def["gold_loss"])
            if "unrest" in evt_def:
                province.unrest = max(0, min(100, province.unrest + evt_def["unrest"]))
            if "fort_damage" in evt_def:
                province.fortification = max(0, province.fortification - evt_def["fort_damage"])
            if evt_def.get("new_character"):
                char = Character(
                    name=_random_chinese_name(),
                    role=CharacterRole.FREE,
                    faction_id=province.owner_faction_id,
                    province_id=province.id,
                    stats=_random_stats("good"),
                )
                self.state.characters[char.id] = char
                desc += f" {char.name} joins {self.state.factions[province.owner_faction_id].name}!"
            if "stability" in evt_def:
                faction = self.state.factions.get(province.owner_faction_id)
                if faction:
                    faction.stability = min(100, max(0, faction.stability + evt_def["stability"]))
            if "prestige" in evt_def:
                faction = self.state.factions.get(province.owner_faction_id)
                if faction:
                    faction.prestige += evt_def["prestige"]

            events.append(self._make_event(
                evt_def["type"], desc,
                faction_id=province.owner_faction_id,
                province_id=province.id,
            ))
        return events

    # -----------------------------------------------------------------------
    # Characters
    # -----------------------------------------------------------------------

    def _process_characters(self) -> List[GameEvent]:
        events = []
        for char in list(self.state.characters.values()):
            if not char.alive:
                continue
            # Aging (1 year = 4 turns)
            if self.state.turn % 4 == 0:
                char.age += 1
                # Death chance increases with age
                if char.age > 60:
                    death_chance = (char.age - 60) * 0.02
                    if random.random() < death_chance:
                        char.alive = False
                        faction = self.state.factions.get(char.faction_id)
                        fname = faction.name if faction else "Unknown"
                        events.append(self._make_event(
                            "character_death",
                            f"{char.name} of {fname} has passed away at age {char.age}",
                            faction_id=char.faction_id,
                        ))
                        # If ruler dies, pick new ruler
                        if faction and faction.ruler_character_id == char.id:
                            self._succession(faction)

            # Loyalty decay for characters without faction interaction
            if char.role != CharacterRole.RULER:
                char.stats.loyalty = max(0, min(100, char.stats.loyalty - random.uniform(0, 1)))
                # Defection
                if char.stats.loyalty < 10 and random.random() < 0.1:
                    char.faction_id = None
                    char.role = CharacterRole.FREE
                    events.append(self._make_event(
                        "defection",
                        f"{char.name} has left their faction due to low loyalty!",
                    ))

        return events

    def _succession(self, faction: Faction) -> None:
        """Handle ruler succession."""
        candidates = [
            c for c in self.state.faction_characters(faction.id)
            if c.alive and c.role != CharacterRole.RULER
        ]
        if candidates:
            # Pick best candidate by total power
            best = max(candidates, key=lambda c: c.total_power())
            best.role = CharacterRole.RULER
            faction.ruler_character_id = best.id
            faction.stability = max(0, faction.stability - 15)
        else:
            faction.ruler_character_id = None
            faction.stability = 0

    # -----------------------------------------------------------------------
    # Elimination check
    # -----------------------------------------------------------------------

    def _check_elimination(self) -> List[GameEvent]:
        events = []
        for faction in list(self.state.factions.values()):
            provinces = self.state.faction_provinces(faction.id)
            if len(provinces) == 0:
                events.append(self._make_event(
                    "faction_eliminated",
                    f"{faction.name} has been eliminated!",
                    faction_id=faction.id,
                ))
                # Don't remove faction, just mark it
                faction.stability = 0
        return events

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_relation(self, fid_a: str, fid_b: str) -> DiplomacyRelation:
        key1 = f"{fid_a}_{fid_b}"
        key2 = f"{fid_b}_{fid_a}"
        if key1 in self.state.diplomacy:
            return self.state.diplomacy[key1]
        if key2 in self.state.diplomacy:
            return self.state.diplomacy[key2]
        # Create new
        rel = DiplomacyRelation(faction_a_id=fid_a, faction_b_id=fid_b)
        self.state.diplomacy[key1] = rel
        return rel

    def _make_event(
        self, event_type: str, description: str,
        faction_id: Optional[str] = None, province_id: Optional[str] = None,
    ) -> GameEvent:
        return GameEvent(
            turn=self.state.turn,
            event_type=event_type,
            description=description,
            faction_id=faction_id,
            province_id=province_id,
        )

    def _add_event(self, event_type: str, description: str, **kwargs: Any) -> None:
        evt = self._make_event(event_type, description, **kwargs)
        self.state.events.append(evt)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save_game(self, filepath: str) -> None:
        """Save game state to JSON file."""
        data = self.state.to_dict()
        # Add metadata for reload
        data["_meta"] = {
            "version": "0.1.0",
            "turn": self.state.turn,
            "year": self.state.year,
            "season": self.state.season.value,
            "map_cols": self.state.map_cols,
            "map_rows": self.state.map_rows,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_game(self, filepath: str) -> GameState:
        """Load game state from JSON file (simplified - reconstructs core state)."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = GameState()
        state.turn = data.get("turn", 0)
        state.year = data.get("_meta", {}).get("year", 190)
        state.season = Season(data.get("season", "spring"))
        state.map_cols = data.get("_meta", {}).get("map_cols", 10)
        state.map_rows = data.get("_meta", {}).get("map_rows", 8)

        # Reconstruct provinces
        for pid, pdata in data.get("provinces", {}).items():
            province = Province(
                id=pid,
                name=pdata["name"],
                x=pdata["x"],
                y=pdata["y"],
                terrain=TerrainType(pdata["terrain"]),
                owner_faction_id=pdata.get("owner_faction_id"),
                population=pdata["population"],
                food=pdata["food"],
                gold=pdata["gold"],
                manpower=pdata["manpower"],
                development=pdata["development"],
                fortification=pdata["fortification"],
                adjacent_province_ids=pdata.get("adjacent_province_ids", []),
                unrest=pdata.get("unrest", 0),
                culture=pdata.get("culture", "default"),
            )
            for bdata in pdata.get("buildings", []):
                province.buildings.append(
                    Building(type=BuildingType(bdata["type"]), level=bdata["level"])
                )
            state.provinces[pid] = province

        # Reconstruct factions
        for fid, fdata in data.get("factions", {}).items():
            faction = Faction(
                id=fid,
                name=fdata["name"],
                color=fdata["color"],
                player_id=fdata.get("player_id"),
                ruler_character_id=fdata.get("ruler_character_id"),
                gold=fdata["gold"],
                food=fdata["food"],
                manpower=fdata["manpower"],
                stability=fdata.get("stability", 50),
                legitimacy=fdata.get("legitimacy", 50),
                prestige=fdata.get("prestige", 0),
                war_exhaustion=fdata.get("war_exhaustion", 0),
                tax_rate=fdata.get("tax_rate", 0.1),
            )
            faction.tech_levels = fdata.get("tech_levels", {"military": 1, "economy": 1, "culture": 1, "diplomacy": 1})
            faction.researched_techs = set(fdata.get("researched_techs", []))
            faction.current_research = fdata.get("current_research")
            state.factions[fid] = faction

        # Reconstruct characters
        for cid, cdata in data.get("characters", {}).items():
            char = Character(
                id=cid,
                name=cdata["name"],
                role=CharacterRole(cdata["role"]),
                faction_id=cdata.get("faction_id"),
                province_id=cdata.get("province_id"),
                stats=CharacterStats(**cdata.get("stats", {})),
                level=cdata.get("level", 1),
                experience=cdata.get("experience", 0),
                skills=cdata.get("skills", []),
                alive=cdata.get("alive", True),
                age=cdata.get("age", 25),
            )
            state.characters[cid] = char

        # Reconstruct armies
        for aid, adata in data.get("armies", {}).items():
            army = Army(
                id=aid,
                faction_id=adata["faction_id"],
                general_id=adata.get("general_id"),
                province_id=adata["province_id"],
                soldiers=adata["soldiers"],
                morale=adata.get("morale", 100),
                status=ArmyStatus(adata.get("status", "idle")),
                infantry=adata.get("infantry", 0),
                cavalry=adata.get("cavalry", 0),
                archers=adata.get("archers", 0),
            )
            state.armies[aid] = army

        # Reconstruct diplomacy
        for key, ddata in data.get("diplomacy", {}).items():
            rel = DiplomacyRelation(
                faction_a_id=ddata["faction_a_id"],
                faction_b_id=ddata["faction_b_id"],
                status=DiplomacyStatus(ddata["status"]),
                opinion=ddata.get("opinion", 0),
                trade_value=ddata.get("trade_value", 0),
                truce_turns=ddata.get("truce_turns", 0),
            )
            state.diplomacy[key] = rel

        self.state = state
        return state
