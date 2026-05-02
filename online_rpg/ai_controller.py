"""AI Controller - Autonomous behavior for AI-controlled factions.

Implements strategic decision-making for AI factions inspired by
Romance of the Three Kingdoms 14 and Europa Universalis 4 AI systems.
Each AI faction evaluates its situation and submits actions each turn.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .models import (
    ArmyStatus,
    BuildingType,
    CharacterRole,
    DiplomacyStatus,
    Faction,
    Province,
)
from .game_engine import GameEngine, GameState
from .tech_tree import get_available_techs


class AIPersonality:
    """Defines an AI faction's behavioral tendencies."""

    def __init__(
        self,
        aggression: float = 0.5,
        expansion: float = 0.5,
        diplomacy: float = 0.5,
        economy: float = 0.5,
        research: float = 0.5,
    ):
        self.aggression = aggression    # 0 = peaceful, 1 = warlike
        self.expansion = expansion      # 0 = defensive, 1 = expansionist
        self.diplomacy = diplomacy      # 0 = isolationist, 1 = diplomatic
        self.economy = economy          # 0 = military focus, 1 = economy focus
        self.research = research        # 0 = ignore tech, 1 = tech priority

    @classmethod
    def random(cls) -> "AIPersonality":
        return cls(
            aggression=random.uniform(0.2, 0.8),
            expansion=random.uniform(0.2, 0.8),
            diplomacy=random.uniform(0.2, 0.8),
            economy=random.uniform(0.3, 0.9),
            research=random.uniform(0.3, 0.8),
        )

    @classmethod
    def warlike(cls) -> "AIPersonality":
        return cls(aggression=0.85, expansion=0.7, diplomacy=0.2, economy=0.4, research=0.5)

    @classmethod
    def peaceful(cls) -> "AIPersonality":
        return cls(aggression=0.15, expansion=0.3, diplomacy=0.8, economy=0.8, research=0.7)

    @classmethod
    def balanced(cls) -> "AIPersonality":
        return cls(aggression=0.5, expansion=0.5, diplomacy=0.5, economy=0.6, research=0.6)


class AIController:
    """Controls all AI factions each turn."""

    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.personalities: Dict[str, AIPersonality] = {}
        self._threat_memory: Dict[str, Dict[str, float]] = {}  # faction -> {enemy -> threat}

    def initialize_personalities(self) -> None:
        """Assign random personalities to AI factions."""
        personality_types = [
            AIPersonality.warlike,
            AIPersonality.peaceful,
            AIPersonality.balanced,
            AIPersonality.random,
        ]
        for fid, faction in self.engine.state.factions.items():
            if faction.player_id is None:  # AI faction
                creator = random.choice(personality_types)
                self.personalities[fid] = creator()
                self._threat_memory[fid] = {}

    def process_ai_turns(self) -> None:
        """Generate and submit actions for all AI factions."""
        state = self.engine.state
        for fid, faction in state.factions.items():
            if faction.player_id is not None:
                continue  # Skip human players
            if fid not in self.personalities:
                self.personalities[fid] = AIPersonality.random()
                self._threat_memory[fid] = {}

            personality = self.personalities[fid]
            actions = self._decide_actions(faction, personality)
            for action in actions:
                self.engine.submit_action(fid, action)

    def _decide_actions(self, faction: Faction, p: AIPersonality) -> List[dict]:
        """Decide what actions an AI faction should take this turn."""
        actions: List[dict] = []
        state = self.engine.state

        provinces = state.faction_provinces(faction.id)
        characters = state.faction_characters(faction.id)
        armies = state.faction_armies(faction.id)

        if not provinces:
            return actions

        # 1. Economy: Build in provinces
        actions.extend(self._economy_actions(faction, provinces, p))

        # 2. Military: Recruit and move armies
        actions.extend(self._military_actions(faction, provinces, armies, p))

        # 3. Diplomacy: Manage relations
        actions.extend(self._diplomacy_actions(faction, p))

        # 4. Research: Pick technology
        actions.extend(self._research_actions(faction, p))

        # 5. Characters: Recruit if needed
        actions.extend(self._character_actions(faction, provinces, characters, p))

        return actions

    # -------------------------------------------------------------------
    # Economy
    # -------------------------------------------------------------------

    def _economy_actions(
        self, faction: Faction, provinces: List[Province], p: AIPersonality
    ) -> List[dict]:
        actions = []

        # Sort provinces by development (invest in best first)
        sorted_provs = sorted(provinces, key=lambda pv: pv.development, reverse=True)

        budget = faction.gold * 0.4  # Spend up to 40% on buildings

        for province in sorted_provs[:3]:  # Max 3 builds per turn
            if budget < 50:
                break

            # Decide what to build based on personality and needs
            if faction.food < 200 or (p.economy > 0.5 and random.random() < 0.4):
                btype = "farm"
                cost = 50
            elif faction.gold < 300 and random.random() < 0.3:
                btype = "market"
                cost = 60
            elif p.aggression > 0.5 and faction.manpower < 3000:
                btype = "barracks"
                cost = 80
            elif p.research > 0.5 and random.random() < 0.3:
                btype = "academy"
                cost = 120
            elif province.fortification < 2 and p.expansion < 0.5:
                btype = "wall"
                cost = 100
            else:
                btype = random.choice(["farm", "market", "workshop"])
                cost = 60

            if faction.gold >= cost:
                actions.append({
                    "type": "build",
                    "province_id": province.id,
                    "building_type": btype,
                })
                budget -= cost

        return actions

    # -------------------------------------------------------------------
    # Military
    # -------------------------------------------------------------------

    def _military_actions(
        self,
        faction: Faction,
        provinces: List[Province],
        armies: list,
        p: AIPersonality,
    ) -> List[dict]:
        actions = []
        state = self.engine.state

        # Recruit if we have few armies or are at war
        at_war = self._is_at_war(faction.id)
        total_soldiers = sum(a.soldiers for a in armies)

        if (total_soldiers < 2000 or (at_war and total_soldiers < 5000)) and faction.gold > 300:
            # Find province with most manpower
            best_prov = max(provinces, key=lambda pv: pv.manpower) if provinces else None
            if best_prov and best_prov.manpower > 500:
                recruit_count = min(1500, best_prov.manpower, faction.manpower)
                if recruit_count > 300:
                    actions.append({
                        "type": "recruit_army",
                        "province_id": best_prov.id,
                        "soldiers": recruit_count,
                    })

        # Move idle armies
        for army in armies:
            if army.status != ArmyStatus.IDLE:
                continue

            current_prov = state.provinces.get(army.province_id)
            if not current_prov:
                continue

            target = self._choose_army_target(faction, army, current_prov, p)
            if target:
                actions.append({
                    "type": "move_army",
                    "army_id": army.id,
                    "target_province_id": target,
                })

        return actions

    def _choose_army_target(
        self, faction: Faction, army, current_prov: Province, p: AIPersonality
    ) -> Optional[str]:
        """Choose where to move an army."""
        state = self.engine.state
        candidates: List[Tuple[str, float]] = []

        for adj_id in current_prov.adjacent_province_ids:
            adj = state.provinces.get(adj_id)
            if not adj:
                continue

            score = 0.0

            if adj.owner_faction_id == faction.id:
                # Own territory - low priority unless defending
                score = 0.1
            elif adj.owner_faction_id == "" or adj.owner_faction_id is None:
                # Unowned - expand
                score = 0.5 * p.expansion
            else:
                # Enemy territory
                rel = self.engine._get_relation(faction.id, adj.owner_faction_id)
                if rel.status == DiplomacyStatus.WAR:
                    # At war - high priority to attack
                    score = 1.0 * p.aggression
                    # Prefer weaker targets
                    enemy_armies = state.get_armies_at_province(adj_id) if hasattr(state, 'get_armies_at_province') else []
                    if not enemy_armies:
                        score += 0.3
                elif rel.status in (DiplomacyStatus.NEUTRAL, DiplomacyStatus.TRADE):
                    score = 0.05  # Low score - don't attack friends

            if score > 0.2:
                candidates.append((adj_id, score))

        if not candidates:
            return None

        # Weighted random selection
        candidates.sort(key=lambda c: c[1], reverse=True)
        if random.random() < 0.7:
            return candidates[0][0]
        return random.choice(candidates)[0]

    def _is_at_war(self, faction_id: str) -> bool:
        for rel in self.engine.state.diplomacy.values():
            if rel.status == DiplomacyStatus.WAR:
                if rel.faction_a_id == faction_id or rel.faction_b_id == faction_id:
                    return True
        return False

    # -------------------------------------------------------------------
    # Diplomacy
    # -------------------------------------------------------------------

    def _diplomacy_actions(self, faction: Faction, p: AIPersonality) -> List[dict]:
        actions = []
        state = self.engine.state

        for other_id, other in state.factions.items():
            if other_id == faction.id:
                continue

            rel = self.engine._get_relation(faction.id, other_id)

            # Consider declaring war
            if rel.status == DiplomacyStatus.NEUTRAL and p.aggression > 0.6:
                # Check if we're stronger
                our_soldiers = sum(a.soldiers for a in state.faction_armies(faction.id))
                their_soldiers = sum(a.soldiers for a in state.faction_armies(other_id))
                our_provs = len(state.faction_provinces(faction.id))
                their_provs = len(state.faction_provinces(other_id))

                if our_soldiers > their_soldiers * 1.5 and our_provs >= their_provs:
                    if random.random() < p.aggression * 0.15:
                        actions.append({
                            "type": "diplomacy",
                            "target_faction_id": other_id,
                            "action": "war",
                        })
                        continue

            # Consider alliances
            if rel.status == DiplomacyStatus.NEUTRAL and p.diplomacy > 0.5:
                if rel.opinion > 10 and random.random() < p.diplomacy * 0.1:
                    actions.append({
                        "type": "diplomacy",
                        "target_faction_id": other_id,
                        "action": "ally",
                    })
                    continue

            # Consider trade
            if rel.status == DiplomacyStatus.NEUTRAL and p.economy > 0.5:
                if random.random() < p.economy * 0.1:
                    actions.append({
                        "type": "diplomacy",
                        "target_faction_id": other_id,
                        "action": "trade",
                    })

            # Consider peace if exhausted
            if rel.status == DiplomacyStatus.WAR and faction.war_exhaustion > 50:
                if random.random() < 0.3:
                    actions.append({
                        "type": "diplomacy",
                        "target_faction_id": other_id,
                        "action": "truce",
                    })

        return actions

    # -------------------------------------------------------------------
    # Research
    # -------------------------------------------------------------------

    def _research_actions(self, faction: Faction, p: AIPersonality) -> List[dict]:
        if faction.current_research:
            return []

        available = get_available_techs(faction.researched_techs)
        if not available:
            return []

        # Score techs by personality
        scored = []
        for tech in available:
            score = 0.0
            cat = tech.category.value
            if cat == "military":
                score = p.aggression * 0.8 + (1 - p.economy) * 0.2
            elif cat == "economy":
                score = p.economy * 0.8 + 0.2
            elif cat == "culture":
                score = p.research * 0.6 + 0.3
            elif cat == "diplomacy":
                score = p.diplomacy * 0.7 + 0.1

            # Prefer lower cost (faster completion)
            score += (1.0 / max(1, tech.cost)) * 50
            scored.append((tech, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_tech = scored[0][0]

        return [{"type": "research", "tech_id": best_tech.id}]

    # -------------------------------------------------------------------
    # Characters
    # -------------------------------------------------------------------

    def _character_actions(
        self,
        faction: Faction,
        provinces: List[Province],
        characters: list,
        p: AIPersonality,
    ) -> List[dict]:
        actions = []

        # Recruit characters if we have few
        generals = [c for c in characters if c.role == CharacterRole.GENERAL]
        if len(generals) < 2 and faction.gold > 150 and provinces:
            prov = random.choice(provinces)
            actions.append({
                "type": "recruit_character",
                "province_id": prov.id,
            })

        return actions
