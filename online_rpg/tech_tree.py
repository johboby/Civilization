"""Technology tree definitions.

Provides a pre-built tech tree inspired by EU4 and RTK14
covering military, economy, culture, and diplomacy branches.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import TechCategory, Technology

# ---------------------------------------------------------------------------
# Technology definitions
# ---------------------------------------------------------------------------

_TECH_DEFS: List[dict] = [
    # -- Military branch --
    {"id": "mil_1", "name": "Bronze Weapons", "cat": TechCategory.MILITARY, "level": 1,
     "cost": 100, "prereqs": [], "effects": {"army_attack": 0.1},
     "desc": "Basic bronze weaponry for infantry."},
    {"id": "mil_2", "name": "Iron Forging", "cat": TechCategory.MILITARY, "level": 2,
     "cost": 200, "prereqs": ["mil_1"], "effects": {"army_attack": 0.15, "army_defense": 0.05},
     "desc": "Iron weapons and basic armor."},
    {"id": "mil_3", "name": "Cavalry Tactics", "cat": TechCategory.MILITARY, "level": 2,
     "cost": 250, "prereqs": ["mil_1"], "effects": {"cavalry_power": 0.2},
     "desc": "Organized cavalry maneuvers."},
    {"id": "mil_4", "name": "Siege Engineering", "cat": TechCategory.MILITARY, "level": 3,
     "cost": 400, "prereqs": ["mil_2"], "effects": {"siege_power": 0.3},
     "desc": "Siege towers and battering rams."},
    {"id": "mil_5", "name": "Steel Weapons", "cat": TechCategory.MILITARY, "level": 3,
     "cost": 500, "prereqs": ["mil_2"], "effects": {"army_attack": 0.2, "army_defense": 0.1},
     "desc": "High-quality steel weaponry."},
    {"id": "mil_6", "name": "Advanced Formations", "cat": TechCategory.MILITARY, "level": 4,
     "cost": 700, "prereqs": ["mil_3", "mil_5"], "effects": {"army_attack": 0.15, "army_defense": 0.15},
     "desc": "Complex military formations and discipline."},
    {"id": "mil_7", "name": "War Machines", "cat": TechCategory.MILITARY, "level": 5,
     "cost": 1000, "prereqs": ["mil_4", "mil_6"], "effects": {"siege_power": 0.5, "army_attack": 0.1},
     "desc": "Trebuchets and advanced siege weaponry."},

    # -- Economy branch --
    {"id": "eco_1", "name": "Agriculture", "cat": TechCategory.ECONOMY, "level": 1,
     "cost": 100, "prereqs": [], "effects": {"food_production": 0.15},
     "desc": "Organized farming techniques."},
    {"id": "eco_2", "name": "Irrigation", "cat": TechCategory.ECONOMY, "level": 2,
     "cost": 200, "prereqs": ["eco_1"], "effects": {"food_production": 0.2},
     "desc": "Canal and irrigation systems."},
    {"id": "eco_3", "name": "Currency", "cat": TechCategory.ECONOMY, "level": 2,
     "cost": 200, "prereqs": ["eco_1"], "effects": {"gold_production": 0.15, "trade_efficiency": 0.1},
     "desc": "Standardized currency for trade."},
    {"id": "eco_4", "name": "Road Network", "cat": TechCategory.ECONOMY, "level": 3,
     "cost": 350, "prereqs": ["eco_3"], "effects": {"trade_efficiency": 0.2, "march_speed": 0.1},
     "desc": "Built roads connecting provinces."},
    {"id": "eco_5", "name": "Banking", "cat": TechCategory.ECONOMY, "level": 3,
     "cost": 400, "prereqs": ["eco_3"], "effects": {"gold_production": 0.2, "tax_efficiency": 0.15},
     "desc": "Financial institutions and lending."},
    {"id": "eco_6", "name": "Advanced Agriculture", "cat": TechCategory.ECONOMY, "level": 4,
     "cost": 600, "prereqs": ["eco_2", "eco_4"], "effects": {"food_production": 0.25, "population_growth": 0.1},
     "desc": "Crop rotation and advanced farming."},
    {"id": "eco_7", "name": "Manufacturing", "cat": TechCategory.ECONOMY, "level": 5,
     "cost": 900, "prereqs": ["eco_5", "eco_6"], "effects": {"gold_production": 0.3, "development_speed": 0.2},
     "desc": "Proto-industrial manufacturing."},

    # -- Culture branch --
    {"id": "cul_1", "name": "Writing", "cat": TechCategory.CULTURE, "level": 1,
     "cost": 100, "prereqs": [], "effects": {"research_speed": 0.1},
     "desc": "Development of written language."},
    {"id": "cul_2", "name": "Philosophy", "cat": TechCategory.CULTURE, "level": 2,
     "cost": 200, "prereqs": ["cul_1"], "effects": {"stability_bonus": 0.1, "research_speed": 0.1},
     "desc": "Schools of philosophical thought."},
    {"id": "cul_3", "name": "Education", "cat": TechCategory.CULTURE, "level": 2,
     "cost": 250, "prereqs": ["cul_1"], "effects": {"research_speed": 0.15, "character_xp": 0.1},
     "desc": "Formal education systems."},
    {"id": "cul_4", "name": "Literature", "cat": TechCategory.CULTURE, "level": 3,
     "cost": 350, "prereqs": ["cul_2"], "effects": {"prestige_gain": 0.2, "stability_bonus": 0.05},
     "desc": "Great works of literature."},
    {"id": "cul_5", "name": "Academy", "cat": TechCategory.CULTURE, "level": 4,
     "cost": 500, "prereqs": ["cul_3", "cul_4"], "effects": {"research_speed": 0.25, "character_xp": 0.15},
     "desc": "Institutions of higher learning."},
    {"id": "cul_6", "name": "Renaissance", "cat": TechCategory.CULTURE, "level": 5,
     "cost": 800, "prereqs": ["cul_5"], "effects": {"research_speed": 0.2, "prestige_gain": 0.3, "development_speed": 0.1},
     "desc": "A cultural and intellectual rebirth."},

    # -- Diplomacy branch --
    {"id": "dip_1", "name": "Envoys", "cat": TechCategory.DIPLOMACY, "level": 1,
     "cost": 100, "prereqs": [], "effects": {"diplomacy_power": 0.1},
     "desc": "Basic diplomatic envoy system."},
    {"id": "dip_2", "name": "Trade Agreements", "cat": TechCategory.DIPLOMACY, "level": 2,
     "cost": 200, "prereqs": ["dip_1"], "effects": {"trade_efficiency": 0.15, "diplomacy_power": 0.05},
     "desc": "Formal trade agreements between nations."},
    {"id": "dip_3", "name": "Alliances", "cat": TechCategory.DIPLOMACY, "level": 2,
     "cost": 250, "prereqs": ["dip_1"], "effects": {"diplomacy_power": 0.15, "ally_trust": 0.1},
     "desc": "Formal military alliances."},
    {"id": "dip_4", "name": "Espionage", "cat": TechCategory.DIPLOMACY, "level": 3,
     "cost": 400, "prereqs": ["dip_2"], "effects": {"spy_efficiency": 0.2, "counter_espionage": 0.1},
     "desc": "Organized intelligence networks."},
    {"id": "dip_5", "name": "Vassalage", "cat": TechCategory.DIPLOMACY, "level": 3,
     "cost": 400, "prereqs": ["dip_3"], "effects": {"diplomacy_power": 0.2, "vassal_income": 0.15},
     "desc": "Formal vassal-suzerain relationships."},
    {"id": "dip_6", "name": "Imperial Authority", "cat": TechCategory.DIPLOMACY, "level": 5,
     "cost": 900, "prereqs": ["dip_4", "dip_5"], "effects": {"diplomacy_power": 0.3, "prestige_gain": 0.2},
     "desc": "Supreme diplomatic authority."},
]


def build_tech_tree() -> Dict[str, Technology]:
    """Build the complete technology tree.

    Returns:
        Dict mapping tech_id -> Technology.
    """
    tree: Dict[str, Technology] = {}
    for d in _TECH_DEFS:
        tech = Technology(
            id=d["id"],
            name=d["name"],
            category=d["cat"],
            level=d["level"],
            cost=d["cost"],
            description=d["desc"],
            prerequisites=d["prereqs"],
            effects=d["effects"],
        )
        tree[tech.id] = tech
    return tree


def get_available_techs(
    researched: set,
    category: Optional[TechCategory] = None,
) -> List[Technology]:
    """Get technologies available for research.

    Args:
        researched: Set of already-researched tech IDs.
        category: Optional filter by category.

    Returns:
        List of available Technology objects.
    """
    tree = build_tech_tree()
    available = []
    for tech in tree.values():
        if tech.id in researched:
            continue
        if category and tech.category != category:
            continue
        if all(p in researched for p in tech.prerequisites):
            available.append(tech)
    return available
