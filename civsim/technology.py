"""
Civilization simulation system - Technology management module

This module is refactored from tech_tree.py to provide a clearer technology management interface.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class TechLevel(Enum):
    """Technology level enum"""
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    TOP = 4


@dataclass
class TechInfo:
    """Technology information data class"""
    name: str
    level: int
    cost: int
    description: str
    prerequisites: List[str] = field(default_factory=list)
    category: str = "other"


@dataclass
class TechBonus:
    """Technology bonus data class"""
    resources: float = 1.0
    strength: float = 1.0
    defense: float = 1.0
    research_speed: float = 1.0
    diplomacy: float = 1.0
    territory_growth: float = 1.0
    territory_value: float = 1.0
    trade_efficiency: float = 1.0
    infrastructure: float = 1.0
    population_growth: float = 1.0
    tactical_advantage: float = 1.0
    tech_discovery: float = 1.0
    stability: float = 1.0
    innovation: float = 1.0
    health: float = 1.0
    energy_efficiency: float = 1.0
    resource_acquisition: float = 1.0
    global_influence: float = 1.0
    decision_quality: float = 1.0


class TechnologyManager:
    """Technology manager, handles tech tree and technology bonuses"""

    def __init__(self):
        """Initialize technology manager"""
        self.techs = self._initialize_tech_tree()
        self.tech_effects = self._initialize_tech_effects()

    def _initialize_tech_tree(self) -> Dict[str, TechInfo]:
        """Initialize technology tree

        Returns:
            Technology information dictionary
        """
        return {
            # Basic technologies
            "agriculture": TechInfo(
                name="agriculture",
                level=1,
                cost=50,
                description="Agriculture technology",
                prerequisites=[],
                category="Agriculture"
            ),
            "military": TechInfo(
                name="military",
                level=1,
                cost=80,
                description="Military technology",
                prerequisites=[],
                category="Military"
            ),
            "trade": TechInfo(
                name="trade",
                level=1,
                cost=60,
                description="Trade technology",
                prerequisites=[],
                category="Trade"
            ),
            "science": TechInfo(
                name="science",
                level=1,
                cost=100,
                description="Science technology",
                prerequisites=[],
                category="Science"
            ),

            # Intermediate technologies
            "irrigation": TechInfo(
                name="irrigation",
                level=2,
                cost=200,
                description="Irrigation system",
                prerequisites=["agriculture"],
                category="Agriculture"
            ),
            "fortification": TechInfo(
                name="fortification",
                level=2,
                cost=250,
                description="Fortifications",
                prerequisites=["military"],
                category="Military"
            ),
            "currency": TechInfo(
                name="currency",
                level=2,
                cost=220,
                description="Currency system",
                prerequisites=["trade"],
                category="Trade"
            ),
            "engineering": TechInfo(
                name="engineering",
                level=2,
                cost=300,
                description="Engineering",
                prerequisites=["science"],
                category="Science"
            ),

            # Advanced technologies
            "industrial_agriculture": TechInfo(
                name="industrial_agriculture",
                level=3,
                cost=800,
                description="Industrial agriculture",
                prerequisites=["irrigation", "engineering"],
                category="Agriculture"
            ),
            "advanced_tactics": TechInfo(
                name="advanced_tactics",
                level=3,
                cost=900,
                description="Advanced tactics",
                prerequisites=["fortification", "engineering"],
                category="Military"
            ),
            "global_trade": TechInfo(
                name="global_trade",
                level=3,
                cost=750,
                description="Global trade",
                prerequisites=["currency", "engineering"],
                category="Trade"
            ),
            "advanced_science": TechInfo(
                name="advanced_science",
                level=3,
                cost=1000,
                description="Advanced science",
                prerequisites=["engineering", "science"],
                category="Science"
            ),

            # Top-level technologies
            "genetic_engineering": TechInfo(
                name="genetic_engineering",
                level=4,
                cost=2000,
                description="Genetic engineering",
                prerequisites=["industrial_agriculture", "advanced_science"],
                category="Top-level Technology"
            ),
            "nuclear_technology": TechInfo(
                name="nuclear_technology",
                level=4,
                cost=2500,
                description="Nuclear technology",
                prerequisites=["advanced_tactics", "advanced_science"],
                category="Top-level Technology"
            ),
            "space_colonization": TechInfo(
                name="space_colonization",
                level=4,
                cost=3000,
                description="Space colonization",
                prerequisites=["global_trade", "advanced_science"],
                category="Top-level Technology"
            ),
            "artificial_intelligence": TechInfo(
                name="artificial_intelligence",
                level=4,
                cost=3500,
                description="Artificial intelligence",
                prerequisites=["advanced_science"],
                category="Top-level Technology"
            )
        }

    def _initialize_tech_effects(self) -> Dict[str, TechBonus]:
        """Initialize technology effects

        Returns:
            Technology bonus dictionary
        """
        return {
            "agriculture": TechBonus(resources=1.1, territory_growth=1.05),
            "military": TechBonus(strength=1.15, defense=1.1),
            "trade": TechBonus(resources=1.08, diplomacy=1.15),
            "science": TechBonus(research_speed=1.2, tech_discovery=1.1),
            "irrigation": TechBonus(resources=1.15, territory_value=1.1),
            "fortification": TechBonus(defense=1.2, stability=1.1),
            "currency": TechBonus(resources=1.12, trade_efficiency=1.15),
            "engineering": TechBonus(research_speed=1.15, infrastructure=1.2),
            "industrial_agriculture": TechBonus(resources=1.3, population_growth=1.2),
            "advanced_tactics": TechBonus(strength=1.3, tactical_advantage=1.25),
            "global_trade": TechBonus(resources=1.25, diplomacy=1.2),
            "advanced_science": TechBonus(research_speed=1.35, innovation=1.3),
            "genetic_engineering": TechBonus(
                resources=1.5, population_growth=1.35, health=1.3
            ),
            "nuclear_technology": TechBonus(
                strength=1.6, defense=1.4, energy_efficiency=1.5
            ),
            "space_colonization": TechBonus(
                territory_growth=1.8, resource_acquisition=1.6, global_influence=1.5
            ),
            "artificial_intelligence": TechBonus(
                research_speed=1.8, innovation=1.6, decision_quality=1.7
            )
        }

    def get_tech_info(self, tech_name: str) -> Optional[TechInfo]:
        """Get technology information

        Args:
            tech_name: Technology name

        Returns:
            Technology information, None if not exists
        """
        return self.techs.get(tech_name)

    def can_research(self, tech_name: str, current_techs: Dict[str, int]) -> tuple[bool, str]:
        """Check if specified technology can be researched

        Args:
            tech_name: Technology name
            current_techs: Current researched technologies and their levels

        Returns:
            (Can research, Reason explanation)
        """
        if tech_name not in self.techs:
            return False, "Technology does not exist"

        tech = self.techs[tech_name]

        # Check prerequisite technologies
        for prerequisite in tech.prerequisites:
            if prerequisite not in current_techs or current_techs[prerequisite] < 1:
                return False, f"Missing prerequisite technology: {prerequisite}"

        return True, "Can be researched"

    def get_research_cost(self, tech_name: str, current_level: int) -> int:
        """Calculate research cost

        Args:
            tech_name: Technology name
            current_level: Current level

        Returns:
            Research cost
        """
        if tech_name not in self.techs:
            return 0

        base_cost = self.techs[tech_name].cost
        cost_multiplier = 1.5 ** (current_level - 1)

        return int(base_cost * cost_multiplier)

    def calculate_bonuses(self, current_techs: Dict[str, int]) -> TechBonus:
        """Calculate technology bonuses

        Args:
            current_techs: Current researched technologies and their levels

        Returns:
            Technology bonus object
        """
        bonus = TechBonus()

        for tech_name, level in current_techs.items():
            if tech_name in self.tech_effects and level > 0:
                tech_bonus = self.tech_effects[tech_name]
                # Accumulate bonuses for each technology level
                for field_name in TechBonus.__dataclass_fields__:
                    current_value = getattr(bonus, field_name)
                    tech_value = getattr(tech_bonus, field_name)
                    setattr(bonus, field_name, current_value + (tech_value - 1.0) * level)

        return bonus

    def get_available_techs(self, current_techs: Dict[str, int]) -> List[Dict]:
        """Get list of researchable technologies

        Args:
            current_techs: Current researched technologies and their levels

        Returns:
            List of researchable technology information
        """
        available = []

        for tech_name, tech_info in self.techs.items():
            current_level = current_techs.get(tech_name, 0)

            # If current level is lower than tech level, check if it can be researched
            if current_level < tech_info.level:
                can_research, _ = self.can_research(tech_name, current_techs)
                if can_research:
                    cost = self.get_research_cost(tech_name, current_level + 1)
                    available.append({
                        "name": tech_name,
                        "level": tech_info.level,
                        "cost": cost,
                        "description": tech_info.description,
                        "category": tech_info.category
                    })

        return available

    def get_tech_category(self, tech_name: str) -> str:
        """Get technology category

        Args:
            tech_name: Technology name

        Returns:
            Technology category name
        """
        if tech_name in ["agriculture", "irrigation", "industrial_agriculture"]:
            return "Agriculture"
        elif tech_name in ["military", "fortification", "advanced_tactics"]:
            return "Military"
        elif tech_name in ["trade", "currency", "global_trade"]:
            return "Trade"
        elif tech_name in ["science", "engineering", "advanced_science"]:
            return "Science"
        elif tech_name in ["genetic_engineering", "nuclear_technology",
                         "space_colonization", "artificial_intelligence"]:
            return "Top-level Technology"
        else:
            return "Other"

    def get_impact_description(self, tech_name: str) -> str:
        """Get technology impact description

        Args:
            tech_name: Technology name

        Returns:
            Technology impact description
        """
        if tech_name not in self.tech_effects:
            return "No special effects"

        effects = self.tech_effects[tech_name]
        descriptions = []

        for field_name in TechBonus.__dataclass_fields__:
            value = getattr(effects, field_name)

            # Only describe attributes with bonuses
            if value != 1.0:
                field_translations = {
                    "resources": "Resource Output",
                    "strength": "Military Strength",
                    "defense": "Defense Capability",
                    "research_speed": "Research Speed",
                    "population_growth": "Population Growth",
                    "territory_growth": "Territory Expansion",
                    "health": "Health Level",
                    "energy_efficiency": "Energy Efficiency",
                    "resource_acquisition": "Resource Acquisition",
                    "global_influence": "Global Influence",
                    "decision_quality": "Decision Quality"
                }

                translation = field_translations.get(field_name, field_name)
                bonus_percent = (value - 1.0) * 100
                descriptions.append(f"{translation}+{bonus_percent:.0f}%")

        return ",".join(descriptions)

    def get_tech_tree_summary(self) -> str:
        """Get technology tree summary

        Returns:
            Technology tree text summary
        """
        summary = "Technology Tree System\n"
        summary += "=" * 50 + "\n"

        # Group by level
        techs_by_level = {}
        for tech_name, tech_info in self.techs.items():
            level = tech_info.level
            techs_by_level.setdefault(level, []).append((tech_name, tech_info))

        for level in sorted(techs_by_level.keys()):
            summary += f"\nLevel {level} Technologies:\n"
            for tech_name, tech_info in techs_by_level[level]:
                prereqs = ", ".join(tech_info.prerequisites) if tech_info.prerequisites else "None"
                summary += f"  - {tech_name}: {tech_info.description} (Prerequisites: {prereqs})\n"

        return summary


if __name__ == "__main__":
    # Test technology manager
    manager = TechnologyManager()

    # Test initial technology state
    initial_techs = {"agriculture": 1, "military": 1, "trade": 1}

    # Get available technologies
    available = manager.get_available_techs(initial_techs)
    print("Available technologies:")
    for tech in available:
        print(f"  {tech['name']}: {tech['description']} (Cost: {tech['cost']})")

    # Calculate technology bonuses
    bonuses = manager.calculate_bonuses(initial_techs)
    print("\nCurrent technology bonuses:")
    for field_name in TechBonus.__dataclass_fields__:
        value = getattr(bonuses, field_name)
        if value != 1.0:
            print(f"  {field_name}: {value:.2f}x")

    # Print technology tree summary
    print("\n" + manager.get_tech_tree_summary())
