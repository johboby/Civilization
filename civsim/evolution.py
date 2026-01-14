"""Advanced evolution module for civilization simulation.

This module implements sophisticated reasoning and evolution capabilities
based on game theory, complex systems theory, and modern evolution algorithms,
providing CivilizationAgent with more intelligent decision-making and richer evolution paths.
"""
import copy
import random
from collections import defaultdict, deque
from math import exp, sin, pi
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .logger import get_logger


class AdvancedEvolution:
    """Advanced evolution engine providing complex decision-making and evolution capabilities."""

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the advanced evolution engine.

        Args:
            config: Simulation configuration object with various parameter settings.
        """
        self.config = config
        self.strategy_history: Dict[int, deque] = defaultdict(deque)
        self.memory_window = (
            getattr(config, 'MEMORY_WINDOW_SIZE', 10) if config else 10
        )
        self.exploration_rate = (
            getattr(config, 'STRATEGY_EXPLORATION_PROBABILITY', 0.05) if config else 0.05
        )
        self.metacognition_rate = (
            getattr(config, 'METACOGNITION_LEARNING_RATE', 0.05) if config else 0.05
        )
        self.enable_metacognition = (
            getattr(config, 'ENABLE_METACOGNITION', True) if config else True
        )
        self.logger = get_logger(__name__)

    def calculate_strategy_tendency(
        self,
        agent: Any,
        neighbors: Dict[Any, float],
        global_resources: Dict[Any, float]
    ) -> Dict[str, float]:
        """Calculate strategy tendency based on complex systems theory and game theory.

        Args:
            agent: Current civilization agent.
            neighbors: Dictionary of neighbor civilizations to relationship values.
            global_resources: Global resource distribution.

        Returns:
            Dictionary mapping strategy names to tendency scores.
        """
        # Base strategy tendencies
        strategy_scores = {
            'expansion': 0.0,
            'defense': 0.0,
            'trade': 0.0,
            'research': 0.0
        }

        # 1. Resource pressure assessment
        resource_pressure = self._assess_resource_pressure(agent, global_resources)

        # 2. Security risk assessment
        security_risk = self._assess_security_risk(agent, neighbors)

        # 3. Development potential assessment
        development_potential = self._assess_development_potential(agent, neighbors)

        # 4. Game theory-based strategy selection
        game_theory_influence = self._calculate_game_theory_influence(agent, neighbors)

        # 5. Historical experience learning
        historical_influence = self._learn_from_history(agent.agent_id)

        # Combine factors to calculate final strategy tendencies
        strategy_scores['expansion'] += (
            resource_pressure * 0.3 + (1 - security_risk) * 0.2 +
            game_theory_influence.get('expansion', 0) * 0.2 +
            historical_influence.get('expansion', 0) * 0.3
        )

        strategy_scores['defense'] += (
            security_risk * 0.4 + resource_pressure * 0.1 +
            game_theory_influence.get('defense', 0) * 0.2 +
            historical_influence.get('defense', 0) * 0.3
        )

        strategy_scores['trade'] += (
            (1 - resource_pressure) * 0.3 + development_potential * 0.2 +
            game_theory_influence.get('trade', 0) * 0.25 +
            historical_influence.get('trade', 0) * 0.25
        )

        strategy_scores['research'] += (
            development_potential * 0.4 + (1 - resource_pressure) * 0.2 +
            game_theory_influence.get('research', 0) * 0.2 +
            historical_influence.get('research', 0) * 0.2
        )

        # Apply metacognition adjustment
        if self.enable_metacognition:
            strategy_scores = self._apply_metacognition(agent, strategy_scores)

        # Add random exploration
        if random.random() < self.exploration_rate:
            random_strategy = random.choice(list(strategy_scores.keys()))
            strategy_scores[random_strategy] += 0.5

        # Normalize strategy scores
        total = sum(strategy_scores.values())
        if total > 0:
            for key in strategy_scores:
                strategy_scores[key] /= total

        # Record strategy history
        self._record_strategy_history(agent.agent_id, strategy_scores)

        return strategy_scores

    def _assess_resource_pressure(
        self,
        agent: Any,
        global_resources: Dict[Any, float]
    ) -> float:
        """Assess resource pressure faced by the civilization.

        Factors: current resource level, population needs, territory output, distribution.

        Args:
            agent: Civilization agent.
            global_resources: Global resource distribution.

        Returns:
            Resource pressure score (0-1).
        """
        # Current resource to demand ratio
        resource_need_ratio = agent.resources / (agent.population * 0.1)

        # Territory resource output efficiency
        territory_resource_efficiency = (
            sum(global_resources.get(pos, 0) for pos in agent.territory) / len(agent.territory)
            if agent.territory else 0
        )

        # Estimate consumption and production balance
        estimated_consumption = agent.population * 0.01
        estimated_production = (
            territory_resource_efficiency *
            getattr(agent, 'resource_acquisition', 1.0) *
            getattr(agent, 'infrastructure', 1.0)
        )

        consumption_production_ratio = estimated_consumption / (estimated_production + 0.001)

        # Calculate comprehensive resource pressure
        resource_pressure = 1.0 - exp(
            -(consumption_production_ratio / (resource_need_ratio + 0.1) *
              (1 / (territory_resource_efficiency + 0.1)))
        )

        return min(max(resource_pressure, 0.0), 1.0)

    def _assess_security_risk(
        self,
        agent: Any,
        neighbors: Dict[Any, float]
    ) -> float:
        """Assess security risk faced by the civilization.

        Factors: enemy strength, ally strength, territory adjacency, strategic environment.

        Args:
            agent: Civilization agent.
            neighbors: Dictionary of neighbor civilizations to relationship values.

        Returns:
            Security risk score (0-1).
        """
        if not neighbors:
            return 0.0

        enemy_strength = 0.0
        ally_strength = 0.0

        for neighbor, relation in neighbors.items():
            if relation < -0.4:  # Hostile threshold
                distance_factor = 1.0 / (getattr(agent, 'get_distance', lambda _: 1)(neighbor) + 1)
                enemy_strength += neighbor.strength * distance_factor
            elif relation > 0.6:  # Ally threshold
                ally_strength += neighbor.strength

        relative_strength = (enemy_strength + 1) / (ally_strength + agent.strength + 1)

        # Calculate security risk using logistic function
        security_risk = 1.0 - 1.0 / (1.0 + exp(relative_strength - 1.0))

        return min(max(security_risk, 0.0), 1.0)

    def _assess_development_potential(
        self,
        agent: Any,
        neighbors: Dict[Any, float]
    ) -> float:
        """Assess development potential of the civilization.

        Factors: technology level, population quality, infrastructure, exchange opportunities.

        Args:
            agent: Civilization agent.
            neighbors: Dictionary of neighbor civilizations to relationship values.

        Returns:
            Development potential score (0-1).
        """
        # Technology level score
        tech_score = sum(agent.technology.values()) / (len(agent.technology) * 5.0)

        # Population quality score
        population_quality = min(agent.population / 1000.0, 1.0)
        population_quality *= (1 + tech_score * 0.5)

        # Technology exchange potential with advanced civilizations
        tech_exchange_potential = 0.0
        if neighbors:
            avg_neighbor_tech = sum(sum(n.technology.values()) for n in neighbors) / len(neighbors)
            own_tech = sum(agent.technology.values())
            if avg_neighbor_tech > own_tech:
                tech_exchange_potential = min((avg_neighbor_tech - own_tech) / own_tech, 1.0) if own_tech > 0 else 1.0

        development_potential = (
            tech_score * 0.4 + population_quality * 0.3 + tech_exchange_potential * 0.3
        )

        return min(max(development_potential, 0.0), 1.0)

    def _calculate_game_theory_influence(
        self,
        _agent: Any,
        neighbors: Dict[Any, float]
    ) -> Dict[str, float]:
        """Calculate strategy influence based on game theory.

        Uses evolutionary game theory to analyze long-term benefits of different strategies.

        Args:
            agent: Civilization agent.
            neighbors: Dictionary of neighbor civilizations to relationship values.

        Returns:
            Dictionary mapping strategy names to game theory influence scores.
        """
        influence = {
            'expansion': 0.0,
            'defense': 0.0,
            'trade': 0.0,
            'research': 0.0
        }

        if not neighbors:
            return influence

        # Analyze neighbor strategy distribution
        neighbor_strategies = defaultdict(float)
        for neighbor, _ in neighbors.items():
            if hasattr(neighbor, 'last_strategy'):
                neighbor_strategies[neighbor.last_strategy] += 1.0

        # Normalize neighbor strategy distribution
        total = sum(neighbor_strategies.values())
        if total > 0:
            for key in neighbor_strategies:
                neighbor_strategies[key] /= total

        # Calculate benefits for each strategy based on opponent strategies
        expansion_benefit = 1.0 - neighbor_strategies.get('defense', 0) * 0.7
        expansion_benefit += neighbor_strategies.get('expansion', 0) * 0.3
        influence['expansion'] = expansion_benefit

        defense_benefit = neighbor_strategies.get('expansion', 0) * 0.8
        defense_benefit -= neighbor_strategies.get('research', 0) * 0.2
        influence['defense'] = defense_benefit

        peaceful_strategies = neighbor_strategies.get('trade', 0) + neighbor_strategies.get('research', 0)
        trade_benefit = peaceful_strategies * 0.9
        trade_benefit -= neighbor_strategies.get('expansion', 0) * 0.5
        influence['trade'] = trade_benefit

        research_benefit = 0.5
        research_benefit += (1 - sum(neighbor_strategies.values())) * 0.3
        research_benefit += neighbor_strategies.get('research', 0) * 0.2
        influence['research'] = research_benefit

        # Normalize influence scores
        total_influence = sum(influence.values())
        if total_influence > 0:
            for key in influence:
                influence[key] /= total_influence

        return influence

    def _learn_from_history(self, agent_id: int) -> Dict[str, float]:
        """Learn from historical experience.

        Analyze past strategy performance to adjust future strategy selection.

        Args:
            agent_id: Agent identifier.

        Returns:
            Dictionary mapping strategy names to history-based adjustment scores.
        """
        influence = {
            'expansion': 0.0,
            'defense': 0.0,
            'trade': 0.0,
            'research': 0.0
        }

        if agent_id not in self.strategy_history or len(self.strategy_history[agent_id]) < 2:
            return influence

        # Calculate average performance of recent strategies
        recent_strategies = list(self.strategy_history[agent_id])
        avg_strategies = defaultdict(float)
        for strategy in recent_strategies:
            for key, value in strategy.items():
                avg_strategies[key] += value

        total = sum(avg_strategies.values())
        if total > 0:
            for key in avg_strategies:
                avg_strategies[key] /= total
                influence[key] = avg_strategies[key]

        return influence

    def _apply_metacognition(
        self,
        _agent: Any,
        strategy_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply metacognitive adjustment to strategy selection.

        Civilization reflects on and adjusts its own decision-making process.

        Args:
            agent: Civilization agent.
            strategy_scores: Current strategy scores.

        Returns:
            Adjusted strategy scores.
        """
        adjusted_scores = copy.deepcopy(strategy_scores)

        # Detect strategy inertia (over-reliance on a single strategy)
        max_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
        max_score = strategy_scores[max_strategy]
        avg_score = sum(strategy_scores.values()) / len(strategy_scores)

        # Reduce weight for overused strategies to encourage diversity
        if max_score > avg_score * 1.5:
            adjustment = (max_score - avg_score * 1.5) * self.metacognition_rate
            adjusted_scores[max_strategy] -= adjustment

            other_strategies = [s for s in strategy_scores if s != max_strategy]
            if other_strategies:
                for s in other_strategies:
                    adjusted_scores[s] += adjustment / len(other_strategies)

        # Ensure all scores are positive
        for key in adjusted_scores:
            adjusted_scores[key] = max(adjusted_scores[key], 0.001)

        # Renormalize
        total = sum(adjusted_scores.values())
        if total > 0:
            for key in adjusted_scores:
                adjusted_scores[key] /= total

        return adjusted_scores

    def _calculate_tech_research_priority(self, agent: Any) -> Dict[str, float]:
        """Evaluate technology research priority for a civilization.

        Args:
            agent: Civilization agent.

        Returns:
            Dictionary mapping technology names to priority scores.
        """
        # Get available technologies
        available_techs = getattr(agent, 'tech_tree', None)
        if not available_techs:
            return {}

        tech_priorities: Dict[str, float] = {}

        for tech_name, current_level in agent.technology.items():
            priority_score = 0.0

            # Tech effects
            tech_effects = getattr(available_techs, 'tech_effects', {})

            for attribute, effect_value in tech_effects.get(tech_name, {}).items():
                if attribute in ['resources']:
                    priority_score += effect_value * 0.3
                elif attribute in ['strength', 'defense']:
                    priority_score += effect_value * 0.2
                elif attribute in ['research_speed']:
                    priority_score += effect_value * 0.4
                elif attribute in ['population_growth']:
                    priority_score += effect_value * 0.1

            # Strategy-based adjustment
            last_strategy = getattr(agent, 'last_strategy', None)
            if last_strategy == 'expansion':
                if 'resources' in tech_effects.get(tech_name, {}):
                    priority_score *= 1.2
            elif last_strategy == 'defense':
                if 'strength' in tech_effects.get(tech_name, {}) or 'defense' in tech_effects.get(tech_name, {}):
                    priority_score *= 1.2
            elif last_strategy == 'research':
                if 'research_speed' in tech_effects.get(tech_name, {}):
                    priority_score *= 1.3

            # Level adjustment
            level_factor = 1.0 + (current_level - 1) * 0.1

            priority_score *= level_factor
            tech_priorities[tech_name] = priority_score

        # Normalize
        total_priority = sum(tech_priorities.values())
        if total_priority > 0:
            for tech_name in tech_priorities:
                tech_priorities[tech_name] /= total_priority

        return tech_priorities

    def _record_strategy_history(
        self,
        agent_id: int,
        strategy_scores: Dict[str, float]
    ) -> None:
        """Record strategy selection history.

        Saves civilization's strategy choices for historical learning.

        Args:
            agent_id: Agent identifier.
            strategy_scores: Current strategy scores.
        """
        self.strategy_history[agent_id].append(copy.deepcopy(strategy_scores))

        # Keep history within memory window
        while len(self.strategy_history[agent_id]) > self.memory_window:
            self.strategy_history[agent_id].popleft()


class ComplexResourceManager:
    """Complex resource management system implementing scientific resource distribution and management mechanisms."""

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize complex resource management system.

        Args:
            config: Simulation configuration object.
        """
        self.config = config
        # Resource type definitions
        self.resource_types = {
            'food': {'base_value': 1.0, 'regeneration_rate': 0.01, 'consumption_rate': 0.02},
            'energy': {'base_value': 1.5, 'regeneration_rate': 0.005, 'consumption_rate': 0.015},
            'materials': {'base_value': 2.0, 'regeneration_rate': 0.003, 'consumption_rate': 0.01},
            'technology': {'base_value': 3.0, 'regeneration_rate': 0.002, 'consumption_rate': 0.005}
        }
        self.logger = get_logger(__name__)

    def generate_resource_map(self, grid_size: int) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Generate complex resource distribution map.

        Creates more realistic resource distribution based on GIS principles.

        Args:
            grid_size: Grid size.

        Returns:
            Dictionary mapping positions to resource dictionaries.
        """
        resources: Dict[Tuple[int, int], Dict[str, float]] = {}

        # Create basic terrain features
        elevation_map = self._generate_elevation_map(grid_size)
        moisture_map = self._generate_moisture_map(grid_size)

        # Generate resources for each position
        for i in range(grid_size):
            for j in range(grid_size):
                pos = (i, j)
                resources[pos] = self._generate_resources_for_position(pos, elevation_map, moisture_map)

        return resources

    def _generate_elevation_map(self, grid_size: int) -> np.ndarray:
        """Generate elevation map using Perlin-like noise.

        Args:
            grid_size: Grid size.

        Returns:
            Elevation map as numpy array.
        """
        # Simplified implementation with multi-scale features
        elevation = np.random.rand(grid_size, grid_size) * 0.5

        # Add large-scale features
        for scale in [grid_size // 4, grid_size // 8, grid_size // 16]:
            if scale > 0:
                large_size = int(np.ceil(grid_size / scale))
                large_features = np.random.rand(large_size, large_size)
                upsampled = np.repeat(np.repeat(large_features, scale, axis=0), scale, axis=1)
                upsampled = upsampled[:grid_size, :grid_size]
                if upsampled.shape == elevation.shape:
                    elevation += upsampled * 0.1

        # Normalize to 0-1 range
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())

        return elevation

    def _generate_moisture_map(self, grid_size: int) -> np.ndarray:
        """Generate moisture map based on simplified climate model.

        Args:
            grid_size: Grid size.

        Returns:
            Moisture map as numpy array.
        """
        # Base moisture distribution
        moisture = np.random.rand(grid_size, grid_size) * 0.5

        # Add latitude effect (simplified model)
        for i in range(grid_size):
            lat_factor = 1.0 - abs(i / grid_size - 0.5) * 2.0
            moisture[i, :] += lat_factor * 0.3

        # Normalize to 0-1 range
        moisture = (moisture - moisture.min()) / (moisture.max() - moisture.min())

        return moisture

    def _generate_resources_for_position(
        self,
        pos: Tuple[int, int],
        elevation_map: np.ndarray,
        moisture_map: np.ndarray
    ) -> Dict[str, float]:
        """Generate resources for a specific position based on terrain.

        Args:
            pos: Position coordinates.
            elevation_map: Elevation map.
            moisture_map: Moisture map.

        Returns:
            Resource dictionary.
        """
        x, y = pos
        elevation = elevation_map[x, y]
        moisture = moisture_map[x, y]

        resources: Dict[str, float] = {}

        # Food: abundant at suitable elevation and moisture
        food_factor = exp(-((elevation - 0.3) ** 2 / (2 * 0.1 ** 2)) - ((moisture - 0.7) ** 2 / (2 * 0.15 ** 2)))
        resources['food'] = max(0, np.random.normal(food_factor * 5, 1))

        # Energy: abundant in specific terrains
        energy_factor = 0
        if 0.6 < elevation < 0.8:
            energy_factor = 0.7
        elif moisture < 0.2:
            energy_factor = 0.5
        resources['energy'] = max(0, np.random.normal(energy_factor * 4, 1))

        # Materials: distributed in mountains and hills
        materials_factor = elevation * 0.8 + np.random.normal(0, 0.1)
        resources['materials'] = max(0, np.random.normal(materials_factor * 3, 1))

        # Technology: related to special geographical features
        tech_factor = 0.1 + np.random.random() * 0.2
        resources['technology'] = max(0, np.random.normal(tech_factor * 2, 0.5))

        return resources

    def calculate_resource_value(self, resources: Dict[str, float]) -> float:
        """Calculate total value of resources.

        Considers relative value and scarcity of different resource types.

        Args:
            resources: Resource dictionary.

        Returns:
            Total resource value.
        """
        total_value = 0.0

        for resource_type, amount in resources.items():
            if resource_type in self.resource_types:
                total_value += self.resource_types[resource_type]['base_value'] * amount

        return total_value

    def regenerate_resources(
        self,
        resources: Dict[str, float],
        _position: Tuple[int, int],
        cycle: int
    ) -> Dict[str, float]:
        """Regenerate resources simulating natural regeneration.

        Args:
            resources: Current resource dictionary.
            position: Position coordinates.
            cycle: Current simulation cycle.

        Returns:
            Updated resource dictionary.
        """
        updated_resources = copy.deepcopy(resources)

        for resource_type, amount in updated_resources.items():
            if resource_type in self.resource_types:
                regeneration_rate = self.resource_types[resource_type]['regeneration_rate']
                seasonal_factor = 1.0 + 0.3 * sin(2 * pi * cycle / 100)

                updated_amount = amount * (1 + regeneration_rate * seasonal_factor)
                max_amount = amount * 1.5 if amount > 0 else 10.0
                updated_resources[resource_type] = min(updated_amount, max_amount)

        return updated_resources


class CulturalInfluence:
    """Cultural influence system simulating cultural exchange and influence between civilizations."""

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize cultural influence system.

        Args:
            config: Simulation configuration object.
        """
        self.config = config
        # Cultural trait definitions
        self.cultural_traits = {
            'collectivism': {'description': 'Collectivism', 'influence_factor': 0.1},
            'individualism': {'description': 'Individualism', 'influence_factor': 0.1},
            'militarism': {'description': 'Militarism', 'influence_factor': 0.15},
            'pacifism': {'description': 'Pacifism', 'influence_factor': 0.15},
            'tradition': {'description': 'Tradition', 'influence_factor': 0.08},
            'innovation': {'description': 'Innovation', 'influence_factor': 0.08},
            'expansionism': {'description': 'Expansionism', 'influence_factor': 0.12},
            'isolationism': {'description': 'Isolationism', 'influence_factor': 0.12}
        }
        self.logger = get_logger(__name__)

    def initialize_culture(self, agent: Any) -> None:
        """Initialize cultural traits for a civilization.

        Args:
            agent: Civilization agent.
        """
        agent.culture = {}
        if not hasattr(agent, 'culture'):
            setattr(agent, 'culture', {})

        trait_pairs = [
            ('collectivism', 'individualism'),
            ('militarism', 'pacifism'),
            ('tradition', 'innovation'),
            ('expansionism', 'isolationism')
        ]

        for trait1, trait2 in trait_pairs:
            value = random.random()
            agent.culture[trait1] = value
            agent.culture[trait2] = 1.0 - value

    def update_cultural_influence(
        self,
        agent: Any,
        neighbors: Dict[Any, float]
    ) -> None:
        """Update cultural influence between civilizations.

        Args:
            agent: Current civilization agent.
            neighbors: Dictionary of neighbor civilizations to relationship values.
        """
        if not hasattr(agent, 'culture'):
            self.initialize_culture(agent)

        for neighbor, relation in neighbors.items():
            if hasattr(neighbor, 'culture'):
                influence_strength = abs(relation) * 0.02
                similarity = self._calculate_cultural_similarity(agent.culture, neighbor.culture)
                actual_influence = influence_strength * (1 - similarity)

                for trait in agent.culture:
                    if trait in neighbor.culture:
                        agent.culture[trait] += actual_influence * (neighbor.culture[trait] - agent.culture[trait])
                        agent.culture[trait] = max(0.0, min(1.0, agent.culture[trait]))

    def _calculate_cultural_similarity(
        self,
        culture1: Dict[str, float],
        culture2: Dict[str, float]
    ) -> float:
        """Calculate similarity between two cultures.

        Args:
            culture1: First culture.
            culture2: Second culture.

        Returns:
            Similarity score (0-1).
        """
        similarity = 0.0
        common_traits = set(culture1.keys()) & set(culture2.keys())

        if not common_traits:
            return 0.0

        for trait in common_traits:
            similarity += 1.0 - abs(culture1[trait] - culture2[trait])

        similarity /= len(common_traits)

        return similarity

    def get_cultural_bonuses(self, agent: Any) -> Dict[str, float]:
        """Calculate attribute bonuses from cultural traits.

        Args:
            agent: Civilization agent.

        Returns:
            Dictionary of attribute bonuses.
        """
        if not hasattr(agent, 'culture'):
            self.initialize_culture(agent)

        bonuses: Dict[str, float] = {
            'research_speed': 0.0,
            'resource_collection': 0.0,
            'military_strength': 0.0,
            'population_growth': 0.0,
            'diplomacy_effectiveness': 0.0
        }

        # Innovation and individualism boost research speed
        bonuses['research_speed'] += agent.culture.get('innovation', 0) * 0.3
        bonuses['research_speed'] += agent.culture.get('individualism', 0) * 0.15

        # Collectivism and tradition boost resource collection
        bonuses['resource_collection'] += agent.culture.get('collectivism', 0) * 0.2
        bonuses['resource_collection'] += agent.culture.get('tradition', 0) * 0.1

        # Militarism and expansionism boost military strength
        bonuses['military_strength'] += agent.culture.get('militarism', 0) * 0.3
        bonuses['military_strength'] += agent.culture.get('expansionism', 0) * 0.2

        # Collectivism and pacifism boost population growth
        bonuses['population_growth'] += agent.culture.get('collectivism', 0) * 0.2
        bonuses['population_growth'] += agent.culture.get('pacifism', 0) * 0.15

        # Pacifism boosts diplomacy, individualism reduces it
        bonuses['diplomacy_effectiveness'] += agent.culture.get('pacifism', 0) * 0.25
        bonuses['diplomacy_effectiveness'] -= agent.culture.get('individualism', 0) * 0.1

        return bonuses


__all__ = ['AdvancedEvolution', 'ComplexResourceManager', 'CulturalInfluence']
