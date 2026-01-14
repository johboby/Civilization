"""
Civilization simulation system - Strategy decision module

This module implements civilization strategy decision logic, supporting multiple strategy types and decision algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class StrategyResult:
    """Strategy decision result"""

    name: str
    probability: float
    confidence: float = 1.0


class StrategyDecisionEngine(ABC):
    """Abstract base class for strategy decision engines"""

    @abstractmethod
    def decide(
        self, agent_state: "AgentState", neighbors: Dict, global_resources: Dict
    ) -> List[StrategyResult]:
        """Determine strategy based on current state

        Args:
            agent_state: Civilization agent state
            neighbors: Neighbor civilization information
            global_resources: Global resource distribution

        Returns:
            List of strategy decision results
        """
        pass


@dataclass
class AgentState:
    """Civilization agent state data class"""

    agent_id: int
    strength: float
    resources: float
    population: float
    infrastructure: float
    stability: float
    global_influence: float
    research_speed: float
    cultural_influence: float = 0.0
    religious_influence: float = 0.0
    resource_acquisition: float = 1.0
    decision_quality: float = 1.0

    # Technology bonuses
    tech_bonuses: Optional[Dict[str, float]] = None

    # Territory and relationships
    territory_size: int = 0
    allies_count: int = 0
    enemies_count: int = 0

    def __post_init__(self):
        if self.tech_bonuses is None:
            self.tech_bonuses = {}


class DefaultDecisionEngine(StrategyDecisionEngine):
    """Default strategy decision engine"""

    # Strategy name list
    STRATEGY_NAMES = [
        "expansion",
        "defense",
        "trade",
        "research",
        "diplomacy",
        "culture",
        "religion",
    ]

    def __init__(self, config=None):
        """Initialize decision engine

        Args:
            config: Configuration object
        """
        self.config = config
        self.relationship_threshold = (
            getattr(config, "RELATIONSHIP_THRESHOLD", 0.5) if config else 0.5
        )

    def decide(
        self, agent_state: AgentState, neighbors: Dict, global_resources: Dict
    ) -> List[StrategyResult]:
        """Decide strategy

        Args:
            agent_state: Civilization agent state
            neighbors: Neighbor civilization information
            global_resources: Global resource distribution

        Returns:
            List of strategy decision results
        """
        # Calculate base weights
        strategy_weights = self._calculate_base_weights(agent_state)

        # Apply threat adjustment
        threat_adjustment = self._calculate_threat_adjustment(agent_state, neighbors)
        for i, adj in enumerate(threat_adjustment):
            strategy_weights[i] += adj

        # Apply resource adjustment
        resource_adjustment = self._calculate_resource_adjustment(agent_state, global_resources)
        for i, adj in enumerate(resource_adjustment):
            strategy_weights[i] += adj

        # Apply research adjustment
        research_adjustment = self._calculate_research_adjustment(agent_state)
        for i, adj in enumerate(research_adjustment):
            strategy_weights[i] += adj

        # Apply diplomacy and culture adjustment
        diplomacy_adjustment = self._calculate_diplomacy_culture_adjustment(agent_state, neighbors)
        for i, adj in enumerate(diplomacy_adjustment):
            strategy_weights[i] += adj

        # Ensure all values are positive
        strategy_weights = np.clip(strategy_weights, 0.0, None)

        # Normalize
        total = np.sum(strategy_weights)
        if total <= 0.0:
            strategy_weights = np.ones(len(strategy_weights)) / len(strategy_weights)
        else:
            strategy_weights = strategy_weights / total

        # Create strategy results
        results = []
        for i, (name, weight) in enumerate(zip(self.STRATEGY_NAMES, strategy_weights)):
            if i < len(strategy_weights):  # Ensure not exceeding strategy count
                results.append(StrategyResult(name=name, probability=float(weight)))

        return results

    def _calculate_base_weights(self, state: AgentState) -> np.ndarray:
        """Calculate base weights"""
        infra = state.infrastructure
        stability = state.stability if state.stability > 0 else 1.0
        global_inf = state.global_influence
        research_speed = state.tech_bonuses.get("research_speed", 1.0)

        weights = np.zeros(len(self.STRATEGY_NAMES))

        # expansion
        weights[0] = 0.2 * infra

        # defense
        weights[1] = 0.15 * (1.0 / stability)

        # trade
        weights[2] = 0.25 * global_inf

        # research
        weights[3] = 0.4 * research_speed

        # diplomacy
        weights[4] = 0.2 * global_inf

        # culture
        if len(weights) > 5:
            weights[5] = 0.15 * state.cultural_influence

        # religion
        if len(weights) > 6:
            weights[6] = 0.1 * state.religious_influence

        return weights

    def _calculate_threat_adjustment(self, state: AgentState, neighbors: Dict) -> np.ndarray:
        """Calculate threat adjustment"""
        adjustment = np.zeros(len(self.STRATEGY_NAMES))

        if not neighbors:
            return adjustment

        # Calculate enemy civilization strength
        enemy_strength = sum(
            n_strength
            for n_strength, rel in neighbors.values()
            if rel < -self.relationship_threshold
        )

        effective_strength = state.strength * state.tech_bonuses.get("strength", 1.0)

        if enemy_strength > effective_strength * 1.2 and len(adjustment) > 1:
            threat_factor = min(enemy_strength / max(effective_strength, 1e-6), 2.0)
            adjustment[1] = 0.6 + min(0.3, threat_factor * 0.1)

        return adjustment

    def _calculate_resource_adjustment(
        self, state: AgentState, global_resources: Dict
    ) -> np.ndarray:
        """Calculate resource adjustment"""
        adjustment = np.zeros(len(self.STRATEGY_NAMES))

        if state.territory_size == 0:
            return adjustment

        controlled_resources = state.resources * state.territory_size

        if controlled_resources <= 0:
            return adjustment

        resource_pressure = 1.0 - (state.resources / (controlled_resources * 10.0))

        if resource_pressure > 0.8:
            if state.allies_count > 0 and len(adjustment) > 2:
                trade_efficiency = state.tech_bonuses.get("trade_efficiency", 1.0)
                adjustment[2] = 0.5 + (1.0 - resource_pressure) * 0.4 * trade_efficiency
            else:
                adjustment[0] = 0.5 + resource_pressure * 0.3 * state.resource_acquisition

        return adjustment

    def _calculate_research_adjustment(self, state: AgentState) -> np.ndarray:
        """Calculate research adjustment"""
        adjustment = np.zeros(len(self.STRATEGY_NAMES))

        # Simplified implementation, should check available technologies in practice
        if state.resources > 1000 and len(adjustment) > 3:
            research_speed = state.tech_bonuses.get("research_speed", 1.0)
            adjustment[3] = 0.6 + 0.1 * research_speed

        return adjustment

    def _calculate_diplomacy_culture_adjustment(
        self, state: AgentState, neighbors: Dict
    ) -> np.ndarray:
        """Calculate diplomacy and culture adjustment"""
        adjustment = np.zeros(len(self.STRATEGY_NAMES))

        # Simplified implementation
        if state.global_influence > 1.0 and len(adjustment) > 4:
            adjustment[4] = 0.2 * state.global_influence

        return adjustment


class RandomDecisionEngine(StrategyDecisionEngine):
    """Random strategy decision engine (for testing)"""

    def decide(
        self, agent_state: AgentState, neighbors: Dict, global_resources: Dict
    ) -> List[StrategyResult]:
        """Randomly select strategy"""
        strategy_names = DefaultDecisionEngine.STRATEGY_NAMES
        probabilities = np.random.rand(len(strategy_names))
        probabilities = probabilities / np.sum(probabilities)

        results = [
            StrategyResult(name=name, probability=float(prob))
            for name, prob in zip(strategy_names, probabilities)
        ]

        return results


def create_decision_engine(engine_type: str = "default", config=None) -> StrategyDecisionEngine:
    """Create strategy decision engine

    Args:
        engine_type: Engine type ("default" or "random")
        config: Configuration object

    Returns:
        Strategy decision engine instance
    """
    engines = {"default": DefaultDecisionEngine, "random": RandomDecisionEngine}

    if engine_type not in engines:
        raise ValueError(f"Unknown engine type: {engine_type}. Available: {list(engines.keys())}")

    return engines[engine_type](config=config)


if __name__ == "__main__":
    # Test decision engine
    state = AgentState(
        agent_id=1,
        strength=100.0,
        resources=200.0,
        population=100.0,
        infrastructure=1.5,
        stability=1.2,
        global_influence=1.0,
        research_speed=1.0,
    )

    neighbors = {
        2: (50.0, -0.8),  # Hostile
        3: (80.0, 0.7),  # Ally
    }

    global_resources = {"cell_1": 50.0, "cell_2": 60.0}

    engine = create_decision_engine("default")
    results = engine.decide(state, neighbors, global_resources)

    print("Strategy decision results:")
    for result in results:
        print(f"  {result.name}: {result.probability:.4f}")
