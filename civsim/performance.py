"""
Performance optimization utilities for civilization simulation.

This module provides vectorized operations to improve simulation performance.
"""
import numpy as np
from typing import Any, Dict, List


class VectorizedOperations:
    """Vectorized operations for faster simulation loops."""

    @staticmethod
    def compute_resource_production(
        territories: Dict[Any, float],
        acquisition_rates: List[float],
        infrastructure_levels: List[float]
    ) -> np.ndarray:
        """Compute resource production for all agents using vectorized operations.

        Args:
            territories: Dictionary of territory positions to resource values.
            acquisition_rates: List of resource acquisition rates for each agent.
            infrastructure_levels: List of infrastructure levels for each agent.

        Returns:
            Numpy array of production values for each agent.
        """
        if not territories:
            return np.zeros(len(acquisition_rates))

        # Convert territories to array
        territory_values = np.array(list(territories.values()))

        # Vectorized computation
        acquisition_array = np.array(acquisition_rates)
        infrastructure_array = np.array(infrastructure_levels)

        # Compute production using broadcasting
        production = np.sum(territory_values) * acquisition_array * infrastructure_array

        return production

    @staticmethod
    def compute_population_growth(
        populations: np.ndarray,
        growth_rates: np.ndarray,
        stability_factors: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """Compute population growth using vectorized operations.

        Args:
            populations: Current population array.
            growth_rates: Growth rate array.
            stability_factors: Stability factor array.
            dt: Time step.

        Returns:
            New population array.
        """
        # Logistic growth model: dN/dt = r*N*(1-N/K)
        # Simplified: N_new = N * (1 + r * stability * dt)
        return populations * (1.0 + growth_rates * stability_factors * dt)

    @staticmethod
    def compute_relationship_updates(
        relationship_matrix: np.ndarray,
        interaction_strength: float,
        decay_factor: float = 0.01
    ) -> np.ndarray:
        """Compute relationship updates using vectorized operations.

        Args:
            relationship_matrix: Current relationship matrix (N x N).
            interaction_strength: Strength of interactions.
            decay_factor: Natural decay of relationships.

        Returns:
            Updated relationship matrix.
        """
        # Apply decay
        relationship_matrix = relationship_matrix * (1.0 - decay_factor)

        # Apply interaction strength (simplified)
        n = len(relationship_matrix)
        interaction_matrix = np.random.randn(n, n) * interaction_strength * 0.01

        # Update relationships
        updated_matrix = relationship_matrix + interaction_matrix

        # Clip to valid range [-1, 1]
        updated_matrix = np.clip(updated_matrix, -1.0, 1.0)

        return updated_matrix

    @staticmethod
    def compute_strategy_probabilities(
        scores: Dict[str, float],
        temperature: float = 1.0
    ) -> np.ndarray:
        """Compute strategy probabilities using softmax.

        Args:
            scores: Dictionary of strategy scores.
            temperature: Temperature for softmax (higher = more uniform).

        Returns:
            Numpy array of probabilities.
        """
        # Convert scores to array
        score_array = np.array(list(scores.values()))

        # Apply temperature
        if temperature > 0:
            score_array = score_array / temperature

        # Compute softmax
        exp_scores = np.exp(score_array - np.max(score_array))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)

        return probabilities

    @staticmethod
    def compute_technology_progress(
        current_techs: np.ndarray,
        research_investments: np.ndarray,
        tech_costs: np.ndarray,
        max_levels: np.ndarray
    ) -> np.ndarray:
        """Compute technology progress using vectorized operations.

        Args:
            current_techs: Current technology levels.
            research_investments: Research investment amounts.
            tech_costs: Technology cost per level.
            max_levels: Maximum technology levels.

        Returns:
            Updated technology levels.
        """
        # Compute progress
        progress = research_investments / tech_costs

        # Update levels
        new_techs = current_techs + progress

        # Clip to max levels
        new_techs = np.minimum(new_techs, max_levels)

        return new_techs

    @staticmethod
    def batch_distance_matrix(
        positions: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise distance matrix using vectorized operations.

        Args:
            positions: Array of shape (N, 2) containing x, y coordinates.

        Returns:
            Distance matrix of shape (N, N).
        """
        # Compute squared differences
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

        # Compute squared distances
        squared_distances = np.sum(diff ** 2, axis=-1)

        # Compute distances
        distances = np.sqrt(squared_distances)

        return distances

    @staticmethod
    def compute_influence_spread(
        influence_matrix: np.ndarray,
        spread_rate: float,
        decay_rate: float = 0.02
    ) -> np.ndarray:
        """Compute spread of influence using vectorized operations.

        Args:
            influence_matrix: Current influence matrix (N x N).
            spread_rate: Rate of influence spread.
            decay_rate: Rate of natural influence decay.

        Returns:
            Updated influence matrix.
        """
        # Apply decay
        influence_matrix = influence_matrix * (1.0 - decay_rate)

        # Apply spread (diffusion)
        n = len(influence_matrix)
        spread = np.zeros_like(influence_matrix)

        for i in range(n):
            # Influence spreads to neighbors
            spread[i, :] = np.sum(influence_matrix[:, i]) * spread_rate / n

        influence_matrix = influence_matrix + spread

        # Clip to valid range
        influence_matrix = np.clip(influence_matrix, 0.0, 1.0)

        return influence_matrix


class OptimizedSimulationLoop:
    """Optimized simulation loop using vectorized operations."""

    def __init__(self, num_agents: int):
        """Initialize optimized simulation loop.

        Args:
            num_agents: Number of agents in simulation.
        """
        self.num_agents = num_agents
        self.vector_ops = VectorizedOperations()

        # Initialize state arrays
        self.populations = np.zeros(num_agents)
        self.resources = np.zeros(num_agents)
        self.strengths = np.zeros(num_agents)
        self.stability = np.ones(num_agents)
        self.infrastructure = np.ones(num_agents)
        self.relationships = np.zeros((num_agents, num_agents))

    def update_state(self, **kwargs):
        """Update simulation state using vectorized operations.

        Args:
            **kwargs: State updates.
        """
        if 'populations' in kwargs:
            self.populations = np.array(kwargs['populations'])
        if 'resources' in kwargs:
            self.resources = np.array(kwargs['resources'])
        if 'strengths' in kwargs:
            self.strengths = np.array(kwargs['strengths'])
        if 'stability' in kwargs:
            self.stability = np.array(kwargs['stability'])
        if 'infrastructure' in kwargs:
            self.infrastructure = np.array(kwargs['infrastructure'])
        if 'relationships' in kwargs:
            self.relationships = np.array(kwargs['relationships'])

    def get_agent_state(self, agent_id: int) -> Dict[str, float]:
        """Get state for a specific agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Dictionary of agent state.
        """
        return {
            'population': float(self.populations[agent_id]),
            'resources': float(self.resources[agent_id]),
            'strength': float(self.strengths[agent_id]),
            'stability': float(self.stability[agent_id]),
            'infrastructure': float(self.infrastructure[agent_id]),
        }

    def compute_all_updates(self, dt: float = 1.0):
        """Compute all state updates in a single vectorized operation.

        Args:
            dt: Time step.

        Returns:
            Dictionary of updated states.
        """
        # Compute population growth
        growth_rates = np.ones(self.num_agents) * 0.01  # Base growth rate
        new_populations = self.vector_ops.compute_population_growth(
            self.populations, growth_rates, self.stability, dt
        )

        # Compute relationship updates
        new_relationships = self.vector_ops.compute_relationship_updates(
            self.relationships, interaction_strength=0.1
        )

        # Update state
        updates = {
            'populations': new_populations,
            'relationships': new_relationships
        }

        return updates


__all__ = ['VectorizedOperations', 'OptimizedSimulationLoop']
