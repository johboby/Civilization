"""Relationship management module

This module is responsible for managing diplomatic relationships between civilization agents.
"""
from typing import Dict, Set, Tuple, Any
import numpy as np


class RelationshipManager:
    """Relationship manager - manages diplomatic relationships"""

    def __init__(self, agent):
        """Initialize relationship manager

        Args:
            agent: Civilization agent instance
        """
        self.agent = agent
        self.relationship_weights: Dict[int, float] = {}
        self.allies: Set[int] = set()
        self.enemies: Set[int] = set()

    def initialize_relationships(self, num_agents: int) -> None:
        """Initialize relationship weights

        Args:
            num_agents: Total number of agents
        """
        for i in range(num_agents):
            if i != self.agent.agent_id:
                self.relationship_weights[i] = np.random.uniform(-0.3, 0.3)

    def get_neighbors(self, all_agents: list) -> Dict[int, Tuple[float, float]]:
        """Get neighboring agents

        Args:
            all_agents: List of all agents

        Returns:
            Neighboring agent mapping {agent_id -> (strength, relationship)}
        """
        neighbors = {}
        for other_agent in all_agents:
            if other_agent.agent_id != self.agent.agent_id:
                # Calculate distance (based on territory overlap)
                shared_border = len(set(self.agent.territory) & set(other_agent.territory))
                if shared_border > 0:
                    neighbors[other_agent.agent_id] = (other_agent.strength, self.relationship_weights.get(other_agent.agent_id, 0.0))

        return neighbors

    def update_relationships(self, other_id: int, change: float) -> None:
        """Update relationship with another agent

        Args:
            other_id: Other agent ID
            change: Relationship change amount
        """
        if other_id in self.relationship_weights:
            self.relationship_weights[other_id] += change

            # Clamp within reasonable range
            self.relationship_weights[other_id] = max(-1.0, min(1.0, self.relationship_weights[other_id]))

            # Update ally/enemy sets
            if self.relationship_weights[other_id] > 0.4:
                self.allies.add(other_id)
            elif self.relationship_weights[other_id] < -0.4:
                self.enemies.add(other_id)
            else:
                self.allies.discard(other_id)
                self.enemies.discard(other_id)

    def calculate_threat_level(self, neighbors: Dict[int, Tuple[float, float]]) -> float:
        """Calculate threat level

        Args:
            neighbors: Neighboring agents

        Returns:
            Threat level (0-1)
        """
        rel_threshold = 0.5
        enemy_strength = sum(n_strength for _, (n_strength, rel) in neighbors.items() if rel < -rel_threshold)

        effective_strength = float(self.agent.strength) * float(self.agent.tech_bonuses.get('strength', 1.0))
        if effective_strength <= 0:
            return 0.0

        if enemy_strength > effective_strength * 1.2:
            threat_factor = min(enemy_strength / max(effective_strength, 1e-6), 2.0)
            return min(1.0, threat_factor * 0.5)
        return 0.0
