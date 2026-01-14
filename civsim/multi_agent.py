"""Lightweight MultiAgentSimulation implementation used during incremental refactor.

This module provides a minimal but functional MultiAgentSimulation class that
uses the CivilizationAgent defined in `civsim.simulation`. It's intentionally
small to avoid depending on the large, partially-migrated top-level module.
"""
from typing import Dict
import os
import numpy as np

from civsim.simulation import CivilizationAgent
from civsim.config import SimulationConfig


class MultiAgentSimulation:
    def __init__(self, config=None):
        from civsim.config import default_config
        self.config = config if config is not None else default_config
        self.num_agents = getattr(self.config, 'num_civilizations', 3)
        self.grid_size = getattr(self.config, 'grid_size', 10)

        # Seed RNG if provided
        seed = getattr(self.config, 'random_seed', None)
        if seed is not None:
            np.random.seed(seed)

        # Simple global resource map
        self.global_resources = self._generate_resources(self.grid_size)
        self.h3_grid = list(self.global_resources.keys())

        # Create agents
        self.agents = [CivilizationAgent(i) for i in range(self.num_agents)]

        # Initialize per-agent relationship weights and territories
        for agent in self.agents:
            agent.relationship_weights = {}
            agent.allies = set()
            agent.enemies = set()

        self.initialize_territory()
        self.initialize_relationships()

        # Minimal history containers used by tests
        self.history = {
            "strategy": [],
            "resources": [],
            "strength": [],
            "technology": [],
            "population": [],
            "territory": [],
        }
        self.attribute_history = []
        self.tech_spillover_history = []
        self.relationship_history = []
        self.technology_history = {agent.agent_id: [] for agent in self.agents}

    def _generate_resources(self, grid_size: int) -> Dict[str, float]:
        return {f"h3_{i}": float(np.random.rand() * 100) for i in range(max(1, grid_size))}

    def initialize_territory(self):
        territory_per_agent = max(1, len(self.h3_grid) // max(1, len(self.agents)))
        for i, agent in enumerate(self.agents):
            start_idx = i * territory_per_agent
            end_idx = start_idx + territory_per_agent
            agent.territory = set(self.h3_grid[start_idx:end_idx])

    def initialize_relationships(self):
        for agent in self.agents:
            for other in self.agents:
                if agent.agent_id == other.agent_id:
                    continue
                weight = float(np.random.uniform(-0.1, 0.1))
                agent.relationship_weights[other.agent_id] = weight
                if weight > 0.5:
                    agent.allies.add(other.agent_id)
                elif weight < -0.5:
                    agent.enemies.add(other.agent_id)

    def _get_neighbors(self, agent):
        neighbors = {}
        for other in self.agents:
            if agent.agent_id == other.agent_id:
                continue
            shared_border = len(agent.territory & other.territory) > 0
            if shared_border:
                weight = agent.relationship_weights.get(other.agent_id, 0.0)
                neighbors[other.agent_id] = (other.strength, weight)
        return neighbors

    def step(self, cycle=None):
        agent_neighbors = {agent.agent_id: self._get_neighbors(agent) for agent in self.agents}

        # --- Technology spillover ---
        tech_levels = {agent.agent_id: sum(agent.technology.values()) for agent in self.agents}
        avg_tech_level = sum(tech_levels.values()) / max(1, len(tech_levels))

        spillover_amounts = []
        for agent in self.agents:
            neighbor_tech_bonus = 0.0
            neighbors = agent_neighbors[agent.agent_id]
            for neighbor_id, (_, rel) in neighbors.items():
                neighbor_agent = next((a for a in self.agents if a.agent_id == neighbor_id), None)
                if neighbor_agent is None:
                    continue
                neighbor_total_tech = sum(neighbor_agent.technology.values())
                agent_total_tech = sum(agent.technology.values())
                if neighbor_total_tech > agent_total_tech:
                    tech_gap = neighbor_total_tech - agent_total_tech
                    rel_threshold = getattr(self.config, 'RELATIONSHIP_THRESHOLD', 0.5)
                    if rel > rel_threshold:
                        spillover_coefficient = 0.05
                    elif rel < -rel_threshold:
                        spillover_coefficient = 0.01
                    else:
                        spillover_coefficient = 0.02
                    neighbor_tech_bonus += tech_gap * spillover_coefficient * getattr(self.config, 'TECH_SPILLOVER_EFFECT', 0.0)

            global_spillover = max(0.0, avg_tech_level - sum(agent.technology.values())) * 0.01 * getattr(self.config, 'TECH_SPILLOVER_EFFECT', 0.0)
            total_spillover = neighbor_tech_bonus + global_spillover
            spillover_amounts.append(total_spillover)
            agent.tech_spillover_received = total_spillover

        self.tech_spillover_history = getattr(self, 'tech_spillover_history', [])
        self.tech_spillover_history.append(spillover_amounts)

        # --- Per-agent resource/population updates (simple model) ---
        cycle_attributes = []
        for agent in self.agents:
            # compute base output from territory
            base_output = sum(self.global_resources.get(h3, 0.0) for h3 in agent.territory) * getattr(self.config, 'RESOURCE_BASE_OUTPUT', 0.1)
            resource_bonus = agent.tech_bonuses.get('resources', 1.0)
            territory_value_bonus = agent.tech_bonuses.get('territory_value', 1.0)
            actual_output = base_output * resource_bonus * territory_value_bonus
            # add spillover-derived resources
            actual_output += getattr(agent, 'tech_spillover_received', 0.0) * 0.1
            agent.resources += actual_output

            # consumption
            base_consumption = getattr(self.config, 'RESOURCE_CONSUMPTION_RATE', 0.01) * agent.resources
            population_consumption = agent.population * getattr(self.config, 'POPULATION_RESOURCE_CONSUMPTION', 0.05)
            agent.resources = max(0.0, agent.resources - (base_consumption + population_consumption))

            # population growth
            if agent.resources > 0 and agent.population > 0:
                resource_per_capita = agent.resources / agent.population
                resource_factor = min(resource_per_capita / getattr(self.config, 'RESOURCE_PER_CAPITA_FOR_MAX_GROWTH', 10), 1.0)
                max_growth_rate = getattr(self.config, 'POPULATION_GROWTH_RATE', 0.02)
                growth_rate = max_growth_rate * resource_factor * agent.tech_bonuses.get('population_growth', 1.0) * agent.health
                population_increase = agent.population * growth_rate
                cap = getattr(self.config, 'POPULATION_GROWTH_CAP', None)
                if cap is not None and cap > 0:
                    agent.population = min(cap, agent.population + population_increase)
                else:
                    agent.population += population_increase

            cycle_attributes.append([agent.resources, agent.strength, agent.population, len(agent.territory)])

        # Save histories
        self.history['strategy'].append({})
        self.history['resources'].append([agent.resources for agent in self.agents])
        self.history['strength'].append([agent.strength for agent in self.agents])
        self.history.setdefault('population', []).append([agent.population for agent in self.agents])
        self.history.setdefault('territory', []).append([len(agent.territory) for agent in self.agents])
        self.attribute_history.append(np.mean(cycle_attributes, axis=0))

        # relationship history snapshot
        rel_snapshot = {agent.agent_id: {'allies': set(agent.allies), 'enemies': set(agent.enemies)} for agent in self.agents}
        self.relationship_history.append(rel_snapshot)

    def run(self, num_cycles=None):
        cycles = num_cycles if num_cycles is not None else getattr(self.config, 'SIMULATION_CYCLES', 1)
        for c in range(cycles):
            self.step(c)
        return np.array(self.history["strategy"], dtype=object)
