"""
Optimized multi-agent simulation with performance improvements.

This module provides an optimized version of MultiAgentSimulation that uses
vectorized operations, caching, and memory-efficient storage.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

from civsim.simulation import CivilizationAgent
from civsim.config import SimulationConfig
from civsim.optimizations import VectorizedAgentUpdater, MemoryEfficientHistory, PerformanceMetrics
from civsim.validation import validate_positive, validate_list
from civsim.logger import get_logger


class OptimizedMultiAgentSimulation:
    """Optimized multi-agent simulation with performance enhancements."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        enable_vectorization: bool = True,
        enable_history_compression: bool = True,
    ):
        """Initialize optimized simulation.

        Args:
            config: Simulation configuration
            enable_vectorization: Enable vectorized operations
            enable_history_compression: Enable history compression
        """
        from civsim.config import default_config

        self.config = config if config is not None else default_config
        self.num_agents = validate_positive(
            getattr(self.config, "num_civilizations", 4), "num_civilizations"
        )
        self.grid_size = validate_positive(getattr(self.config, "grid_size", 20), "grid_size")

        # Performance options
        self.enable_vectorization = enable_vectorization
        self.enable_history_compression = enable_history_compression

        # Performance metrics
        self.metrics = PerformanceMetrics()

        # Seed RNG if provided
        seed = getattr(self.config, "random_seed", None)
        if seed is not None:
            np.random.seed(seed)

        # Initialize components
        self.global_resources = self._generate_resources(self.grid_size)
        self.h3_grid = list(self.global_resources.keys())

        # Create agents
        self.agents = [CivilizationAgent(i) for i in range(self.num_agents)]

        # Initialize per-agent relationship weights and territories
        self._initialize_agents()

        # Vectorized updater
        self.vectorized_updater = VectorizedAgentUpdater(self.metrics)

        # History storage
        self.history = (
            MemoryEfficientHistory(
                max_memory_size_mb=getattr(self.config, "max_memory_mb", 500),
                compression_threshold=getattr(self.config, "compression_threshold", 100),
            )
            if enable_history_compression
            else {}
        )

        self.technology_history = {agent.agent_id: [] for agent in self.agents}
        self.relationship_history = []
        self.event_history = []

        self.logger = get_logger(__name__)

    def _generate_resources(self, grid_size: int) -> Dict[str, float]:
        """Generate resource map.

        Args:
            grid_size: Size of grid

        Returns:
            Resource dictionary
        """
        return {f"h3_{i}": float(np.random.rand() * 100) for i in range(max(1, grid_size))}

    def _initialize_agents(self) -> None:
        """Initialize agent territories and relationships."""
        # Initialize territories
        territory_per_agent = max(1, len(self.h3_grid) // max(1, len(self.agents)))
        for i, agent in enumerate(self.agents):
            start_idx = i * territory_per_agent
            end_idx = start_idx + territory_per_agent
            agent.territory = set(self.h3_grid[start_idx:end_idx])

        # Initialize relationships
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

    def _get_neighbors(self, agent: CivilizationAgent) -> Dict[int, tuple]:
        """Get neighbors for an agent.

        Args:
            agent: Agent to get neighbors for

        Returns:
            Dictionary mapping neighbor_id -> (strength, relationship)
        """
        neighbors = {}
        for other in self.agents:
            if agent.agent_id == other.agent_id:
                continue

            shared_border = len(agent.territory & other.territory) > 0
            if shared_border:
                weight = agent.relationship_weights.get(other.agent_id, 0.0)
                neighbors[other.agent_id] = (other.strength, weight)

        return neighbors

    def _compute_tech_spillover(self, agent_neighbors: Dict[int, Dict]) -> List[float]:
        """Compute technology spillover for all agents.

        Args:
            agent_neighbors: Dictionary mapping agent_id -> neighbors

        Returns:
            List of spillover amounts for each agent
        """
        tech_levels = {agent.agent_id: sum(agent.technology.values()) for agent in self.agents}
        avg_tech_level = sum(tech_levels.values()) / max(1, len(tech_levels))

        spillover_amounts = []
        tech_spillover_effect = getattr(self.config, "TECH_SPILLOVER_EFFECT", 0.3)

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

                    rel_threshold = getattr(self.config, "RELATIONSHIP_THRESHOLD", 0.5)
                    if rel > rel_threshold:
                        spillover_coefficient = 0.05
                    elif rel < -rel_threshold:
                        spillover_coefficient = 0.01
                    else:
                        spillover_coefficient = 0.02

                    neighbor_tech_bonus += tech_gap * spillover_coefficient * tech_spillover_effect

            global_spillover = (
                max(0.0, avg_tech_level - sum(agent.technology.values()))
                * 0.01
                * tech_spillover_effect
            )

            total_spillover = neighbor_tech_bonus + global_spillover
            spillover_amounts.append(total_spillover)
            agent.tech_spillover_received = total_spillover

        return spillover_amounts

    def _batch_update_resources_population(self) -> None:
        """Batch update resources and population using vectorized operations."""
        if not self.enable_vectorization:
            self._sequential_update_resources_population()
            return

        # Prepare arrays for vectorized operations
        agent_ids = [agent.agent_id for agent in self.agents]
        resources = np.array([agent.resources for agent in self.agents])
        populations = np.array([agent.population for agent in self.agents])
        health = np.array([agent.health for agent in self.agents])
        resource_bonuses = np.array(
            [agent.tech_bonuses.get("resources", 1.0) for agent in self.agents]
        )
        territory_value_bonuses = np.array(
            [agent.tech_bonuses.get("territory_value", 1.0) for agent in self.agents]
        )
        population_growth_bonuses = np.array(
            [agent.tech_bonuses.get("population_growth", 1.0) for agent in self.agents]
        )
        infrastructure = np.array([agent.infrastructure for agent in self.agents])

        # Compute base outputs
        base_outputs = np.array(
            [
                sum(self.global_resources.get(h3, 0.0) for h3 in agent.territory)
                * getattr(self.config, "RESOURCE_BASE_OUTPUT", 0.1)
                for agent in self.agents
            ]
        )

        # Update resources using vectorized operations
        consumption_rates = np.array(
            [
                getattr(self.config, "RESOURCE_CONSUMPTION_RATE", 0.01)
                + agent.population * getattr(self.config, "POPULATION_RESOURCE_CONSUMPTION", 0.05)
                for agent in self.agents
            ]
        ) / np.maximum(resources, 1.0)

        updated_resources = self.vectorized_updater.batch_update_resources(
            agent_ids,
            resources,
            base_outputs,
            consumption_rates,
            resource_bonuses,
            territory_value_bonuses,
        )

        # Update population using vectorized operations
        resource_per_capita = np.divide(
            updated_resources, populations, out=np.zeros_like(populations), where=populations > 0
        )

        population_growth_rates = np.array(
            [getattr(self.config, "POPULATION_GROWTH_RATE", 0.02) for _ in self.agents]
        )

        population_caps = np.array(
            [getattr(self.config, "POPULATION_GROWTH_CAP", float("inf")) for _ in self.agents]
        )

        updated_populations = self.vectorized_updater.batch_update_population(
            populations,
            resource_per_capita,
            population_growth_rates,
            health,
            population_growth_bonuses,
            population_caps if np.all(np.isfinite(population_caps)) else None,
        )

        # Apply updates
        for i, agent in enumerate(self.agents):
            agent.resources = updated_resources[i]
            agent.population = updated_populations[i]

    def _sequential_update_resources_population(self) -> None:
        """Sequential update of resources and population (fallback)."""
        for agent in self.agents:
            # Resource output
            base_output = sum(
                self.global_resources.get(h3, 0.0) for h3 in agent.territory
            ) * getattr(self.config, "RESOURCE_BASE_OUTPUT", 0.1)

            resource_bonus = agent.tech_bonuses.get("resources", 1.0)
            territory_value_bonus = agent.tech_bonuses.get("territory_value", 1.0)
            actual_output = base_output * resource_bonus * territory_value_bonus

            # Add spillover
            actual_output += getattr(agent, "tech_spillover_received", 0.0) * 0.1
            agent.resources += actual_output

            # Consumption
            base_consumption = (
                getattr(self.config, "RESOURCE_CONSUMPTION_RATE", 0.01) * agent.resources
            )
            population_consumption = agent.population * getattr(
                self.config, "POPULATION_RESOURCE_CONSUMPTION", 0.05
            )
            agent.resources = max(
                0.0, agent.resources - (base_consumption + population_consumption)
            )

            # Population growth
            if agent.resources > 0 and agent.population > 0:
                resource_per_capita = agent.resources / agent.population
                resource_factor = min(
                    resource_per_capita
                    / getattr(self.config, "RESOURCE_PER_CAPITA_FOR_MAX_GROWTH", 10),
                    1.0,
                )
                growth_rate = (
                    getattr(self.config, "POPULATION_GROWTH_RATE", 0.02)
                    * resource_factor
                    * agent.tech_bonuses.get("population_growth", 1.0)
                    * agent.health
                )
                population_increase = agent.population * growth_rate

                cap = getattr(self.config, "POPULATION_GROWTH_CAP", None)
                if cap is not None and cap > 0:
                    agent.population = min(cap, agent.population + population_increase)
                else:
                    agent.population += population_increase

    def step(self, cycle: Optional[int] = None) -> None:
        """Execute one simulation step.

        Args:
            cycle: Current cycle number
        """
        cycle = cycle if cycle is not None else 0

        # Get neighbors for all agents
        agent_neighbors = {agent.agent_id: self._get_neighbors(agent) for agent in self.agents}

        # Compute technology spillover
        spillover_amounts = self._compute_tech_spillover(agent_neighbors)

        # Update resources and population
        self._batch_update_resources_population()

        # Strategy decisions and execution
        for agent in self.agents:
            neighbors = agent_neighbors[agent.agent_id]
            strategy = agent.decide_strategy(neighbors, self.global_resources)
            agent.execute_strategy(strategy, neighbors, self.global_resources)

        # Update relationships
        self._update_relationships()

        # Record history
        self._record_history(cycle)

        # Log if enabled
        if (
            getattr(self.config, "print_logs", True)
            and cycle % getattr(self.config, "log_interval", 10) == 0
        ):
            self.logger.info(f"Completed cycle {cycle}")

    def _update_relationships(self) -> None:
        """Update relationships between agents."""
        relationship_change_rate = getattr(self.config, "relationship_change_rate", 0.05)

        for agent in self.agents:
            for other in self.agents:
                if agent.agent_id == other.agent_id:
                    continue

                current = agent.relationship_weights.get(other.agent_id, 0.0)
                change = np.random.uniform(-relationship_change_rate, relationship_change_rate)
                new_weight = np.clip(current + change, -1.0, 1.0)

                agent.relationship_weights[other.agent_id] = new_weight

                # Update allies/enemies sets
                if new_weight > 0.5:
                    agent.allies.add(other.agent_id)
                    agent.enemies.discard(other.agent_id)
                elif new_weight < -0.5:
                    agent.enemies.add(other.agent_id)
                    agent.allies.discard(other.agent_id)
                else:
                    agent.allies.discard(other.agent_id)
                    agent.enemies.discard(other.agent_id)

    def _record_history(self, cycle: int) -> None:
        """Record history for current cycle.

        Args:
            cycle: Current cycle number
        """
        cycle_data = {
            "resources": np.array([agent.resources for agent in self.agents]),
            "strength": np.array([agent.strength for agent in self.agents]),
            "population": np.array([agent.population for agent in self.agents]),
            "territory": np.array([len(agent.territory) for agent in self.agents]),
            "infrastructure": np.array([agent.infrastructure for agent in self.agents]),
            "stability": np.array([agent.stability for agent in self.agents]),
        }

        if self.enable_history_compression:
            self.history.append(cycle, cycle_data)
        else:
            for key, value in cycle_data.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

        # Record technology history
        for agent in self.agents:
            self.technology_history[agent.agent_id].append(
                {
                    "cycle": cycle,
                    "technologies": dict(agent.technology),
                    "current_research": agent.current_research,
                    "research_progress": agent.research_progress,
                    "research_cost": agent.research_cost,
                    "tech_bonuses": dict(agent.tech_bonuses),
                }
            )

        # Record relationships
        rel_snapshot = {
            agent.agent_id: {"allies": set(agent.allies), "enemies": set(agent.enemies)}
            for agent in self.agents
        }
        self.relationship_history.append(rel_snapshot)

    def run(self, num_cycles: Optional[int] = None) -> np.ndarray:
        """Run simulation for specified number of cycles.

        Args:
            num_cycles: Number of cycles to run

        Returns:
            Strategy history array
        """
        cycles = (
            num_cycles if num_cycles is not None else getattr(self.config, "simulation_cycles", 100)
        )

        self.logger.info(f"Starting simulation for {cycles} cycles")
        self.logger.info(f"Agents: {self.num_agents}, Grid size: {self.grid_size}")
        self.logger.info(
            f"Vectorization: {self.enable_vectorization}, "
            f"Compression: {self.enable_history_compression}"
        )

        import time

        start_time = time.perf_counter()

        for cycle in range(cycles):
            self.step(cycle)

        elapsed = time.perf_counter() - start_time

        self.logger.info(f"Simulation completed in {elapsed:.2f} seconds")
        self.logger.info(f"Vectorized operations: {self.metrics.vectorized_operations}")

        # Return strategy history
        if "strategy" in self.history:
            return self.history.get("strategy", np.array([]))
        else:
            return np.array([])

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics.

        Returns:
            Performance metrics object
        """
        return self.metrics

    def save_results(self, output_dir: str = "results") -> None:
        """Save simulation results to files.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save history
        if self.enable_history_compression:
            history_file = output_path / "simulation_history.npz"
            self.history.save_to_disk(str(history_file))
            self.logger.info(f"History saved to {history_file}")
        else:
            history_file = output_path / "simulation_history.npz"
            np.savez_compressed(history_file, **self.history)
            self.logger.info(f"History saved to {history_file}")

        # Save technology history
        import json

        tech_file = output_path / "technology_history.json"
        with open(tech_file, "w", encoding="utf-8") as f:
            json.dump(self.technology_history, f, indent=2)
        self.logger.info(f"Technology history saved to {tech_file}")

        # Save relationships
        rel_file = output_path / "relationship_history.json"
        with open(rel_file, "w", encoding="utf-8") as f:
            json.dump(self.relationship_history, f, indent=2, default=list)
        self.logger.info(f"Relationship history saved to {rel_file}")

        # Save performance metrics
        metrics_file = output_path / "performance_metrics.txt"
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("Performance Metrics:\n")
            f.write(f"Vectorized operations: {self.metrics.vectorized_operations}\n")
            f.write(f"Cache hit rate: {self.metrics.cache_hit_rate():.2%}\n")
        self.logger.info(f"Performance metrics saved to {metrics_file}")


__all__ = ["OptimizedMultiAgentSimulation"]
