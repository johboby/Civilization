"""
Optimization utilities for civilization simulation.

This module provides performance optimizations including vectorized operations,
caching mechanisms, and memory-efficient data structures.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from functools import lru_cache
from dataclasses import dataclass
import time


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""

    tech_bonus_cache_hits: int = 0
    tech_bonus_cache_misses: int = 0
    vectorized_operations: int = 0
    total_computation_time: float = 0.0

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.tech_bonus_cache_hits + self.tech_bonus_cache_misses
        return self.tech_bonus_cache_hits / total if total > 0 else 0.0


class TechBonusCache:
    """Cache for technology bonus calculations."""

    def __init__(self, max_size: int = 1000):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached entries
        """
        self.max_size = max_size
        self._cache: Dict[int, Dict] = {}
        self._access_order: List[int] = []

    def get(self, tech_hash: int) -> Optional[Dict]:
        """Get cached bonuses.

        Args:
            tech_hash: Hash of technology state

        Returns:
            Cached bonuses or None if not found
        """
        if tech_hash in self._cache:
            self._access_order.remove(tech_hash)
            self._access_order.append(tech_hash)
            return self._cache[tech_hash]
        return None

    def set(self, tech_hash: int, bonuses: Dict) -> None:
        """Cache bonuses.

        Args:
            tech_hash: Hash of technology state
            bonuses: Technology bonuses
        """
        if tech_hash in self._cache:
            self._access_order.remove(tech_hash)
        elif len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[tech_hash] = bonuses
        self._access_order.append(tech_hash)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class VectorizedAgentUpdater:
    """Vectorized operations for updating multiple agents."""

    def __init__(self, metrics: Optional[PerformanceMetrics] = None):
        """Initialize vectorized updater.

        Args:
            metrics: Performance metrics to track
        """
        self.metrics = metrics

    def batch_update_resources(
        self,
        agent_ids: List[int],
        current_resources: np.ndarray,
        base_outputs: np.ndarray,
        consumption_rates: np.ndarray,
        resource_bonuses: np.ndarray,
        territory_value_bonuses: np.ndarray,
    ) -> np.ndarray:
        """Update resources for multiple agents using vectorized operations.

        Args:
            agent_ids: List of agent IDs
            current_resources: Current resource levels
            base_outputs: Base resource outputs
            consumption_rates: Consumption rates
            resource_bonuses: Resource bonuses
            territory_value_bonuses: Territory value bonuses

        Returns:
            Updated resource levels
        """
        if self.metrics:
            self.metrics.vectorized_operations += 1

        # Vectorized production calculation
        production = base_outputs * resource_bonuses * territory_value_bonuses

        # Vectorized consumption calculation
        consumption = current_resources * consumption_rates

        # Vectorized update
        updated_resources = current_resources + production - consumption

        return np.maximum(updated_resources, 0.0)

    def batch_update_population(
        self,
        current_populations: np.ndarray,
        resource_per_capita: np.ndarray,
        population_growth_rates: np.ndarray,
        health_factors: np.ndarray,
        tech_bonuses: np.ndarray,
        population_caps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Update population for multiple agents using vectorized operations.

        Args:
            current_populations: Current population levels
            resource_per_capita: Resources per capita
            population_growth_rates: Base growth rates
            health_factors: Health factors
            tech_bonuses: Technology bonuses
            population_caps: Optional population caps

        Returns:
            Updated population levels
        """
        if self.metrics:
            self.metrics.vectorized_operations += 1

        # Vectorized resource factor calculation
        resource_factor = np.minimum(resource_per_capita / 10.0, 1.0)

        # Vectorized growth rate calculation
        growth_rates = population_growth_rates * resource_factor * health_factors * tech_bonuses

        # Vectorized population update
        population_increase = current_populations * growth_rates
        updated_populations = current_populations + population_increase

        # Apply caps if provided
        if population_caps is not None:
            updated_populations = np.minimum(updated_populations, population_caps)

        return updated_populations

    def batch_update_strength(
        self,
        current_strengths: np.ndarray,
        defense_probs: np.ndarray,
        defense_bonuses: np.ndarray,
        resource_consumption: float = 0.05,
    ) -> np.ndarray:
        """Update strength for multiple agents using vectorized operations.

        Args:
            current_strengths: Current strength levels
            defense_probs: Defense probabilities
            defense_bonuses: Defense bonuses
            resource_consumption: Resource consumption rate

        Returns:
            Updated strength levels
        """
        if self.metrics:
            self.metrics.vectorized_operations += 1

        # Vectorized strength increase
        strength_multiplier = 1.0 + (0.05 * defense_probs * defense_bonuses)

        # Vectorized strength update
        updated_strengths = current_strengths * strength_multiplier * (1.0 - resource_consumption)

        return np.maximum(updated_strengths, 0.0)

    def batch_normalize_strategies(self, strategy_arrays: np.ndarray) -> np.ndarray:
        """Normalize strategy probabilities for multiple agents.

        Args:
            strategy_arrays: Raw strategy arrays (N x M where N=agents, M=strategies)

        Returns:
            Normalized strategy arrays
        """
        if self.metrics:
            self.metrics.vectorized_operations += 1

        # Ensure non-negative
        clipped = np.maximum(strategy_arrays, 0.0)

        # Calculate sums
        sums = np.sum(clipped, axis=1, keepdims=True)

        # Handle zero sums
        zero_mask = sums.flatten() == 0.0
        if np.any(zero_mask):
            uniform = np.ones_like(clipped) / clipped.shape[1]
            clipped[zero_mask] = uniform[zero_mask]
            sums[zero_mask] = 1.0

        # Normalize
        normalized = clipped / sums

        return normalized

    def compute_neighbor_matrices(
        self, num_agents: int, territories: List[set], relationship_weights: List[Dict[int, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute neighbor matrices using vectorized operations.

        Args:
            num_agents: Number of agents
            territories: List of territory sets for each agent
            relationship_weights: List of relationship weight dictionaries

        Returns:
            Tuple of (neighbor_strength_matrix, neighbor_relationship_matrix)
        """
        if self.metrics:
            self.metrics.vectorized_operations += 1

        neighbor_strength = np.zeros((num_agents, num_agents))
        neighbor_relationship = np.zeros((num_agents, num_agents))

        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    shared = len(territories[i] & territories[j])
                    if shared > 0:
                        neighbor_relationship[i, j] = relationship_weights[i].get(j, 0.0)
                        # Strength will be set later by agent objects
                        neighbor_strength[i, j] = 1.0  # Placeholder

        return neighbor_strength, neighbor_relationship


class MemoryEfficientHistory:
    """Memory-efficient history storage with compression."""

    def __init__(self, max_memory_size_mb: int = 500, compression_threshold: int = 1000):
        """Initialize memory-efficient history.

        Args:
            max_memory_size_mb: Maximum memory size in MB
            compression_threshold: Number of cycles after which to compress
        """
        self.max_memory_size_mb = max_memory_size_mb
        self.compression_threshold = compression_threshold
        self.history: Dict[str, List] = {}
        self.cycle_index: int = 0
        self.compressed_cycles: List[int] = []

    def append(self, cycle: int, data: Dict[str, np.ndarray]) -> None:
        """Append data for a cycle.

        Args:
            cycle: Cycle number
            data: Dictionary of arrays to store
        """
        for key, value in data.items():
            if key not in self.history:
                self.history[key] = []

            self.history[key].append(value)

        self.cycle_index += 1

        # Check if compression is needed
        if self.cycle_index >= self.compression_threshold:
            self._compress()

    def _compress(self) -> None:
        """Compress history data."""
        for key in self.history:
            if isinstance(self.history[key], list) and len(self.history[key]) > 0:
                # Convert to numpy array for better memory efficiency
                arr = np.array(self.history[key])
                self.history[key] = arr
                self.compressed_cycles.append(self.cycle_index)

    def get(self, key: str, cycle: int | None = None) -> np.ndarray:
        """Get history data.

        Args:
            key: Data key
            cycle: Optional cycle index

        Returns:
            Data array
        """
        if key not in self.history:
            return np.array([])

        data = self.history[key]

        if cycle is not None:
            if isinstance(data, np.ndarray):
                return data[cycle]
            else:
                return data[cycle]

        return data if isinstance(data, np.ndarray) else np.array(data)

    def get_slice(self, key: str, start_cycle: int, end_cycle: int) -> np.ndarray:
        """Get slice of history data.

        Args:
            key: Data key
            start_cycle: Start cycle (inclusive)
            end_cycle: End cycle (exclusive)

        Returns:
            Data array slice
        """
        if key not in self.history:
            return np.array([])

        data = self.history[key]
        arr = data if isinstance(data, np.ndarray) else np.array(data)

        return arr[start_cycle:end_cycle]

    def save_to_disk(self, filename: str) -> None:
        """Save history to disk and clear from memory.

        Args:
            filename: Output filename
        """
        np.savez_compressed(filename, **self.history)
        self.clear()

    def load_from_disk(self, filename: str) -> None:
        """Load history from disk.

        Args:
            filename: Input filename
        """
        data = np.load(filename, allow_pickle=True)
        self.history = {key: data[key] for key in data.files}
        self.cycle_index = len(next(iter(self.history.values())))

    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()
        self.cycle_index = 0
        self.compressed_cycles.clear()

    def estimate_memory_size_mb(self) -> float:
        """Estimate current memory usage in MB.

        Returns:
            Estimated memory size in MB
        """
        total_size = 0
        for key, data in self.history.items():
            if isinstance(data, np.ndarray):
                total_size += data.nbytes
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, np.ndarray):
                        total_size += item.nbytes
                    else:
                        total_size += np.array(item).nbytes

        return total_size / (1024 * 1024)


def compute_tech_hash(technology: Dict[str, int]) -> int:
    """Compute hash of technology state.

    Args:
        technology: Technology dictionary

    Returns:
        Hash value
    """
    return hash(frozenset(sorted(technology.items())))


def benchmark_function(
    func: Callable, *args, iterations: int = 100, **kwargs
) -> tuple[float, float]:
    """Benchmark a function execution time.

    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of iterations
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (mean_time, std_time)
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time


__all__ = [
    "PerformanceMetrics",
    "TechBonusCache",
    "VectorizedAgentUpdater",
    "MemoryEfficientHistory",
    "compute_tech_hash",
    "benchmark_function",
]
