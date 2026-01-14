"""
Common utility functions for civilization simulation.

This module provides reusable utility functions to eliminate code duplication.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np


def calculate_bonus_multiplier(
    base_value: float, bonus_dict: Dict[str, float], bonus_key: str, default_multiplier: float = 1.0
) -> float:
    """Calculate final value with bonus multiplier applied.

    Args:
        base_value: Base value
        bonus_dict: Dictionary of bonuses
        bonus_key: Key to lookup in bonus dict
        default_multiplier: Default multiplier if key not found

    Returns:
        Value with bonus applied
    """
    multiplier = bonus_dict.get(bonus_key, default_multiplier)
    return base_value * multiplier


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Normalize probability array to sum to 1.

    Args:
        probabilities: Array of probabilities

    Returns:
        Normalized probabilities
    """
    total = np.sum(probabilities)
    if total <= 0:
        return np.ones_like(probabilities) / len(probabilities)
    return probabilities / total


def weighted_choice(
    items: List[Any], weights: List[float], random_state: Optional[np.random.Generator] = None
) -> Any:
    """Make weighted random choice from items.

    Args:
        items: List of items to choose from
        weights: List of weights for each item
        random_state: Optional random number generator

    Returns:
        Selected item
    """
    if random_state is None:
        random_state = np.random

    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)

    return random_state.choice(items, p=weights)


def compute_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two positions.

    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)

    Returns:
        Euclidean distance
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return np.sqrt(dx * dx + dy * dy)


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix.

    Args:
        positions: Array of shape (N, 2) containing positions

    Returns:
        Distance matrix of shape (N, N)
    """
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    return distances


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Division result or default
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def clip_and_log(
    value: float, min_val: float, max_val: float, name: str = "value", logger: Optional[Any] = None
) -> float:
    """Clip value to range and log if clamped.

    Args:
        value: Value to clip
        min_val: Minimum value
        max_val: Maximum value
        name: Name for logging
        logger: Optional logger

    Returns:
        Clipped value
    """
    original = value
    value = np.clip(value, min_val, max_val)

    if value != original and logger is not None:
        logger.debug(f"Clipped {name} from {original:.4f} to {value:.4f}")

    return value


def exponential_decay(current: float, target: float, rate: float, dt: float = 1.0) -> float:
    """Apply exponential decay towards target.

    Args:
        current: Current value
        target: Target value
        rate: Decay rate
        dt: Time step

    Returns:
        New value after decay
    """
    return current + (target - current) * (1 - np.exp(-rate * dt))


def sigmoid(x: float, center: float = 0.0, steepness: float = 1.0) -> float:
    """Compute sigmoid function.

    Args:
        x: Input value
        center: Center of sigmoid
        steepness: Steepness of sigmoid

    Returns:
        Sigmoid output between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax of scores.

    Args:
        scores: Input scores
        temperature: Temperature for softmax (higher = more uniform)

    Returns:
        Softmax probabilities
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    scores = scores / temperature
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def get_top_k_indices(values: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    """Get indices of top k values.

    Args:
        values: Array of values
        k: Number of top values to return
        largest: If True, get largest; otherwise get smallest

    Returns:
        Indices of top k values
    """
    if largest:
        return np.argpartition(values, -k)[-k:]
    else:
        return np.argpartition(values, k)[:k]


def rescale_range(
    value: float, old_min: float, old_max: float, new_min: float, new_max: float
) -> float:
    """Rescale value from one range to another.

    Args:
        value: Value to rescale
        old_min: Old range minimum
        old_max: Old range maximum
        new_min: New range minimum
        new_max: New range maximum

    Returns:
        Rescaled value
    """
    old_range = old_max - old_min
    new_range = new_max - new_min

    if old_range == 0:
        return new_min + new_range / 2

    return new_min + ((value - old_min) / old_range) * new_range


def format_number(value: float, precision: int = 2) -> str:
    """Format number for display.

    Args:
        value: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if abs(value) >= 1e6:
        return f"{value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary (later dicts override earlier ones)
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def compute_hash_dict(d: Dict[str, Any]) -> int:
    """Compute hash of dictionary.

    Args:
        d: Dictionary to hash

    Returns:
        Hash value
    """
    import hashlib
    import json

    sorted_str = json.dumps(d, sort_keys=True)
    return int(hashlib.md5(sorted_str.encode()).hexdigest(), 16)


def compute_hash_list(l: List[Any]) -> int:
    """Compute hash of list.

    Args:
        l: List to hash

    Returns:
        Hash value
    """
    import hashlib
    import json

    sorted_str = json.dumps(l, sort_keys=True)
    return int(hashlib.md5(sorted_str.encode()).hexdigest(), 16)


__all__ = [
    "calculate_bonus_multiplier",
    "normalize_probabilities",
    "weighted_choice",
    "compute_distance",
    "compute_distance_matrix",
    "safe_divide",
    "clip_and_log",
    "exponential_decay",
    "sigmoid",
    "softmax",
    "get_top_k_indices",
    "rescale_range",
    "format_number",
    "merge_dicts",
    "chunk_list",
    "compute_hash_dict",
    "compute_hash_list",
]
