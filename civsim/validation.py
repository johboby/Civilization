"""
Input validation utilities for civilization simulation.

This module provides validation functions to ensure data integrity and prevent errors.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, field: Optional[str] = None):
        """Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
        """
        self.message = message
        self.field = field
        super().__init__(f"{field}: {message}" if field else message)


def validate_positive(value: Any, field_name: str, allow_zero: bool = False) -> float:
    """Validate that a value is positive.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        allow_zero: Whether zero is allowed

    Returns:
        Validated float value

    Raises:
        ValidationError: If validation fails
    """
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"Must be a number", field_name)

    if allow_zero:
        if float_value < 0:
            raise ValidationError(f"Must be non-negative, got {float_value}", field_name)
    else:
        if float_value <= 0:
            raise ValidationError(f"Must be positive, got {float_value}", field_name)

    return float_value


def validate_probability(value: Any, field_name: str) -> float:
    """Validate that a value is a valid probability.

    Args:
        value: Value to validate
        field_name: Name of the field being validated

    Returns:
        Validated float value

    Raises:
        ValidationError: If validation fails
    """
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"Must be a number", field_name)

    if not 0.0 <= float_value <= 1.0:
        raise ValidationError(f"Must be between 0 and 1, got {float_value}", field_name)

    return float_value


def validate_range(
    value: Any, field_name: str, min_value: float, max_value: float, inclusive: bool = True
) -> float:
    """Validate that a value is within a range.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        inclusive: Whether range is inclusive

    Returns:
        Validated float value

    Raises:
        ValidationError: If validation fails
    """
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"Must be a number", field_name)

    if inclusive:
        if not min_value <= float_value <= max_value:
            raise ValidationError(
                f"Must be between {min_value} and {max_value}, got {float_value}", field_name
            )
    else:
        if not min_value < float_value < max_value:
            raise ValidationError(
                f"Must be strictly between {min_value} and {max_value}, got {float_value}",
                field_name,
            )

    return float_value


def validate_dict(
    value: Any, field_name: str, key_type: Optional[type] = None, value_type: Optional[type] = None
) -> Dict:
    """Validate that a value is a dictionary.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        key_type: Expected type for keys (optional)
        value_type: Expected type for values (optional)

    Returns:
        Validated dictionary

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationError(f"Must be a dictionary", field_name)

    if key_type is not None:
        for key in value.keys():
            if not isinstance(key, key_type):
                raise ValidationError(
                    f"Dictionary keys must be of type {key_type.__name__}", field_name
                )

    if value_type is not None:
        for val in value.values():
            if not isinstance(val, value_type):
                raise ValidationError(
                    f"Dictionary values must be of type {value_type.__name__}", field_name
                )

    return value


def validate_list(
    value: Any,
    field_name: str,
    item_type: Optional[type] = None,
    min_length: int = 0,
    max_length: Optional[int] = None,
) -> List:
    """Validate that a value is a list.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        item_type: Expected type for items (optional)
        min_length: Minimum allowed length
        max_length: Maximum allowed length (optional)

    Returns:
        Validated list

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, list):
        raise ValidationError(f"Must be a list", field_name)

    if len(value) < min_length:
        raise ValidationError(
            f"Must have at least {min_length} items, got {len(value)}", field_name
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"Must have at most {max_length} items, got {len(value)}", field_name)

    if item_type is not None:
        for i, item in enumerate(value):
            if not isinstance(item, item_type):
                raise ValidationError(
                    f"Item at index {i} must be of type {item_type.__name__}", field_name
                )

    return value


def validate_strategy_array(value: Any, field_name: str, expected_length: int) -> np.ndarray:
    """Validate that a value is a valid strategy array.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        expected_length: Expected length of strategy array

    Returns:
        Validated numpy array

    Raises:
        ValidationError: If validation fails
    """
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Must be convertible to array: {e}", field_name)

    if len(arr.shape) != 1:
        raise ValidationError(f"Must be 1-dimensional array, got shape {arr.shape}", field_name)

    if arr.shape[0] != expected_length:
        raise ValidationError(f"Must have length {expected_length}, got {arr.shape[0]}", field_name)

    if np.any(np.isnan(arr)):
        raise ValidationError(f"Must not contain NaN values", field_name)

    if np.any(arr < 0):
        raise ValidationError(f"Must not contain negative values", field_name)

    return arr


def validate_technology_dict(
    value: Any, field_name: str, valid_techs: Optional[set] = None
) -> Dict[str, int]:
    """Validate technology dictionary.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        valid_techs: Set of valid technology names (optional)

    Returns:
        Validated technology dictionary

    Raises:
        ValidationError: If validation fails
    """
    tech_dict = validate_dict(value, field_name, key_type=str, value_type=(int, float))

    for tech_name, level in tech_dict.items():
        if level < 0:
            raise ValidationError(
                f"Technology level for '{tech_name}' must be non-negative, got {level}", field_name
            )

        if valid_techs is not None and tech_name not in valid_techs:
            raise ValidationError(f"Invalid technology name: '{tech_name}'", field_name)

    return {k: int(v) for k, v in tech_dict.items()}


def validate_neighbors(
    value: Any, field_name: str, num_agents: Optional[int] = None
) -> Dict[int, Tuple[float, float]]:
    """Validate neighbors dictionary.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        num_agents: Total number of agents (optional)

    Returns:
        Validated neighbors dictionary

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationError(f"Must be a dictionary", field_name)

    for agent_id, neighbor_data in value.items():
        if not isinstance(agent_id, int):
            raise ValidationError(
                f"Agent ID must be integer, got {type(agent_id).__name__}", field_name
            )

        if num_agents is not None and (agent_id < 0 or agent_id >= num_agents):
            raise ValidationError(
                f"Agent ID {agent_id} out of valid range [0, {num_agents})", field_name
            )

        if not isinstance(neighbor_data, (list, tuple)) or len(neighbor_data) != 2:
            raise ValidationError(
                f"Neighbor data for agent {agent_id} must be a tuple of (strength, relationship)",
                field_name,
            )

        try:
            float(neighbor_data[0])
            float(neighbor_data[1])
        except (TypeError, ValueError):
            raise ValidationError(
                f"Neighbor data for agent {agent_id} must contain numbers", field_name
            )

        if not -1.0 <= neighbor_data[1] <= 1.0:
            raise ValidationError(
                f"Relationship weight for agent {agent_id} must be between -1 and 1", field_name
            )

    return value


def validate_config(config: Any) -> List[str]:
    """Validate simulation configuration.

    Args:
        config: Configuration object or dict

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        if hasattr(config, "validate"):
            errors.extend(config.validate())
        else:
            errors.append("Configuration must have a validate() method")
    except Exception as e:
        errors.append(f"Configuration validation failed: {e}")

    return errors


def sanitize_input(
    value: Any, field_name: str, default: Any = None, validator: Optional[callable] = None
) -> Any:
    """Sanitize and validate input, providing default on failure.

    Args:
        value: Input value
        field_name: Name of the field
        default: Default value to use if validation fails
        validator: Optional validation function

    Returns:
        Sanitized value or default
    """
    if validator is not None:
        try:
            return validator(value, field_name)
        except ValidationError:
            return default

    return value if value is not None else default


__all__ = [
    "ValidationError",
    "validate_positive",
    "validate_probability",
    "validate_range",
    "validate_dict",
    "validate_list",
    "validate_strategy_array",
    "validate_technology_dict",
    "validate_neighbors",
    "validate_config",
    "sanitize_input",
]
