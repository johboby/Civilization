"""Constants configuration module

This module contains all magic numbers and configuration constants to avoid hardcoding in code.
"""
from dataclasses import dataclass


@dataclass
class StrategyWeights:
    """Strategy weight constants"""
    EXPANSION: float = 0.2
    DEFENSE: float = 0.15
    TRADE: float = 0.25
    RESEARCH: float = 0.4
    DIPLOMACY: float = 0.2
    CULTURE: float = 0.15
    RELIGION: float = 0.1


@dataclass
class Thresholds:
    """Threshold constants"""
    RELATION_HOSTILE: float = -0.4
    RELATION_ALLY: float = 0.6
    RESOURCE_PRESSURE: float = 0.8
    POPULATION_THRESHOLD: float = 500.0
    MAX_THREAT_FACTOR: float = 2.0
    TERRITORY_OUTPUT_FRACTION: float = 0.1


@dataclass
class Multipliers:
    """Multiplier constants"""
    DEFENSE_ENHANCEMENT: float = 1.05
    RESOURCE_CONSUMPTION: float = 0.95
    TRADE_AMOUNT_BASE: float = 20.0
    CULTURE_GROWTH: float = 0.02
    RELIGION_SPREAD: float = 0.02
    DIPLOMACY_SUCCESS: float = 0.05
    TERRITORY_VALUE: float = 0.3
    THREAT_ADJUSTMENT: float = 0.1
    DEFENSE_BASE: float = 0.6
    MAX_DEFENSE_ADJUSTMENT: float = 0.3
    POPULATION_RESEARCH_CAP: float = 5.0
    INFRASTRUCTURE_BONUS_FACTOR: float = 0.5


@dataclass
class ResourceCoefficients:
    """Resource coefficient constants"""
    POPULATION_CONSUMPTION_RATE: float = 0.5
    RESOURCE_PER_CAPITA_MAX_GROWTH: float = 10.0
    BASE_OUTPUT_FRACTION: float = 0.1
    RESOURCE_REGENERATION_BASE: float = 0.02


@dataclass
class TechnologyPriorities:
    """Technology priority constants"""
    TOP_LEVEL_BONUS: float = 10.0
    LAGGING_MULTIPLIER: float = 2.0
    WEAK_MULTIPLIER: float = 1.5
    RESOURCE_BONUS_WEIGHT: float = 1.0
    TERRITORY_BONUS_WEIGHT: float = 0.8
    POPULATION_BONUS_WEIGHT: float = 0.7
    COST_NORMALIZATION: float = 100.0


@dataclass
class EventProbabilities:
    """Event probability constants"""
    MINOR_DISASTER: float = 0.15
    MAJOR_DISASTER: float = 0.05
    TECH_BREAKTHROUGH: float = 0.08
    SOCIAL_REFORM: float = 0.07
    RESOURCE_DISCOVERY: float = 0.10
    PANDEMIC: float = 0.04
    MIGRATION_WAVE: float = 0.06
    TRADE_AGREEMENT: float = 0.07
    CULTURAL_EXCHANGE: float = 0.06
    RESOURCE_DEPLETION: float = 0.05
    DIPLOMATIC_CRISIS: float = 0.05


@dataclass
class ResearchParameters:
    """Research parameter constants"""
    BASE_SPEED_MULTIPLIER: float = 1.0
    POPULATION_DIVISOR: float = 100.0
    INFRASTRUCTURE_BASE: float = 1.0
    INFRASTRUCTURE_BONUS_MULTIPLIER: float = 0.3
    RESOURCE_INVESTMENT_FRACTION: float = 0.1


# Create global constant instances
STRATEGY_WEIGHTS = StrategyWeights()
THRESHOLDS = Thresholds()
MULTIPLIERS = Multipliers()
RESOURCE_COEFFICIENTS = ResourceCoefficients()
TECHNOLOGY_PRIORITIES = TechnologyPriorities()
EVENT_PROBABILITIES = EventProbabilities()
RESEARCH_PARAMETERS = ResearchParameters()
