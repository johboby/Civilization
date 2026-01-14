"""civsim package — core simulation modules.

This package contains refactored civilization simulation modules,
providing a clean, maintainable architecture for the simulation system.
"""
from civsim.config import SimulationConfig, get_preset, config, default_config
from civsim.logger import SimulationLogger, init_logging, get_logger, debug, info, warning, error, critical
from civsim.technology import TechnologyManager, TechInfo, TechBonus
from civsim.strategy import StrategyDecisionEngine, DefaultDecisionEngine, RandomDecisionEngine, create_decision_engine, AgentState, StrategyResult
from civsim.events import RandomEventManager, EventRecord, EventInfo
from civsim.evolution import AdvancedEvolution, ComplexResourceManager, CulturalInfluence
from civsim.multi_agent import MultiAgentSimulation
from civsim.strategy_executor import StrategyExecutor
from civsim.relationship_manager import RelationshipManager
from civsim.constants import (
    StrategyWeights, Thresholds, Multipliers, ResourceCoefficients,
    TechnologyPriorities, EventProbabilities, ResearchParameters,
    STRATEGY_WEIGHTS, THRESHOLDS, MULTIPLIERS, RESOURCE_COEFFICIENTS,
    TECHNOLOGY_PRIORITIES, EVENT_PROBABILITIES, RESEARCH_PARAMETERS
)

__all__ = [
    # Configuration
    "SimulationConfig",
    "get_preset",
    "config",
    "default_config",

    # Logging
    "SimulationLogger",
    "init_logging",
    "get_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",

    # Technology
    "TechnologyManager",
    "TechInfo",
    "TechBonus",

    # Strategy
    "StrategyDecisionEngine",
    "DefaultDecisionEngine",
    "RandomDecisionEngine",
    "create_decision_engine",
    "AgentState",
    "StrategyResult",

    # Events
    "RandomEventManager",
    "EventRecord",
    "EventInfo",

    # Evolution
    "AdvancedEvolution",
    "ComplexResourceManager",
    "CulturalInfluence",

    # Multi-agent
    "MultiAgentSimulation",

    # Refactored components
    "StrategyExecutor",
    "RelationshipManager",

    # Constants
    "StrategyWeights",
    "Thresholds",
    "Multipliers",
    "ResourceCoefficients",
    "TechnologyPriorities",
    "EventProbabilities",
    "ResearchParameters",
    "STRATEGY_WEIGHTS",
    "THRESHOLDS",
    "MULTIPLIERS",
    "RESOURCE_COEFFICIENTS",
    "TECHNOLOGY_PRIORITIES",
    "EVENT_PROBABILITIES",
    "RESEARCH_PARAMETERS",
]
