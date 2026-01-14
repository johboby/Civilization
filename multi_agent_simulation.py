"""Top-level compatibility shim.

This module re-exports the small, stable APIs from the civsim package so
existing imports like ``from multi_agent_simulation import MultiAgentSimulation``
continue to work during the refactor.
"""
from civsim.multi_agent import MultiAgentSimulation
from civsim.simulation import CivilizationAgent
from civsim.technology import TechnologyManager
from civsim.strategy import DefaultDecisionEngine, AgentState
from civsim.config import SimulationConfig
from civsim.events import RandomEventManager
from civsim.evolution import AdvancedEvolution, ComplexResourceManager, CulturalInfluence

__all__ = [
    "MultiAgentSimulation",
    "CivilizationAgent",
    "TechnologyManager",
    "DefaultDecisionEngine",
    "AgentState",
    "SimulationConfig",
    "RandomEventManager",
    "AdvancedEvolution",
    "ComplexResourceManager",
    "CulturalInfluence",
]