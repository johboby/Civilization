"""Random events module for the civilization simulation.

This module implements a random event system that adds uncertainty and
realism to the simulation, including natural disasters, technological breakthroughs,
and social changes.
"""
import json
import random
from typing import Any, Callable, Dict, List, Optional

from .logger import get_logger


class EventRecord:
    """Record of a triggered event."""
    
    def __init__(
        self,
        cycle: int,
        event_id: str,
        event_name: str,
        description: str,
        affected_agents: List[int],
        severity: str,
        categories: List[str]
    ):
        self.cycle = cycle
        self.event_id = event_id
        self.event_name = event_name
        self.description = description
        self.affected_agents = affected_agents
        self.severity = severity
        self.categories = categories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cycle': self.cycle,
            'event_id': self.event_id,
            'event_name': self.event_name,
            'description': self.description,
            'affected_agents': self.affected_agents,
            'severity': self.severity,
            'categories': self.categories
        }


class EventInfo:
    """Information about an event type."""
    
    def __init__(
        self,
        name: str,
        description: str,
        probability: float,
        effect: Callable,
        severity: str,
        categories: List[str]
    ):
        self.name = name
        self.description = description
        self.probability = probability
        self.effect = effect
        self.severity = severity
        self.categories = categories


class RandomEventManager:
    """Manager for random events in the simulation."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize the random event manager.
        
        Args:
            config: Simulation configuration object.
        """
        self.config = config
        self.events: Dict[str, EventInfo] = self._initialize_events()
        self.event_history: List[EventRecord] = []
        self.logger = get_logger(__name__)
    
    def _initialize_events(self) -> Dict[str, EventInfo]:
        """Initialize the event library.
        
        Returns:
            Dictionary mapping event IDs to event information.
        """
        return {
            # Natural disaster events
            'minor_disaster': EventInfo(
                name="Minor Natural Disaster",
                description="A small-scale natural disaster occurred, affecting some civilizations",
                probability=0.15,
                effect=self._minor_disaster_effect,
                severity="minor",
                categories=["natural_disaster"]
            ),
            'major_disaster': EventInfo(
                name="Major Natural Disaster",
                description="A major natural disaster occurred, severely affecting multiple civilizations",
                probability=0.05,
                effect=self._major_disaster_effect,
                severity="major",
                categories=["natural_disaster"]
            ),

            # Technology breakthrough events
            'tech_breakthrough': EventInfo(
                name="Technology Breakthrough",
                description="An important technology was accidentally discovered, accelerating technological development",
                probability=0.08,
                effect=self._tech_breakthrough_effect,
                severity="positive",
                categories=["technology"]
            ),

            # Social reform events
            'social_reform': EventInfo(
                name="Social Reform",
                description="An important social reform improved civilization's organizational efficiency",
                probability=0.07,
                effect=self._social_reform_effect,
                severity="positive",
                categories=["social"]
            ),

            # Resource discovery events
            'resource_discovery': EventInfo(
                name="Resource Discovery",
                description="New resource reserves were discovered, increasing resource production",
                probability=0.1,
                effect=self._resource_discovery_effect,
                severity="positive",
                categories=["resource"]
            ),

            # New event types
            'pandemic': EventInfo(
                name="Pandemic Outbreak",
                description="A severe pandemic spread among civilizations, causing significant population decline",
                probability=0.04,
                effect=self._pandemic_effect,
                severity="major",
                categories=["natural_disaster", "health"]
            ),

            'migration_wave': EventInfo(
                name="Migration Wave",
                description="Large numbers of migrants flooded certain civilizations, bringing opportunities and challenges",
                probability=0.06,
                effect=self._migration_wave_effect,
                severity="positive",
                categories=["social", "population"]
            ),

            'trade_agreement': EventInfo(
                name="Trade Agreement",
                description="Trade agreements were signed between civilizations, promoting economic prosperity",
                probability=0.07,
                effect=self._trade_agreement_effect,
                severity="positive",
                categories=["diplomatic", "economic"]
            ),

            'cultural_exchange': EventInfo(
                name="Cultural Exchange",
                description="Cultural exchanges between civilizations promoted technological progress and social stability",
                probability=0.06,
                effect=self._cultural_exchange_effect,
                severity="positive",
                categories=["cultural", "technology"]
            ),

            'resource_depletion': EventInfo(
                name="Resource Depletion",
                description="Some key resources began to deplete, affecting civilization development",
                probability=0.05,
                effect=self._resource_depletion_effect,
                severity="minor",
                categories=["resource", "negative"]
            ),

            'diplomatic_crisis': EventInfo(
                name="Diplomatic Crisis",
                description="Diplomatic relations between civilizations deteriorated, potentially leading to conflicts",
                probability=0.05,
                effect=self._diplomatic_crisis_effect,
                severity="major",
                categories=["diplomatic", "negative"]
            ),
        }
    
    def trigger_event(
        self,
        agents: List[Any],
        cycle: int
    ) -> Optional[EventRecord]:
        """Trigger a random event.
        
        Args:
            agents: List of civilization agents.
            cycle: Current simulation cycle.
            
        Returns:
            Event record if an event was triggered, None otherwise.
        """
        event_probability_modifier = getattr(
            self.config, 'EVENT_PROBABILITY_MODIFIER', 1.0
        )
        
        for event_id, event_info in self.events.items():
            adjusted_probability = event_info.probability * event_probability_modifier
            
            if random.random() < adjusted_probability:
                affected_agents = event_info.effect(agents)
                
                event_record = EventRecord(
                    cycle=cycle,
                    event_id=event_id,
                    event_name=event_info.name,
                    description=event_info.description,
                    affected_agents=affected_agents,
                    severity=event_info.severity,
                    categories=event_info.categories
                )
                
                self.event_history.append(event_record)

                print_logs = getattr(self.config, 'PRINT_LOGS', True) if self.config else True
                if print_logs:
                    self.logger.info(
                        f"Cycle {cycle}: {event_info.name} - {event_info.description}"
                    )
                    if affected_agents:
                        self.logger.info(f"  Affected civilizations: {affected_agents}")
                
                return event_record
        
        return None
    
    def _minor_disaster_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a minor natural disaster.
        
        Args:
            agents: List of civilization agents.
            
        Returns:
            List of affected agent IDs.
        """
        affected_count = min(random.randint(1, 2), len(agents))
        affected_agents = random.sample(agents, affected_count)
        
        for agent in affected_agents:
            agent.resources *= random.uniform(0.7, 0.9)
            agent.population *= random.uniform(0.8, 0.95)
            agent.strength *= random.uniform(0.85, 0.95)
            
        return [agent.agent_id for agent in affected_agents]
    
    def _major_disaster_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a major natural disaster.
        
        Args:
            agents: List of civilization agents.
            
        Returns:
            List of affected agent IDs.
        """
        affected_count = max(1, len(agents) // 2 + random.randint(0, len(agents) // 2))
        affected_agents = random.sample(agents, min(affected_count, len(agents)))
        
        for agent in affected_agents:
            agent.resources *= random.uniform(0.4, 0.7)
            agent.population *= random.uniform(0.6, 0.8)
            agent.strength *= random.uniform(0.5, 0.7)
            
        return [agent.agent_id for agent in affected_agents]
    
    def _tech_breakthrough_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a technological breakthrough.
        
        Args:
            agents: List of civilization agents.
            
        Returns:
            List of affected agent IDs.
        """
        affected_agent = random.choice(agents)
        
        tech_keys = list(affected_agent.technology.keys())
        if tech_keys:
            tech_to_boost = random.choice(tech_keys)
            current_level = affected_agent.technology[tech_to_boost]
            affected_agent.technology[tech_to_boost] = min(
                5, current_level + random.randint(1, 2)
            )
            
            if hasattr(affected_agent, 'update_tech_bonuses'):
                affected_agent.update_tech_bonuses()
        
        return [affected_agent.agent_id]
    
    def _social_reform_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a social reform.
        
        Args:
            agents: List of civilization agents.
            
        Returns:
            List of affected agent IDs.
        """
        affected_agent = random.choice(agents)
        
        if hasattr(affected_agent, 'infrastructure'):
            affected_agent.infrastructure *= random.uniform(1.1, 1.3)
        if hasattr(affected_agent, 'stability'):
            affected_agent.stability = min(1.5, affected_agent.stability * random.uniform(1.1, 1.2))
        if hasattr(affected_agent, 'decision_quality'):
            affected_agent.decision_quality *= random.uniform(1.1, 1.2)
        
        return [affected_agent.agent_id]
    
    def _resource_discovery_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a resource discovery.
        
        Args:
            agents: List of civilization agents.
            
        Returns:
            List of affected agent IDs.
        """
        affected_count = random.randint(1, min(2, len(agents)))
        affected_agents = random.sample(agents, affected_count)
        
        for agent in affected_agents:
            agent.resources *= random.uniform(1.3, 1.8)
            
        return [agent.agent_id for agent in affected_agents]
    
    def get_event_history(self) -> List[EventRecord]:
        """Get the event history.
        
        Returns:
            List of event records.
        """
        return self.event_history
    
    def save_event_history(self, filename: str = "event_history.json") -> None:
        """Save the event history to a file.
        
        Args:
            filename: Path to the output file.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                [record.to_dict() for record in self.event_history],
                f,
                ensure_ascii=False,
                indent=2
            )
        self.logger.info(f"Event history saved to {filename}")
    
    def get_events_by_category(self, category: str) -> Dict[str, EventInfo]:
        """Get events filtered by category.
        
        Args:
            category: Category to filter by.
            
        Returns:
            Dictionary of events in the specified category.
        """
        return {
            event_id: event_info
            for event_id, event_info in self.events.items()
            if category in event_info.categories
        }
    
    def get_events_by_severity(self, severity: str) -> Dict[str, EventInfo]:
        """Get events filtered by severity.

        Args:
            severity: Severity level to filter by.

        Returns:
            Dictionary of events with the specified severity.
        """
        return {
            event_id: event_info
            for event_id, event_info in self.events.items()
            if event_info.severity == severity
        }

    def _pandemic_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a pandemic outbreak.

        Args:
            agents: List of civilization agents.

        Returns:
            List of affected agent IDs.
        """
        # Pandemic affects random agents based on population density
        affected_count = random.randint(1, min(3, len(agents)))
        affected_agents = random.sample(agents, affected_count)

        for agent in affected_agents:
            # Severe population reduction
            agent.population *= random.uniform(0.5, 0.7)
            # Moderate resource reduction
            agent.resources *= random.uniform(0.6, 0.8)
            # Stability decreases significantly
            if hasattr(agent, 'stability'):
                agent.stability = max(0.5, agent.stability * random.uniform(0.5, 0.7))

        return [agent.agent_id for agent in affected_agents]

    def _migration_wave_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a migration wave.

        Args:
            agents: List of civilization agents.

        Returns:
            List of affected agent IDs.
        """
        # Select 2-3 agents to receive migrants
        affected_count = random.randint(2, min(3, len(agents)))
        affected_agents = random.sample(agents, affected_count)

        for agent in affected_agents:
            # Population increases significantly
            agent.population *= random.uniform(1.3, 1.6)
            # Resources need to accommodate new population
            agent.resources *= random.uniform(0.9, 1.1)
            # Cultural exchange potential increases
            if hasattr(agent, 'infrastructure'):
                agent.infrastructure *= random.uniform(1.05, 1.15)
            # Temporary stability decrease due to integration
            if hasattr(agent, 'stability'):
                agent.stability = max(0.8, agent.stability * random.uniform(0.9, 1.0))

        return [agent.agent_id for agent in affected_agents]

    def _trade_agreement_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a trade agreement.

        Args:
            agents: List of civilization agents.

        Returns:
            List of affected agent IDs.
        """
        # Select 2-4 agents to participate in trade
        affected_count = random.randint(2, min(4, len(agents)))
        affected_agents = random.sample(agents, affected_count)

        for agent in affected_agents:
            # Resources increase due to trade
            agent.resources *= random.uniform(1.2, 1.5)
            # Infrastructure improves with trade routes
            if hasattr(agent, 'infrastructure'):
                agent.infrastructure *= random.uniform(1.1, 1.2)
            # Stability increases with economic prosperity
            if hasattr(agent, 'stability'):
                agent.stability = min(1.5, agent.stability * random.uniform(1.05, 1.15))
            # Improve relationships with other participating agents
            for other_agent in affected_agents:
                if other_agent.agent_id != agent.agent_id:
                    agent.relationship_weights[other_agent.agent_id] = min(
                        1.0, agent.relationship_weights.get(other_agent.agent_id, 0) + 0.2
                    )

        return [agent.agent_id for agent in affected_agents]

    def _cultural_exchange_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a cultural exchange.

        Args:
            agents: List of civilization agents.

        Returns:
            List of affected agent IDs.
        """
        # Select 2-3 agents for cultural exchange
        affected_count = random.randint(2, min(3, len(agents)))
        affected_agents = random.sample(agents, affected_count)

        for agent in affected_agents:
            # Boost research due to cultural exchange
            if hasattr(agent, 'research_speed'):
                agent.research_speed *= random.uniform(1.1, 1.3)
            # Stability increases with cultural enrichment
            if hasattr(agent, 'stability'):
                agent.stability = min(1.5, agent.stability * random.uniform(1.05, 1.15))
            # Slight population growth from cultural openness
            agent.population *= random.uniform(1.02, 1.08)

        return [agent.agent_id for agent in affected_agents]

    def _resource_depletion_effect(self, agents: List[Any]) -> List[int]:
        """Effect of resource depletion.

        Args:
            agents: List of civilization agents.

        Returns:
            List of affected agent IDs.
        """
        # Affect 2-4 agents
        affected_count = random.randint(2, min(4, len(agents)))
        affected_agents = random.sample(agents, affected_count)

        for agent in affected_agents:
            # Resources decrease significantly
            agent.resources *= random.uniform(0.4, 0.7)
            # Production slows down
            if hasattr(agent, 'infrastructure'):
                agent.infrastructure *= random.uniform(0.8, 0.95)
            # Stability may decrease
            if hasattr(agent, 'stability'):
                agent.stability = max(0.7, agent.stability * random.uniform(0.85, 1.0))

        return [agent.agent_id for agent in affected_agents]

    def _diplomatic_crisis_effect(self, agents: List[Any]) -> List[int]:
        """Effect of a diplomatic crisis.

        Args:
            agents: List of civilization agents.

        Returns:
            List of affected agent IDs.
        """
        # Select 2-3 agents for diplomatic crisis
        affected_count = random.randint(2, min(3, len(agents)))
        affected_agents = random.sample(agents, affected_count)

        # Deteriorate relationships between affected agents
        for i, agent in enumerate(affected_agents):
            for j, other_agent in enumerate(affected_agents):
                if i != j:
                    # Relationship deteriorates
                    current_relation = agent.relationship_weights.get(other_agent.agent_id, 0)
                    agent.relationship_weights[other_agent.agent_id] = max(
                        -1.0, current_relation - random.uniform(0.4, 0.7)
                    )

            # Stability decreases
            if hasattr(agent, 'stability'):
                agent.stability = max(0.6, agent.stability * random.uniform(0.7, 0.9))

        return [agent.agent_id for agent in affected_agents]


__all__ = [
    'EventRecord',
    'EventInfo',
    'RandomEventManager',
]
