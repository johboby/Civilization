# API Reference

## Core Classes

### CivilizationAgent

The main agent class representing a civilization.

#### Attributes
- `agent_id`: Unique identifier
- `strength`: Military strength
- `resources`: Resource reserves
- `population`: Population size
- `infrastructure`: Infrastructure level
- `stability`: Social stability
- `territory`: Controlled territory
- `technology`: Researched technologies

#### Methods
- `decide_strategy()`: Make strategic decisions
- `execute_strategy()`: Execute chosen strategy
- `update_tech_bonuses()`: Update technology bonuses

### MultiAgentSimulation

Main simulation orchestrator.

#### Methods
- `run(cycles)`: Run simulation for specified cycles
- `__init__(config)`: Initialize with configuration

### TechnologyManager

Manages the technology tree and research.

#### Methods
- `get_available_techs()`: Get researchable technologies
- `calculate_bonuses()`: Calculate technology bonuses
- `can_research()`: Check research eligibility

## Strategy Types

1. **Expansion**: Acquire more territory and resources
2. **Defense**: Enhance military strength and defensive capability
3. **Trade**: Exchange resources with allies
4. **Research**: Invest resources in technology development
5. **Diplomacy**: Establish and maintain relationships
6. **Culture**: Promote cultural influence
7. **Religion**: Spread religious beliefs

## Random Events

The system includes 11 random event types:

- Natural Disasters (minor/major)
- Technology Breakthrough
- Social Reform
- Resource Discovery
- Pandemic Outbreak
- Migration Wave
- Trade Agreement
- Cultural Exchange
- Resource Depletion
- Diplomatic Crisis

## Data Structures

### Simulation History
```python
history = {
    "strategy": np.array,    # Strategy probabilities per cycle
    "resources": np.array,   # Resource levels per cycle
    "strength": np.array,    # Military strength per cycle
    "technology": np.array,  # Technology levels per cycle
    "population": np.array,  # Population per cycle
    "territory": np.array,   # Territory per cycle
    "events": List[EventRecord]  # Event history
}
```

### Technology History
```python
technology_history = {
    agent_id: [
        {
            "cycle": int,
            "technologies": {tech_name: level},
            "current_research": str | None,
            "research_progress": float,
            "research_cost": float,
            "tech_bonuses": dict
        },
        ...
    ]
}
```