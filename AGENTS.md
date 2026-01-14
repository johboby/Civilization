# AGENTS.md - AI Assistant & Automation Guide

This document provides technical information for AI assistants and automation tools working with the Civilization Evolution Simulation System.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Flow & Execution](#data-flow--execution)
4. [Extension Points](#extension-points)
5. [Common Tasks](#common-tasks)
6. [Testing & Validation](#testing--validation)

---

## Architecture Overview

### System Design

The simulation system is built on a multi-agent architecture with the following key principles:

- **Modular Design**: Each component is self-contained with clear interfaces
- **Type Safety**: Extensive use of type hints and dataclasses
- **Configurable**: Behavior controlled through `SimulationConfig`
- **Extensible**: Pluggable decision engines and event systems

### Package Structure

```
civsim/
├── config.py              # Configuration management (SimulationConfig)
├── logger.py              # Logging infrastructure
├── constants.py           # Global constants and enums
├── technology.py          # Tech tree (TechnologyManager, TechInfo, TechBonus)
├── strategy.py            # Decision engines (StrategyDecisionEngine, AgentState)
├── events.py              # Random events (RandomEventManager, EventInfo)
├── evolution.py           # Advanced AI (AdvancedEvolution, ComplexResourceManager)
├── relationship_manager.py  # Inter-civilization relationships
├── strategy_executor.py    # Strategy implementation
├── simulation.py          # Core simulation loop (CivilizationAgent, MultiAgentSimulation)
├── multi_agent.py         # Multi-agent orchestration
├── performance.py         # Performance optimizations
└── animation.py           # Visualization components
```

---

## Core Components

### 1. Configuration System (`civsim.config`)

**Primary Classes:**
- `SimulationConfig`: Main configuration container
- `get_preset(name: str) -> SimulationConfig`: Retrieve preset configurations

**Key Configuration Parameters:**
```python
config = SimulationConfig(
    num_civilizations: int = 5,
    simulation_cycles: int = 200,
    grid_size: int = 150,
    random_seed: int | None = None,
    enable_random_events: bool = True,
    event_probability_modifier: float = 1.0,
    use_advanced_evolution: bool = False,
    use_complex_resources: bool = False,
    use_cultural_influence: bool = False
)
```

**Usage Pattern:**
```python
from civsim.config import SimulationConfig, get_preset

# Use preset
config = get_preset("medium")

# Or customize
config = SimulationConfig(num_civilizations=8, simulation_cycles=500)
```

### 2. Civilization Agent (`civsim.simulation.CivilizationAgent`)

**Key Attributes:**
```python
class CivilizationAgent:
    agent_id: int
    strength: float           # Military strength
    resources: float         # Resource reserves
    population: float        # Population size
    infrastructure: float     # Infrastructure level
    stability: float         # Social stability
    territory: set           # Controlled territory (grid positions)
    allies: set              # Allied civilization IDs
    enemies: set             # Enemy civilization IDs
    technology: dict         # Researched technologies {name: level}
    tech_tree: TechnologyManager
    tech_bonuses: dict      # Current technology bonuses
    relationship_weights: dict  # Relationship weights {agent_id: value}
```

**Key Methods:**
- `decide_strategy(neighbors, global_resources, advanced_evolution) -> np.ndarray`
- `execute_strategy(strategy, neighbors, global_resources) -> None`
- `update_tech_bonuses() -> None`
- `_research_technology() -> None`

**Strategy Vector Structure:**
```python
# 7-element array: [expansion, defense, trade, research, diplomacy, culture, religion]
strategy_probs = agent.decide_strategy(neighbors, global_resources)
```

### 3. Technology Management (`civsim.technology`)

**Technology Tree Structure:**
```python
technology_levels = {
    1: ["agriculture", "military", "trade", "science"],
    2: ["irrigation", "fortification", "currency", "engineering"],
    3: ["industrial_agriculture", "advanced_tactics", "global_trade", "advanced_science"],
    4: ["genetic_engineering", "nuclear_technology", "space_colonization", "artificial_intelligence"]
}
```

**TechBonus Attributes (19 total):**
```python
@dataclass
class TechBonus:
    resources: float
    strength: float
    defense: float
    research_speed: float
    diplomacy: float
    territory_growth: float
    territory_value: float
    trade_efficiency: float
    infrastructure: float
    population_growth: float
    tactical_advantage: float
    tech_discovery: float
    stability: float
    innovation: float
    health: float
    energy_efficiency: float
    resource_acquisition: float
    global_influence: float
    decision_quality: float
```

**Usage Pattern:**
```python
from civsim.technology import TechnologyManager

tech_manager = TechnologyManager()

# Get available technologies
available = tech_manager.get_available_techs(current_techs)

# Calculate bonuses
bonuses = tech_manager.calculate_bonuses(current_techs)

# Check research eligibility
can_research, reason = tech_manager.can_research("irrigation", current_techs)
```

### 4. Strategy Decision Engine (`civsim.strategy`)

**Available Engines:**
- `DefaultDecisionEngine`: Rule-based heuristic decision making
- `RandomDecisionEngine`: Random strategy selection
- Custom engines can be created by inheriting from `StrategyDecisionEngine`

**AgentState Structure:**
```python
@dataclass
class AgentState:
    agent_id: int
    strength: float
    resources: float
    population: float
    infrastructure: float
    stability: float
    global_influence: float
    research_speed: float
```

**Creating Custom Decision Engine:**
```python
from civsim.strategy import StrategyDecisionEngine, StrategyResult, AgentState

class MyDecisionEngine(StrategyDecisionEngine):
    def decide(self, state: AgentState, neighbors: Dict, global_resources: Dict) -> List[StrategyResult]:
        # Implement custom decision logic
        results = []
        # ... decision logic ...
        return results
```

### 5. Random Events (`civsim.events`)

**Event Structure:**
```python
@dataclass
class EventInfo:
    name: str
    description: str
    probability: float
    effect: Callable  # Function that applies event effects
    severity: str     # "minor", "major", "positive"
    categories: List[str]

# Built-in events
events = {
    'minor_disaster': EventInfo(..., probability=0.15, effect=minor_disaster_effect),
    'major_disaster': EventInfo(..., probability=0.05, effect=major_disaster_effect),
    'tech_breakthrough': EventInfo(..., probability=0.08, effect=tech_breakthrough_effect),
    # ... 8 more events
}
```

**Adding Custom Event:**
```python
def my_event_effect(agents: List[Any]) -> List[int]:
    # Apply effect to affected agents
    return [agent.agent_id for agent in affected_agents]

# Add to event manager
event_manager.events['my_event'] = EventInfo(
    name="My Custom Event",
    description="Description of the event",
    probability=0.1,
    effect=my_event_effect,
    severity="minor",
    categories=["custom"]
)
```

### 6. Advanced Evolution (`civsim.evolution`)

**AdvancedEvolution Class:**
Implements sophisticated AI decision-making based on:
- Resource pressure assessment
- Security risk evaluation
- Development potential analysis
- Game theory-based strategy selection
- Historical experience learning
- Metacognitive adjustment

**Key Methods:**
```python
class AdvancedEvolution:
    def calculate_strategy_tendency(agent, neighbors, global_resources) -> Dict[str, float]:
        # Returns strategy probabilities {strategy_name: probability}
        pass

    def _assess_resource_pressure(agent, global_resources) -> float:
        # Calculate resource pressure score (0-1)
        pass

    def _assess_security_risk(agent, neighbors) -> float:
        # Calculate security risk score (0-1)
        pass
```

**ComplexResourceManager:**
Generates realistic resource distribution based on:
- Elevation maps (multi-scale noise)
- Moisture maps (climate model)
- Terrain-based resource allocation

**CulturalInfluence:**
Models cultural traits and their influence:
```python
cultural_traits = {
    'collectivism', 'individualism',
    'militarism', 'pacifism',
    'tradition', 'innovation',
    'expansionism', 'isolationism'
}
```

### 7. Visualization (`civilization_visualizer.py`)

**CivilizationVisualizer Class:**

**Available Visualization Methods:**
```python
class CivilizationVisualizer:
    def plot_strategy_heatmap(self, strategy_matrix, title, filename, cmap_type, colorbar_label)
    def plot_evolution_curve(self, history_data, resource_history, attribute_history, filename)
    def plot_technology_progress(self, technology_history, num_cycles, filename_prefix)
    def plot_tech_tree_comparison(self, technology_history, filename_prefix)
    def plot_top_tech_comparison(self, technology_history, filename_prefix)
    def plot_attribute_comparison(self, attribute_data, attribute_names, title, filename)
    def plot_radar_chart(self, attribute_data, attribute_names, agent_names, title, filename)
    def plot_relationships_network(self, relationships_data, filename_prefix)
    def create_evolution_animation(self, grid_history, strategy_history, filename_prefix)
    def save_to_csv(self, data, filename, headers)
    def create_summary_report(self, history_data, attribute_history, technology_history, filename_prefix)
```

**Color Mapping:**
```python
colors = {
    'expansion': '#0868ac',
    'defense': '#d95f0e',
    'trade': '#3f007d',
    'research': '#4E9CAF',
    'diplomacy': '#c51b8a',
    'population': '#807dba',
    'health': '#74c476',
    'resources': '#ffb72b'
}
```

---

## Data Flow & Execution

### Simulation Execution Flow

```python
# 1. Initialize
config = SimulationConfig(...)
sim = MultiAgentSimulation(config)

# 2. Run simulation
history = sim.run(config.simulation_cycles)

# Inside sim.run(cycles):
for cycle in range(cycles):
    # a. Each agent decides strategy
    for agent in agents:
        neighbors = sim.get_neighbors(agent)
        strategy = agent.decide_strategy(neighbors, global_resources, advanced_evolution)

    # b. Each agent executes strategy
    for agent in agents:
        agent.execute_strategy(strategy, neighbors, global_resources)

    # c. Update relationships
    relationship_manager.update_relationships(agents)

    # d. Trigger random events
    event_manager.trigger_event(agents, cycle)

    # e. Record history
    record_history()

# 3. Visualize results
visualizer = CivilizationVisualizer()
visualizer.plot_evolution_curve(history)
visualizer.plot_technology_progress(technology_history)
```

### Key Data Structures

**Simulation History:**
```python
history = {
    "strategy": np.array,       # Strategy probabilities per cycle
    "resources": np.array,       # Resource levels per cycle
    "strength": np.array,        # Military strength per cycle
    "technology": np.array,      # Technology levels per cycle
    "population": np.array,      # Population per cycle
    "territory": np.array,       # Territory per cycle
    "events": List[EventRecord]   # Event history
}
```

**Technology History:**
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

---

## Extension Points

### 1. Custom Decision Engine

Create a new decision engine by inheriting from `StrategyDecisionEngine`:

```python
from civsim.strategy import StrategyDecisionEngine, StrategyResult
from typing import Dict, List

class MyDecisionEngine(StrategyDecisionEngine):
    """Custom decision engine example."""

    def decide(
        self,
        state: AgentState,
        neighbors: Dict[int, float],
        global_resources: Dict[int, float]
    ) -> List[StrategyResult]:
        """
        Make strategic decisions based on custom logic.

        Args:
            state: Current agent state
            neighbors: Neighbor relationships {agent_id: weight}
            global_resources: Resource distribution {position: value}

        Returns:
            List of strategy results with probabilities
        """
        results = []

        # Your custom decision logic here
        # Example: Prioritize research when resources are abundant
        if state.resources > 500:
            results.append(StrategyResult(
                name="research",
                probability=0.6,
                reasoning="High resources available for research"
            ))
        else:
            results.append(StrategyResult(
                name="expansion",
                probability=0.7,
                reasoning="Need to acquire more resources"
            ))

        return results

# Use custom engine
from civsim.strategy import create_decision_engine
engine = create_decision_engine("my_custom_engine")  # Register your engine
```

### 2. Custom Technology

Add new technologies to the technology tree:

```python
from civsim.technology import TechnologyManager, TechInfo, TechBonus

# Extend TechnologyManager
class ExtendedTechnologyManager(TechnologyManager):
    def _initialize_tech_tree(self) -> Dict[str, TechInfo]:
        techs = super()._initialize_tech_tree()

        # Add new technology
        techs["quantum_computing"] = TechInfo(
            name="quantum_computing",
            level=4,
            cost=4000,
            description="Quantum computing technology",
            prerequisites=["artificial_intelligence"],
            category="Top-level Technology"
        )

        return techs

    def _initialize_tech_effects(self) -> Dict[str, TechBonus]:
        effects = super()._initialize_tech_effects()

        # Define technology bonuses
        effects["quantum_computing"] = TechBonus(
            research_speed=2.0,
            innovation=1.8,
            decision_quality=1.9
        )

        return effects
```

### 3. Custom Event

Add new random events:

```python
from civsim.events import RandomEventManager, EventInfo
from typing import List, Any

def alien_encounter_effect(agents: List[Any]) -> List[int]:
    """Effect of alien encounter."""
    affected = []
    for agent in agents:
        if agent.technology.get("space_colonization", 0) >= 1:
            agent.research_speed *= 1.5
            agent.global_influence *= 1.2
            affected.append(agent.agent_id)
    return affected

# Add to event manager
class ExtendedEventManager(RandomEventManager):
    def _initialize_events(self) -> Dict[str, EventInfo]:
        events = super()._initialize_events()

        events['alien_encounter'] = EventInfo(
            name="Alien Encounter",
            description="First contact with extraterrestrial civilization",
            probability=0.02,
            effect=alien_encounter_effect,
            severity="positive",
            categories=["special", "space"]
        )

        return events
```

### 4. Custom Visualization

Create custom visualization methods:

```python
from civilization_visualizer import CivilizationVisualizer
import matplotlib.pyplot as plt

class CustomVisualizer(CivilizationVisualizer):
    def plot_custom_metric(self, data, title, filename):
        """Plot custom metric visualization."""
        plt.figure(figsize=(12, 6))
        plt.plot(data, linewidth=2)
        plt.title(title)
        plt.xlabel("Cycle")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300)
        plt.close()
```

---

## Common Tasks

### Task 1: Run a Basic Simulation

```python
from civsim import MultiAgentSimulation, SimulationConfig, get_preset
from civilization_visualizer import CivilizationVisualizer

# Configure simulation
config = get_preset("medium")
config.num_civilizations = 6
config.simulation_cycles = 200

# Run simulation
sim = MultiAgentSimulation(config)
history = sim.run(config.simulation_cycles)

# Visualize results
visualizer = CivilizationVisualizer(output_dir="results")
visualizer.plot_evolution_curve(history)
visualizer.plot_technology_progress(sim.technology_history)
```

### Task 2: Analyze Simulation Results

```python
import numpy as np
import pandas as pd

# Load simulation results
data = np.load("results/simulation.npz", allow_pickle=True)

# Extract strategy history
strategy_history = data["strategy"]

# Calculate average strategy usage
avg_strategy = np.mean(strategy_history, axis=0)
print(f"Average strategy usage:")
print(f"  Expansion: {avg_strategy[0]:.3f}")
print(f"  Defense: {avg_strategy[1]:.3f}")
print(f"  Trade: {avg_strategy[2]:.3f}")
print(f"  Research: {avg_strategy[3]:.3f}")

# Find dominant strategy
dominant_strategy = np.argmax(avg_strategy)
strategy_names = ["expansion", "defense", "trade", "research"]
print(f"Dominant strategy: {strategy_names[dominant_strategy]}")
```

### Task 3: Compare Multiple Simulations

```python
from civsim import MultiAgentSimulation, SimulationConfig
import matplotlib.pyplot as plt

results = []

for seed in [42, 123, 456, 789]:
    config = SimulationConfig(random_seed=seed, num_civilizations=5, simulation_cycles=100)
    sim = MultiAgentSimulation(config)
    history = sim.run(config.simulation_cycles)
    results.append(history)

# Plot comparison
plt.figure(figsize=(12, 6))
for i, history in enumerate(results):
    plt.plot(history[:, 0], label=f"Run {i+1}", alpha=0.7)
plt.xlabel("Cycle")
plt.ylabel("Expansion Strategy")
plt.title("Strategy Comparison Across Different Seeds")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("strategy_comparison.png", dpi=300)
plt.close()
```

### Task 4: Add Logging to Custom Module

```python
from civsim.logger import get_logger

logger = get_logger(__name__)

def my_function():
    logger.info("Starting my function")
    try:
        # Do something
        result = perform_calculation()
        logger.info(f"Calculation completed successfully: {result}")
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise

def perform_calculation():
    logger.debug("Performing calculation...")
    return 42
```

### Task 5: Extend Configuration

```python
from civsim.config import SimulationConfig

class MyConfig(SimulationConfig):
    """Custom configuration with additional parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_custom_param = kwargs.get("my_custom_param", 10)

    def validate(self) -> List[str]:
        errors = super().validate()
        if self.my_custom_param < 0:
            errors.append("my_custom_param must be non-negative")
        return errors

# Use custom config
config = MyConfig(
    num_civilizations=5,
    simulation_cycles=200,
    my_custom_param=20
)
```

---

## Testing & Validation

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_simulation.py -v

# Run with coverage
pytest tests/ --cov=civsim --cov-report=html

# Run specific test
pytest tests/test_simulation.py::test_agent_initialization -v
```

### Writing Tests

```python
import pytest
import numpy as np
from civsim.simulation import CivilizationAgent
from civsim.technology import TechnologyManager

def test_agent_initialization():
    """Test that agent is initialized correctly."""
    agent = CivilizationAgent(agent_id=0, initial_strength=100, resources=200)

    assert agent.agent_id == 0
    assert agent.strength == 100.0
    assert agent.resources == 200.0
    assert len(agent.technology) == 4  # Basic technologies
    assert agent.tech_tree is not None

def test_strategy_decision():
    """Test strategy decision making."""
    agent = CivilizationAgent(agent_id=0)
    neighbors = {1: (50.0, 0.5)}  # agent_id: (strength, relationship)
    global_resources = {(0, 0): 10.0, (1, 1): 15.0}

    strategy = agent.decide_strategy(neighbors, global_resources)

    assert len(strategy) == 7  # 7 strategy types
    assert np.allclose(np.sum(strategy), 1.0)  # Normalized
    assert np.all(strategy >= 0)  # Non-negative

def test_technology_research():
    """Test technology research mechanics."""
    agent = CivilizationAgent(agent_id=0)
    agent.resources = 500

    available = agent.tech_tree.get_available_techs(agent.technology)
    assert len(available) > 0

    # Research a technology
    tech_name = available[0]["name"]
    cost = agent.tech_tree.get_research_cost(tech_name, 1)

    assert cost <= agent.resources
```

### Validation Checklist

When modifying code, verify:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Type checking passes (`mypy civsim`)
- [ ] Code formatting consistent (`black civsim`)
- [ ] Documentation updated (docstrings, comments)
- [ ] Performance not degraded (benchmark if needed)
- [ ] Backward compatibility maintained (if applicable)
- [ ] Edge cases handled
- [ ] Error handling in place
- [ ] Logging added for debugging

### Performance Benchmarking

```python
import time
import numpy as np
from civsim import MultiAgentSimulation, SimulationConfig

def benchmark_simulation(num_civs, cycles, trials=3):
    """Benchmark simulation performance."""
    times = []

    for i in range(trials):
        config = SimulationConfig(
            num_civilizations=num_civs,
            simulation_cycles=cycles
        )
        sim = MultiAgentSimulation(config)

        start = time.time()
        history = sim.run(cycles)
        elapsed = time.time() - start

        times.append(elapsed)
        print(f"Trial {i+1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nAverage: {avg_time:.3f}s ± {std_time:.3f}s")
    return avg_time

# Run benchmarks
print("Benchmarking small simulation...")
benchmark_simulation(num_civs=5, cycles=100)

print("\nBenchmarking large simulation...")
benchmark_simulation(num_civs=15, cycles=500)
```

---

## Important Notes for AI Agents

### Code Conventions

1. **Type Hints**: Use type hints for all function signatures
2. **Docstrings**: Google-style docstrings for public APIs
3. **Naming**: snake_case for variables/functions, PascalCase for classes
4. **Imports**: Group imports (stdlib, third-party, local)
5. **Error Handling**: Use exceptions for error conditions, not return codes

### Design Patterns

- **Factory Pattern**: `create_decision_engine()`, `get_preset()`
- **Strategy Pattern**: Pluggable decision engines
- **Observer Pattern**: Event system
- **Dataclass Pattern**: Configuration and data structures

### Key Dependencies

- `numpy >= 1.24.0` - Numerical computations
- `matplotlib >= 3.7.0` - Visualization
- `pandas >= 2.0.0` - Data handling
- `networkx >= 3.0` - Graph structures

### Limitations

- Maximum civilizations: 20 (performance degrades beyond)
- Grid size limit: 500x500 (memory constraints)
- Strategy count: Fixed at 7 (requires code modification to change)
- Technology levels: Fixed at 4 (extensible via inheritance)

### Common Pitfalls

1. **Forgetting to normalize strategy vectors**: Always ensure probabilities sum to 1.0
2. **Not updating tech bonuses**: Call `update_tech_bonuses()` after tech changes
3. **Modifying simulation during iteration**: Create copies if modifying agent lists
4. **Ignoring edge cases**: Handle empty lists, zero divisions, None values
5. **Missing type conversions**: Ensure numpy arrays are properly typed

### Debugging Tips

```python
# Enable debug logging
from civsim.logger import init_logging
init_logging(level="DEBUG")

# Check agent state
print(f"Agent {agent.agent_id}:")
print(f"  Resources: {agent.resources}")
print(f"  Technology: {agent.technology}")
print(f"  Territory: {len(agent.territory)} tiles")

# Monitor strategy changes
strategy_history = []
for cycle in range(100):
    strategy = agent.decide_strategy(neighbors, global_resources)
    strategy_history.append(strategy)
    # Log significant changes
    if cycle > 0 and np.linalg.norm(strategy - strategy_history[-2]) > 0.3:
        print(f"Cycle {cycle}: Major strategy change")
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-30
**Maintained For**: AI assistants and automation tools
