# Agent Documentation for Civilization Evolution System

This documentation is specifically designed for AI assistants and automation tools working with the Civilization Evolution System codebase.

## Project Structure

```
.
├── multi_agent_simulation.py     # Core multi-agent simulation engine
├── civilization_visualizer.py    # Data visualization and plotting tools
├── tech_tree.py                  # Technology tree implementation
├── simulation_config.py          # Configuration parameters and presets
├── simulation_cli.py             # Command-line interface
├── advanced_evolution.py         # Advanced evolution mechanisms
├── example_config.py             # Example configuration files
├── demo_*.py                     # Various demonstration scripts
├── test_*.py                     # Test scripts for system validation
├── requirements.txt              # Python dependencies
├── run_simulation.bat            # Windows batch script for execution
├── *.md                          # Documentation files
└── test_results/                 # Directory for test outputs
```

## Core Classes and Components

### CivilizationAgent (multi_agent_simulation.py)

The primary agent class representing a civilization in the simulation.

**Key Attributes:**
- `agent_id`: Unique identifier for the civilization
- `strength`: Military power affecting combat and defense capabilities
- `resources`: Resource reserves for development and expansion
- `territory`: Set of controlled hexagonal grid cells
- `allies`/`enemies`: Sets of diplomatic relationship identifiers
- `technology`: Dictionary of technology levels in different fields
- `population`: Population size affecting production and research
- `infrastructure`: Infrastructure level affecting various capabilities

**Key Methods:**
- `decide_strategy()`: Determine strategy based on neighbors and resources
- `execute_strategy()`: Execute the chosen strategy
- `update_tech_bonuses()`: Update technology-based attribute bonuses

### TechTree (tech_tree.py)

Manages the technology research system with dependencies and effects.

**Key Methods:**
- `get_available_techs()`: Return technologies available for research
- `calculate_tech_bonuses()`: Calculate attribute bonuses from technologies
- `research_technology()`: Process technology research completion

### CivilizationVisualizer (civilization_visualizer.py)

Handles data visualization and result presentation.

**Key Methods:**
- `plot_strategy_heatmap()`: Display strategy distribution among civilizations
- `plot_evolution_curve()`: Show attribute evolution over time
- `plot_technology_progress()`: Compare technology development paths
- `create_summary_report()`: Generate textual summary of simulation results

### SimulationConfig (simulation_config.py)

Centralized configuration management with preset scenarios.

**Key Features:**
- Configurable simulation parameters
- Preset configurations for different scenarios
- Method to apply preset configurations

## Key Workflows

### 1. Initialization Process

1. Load configuration from `simulation_config.py`
2. Initialize `CivilizationAgent` instances
3. Set up resource distribution
4. Establish initial territory assignments

### 2. Simulation Cycle

1. Each agent decides strategy via `decide_strategy()`
2. Agents execute strategies via `execute_strategy()`
3. Update agent states (resources, territory, relationships)
4. Handle technology research and completion
5. Record historical data for analysis

### 3. Visualization and Output

1. Generate plots using `CivilizationVisualizer`
2. Save data in various formats (CSV, JSON, NPZ)
3. Create summary reports

## Code Patterns to Recognize

### Multi-Agent Interaction
Agents interact through:
- Territory competition (overlapping territorial claims)
- Diplomatic relationships (allies/enemies)
- Resource competition
- Technology spillover effects

### Configuration Management
- Centralized in `SimulationConfig` class
- Preset scenarios via `apply_preset()` method
- Overridable through custom configuration files

### Data Flow
1. Configuration → Simulation Setup → Execution Loop
2. Execution Loop → Data Collection → Historical Data
3. Historical Data → Visualization → Reports

## File Dependencies

- `multi_agent_simulation.py` imports:
  - `tech_tree.py`
  - `simulation_config.py`
  - `advanced_evolution.py`

- `civilization_visualizer.py` imports:
  - Standard libraries (matplotlib, numpy, pandas)
  - `simulation_config.py`

- `simulation_cli.py` imports:
  - All core modules
  - Standard libraries for CLI handling

## Common Tasks for Agents

### Understanding Agent Decision Making
- Strategy decision in `CivilizationAgent.decide_strategy()`
- Factors include neighbor relationships, resource pressure, technology levels
- Returns probability distribution over 4 strategies: expansion, defense, trade, research

### Tracking Technology Progress
- Technologies defined in `TechTree.techs`
- Research progress tracked per agent
- Bonuses applied through `TechTree.calculate_tech_bonuses()`

### Analyzing Simulation Results
- Historical data collected in simulation loop
- Visualization methods in `CivilizationVisualizer`
- Output formats: plots, CSV, JSON, binary archives

## Important Constants and Parameters

Refer to `simulation_config.py` for:
- `NUM_CIVILIZATIONS`: Number of civilization agents
- `GRID_SIZE`: Size of the hexagonal grid world
- `SIMULATION_CYCLES`: Number of simulation steps
- Resource, population, technology, and relationship parameters
- Visualization and output settings

This documentation should provide AI assistants with the necessary context to understand, navigate, and work with the Civilization Evolution System codebase effectively.