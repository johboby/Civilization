# Civilization Evolution Simulation System

<div align="center">
  <img src="demo_resource_evolution.png" alt="Civilization Resource Evolution Example" width="600">
</div>

## Project Overview

The Civilization Evolution Simulation System is a multi-agent based simulation platform designed to simulate the competition and development of multiple civilizations in a resource-limited environment. The system combines artificial intelligence, complex systems theory, and game theory to drive civilizations to make strategic choices through algorithms, achieving multi-dimensional civilization evolution including technology research and development, resource acquisition, territorial expansion, and diplomatic relations.

## System Features

- **Multi-Agent Interaction**: Simulate multiple independent decision-making civilization agents
- **Complete Technology Tree System**: Including basic, intermediate, and advanced technologies
- **Complex Strategy Decision-Making**: Dynamically adjust strategies based on resources, military, technology, and diplomatic factors
- **Territory and Resource Management**: Simulate uneven resource distribution and territorial expansion mechanisms
- **Civilization Relationship Network**: Establishment and evolution of ally and enemy relationships
- **Rich Visualization Output**: Heatmaps, evolution curves, radar charts, and various other charts
- **Complete Data Export**: Support for CSV, JSON, and binary formats
- **Command-Line Interface**: Provides both interactive and batch processing modes

## Directory Structure

```
├── multi_agent_simulation.py  # Core simulation system
├── civilization_visualizer.py # Visualization tools
├── tech_tree.py               # Technology tree system
├── simulation_config.py       # Configuration file
├── demo_simulation.py         # Demo script
├── simulation_cli.py          # Command-line interface
├── test_system.py             # System test script
├── requirements.txt           # Dependency list
├── run_simulation.bat         # Quick start script (Windows)
├── example_config.py          # Configuration example
└── test_results/              # Test results directory
```

## Installation

### Requirements
- Python 3.8+ 
- PyTorch 2.0+ 
- NumPy 1.24+ 
- Matplotlib 3.7+ 
- Pandas 2.0+ 
- NetworkX 3.0+

### Installation Steps

1. Clone or download the project code
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) Create a virtual environment:
```bash
python -m venv venv
# Windows
source venv/Scripts/activate
# Linux/Mac
source venv/bin/activate
```

## Usage

### Quick Start

The easiest way to start is using the batch file:
- Windows users: Double-click `run_simulation.bat`

Or run the demo script directly through the command line:
```bash
python demo_simulation.py
```

### Command-Line Interface

The system provides a feature-rich command-line interface that supports various parameter configurations:

```bash
# Basic usage
python simulation_cli.py --num-civs 5 --cycles 100 --grid-size 200

# Interactive mode
python simulation_cli.py --interactive

# Using custom configuration file
python simulation_cli.py --config example_config.py

# Fast mode (reduced logs and visualization)
python simulation_cli.py --fast-mode
```

### Main Parameters

| Parameter | Description | Default |
|------|--------|--------|
| `--cycles` | Simulation cycle count | 50 |
| `--num-civs` | Number of civilizations | 5 |
| `--grid-size` | Grid size | 200 |
| `--output-dir` | Output results directory | results |
| `--seed` | Random seed | Random |
| `--interactive` | Enable interactive mode | Disabled |
| `--batch` | Enable batch mode | Disabled |
| `--resume` | Resume from saved results | Disabled |
| `--config` | Custom configuration file | None |

## Core System Functions

### 1. Civilization Agent (CivilizationAgent)

Each civilization has the following core attributes:
- Military strength (strength): Affects combat and defense capabilities
- Resource reserves (resources): Foundation for development and expansion
- Territory control (territory): Controlled hexagonal grid area
- Technology level (technology): Progress in various fields of technological research and development
- Diplomatic relations (allies/enemies): Relationships with other civilizations
- Population (population): Affects production and research capabilities
- Infrastructure (infrastructure): Affects the efficiency of various capabilities

### 2. Strategy Decision System

Civilizations can adopt four main strategies:
- **Expansion Strategy**: Acquire more territory and resources
- **Defense Strategy**: Enhance military strength and defensive capabilities
- **Trade Strategy**: Resource exchange and trade with allies
- **Research Strategy**: Invest resources in technological research and development

Strategy selection is dynamically adjusted based on multiple factors, including:
- Neighbor threat level
- Resource pressure
- Technology development level
- Population and infrastructure conditions

### 3. Technology Tree System (TechTree)

The technology tree is divided into four levels:
- **Basic Technologies**: Agriculture, Military, Trade, Science
- **Intermediate Technologies**: Irrigation, Defensive Works, Monetary System, Engineering
- **Advanced Technologies**: Industrial Agriculture, Advanced Tactics, Global Trade, Advanced Science
- **Top-Level Technologies**: Genetic Engineering, Nuclear Technology, Space Colonization, Artificial Intelligence

Each technology provides specific attribute bonuses, and advanced technologies require prerequisite conditions to be researched.

### 4. Visualization Features

The system provides multiple visualization charts:
- Strategy heatmap: Display civilization strategy distribution
- Evolution trend chart: Track strategy and resource changes
- Technology progress chart: Compare technology development among civilizations
- Technology tree comparison chart: Display different civilization technology tree paths
- Attribute radar chart: Multi-dimensional comparison of civilization attributes
- Relationship network chart: Display diplomatic relationships between civilizations

## Output Data

The system supports saving output data in multiple formats:
- **CSV Format**: Strategy history, attribute history, etc.
- **JSON Format**: Technology development history, civilization status, etc.
- **Binary Format**: Complete simulation results
- **Summary Report**: Simulation result analysis report

## Development Guide

### Extending the Technology Tree

In the `tech_tree.py` file, you can add new technologies by modifying the `self.techs` dictionary:
```python
self.techs = {
    # Add new technology
    "new_technology": {
        "level": 2,  # Technology level
        "cost": 300,  # Base research cost
        "description": "New technology description",
        "prerequisites": ["existing_tech1", "existing_tech2"]  # Prerequisite technologies
    },
    # ...other technologies
}

# Add technology effects
self.tech_effects = {
    "new_technology": {"attribute1": 0.2, "attribute2": 0.15},
    # ...other technology effects
}
```

### Custom Configuration

You can create a configuration file to override default parameters:
```python
# example_config.py
from simulation_config import config

# Modify configuration parameters
config.NUM_CIVILIZATIONS = 8
config.SIMULATION_CYCLES = 200
config.GRID_SIZE = 300
config.INITIAL_RESOURCES = 300
```

Then load it using the `--config` parameter:
```bash
python simulation_cli.py --config example_config.py
```

### Adding New Visualization Charts

In `civilization_visualizer.py`, you can add new visualization methods:
```python
def plot_new_visualization(self, data, title="New Visualization Chart", filename="new_visualization.png"):
    """Draw custom visualization chart"""
    plt.figure(figsize=(10, 8))
    # Chart drawing logic
    plt.title(title, fontsize=14)
    plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
```

## Testing System

The system includes a complete test script to verify that core functions are working properly:
```bash
python test_system.py
```

Test results will be saved in the `test_results` directory.

## Future Improvements

1. **GUI Development**: Develop a graphical user interface for a more intuitive interactive experience
2. **Algorithm Optimization**: Improve agent decision-making algorithms and add more strategy types
3. **Expanded Technology Tree**: Add more technology types and branches
4. **Enhanced Diplomatic System**: Add more complex diplomatic mechanisms and interaction methods between civilizations
5. **Random Event System**: Introduce random events such as natural disasters and new technology discoveries
6. **Parallel Computing**: Optimize performance for large-scale simulations
7. **Web Demo**: Develop a web version for online demonstration and sharing

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contact

If you have any questions or suggestions, please contact the project maintainer.