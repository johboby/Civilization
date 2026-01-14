# Civilization Evolution Simulation System

<div align="center">
  <img src="docs/assets/logo.png" alt="Civilization Evolution Simulation" width="200">
</div>

A multi-agent civilization evolution simulation platform combining artificial intelligence, complex systems theory, and game theory.

## Features

### Core Features
- 🤖 **Multi-Agent Interaction**: Simulate multiple independently evolving civilizations, each with unique attributes and behavior patterns
- 🌳 **Complete Technology Tree**: Four levels (Basic, Intermediate, Advanced, Top) with 16 distinct technologies
- 🎯 **Complex Strategic Decisions**: 7 strategy types (Expansion, Defense, Trade, Research, Diplomacy, Culture, Religion) with dynamic adjustment based on resources, military, technology, and diplomacy
- 🗺️ **Territory & Resource Management**: Simulate resource distribution and territorial expansion with dynamic resource generation based on terrain and climate

### Advanced Features
- 🤝 **Civilization Relationship Network**: Establishment and evolution of alliances and rivalries based on behavioral history and cultural similarity
- 📊 **Rich Visualization**: Heatmaps, evolution curves, radar charts, relationship network graphs, and more
- 💾 **Complete Data Export**: Support for CSV, JSON, and NPZ formats
- 🖥️ **Command Line Interface**: Interactive and batch modes with parameter customization
- 🎲 **Random Event System**: 11 event types (natural disasters, technological breakthroughs, pandemics, trade agreements, etc.)

### Intelligent Evolution
- 🧬 **Advanced Evolution Engine**: Intelligent decision-making based on game theory and metacognition
- 🌍 **Cultural Influence System**: Simulate cultural transmission and influence between civilizations
- 💎 **Complex Resource Management**: Dynamic resource distribution based on terrain and climate
- ⚡ **Performance Optimization**: Vectorized operations supporting large-scale simulations
- 🎨 **Real-time Animation**: Dynamic visualization of the simulation process

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Demos

#### Basic Demo
```bash
python run_demo.py basic
```

#### Advanced Evolution Demo
```bash
python run_demo.py advanced
```

#### New Features Demo
```bash
python run_demo.py new-features --save --output-dir my_results
```

### Using Command Line Interface

#### Basic Usage
```bash
python simulation_cli.py --cycles 100 --num-civs 5
```

#### Interactive Mode
```bash
python simulation_cli.py --interactive
```

#### Enable Random Events
```bash
python simulation_cli.py --enable-events --event-modifier 1.5
```

### Using Interactive Demo

```bash
# Basic interactive demo
python interactive_demo.py

# With custom configuration
python interactive_demo.py --preset medium --num-civs 5 --cycles 200

# With specific random seed
python interactive_demo.py --preset large --seed 12345
```

## Project Structure

```
Civilization/
├── civsim/                    # Core package
│   ├── __init__.py           # Package initialization and exports
│   ├── config.py             # Configuration management
│   ├── logger.py            # Logging system
│   ├── strategy.py          # Strategy engine
│   ├── technology.py        # Technology management
│   ├── events.py           # Random events (11 types)
│   ├── evolution.py         # Advanced evolution
│   ├── performance.py      # Performance optimization
│   ├── animation.py        # Real-time animation
│   ├── simulation.py       # Simulation core
│   ├── multi_agent.py      # Multi-agent system
│   ├── relationship_manager.py  # Relationship management
│   ├── strategy_executor.py     # Strategy execution
│   └── constants.py       # Constant definitions
├── results/                  # Simulation results
├── run_demo.py             # Demo entry point
├── simulation_cli.py       # Command line interface
├── interactive_demo.py     # Interactive demo
├── civilization_visualizer.py  # Visualization tools
├── example_config.py       # Configuration examples
├── requirements.txt        # Dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This document
```

## Core Modules

### Configuration Management (`civsim.config`)
- Type-safe configuration classes
- Preset configurations (demo, small, medium, large, resource_scarcity, tech_focus, advanced_evolution)
- Configuration validation

### Strategy Engine (`civsim.strategy`)
- Pluggable decision engines
- Multiple strategy types
- Strategy weight calculation

### Technology Management (`civsim.technology`)
- Complete technology tree (16 technologies, 4 levels)
- Technology bonus system (19 attribute bonuses)
- Technology dependency relationships

### Random Events (`civsim.events`)
11 random event types:
- **Natural Disasters**: Minor Natural Disaster, Major Natural Disaster
- **Technology Breakthrough**: Accidental technology discovery
- **Social Reform**: Improve organizational efficiency
- **Resource Discovery**: Increase resource reserves
- **Pandemic Outbreak**: Significant population and stability decline
- **Migration Wave**: Population growth bringing opportunities and challenges
- **Trade Agreement**: Promote economic prosperity
- **Cultural Exchange**: Promote technological progress and social stability
- **Resource Depletion**: Affect civilization development
- **Diplomatic Crisis**: Deteriorated relations, potentially leading to conflicts

### Advanced Evolution (`civsim.evolution`)
- Game theory decision-making
- Metacognitive learning
- Complex resource management
- Cultural influence system

### Performance Optimization (`civsim.performance`)
- Vectorized operations
- Batch computations
- Optimized simulation loops

### Real-time Animation (`civsim.animation`)
- Animation visualization
- Real-time updates
- Multiple chart types

## Technology Tree

Technologies are organized into four levels:

### Basic Technologies (Level 1)
- **Agriculture** - Improve resource output and territorial growth
- **Military** - Enhance military strength and defense capability
- **Trade** - Increase resources and diplomatic capability
- **Science** - Improve research speed and technology discovery

### Intermediate Technologies (Level 2)
- **Irrigation System** - Improve resource output and territorial value
- **Fortifications** - Enhance defense capability and stability
- **Currency System** - Improve resources and trade efficiency
- **Engineering** - Improve research speed and infrastructure

### Advanced Technologies (Level 3)
- **Industrial Agriculture** - Significantly improve resources and population growth
- **Advanced Tactics** - Enhance military strength and tactical advantage
- **Global Trade** - Improve resources and diplomatic capability
- **Advanced Science** - Significantly improve research speed and innovation capability

### Top-level Technologies (Level 4)
- **Genetic Engineering** - Improve resources, population growth, and health
- **Nuclear Technology** - Significantly enhance military strength, defense, and energy efficiency
- **Space Colonization** - Improve territorial growth, resource acquisition, and global influence
- **Artificial Intelligence** - Significantly improve research speed, innovation, and decision quality

## Strategy Types

Civilizations can adopt 7 strategies:

1. **Expansion** - Acquire more territory and resources
2. **Defense** - Enhance military strength and defensive capability
3. **Trade** - Exchange resources with allies
4. **Research** - Invest resources in technology development
5. **Diplomacy** - Establish and maintain relationships with other civilizations
6. **Culture** - Promote cultural influence
7. **Religion** - Spread religious beliefs

## Performance

### Optimization Features
- ✅ NumPy vectorized computing
- ✅ Batch operation optimization
- ✅ Memory-efficient data structures
- ✅ Cached calculation results

### Performance Metrics

| Scenario | Civilizations | Cycles | Simulation Time | Peak Memory |
|----------|--------------|---------|-----------------|-------------|
| Small | 3 | 100 | ~0.3s | ~30MB |
| Medium | 5 | 200 | ~0.8s | ~50MB |
| Large | 10 | 500 | ~3.5s | ~150MB |
| Extra Large | 20 | 1000 | ~12s | ~350MB |

## Configuration Presets

The system provides multiple preset configurations:

- `demo` - Demo configuration (4 civilizations, 100 cycles)
- `small` - Small-scale configuration (5 civilizations, 200 cycles)
- `medium` - Medium-scale configuration (8 civilizations, 300 cycles)
- `large` - Large-scale configuration (10 civilizations, 500 cycles)
- `resource_scarcity` - Resource scarcity configuration
- `tech_focus` - Technology-focused configuration
- `advanced_evolution` - Advanced evolution configuration

## Visualization Features

### Supported Chart Types
- Strategy heatmaps
- Evolution trend curves
- Technology progress charts
- Technology tree comparison
- Attribute comparison charts
- Civilization comprehensive capability radar charts
- Relationship network graphs

### Data Export Formats
- CSV - Tabular data
- JSON - Structured data
- NPZ - NumPy compressed data

## Development

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_core.py::test_config_creation -v

# View test coverage
pytest tests/ --cov=civsim --cov-report=html
```

### Code Quality

```bash
# Run type checking
mypy civsim

# Run code formatting
black civsim

# Run linting
flake8 civsim
```

## FAQ

### Q: How do I adjust simulation parameters?
A: You can adjust parameters through command line arguments or custom configuration files. Refer to `example_config.py` for all available parameters.

### Q: How do I save and load simulation states?
A: Use the `--save` parameter to save results, or use `--resume <file>` to resume from a saved file.

### Q: What visualization formats are supported?
A: PNG format for static images and GIF format for animations are supported.

### Q: How do I add new random events?
A: Add new event definitions and effect functions in `civsim/events.py`.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Submit a Pull Request

## License

MIT License - See [LICENSE](LICENSE) file for details

## Acknowledgments

Thanks to all contributors for their support and help!

---

<div align="center">
  <sub>Build with ❤️ for simulation research</sub>
</div>

**Version**: v0.2.0
**Last Updated**: 2025-12-30
