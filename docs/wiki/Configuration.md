# Configuration

## Configuration System

The simulation uses a type-safe configuration system defined in `civsim.config`.

### Basic Configuration

```python
from civsim.config import SimulationConfig

config = SimulationConfig(
    num_civilizations=5,
    simulation_cycles=200,
    grid_size=150,
    random_seed=42,
    enable_random_events=True,
    use_advanced_evolution=False,
    use_complex_resources=False,
    use_cultural_influence=False
)
```

### Preset Configurations

The system provides several preset configurations:

#### Demo Preset
```python
config = get_preset("demo")  # 4 civilizations, 100 cycles
```

#### Small Preset
```python
config = get_preset("small")  # 5 civilizations, 200 cycles
```

#### Medium Preset
```python
config = get_preset("medium")  # 8 civilizations, 300 cycles
```

#### Large Preset
```python
config = get_preset("large")  # 10 civilizations, 500 cycles
```

### Advanced Options

#### Random Events
- `enable_random_events`: Enable/disable random events
- `event_probability_modifier`: Adjust event frequency (default: 1.0)

#### Evolution Features
- `use_advanced_evolution`: Enable advanced AI decision making
- `use_complex_resources`: Enable complex resource distribution
- `use_cultural_influence`: Enable cultural influence system

### Custom Configuration

You can create custom configurations by extending `SimulationConfig`:

```python
class MyConfig(SimulationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = kwargs.get("custom_param", 10)
```

## Command Line Options

The CLI supports various options:

```bash
python simulation_cli.py --help
```

Common options:
- `--cycles`: Number of simulation cycles
- `--num-civs`: Number of civilizations
- `--preset`: Use preset configuration
- `--save`: Save results
- `--interactive`: Enable interactive mode