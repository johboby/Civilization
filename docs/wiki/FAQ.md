# FAQ

## General Questions

### What is this project?
This is a multi-agent civilization evolution simulation that models how civilizations develop, interact, and evolve over time using artificial intelligence and game theory.

### What technologies are used?
- Python 3.8+
- NumPy for numerical computations
- Matplotlib for visualization
- Type hints and dataclasses for type safety

## Usage Questions

### How do I adjust simulation parameters?
You can modify parameters through:
- Command line arguments: `python simulation_cli.py --cycles 500 --num-civs 10`
- Configuration presets: `get_preset("large")`
- Custom configuration classes

### How do I save and load simulation results?
Use the `--save` flag to save results:
```bash
python simulation_cli.py --save --output-dir my_results
```

Results are saved in NPZ, JSON, and CSV formats.

### What do the strategy types mean?
- **Expansion**: Focus on acquiring territory and resources
- **Defense**: Build military strength and fortifications
- **Trade**: Economic cooperation with other civilizations
- **Research**: Invest in technological development
- **Diplomacy**: Build alliances and relationships
- **Culture**: Spread cultural influence
- **Religion**: Expand religious beliefs

## Technical Questions

### How does the AI decision making work?
The system uses multiple decision engines:
- Rule-based heuristics (default)
- Game theory optimization (advanced evolution)
- Strategy probability distributions
- Dynamic adaptation based on current state

### What are technology bonuses?
Each technology provides bonuses to various attributes:
- Resources, Strength, Defense
- Research Speed, Diplomacy
- Territory Growth, Trade Efficiency
- Population Growth, Infrastructure
- And 9 other attributes

### How are random events implemented?
Events are defined with:
- Probability of occurrence
- Effect functions that modify agent states
- Severity levels (minor, major, positive)
- Categories for organization

## Performance Questions

### How fast is the simulation?
Performance depends on scale:

| Scale | Civilizations | Cycles | Time | Memory |
|-------|---------------|--------|------|--------|
| Small | 3 | 100 | ~0.3s | ~30MB |
| Medium | 5 | 200 | ~0.8s | ~50MB |
| Large | 10 | 500 | ~3.5s | ~150MB |

### Can it run larger simulations?
Yes, but performance degrades with scale. The system uses:
- NumPy vectorized operations
- Batch processing
- Memory-efficient data structures

## Development Questions

### How do I add a new technology?
1. Add to `TechnologyManager._initialize_tech_tree()`
2. Define bonuses in `_initialize_tech_effects()`
3. Update dependencies if needed

### How do I create a custom decision engine?
Extend `StrategyDecisionEngine` and implement the `decide()` method:

```python
class MyEngine(StrategyDecisionEngine):
    def decide(self, state, neighbors, global_resources):
        # Your logic here
        return [StrategyResult(...)]
```

### How do I add random events?
Add to `RandomEventManager._initialize_events()`:

```python
events['my_event'] = EventInfo(
    name="My Event",
    description="Description",
    probability=0.1,
    effect=my_effect_function,
    severity="minor",
    categories=["custom"]
)
```

## Troubleshooting

### Simulation runs slowly
- Reduce number of civilizations or cycles
- Disable advanced features if not needed
- Check for memory leaks in long runs

### Visualization doesn't work
- Ensure matplotlib is installed
- Check display settings for headless environments
- Try saving to file instead of showing

### Import errors
- Install all requirements: `pip install -r requirements.txt`
- Check Python version compatibility
- Ensure you're in the correct directory

### Random events not triggering
- Check `enable_random_events` setting
- Adjust `event_probability_modifier`
- Events may be rare by design

## Getting Help

- Check existing [issues](https://github.com/johboby/Civilization/issues)
- Read the [documentation](https://github.com/johboby/Civilization/blob/main/README.md)
- Create a new issue with detailed information