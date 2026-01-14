# Getting Started

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
git clone https://github.com/johboby/Civilization.git
cd Civilization
pip install -r requirements.txt
```

## Basic Usage

### Run Demo

```bash
# Basic demo
python run_demo.py basic

# Advanced evolution demo
python run_demo.py advanced
```

### Command Line Interface

```bash
# Basic simulation
python simulation_cli.py --cycles 100 --num-civs 5

# Interactive mode
python simulation_cli.py --interactive
```

### Interactive Demo

```bash
python interactive_demo.py --preset medium
```

## Configuration

The system supports multiple configuration presets:

- `demo`: Quick demonstration (4 civilizations, 100 cycles)
- `small`: Small scale (5 civilizations, 200 cycles)
- `medium`: Medium scale (8 civilizations, 300 cycles)
- `large`: Large scale (10 civilizations, 500 cycles)

## Next Steps

- Read the [Configuration](Configuration) guide
- Explore the [API Reference](API-Reference)
- Check out the [FAQ](FAQ)