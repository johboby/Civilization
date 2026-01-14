"""
Civilization simulation system - Configuration management module

This module provides type-safe configuration management using dataclasses instead of traditional class attribute approach.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class SimulationConfig:
    """Simulation configuration class, using dataclasses to provide type safety and better maintainability"""

    # ====================== Basic simulation parameters ======================
    num_civilizations: int = 4
    grid_size: int = 20
    simulation_cycles: int = 100
    random_seed: Optional[int] = None
    initial_strength: float = 50.0
    initial_resources: float = 200.0
    initial_population: float = 100.0

    # ====================== Resource system parameters ======================
    resource_abundance: float = 10.0
    resource_regeneration_rate: float = 0.02
    resource_cap: float = 100.0
    resource_consumption_per_action: float = 1.0
    resource_distribution_method: str = "random"
    resource_per_capita_for_max_growth: float = 10.0
    max_resource_per_cell: float = 100.0
    territory_value_coefficient: float = 1.0
    global_resource_regeneration: bool = True

    # ====================== Population system parameters ======================
    population_growth_rate: float = 0.01
    population_consumption_rate: float = 0.5
    population_cap_per_territory_value: float = 2.0
    population_growth_cap: float = float('inf')

    # ====================== Technology research parameters ======================
    tech_research_rate: float = 0.1
    tech_research_cost: float = 10.0
    tech_spillover_effect: float = 0.3
    tech_bonus_diminishing_rate: float = 0.1
    population_research_bonus_cap: float = 5.0
    base_research_speed: float = 1.0
    research_resource_efficiency: float = 1.0

    # ====================== Territory expansion parameters ======================
    territory_expansion_cost: float = 2.0
    territory_base_value: float = 1.0
    territory_development_bonus: float = 0.1
    territory_expansion_chance: float = 0.3
    max_expansion_per_cycle: int = 1

    # ====================== Relationship adjustment parameters ======================
    relationship_change_rate: float = 0.05
    allyship_bonus: float = 0.2
    enemy_penalty: float = 0.2
    relationship_threshold: float = 0.3

    # ====================== Randomization parameters ======================
    random_fluctuation_amplitude: float = 0.1
    stochasticity_factor: float = 0.2
    random_territory_distribution: bool = False

    # ====================== Run mode parameters ======================
    parallel_processing: bool = False
    fast_mode: bool = False
    print_logs: bool = True
    log_interval: int = 10

    # ====================== Visualization parameters ======================
    visualization_enabled: bool = True
    visualization_interval: int = 5
    figure_size: tuple = (12, 8)
    dpi: int = 300

    # ====================== File output parameters ======================
    output_dir: str = "results"
    save_results: bool = True
    export_format: str = "csv"

    # ====================== Advanced evolution system parameters ======================
    use_advanced_evolution: bool = False
    use_complex_resources: bool = False
    use_cultural_influence: bool = False

    # ====================== Random event system parameters ======================
    enable_random_events: bool = True
    event_probability_modifier: float = 1.0

    # ====================== Advanced evolution parameters ======================
    evolution_learning_rate: float = 0.1
    strategy_exploration_rate: float = 0.2
    memory_window_size: int = 10
    evolution_randomness_factor: float = 0.1

    # ====================== Complex resource system parameters ======================
    resource_types: List[str] = field(default_factory=lambda: ["food", "energy", "minerals", "technology"])
    resource_spot_variance: float = 0.3
    resource_interaction_weight: float = 0.2

    # ====================== Cultural influence system parameters ======================
    cultural_diffusion_rate: float = 0.05
    cultural_resistance_factor: float = 0.1
    cultural_bonus_strength: float = 0.2
    num_cultural_traits: int = 5

    # ====================== Religion system parameters ======================
    enable_religious_influence: bool = False
    religious_influence_factor: float = 0.1
    religious_conversion_rate: float = 0.05

    def validate(self) -> List[str]:
        """Validate configuration parameter validity

        Returns:
            List of error messages, returns empty list if configuration is valid
        """
        errors = []

        if self.num_civilizations <= 0:
            errors.append("num_civilizations must be positive")
        if self.grid_size <= 0:
            errors.append("grid_size must be positive")
        if self.simulation_cycles <= 0:
            errors.append("simulation_cycles must be positive")
        if self.initial_resources < 0:
            errors.append("initial_resources must be non-negative")
        if self.initial_population <= 0:
            errors.append("initial_population must be positive")
        if not 0 <= self.population_growth_rate <= 1:
            errors.append("population_growth_rate must be between 0 and 1")
        if not 0 <= self.tech_spillover_effect <= 1:
            errors.append("tech_spillover_effect must be between 0 and 1")

        return errors

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SimulationConfig':
        """Create configuration instance from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            SimulationConfig instance
        """
        # Filter out unknown keys
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary

        Returns:
            Configuration dictionary
        """
        from dataclasses import asdict
        return asdict(self)


# Preset configurations
def get_preset(name: str) -> SimulationConfig:
    """Get preset configuration

    Args:
        name: Preset name

    Returns:
        Corresponding configuration instance
    """
    presets = {
        "small_scale": SimulationConfig(
            num_civilizations=2,
            grid_size=10,
            simulation_cycles=50,
            tech_spillover_effect=0.5
        ),
        "medium_scale": SimulationConfig(
            num_civilizations=4,
            grid_size=20,
            simulation_cycles=100,
            tech_spillover_effect=0.3
        ),
        "large_scale": SimulationConfig(
            num_civilizations=8,
            grid_size=30,
            simulation_cycles=200,
            tech_spillover_effect=0.2
        ),
        "demo": SimulationConfig(
            num_civilizations=4,
            grid_size=15,
            simulation_cycles=100,
            tech_spillover_effect=0.3,
            print_logs=True,
            log_interval=10,
            visualization_enabled=True
        ),
    }

    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

    return presets[name]


# Global default configuration instance
config = SimulationConfig()
default_config = config  # Compatibility alias
