"""Core simulation module moved into civsim package.

This file is a direct copy of the top-level `multi_agent_simulation.py` to
allow an incremental migration into the package. It intentionally keeps the
same imports for now to minimize required changes during the move.
"""

import numpy as np
from .technology import TechnologyManager as TechTree

# NOTE: file contents copied from top-level module to preserve behavior


class CivilizationAgent:
    """Refactored CivilizationAgent with clearer strategy stages and safety checks.

    This refactor keeps backward-compatible attributes while splitting
    decision logic into smaller methods that are easier to extend and test.
    """

    DEFAULT_STRATEGY_NAMES = [
        "expansion",
        "defense",
        "trade",
        "research",
        "diplomacy",
        "culture",
        "religion",
    ]

    def __init__(
        self,
        agent_id: int,
        initial_strength: float | None = None,
        resources: float | None = None,
        strategy_names: list | None = None,
    ):
        self.agent_id: int = int(agent_id)
        # Use default values if not provided
        self.strength: float = float(initial_strength) if initial_strength is not None else 50.0
        self.defense: float = float(self.strength)
        self.resources: float = float(resources) if resources is not None else 200.0

        # State containers
        self.territory: set = set()
        self.allies: set = set()
        self.enemies: set = set()

        # Technology
        self.technology: dict = {"agriculture": 1, "military": 1, "trade": 1, "science": 1}
        self.research_progress: float = 0.0
        self.current_research: str | None = None
        self.research_cost: float = 0.0
        self.tech_tree: TechTree = TechTree()
        self.tech_bonuses: dict = {}
        self._tech_bonuses_cached: bool = False
        self._tech_bonuses_hash: int = 0

        # Demographics & attributes
        self.population: float = 100.0
        self.infrastructure: float = 1.0
        self.stability: float = 1.0
        self.health: float = 1.0
        self.energy_efficiency: float = 1.0
        self.resource_acquisition: float = 1.0
        self.global_influence: float = 1.0
        self.decision_quality: float = 1.0

        # Misc
        self.trade_history: list = []
        self.conflict_history: list = []
        self.tech_spillover_received: float = 0.0
        self.last_strategy = None
        self.culture = None
        self.cultural_bonuses: dict = {}
        self.religious_influence: float = 1.0
        self.religious_followers: int = 0
        self.relationship_weights: dict = {}
        # Defensive defaults for optional attributes used in decision adjustments
        self.cultural_influence: float = 0.0

        # Strategy configuration
        self.strategy_names = (
            strategy_names if strategy_names is not None else list(self.DEFAULT_STRATEGY_NAMES)
        )
        self.strategy_count = len(self.strategy_names)

        # initialize derived values
        self.update_tech_bonuses()

    # --------------------------- Tech / bonuses ---------------------------
    def _compute_tech_hash(self) -> int:
        """Compute hash of current technology state for caching."""
        return hash(frozenset(sorted(self.technology.items())))

    def update_tech_bonuses(self, force: bool = False) -> None:
        """Recompute tech bonuses and apply them to derived attributes.

        Args:
            force: Force recomputation even if cached
        """
        current_hash = self._compute_tech_hash()

        if not force and self._tech_bonuses_cached and self._tech_bonuses_hash == current_hash:
            return

        bonuses_obj = self.tech_tree.calculate_bonuses(self.technology)
        self.tech_bonuses = {
            "infrastructure": bonuses_obj.infrastructure,
            "stability": bonuses_obj.stability,
            "health": bonuses_obj.health,
            "energy_efficiency": bonuses_obj.energy_efficiency,
            "resource_acquisition": bonuses_obj.resource_acquisition,
            "global_influence": bonuses_obj.global_influence,
            "decision_quality": bonuses_obj.decision_quality,
            "resources": bonuses_obj.resources,
            "strength": bonuses_obj.strength,
            "defense": bonuses_obj.defense,
            "research_speed": bonuses_obj.research_speed,
            "diplomacy": bonuses_obj.diplomacy,
            "territory_growth": bonuses_obj.territory_growth,
            "territory_value": bonuses_obj.territory_value,
            "trade_efficiency": bonuses_obj.trade_efficiency,
            "population_growth": bonuses_obj.population_growth,
            "tactical_advantage": bonuses_obj.tactical_advantage,
            "tech_discovery": bonuses_obj.tech_discovery,
            "innovation": bonuses_obj.innovation,
        }

        default_attrs = {
            "infrastructure": self.infrastructure,
            "stability": self.stability,
            "health": self.health,
            "energy_efficiency": self.energy_efficiency,
            "resource_acquisition": self.resource_acquisition,
            "global_influence": self.global_influence,
            "decision_quality": self.decision_quality,
        }

        for attr, default_val in default_attrs.items():
            setattr(self, attr, float(self.tech_bonuses.get(attr, default_val)))

        self._tech_bonuses_cached = True
        self._tech_bonuses_hash = current_hash

    # --------------------------- Decision pipeline ------------------------
    def decide_strategy(
        self, neighbors: dict, global_resources: dict, advanced_evolution=None
    ) -> np.ndarray:
        """Return a normalized strategy distribution (length = strategy_count).

        neighbors: dict mapping agent_id -> (strength, relationship_weight)
        global_resources: mapping of territory id -> resource value
        advanced_evolution: optional pluggable decision engine (ignored here if None)
        """
        if advanced_evolution is not None:
            try:
                out = advanced_evolution.evaluate(self, neighbors, global_resources)
                arr = np.asarray(out, dtype=float)
                if arr.size == self.strategy_count:
                    if np.sum(arr) > 0:
                        return arr / np.sum(arr)
                    return np.ones(self.strategy_count) / self.strategy_count
            except (AttributeError, ValueError, TypeError) as e:
                from .logger import get_logger

                logger = get_logger(__name__)
                logger.warning(f"Advanced evolution failed, falling back to default logic: {e}")
            except Exception as e:
                from .logger import get_logger

                logger = get_logger(__name__)
                logger.warning(f"Unexpected error in advanced evolution: {e}")

        # Build strategy vector from modular assessments
        strategy = np.zeros(self.strategy_count, dtype=float)
        strategy += self._base_weights()
        strategy += self._threat_adjustment(neighbors)
        strategy += self._resource_adjustment(global_resources)
        strategy += self._research_adjustment()
        strategy += self._diplomacy_and_culture_adjustment(neighbors)
        strategy = np.clip(strategy, 0.0, None)

        # Ensure non-zero and normalized
        total = np.sum(strategy)
        if total <= 0.0:
            return np.ones(self.strategy_count) / self.strategy_count
        return strategy / total

    def _base_weights(self) -> np.ndarray:
        """Base weights derived from stable attributes like infrastructure and influence."""
        infra = float(self.infrastructure)
        stability = float(self.stability) if self.stability > 0 else 1.0
        global_inf = float(self.global_influence)
        research_speed = float(self.tech_bonuses.get("research_speed", 1.0))

        # Map to the expected strategy_count (default 7)
        w = np.zeros(self.strategy_count)
        # expansion
        w[0] = 0.2 * infra
        # defense
        w[1] = 0.15 * (1.0 / stability)
        # trade
        w[2] = 0.25 * global_inf
        # research
        w[3] = 0.4 * research_speed
        # diplomacy
        w[4] = 0.2 * global_inf
        # culture
        if self.strategy_count > 5:
            w[5] = 0.15 * self.cultural_influence
        # religion
        if self.strategy_count > 6:
            w[6] = 0.1 * self.religious_influence
        return w

    def _threat_adjustment(self, neighbors: dict) -> np.ndarray:
        """Increase defense when enemy strength nearby exceeds agent strength."""
        adj = np.zeros(self.strategy_count)
        rel_threshold = 0.5
        enemy_strength = sum(
            n_strength for _, (n_strength, rel) in neighbors.items() if rel < -rel_threshold
        )
        effective_strength = float(self.strength) * float(self.tech_bonuses.get("strength", 1.0))
        if effective_strength <= 0:
            return adj
        if enemy_strength > effective_strength * 1.2 and self.strategy_count > 1:
            threat_factor = min(enemy_strength / max(effective_strength, 1e-6), 2.0)
            adj[1] = 0.6 + min(0.3, threat_factor * 0.1)
        return adj

    def _resource_adjustment(self, global_resources: dict) -> np.ndarray:
        """Adjust expansion/trade preferences based on resource pressure."""
        adj = np.zeros(self.strategy_count)
        if not self.territory:
            return adj
        controlled_resources = sum(global_resources.get(h3_id, 0.0) for h3_id in self.territory)
        if controlled_resources <= 0:
            return adj
        resource_pressure = 1.0 - (self.resources / (controlled_resources * 10.0))
        # high pressure -> expansion or trade
        if resource_pressure > 0.8:
            if len(self.allies) > 0 and self.strategy_count > 2:
                adj[2] = 0.5 + (1.0 - resource_pressure) * 0.4 * float(
                    self.tech_bonuses.get("trade_efficiency", 1.0)
                )
            else:
                adj[0] = 0.5 + resource_pressure * 0.3 * float(self.resource_acquisition)
        # population-driven expansion bonus
        effective_population = (
            float(self.population)
            * float(self.health)
            * float(self.tech_bonuses.get("population_growth", 1.0))
        )
        if effective_population > 500 and self.health > 1.5 and self.strategy_count > 0:
            adj[0] += 0.2
        return adj

    def _research_adjustment(self) -> np.ndarray:
        adj = np.zeros(self.strategy_count)
        available_techs = self.tech_tree.get_available_techs(self.technology)
        if self.resources > 1000 and available_techs and self.strategy_count > 3:
            has_high_level = any(tech["level"] >= 3 for tech in available_techs)
            if has_high_level:
                adj[3] = 0.6 + 0.1 * float(self.tech_bonuses.get("research_speed", 1.0))
        # boost research if we already have top tech
        if (
            any(t in self.technology for t in ("artificial_intelligence", "nuclear_technology"))
            and self.strategy_count > 3
        ):
            adj[3] += 0.2 * float(self.decision_quality)
        return adj

    def _diplomacy_and_culture_adjustment(self, neighbors: dict) -> np.ndarray:
        adj = np.zeros(self.strategy_count)
        if len(neighbors) > 0 and self.strategy_count > 4:
            adj[4] = 0.3 + 0.2 * (len(self.allies) / max(1, len(neighbors)))
        if self.strategy_count > 5 and hasattr(self, "cultural_influence"):
            adj[5] = 0.2 * float(self.cultural_influence)
        if self.strategy_count > 6:
            adj[6] = 0.1 * float(self.religious_influence)
        return adj

    # --------------------------- Actions / effects -----------------------
    def apply_cultural_bonuses(self) -> None:
        if self.cultural_bonuses:
            self.strength *= 1 + self.cultural_bonuses.get("military_strength", 0.0)
            self.population *= 1 + self.cultural_bonuses.get("population_growth", 0.0)

    def _should_research(self) -> bool:
        return len(self.tech_tree.get_available_techs(self.technology)) > 0

    # --------------------------- Small action helpers -------------------
    def _collect_resource_output(self, global_resources: dict) -> None:
        """Collect periodic resources from territory into agent reserves."""
        if not self.territory:
            return
        base_output = sum(global_resources.get(h3, 0.0) for h3 in self.territory) * 0.1
        resource_bonus = float(self.tech_bonuses.get("resources", 1.0))
        territory_value_bonus = float(self.tech_bonuses.get("territory_value", 1.0))
        self.resources += base_output * resource_bonus * territory_value_bonus

    def _conduct_diplomacy(self, neighbors: dict) -> None:
        """Simple diplomacy: try to convert neutral relationships into allies if resources allow."""
        for nid, (_, rel) in neighbors.items():
            if rel > 0.2 and nid not in self.allies:
                if np.random.rand() < 0.05:
                    self.allies.add(nid)

    def _promote_culture(self) -> None:
        """Promote culture to gain small population or influence bonuses."""
        bonus = 0.01 * 0.2
        self.population += self.population * bonus

    def _spread_religion(self, neighbors: dict) -> None:
        """Spread religion to neighbors increasing religious_followers metric slightly."""
        for _ in neighbors.keys():
            if np.random.rand() < 0.02:
                self.religious_followers += 1

    def execute_strategy(self, strategy, neighbors, global_resources):
        """Execute strategy and update status"""
        # Ensure parameters exist and are valid
        if strategy is None or not hasattr(strategy, "__len__"):
            return
        # Handle possible 7-dimensional strategy (expansion/defense/trade/research/diplomacy/culture/religion)
        if len(strategy) == 7:
            (
                expansion_prob,
                defense_prob,
                trade_prob,
                research_prob,
                diplomacy_prob,
                culture_prob,
                religion_prob,
            ) = strategy
        elif len(strategy) == 6:
            (
                expansion_prob,
                defense_prob,
                trade_prob,
                research_prob,
                diplomacy_prob,
                culture_prob,
            ) = strategy
            religion_prob = 0
        elif len(strategy) == 4:
            expansion_prob, defense_prob, trade_prob, research_prob = strategy
            diplomacy_prob, culture_prob, religion_prob = 0, 0, 0
        else:
            expansion_prob, defense_prob, trade_prob = strategy
            research_prob, diplomacy_prob, culture_prob, religion_prob = 0, 0, 0, 0

        # Expansion logic
        if np.random.rand() < expansion_prob and len(self.territory) < 50:
            # Select the most resource-rich uncontrolled area
            available_h3 = [h3_id for h3_id in global_resources if h3_id not in self.territory]
            if available_h3:
                target = max(available_h3, key=lambda x: global_resources[x])
                # Apply territory growth bonus
                territory_bonus = self.tech_bonuses.get("territory_growth", 1.0)
                self.territory.add(target)
                # Apply territory value bonus
                territory_value_bonus = self.tech_bonuses.get("territory_value", 1.0)
                self.resources += (
                    global_resources[target] * 0.3 * territory_value_bonus * territory_bonus
                )

        # Defense logic
        if np.random.rand() < defense_prob:
            # Apply military bonus
            defense_bonus = self.tech_bonuses.get("defense", 1.0)
            self.strength *= 1.05 * defense_bonus  # Military enhancement
            self.resources *= 0.95  # Resource consumption

        # Trade logic
        if np.random.rand() < trade_prob and len(self.allies) > 0:
            # Randomly select ally for resource exchange
            ally_id = np.random.choice(list(self.allies)) if self.allies else None
            if ally_id:
                # Apply trade efficiency bonus
                trade_efficiency_bonus = self.tech_bonuses.get("trade_efficiency", 1.0)
                trade_amount = int(self.resources * 0.1)
                self.resources += trade_amount * 0.1 * trade_efficiency_bonus  # Trade profit
                self.resources -= trade_amount

        # Research logic
        if np.random.rand() < research_prob:
            self._research_technology()

        # Diplomacy logic
        if np.random.rand() < diplomacy_prob:
            self._conduct_diplomacy(neighbors)

        # Culture logic
        if np.random.rand() < culture_prob:
            self._promote_culture()

        # Religion logic
        if np.random.rand() < religion_prob:
            self._spread_religion(neighbors)

        # Periodically collect resource output
        self._collect_resource_output(global_resources)

    def _research_technology(self):
        """Conduct technology research"""
        # If there is no technology currently being researched, select a new one
        if self.current_research is None:
            available_techs = self.tech_tree.get_available_techs(self.technology)
            if available_techs:
                # Select available technologies and calculate costs
                affordable_techs = []
                for tech_info in available_techs:
                    tech_name = tech_info["name"]
                    current_level = self.technology.get(tech_name, 0)
                    cost = self.tech_tree.get_research_cost(tech_name, current_level + 1)
                    if cost <= self.resources:
                        affordable_techs.append((tech_name, cost, tech_info["level"]))

                if affordable_techs:
                    # Enhanced priority selection strategy
                    def tech_priority(tech_tuple):
                        tech_name, cost, tech_level = tech_tuple
                        # Get the bonus effects of the technology (temporarily build a single technology dictionary to calculate)
                        current_level = self.technology.get(tech_name, 0)
                        temp_tech = {tech_name: current_level + 1}
                        bonuses_obj = self.tech_tree.calculate_bonuses(temp_tech)
                        bonuses = {
                            "research_speed": bonuses_obj.research_speed,
                            "strength": bonuses_obj.strength,
                            "defense": bonuses_obj.defense,
                            "resources": bonuses_obj.resources,
                            "infrastructure": bonuses_obj.infrastructure,
                            "stability": bonuses_obj.stability,
                            "health": bonuses_obj.health,
                            "energy_efficiency": bonuses_obj.energy_efficiency,
                            "resource_acquisition": bonuses_obj.resource_acquisition,
                            "global_influence": bonuses_obj.global_influence,
                            "decision_quality": bonuses_obj.decision_quality,
                            "diplomacy": bonuses_obj.diplomacy,
                            "territory_growth": bonuses_obj.territory_growth,
                            "territory_value": bonuses_obj.territory_value,
                            "trade_efficiency": bonuses_obj.trade_efficiency,
                            "population_growth": bonuses_obj.population_growth,
                            "tactical_advantage": bonuses_obj.tactical_advantage,
                            "tech_discovery": bonuses_obj.tech_discovery,
                            "innovation": bonuses_obj.innovation,
                        }

                        priority = 0

                        # Calculate effective attribute values (considering current state needs)
                        effective_research_speed = 1.0 + self.tech_bonuses.get("research_speed", 0)
                        effective_strength = 1.0 + self.tech_bonuses.get("strength", 0)
                        effective_defense = 1.0 + self.tech_bonuses.get("defense", 0)

                        # Special handling for top-level technologies
                        if tech_level == 4:
                            priority += 10  # Top-level technology has extra priority

                        # Dynamically adjust weights based on current state
                        if (
                            effective_research_speed < effective_strength
                            and effective_research_speed < effective_defense
                        ):
                            if "research_speed" in bonuses:
                                priority += (
                                    bonuses["research_speed"] * 2.0
                                )  # Higher priority when research is lagging
                        elif effective_strength < effective_defense and self.enemies:
                            if "strength" in bonuses:
                                priority += (
                                    bonuses["strength"] * 1.5
                                )  # Prioritize military when there are enemies and strength is weak

                        # Basic attribute bonuses
                        if "resources" in bonuses:
                            priority += bonuses["resources"] * 1.0
                        if "territory_value" in bonuses:
                            priority += bonuses["territory_value"] * 0.8
                        if "population_growth" in bonuses:
                            priority += bonuses["population_growth"] * 0.7

                        # Cost-benefit analysis
                        priority /= cost / 100  # Divide by normalized cost

                        return priority

                    affordable_techs.sort(key=tech_priority, reverse=True)
                    self.current_research, self.research_cost, _ = affordable_techs[0]
            # Conduct research
        if self.current_research is not None:
            # Calculate research speed (considering population, infrastructure, and research investment)
            research_base_speed = self.tech_bonuses.get("research_speed", 1.0) * 1.0
            population_bonus = min(self.population / 100, 5.0)  # Population bonus cap
            infrastructure_bonus = (
                self.infrastructure / 100
            )  # Infrastructure provides additional speed

            # Apply technology spillover effect
            tech_spillover_effect = 0.0
            spillover_bonus = self.tech_spillover_received * tech_spillover_effect

            total_research_speed = research_base_speed * (
                1 + population_bonus + infrastructure_bonus + spillover_bonus
            )

            # Invest resources in research (file truncated)
