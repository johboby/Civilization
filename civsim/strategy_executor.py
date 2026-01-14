"""Strategy execution module

This module is responsible for executing various strategies for civilization agents, separating strategy execution logic from CivilizationAgent.
"""
import numpy as np
from typing import Dict, Set, Any, Tuple


class StrategyExecutor:
    """Strategy executor - responsible for executing various civilization strategies"""

    def __init__(self, agent):
        """Initialize strategy executor

        Args:
            agent: Civilization agent instance
        """
        self.agent = agent

    def execute(self, strategy: np.ndarray, neighbors: Dict[int, Tuple[float, float]], global_resources: Dict[Any, float]) -> None:
        """Execute strategy and update agent state

        Args:
            strategy: Strategy probability array
            neighbors: Neighbor agent mapping {agent_id -> (strength, relationship)}
            global_resources: Global resource mapping
        """
        # Ensure parameters exist and are valid
        if strategy is None or not hasattr(strategy, '__len__'):
            return

        # Handle possible 7-dimensional strategy (expansion/defense/trade/research/diplomacy/culture/religion)
        if len(strategy) == 7:
            expansion_prob, defense_prob, trade_prob, research_prob, diplomacy_prob, culture_prob, religion_prob = strategy
        elif len(strategy) == 6:
            expansion_prob, defense_prob, trade_prob, research_prob, diplomacy_prob, culture_prob = strategy
            religion_prob = 0
        elif len(strategy) == 4:
            expansion_prob, defense_prob, trade_prob, research_prob = strategy
            diplomacy_prob = culture_prob = religion_prob = 0
        elif len(strategy) == 3:
            expansion_prob, defense_prob, trade_prob = strategy
            research_prob = diplomacy_prob = culture_prob = religion_prob = 0
        else:
            return

        # Expansion logic - acquire new territory
        if np.random.rand() < expansion_prob:
            self._execute_expansion(global_resources)

        # Defense logic - enhance military strength
        if np.random.rand() < defense_prob:
            self._execute_defense()

        # Trade logic - exchange resources with allies
        if np.random.rand() < trade_prob and len(self.agent.allies) > 0:
            self._execute_trade()

        # Research logic - invest in technology research
        if np.random.rand() < research_prob:
            self._execute_research()

        # Diplomacy logic - establish diplomatic relations
        if np.random.rand() < diplomacy_prob:
            self._execute_diplomacy(neighbors)

        # Culture logic - promote cultural influence
        if np.random.rand() < culture_prob:
            self._execute_culture()

        # Religion logic - spread religious beliefs
        if np.random.rand() < religion_prob:
            self._execute_religion(neighbors)

        # Periodically collect resource output
        self._collect_resource_output(global_resources)

    def _execute_expansion(self, global_resources: Dict[Any, float]) -> None:
        """Execute expansion strategy"""
        if not self.agent.territory:
            return

        # Find adjacent unoccupied cells
        for territory in self.agent.territory:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (territory[0] + dx, territory[1] + dy)
                if neighbor in global_resources and neighbor not in self.agent.territory:
                    border_cells.append(neighbor)

        if not border_cells:
            return

        # Select resource-rich target
        available_h3 = [h3 for h3 in border_cells if h3 not in self.agent.territory]
        if available_h3:
            target = max(available_h3, key=lambda x: global_resources[x])
            # Apply territory growth bonus
            territory_bonus = self.agent.tech_bonuses.get("territory_growth", 1.0)
            self.agent.territory.add(target)
            # Apply territory value bonus
            territory_value_bonus = self.agent.tech_bonuses.get("territory_value", 1.0)
            self.agent.resources += global_resources[target] * 0.3 * territory_value_bonus * territory_bonus

    def _execute_defense(self) -> None:
        """Execute defense strategy"""
        # Apply military bonus
        defense_bonus = self.agent.tech_bonuses.get("defense", 1.0)
        self.agent.strength *= 1.05 * defense_bonus
        self.agent.resources *= 0.95

    def _execute_trade(self) -> None:
        """Execute trade strategy"""
        # Randomly select ally for resource exchange
        ally = np.random.choice(list(self.agent.allies))
        trade_efficiency = self.agent.tech_bonuses.get("trade_efficiency", 1.0)
        trade_amount = 20.0 * trade_efficiency
        self.agent.resources += trade_amount

    def _execute_research(self) -> None:
        """Execute research strategy"""
        self._research_technology()

    def _execute_diplomacy(self, neighbors: Dict[int, Tuple[float, float]]) -> None:
        """Execute diplomacy strategy"""
        # Try to establish diplomatic relations
        if not neighbors:
            return

        neighbor_id = np.random.choice(list(neighbors.keys()))
        if np.random.rand() < 0.05:
            self.agent.relationship_weights[neighbor_id] = 0.5
            self.agent.allies.add(neighbor_id)

    def _execute_culture(self) -> None:
        """Execute culture strategy"""
        culture_bonus = self.agent.tech_bonuses.get("global_influence", 1.0)
        self.agent.global_influence += 0.02 * culture_bonus

    def _execute_religion(self, neighbors: Dict[int, Tuple[float, float]]) -> None:
        """Execute religion strategy"""
        for _ in neighbors.keys():
            if np.random.rand() < 0.02:
                self.agent.religious_followers += 1

    def _collect_resource_output(self, global_resources: Dict[Any, float]) -> None:
        """Collect resource output"""
        if not self.agent.territory:
            return

        base_output = sum(global_resources.get(h3, 0.0) for h3 in self.agent.territory) * 0.1
        resource_bonus = self.agent.tech_bonuses.get("resources", 1.0)
        acquisition_bonus = self.agent.tech_bonuses.get("resource_acquisition", 1.0)
        infrastructure_bonus = self.agent.infrastructure * 0.5

        self.agent.resources += base_output * resource_bonus * acquisition_bonus + infrastructure_bonus

    def _research_technology(self) -> None:
        """Technology research logic"""
        # If no technology is currently being researched, select new one
        if self.agent.current_research is None:
            available_techs = self.agent.tech_tree.get_available_techs(self.agent.technology)
            if available_techs:
                affordable_techs = []
                for tech_info in available_techs:
                    tech_name = tech_info["name"]
                    current_level = self.agent.technology.get(tech_name, 0)
                    cost = self.agent.tech_tree.get_research_cost(tech_name, current_level + 1)
                    if cost <= self.agent.resources:
                        affordable_techs.append((tech_name, cost, tech_info["level"]))

                if affordable_techs:
                    # Technology priority selection
                    def tech_priority(tech_tuple):
                        tech_name, cost, tech_level = tech_tuple
                        temp_tech = {tech_name: current_level + 1}
                        bonuses_obj = self.agent.tech_tree.calculate_bonuses(temp_tech)
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
                        effective_research_speed = 1.0 + self.agent.tech_bonuses.get("research_speed", 0)
                        effective_strength = 1.0 + self.agent.tech_bonuses.get("strength", 0)
                        effective_defense = 1.0 + self.agent.tech_bonuses.get("defense", 0)

                        # Special handling for top-level technologies
                        if tech_level == 4:
                            priority += 10

                        # Dynamically adjust weights based on current state
                        if effective_research_speed < effective_strength and effective_research_speed < effective_defense:
                            if "research_speed" in bonuses:
                                priority += bonuses["research_speed"] * 2.0
                        elif effective_strength < effective_defense and self.agent.enemies:
                            if "strength" in bonuses:
                                priority += bonuses["strength"] * 1.5

                        # Basic attribute bonuses
                        if "resources" in bonuses:
                            priority += bonuses["resources"] * 1.0
                        if "territory_value" in bonuses:
                            priority += bonuses["territory_value"] * 0.8
                        if "population_growth" in bonuses:
                            priority += bonuses["population_growth"] * 0.7

                        # Cost-benefit analysis
                        priority /= (cost / 100)

                        return priority

                    affordable_techs.sort(key=tech_priority, reverse=True)
                    self.agent.current_research, self.agent.research_cost, _ = affordable_techs[0]

        # Conduct research
        if self.agent.current_research is not None:
            # Calculate research speed (considering population, infrastructure, and research investment)
            research_base_speed = self.agent.tech_bonuses.get("research_speed", 1.0) * 1.0
            population_bonus = min(self.agent.population / 100, 5.0)
            infrastructure_bonus = (self.agent.infrastructure - 1.0) * 0.3

            # Technology spillover effect
            spillover_bonus = self.agent.tech_spillover_received * self.agent.tech_spillover_effect

            total_research_speed = research_base_speed * (1 + population_bonus + infrastructure_bonus + spillover_bonus)

            # Invest resources in research
            research_invest = min(self.agent.resources * 0.1, self.agent.research_cost)
            self.agent.resources -= research_invest

            # Research progress increases
            self.agent.research_progress += research_invest * total_research_speed

            # Research completed
            if self.agent.research_progress >= self.agent.research_cost:
                tech_name = self.agent.current_research
                current_level = self.agent.technology.get(tech_name, 0)

                self.agent.technology[tech_name] = current_level + 1
                self.agent.current_research = None
                self.agent.research_progress = 0

                # Update technology bonuses
                self.agent.update_tech_bonuses()
