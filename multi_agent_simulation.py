import numpy as np
import torch
import os
from tech_tree import TechTree
from simulation_config import config
from advanced_evolution import AdvancedEvolution, ComplexResourceManager, CulturalInfluence
from random_events import RandomEventManager

class CivilizationAgent:
    def __init__(self, agent_id, initial_strength=None, resources=None):
        self.agent_id = agent_id
        self.strength = initial_strength if initial_strength is not None else config.INITIAL_STRENGTH  # Military strength
        self.defense = self.strength   # Defense capability (initially same as military strength)
        self.resources = resources if resources is not None else config.INITIAL_RESOURCE_AMOUNT        # Resource reserves
        self.territory = set()            # Controlled hexagonal grid IDs
        self.allies = set()               # Ally IDs set
        self.enemies = set()              # Enemy civilization IDs set
        self.technology = {"agriculture": 1, "military": 1, "trade": 1, "science": 1}  # Technology levels
        self.research_progress = 0        # Current technology research progress
        self.current_research = None      # Currently researched technology
        self.research_cost = 0            # Current research cost
        self.tech_bonuses = {}            # Technology bonuses
        self.population = getattr(config, 'INITIAL_POPULATION', 100)             # Population size
        self.infrastructure = 1.0         # Infrastructure level
        self.stability = 1.0              # Social stability
        self.health = 1.0                 # Health level
        self.energy_efficiency = 1.0      # Energy efficiency
        self.resource_acquisition = 1.0   # Resource acquisition ability
        self.global_influence = 1.0       # Global influence
        self.decision_quality = 1.0       # Decision quality
        self.tech_tree = TechTree()
        self.update_tech_bonuses()
        self.trade_history = []           # Trade history records
        self.conflict_history = []        # Conflict history records
        self.tech_spillover_received = 0.0  # Received technology spillover amount
        self.last_strategy = None         # Strategy used in the previous round
        self.culture = None               # Cultural traits (to be initialized by CulturalInfluence)
        self.cultural_bonuses = {}        # Cultural bonuses
        self.religious_influence = 1.0    # 宗教影响力
        self.religious_followers = 0      # 宗教追随者数量
        
    def update_tech_bonuses(self):
        """Update technology bonuses"""
        self.tech_bonuses = self.tech_tree.calculate_tech_bonuses(self.technology)
        
        # Apply attribute bonuses
        self.infrastructure = self.tech_bonuses.get("infrastructure", 1.0)
        self.stability = self.tech_bonuses.get("stability", 1.0)
        self.health = self.tech_bonuses.get("health", 1.0)
        self.energy_efficiency = self.tech_bonuses.get("energy_efficiency", 1.0)
        self.resource_acquisition = self.tech_bonuses.get("resource_acquisition", 1.0)
        self.global_influence = self.tech_bonuses.get("global_influence", 1.0)
        self.decision_quality = self.tech_bonuses.get("decision_quality", 1.0)
    
    def decide_strategy(self, neighbors, global_resources, advanced_evolution=None):
        """
        Decide strategy based on neighbor status and global resource distribution
        neighbors: Dictionary {agent_id: (strength, relationship)}
        global_resources: Resource abundance map (H3 grid)
        advanced_evolution: Advanced evolution engine instance
        """
        # If advanced evolution engine is provided, use it to calculate strategy tendency
        if advanced_evolution is not None:
            # Convert neighbors to the format required by the advanced evolution engine
            neighbor_agents = {}
            for agent_id, (_, rel) in neighbors.items():
                # Assume that the corresponding agent instance can be found by agent_id
                # In actual calls, ensure that the MultiAgentSimulation class correctly passes neighbor agent objects
                pass
            
            # In the MultiAgentSimulation class, this method will be overridden to ensure correct passing of neighbor agents
            # Return default strategy here, actual strategy calculation is handled in MultiAgentSimulation.step()
            # 修改为7种策略（增加宗教策略）
            return np.ones(7) / 7
        
        # Fall back to traditional strategy decision logic
        # Strategy tendency (expansion/defense/trade/research/diplomacy/culture/religion)
        strategy = np.zeros(7)  # 修改为7种策略类型
        
        # 1. Analyze neighbor threats
        enemy_strength = sum(
            n_strength for n_id, (n_strength, rel) in neighbors.items() 
            if rel == "enemy"
        )
        ally_strength = sum(
            n_strength for n_id, (n_strength, rel) in neighbors.items()
            if rel == "ally"
        )
        
        # 2. Resource pressure assessment
        controlled_resources = sum(global_resources[h3_id] for h3_id in self.territory)
        resource_pressure = 1.0 - (self.resources / (controlled_resources * 10))
        
        # 3. Calculate effective attributes after various bonuses
        effective_strength = self.strength * self.tech_bonuses.get("strength", 1.0)
        effective_defense = self.defense * self.tech_bonuses.get("defense", 1.0)
        effective_research_speed = self.tech_bonuses.get("research_speed", 1.0)
        effective_population = self.population * self.health * self.tech_bonuses.get("population_growth", 1.0)
        
        # 4. Dynamically adjust strategy weights
        # Base weights based on population and infrastructure
        base_weights = np.array([
            0.2 * self.infrastructure,  # 扩张基础权重
            0.15 * (1.0 / self.stability) if self.stability > 0 else 0.15,  # 防御基础权重
            0.25 * self.global_influence,  # 贸易基础权重
            0.4 * effective_research_speed,  # 研发基础权重
            0.2 * self.global_influence,  # 外交基础权重
            0.15 * (self.cultural_influence if hasattr(self, 'cultural_influence') else 0.15),  # 文化基础权重
            0.1 * self.religious_influence  # 宗教基础权重
        ])
        
        # Threat response adjustment
        if enemy_strength > effective_strength * 1.2:
            # Defense priority
            threat_factor = min(enemy_strength / effective_strength, 2.0)
            strategy[1] = 0.6 + min(resource_pressure, 0.3) * threat_factor
        
        # Resource pressure adjustment
        if resource_pressure > 0.8:
            # High resource pressure
            if len(self.allies) > 0:
                # Has allies, prioritize trade
                strategy[2] = 0.5 + (1.0 - resource_pressure) * 0.4 * self.tech_bonuses.get("trade_efficiency", 1.0)
            else:
                # No allies, prioritize expansion to acquire resources
                strategy[0] = 0.5 + resource_pressure * 0.3 * self.resource_acquisition
        
        # Research condition adjustment
        available_techs = self.tech_tree.get_available_techs(self.technology)
        if self.resources > 1000 and available_techs:
            # Has resources and available technologies to research
            # Adjust research priority based on technology type
            has_high_level_tech = any(tech['level'] >= 3 for tech in available_techs)
            if has_high_level_tech:
                # Prioritize researching advanced technologies
                strategy[3] = 0.6 + (1.0 - resource_pressure) * 0.3 * effective_research_speed
        
        # Diplomacy adjustment - more allies when there are neighbors
        if len(neighbors) > 0:
            strategy[4] = 0.3 + 0.2 * len(self.allies) / len(neighbors)
        
        # Cultural influence adjustment
        if hasattr(self, 'cultural_influence'):
            strategy[5] = 0.2 * self.cultural_influence
            
        # Religious influence adjustment
        strategy[6] = 0.1 * self.religious_influence
        
        # Population and health factors
        if effective_population > 500 and self.health > 1.5:
            # Sufficient population and good health, suitable for expansion
            strategy[0] += 0.2
        
        # Special handling for top-level technologies
        if any(tech in self.technology for tech in ["artificial_intelligence", "nuclear_technology"]):
            # Having top-level technology, increase research and defense priority
            strategy[3] += 0.2 * self.decision_quality
            strategy[1] += 0.1 * self.decision_quality
        
        # Combine base weights and dynamic adjustments
        strategy = strategy + base_weights
        
        # Normalize to ensure sum is 1
        if np.sum(strategy) > 0:
            return strategy / np.sum(strategy)
        else:
            # If all strategy weights are 0, return uniform distribution
            # 修改为7种策略的均匀分布
            return np.ones(7) / 7
    
    def apply_cultural_bonuses(self):
        """Apply cultural bonuses to civilization attributes"""
        if hasattr(self, 'cultural_bonuses') and self.cultural_bonuses:
            # Apply cultural bonuses to various attributes
            self.strength *= (1 + self.cultural_bonuses.get('military_strength', 0))
            self.population *= (1 + self.cultural_bonuses.get('population_growth', 0))
            # Note: This only modifies temporary attribute values, not the base values directly
            # In actual simulations, the application method of bonuses may need to be adjusted according to specific situations
        
    def _should_research(self):
        """Determine whether technology research should be conducted"""
        available_techs = self.tech_tree.get_available_techs(self.technology)
        return len(available_techs) > 0
    
    def execute_strategy(self, strategy, neighbors, global_resources):
        """Execute strategy and update status"""
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
            diplomacy_prob, culture_prob, religion_prob = 0, 0, 0
        else:
            expansion_prob, defense_prob, trade_prob = strategy
            research_prob, diplomacy_prob, culture_prob, religion_prob = 0, 0, 0, 0
        
        # Expansion logic
        if np.random.rand() < expansion_prob and len(self.territory) < 50:
            # Select the most resource-rich uncontrolled area
            available_h3 = [h3_id for h3_id in global_resources 
                          if h3_id not in self.territory]
            if available_h3:
                target = max(available_h3, key=lambda x: global_resources[x])
                # Apply territory growth bonus
                territory_bonus = self.tech_bonuses.get("territory_growth", 1.0)
                self.territory.add(target)
                # Apply territory value bonus
                territory_value_bonus = self.tech_bonuses.get("territory_value", 1.0)
                self.resources += global_resources[target] * 0.3 * territory_value_bonus
        
        # Defense logic
        if np.random.rand() < defense_prob:
            # Apply military bonus
            defense_bonus = self.tech_bonuses.get("defense", 1.0)
            self.strength *= 1.05 * defense_bonus  # 军事强化
            self.resources *= 0.95  # 消耗资源
        
        # Trade logic
        if np.random.rand() < trade_prob and len(self.allies) > 0:
            # 随机选择盟友进行资源交换
            ally_id = np.random.choice(list(self.allies)) if self.allies else None
            if ally_id:
                # 应用贸易效率加成
                trade_efficiency_bonus = self.tech_bonuses.get("trade_efficiency", 1.0)
                trade_amount = int(self.resources * 0.1)
                self.resources += trade_amount * 0.1 * trade_efficiency_bonus  # 贸易收益
                self.resources -= trade_amount
        
        # 研发逻辑
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
        
        # 定期获取资源产出
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
                        temp_tech = {tech_name: current_level + 1}
                        bonuses = self.tech_tree.calculate_tech_bonuses(temp_tech)
                        
                        priority = 0
                        
                        # Calculate effective attribute values (considering current state needs)
                        effective_research_speed = 1.0 + self.tech_bonuses.get("research_speed", 0)
                        effective_strength = 1.0 + self.tech_bonuses.get("strength", 0)
                        effective_defense = 1.0 + self.tech_bonuses.get("defense", 0)
                        
                        # Special handling for top-level technologies
                        if tech_level == 4:
                            priority += 10  # 顶级科技有额外优先级
                        
                        # Dynamically adjust weights based on current state
                        if effective_research_speed < effective_strength and effective_research_speed < effective_defense:
                            if "research_speed" in bonuses:
                                priority += bonuses["research_speed"] * 2.0  # Higher priority when research is lagging
                        elif effective_strength < effective_defense and self.enemies:
                            if "strength" in bonuses:
                                priority += bonuses["strength"] * 1.5  # Prioritize military when there are enemies and strength is weak
                        
                        # Basic attribute bonuses
                        if "resources" in bonuses:
                            priority += bonuses["resources"] * 1.0
                        if "territory_value" in bonuses:
                            priority += bonuses["territory_value"] * 0.8
                        if "population_growth" in bonuses:
                            priority += bonuses["population_growth"] * 0.7
                        
                        # Cost-benefit analysis
                        priority /= (cost / 100)  # Divide by normalized cost
                        
                        return priority
                    
                    affordable_techs.sort(key=tech_priority, reverse=True)
                    self.current_research, self.research_cost, _ = affordable_techs[0]
                    if hasattr(config, 'PRINT_LOGS') and config.PRINT_LOGS:
                        print(f"文明{self.agent_id}开始研发: {self.current_research} (成本: {self.research_cost})")
        
        # 进行研发
        if self.current_research is not None:
            # 计算研发速度（考虑人口、基础设施和科研投入）
            research_base_speed = self.tech_bonuses.get("research_speed", 1.0) * getattr(config, 'BASE_RESEARCH_SPEED', 1.0)
            population_bonus = min(self.population / 100, getattr(config, 'POPULATION_RESEARCH_BONUS_CAP', 5.0))  # 人口红利上限
            infrastructure_bonus = self.infrastructure / 100  # 基础设施提供额外速度
            
            # 应用科技溢出效应
            tech_spillover_effect = getattr(config, 'TECH_SPILLOVER_EFFECT', 0.0)
            spillover_bonus = self.tech_spillover_received * tech_spillover_effect
            
            # 计算资源效率
            research_resource_efficiency = getattr(config, 'RESEARCH_RESOURCE_EFFICIENCY', 1.0)
            
            total_research_speed = research_base_speed * (1 + population_bonus + infrastructure_bonus + spillover_bonus)
            
            # 投入资源进行研发
            research_resource_ratio = getattr(config, 'RESEARCH_RESOURCE_RATIO', 0.15)
            research_investment = min(self.resources * research_resource_ratio, self.resources)
            self.resources -= research_investment
            
            # 应用资源效率
            effective_investment = research_investment * research_resource_efficiency
            self.research_progress += effective_investment * total_research_speed
            
            # 添加科技溢出带来的额外研发进度
            self.research_progress += self.tech_spillover_received * 0.5  # 科技溢出50%转化为研发进度
            
            # 记录科技溢出的使用情况
            if spillover_bonus > 0 and hasattr(config, 'PRINT_LOGS') and config.PRINT_LOGS:
                pass  # 可以在这里添加科技溢出使用日志
            
            # 检查是否完成研发
            if self.research_progress >= self.research_cost:
                # 完成研发
                current_level = self.technology.get(self.current_research, 0)
                self.technology[self.current_research] = current_level + 1
                if hasattr(config, 'PRINT_LOGS') and config.PRINT_LOGS:
                    print(f"文明{self.agent_id}研发完成: {self.current_research} (等级 {current_level + 1})")
                
                # 更新科技加成
                self.update_tech_bonuses()
                
                # 重置研发状态
                self.current_research = None
                self.research_progress = 0
                self.research_cost = 0
                
                # 不再重置科技溢出接收量，让它在下一周期继续生效
    
    def _collect_resource_output(self, global_resources):
        """Collect territory resource output"""
        # Calculate base output
        base_output = sum(global_resources[h3_id] for h3_id in self.territory) * getattr(config, 'RESOURCE_BASE_OUTPUT', 0.1)
        
        # Apply various bonuses
        resource_bonus = self.tech_bonuses.get("resources", 1.0)
        territory_value_bonus = self.tech_bonuses.get("territory_value", 1.0)
        resource_acquisition_bonus = self.resource_acquisition
        energy_efficiency_bonus = self.energy_efficiency
        
        # Apply territory value coefficient
        territory_value_coefficient = getattr(config, 'TERRITORY_VALUE_COEFFICIENT', 1.0)
        
        # Comprehensive calculation of actual output
        total_bonus = resource_bonus * territory_value_bonus * resource_acquisition_bonus * energy_efficiency_bonus * territory_value_coefficient
        actual_output = base_output * total_bonus
        
        # Add additional resource output from technology spillover
        if hasattr(self, 'tech_spillover_received'):
            actual_output += self.tech_spillover_received * 0.1  # 科技溢出10%转化为资源增益
        
        # Add to resource reserves
        self.resources += actual_output
        
        # Resource regeneration
        if hasattr(config, 'RESOURCE_REGENERATION_RATE'):
            resource_regeneration = sum(global_resources[h3_id] for h3_id in self.territory) * config.RESOURCE_REGENERATION_RATE
            self.resources += resource_regeneration
        
        # Resource consumption (base and population)
        base_consumption = getattr(config, 'RESOURCE_CONSUMPTION_RATE', 0.01) * self.resources
        population_consumption = self.population * getattr(config, 'POPULATION_RESOURCE_CONSUMPTION', 0.05)
        total_consumption = base_consumption + population_consumption
        self.resources = max(0, self.resources - total_consumption)
        
        # 人口增长（受健康和资源影响）
        if self.resources > 0:
            # Calculate available resource per capita
            resource_per_capita = self.resources / self.population if self.population > 0 else 0
            
            # Population growth rate is affected by resource per capita, health status and technology bonuses
            max_growth_rate = getattr(config, 'POPULATION_GROWTH_RATE', 0.02)
            resource_factor = min(resource_per_capita / getattr(config, 'RESOURCE_PER_CAPITA_FOR_MAX_GROWTH', 10), 1.0)
            population_growth_rate = max_growth_rate * resource_factor * self.health * self.tech_bonuses.get("population_growth", 1.0)
            
            # Calculate population growth
            population_increase = self.population * population_growth_rate
            
            # Apply population cap
            population_cap = getattr(config, 'POPULATION_GROWTH_CAP', float('inf'))
            self.population_cap = population_cap
            population_after_growth = self.population + population_increase
            
            # Limit population not to exceed the cap
            if population_after_growth > population_cap:
                population_increase = population_cap - self.population
                self.population = population_cap
            else:
                self.population += population_increase
            
            # Record population changes
            if hasattr(config, 'PRINT_LOGS') and config.PRINT_LOGS and population_increase > 0 and len(self.trade_history) % 100 == 0:
                pass  # 可以在这里添加详细日志
    
    def update_relationships(self, neighbors, threshold=None):
        """Dynamically update ally/enemy relationships"""
        # Use threshold from configuration or default value
        rel_threshold = threshold if threshold is not None else config.RELATIONSHIP_THRESHOLD
        
        for agent_id, (strength, rel) in neighbors.items():
            if rel == "neutral":
                # Dynamically adjust relationships based on resource competition
                shared_interest = len(set(neighbors.keys()) & self.territory)
                if shared_interest > rel_threshold * len(self.territory):
                    self.enemies.add(agent_id)
                    if config.PRINT_LOGS:
                        print(f"文明{self.agent_id}与文明{agent_id}成为敌人")
                elif strength > self.strength * 1.5:
                    self.allies.add(agent_id)
                    if config.PRINT_LOGS:
                        print(f"文明{self.agent_id}与文明{agent_id}成为盟友")

class MultiAgentSimulation:
    def __init__(self, config=None):
        # Use parameters from configuration or passed parameters
        self.config = config if config is not None else __import__('simulation_config').config
        self.num_agents = self.config.NUM_CIVILIZATIONS
        self.grid_size = self.config.GRID_SIZE
        
        # Initialize random seed
        if hasattr(self.config, 'RANDOM_SEED') and self.config.RANDOM_SEED is not None:
            np.random.seed(self.config.RANDOM_SEED)
        
        # Initialize advanced evolution modules
        self.advanced_evolution = AdvancedEvolution(self.config)
        self.resource_manager = ComplexResourceManager(self.config)
        self.cultural_influence = CulturalInfluence(self.config)
        self.random_events = RandomEventManager(self.config)  # 添加随机事件管理器
        
        self.agents = [CivilizationAgent(i) for i in range(self.num_agents)]
        
        # Initialize cultural traits
        for agent in self.agents:
            self.cultural_influence.initialize_culture(agent)
        
        # Generate resources using complex resource management system
        if hasattr(config, 'USE_COMPLEX_RESOURCES') and config.USE_COMPLEX_RESOURCES:
            # Complex resource system uses coordinate positions instead of string IDs
            # To be compatible with existing code, we need to do some conversions
            self.complex_resources = self.resource_manager.generate_resource_map(self.grid_size)
            # Create a simplified resource mapping for compatibility with existing code
            self.global_resources = {}
            self.h3_grid = []
            for pos, resources in self.complex_resources.items():
                h3_id = f"h3_{pos[0]}_{pos[1]}"
                self.h3_grid.append(h3_id)
                # Convert complex resources to a single numerical representation (using total value)
                resource_value = self.resource_manager.calculate_resource_value(resources)
                self.global_resources[h3_id] = resource_value
        else:
            # Use traditional resource generation method
            self.global_resources = self._generate_resources(self.grid_size)
            self.h3_grid = list(self.global_resources.keys())
        
        self.initialize_territory()
        
        # Save historical data for visualization
        self.history = {"strategy": [], "resources": [], "strength": [], "technology": []}
        
        # Added historical data collection structures
        self.attribute_history = []
        self.technology_history = {agent.agent_id: [] for agent in self.agents}
        self.relationship_history = []
        
        # Define attributes to track
        self.attribute_names = ['resources', 'strength', 'population', 'territory', 
                              'tech_level', 'global_influence', 'stability', 'health']
        
        # Initialize relationship network
        self.initialize_relationships()
        
        # Save advanced evolution related historical data
        self.strategy_tendency_history = []
        self.cultural_history = []
        self.event_history = []  # 添加事件历史记录
    
    def _generate_resources(self, grid_size):
        """Generate H3 grid resource distribution (simplified to random IDs)"""
        # 实际应用中应使用真实H3索引
        return {f"h3_{i}": np.random.rand() * 100 for i in range(grid_size)}
        
    def _get_agent_by_id(self, agent_id):
        """Get agent instance by ID"""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def initialize_territory(self):
        """Initialize civilization territory distribution"""
        territory_per_agent = len(self.h3_grid) // len(self.agents)
        
        # If configuration allows random initial distribution
        if hasattr(self.config, 'RANDOM_TERRITORY_DISTRIBUTION') and self.config.RANDOM_TERRITORY_DISTRIBUTION:
            # Randomly shuffle grid
            shuffled_grid = self.h3_grid.copy()
            np.random.shuffle(shuffled_grid)
            
            # Allocate territory
            for i, agent in enumerate(self.agents):
                start_idx = i * territory_per_agent
                end_idx = start_idx + territory_per_agent
                agent.territory = set(shuffled_grid[start_idx:end_idx])
        else:
            # Default uniform distribution
            for i, agent in enumerate(self.agents):
                start_idx = i * territory_per_agent
                end_idx = start_idx + territory_per_agent
                agent.territory = set(self.h3_grid[start_idx:end_idx])
        
    def initialize_relationships(self):
        """Initialize relationships between civilizations"""
        # Can initialize specific relationship patterns based on configuration here
        pass
    
    def step(self, cycle=None):
        """Execute one cycle of multi-agent interaction"""
        # 1. Collect neighbor relationships
        agent_neighbors = {
            agent.agent_id: self._get_neighbors(agent) 
            for agent in self.agents
        }
        
        # 2. 触发随机事件
        event = self.random_events.trigger_event(self.agents, cycle)
        if event:
            self.event_history.append(event)
        
        # 3. Apply technology spillover effect
        if hasattr(self.config, 'TECH_SPILLOVER_EFFECT') and self.config.TECH_SPILLOVER_EFFECT > 0:
            self._apply_tech_spillover()
        
        # 4. Update cultural influence
        for agent in self.agents:
            # Get neighbor agent objects
            neighbor_agents = {}
            for neighbor_id, (_, rel) in agent_neighbors[agent.agent_id].items():
                neighbor_agent = self._get_agent_by_id(neighbor_id)
                if neighbor_agent:
                    # Convert relationship strength to numerical representation
                    rel_value = 0.0
                    if rel == "ally":
                        rel_value = 0.8
                    elif rel == "enemy":
                        rel_value = -0.8
                    else:  # neutral
                        rel_value = 0.0
                    neighbor_agents[neighbor_agent] = rel_value
            
            # Update cultural influence
            self.cultural_influence.update_cultural_influence(agent, neighbor_agents)
            
            # Calculate and apply cultural bonuses
            agent.cultural_bonuses = self.cultural_influence.get_cultural_bonuses(agent)
        
        # 5. Each civilization makes independent decisions
        strategies = {}
        current_strategy_tendencies = []
        
        for agent in self.agents:
            neighbors = agent_neighbors[agent.agent_id]
            
            # If advanced evolution is enabled, use it to calculate strategy tendency
            if hasattr(self.config, 'USE_ADVANCED_EVOLUTION') and self.config.USE_ADVANCED_EVOLUTION:
                # Get neighbor agent objects and relationships
                neighbor_agents = {}
                for neighbor_id, (_, rel) in neighbors.items():
                    neighbor_agent = self._get_agent_by_id(neighbor_id)
                    if neighbor_agent:
                        # Convert relationships to numerical values
                        rel_value = 0.0
                        if rel == "ally":
                            rel_value = 0.8
                        elif rel == "enemy":
                            rel_value = -0.8
                        else:  # neutral
                            rel_value = 0.0
                        neighbor_agents[neighbor_agent] = rel_value
                
                # Use advanced evolution engine to calculate strategy tendency
                strategy_tendency = self.advanced_evolution.calculate_strategy_tendency(
                    agent, neighbor_agents, self.global_resources
                )
                current_strategy_tendencies.append(strategy_tendency)
                
                # Convert strategy tendency to array format
                strategy_array = np.array([
                    strategy_tendency.get('expansion', 0.0),
                    strategy_tendency.get('defense', 0.0),
                    strategy_tendency.get('trade', 0.0),
                    strategy_tendency.get('research', 0.0),
                    strategy_tendency.get('diplomacy', 0.0),
                    strategy_tendency.get('culture', 0.0)
                ])
                
                # Record the last used strategy
                max_strategy_idx = np.argmax(strategy_array)
                strategy_names = ['expansion', 'defense', 'trade', 'research', 'diplomacy', 'culture']
                agent.last_strategy = strategy_names[max_strategy_idx]
                
                strategies[agent.agent_id] = strategy_array
            else:
                # Use traditional strategy decision
                strategies[agent.agent_id] = agent.decide_strategy(neighbors, self.global_resources)
        
        # Save strategy tendency history
        if current_strategy_tendencies:
            self.strategy_tendency_history.append(current_strategy_tendencies)
        
        # 6. Execute strategies and update relationships
        cycle_attributes = []
        for agent in self.agents:
            neighbors = agent_neighbors[agent.agent_id]
            
            # 应用文化加成
            agent.apply_cultural_bonuses()
            
            agent.execute_strategy(strategies[agent.agent_id], neighbors, self.global_resources)
            agent.update_relationships(neighbors)
            
            # Collect attribute data
            attributes = [
                agent.resources, agent.strength, agent.population,
                len(agent.territory), sum(agent.technology.values()),
                agent.global_influence, agent.stability, agent.health
            ]
            cycle_attributes.append(attributes)
            
            # Collect technology data
            tech_data = {
                'cycle': len(self.history["resources"]) if cycle is None else cycle,
                'technologies': agent.technology.copy(),
                'current_research': agent.current_research,
                'research_progress': agent.research_progress,
                'research_cost': agent.research_cost,
                'tech_bonuses': agent.tech_bonuses.copy()
            }
            self.technology_history[agent.agent_id].append(tech_data)
        
        # 7. Global resource regeneration
        if hasattr(self.config, 'USE_COMPLEX_RESOURCES') and self.config.USE_COMPLEX_RESOURCES:
            # Use complex resource management system for resource regeneration
            for pos in self.complex_resources:
                self.complex_resources[pos] = self.resource_manager.regenerate_resources(
                    self.complex_resources[pos], pos, len(self.history['resources']) if cycle is None else cycle
                )
            
            # Update simplified resource mapping
            for pos in self.complex_resources:
                h3_id = f"h3_{pos[0]}_{pos[1]}"
                if h3_id in self.global_resources:
                    resource_value = self.resource_manager.calculate_resource_value(self.complex_resources[pos])
                    self.global_resources[h3_id] = resource_value
        else:
            # Use traditional resource regeneration method
            self._regenerate_global_resources()
            
        # 8. Save cultural history data
        cultural_data = {}
        for agent in self.agents:
            if hasattr(agent, 'culture'):
                cultural_data[agent.agent_id] = agent.culture.copy()
        self.cultural_history.append(cultural_data)
        
        # 9. Save historical data
        self._save_history()
        self.attribute_history.append(np.mean(cycle_attributes, axis=0))
        
        # 10. Record relationship history
        relationship_data = {}
        for agent in self.agents:
            relationship_data[agent.agent_id] = {
                'allies': agent.allies.copy(),
                'enemies': agent.enemies.copy()
            }
        self.relationship_history.append(relationship_data)
        
        # 11. Check if termination condition is reached
        if cycle is not None and cycle % 100 == 0 and hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS:
            self._print_status(cycle)
            
        return strategies
        
    def _apply_tech_spillover(self):
        """Apply technology spillover effect"""
        # Check if technology spillover is enabled
        if not hasattr(self.config, 'TECH_SPILLOVER_EFFECT') or self.config.TECH_SPILLOVER_EFFECT <= 0:
            return
        
        # Calculate technology level of each civilization
        tech_levels = {agent.agent_id: sum(agent.technology.values()) for agent in self.agents}
        avg_tech_level = sum(tech_levels.values()) / len(tech_levels)
        
        # Calculate technology spillover received from neighbors for each agent
        spillover_amounts = []
        
        # Apply technology spillover to each civilization
        for agent in self.agents:
            # Spillover amount is based on technology gap and geographical proximity between civilizations
            neighbor_tech_bonus = 0
            neighbors = self._get_neighbors(agent)
            
            for neighbor_id, (_, rel) in neighbors.items():
                neighbor_agent = next(a for a in self.agents if a.agent_id == neighbor_id)
                neighbor_total_tech = sum(neighbor_agent.technology.values())
                agent_total_tech = sum(agent.technology.values())
                
                # If neighbor's technology is more advanced, generate spillover
                if neighbor_total_tech > agent_total_tech:
                    tech_gap = neighbor_total_tech - agent_total_tech
                    
                    # Adjust spillover coefficient based on relationship
                    if rel == "ally":
                        spillover_coefficient = 0.05  # Higher spillover between allies
                    elif rel == "neutral":
                        spillover_coefficient = 0.02  # Normal spillover for neutral relationships
                    else:  # enemy
                        spillover_coefficient = 0.01  # Lower spillover for enemy relationships
                    
                    neighbor_tech_bonus += tech_gap * spillover_coefficient * self.config.TECH_SPILLOVER_EFFECT
            
            # Base spillover from global technology level
            global_spillover = max(0, avg_tech_level - sum(agent.technology.values())) * 0.01 * self.config.TECH_SPILLOVER_EFFECT
            
            # 10% of technology spillover converted to resource gain
            total_spillover = neighbor_tech_bonus + global_spillover
            spillover_amounts.append(total_spillover)
            
            # Apply spillover
            agent.tech_spillover_received = total_spillover
            
            # Record technology spillover reception
            if hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS and total_spillover > 0:
                pass  # 可以在这里添加科技溢出接收日志
        
        # 记录科技溢出历史
        if hasattr(self, 'tech_spillover_history'):
            self.tech_spillover_history.append(spillover_amounts)
        else:
            # 如果没有tech_spillover_history属性，创建它
            self.tech_spillover_history = [spillover_amounts]
            
    def _regenerate_global_resources(self):
        """Global resource regeneration"""
        if hasattr(self.config, 'GLOBAL_RESOURCE_REGENERATION') and self.config.GLOBAL_RESOURCE_REGENERATION:
            for h3_id in self.global_resources:
                # Basic resource regeneration
                regen_rate = getattr(self.config, 'RESOURCE_REGENERATION_RATE', 0.01)
                max_resource = getattr(self.config, 'MAX_RESOURCE_PER_CELL', 100.0)
                
                # Resource regeneration, not exceeding maximum value
                self.global_resources[h3_id] = min(max_resource, 
                                                  self.global_resources[h3_id] * (1 + regen_rate))
        
    def _save_history(self):
        """Save historical data for visualization"""
        # Save strategy distribution
        strategy_dist = {agent.agent_id: agent.decide_strategy(self._get_neighbors(agent), self.global_resources) 
                        for agent in self.agents}
        self.history["strategy"].append(strategy_dist)
        
        # Save resources and strength
        self.history["resources"].append([agent.resources for agent in self.agents])
        self.history["strength"].append([agent.strength for agent in self.agents])
        
        # Save technology level (total technology of each civilization)
        tech_levels = []
        for agent in self.agents:
            total_tech = sum(agent.technology.values())
            tech_levels.append(total_tech)
        self.history["technology"].append(tech_levels)
        
        # Save population data
        self.history.setdefault("population", []).append([agent.population for agent in self.agents])
        
        # Save territory data
        self.history.setdefault("territory", []).append([len(agent.territory) for agent in self.agents])
        
        # Save technology spillover data
        if hasattr(self, 'tech_spillover_history') and self.tech_spillover_history:
            self.history.setdefault("tech_spillover", []).append(self.tech_spillover_history[-1])
        
        # Print cycle logs
        if hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS:
            log_interval = getattr(self.config, 'LOG_INTERVAL', 100)
            if len(self.history["resources"]) % log_interval == 0:
                cycle = len(self.history["resources"]) - 1
                print(f"\n周期{cycle + 1}统计:")
                print(f"资源分布: {[round(res, 2) for res in self.history['resources'][-1]]}")
                print(f"力量分布: {[round(stren, 2) for stren in self.history['strength'][-1]]}")
                print(f"科技总览: {[round(tech, 2) for tech in self.history['technology'][-1]]}")
                if "population" in self.history:
                    print(f"人口分布: {[round(pop, 2) for pop in self.history['population'][-1]]}")
                if "tech_spillover" in self.history and self.history['tech_spillover']:
                    print(f"科技溢出接收: {[round(spill, 4) for spill in self.history['tech_spillover'][-1]]}")
    
    def _print_status(self, cycle):
        """Print current simulation status"""
        if hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS:
            print(f"\n==== 周期 {cycle} 状态报告 ====")
            avg_resources = np.mean([agent.resources for agent in self.agents])
            avg_strength = np.mean([agent.strength for agent in self.agents])
            avg_tech_level = np.mean([sum(agent.technology.values()) for agent in self.agents])
            avg_territory = np.mean([len(agent.territory) for agent in self.agents])
            avg_population = np.mean([agent.population for agent in self.agents])
            
            print(f"平均资源: {round(avg_resources, 2)}")
            print(f"平均力量: {round(avg_strength, 2)}")
            print(f"平均科技等级: {round(avg_tech_level, 2)}")
            print(f"平均领土大小: {round(avg_territory, 2)}")
            print(f"平均人口: {round(avg_population, 2)}")
            
            # Print technology research status
            research_in_progress = sum(1 for agent in self.agents if agent.current_research is not None)
            print(f"正在进行的研发项目数: {research_in_progress}")
    
    def _get_neighbors(self, agent):
        """Get neighboring civilizations and relationships"""
        neighbors = {}
        for other in self.agents:
            if agent.agent_id == other.agent_id:
                continue
            
            # Simplified neighbor detection (should actually be based on H3 grid distance)
            shared_border = len(agent.territory & other.territory) > 0
            if shared_border:
                rel = "ally" if other.agent_id in agent.allies else \
                      "enemy" if other.agent_id in agent.enemies else "neutral"
                neighbors[other.agent_id] = (other.strength, rel)
        
        return neighbors

    # Run complete simulation
    def run(self, num_cycles=None):
        """Run complete multi-agent simulation"""
        simulation_cycles = num_cycles if num_cycles is not None else getattr(self.config, 'SIMULATION_CYCLES', 1000)
        
        # Fast mode detection
        fast_mode = getattr(self.config, 'FAST_MODE', False)
        
        for cycle in range(simulation_cycles):
            # Reduce log output in fast mode
            if not fast_mode or (fast_mode and cycle % 100 == 0):
                if hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS and cycle % 100 == 0:
                    print(f"运行周期 {cycle}/{simulation_cycles}")
            
            self.step(cycle)
        
        # Save simulation results
        output_dir = getattr(self.config, 'DEFAULT_OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "多智能体演化历史.npz")
        
        try:
            np.savez(output_file, 
                     strategies=np.array(self.history["strategy"], dtype=object),
                     resources=np.array(self.history["resources"]),
                     strength=np.array(self.history["strength"]),
                     technology=np.array(self.history["technology"]),
                     attribute_history=np.array(self.attribute_history),
                     technology_history=np.array(self.technology_history, dtype=object),
                     relationship_history=np.array(self.relationship_history, dtype=object),
                     event_history=np.array(self.event_history, dtype=object))  # 添加事件历史数据
            
            if hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS:
                print("\n==== 模拟完成 ====")
                print(f"共运行{simulation_cycles}个周期")
                print(f"多智能体模拟结果保存至：{output_file}")
                if self.event_history:
                    print(f"共发生{len(self.event_history)}次随机事件")
        except Exception as e:
            if hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS:
                print(f"\n保存模拟结果失败: {e}")
            
        return np.array(self.history["strategy"])

# 添加一个新方法用于导出更详细的JSON格式数据
    def export_detailed_json(self, filename=" detailed_civilization_history.json"):
        """导出详细的文明演化历史数据为JSON格式"""
        import json
        
        detailed_data = {
            "simulation_config": {
                "num_civilizations": self.num_agents,
                "grid_size": self.grid_size,
                "simulation_cycles": len(self.history["strategy"])
            },
            "civilizations": {}
        }
        
        # 为每个文明添加详细数据
        for agent in self.agents:
            detailed_data["civilizations"][agent.agent_id] = {
                "technology_development": self.technology_history[agent.agent_id],
                "relationships": {
                    "allies": list(agent.allies),
                    "enemies": list(agent.enemies)
                },
                "final_state": {
                    "strength": agent.strength,
                    "resources": agent.resources,
                    "population": agent.population,
                    "territory_size": len(agent.territory),
                    "technology_levels": agent.technology,
                    "infrastructure": agent.infrastructure,
                    "stability": agent.stability,
                    "health": agent.health
                }
            }
        
        # 添加整体历史数据
        detailed_data["overall_history"] = {
            "strategies": self.history["strategy"],
            "resources": self.history["resources"],
            "strength": self.history["strength"],
            "technology": self.history["technology"]
        }
        
        output_dir = getattr(self.config, 'DEFAULT_OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        if hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS:
            print(f"详细JSON数据已保存至: {output_file}")
            
    def _spread_religion(self, neighbors):
        """传播宗教影响力"""
        # 增加宗教影响力
        religious_growth = 0.03 * self.tech_bonuses.get("cultural_efficiency", 1.0)
        self.religious_influence = min(2.0, self.religious_influence + religious_growth)  # 宗教影响力上限为2.0
        
        # 尝试转化邻居文明的民众
        for neighbor_id, (_, relationship) in neighbors.items():
            if np.random.rand() < 0.1:  # 10%几率影响邻居
                self.religious_followers += 1
                
        # 宗教传播消耗少量资源
        self.resources *= 0.99

if __name__ == "__main__":
    from simulation_config import config
    from civilization_visualizer import CivilizationVisualizer
    
    # 初始化多智能体模拟
    sim = MultiAgentSimulation(config)
    
    # 运行模拟
    history = sim.run(500)
    
    # 导出详细JSON数据
    sim.export_detailed_json()
    
    # 可视化结果
    if len(history) > 0:
        visualizer = CivilizationVisualizer()
        
        # 生成所有可视化图表
        visualizer.plot_strategy_heatmap(history)
        visualizer.plot_evolution_curve(history)
        visualizer.plot_technology_progress(sim.technology_history)
        visualizer.plot_tech_tree_comparison(sim.technology_history)
        visualizer.plot_attribute_comparison(sim.agents)
        visualizer.plot_radar_chart(sim.agents)
        visualizer.plot_relationships_network(sim.agents)
        
        # 保存所有结果和报告
        if hasattr(visualizer, 'save_to_csv'):
            visualizer.save_to_csv(np.array(sim.history["strategy"]))
        if hasattr(visualizer, 'save_attribute_history'):
            visualizer.save_attribute_history(np.array(sim.attribute_history), sim.attribute_names)
        if hasattr(visualizer, 'save_technology_data'):
            visualizer.save_technology_data(sim.technology_history)
        
        # Display charts
        visualizer.show()            visualizer.save_attribute_history(np.array(sim.attribute_history), sim.attribute_names)
        if hasattr(visualizer, 'save_technology_data'):
            visualizer.save_technology_data(sim.technology_history)
        
        # Display charts
        visualizer.show()    if len(history) > 0:
        visualizer = CivilizationVisualizer()
        
        # 生成所有可视化图表
        visualizer.plot_strategy_heatmap(history)
        visualizer.plot_evolution_curve(history)
        visualizer.plot_technology_progress(sim.technology_history)
        visualizer.plot_tech_tree_comparison(sim.technology_history)
        visualizer.plot_attribute_comparison(sim.agents)
        visualizer.plot_radar_chart(sim.agents)
        visualizer.plot_relationships_network(sim.agents)
        
        # 保存所有结果和报告
        if hasattr(visualizer, 'save_to_csv'):
            visualizer.save_to_csv(np.array(sim.history["strategy"]))
        if hasattr(visualizer, 'save_attribute_history'):
            visualizer.save_attribute_history(np.array(sim.attribute_history), sim.attribute_names)
        if hasattr(visualizer, 'save_technology_data'):
            visualizer.save_technology_data(sim.technology_history)
        
        # Display charts
        visualizer.show()            visualizer.save_attribute_history(np.array(sim.attribute_history), sim.attribute_names)
        if hasattr(visualizer, 'save_technology_data'):
            visualizer.save_technology_data(sim.technology_history)
        
        # Display charts
        visualizer.show()