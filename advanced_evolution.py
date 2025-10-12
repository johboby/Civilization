#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统 - 高级演化模块

此模块实现了更复杂和科学的推理与演化功能，基于博弈论、复杂系统理论和现代演化算法
为CivilizationAgent提供更智能的决策能力和更丰富的演化路径
"""
import numpy as np
import copy
from collections import defaultdict, deque
import random
from math import exp, log, sqrt


class AdvancedEvolution:
    """高级演化引擎，为文明提供复杂的决策和演化能力"""
    
    def __init__(self, config=None):
        """初始化高级演化引擎
        
        Args:
            config: 模拟配置对象，包含各种参数设置
        """
        self.config = config
        # 用于存储策略评估历史
        self.strategy_history = defaultdict(deque)
        # 记忆窗口大小
        self.memory_window = config.MEMORY_WINDOW_SIZE if config and hasattr(config, 'MEMORY_WINDOW_SIZE') else 10
        # 探索率
        self.exploration_rate = config.STRATEGY_EXPLORATION_PROBABILITY if config and hasattr(config, 'STRATEGY_EXPLORATION_PROBABILITY') else 0.05
        # 元认知学习率
        self.metacognition_rate = config.METACOGNITION_LEARNING_RATE if config and hasattr(config, 'METACOGNITION_LEARNING_RATE') else 0.05
        # 启用元认知
        self.enable_metacognition = config.ENABLE_METACOGNITION if config and hasattr(config, 'ENABLE_METACOGNITION') else True
        
    def calculate_strategy_tendency(self, agent, neighbors, global_resources):
        """基于复杂系统理论和博弈论计算策略倾向
        
        Args:
            agent: 当前文明智能体
            neighbors: 邻居文明列表
            global_resources: 全局资源分布
            
        Returns:
            dict: 各策略的倾向分数
        """
        # 基础策略倾向
        strategy_scores = {
            'expansion': 0.0,
            'defense': 0.0,
            'trade': 0.0,
            'research': 0.0
        }
        
        # 1. 资源压力评估 (Resource Pressure Assessment)
        resource_pressure = self._assess_resource_pressure(agent, global_resources)
        
        # 2. 安全风险评估 (Security Risk Assessment)
        security_risk = self._assess_security_risk(agent, neighbors)
        
        # 3. 发展潜力评估 (Development Potential Assessment)
        development_potential = self._assess_development_potential(agent, neighbors)
        
        # 4. 基于博弈论的策略选择 (Game Theory-based Strategy Selection)
        game_theory_influence = self._calculate_game_theory_influence(agent, neighbors)
        
        # 5. 历史经验学习 (Historical Experience Learning)
        historical_influence = self._learn_from_history(agent)
        
        # 综合各种因素计算最终策略倾向
        strategy_scores['expansion'] += (resource_pressure * 0.3 + (1 - security_risk) * 0.2 + 
                                        game_theory_influence.get('expansion', 0) * 0.2 + 
                                        historical_influence.get('expansion', 0) * 0.3)
        
        strategy_scores['defense'] += (security_risk * 0.4 + resource_pressure * 0.1 + 
                                      game_theory_influence.get('defense', 0) * 0.2 + 
                                      historical_influence.get('defense', 0) * 0.3)
        
        strategy_scores['trade'] += ((1 - resource_pressure) * 0.3 + development_potential * 0.2 + 
                                   game_theory_influence.get('trade', 0) * 0.25 + 
                                   historical_influence.get('trade', 0) * 0.25)
        
        strategy_scores['research'] += (development_potential * 0.4 + (1 - resource_pressure) * 0.2 + 
                                      game_theory_influence.get('research', 0) * 0.2 + 
                                      historical_influence.get('research', 0) * 0.2)
        
        # 应用元认知调整
        if self.enable_metacognition:
            strategy_scores = self._apply_metacognition(agent, strategy_scores)
        
        # 添加随机探索
        if random.random() < self.exploration_rate:
            # 随机选择一个策略增加其分数，促进探索
            random_strategy = random.choice(list(strategy_scores.keys()))
            strategy_scores[random_strategy] += 0.5
        
        # 归一化策略分数
        total = sum(strategy_scores.values())
        if total > 0:
            for key in strategy_scores:
                strategy_scores[key] /= total
        
        # 记录策略历史
        self._record_strategy_history(agent.agent_id, strategy_scores)
        
        return strategy_scores
    
    def evaluate_tech_research_priority(self, agent):
        """评估科技研发优先级，为文明提供更智能的科技选择建议
            
        Args:
            agent: 文明智能体
                
        Returns:
            dict: 各科技的优先级评分
        """
        # 获取可研发的科技
        available_techs = agent.tech_tree.get_available_techs(agent.technology)
            
        tech_priorities = {}
            
        for tech_info in available_techs:
            tech_name = tech_info["name"]
            tech_level = tech_info["level"]
                
            priority_score = 0.0
                
            # 1. 基于当前文明状态的评估
            # 资源充足度影响
            resource_factor = min(agent.resources / 500, 1.0)  # 假设500是充足资源的标准
                
            # 科技效果评估
            tech_effects = agent.tech_tree.tech_effects.get(tech_name, {})
                
            # 根据科技效果计算优先级
            for attribute, effect_value in tech_effects.items():
                if attribute in ['resources']:
                    priority_score += effect_value * 0.3  # 资源效果权重
                elif attribute in ['strength', 'defense']:
                    priority_score += effect_value * 0.2  # 军事效果权重
                elif attribute in ['research_speed']:
                    priority_score += effect_value * 0.4  # 研发效果权重（复合效应）
                elif attribute in ['population_growth']:
                    priority_score += effect_value * 0.1  # 人口效果权重
                
            # 2. 战略需求评估
            # 如果文明处于扩张阶段，更需要资源和领土相关科技
            if hasattr(agent, 'last_strategy') and agent.last_strategy == 'expansion':
                if any(attr in tech_effects for attr in ['resources', 'territory_growth']):
                    priority_score *= 1.2
                
            # 如果文明处于防御阶段，更需要军事和防御相关科技
            if hasattr(agent, 'last_strategy') and agent.last_strategy == 'defense':
                if any(attr in tech_effects for attr in ['strength', 'defense']):
                    priority_score *= 1.2
                
            # 如果文明处于研发阶段，更需要科研相关科技
            if hasattr(agent, 'last_strategy') and agent.last_strategy == 'research':
                if 'research_speed' in tech_effects:
                    priority_score *= 1.3
                
            # 3. 科技等级调整
            # 高级科技应该有更高的优先级，但需要考虑研发成本
            level_factor = 1.0 + (tech_level - 1) * 0.1  # 每级增加10%优先级
            cost_factor = max(0.5, 1.0 - (tech_info["cost"] / agent.resources))  # 基于资源的可负担性
                
            priority_score *= level_factor * cost_factor
                
            tech_priorities[tech_name] = priority_score
            
        # 归一化优先级分数
        total_priority = sum(tech_priorities.values())
        if total_priority > 0:
            for tech_name in tech_priorities:
                tech_priorities[tech_name] /= total_priority
                    
        return tech_priorities
    
    def _assess_resource_pressure(self, agent, global_resources):
        """评估文明面临的资源压力
        
        考虑因素：当前资源水平、人口需求、领土资源产出、资源分布情况
        
        Returns:
            float: 资源压力分数 (0-1)
        """
        # 当前资源与需求比例
        resource_need_ratio = agent.resources / (agent.population * 0.1)  # 假设每人需要0.1单位资源
        
        # 领土资源产出效率
        territory_resource_efficiency = sum(global_resources.get(pos, 0) for pos in agent.territory) / len(agent.territory) if agent.territory else 0
        
        # 使用现有属性计算资源消耗与产出平衡
        # 1. 基于人口的消耗估计
        estimated_consumption = agent.population * 0.01  # 估计每人消耗0.01单位资源
        # 2. 基于基础设施和资源获取能力的产出估计
        estimated_production = territory_resource_efficiency * agent.resource_acquisition * agent.infrastructure
        
        # 计算消耗与产出的比例
        consumption_production_ratio = estimated_consumption / (estimated_production + 0.001)  # 避免除以零
        
        # 综合计算资源压力
        resource_pressure = 1.0 - exp(-(consumption_production_ratio / (resource_need_ratio + 0.1) * (1 / (territory_resource_efficiency + 0.1))))
        
        # 确保压力值在0-1范围内
        return min(max(resource_pressure, 0.0), 1.0)
    
    def _assess_security_risk(self, agent, neighbors):
        """评估文明面临的安全风险
        
        考虑因素：敌对文明的实力、盟友的实力、领土接壤情况、整体战略环境
        
        Returns:
            float: 安全风险分数 (0-1)
        """
        if not neighbors:
            return 0.0
        
        # 计算敌对文明的综合实力
        enemy_strength = 0.0
        ally_strength = 0.0
        
        for neighbor, relation in neighbors.items():
            if relation < -0.4:  # 敌对关系阈值
                # 考虑距离因素的敌对实力
                distance_factor = 1.0 / (agent.get_distance(neighbor) + 1)  # 距离越近威胁越大
                enemy_strength += neighbor.strength * distance_factor
            elif relation > 0.6:  # 盟友关系阈值
                ally_strength += neighbor.strength
        
        # 计算相对实力比
        relative_strength = (enemy_strength + 1) / (ally_strength + agent.strength + 1)  # 避免除以零
        
        # 计算安全风险（基于对数函数，使风险增长更加合理）
        security_risk = 1.0 - 1.0 / (1.0 + exp(relative_strength - 1.0))
        
        return min(max(security_risk, 0.0), 1.0)
    
    def _assess_development_potential(self, agent, neighbors):
        """评估文明的发展潜力
        
        考虑因素：科技水平、人口质量、现有基础设施、与先进文明的交流机会
        
        Returns:
            float: 发展潜力分数 (0-1)
        """
        # 科技水平评分
        tech_score = sum(agent.technology.values()) / (len(agent.technology) * 5.0)  # 假设每项科技满分5分
        
        # 人口质量评分（考虑人口规模和科技加成）
        population_quality = min(agent.population / 1000.0, 1.0)  # 人口基数
        population_quality *= (1 + tech_score * 0.5)  # 科技对人口质量的提升
        
        # 计算与先进文明的科技差距和交流机会
        tech_exchange_potential = 0.0
        if neighbors:
            avg_neighbor_tech = sum(sum(n.technology.values()) for n in neighbors) / len(neighbors)
            own_tech = sum(agent.technology.values())
            # 与更先进文明的科技差距带来的学习潜力
            if avg_neighbor_tech > own_tech:
                tech_exchange_potential = min((avg_neighbor_tech - own_tech) / own_tech, 1.0) if own_tech > 0 else 1.0
        
        # 综合发展潜力
        development_potential = (tech_score * 0.4 + population_quality * 0.3 + tech_exchange_potential * 0.3)
        
        return min(max(development_potential, 0.0), 1.0)
    
    def _calculate_game_theory_influence(self, agent, neighbors):
        """基于博弈论计算策略影响
        
        使用演化博弈论模型分析不同策略的长期收益
        
        Returns:
            dict: 各策略的博弈论影响分数
        """
        influence = {
            'expansion': 0.0,
            'defense': 0.0,
            'trade': 0.0,
            'research': 0.0
        }
        
        if not neighbors:
            return influence
        
        # 简单的演化博弈分析
        # 计算不同策略组合的收益矩阵
        # 这里使用简化的收益矩阵，实际应用中可以根据具体模型进行扩展
        
        # 统计邻居的策略分布
        neighbor_strategies = defaultdict(float)
        for neighbor, relation in neighbors.items():
            # 假设邻居上一轮的策略选择对当前决策有影响
            if hasattr(neighbor, 'last_strategy'):
                neighbor_strategies[neighbor.last_strategy] += 1.0
        
        # 归一化邻居策略分布
        total = sum(neighbor_strategies.values())
        if total > 0:
            for key in neighbor_strategies:
                neighbor_strategies[key] /= total
        
        # 基于博弈论的策略收益计算
        # 这里使用简化的收益矩阵，实际应用中可以根据具体模型进行扩展
        
        # 扩张策略的收益取决于敌对策略的比例
        expansion_benefit = 1.0 - neighbor_strategies.get('defense', 0) * 0.7  # 防御克制扩张
        expansion_benefit += neighbor_strategies.get('expansion', 0) * 0.3  # 扩张对抗扩张的收益较低
        influence['expansion'] = expansion_benefit
        
        # 防御策略的收益取决于扩张策略的比例
        defense_benefit = neighbor_strategies.get('expansion', 0) * 0.8  # 防御对抗扩张有高收益
        defense_benefit -= neighbor_strategies.get('research', 0) * 0.2  # 防御对研发的影响
        influence['defense'] = defense_benefit
        
        # 贸易策略的收益取决于和平策略的比例
        peaceful_strategies = neighbor_strategies.get('trade', 0) + neighbor_strategies.get('research', 0)
        trade_benefit = peaceful_strategies * 0.9
        trade_benefit -= neighbor_strategies.get('expansion', 0) * 0.5  # 贸易受扩张威胁
        influence['trade'] = trade_benefit
        
        # 研发策略的长期收益
        research_benefit = 0.5  # 基础长期收益
        research_benefit += (1 - sum(neighbor_strategies.values())) * 0.3  # 独立发展的收益
        research_benefit += neighbor_strategies.get('research', 0) * 0.2  # 研发合作的收益
        influence['research'] = research_benefit
        
        # 归一化影响力分数
        total_influence = sum(influence.values())
        if total_influence > 0:
            for key in influence:
                influence[key] /= total_influence
        
        return influence
    
    def _learn_from_history(self, agent_id):
        """从历史经验中学习
        
        分析过去策略的效果，调整未来的策略选择
        
        Returns:
            dict: 基于历史经验的策略调整分数
        """
        influence = {
            'expansion': 0.0,
            'defense': 0.0,
            'trade': 0.0,
            'research': 0.0
        }
        
        # 获取该文明的策略历史
        if agent_id not in self.strategy_history or len(self.strategy_history[agent_id]) < 2:
            return influence
        
        # 简单的历史学习机制：如果过去选择的策略在当时的环境下表现良好，则增加该策略的权重
        # 这里使用简化的假设，实际应用中需要结合策略执行后的实际效果数据
        
        # 计算最近策略的平均表现
        recent_strategies = list(self.strategy_history[agent_id])
        avg_strategies = defaultdict(float)
        for strategy in recent_strategies:
            for key, value in strategy.items():
                avg_strategies[key] += value
        
        # 归一化
        total = sum(avg_strategies.values())
        if total > 0:
            for key in avg_strategies:
                avg_strategies[key] /= total
                # 假设过去表现好的策略在类似环境下仍然有效
                # 给予表现稳定的策略更高的权重
                influence[key] = avg_strategies[key]
        
        return influence
    
    def _apply_metacognition(self, agent, strategy_scores):
        """应用元认知调整策略选择
        
        文明对自身决策过程进行反思和调整
        
        Returns:
            dict: 调整后的策略分数
        """
        # 创建策略分数的副本以避免修改原数据
        adjusted_scores = copy.deepcopy(strategy_scores)
        
        # 元认知调整逻辑：
        # 1. 识别当前策略选择中的偏见或惯性
        # 2. 根据长期目标调整短期策略
        # 3. 考虑策略多样性和适应性
        
        # 检测策略惯性（过度依赖某一策略）
        max_strategy = max(strategy_scores, key=strategy_scores.get)
        max_score = strategy_scores[max_strategy]
        avg_score = sum(strategy_scores.values()) / len(strategy_scores)
        
        # 如果某一策略的分数明显高于平均值，适当降低其权重以鼓励多样化
        if max_score > avg_score * 1.5:
            adjustment = (max_score - avg_score * 1.5) * self.metacognition_rate
            adjusted_scores[max_strategy] -= adjustment
            
            # 将减少的权重分配给其他策略
            other_strategies = [s for s in strategy_scores if s != max_strategy]
            if other_strategies:
                for s in other_strategies:
                    adjusted_scores[s] += adjustment / len(other_strategies)
        
        # 确保所有分数为正
        for key in adjusted_scores:
            adjusted_scores[key] = max(adjusted_scores[key], 0.001)
        
        # 重新归一化
        total = sum(adjusted_scores.values())
        if total > 0:
            for key in adjusted_scores:
                adjusted_scores[key] /= total
        
        return adjusted_scores
    
    def _record_strategy_history(self, agent_id, strategy_scores):
        """记录策略选择历史
        
        保存文明的策略选择记录，用于历史学习
        """
        # 记录当前策略选择
        self.strategy_history[agent_id].append(copy.deepcopy(strategy_scores))
        
        # 保持历史记录不超过记忆窗口大小
        while len(self.strategy_history[agent_id]) > self.memory_window:
            self.strategy_history[agent_id].popleft()


class ComplexResourceManager:
    """复杂资源管理系统，实现更科学的资源分布和管理机制"""
    
    def __init__(self, config=None):
        """初始化复杂资源管理系统
        
        Args:
            config: 模拟配置对象
        """
        self.config = config
        # 资源类型定义
        self.resource_types = {
            'food': {'base_value': 1.0, 'regeneration_rate': 0.01, 'consumption_rate': 0.02},
            'energy': {'base_value': 1.5, 'regeneration_rate': 0.005, 'consumption_rate': 0.015},
            'materials': {'base_value': 2.0, 'regeneration_rate': 0.003, 'consumption_rate': 0.01},
            'technology': {'base_value': 3.0, 'regeneration_rate': 0.002, 'consumption_rate': 0.005}
        }
        
    def generate_resource_map(self, grid_size):
        """生成复杂的资源分布图
        
        基于地理信息系统(GIS)原理，创建更真实的资源分布
        
        Args:
            grid_size: 网格大小
        
        Returns:
            dict: 资源分布地图，键为位置坐标，值为资源字典
        """
        resources = {}
        
        # 创建基础地形特征
        elevation_map = self._generate_elevation_map(grid_size)
        moisture_map = self._generate_moisture_map(grid_size)
        
        # 为每个位置生成资源
        for i in range(grid_size):
            for j in range(grid_size):
                pos = (i, j)
                # 根据地形特征生成资源
                resources[pos] = self._generate_resources_for_position(pos, elevation_map, moisture_map)
        
        return resources
    
    def _generate_elevation_map(self, grid_size):
        """生成海拔高度图
        
        使用Perlin噪声或类似算法生成更真实的地形
        
        Returns:
            np.ndarray: 海拔高度图
        """
        # 简化实现，实际应用中可以使用更复杂的噪声算法
        elevation = np.random.rand(grid_size, grid_size) * 0.5  # 基础随机高度
        
        # 添加一些大尺度特征
        for scale in [grid_size // 4, grid_size // 8, grid_size // 16]:
            if scale > 0:
                # 计算大尺度特征的大小
                large_size = int(np.ceil(grid_size / scale))
                large_features = np.random.rand(large_size, large_size)
                # 上采样到大网格
                upsampled = np.repeat(np.repeat(large_features, scale, axis=0), scale, axis=1)
                # 截取到原始网格大小
                upsampled = upsampled[:grid_size, :grid_size]
                # 确保upsampled的形状与elevation匹配
                if upsampled.shape == elevation.shape:
                    # 叠加到大尺度特征
                    elevation += upsampled * 0.1
        
        # 归一化到0-1范围
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        
        return elevation
    
    def _generate_moisture_map(self, grid_size):
        """生成湿度图
        
        基于简化的气候模型生成湿度分布
        
        Returns:
            np.ndarray: 湿度图
        """
        # 基础湿度分布
        moisture = np.random.rand(grid_size, grid_size) * 0.5
        
        # 添加纬度效应（简化模型）
        for i in range(grid_size):
            # 模拟赤道附近湿度较高，两极较低
            lat_factor = 1.0 - abs(i / grid_size - 0.5) * 2.0  # 0.5 - 0 - 0.5
            moisture[i, :] += lat_factor * 0.3
        
        # 归一化到0-1范围
        moisture = (moisture - moisture.min()) / (moisture.max() - moisture.min())
        
        return moisture
    
    def _generate_resources_for_position(self, pos, elevation_map, moisture_map):
        """为特定位置生成资源
        
        根据地形特征确定资源类型和数量
        
        Args:
            pos: 位置坐标
            elevation_map: 海拔高度图
            moisture_map: 湿度图
        
        Returns:
            dict: 资源字典
        """
        x, y = pos
        elevation = elevation_map[x, y]
        moisture = moisture_map[x, y]
        
        resources = {}
        
        # 根据地形特征确定各资源的丰富程度
        # 食物资源：适宜的海拔和湿度下丰富
        food_factor = exp(-((elevation - 0.3) ** 2 / (2 * 0.1 ** 2)) - ((moisture - 0.7) ** 2 / (2 * 0.15 ** 2)))
        resources['food'] = max(0, np.random.normal(food_factor * 5, 1))
        
        # 能源资源：某些特定地形下丰富
        energy_factor = 0
        if 0.6 < elevation < 0.8:  # 山区可能有矿物能源
            energy_factor = 0.7
        elif moisture < 0.2:  # 沙漠可能有太阳能或石油
            energy_factor = 0.5
        resources['energy'] = max(0, np.random.normal(energy_factor * 4, 1))
        
        # 材料资源：多分布在山区和丘陵
        materials_factor = elevation * 0.8 + np.random.normal(0, 0.1)
        resources['materials'] = max(0, np.random.normal(materials_factor * 3, 1))
        
        # 科技资源：与特殊地理特征相关
        # 这里使用随机分布模拟特殊地理位置对科技发展的影响
        tech_factor = 0.1 + np.random.random() * 0.2
        resources['technology'] = max(0, np.random.normal(tech_factor * 2, 0.5))
        
        return resources
    
    def calculate_resource_value(self, resources):
        """计算资源的总价值
        
        考虑不同资源类型的相对价值和稀缺性
        
        Args:
            resources: 资源字典
        
        Returns:
            float: 资源总价值
        """
        total_value = 0.0
        
        for resource_type, amount in resources.items():
            if resource_type in self.resource_types:
                # 基础价值乘以数量
                total_value += self.resource_types[resource_type]['base_value'] * amount
        
        return total_value
    
    def regenerate_resources(self, resources, position, cycle):
        """资源再生逻辑
        
        模拟资源的自然再生过程
        
        Args:
            resources: 当前资源字典
            position: 位置坐标
            cycle: 当前周期
        
        Returns:
            dict: 更新后的资源字典
        """
        updated_resources = copy.deepcopy(resources)
        
        for resource_type, amount in updated_resources.items():
            if resource_type in self.resource_types:
                # 基础再生率
                regeneration_rate = self.resource_types[resource_type]['regeneration_rate']
                
                # 添加一些周期变化（模拟季节等因素）
                seasonal_factor = 1.0 + 0.3 * sin(2 * pi * cycle / 100)  # 每100个周期一个周期
                
                # 应用再生
                updated_amount = amount * (1 + regeneration_rate * seasonal_factor)
                
                # 设置上限，避免资源无限增长
                max_amount = amount * 1.5 if amount > 0 else 10.0
                updated_resources[resource_type] = min(updated_amount, max_amount)
        
        return updated_resources


class CulturalInfluence:
    """文化影响系统，模拟文明间的文化传播和影响"""
    
    def __init__(self, config=None):
        """初始化文化影响系统
        
        Args:
            config: 模拟配置对象
        """
        self.config = config
        # 文化特性定义
        self.cultural_traits = {
            'collectivism': {'description': '集体主义', 'influence_factor': 0.1},
            'individualism': {'description': '个人主义', 'influence_factor': 0.1},
            'militarism': {'description': '军国主义', 'influence_factor': 0.15},
            'pacifism': {'description': '和平主义', 'influence_factor': 0.15},
            'tradition': {'description': '传统主义', 'influence_factor': 0.08},
            'innovation': {'description': '创新精神', 'influence_factor': 0.08},
            'expansionism': {'description': '扩张主义', 'influence_factor': 0.12},
            'isolationism': {'description': '孤立主义', 'influence_factor': 0.12}
        }
        
    def initialize_culture(self, agent):
        """为文明初始化文化特性
        
        Args:
            agent: 文明智能体
        """
        # 随机初始化文化特性，但保证内部一致性
        agent.culture = {}
        
        # 生成相互对立的特性对
        trait_pairs = [
            ('collectivism', 'individualism'),
            ('militarism', 'pacifism'),
            ('tradition', 'innovation'),
            ('expansionism', 'isolationism')
        ]
        
        for trait1, trait2 in trait_pairs:
            # 为每对特性生成一个0-1之间的值，表示两种特性的相对强度
            value = random.random()
            agent.culture[trait1] = value
            agent.culture[trait2] = 1.0 - value
            
    def update_cultural_influence(self, agent, neighbors):
        """更新文明间的文化影响
        
        Args:
            agent: 当前文明智能体
            neighbors: 邻居文明列表
        """
        if not hasattr(agent, 'culture'):
            self.initialize_culture(agent)
            
        # 文化影响来自贸易、交流和征服等过程
        for neighbor, relation in neighbors.items():
            if hasattr(neighbor, 'culture'):
                # 根据关系类型和强度确定文化影响程度
                influence_strength = abs(relation) * 0.02  # 关系越强，文化影响越大
                
                # 文化相似性会影响文化传播的效率
                similarity = self._calculate_cultural_similarity(agent.culture, neighbor.culture)
                
                # 文化传播的实际强度
                actual_influence = influence_strength * (1 - similarity)  # 差异越大，传播潜力越大
                
                # 应用文化影响
                for trait in agent.culture:
                    if trait in neighbor.culture:
                        # 文化特性向邻居的方向移动
                        agent.culture[trait] += actual_influence * (neighbor.culture[trait] - agent.culture[trait])
                        # 确保文化特性值在0-1范围内
                        agent.culture[trait] = max(0.0, min(1.0, agent.culture[trait]))
                        
    def _calculate_cultural_similarity(self, culture1, culture2):
        """计算两种文化的相似性
        
        Args:
            culture1: 第一种文化
            culture2: 第二种文化
        
        Returns:
            float: 相似性分数 (0-1)
        """
        similarity = 0.0
        common_traits = set(culture1.keys()) & set(culture2.keys())
        
        if not common_traits:
            return 0.0
            
        for trait in common_traits:
            # 计算每种特性的差异
            similarity += 1.0 - abs(culture1[trait] - culture2[trait])
            
        # 归一化相似度
        similarity /= len(common_traits)
        
        return similarity
        
    def get_cultural_bonuses(self, agent):
        """计算文化带来的属性加成
        
        Args:
            agent: 文明智能体
        
        Returns:
            dict: 属性加成字典
        """
        if not hasattr(agent, 'culture'):
            self.initialize_culture(agent)
            
        bonuses = {
            'research_speed': 0.0,
            'resource_collection': 0.0,
            'military_strength': 0.0,
            'population_growth': 0.0,
            'diplomacy_effectiveness': 0.0
        }
        
        # 应用文化特性带来的加成
        # 创新精神提升研发速度
        bonuses['research_speed'] += agent.culture.get('innovation', 0) * 0.3
        # 个人主义也能提升研发速度（通过鼓励独立思考）
        bonuses['research_speed'] += agent.culture.get('individualism', 0) * 0.15
        
        # 集体主义提升资源收集效率
        bonuses['resource_collection'] += agent.culture.get('collectivism', 0) * 0.2
        # 传统主义提升资源利用效率
        bonuses['resource_collection'] += agent.culture.get('tradition', 0) * 0.1
        
        # 军国主义提升军事实力
        bonuses['military_strength'] += agent.culture.get('militarism', 0) * 0.3
        # 扩张主义也能提升军事实力
        bonuses['military_strength'] += agent.culture.get('expansionism', 0) * 0.2
        
        # 集体主义提升人口增长
        bonuses['population_growth'] += agent.culture.get('collectivism', 0) * 0.2
        # 和平主义创造稳定环境，有利于人口增长
        bonuses['population_growth'] += agent.culture.get('pacifism', 0) * 0.15
        
        # 和平主义提升外交 effectiveness
        bonuses['diplomacy_effectiveness'] += agent.culture.get('pacifism', 0) * 0.25
        # 个人主义可能降低外交 effectiveness
        bonuses['diplomacy_effectiveness'] -= agent.culture.get('individualism', 0) * 0.1
        
        return bonuses


# 确保sin和pi可用
from math import sin, pi


# 示例用法
if __name__ == "__main__":
    # 创建一个简单的配置对象
    class DummyConfig:
        MEMORY_WINDOW_SIZE = 10
        STRATEGY_EXPLORATION_PROBABILITY = 0.05
        METACOGNITION_LEARNING_RATE = 0.05
        ENABLE_METACOGNITION = True
        
    # 初始化高级演化引擎
    config = DummyConfig()
    advanced_evolution = AdvancedEvolution(config)
    
    # 创建一个简单的文明代理模拟对象
    class DummyAgent:
        def __init__(self, agent_id):
            self.id = agent_id
            self.resources = 50.0
            self.population = 100
            self.strength = 10.0
            self.territory = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 初始领土
            self.technology = {'agriculture': 1.0, 'metalworking': 0.5}  # 初始科技
            self.resource_consumption_rate = 0.015
            self.resource_regeneration_rate = 0.008
            
        def get_distance(self, other_agent):
            # 简化的距离计算
            return 1.0  # 假设所有邻居距离相同
            
    # 创建一些测试文明
    agent1 = DummyAgent(1)
    agent2 = DummyAgent(2)
    agent3 = DummyAgent(3)
    
    # 定义邻居关系
    neighbors = {
        agent2: 0.3,  # 友好关系
        agent3: -0.5  # 敌对关系
    }
    
    # 简单的全局资源分布
    global_resources = {
        (0, 0): 10.0,
        (0, 1): 15.0,
        (1, 0): 8.0,
        (1, 1): 12.0
    }
    
    # 计算策略倾向
    strategy_tendency = advanced_evolution.calculate_strategy_tendency(agent1, neighbors, global_resources)
    
    print("=== 高级演化引擎测试 ===")
    print(f"文明1的策略倾向:")
    for strategy, score in strategy_tendency.items():
        print(f"  {strategy}: {score:.4f}")
    
    # 测试复杂资源管理系统
    resource_manager = ComplexResourceManager(config)
    resource_map = resource_manager.generate_resource_map(10)  # 生成10x10的资源地图
    
    print(f"\n=== 复杂资源管理系统测试 ===")
    print(f"生成的资源地图包含 {len(resource_map)} 个位置的资源信息")
    print(f"示例位置 (5,5) 的资源: {resource_map[(5,5)]}")
    
    # 测试文化影响系统
    culture_system = CulturalInfluence(config)
    culture_system.initialize_culture(agent1)
    
    print(f"\n=== 文化影响系统测试 ===")
    print(f"文明1的初始文化特性:")
    for trait, value in agent1.culture.items():
        print(f"  {trait}: {value:.4f}")
    
    # 计算文化加成
    cultural_bonuses = culture_system.get_cultural_bonuses(agent1)
    print(f"\n文明1的文化加成:")
    for bonus_type, bonus_value in cultural_bonuses.items():
        print(f"  {bonus_type}: {bonus_value:.4f}")