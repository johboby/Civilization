#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化高级功能演示
展示如何使用高级演化、复杂资源管理和文化影响系统
"""

import numpy as np
import matplotlib.pyplot as plt
from multi_agent_simulation import MultiAgentSimulation
from simulation_config import SimulationConfig
import os
import time
from datetime import datetime

# 简化字体设置，只使用Windows系统上最常见的中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class AdvancedEvolutionDemo:
    """高级演化功能演示类"""
    def __init__(self):
        self.config = SimulationConfig()
        # 启用高级功能
        self.config.USE_ADVANCED_EVOLUTION = True
        self.config.USE_COMPLEX_RESOURCES = True
        self.config.USE_CULTURAL_INFLUENCE = True
        
        # 设置适合演示的参数
        self.config.NUM_CIVILIZATIONS = 6
        self.config.GRID_SIZE = 25
        self.config.SIMULATION_CYCLES = 100
        self.config.VISUALIZATION_INTERVAL = 5
        self.config.PRINT_LOGS = True
        self.config.LOG_INTERVAL = 10
        
        # 高级演化参数调整
        self.config.EVOLUTION_LEARNING_RATE = 0.15
        self.config.STRATEGY_EXPLORATION_RATE = 0.2
        self.config.CULTURAL_DIFFUSION_RATE = 0.07
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("results", f"advanced_evolution_demo_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.config.OUTPUT_DIR = self.output_dir
        
    def run_demo(self):
        """运行高级演化演示"""
        print("=== 文明演化高级功能演示 ===")
        print(f"配置: {self.config.__class__.__name__}")
        print(f"启用高级演化: {self.config.USE_ADVANCED_EVOLUTION}")
        print(f"启用复杂资源: {self.config.USE_COMPLEX_RESOURCES}")
        print(f"启用文化影响: {self.config.USE_CULTURAL_INFLUENCE}")
        print(f"文明数量: {self.config.NUM_CIVILIZATIONS}")
        print(f"网格大小: {self.config.GRID_SIZE}")
        print(f"模拟周期: {self.config.SIMULATION_CYCLES}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 50)
        
        start_time = time.time()
        
        # 创建模拟实例
        simulation = MultiAgentSimulation(self.config)
        
        # 运行模拟
        print("开始模拟...")
        for cycle in range(self.config.SIMULATION_CYCLES):
            simulation.step(cycle)
            
            if self.config.PRINT_LOGS and cycle % self.config.LOG_INTERVAL == 0:
                print(f"周期 {cycle}/{self.config.SIMULATION_CYCLES}")
                # 打印一些统计信息
                total_resources = sum(agent.resources for agent in simulation.agents)
                max_territory = max(len(agent.territory) for agent in simulation.agents)
                avg_tech_level = np.mean([sum(agent.technology.values()) for agent in simulation.agents])
                print(f"  总资源: {total_resources:.2f}, 最大领土: {max_territory}, 平均科技水平: {avg_tech_level:.2f}")
        
        end_time = time.time()
        print("模拟完成!")
        print(f"模拟耗时: {end_time - start_time:.2f} 秒")
        
        # 生成可视化结果
        self.generate_visualizations(simulation)
        
        print(f"结果已保存至: {self.output_dir}")
        return simulation
    
    def generate_visualizations(self, simulation):
        """生成高级演化功能相关的可视化图表"""
        print("生成可视化结果...")
        
        # 1. 策略倾向变化图
        if hasattr(simulation, 'strategy_tendency_history') and simulation.strategy_tendency_history:
            self._plot_strategy_tendencies(simulation)
        
        # 2. 文化相似性热图
        if hasattr(simulation, 'cultural_history') and simulation.cultural_history:
            self._plot_cultural_similarity(simulation)
        
        # 3. 资源分布和文明发展关系图
        if hasattr(simulation, 'complex_resources') and simulation.complex_resources:
            self._plot_resource_civilization_relationship(simulation)
        
        # 4. 文明属性发展趋势对比
        self._plot_agent_attributes_comparison(simulation)
        
    def _plot_strategy_tendencies(self, simulation):
        """绘制策略倾向变化趋势图"""
        plt.figure(figsize=(12, 8))
        
        # 提取每种策略的平均值
        strategy_types = ['expansion', 'defense', 'trade', 'research']
        strategy_data = {st: [] for st in strategy_types}
        
        for cycle_data in simulation.strategy_tendency_history:
            avg_tendencies = {st: 0.0 for st in strategy_types}
            for agent_tendency in cycle_data:
                for st in strategy_types:
                    avg_tendencies[st] += agent_tendency.get(st, 0.0)
            
            # 计算平均值
            for st in strategy_types:
                avg_tendencies[st] /= len(cycle_data)
                strategy_data[st].append(avg_tendencies[st])
        
        # 绘制每种策略的趋势
        colors = ['r', 'b', 'g', 'purple']
        for i, st in enumerate(strategy_types):
            if strategy_data[st]:
                plt.plot(strategy_data[st], label=st, color=colors[i])
        
        plt.title('平均策略倾向变化趋势')
        plt.xlabel('模拟周期')
        plt.ylabel('策略倾向值')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'strategy_tendencies.png'), dpi=300)
        plt.close()
    
    def _plot_cultural_similarity(self, simulation):
        """绘制文化相似性热图"""
        # 只取最后一个周期的文化数据
        if simulation.cultural_history:
            last_cycle_culture = simulation.cultural_history[-1]
            num_agents = len(last_cycle_culture)
            
            if num_agents > 0:
                # 计算文化相似性矩阵
                similarity_matrix = np.zeros((num_agents, num_agents))
                
                for i in range(num_agents):
                    for j in range(num_agents):
                        if i in last_cycle_culture and j in last_cycle_culture:
                            # 简单的欧几里得距离计算相似性
                            culture_i = last_cycle_culture[i]
                            culture_j = last_cycle_culture[j]
                            
                            # 提取共同的文化特性
                            common_traits = set(culture_i.keys()).intersection(set(culture_j.keys()))
                            if common_traits:
                                # 计算文化向量的差异
                                diffs = []
                                for trait in common_traits:
                                    diffs.append(culture_i[trait] - culture_j[trait])
                                
                                # 计算相似性（1 - 归一化距离）
                                if diffs:
                                    distance = np.linalg.norm(diffs)
                                    # 归一化到0-1范围
                                    similarity = 1.0 / (1.0 + distance)
                                    similarity_matrix[i, j] = similarity
                
                # 绘制热图
                plt.figure(figsize=(10, 8))
                plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
                plt.colorbar(label='文化相似性')
                plt.title('文明间文化相似性矩阵')
                plt.xlabel('文明ID')
                plt.ylabel('文明ID')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'cultural_similarity.png'), dpi=300)
                plt.close()
    
    def _plot_resource_civilization_relationship(self, simulation):
        """绘制资源分布和文明发展的关系图"""
        # 这个图需要根据实际的数据结构调整
        # 这里提供一个简化的版本
        plt.figure(figsize=(12, 8))
        
        # 收集每个文明的领土面积和总资源
        territory_sizes = []
        total_resources = []
        
        for agent in simulation.agents:
            territory_sizes.append(len(agent.territory))
            total_resources.append(agent.resources)
        
        # 绘制散点图
        plt.scatter(territory_sizes, total_resources, alpha=0.7)
        
        # 添加趋势线
        if len(territory_sizes) > 1:
            z = np.polyfit(territory_sizes, total_resources, 1)
            p = np.poly1d(z)
            plt.plot(territory_sizes, p(territory_sizes), "r--", alpha=0.5)
        
        plt.title('领土面积与资源总量关系')
        plt.xlabel('领土面积')
        plt.ylabel('资源总量')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'territory_resource_relationship.png'), dpi=300)
        plt.close()
    
    def _plot_agent_attributes_comparison(self, simulation):
        """绘制各文明属性发展趋势对比图"""
        # 定义要绘制的属性
        attributes = [
            ('resources', '资源', 'blue'),
            ('strength', '军事力量', 'red'),
            ('population', '人口', 'green'),
            ('global_influence', '全球影响力', 'purple')
        ]
        
        for attr_name, attr_label, color in attributes:
            plt.figure(figsize=(12, 8))
            
            # 绘制每个文明的属性趋势
            for agent in simulation.agents:
                # 提取该属性的历史数据
                attr_history = []
                for cycle_data in simulation.attribute_history:
                    # 假设attribute_history中包含了所有属性的数据
                    # 这里需要根据实际的数据结构调整
                    pass
                
                # 简化版本：只绘制最终值
                final_value = getattr(agent, attr_name, 0)
                plt.bar(agent.agent_id, final_value, color=color, alpha=0.7)
            
            plt.title(f'各文明{attr_label}对比')
            plt.xlabel('文明ID')
            plt.ylabel(attr_label)
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{attr_name}_comparison.png'), dpi=300)
            plt.close()

if __name__ == "__main__":
    # 运行高级演化演示
    demo = AdvancedEvolutionDemo()
    simulation = demo.run_demo()
    
    # 输出一些关键发现
    print("\n=== 高级演化模拟关键发现 ===")
    
    # 计算文化多样性
    if hasattr(simulation, 'cultural_history') and simulation.cultural_history:
        last_cycle_culture = simulation.cultural_history[-1]
        # 简化的文化多样性计算
        traits_values = []
        for agent_id, traits in last_cycle_culture.items():
            traits_values.extend(list(traits.values()))
        
        if traits_values:
            culture_diversity = np.std(traits_values)
            print(f"文化多样性指数: {culture_diversity:.4f}")
    
    # 计算策略多样性
    if hasattr(simulation, 'strategy_tendency_history') and simulation.strategy_tendency_history:
        # 检查最后几个周期的策略分布
        recent_cycles = min(5, len(simulation.strategy_tendency_history))
        recent_strategies = simulation.strategy_tendency_history[-recent_cycles:]
        
        # 计算每个策略的使用频率
        strategy_counts = {'expansion': 0, 'defense': 0, 'trade': 0, 'research': 0}
        
        for cycle_data in recent_strategies:
            for agent_tendency in cycle_data:
                # 找出最常用的策略
                max_strategy = max(agent_tendency, key=agent_tendency.get)
                if max_strategy in strategy_counts:
                    strategy_counts[max_strategy] += 1
        
        print("最近策略分布:")
        total = sum(strategy_counts.values())
        for strategy, count in strategy_counts.items():
            if total > 0:
                print(f"  {strategy}: {count/total*100:.1f}%")
    
    print("\n演示完成! 请查看结果目录中的可视化图表了解更多详情。")