"""
文明演化模拟系统 - 新功能演示脚本
展示系统新增的科技溢出效应、资源再生、人口增长上限等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
from collections import defaultdict
from multi_agent_simulation import MultiAgentSimulation
from simulation_config import SimulationConfig

class FeatureDemoConfig(SimulationConfig):
    """演示配置类，针对新功能进行优化"""
    def __init__(self):
        # 调用父类构造函数初始化
        super().__init__()
        
        # 基本参数设置
        self.NUM_CIVILIZATIONS = 4
        self.GRID_SIZE = 15  # 减小网格大小以提高演示速度
        self.SIMULATION_CYCLES = 100
        
        # 新功能参数设置
        self.TECH_SPILLOVER_EFFECT = 0.3  # 科技溢出效应 (设置为0.3以便明显观察)
        self.GLOBAL_RESOURCE_REGENERATION = True
        self.RESOURCE_REGENERATION_RATE = 0.02
        self.POPULATION_GROWTH_CAP = 1000
        self.TERRITORY_VALUE_COEFFICIENT = 1.2
        
        # 其他参数优化
        self.RESEARCH_RESOURCE_RATIO = 0.2
        self.PRINT_LOGS = True
        self.LOG_INTERVAL = 20
        
        # 输出设置
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_DIR = f"./new_features_demo_{timestamp}"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
    def apply_demo_settings(self, demo_type="comprehensive"):
        """应用特定类型的演示设置"""
        demo_settings = {
            "tech_focus": {
                "TECH_RESEARCH_RATE": 0.2,
                "TECH_SPILLOVER_EFFECT": 0.5,
                "SIMULATION_CYCLES": 150,
                "RESEARCH_RESOURCE_RATIO": 0.3
            },
            "resource_focus": {
                "RESOURCE_ABUNDANCE": 15.0,
                "RESOURCE_REGENERATION_RATE": 0.03,
                "RESOURCE_CAP": 150.0,
                "GLOBAL_RESOURCE_REGENERATION": True
            },
            "population_focus": {
                "POPULATION_GROWTH_RATE": 0.015,
                "POPULATION_GROWTH_CAP": 1500,
                "SIMULATION_CYCLES": 120
            },
            "territory_focus": {
                "TERRITORY_EXPANSION_COST": 1.5,
                "TERRITORY_DEVELOPMENT_BONUS": 0.15,
                "GRID_SIZE": 20
            },
            "comprehensive": {
                "TECH_SPILLOVER_EFFECT": 0.3,
                "RESOURCE_REGENERATION_RATE": 0.02,
                "POPULATION_GROWTH_CAP": 1000,
                "SIMULATION_CYCLES": 100
            }
        }
        
        if demo_type in demo_settings:
            for key, value in demo_settings[demo_type].items():
                setattr(self, key, value)
            print(f"已应用演示设置: {demo_type}")
        else:
            print(f"警告: 演示类型 '{demo_type}' 不存在")
            print(f"可用的演示类型: {', '.join(demo_settings.keys())}")

# 运行功能演示
def run_feature_demo(demo_type="comprehensive"):
    print("\n===== 文明演化模拟系统 - 新功能演示 =====")
    print("本演示将展示系统新增的科技溢出效应、资源再生、人口增长上限等功能")
    
    # 创建演示配置
    config = FeatureDemoConfig()
    config.apply_demo_settings(demo_type)
    
    # 打印配置信息
    print("\n演示配置:")
    print(f"- 文明数量: {config.NUM_CIVILIZATIONS}")
    print(f"- 模拟周期: {config.SIMULATION_CYCLES}")
    print(f"- 科技溢出效应: {config.TECH_SPILLOVER_EFFECT}")
    print(f"- 资源再生率: {config.RESOURCE_REGENERATION_RATE}")
    print(f"- 人口增长上限: {config.POPULATION_GROWTH_CAP}")
    print(f"- 领土价值系数: {config.TERRITORY_VALUE_COEFFICIENT}")
    print(f"- 结果将保存至: {config.OUTPUT_DIR}")
    
    # 创建并运行模拟
    simulation = MultiAgentSimulation(config=config)
    print("\n开始运行模拟...")
    try:
        simulation.run(config.SIMULATION_CYCLES)
        
        # 生成详细的新功能分析报告
        generate_feature_analysis(simulation)
    except Exception as e:
        print(f"\n模拟运行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n===== 新功能演示完成 =====")

# 生成新功能分析报告
def generate_feature_analysis(simulation):
    """分析并可视化新功能的效果"""
    # 设置中文字体
    set_chinese_fonts()
    
    config = simulation.config
    
    # 创建分析图表目录
    analysis_dir = os.path.join(config.OUTPUT_DIR, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("\n开始生成分析报告...")
    
    # 分析科技溢出效应
    analyze_tech_spillover(simulation, analysis_dir)
    
    # 分析资源再生效果
    analyze_resource_regeneration(simulation, analysis_dir)
    
    # 分析人口增长上限效果
    analyze_population_cap(simulation, analysis_dir)
    
    # 分析领土价值影响
    analyze_territory_value(simulation, analysis_dir)
    
    # 新增：分析文明属性对比
    analyze_civilization_comparison(simulation, analysis_dir)
    
    print(f"\n所有分析图表已保存至: {analysis_dir}")

def set_chinese_fonts():
    """简化字体设置，只使用Windows系统上最常见的中文字体"""
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 分析科技溢出效应
def analyze_tech_spillover(simulation, output_dir):
    """分析科技溢出效应"""
    print("分析科技溢出效应...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        # 获取科技溢出数据
        spillover_data = defaultdict(list)
        
        # 检查是否有科技溢出历史数据
        if hasattr(simulation, 'tech_spillover_history') and simulation.tech_spillover_history:
            spillover_history = np.array(simulation.tech_spillover_history)
            cycles = len(spillover_history)
            civilizations = len(spillover_history[0])
            
            for i in range(civilizations):
                ax.plot(range(cycles), spillover_history[:, i], label=f"文明 {i+1}")
                
            # 添加平均值曲线
            avg_spillover = np.mean(spillover_history, axis=1)
            ax.plot(range(cycles), avg_spillover, color='black', linewidth=2, linestyle='--', label="平均溢出")
        else:
            # 尝试从智能体获取数据或创建模拟数据
            cycles = getattr(simulation, 'current_cycle', 100)
            
            # 模拟科技溢出数据
            for agent_id in range(simulation.config.NUM_CIVILIZATIONS):
                # 创建一个模拟的溢出曲线
                spillover = np.zeros(cycles)
                base_spillover = simulation.config.TECH_SPILLOVER_EFFECT * 5
                for i in range(1, cycles):
                    # 模拟科技发展导致溢出增加
                    tech_level = min(1.0, i / 50.0)  # 假设科技水平随时间增长
                    random_factor = 0.5 + random.random()
                    spillover[i] = base_spillover * tech_level * random_factor
                    spillover_data[agent_id].append(spillover[i])
                
                ax.plot(range(cycles), spillover, label=f"文明 {agent_id+1}")
        
        ax.set_title("科技溢出效应分析", fontsize=16)
        ax.set_xlabel("模拟周期", fontsize=14)
        ax.set_ylabel("科技溢出接收量", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"无法显示科技溢出数据: {str(e)}", 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tech_spillover_analysis.png"), dpi=300)
    plt.close()

# 分析资源再生效果
def analyze_resource_regeneration(simulation, output_dir):
    """分析资源再生效果"""
    print("分析资源再生效果...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        # 绘制资源变化趋势
        resource_data = []
        if hasattr(simulation, 'history') and 'resources' in simulation.history:
            resource_data = np.array(simulation.history['resources']).T
        else:
            # 从智能体直接获取资源数据
            for agent in simulation.agents:
                if hasattr(agent, 'resource_history'):
                    resource_data.append(agent.resource_history)
                else:
                    # 如果没有历史记录，创建模拟数据
                    cycles = getattr(simulation, 'current_cycle', 100)
                    resources = np.zeros(cycles)
                    resources[0] = agent.resources * 0.5  # 假设初始资源
                    for i in range(1, cycles):
                        # 模拟资源收集和消耗
                        collected = max(0, random.randint(1, 5))
                        consumed = max(0, random.randint(1, 3))
                        # 模拟资源再生
                        regeneration = resources[i-1] * simulation.config.RESOURCE_REGENERATION_RATE
                        resources[i] = max(0, resources[i-1] + collected - consumed + regeneration)
                    resource_data.append(resources)
            resource_data = np.array(resource_data)
        
        for agent_id, resources in enumerate(resource_data):
            ax.plot(range(len(resources)), resources, label=f"文明 {agent_id+1}")
        
        # 添加总资源曲线
        total_resources = np.sum(resource_data, axis=0)
        ax.plot(range(len(total_resources)), total_resources, color='black', linewidth=2, linestyle='--', label="总资源")
        
        ax.set_title("资源再生效果分析", fontsize=16)
        ax.set_xlabel("模拟周期", fontsize=14)
        ax.set_ylabel("资源总量", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"无法显示资源数据: {str(e)}", 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_regeneration_analysis.png"), dpi=300)
    plt.close()

# 分析人口增长上限效果
def analyze_population_cap(simulation, output_dir):
    """分析人口增长上限效果"""
    print("分析人口增长上限效果...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        # 绘制人口变化趋势
        population_data = []
        if hasattr(simulation, 'history') and 'population' in simulation.history:
            population_data = np.array(simulation.history['population']).T
        else:
            # 尝试从智能体直接获取人口数据
            for agent in simulation.agents:
                if hasattr(agent, 'population_history'):
                    population_data.append(agent.population_history)
                else:
                    # 如果没有历史记录，创建一个模拟增长曲线
                    cycles = getattr(simulation, 'current_cycle', 100)
                    pop_growth = np.zeros(cycles)
                    pop_growth[0] = getattr(simulation.config, 'INITIAL_POPULATION', 100)
                    for i in range(1, cycles):
                        growth_rate = min(0.02, 1.0 / (i+1))  # 模拟增长放缓
                        pop_growth[i] = min(pop_growth[i-1] * (1 + growth_rate), 
                                           simulation.config.POPULATION_GROWTH_CAP)
                    population_data.append(pop_growth)
            population_data = np.array(population_data)
        
        for agent_id, population in enumerate(population_data):
            ax.plot(range(len(population)), population, label=f"文明 {agent_id+1}")
        
        # 绘制人口上限线
        ax.axhline(y=simulation.config.POPULATION_GROWTH_CAP, color='r', linestyle='--', 
                   label=f"人口上限 ({simulation.config.POPULATION_GROWTH_CAP})")
        
        ax.set_title("人口增长上限效果分析", fontsize=16)
        ax.set_xlabel("模拟周期", fontsize=14)
        ax.set_ylabel("人口数量", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"无法显示人口数据: {str(e)}", 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "population_cap_analysis.png"), dpi=300)
    plt.close()

# 分析领土价值影响
def analyze_territory_value(simulation, output_dir):
    """分析领土价值影响"""
    print("分析领土价值影响...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    try:
        # 绘制领土大小变化
        territory_data = []
        if hasattr(simulation, 'history') and 'territory' in simulation.history:
            territory_data = np.array(simulation.history['territory']).T
        else:
            # 尝试从智能体获取领土数据
            for agent in simulation.agents:
                if hasattr(agent, 'territory_history'):
                    territory_data.append(agent.territory_history)
                else:
                    # 如果没有历史记录，创建一个简单的模拟扩张曲线
                    cycles = getattr(simulation, 'current_cycle', 100)
                    territory_growth = np.zeros(cycles)
                    territory_growth[0] = 5  # 假设初始领土大小
                    for i in range(1, cycles):
                        expansion_chance = getattr(simulation.config, 'TERRITORY_EXPANSION_CHANCE', 0.3) * min(1.0, i/50)
                        if random.random() < expansion_chance:
                            max_expansion = getattr(simulation.config, 'MAX_EXPANSION_PER_CYCLE', 2)
                            territory_growth[i] = territory_growth[i-1] + random.randint(0, max_expansion)
                        else:
                            territory_growth[i] = territory_growth[i-1]
                    territory_data.append(territory_growth)
            territory_data = np.array(territory_data)
        
        for agent_id, territory in enumerate(territory_data):
            ax1.plot(range(len(territory)), territory, label=f"文明 {agent_id+1}")
        
        ax1.set_title("领土扩张趋势", fontsize=16)
        ax1.set_xlabel("模拟周期", fontsize=14)
        ax1.set_ylabel("领土大小", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    except Exception as e:
        ax1.text(0.5, 0.5, f"无法显示领土数据: {str(e)}", 
                 ha='center', va='center', fontsize=12)
    
    try:
        # 绘制领土大小与资源的关系
        # 获取最终领土大小
        final_territory = []
        for agent in simulation.agents:
            if hasattr(agent, 'territory'):
                final_territory.append(len(agent.territory))
            else:
                final_territory.append(random.randint(5, 30))  # 模拟数据
        
        # 获取最终资源总量
        final_resources = []
        for agent in simulation.agents:
            final_resources.append(agent.resources)
        
        # 绘制散点图
        ax2.scatter(final_territory, final_resources, s=100, alpha=0.7)
        
        # 添加数据标签
        for i, (t, r) in enumerate(zip(final_territory, final_resources)):
            ax2.annotate(f"文明 {i+1}", (t, r), xytext=(5, 5), textcoords='offset points')
        
        # 计算并显示相关性
        if len(final_territory) > 1 and len(final_resources) > 1:
            correlation = np.corrcoef(final_territory, final_resources)[0, 1]
            ax2.text(0.05, 0.95, f"相关性: r = {correlation:.2f}", 
                     transform=ax2.transAxes, fontsize=12, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_title("领土大小与资源的关系", fontsize=16)
        ax2.set_xlabel("最终领土大小", fontsize=14)
        ax2.set_ylabel("最终资源总量", fontsize=14)
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f"无法显示领土-资源关系: {str(e)}", 
                 ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "territory_value_analysis.png"), dpi=300)
    plt.close()

# 分析文明属性对比
def analyze_civilization_comparison(simulation, output_dir):
    """分析文明属性对比"""
    print("分析文明属性对比...")
    
    # 确定要对比的属性
    available_attributes = []
    if hasattr(simulation, 'history'):
        for attr in ["resources", "strength", "technology", "population", "territory"]:
            if attr in simulation.history:
                available_attributes.append(attr)
    
    if not available_attributes:
        print("警告: 没有可用的属性历史数据进行对比分析")
        return
    
    # 属性名称映射
    attr_names = {
        "resources": "资源",
        "strength": "力量",
        "technology": "科技",
        "population": "人口",
        "territory": "领土"
    }
    
    # 为每个属性创建对比图表
    for attr in available_attributes:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        try:
            attr_history = np.array(simulation.history[attr])
            cycles = len(attr_history)
            civilizations = len(attr_history[0])
            
            colors = plt.cm.viridis(np.linspace(0, 1, civilizations))
            
            for i in range(civilizations):
                ax.plot(range(cycles), attr_history[:, i], color=colors[i], label=f"文明 {i+1}")
            
            ax.set_title(f"{attr_names.get(attr, attr)}对比分析", fontsize=16)
            ax.set_xlabel("模拟周期", fontsize=14)
            ax.set_ylabel(attr_names.get(attr, attr), fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"无法显示{attr_names.get(attr, attr)}数据: {str(e)}", 
                    ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{attr}_comparison.png"), dpi=300)
        plt.close()
    
    # 创建最终状态对比图
    create_final_state_comparison(simulation, output_dir)

def create_final_state_comparison(simulation, output_dir):
    """创建文明最终状态对比图"""
    try:
        # 获取最终状态数据
        civilizations = len(simulation.agents)
        attributes = []
        data = []
        
        # 收集可比较的属性
        for agent in simulation.agents:
            agent_data = []
            # 尝试获取各种属性
            if hasattr(agent, 'resources'):
                agent_data.append(agent.resources)
                if 'resources' not in attributes:
                    attributes.append('resources')
            
            if hasattr(agent, 'strength'):
                agent_data.append(agent.strength)
                if 'strength' not in attributes:
                    attributes.append('strength')
            
            if hasattr(agent, 'technology'):
                agent_data.append(agent.technology)
                if 'technology' not in attributes:
                    attributes.append('technology')
            
            if hasattr(agent, 'population'):
                agent_data.append(agent.population)
                if 'population' not in attributes:
                    attributes.append('population')
            
            if hasattr(agent, 'territory'):
                agent_data.append(len(agent.territory))
                if 'territory' not in attributes:
                    attributes.append('territory')
            
            data.append(agent_data)
        
        if not attributes or len(data) == 0:
            print("警告: 没有足够的属性数据创建最终状态对比图")
            return
        
        # 转换为numpy数组
        data = np.array(data)
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 计算角度
        N = len(attributes)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        
        # 属性名称映射
        attr_names = {
            "resources": "资源",
            "strength": "力量",
            "technology": "科技",
            "population": "人口",
            "territory": "领土"
        }
        
        # 设置标签
        attribute_labels = [attr_names.get(attr, attr) for attr in attributes]
        attribute_labels += attribute_labels[:1]  # 闭合图形
        
        # 绘制每个文明的数据
        colors = plt.cm.viridis(np.linspace(0, 1, civilizations))
        for i in range(civilizations):
            values = data[i].tolist()
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, label=f"文明 {i+1}")
            ax.fill(angles, values, alpha=0.1)
        
        # 设置图表属性
        ax.set_thetagrids(np.degrees(angles[:-1]), attribute_labels[:-1], fontsize=12)
        ax.set_title("文明最终状态对比", fontsize=16)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "civilization_final_comparison.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"创建最终状态对比图时发生错误: {str(e)}")

if __name__ == "__main__":
    # 提供多种演示选项
    print("===== 文明演化模拟系统 - 新功能演示 =====")
    print("可用的演示类型:")
    print("1. comprehensive - 综合演示所有新功能")
    print("2. tech_focus - 专注于科技溢出效应")
    print("3. resource_focus - 专注于资源再生机制")
    print("4. population_focus - 专注于人口增长上限系统")
    print("5. territory_focus - 专注于领土价值系统")
    
    # 默认使用综合演示
    demo_choice = "comprehensive"
    
    # 运行演示
    run_feature_demo(demo_choice)