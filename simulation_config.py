"""
文明演化模拟系统配置文件
用于管理所有可调整的模拟参数
"""

class SimulationConfig:
    """模拟配置类，包含所有可调整的参数"""
    def __init__(self):
        # 基本模拟参数
        self.NUM_CIVILIZATIONS = 4  # 文明数量
        self.GRID_SIZE = 20  # 网格大小
        self.SIMULATION_CYCLES = 100  # 模拟周期数
        self.RANDOM_SEED = 42  # 随机种子，用于复现结果
        self.INITIAL_STRENGTH = 50  # 初始军事力量
        self.INITIAL_RESOURCES = 200 # 初始资源（向后兼容）
        self.INITIAL_RESOURCE_AMOUNT = 200 # 初始资源量
        self.INITIAL_POPULATION = 100 # 初始人口
        
        # 资源系统参数
        self.RESOURCE_ABUNDANCE = 10.0  # 初始资源丰度
        self.RESOURCE_REGENERATION_RATE = 0.02  # 资源再生率
        self.RESOURCE_CAP = 100.0  # 资源上限
        self.RESOURCE_CONSUMPTION_PER_ACTION = 1.0  # 每次行动消耗的资源
        self.RESOURCE_DISTRIBUTION_METHOD = "random"  # 资源分布方法：random, clustered, uniform
        self.RESOURCE_PER_CAPITA_FOR_MAX_GROWTH = 10 # 最大人口增长所需人均资源
        self.MAX_RESOURCE_PER_CELL = 100.0  # 每个单元格最大资源量
        self.TERRITORY_VALUE_COEFFICIENT = 1.0 # 领土价值系数
        self.GLOBAL_RESOURCE_REGENERATION = True # 是否启用全局资源再生
        
        # 人口系统参数
        self.POPULATION_GROWTH_RATE = 0.01  # 基础人口增长率
        self.POPULATION_CONSUMPTION_RATE = 0.5  # 人均资源消耗率
        self.POPULATION_CAP_PER_TERRITORY_VALUE = 2.0  # 单位领土价值对应的人口上限
        self.POPULATION_GROWTH_CAP = float('inf')  # 全局人口上限（默认无限）
        
        # 科技研发参数
        self.TECH_RESEARCH_RATE = 0.1  # 科技研发投入比例
        self.TECH_RESEARCH_COST = 10.0  # 科技研发成本
        self.TECH_SPILLOVER_EFFECT = 0.3  # 科技溢出效应强度
        self.TECH_BONUS_DIMINISHING_RATE = 0.1  # 科技加成递减率
        self.POPULATION_RESEARCH_BONUS_CAP = 5.0 # 人口研究加成上限
        self.BASE_RESEARCH_SPEED = 1.0 # 基础研究速度
        self.RESEARCH_RESOURCE_EFFICIENCY = 1.0 # 研发资源效率
        
        # 领土扩张参数
        self.TERRITORY_EXPANSION_COST = 2.0  # 领土扩张成本
        self.TERRITORY_BASE_VALUE = 1.0  # 基础领土价值
        self.TERRITORY_DEVELOPMENT_BONUS = 0.1  # 领土开发加成
        self.TERRITORY_EXPANSION_CHANCE = 0.3 # 扩张几率
        self.MAX_EXPANSION_PER_CYCLE = 1  # 每周期最大扩张数
        
        # 关系调整参数
        self.RELATIONSHIP_CHANGE_RATE = 0.05  # 关系变化速率
        self.ALLYSHIP_BONUS = 0.2  # 盟友关系加成
        self.ENEMY_PENALTY = 0.2  # 敌对关系惩罚
        self.RELATIONSHIP_THRESHOLD = 0.3 # 关系转变阈值
        
        # 随机化参数
        self.RANDOM_FLUCTUATION_AMPLITUDE = 0.1  # 随机波动幅度
        self.STOCHASTICITY_FACTOR = 0.2  # 随机性因子
        self.RANDOM_TERRITORY_DISTRIBUTION = False # 是否随机分配初始领土
        
        # 运行模式参数
        self.PARALLEL_PROCESSING = False  # 是否启用并行处理
        self.FAST_MODE = False # 快速模式（减少日志和可视化）
        self.PRINT_LOGS = True  # 是否打印日志
        self.LOG_INTERVAL = 10  # 日志打印间隔
        
        # 可视化参数
        self.VISUALIZATION_ENABLED = True  # 是否启用可视化
        self.VISUALIZATION_INTERVAL = 5  # 可视化更新间隔
        self.FIGURE_SIZE = (12, 8)        # 图表大小
        self.DPI = 300                    # 图像DPI
        
        # 文件输出参数
        self.OUTPUT_DIR = "results"  # 输出目录
        self.SAVE_RESULTS = True  # 是否保存结果
        self.EXPORT_FORMAT = "csv"  # 导出格式：csv, json, pickle
        self.DEFAULT_OUTPUT_DIR = "results" # 默认输出目录
        
        # 高级演化系统参数
        self.USE_ADVANCED_EVOLUTION = False  # 是否启用高级演化引擎
        self.USE_COMPLEX_RESOURCES = False   # 是否启用复杂资源管理系统
        self.USE_CULTURAL_INFLUENCE = False  # 是否启用文化影响系统
        
        # 随机事件系统参数
        self.ENABLE_RANDOM_EVENTS = True     # 是否启用随机事件系统
        self.EVENT_PROBABILITY_MODIFIER = 1.0  # 事件概率调整因子
        
        # 高级演化参数
        self.EVOLUTION_LEARNING_RATE = 0.1   # 演化学习率
        self.STRATEGY_EXPLORATION_RATE = 0.2 # 策略探索率
        self.MEMORY_WINDOW_SIZE = 10         # 历史记忆窗口大小
        self.EVOLUTION_RANDOMNESS_FACTOR = 0.1  # 演化随机性因子
        
        # 复杂资源系统参数
        self.RESOURCE_TYPES = ["food", "energy", "minerals", "technology"]  # 资源类型
        self.RESOURCE_SPOT_VARIANCE = 0.3     # 资源分布随机度
        self.RESOURCE_INTERACTION_WEIGHT = 0.2  # 资源间相互影响权重
        
        # 文化影响系统参数
        self.CULTURAL_DIFFUSION_RATE = 0.05   # 文化扩散率
        self.CULTURAL_RESISTANCE_FACTOR = 0.1 # 文化抵抗力因子
        self.CULTURAL_BONUS_STRENGTH = 0.2    # 文化加成强度
        self.NUM_CULTURAL_TRAITS = 5          # 文化特性数量
        
        # 新增的宗教系统参数
        self.ENABLE_RELIGIOUS_INFLUENCE = False  # 是否启用宗教影响系统
        self.RELIGIOUS_INFLUENCE_FACTOR = 0.1    # 宗教影响力因子
        self.RELIGIOUS_CONVERSION_RATE = 0.05    # 宗教转化率
        
    def apply_preset(self, preset_name):
        """应用预设配置"""
        presets = {
            "small_scale": {
                "NUM_CIVILIZATIONS": 2,
                "GRID_SIZE": 10,
                "SIMULATION_CYCLES": 50,
                "TECH_SPILLOVER_EFFECT": 0.5
            },
            "medium_scale": {
                "NUM_CIVILIZATIONS": 4,
                "GRID_SIZE": 20,
                "SIMULATION_CYCLES": 100,
                "TECH_SPILLOVER_EFFECT": 0.3
            },
            "large_scale": {
                "NUM_CIVILIZATIONS": 8,
                "GRID_SIZE": 30,
                "SIMULATION_CYCLES": 200,
                "TECH_SPILLOVER_EFFECT": 0.2
            },
            "high_resources": {
                "RESOURCE_ABUNDANCE": 20.0,
                "RESOURCE_REGENERATION_RATE": 0.03,
                "RESOURCE_CAP": 200.0
            },
            "tech_focus": {
                "TECH_RESEARCH_RATE": 0.2,
                "TECH_SPILLOVER_EFFECT": 0.5,
                "TECH_BONUS_DIMINISHING_RATE": 0.05
            },
            "aggressive": {
                "POPULATION_GROWTH_RATE": 0.02,
                "TERRITORY_EXPANSION_COST": 1.5,
                "RESOURCE_CONSUMPTION_PER_ACTION": 1.5
            },
            "peaceful": {
                "POPULATION_GROWTH_RATE": 0.005,
                "TECH_RESEARCH_RATE": 0.15,
                "RESOURCE_CONSUMPTION_PER_ACTION": 0.8
            },
            "demo": {
                "NUM_CIVILIZATIONS": 4,
                "GRID_SIZE": 15,
                "SIMULATION_CYCLES": 100,
                "TECH_SPILLOVER_EFFECT": 0.3,
                "PRINT_LOGS": True,
                "LOG_INTERVAL": 10,
                "VISUALIZATION_ENABLED": True
            },
            "advanced_evolution": {
                "USE_ADVANCED_EVOLUTION": True,
                "USE_COMPLEX_RESOURCES": True,
                "USE_CULTURAL_INFLUENCE": True,
                "SIMULATION_CYCLES": 150,
                "TECH_SPILLOVER_EFFECT": 0.4,
                "EVOLUTION_LEARNING_RATE": 0.15,
                "CULTURAL_DIFFUSION_RATE": 0.07
            },
            "test": {
                "NUM_CIVILIZATIONS": 3,
                "GRID_SIZE": 10,
                "SIMULATION_CYCLES": 50,
                "TECH_SPILLOVER_EFFECT": 0.5,
                "PRINT_LOGS": False
            },
            "detailed_analysis": {
                "NUM_CIVILIZATIONS": 5,
                "GRID_SIZE": 25,
                "SIMULATION_CYCLES": 200,
                "TECH_SPILLOVER_EFFECT": 0.3,
                "PRINT_LOGS": True,
                "LOG_INTERVAL": 20,
                "VISUALIZATION_ENABLED": True,
                "SAVE_DETAILED_JSON": True,
                "SAVE_STRATEGY_HISTORY": True,
                "ENABLE_RANDOM_EVENTS": True,
                "EVENT_PROBABILITY_MODIFIER": 1.2
            },
            "large_scale": {
                "NUM_CIVILIZATIONS": 10,
                "GRID_SIZE": 50,
                "SIMULATION_CYCLES": 300,
                "TECH_SPILLOVER_EFFECT": 0.2,
                "PRINT_LOGS": False,
                "FAST_MODE": True,
                "ENABLE_RANDOM_EVENTS": True
            }
        }
        
        if preset_name in presets:
            for key, value in presets[preset_name].items():
                setattr(self, key, value)
            print(f"已应用预设配置: {preset_name}")
        else:
            print(f"警告: 预设配置 '{preset_name}' 不存在")
            print(f"可用的预设配置: {', '.join(presets.keys())}")

# 创建全局配置实例
config = SimulationConfig()