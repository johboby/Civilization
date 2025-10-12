#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统 - 示例配置文件

此文件展示了如何自定义模拟参数，您可以根据需要修改这些参数，
然后在运行模拟时通过 --config 参数指定此文件。

使用方法:
python simulation_cli.py --config example_config.py
"""

class SimulationConfig:
    """模拟配置类，包含所有可自定义的参数"""
    # ====================== 基础模拟参数 ======================
    # 文明数量
    NUM_CIVILIZATIONS = 6
    # 网格大小 (影响模拟世界的大小)
    GRID_SIZE = 250
    # 模拟周期数
    SIMULATION_CYCLES = 300
    # 是否打印详细日志
    PRINT_LOGS = True
    # 日志打印间隔周期数
    LOG_INTERVAL = 50
    # 是否在每步后可视化结果
    VISUALIZE_EACH_STEP = False
    # 快速模式 (减少日志和可视化以提高性能)
    FAST_MODE = False
    
    # ====================== 资源系统参数 ======================
    # 初始资源分布随机度 (0-1)
    RESOURCE_DISTRIBUTION_VARIANCE = 0.4
    # 初始资源总量
    INITIAL_RESOURCE_AMOUNT = 200.0
    # 资源消耗率
    RESOURCE_CONSUMPTION_RATE = 0.015
    # 资源再生率
    RESOURCE_REGENERATION_RATE = 0.008
    # 最大资源量
    MAX_RESOURCE_AMOUNT = 150.0
    # 最小资源量
    MIN_RESOURCE_AMOUNT = 10.0
    # 资源发现概率
    RESOURCE_DISCOVERY_PROBABILITY = 0.02
    # 资源发现量系数
    RESOURCE_DISCOVERY_AMOUNT = 20.0
    
    # ====================== 科技研发参数 ======================
    # 基础研发速度
    BASE_RESEARCH_SPEED = 0.02
    # 科技研发资源效率
    RESEARCH_RESOURCE_EFFICIENCY = 0.005
    # 科技研发难度系数
    RESEARCH_DIFFICULTY_FACTOR = 1.5
    # 科技树扩展因子
    TECH_TREE_EXPANSION_FACTOR = 1.2
    # 科技溢出效应 (相邻文明共享科技进度)
    TECH_SPILLOVER_EFFECT = 0.05
    # 科技遗忘率
    TECH_FORGETTING_RATE = 0.001
    
    # ====================== 领土扩张参数 ======================
    # 基础扩张概率
    BASE_EXPANSION_PROBABILITY = 0.05
    # 扩张失败概率
    EXPANSION_FAILURE_RATE = 0.2
    # 扩张成本系数
    EXPANSION_COST_FACTOR = 0.1
    # 初始领土大小
    INITIAL_TERRITORY_SIZE = 4
    # 领土价值系数
    TERRITORY_VALUE_FACTOR = 0.02
    # 地形影响系数 (影响不同区域的扩张难度)
    TERRAIN_INFLUENCE_FACTOR = 0.3
    
    # ====================== 文明属性参数 ======================
    # 属性随机初始化范围 (0-1)
    ATTRIBUTE_INITIAL_RANGE = 0.3
    # 属性最大衰减率
    MAX_ATTRIBUTE_DECAY = 0.005
    # 属性提升成本系数
    ATTRIBUTE_IMPROVEMENT_COST = 0.1
    # 属性间影响系数
    INTER_ATTRIBUTE_INFLUENCE = 0.1
    # 属性最大值
    MAX_ATTRIBUTE_VALUE = 5.0
    # 属性最小值
    MIN_ATTRIBUTE_VALUE = 0.1
    # 初始人口
    INITIAL_POPULATION = 100
    # 人口增长率
    POPULATION_GROWTH_RATE = 0.03
    # 人口密度上限
    MAX_POPULATION_DENSITY = 5.0
    
    # ====================== 策略决策参数 ======================
    # 策略转换平滑因子
    STRATEGY_TRANSITION_SMOOTHNESS = 0.1
    # 策略探索概率
    STRATEGY_EXPLORATION_PROBABILITY = 0.05
    # 策略评估周期
    STRATEGY_EVALUATION_PERIOD = 10
    # 策略调整系数
    STRATEGY_ADJUSTMENT_FACTOR = 0.2
    # 记忆窗口大小 (影响历史经验的权重)
    MEMORY_WINDOW_SIZE = 10
    # 启用元认知策略调整
    ENABLE_METACOGNITION = True
    # 元认知学习率
    METACOGNITION_LEARNING_RATE = 0.05
    # 扩张策略权重
    EXPANSION_STRATEGY_WEIGHT = 1.0
    # 防御策略权重
    DEFENSE_STRATEGY_WEIGHT = 1.0
    # 贸易策略权重
    TRADE_STRATEGY_WEIGHT = 1.0
    # 研发策略权重
    RESEARCH_STRATEGY_WEIGHT = 1.0
    
    # ====================== 文明交互参数 ======================
    # 基础攻击概率
    BASE_ATTACK_PROBABILITY = 0.03
    # 基础防御成功率
    BASE_DEFENSE_SUCCESS_RATE = 0.5
    # 贸易成功率
    TRADE_SUCCESS_RATE = 0.7
    # 文化交流概率
    CULTURAL_EXCHANGE_PROBABILITY = 0.05
    # 盟友关系阈值
    ALLIANCE_THRESHOLD = 0.6
    # 敌对关系阈值
    ENEMY_THRESHOLD = -0.4
    # 关系衰减率
    RELATION_DECAY_RATE = 0.01
    # 关系更新率
    RELATION_UPDATE_RATE = 0.1
    
    # ====================== 结果输出参数 ======================
    # 是否保存可视化结果
    SAVE_VISUALIZATION = True
    # 可视化保存间隔
    VISUALIZATION_INTERVAL = 100
    # 结果保存目录
    RESULTS_DIR = "simulation_results"
    # 是否保存详细历史数据
    SAVE_DETAILED_HISTORY = True
    # 图表DPI
    PLOT_DPI = 300
    # 图表保存格式
    PLOT_FORMAT = 'png'
    # 保存CSV数据
    SAVE_CSV_DATA = True
    # 保存JSON数据
    SAVE_JSON_DATA = True
    # 生成总结报告
    GENERATE_REPORT = True
    
    # ====================== 高级参数 ======================
    # 随机种子 (设置为固定值以复现结果)
    RANDOM_SEED = None  # None 表示每次运行使用不同种子
    # 随机事件概率
    RANDOM_EVENT_PROBABILITY = 0.02
    # 自然灾害概率
    NATURAL_DISASTER_PROBABILITY = 0.01
    # 新技术发现概率
    NEW_TECH_DISCOVERY_PROBABILITY = 0.005
    # 模拟加速因子 (大于1时加快模拟速度)
    SIMULATION_SPEED_FACTOR = 1.0


# 方便导入的配置实例
simulation_config = SimulationConfig()

# 示例配置方案
config_presets = {
    # 快速演示配置
    'demo': {
        'NUM_CIVILIZATIONS': 4,
        'SIMULATION_CYCLES': 100,
        'GRID_SIZE': 150,
        'PRINT_LOGS': True,
        'LOG_INTERVAL': 20
    },
    # 标准配置
    'standard': {
        'NUM_CIVILIZATIONS': 6,
        'SIMULATION_CYCLES': 300,
        'GRID_SIZE': 250,
        'PRINT_LOGS': True,
        'LOG_INTERVAL': 50
    },
    # 大规模模拟配置
    'large_scale': {
        'NUM_CIVILIZATIONS': 10,
        'SIMULATION_CYCLES': 500,
        'GRID_SIZE': 400,
        'PRINT_LOGS': True,
        'LOG_INTERVAL': 100,
        'FAST_MODE': True
    },
    # 资源稀缺配置
    'resource_scarcity': {
        'INITIAL_RESOURCE_AMOUNT': 100.0,
        'RESOURCE_REGENERATION_RATE': 0.003,
        'RESOURCE_CONSUMPTION_RATE': 0.02
    },
    # 科技优先配置
    'tech_focus': {
        'BASE_RESEARCH_SPEED': 0.03,
        'RESEARCH_RESOURCE_EFFICIENCY': 0.008,
        'TECH_SPILLOVER_EFFECT': 0.1
    }
}

# 使用预设配置的辅助函数
def apply_preset(preset_name):
    """\应用预设配置方案
    
    Args:
        preset_name: 预设名称，可选值: 'demo', 'standard', 'large_scale', 'resource_scarcity', 'tech_focus'
    
    Returns:
        更新后的配置实例
    """
    if preset_name in config_presets:
        preset = config_presets[preset_name]
        for key, value in preset.items():
            if hasattr(SimulationConfig, key):
                setattr(simulation_config, key, value)
        print(f"应用预设配置: {preset_name}")
    else:
        print(f"警告: 未知的预设配置 '{preset_name}'")
    return simulation_config

# 示例用法:
# from example_config import simulation_config, apply_preset
# config = apply_preset('demo')
    # 多线程模拟
    MULTITHREADING = False
    # 线程数 (多线程模式下)
    NUM_THREADS = 4
    # 性能优化级别 (0-3)
    PERFORMANCE_OPTIMIZATION = 1


# 创建配置实例供导入使用
example_config = SimulationConfig()


def print_config_summary():
    """打印配置摘要"""
    print("==== 文明演化模拟配置摘要 ====")
    print(f"文明数量: {example_config.NUM_CIVILIZATIONS}")
    print(f"网格大小: {example_config.GRID_SIZE}")
    print(f"模拟周期: {example_config.SIMULATION_CYCLES}")
    print(f"是否打印日志: {example_config.PRINT_LOGS} (间隔: {example_config.LOG_INTERVAL}周期)")
    print(f"资源分布随机度: {example_config.RESOURCE_DISTRIBUTION_VARIANCE}")
    print(f"基础研发速度: {example_config.BASE_RESEARCH_SPEED}")
    print(f"基础扩张概率: {example_config.BASE_EXPANSION_PROBABILITY}")
    print(f"随机种子: {'随机' if example_config.RANDOM_SEED is None else example_config.RANDOM_SEED}")
    print(f"结果保存目录: {example_config.RESULTS_DIR}")
    print("============================")


if __name__ == "__main__":
    # 如果直接运行此文件，打印配置摘要
    print_config_summary()