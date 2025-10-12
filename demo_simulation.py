#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统 - 演示脚本

这个脚本提供了一个简单的示例，展示如何使用文明演化模拟系统的核心功能。
它运行一个简化的模拟，并生成一些基本的可视化结果。
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multi_agent_simulation import MultiAgentSimulation
from simulation_config import config
from civilization_visualizer import CivilizationVisualizer


def setup_demo_environment():
    """设置演示环境"""
    # 创建演示结果目录
    demo_dir = "demo_results"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    # 保存原始配置
    original_config = {
        'NUM_CIVILIZATIONS': config.NUM_CIVILIZATIONS,
        'GRID_SIZE': config.GRID_SIZE,
        'SIMULATION_CYCLES': config.SIMULATION_CYCLES,
        'PRINT_LOGS': config.PRINT_LOGS,
        'LOG_INTERVAL': config.LOG_INTERVAL
    }
    
    # 设置演示配置参数
    config.NUM_CIVILIZATIONS = 4  # 4个文明便于可视化
    config.GRID_SIZE = 100        # 适中的网格大小
    config.SIMULATION_CYCLES = 100 # 100个周期足够展示演化过程
    config.PRINT_LOGS = True
    config.LOG_INTERVAL = 10
    
    print("==== 文明演化模拟演示 ====")
    print(f"配置: {config.NUM_CIVILIZATIONS}个文明, {config.GRID_SIZE}网格, {config.SIMULATION_CYCLES}周期")
    
    return demo_dir, original_config


def run_demo_simulation(demo_dir):
    """运行演示模拟"""
    print("\n1. 初始化多智能体模拟...")
    sim = MultiAgentSimulation(config)
    print(f"   ✓ 成功初始化 {len(sim.agents)} 个文明")
    
    print("\n2. 运行模拟...")
    history = sim.run(config.SIMULATION_CYCLES)
    print("   ✓ 模拟运行完成")
    
    return sim, history


def generate_demo_visualizations(sim, history, demo_dir):
    """生成演示可视化结果"""
    print("\n3. 初始化可视化工具...")
    visualizer = CivilizationVisualizer()
    print("   ✓ 成功初始化可视化工具")
    
    print("\n4. 生成可视化结果...")
    
    # 1. 策略热力图
    try:
        plt.figure(figsize=(10, 8))
        visualizer.plot_strategy_heatmap(history)
        plt.title('文明策略分布热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(demo_dir, 'strategy_heatmap.png'), dpi=300)
        print("   ✓ 策略热力图已保存")
    except Exception as e:
        print(f"   ✗ 生成策略热力图失败: {e}")
    
    # 2. 演化趋势图
    try:
        plt.figure(figsize=(12, 8))
        visualizer.plot_evolution_curve(history)
        plt.title('文明属性演化趋势')
        plt.tight_layout()
        plt.savefig(os.path.join(demo_dir, 'evolution_curve.png'), dpi=300)
        print("   ✓ 演化趋势图已保存")
    except Exception as e:
        print(f"   ✗ 生成演化趋势图失败: {e}")
    
    # 3. 科技进展图
    try:
        plt.figure(figsize=(10, 8))
        visualizer.plot_technology_progress(sim.agents)
        plt.title('各文明科技进展对比')
        plt.tight_layout()
        plt.savefig(os.path.join(demo_dir, 'technology_progress.png'), dpi=300)
        print("   ✓ 科技进展图已保存")
    except Exception as e:
        print(f"   ✗ 生成科技进展图失败: {e}")
    
    # 4. 科技树比较图
    try:
        plt.figure(figsize=(12, 10))
        visualizer.plot_tech_tree_comparison(sim.agents)
        plt.title('文明科技树发展对比')
        plt.tight_layout()
        plt.savefig(os.path.join(demo_dir, 'tech_tree_comparison.png'), dpi=300)
        print("   ✓ 科技树比较图已保存")
    except Exception as e:
        print(f"   ✗ 生成科技树比较图失败: {e}")
    
    # 5. 属性雷达图
    try:
        plt.figure(figsize=(12, 8))
        visualizer.plot_radar_chart(sim.agents)
        plt.title('文明属性雷达图对比')
        plt.tight_layout()
        plt.savefig(os.path.join(demo_dir, 'attribute_radar.png'), dpi=300)
        print("   ✓ 属性雷达图已保存")
    except Exception as e:
        print(f"   ✗ 生成属性雷达图失败: {e}")
    
    # 6. 关系网络图
    try:
        plt.figure(figsize=(10, 10))
        visualizer.plot_relationships_network(sim.agents)
        plt.title('文明关系网络图')
        plt.tight_layout()
        plt.savefig(os.path.join(demo_dir, 'relationships_network.png'), dpi=300)
        print("   ✓ 关系网络图已保存")
    except Exception as e:
        print(f"   ✗ 生成关系网络图失败: {e}")
    
    plt.close('all')  # 关闭所有图表以释放资源


def save_demo_data(sim, demo_dir):
    """保存演示数据"""
    print("\n5. 保存演示数据...")
    
    # 保存结果文件
    try:
        results_file = os.path.join(demo_dir, 'demo_simulation_results.npz')
        sim.save_results(results_file)
        print(f"   ✓ 模拟结果已保存到 {results_file}")
    except Exception as e:
        print(f"   ✗ 保存模拟结果失败: {e}")
    
    # 保存CSV数据
    try:
        visualizer = CivilizationVisualizer()
        csv_file = os.path.join(demo_dir, 'demo_civilization_data.csv')
        visualizer.save_to_csv(np.array(sim.history["strategy"]), csv_file)
        print(f"   ✓ CSV数据已保存到 {csv_file}")
    except Exception as e:
        print(f"   ✗ 保存CSV数据失败: {e}")
    
    # 保存属性历史
    try:
        attr_file = os.path.join(demo_dir, 'demo_attribute_history.csv')
        visualizer.save_attribute_history(
            np.array(sim.attribute_history),
            sim.attribute_names,
            attr_file
        )
        print(f"   ✓ 属性历史已保存到 {attr_file}")
    except Exception as e:
        print(f"   ✗ 保存属性历史失败: {e}")
    
    # 保存科技历史
    try:
        tech_file = os.path.join(demo_dir, 'demo_technology_history.json')
        visualizer.save_technology_data(sim.technology_history, tech_file)
        print(f"   ✓ 科技历史已保存到 {tech_file}")
    except Exception as e:
        print(f"   ✗ 保存科技历史失败: {e}")
    
    # 生成总结报告
    try:
        report_file = os.path.join(demo_dir, 'demo_simulation_report.md')
        visualizer.create_summary_report(sim.agents, sim.history, report_file)
        print(f"   ✓ 总结报告已保存到 {report_file}")
    except Exception as e:
        print(f"   ✗ 生成总结报告失败: {e}")


def print_demo_summary(demo_dir):
    """打印演示总结"""
    print("\n==== 演示完成 ====")
    print(f"所有演示结果已保存到 {demo_dir} 目录")
    print("\n您可以查看以下文件：")
    print("- 可视化图表 (.png 文件)")
    print("- 模拟数据 (.npz 和 .csv 文件)")
    print("- 科技历史数据 (.json 文件)")
    print("- 总结报告 (.md 文件)")
    
    print("\n接下来您可以：")
    print("1. 使用命令行界面运行更复杂的模拟：")
    print("   python simulation_cli.py --interactive")
    print("2. 通过自定义配置文件调整模拟参数：")
    print("   python simulation_cli.py --config example_config.py")
    print("3. 使用批处理文件快速启动：")
    print("   双击 run_simulation.bat")


def restore_config(original_config):
    """恢复原始配置"""
    for key, value in original_config.items():
        setattr(config, key, value)


def main():
    """主函数"""
    try:
        # 设置演示环境
        demo_dir, original_config = setup_demo_environment()
        
        # 运行演示模拟
        sim, history = run_demo_simulation(demo_dir)
        
        # 生成可视化结果
        generate_demo_visualizations(sim, history, demo_dir)
        
        # 保存演示数据
        save_demo_data(sim, demo_dir)
        
        # 打印演示总结
        print_demo_summary(demo_dir)
        
    except KeyboardInterrupt:
        print("\n演示被用户中断。")
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复原始配置
        restore_config(original_config)
        print("\n演示结束。")


if __name__ == "__main__":
    main()