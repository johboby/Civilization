#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统 - 统一演示入口

此脚本整合了所有演示功能，提供统一的命令行接口。
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from civsim.config import SimulationConfig, get_preset
from civsim.logger import init_logging, info, error
from multi_agent_simulation import MultiAgentSimulation
from civilization_visualizer import CivilizationVisualizer


def run_basic_demo(args):
    """运行基本模拟演示"""
    info("运行基本模拟演示...")

    # 使用demo预设
    config = get_preset("demo")

    # 应用命令行参数覆盖
    if args.cycles:
        config.simulation_cycles = args.cycles
    if args.num_civs:
        config.num_civilizations = args.num_civs

    # 初始化模拟
    sim = MultiAgentSimulation(config)

    # 运行模拟
    history = sim.run(config.simulation_cycles)

    # 生成可视化
    if not args.no_visualize:
        visualizer = CivilizationVisualizer()
        generate_all_visualizations(visualizer, sim, history)

    # 保存结果
    if args.save:
        save_results(sim, history, args.output_dir)

    info("基本模拟演示完成")


def run_advanced_evolution_demo(args):
    """运行高级演化演示"""
    info("运行高级演化演示...")

    # 使用advanced_evolution预设
    config = get_preset("medium_scale")
    config.use_advanced_evolution = True
    config.use_complex_resources = True
    config.use_cultural_influence = True

    # 应用命令行参数覆盖
    if args.cycles:
        config.simulation_cycles = args.cycles
    if args.num_civs:
        config.num_civilizations = args.num_civs

    # 初始化模拟
    sim = MultiAgentSimulation(config)

    # 运行模拟
    history = sim.run(config.simulation_cycles)

    # 生成可视化
    if not args.no_visualize:
        visualizer = CivilizationVisualizer()
        generate_all_visualizations(visualizer, sim, history)

    # 保存结果
    if args.save:
        save_results(sim, history, args.output_dir)

    info("高级演化演示完成")


def run_new_features_demo(args):
    """运行新功能演示"""
    info("运行新功能演示...")

    # 使用large_scale预设
    config = get_preset("large_scale")
    config.enable_random_events = True

    # 应用命令行参数覆盖
    if args.cycles:
        config.simulation_cycles = args.cycles
    if args.num_civs:
        config.num_civilizations = args.num_civs

    # 初始化模拟
    sim = MultiAgentSimulation(config)

    # 运行模拟
    history = sim.run(config.simulation_cycles)

    # 生成可视化
    if not args.no_visualize:
        visualizer = CivilizationVisualizer()
        generate_all_visualizations(visualizer, sim, history)

    # 保存结果
    if args.save:
        save_results(sim, history, args.output_dir)

    info("新功能演示完成")


def generate_all_visualizations(visualizer, sim, history):
    """生成所有可视化图表"""
    info("生成可视化图表...")

    try:
        # 策略热力图
        visualizer.plot_strategy_heatmap(history)
        info("  ✓ 策略热力图")
    except Exception as e:
        error(f"  ✗ 策略热力图生成失败: {e}")

    try:
        # 演化曲线
        visualizer.plot_evolution_curve(history)
        info("  ✓ 演化曲线")
    except Exception as e:
        error(f"  ✗ 演化曲线生成失败: {e}")

    try:
        # 科技进展
        visualizer.plot_technology_progress(sim.technology_history)
        info("  ✓ 科技进展")
    except Exception as e:
        error(f"  ✗ 科技进展生成失败: {e}")


def save_results(sim, history, output_dir):
    """保存模拟结果"""
    info("保存模拟结果...")

    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 保存NPZ文件
        results_file = os.path.join(output_dir, "demo_results.npz")
        sim.save_results(results_file)
        info(f"  ✓ 模拟结果已保存到 {results_file}")
    except Exception as e:
        error(f"  ✗ 保存模拟结果失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="文明演化模拟系统 - 统一演示入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行基本演示
  python run_demo.py basic

  # 运行高级演化演示
  python run_demo.py advanced

  # 运行新功能演示
  python run_demo.py new-features

  # 自定义参数
  python run_demo.py basic --cycles 200 --num-civs 8 --save --output-dir my_results
        """
    )

    # 子命令
    subparsers = parser.add_subparsers(dest='demo_type', help='演示类型')

    # 基本演示
    basic_parser = subparsers.add_parser('basic', help='基本模拟演示')
    basic_parser.add_argument('--cycles', type=int, help='模拟周期数')
    basic_parser.add_argument('--num-civs', type=int, help='文明数量')
    basic_parser.add_argument('--no-visualize', action='store_true', help='不生成可视化')
    basic_parser.add_argument('--save', action='store_true', help='保存结果')
    basic_parser.add_argument('--output-dir', type=str, default='demo_results', help='输出目录')

    # 高级演化演示
    advanced_parser = subparsers.add_parser('advanced', help='高级演化演示')
    advanced_parser.add_argument('--cycles', type=int, help='模拟周期数')
    advanced_parser.add_argument('--num-civs', type=int, help='文明数量')
    advanced_parser.add_argument('--no-visualize', action='store_true', help='不生成可视化')
    advanced_parser.add_argument('--save', action='store_true', help='保存结果')
    advanced_parser.add_argument('--output-dir', type=str, default='demo_results', help='输出目录')

    # 新功能演示
    new_features_parser = subparsers.add_parser('new-features', help='新功能演示')
    new_features_parser.add_argument('--cycles', type=int, help='模拟周期数')
    new_features_parser.add_argument('--num-civs', type=int, help='文明数量')
    new_features_parser.add_argument('--no-visualize', action='store_true', help='不生成可视化')
    new_features_parser.add_argument('--save', action='store_true', help='保存结果')
    new_features_parser.add_argument('--output-dir', type=str, default='demo_results', help='输出目录')

    # 全局参数
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别')
    parser.add_argument('--log-file', type=str, help='日志文件路径')

    args = parser.parse_args()

    # 初始化日志
    import logging
    log_level = getattr(logging, args.log_level)
    init_logging(level=log_level, log_file=args.log_file)

    # 根据演示类型运行对应的函数
    if args.demo_type == 'basic':
        run_basic_demo(args)
    elif args.demo_type == 'advanced':
        run_advanced_evolution_demo(args)
    elif args.demo_type == 'new-features':
        run_new_features_demo(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        info("\n演示被用户中断")
        sys.exit(0)
    except Exception as e:
        error(f"演示过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
