#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统命令行界面
提供交互式控制和参数配置功能
"""
import os
import sys
import argparse
import json
import numpy as np
import importlib.util
from multi_agent_simulation import MultiAgentSimulation
from civsim.config import SimulationConfig
from civilization_visualizer import CivilizationVisualizer


class SimulationCLI:
    """文明演化模拟系统命令行界面"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.simulation = None
        self.visualizer = None
        self.output_dir = "results"
        self.config = SimulationConfig()
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _create_parser(self):
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(description="文明演化模拟系统")
        
        # 基础参数组
        base_group = parser.add_argument_group('基础参数')
        base_group.add_argument('--cycles', type=int, default=self.config.SIMULATION_CYCLES, 
                              help=f'模拟周期数 (默认: {self.config.SIMULATION_CYCLES})')
        base_group.add_argument('--num-civs', type=int, default=self.config.NUM_CIVILIZATIONS, 
                              help=f'文明数量 (默认: {self.config.NUM_CIVILIZATIONS})')
        base_group.add_argument('--grid-size', type=int, default=self.config.GRID_SIZE, 
                              help=f'网格大小 (默认: {self.config.GRID_SIZE})')
        base_group.add_argument('--output-dir', type=str, default='results', 
                              help='输出结果目录 (默认: results)')
        
        # 配置选项
        base_group.add_argument('--seed', type=int, default=None, 
                              help='随机种子 (默认: 随机)')
        base_group.add_argument('--preset', type=str, choices=['demo', 'standard', 'large_scale', 'resource_scarcity', 'tech_focus', 'advanced_evolution'],
                              help='使用预设配置方案')
        
        # 高级参数组
        advanced_group = parser.add_argument_group('高级参数')
        advanced_group.add_argument('--config', type=str, help='自定义配置文件路径')
        advanced_group.add_argument('--debug', action='store_true', help='启用调试模式')
        advanced_group.add_argument('--no-visualization', action='store_true', help='禁用可视化输出')
        advanced_group.add_argument('--save-only', action='store_true', help='只保存结果不显示图表')
        advanced_group.add_argument('--verbose', action='store_true', help='启用详细日志')
        advanced_group.add_argument('--quiet', action='store_true', help='静默模式，只输出必要信息')
        # 高级演化参数
        advanced_group.add_argument('--use-advanced-evolution', action='store_true', help='启用高级演化引擎')
        advanced_group.add_argument('--use-complex-resources', action='store_true', help='启用复杂资源管理系统')
        advanced_group.add_argument('--use-cultural-influence', action='store_true', help='启用文化影响系统')
        advanced_group.add_argument('--evolution-learning-rate', type=float, help='演化学习率')
        advanced_group.add_argument('--cultural-diffusion-rate', type=float, help='文化扩散率')
        
        # 模式选择
        mode_group = parser.add_argument_group('运行模式')
        mode_group.add_argument('--interactive', action='store_true', help='以交互式模式运行')
        mode_group.add_argument('--batch', action='store_true', help='以批处理模式运行')
        mode_group.add_argument('--resume', type=str, help='从保存的结果文件恢复模拟')
        mode_group.add_argument('--fast-mode', action='store_true', help='快速模式，减少日志和可视化')
        
        return parser
    
    def run(self):
        """运行命令行界面"""
        args = self.parser.parse_args()
        
        # 加载自定义配置文件
        if args.config:
            self._load_custom_config(args.config)
        
        # 如果指定了预设，应用预设配置
        if args.preset:
            self._apply_preset(args.preset)
        
        # 设置随机种子
        if args.seed is not None:
            np.random.seed(args.seed)
            self.config.RANDOM_SEED = args.seed
        
        # 设置命令行参数（覆盖配置文件中的设置）
        if hasattr(self.config, 'SIMULATION_CYCLES'):
            self.config.SIMULATION_CYCLES = args.cycles
        if hasattr(self.config, 'NUM_CIVILIZATIONS'):
            self.config.NUM_CIVILIZATIONS = args.num_civs
        if hasattr(self.config, 'GRID_SIZE'):
            self.config.GRID_SIZE = args.grid_size
        
        # 处理高级演化参数覆盖
        if args.use_advanced_evolution:
            setattr(self.config, 'USE_ADVANCED_EVOLUTION', True)
        if args.use_complex_resources:
            setattr(self.config, 'USE_COMPLEX_RESOURCES', True)
        if args.use_cultural_influence:
            setattr(self.config, 'USE_CULTURAL_INFLUENCE', True)
        if args.evolution_learning_rate is not None:
            setattr(self.config, 'EVOLUTION_LEARNING_RATE', args.evolution_learning_rate)
        if args.cultural_diffusion_rate is not None:
            setattr(self.config, 'CULTURAL_DIFFUSION_RATE', args.cultural_diffusion_rate)
        
        # 更新输出目录
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 根据运行模式调整参数
        if args.fast_mode or args.quiet:
            self.config.PRINT_LOGS = not args.quiet
            self.config.VISUALIZE_EACH_STEP = False
        elif args.verbose:
            self.config.PRINT_LOGS = True
            if hasattr(self.config, 'LOG_INTERVAL'):
                self.config.LOG_INTERVAL = 10
        
        # 根据参数选择运行模式
        if args.interactive:
            self._interactive_mode(args)
        elif args.resume:
            self._resume_simulation(args)
        else:
            self._normal_mode(args)
    
    def _normal_mode(self, args):
        """正常运行模式"""
        print("==== 文明演化模拟系统 ====")
        print(f"配置: 文明数量={args.num_civs}, 周期数={args.cycles}, 网格大小={args.grid_size}")
        
        # 加载自定义配置
        if args.config:
            self._load_custom_config(args.config)
        
        # 更新基本配置
        config.NUM_CIVILIZATIONS = args.num_civs
        config.GRID_SIZE = args.grid_size
        config.SIMULATION_CYCLES = args.cycles
        config.PRINT_LOGS = args.debug or config.PRINT_LOGS
        
        # 初始化模拟
        self.simulation = MultiAgentSimulation(config)
        
        # 运行模拟
        history = self.simulation.run(args.cycles)
        
        # 处理结果
        self._process_results(history, args)
    
    def _interactive_mode(self, args):
        """交互式运行模式"""
        print("==== 文明演化模拟系统 - 交互式模式 ====")
        print("输入 'help' 查看可用命令")
        
        # 初始化模拟
        if args.config:
            self._load_custom_config(args.config)
        
        config.NUM_CIVILIZATIONS = args.num_civs
        config.GRID_SIZE = args.grid_size
        config.PRINT_LOGS = True  # 交互式模式下默认开启日志
        
        self.simulation = MultiAgentSimulation(config)
        self.visualizer = CivilizationVisualizer()
        
        # 主循环
        cycle = 0
        running = True
        while running:
            cmd = input(f"[周期 {cycle}]> ").strip().lower()
            
            if cmd == 'help':
                self._show_help()
            elif cmd == 'run':
                self._run_interactive(args)
                cycle = len(self.simulation.history["resources"])
            elif cmd == 'run n':
                try:
                    n = int(input("输入要运行的周期数: "))
                    self.simulation.run(n)
                    cycle = len(self.simulation.history["resources"])
                except ValueError:
                    print("请输入有效的数字")
            elif cmd == 'visualize':
                self._visualize_current_state()
            elif cmd == 'save':
                self._save_current_state()
            elif cmd == 'config':
                self._show_config()
            elif cmd == 'status':
                self._show_status()
            elif cmd == 'exit' or cmd == 'quit':
                running = False
            else:
                print(f"未知命令: {cmd}")
    
    def _resume_simulation(self, args):
        """从保存的结果恢复模拟"""
        print(f"==== 从 {args.resume} 恢复模拟 ====")
        
        try:
            data = np.load(args.resume, allow_pickle=True)
            
            # 初始化新的模拟
            self.simulation = MultiAgentSimulation(config)
            
            # 加载历史数据（这里只是示意，实际加载需要更复杂的逻辑）
            print(f"已加载数据: {list(data.keys())}")
            
            # 继续模拟
            self.simulation.run(args.cycles)
            
            # 处理结果
            history = np.array(self.simulation.history["strategy"])
            self._process_results(history, args)
        except Exception as e:
            print(f"恢复模拟失败: {e}")
            sys.exit(1)
    
    def _process_results(self, history, args):
        """处理模拟结果"""
        if len(history) == 0:
            print("没有生成历史数据，无法可视化")
            return
        
        # 初始化可视化器
        self.visualizer = CivilizationVisualizer()
        
        # 生成可视化图表
        if not args.no_visualization:
            print("生成可视化图表...")
            
            # 创建所有支持的可视化
            viz_functions = [
                ("策略热力图", self.visualizer.plot_strategy_heatmap, [history]),
                ("演化趋势图", self.visualizer.plot_evolution_curve, [history]),
                ("科技进展图", self.visualizer.plot_technology_progress, [self.simulation.agents]),
                ("科技树比较图", self.visualizer.plot_tech_tree_comparison, [self.simulation.agents]),
                ("属性比较图", self.visualizer.plot_attribute_comparison, [self.simulation.agents]),
                ("属性雷达图", self.visualizer.plot_radar_chart, [self.simulation.agents]),
                ("关系网络图", self.visualizer.plot_relationships_network, [self.simulation.agents])
            ]
            
            for name, func, args_list in viz_functions:
                try:
                    print(f"  - {name}")
                    func(*args_list)
                except Exception as e:
                    print(f"  ! {name} 生成失败: {e}")
        
        # 保存结果
        print("保存结果数据...")
        
        # 保存CSV数据
        if hasattr(self.visualizer, 'save_to_csv'):
            csv_path = os.path.join(self.output_dir, "simulation_results.csv")
            self.visualizer.save_to_csv(np.array(self.simulation.history["strategy"]), csv_path)
            print(f"  - 策略数据已保存到 {csv_path}")
        
        if hasattr(self.visualizer, 'save_attribute_history'):
            attr_path = os.path.join(self.output_dir, "attribute_history.csv")
            self.visualizer.save_attribute_history(
                np.array(self.simulation.attribute_history), 
                self.simulation.attribute_names, 
                attr_path
            )
            print(f"  - 属性历史已保存到 {attr_path}")
        
        if hasattr(self.visualizer, 'save_technology_data'):
            tech_path = os.path.join(self.output_dir, "technology_data.json")
            self.visualizer.save_technology_data(self.simulation.technology_history, tech_path)
            print(f"  - 科技数据已保存到 {tech_path}")
        
        # 创建总结报告
        if hasattr(self.visualizer, 'create_summary_report'):
            report_path = os.path.join(self.output_dir, "simulation_report.md")
            self.visualizer.create_summary_report(
                self.simulation.agents, 
                self.simulation.history, 
                report_path
            )
            print(f"  - 总结报告已保存到 {report_path}")
        
        # 显示图表
        if not args.no_visualization and not args.save_only:
            print("显示可视化图表...")
            self.visualizer.show()
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
可用命令:
  help          - 显示此帮助信息
  run           - 运行一个周期
  run n         - 运行指定数量的周期
  visualize     - 可视化当前状态
  save          - 保存当前状态
  config        - 显示当前配置
  status        - 显示当前模拟状态
  exit/quit     - 退出程序
        """
        print(help_text)
    
    def _run_interactive(self, args):
        """在交互式模式下运行一个周期"""
        self.simulation.step()
        print(f"完成周期 {len(self.simulation.history['resources'])}")
    
    def _visualize_current_state(self):
        """可视化当前模拟状态"""
        if self.visualizer is None:
            self.visualizer = CivilizationVisualizer()
        
        history = np.array(self.simulation.history["strategy"])
        if len(history) > 0:
            self.visualizer.plot_strategy_heatmap(history)
            self.visualizer.plot_evolution_curve(history)
            self.visualizer.show(block=False)
    
    def _save_current_state(self):
        """保存当前模拟状态"""
        cycle = len(self.simulation.history['resources'])
        filename = f"simulation_state_{cycle}.npz"
        filepath = os.path.join(self.output_dir, filename)
        
        np.savez(filepath, 
                 strategies=np.array(self.simulation.history["strategy"], dtype=object),
                 resources=np.array(self.simulation.history["resources"]),
                 strength=np.array(self.simulation.history["strength"]),
                 technology=np.array(self.simulation.history["technology"]),
                 attribute_history=np.array(self.simulation.attribute_history),
                 technology_history=np.array(self.simulation.technology_history, dtype=object))
        
        print(f"当前状态已保存到 {filepath}")
        
        # 保存CSV格式的数据
        csv_filename = os.path.join(self.output_dir, f"civilization_data_{cycle}.csv")
        if hasattr(self.visualizer, 'save_to_csv'):
            self.visualizer.save_to_csv(np.array(self.simulation.history["strategy"]), csv_filename)
            print(f"CSV数据已保存到: {csv_filename}")
    
    def _show_config(self):
        """显示当前配置"""
        print("==== 当前配置 ====")
        for key, value in vars(config).items():
            if not key.startswith('__'):
                print(f"{key}: {value}")
    
    def _show_status(self):
        """显示当前模拟状态"""
        print("==== 当前模拟状态 ====")
        print(f"当前周期: {len(self.simulation.history['resources'])}")
        
        # 显示各文明的基本信息
        for i, agent in enumerate(self.simulation.agents):
            print(f"\n文明 {i}:")
            print(f"  资源: {round(agent.resources, 2)}")
            print(f"  力量: {round(agent.strength, 2)}")
            print(f"  人口: {round(agent.population, 2)}")
            print(f"  领土: {len(agent.territory)}")
            print(f"  盟友: {len(agent.allies)}")
            print(f"  敌人: {len(agent.enemies)}")
            print(f"  科技等级: {sum(agent.technology.values())}")
            
            # 显示正在研发的科技
            if agent.current_research:
                progress = (agent.research_progress / agent.research_cost * 100) if agent.research_cost > 0 else 0
                print(f"  正在研发: {agent.current_research} ({progress:.1f}%)")
    
    def _load_custom_config(self, config_path):
        """加载自定义配置文件"""
        if not os.path.exists(config_path):
            print(f"警告: 配置文件 {config_path} 不存在，将使用默认配置。")
            return False
        
        try:
            # 使用importlib加载配置文件
            spec = importlib.util.spec_from_file_location("custom_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules["custom_config"] = config_module
            spec.loader.exec_module(config_module)
            
            # 更新配置
            if hasattr(config_module, 'config'):
                # 如果配置是对象，转换为字典
                if hasattr(config_module.config, '__dict__'):
                    config_dict = {k: v for k, v in config_module.config.__dict__.items() \
                                 if not k.startswith('__') and not callable(v)}
                    for key, value in config_dict.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                elif isinstance(config_module.config, dict):
                    for key, value in config_module.config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                print(f"已加载自定义配置: {config_path}")
            elif hasattr(config_module, 'example_config'):
                # 兼容旧版配置文件格式
                custom_config = config_module.example_config
                # 应用自定义配置
                for attr_name in dir(custom_config):
                    if not attr_name.startswith('__'):
                        if hasattr(self.config, attr_name):
                            setattr(self.config, attr_name, getattr(custom_config, attr_name))
                print(f"已加载自定义配置: {config_path}")
            else:
                print(f"警告: 在配置文件 {config_path} 中未找到有效的配置实例。")
                return False
            
            return True
        except Exception as e:
            print(f"加载自定义配置失败: {e}")
            return False
    
    def _apply_preset(self, preset_name):
        """应用预设配置方案"""
        try:
            # 尝试从example_config导入apply_preset函数
            from example_config import apply_preset
            preset_config = apply_preset(preset_name)
            
            # 更新配置
            if hasattr(preset_config, '__dict__'):
                config_dict = {k: v for k, v in preset_config.__dict__.items() \
                             if not k.startswith('__') and not callable(v)}
                for key, value in config_dict.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            print(f"已应用预设配置: {preset_name}")
        except Exception as e:
            print(f"应用预设配置失败: {e}")
            # 手动应用预设（以防导入失败）
            self._manual_apply_preset(preset_name)
    
    def _manual_apply_preset(self, preset_name):
        """手动应用预设配置（备用方法）"""
        presets = {
            'demo': {
                'NUM_CIVILIZATIONS': 4,
                'SIMULATION_CYCLES': 100,
                'GRID_SIZE': 150,
                'PRINT_LOGS': True,
                'LOG_INTERVAL': 20
            },
            'standard': {
                'NUM_CIVILIZATIONS': 6,
                'SIMULATION_CYCLES': 300,
                'GRID_SIZE': 250,
                'PRINT_LOGS': True,
                'LOG_INTERVAL': 50
            },
            'large_scale': {
                'NUM_CIVILIZATIONS': 10,
                'SIMULATION_CYCLES': 500,
                'GRID_SIZE': 400,
                'PRINT_LOGS': True,
                'LOG_INTERVAL': 100,
                'FAST_MODE': True
            },
            'resource_scarcity': {
                'INITIAL_RESOURCE_AMOUNT': 100.0,
                'RESOURCE_REGENERATION_RATE': 0.003,
                'RESOURCE_CONSUMPTION_RATE': 0.02
            },
            'tech_focus': {
                'BASE_RESEARCH_SPEED': 0.03,
                'RESEARCH_RESOURCE_EFFICIENCY': 0.008,
                'TECH_SPILLOVER_EFFECT': 0.1
            },
            'advanced_evolution': {
                'USE_ADVANCED_EVOLUTION': True,
                'USE_COMPLEX_RESOURCES': True,
                'USE_CULTURAL_INFLUENCE': True,
                'SIMULATION_CYCLES': 150,
                'TECH_SPILLOVER_EFFECT': 0.4,
                'EVOLUTION_LEARNING_RATE': 0.15,
                'CULTURAL_DIFFUSION_RATE': 0.07
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            for key, value in preset.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            print(f"已手动应用预设配置: {preset_name}")
        else:
            print(f"未知的预设配置: {preset_name}")


if __name__ == "__main__":
    cli = SimulationCLI()
    cli.run()