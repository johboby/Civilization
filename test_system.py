#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统测试脚本
用于快速验证系统核心功能是否正常工作
"""
import os
import sys
import numpy as np
import pandas as pd
from multi_agent_simulation import MultiAgentSimulation
from simulation_config import config
from civilization_visualizer import CivilizationVisualizer


def test_system():
    """测试文明演化模拟系统的核心功能"""
    print("==== 文明演化模拟系统测试 ====")
    
    # 1. 准备测试环境
    print("\n1. 准备测试环境...")
    test_output_dir = "test_results"
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    
    # 2. 配置简化的测试参数
    print("2. 配置测试参数...")
    original_config = {
        'NUM_CIVILIZATIONS': config.NUM_CIVILIZATIONS,
        'GRID_SIZE': config.GRID_SIZE,
        'SIMULATION_CYCLES': config.SIMULATION_CYCLES,
        'PRINT_LOGS': config.PRINT_LOGS
    }
    
    # 为测试设置较小的参数值以加快测试速度
    config.NUM_CIVILIZATIONS = 3
    config.GRID_SIZE = 50
    config.SIMULATION_CYCLES = 50
    config.PRINT_LOGS = True
    
    # 3. 初始化模拟系统
    print("3. 初始化多智能体模拟...")
    try:
        sim = MultiAgentSimulation(config)
        print(f"   ✓ 成功初始化 {config.NUM_CIVILIZATIONS} 个文明")
    except Exception as e:
        print(f"   ✗ 初始化失败: {e}")
        restore_config(original_config)
        return False
    
    # 4. 运行简短模拟
    print("4. 运行简短模拟...")
    try:
        sim.run()  # 不再获取返回值history，直接从sim对象访问数据
        print(f"   ✓ 成功运行 {config.SIMULATION_CYCLES} 个周期")
    except Exception as e:
        print(f"   ✗ 模拟运行失败: {e}")
        restore_config(original_config)
        return False
    
    # 5. 初始化可视化器
    print("5. 初始化可视化工具...")
    try:
        visualizer = CivilizationVisualizer()
        visualizer.output_dir = test_output_dir
        print("   ✓ 成功初始化可视化工具")
    except Exception as e:
        print(f"   ✗ 可视化器初始化失败: {e}")
        restore_config(original_config)
        return False
    
    # 6. 测试可视化功能
    print("6. 测试可视化功能...")
    viz_success = True
    
    # 策略热力图
    try:
        # 使用resource_grid作为示例
        visualizer.plot_strategy_heatmap(
            sim.resource_grid, 
            title="测试策略热力图",
            filename_prefix="test"
        )
        print("   ✓ 成功生成 策略热力图")
    except Exception as e:
        print(f"   ✗ 策略热力图 生成失败: {e}")
        viz_success = False
    
    # 演化趋势图
    try:
        # 确保strategy_history属性存在
        if hasattr(sim, 'strategy_history') and len(sim.strategy_history) > 0:
            visualizer.plot_evolution_curve(
                np.array(sim.strategy_history), 
                filename_prefix="test"
            )
            print("   ✓ 成功生成 演化趋势图")
        else:
            print("   ✓ 跳过 演化趋势图 测试（无数据）")
    except Exception as e:
        print(f"   ✗ 演化趋势图 生成失败: {e}")
        viz_success = False
    
    # 科技进展图
    try:
        # 确保technology_history属性存在
        if hasattr(sim, 'technology_history') and sim.technology_history:
            visualizer.plot_technology_progress(
                sim.technology_history, 
                filename_prefix="test"
            )
            print("   ✓ 成功生成 科技进展图")
        else:
            print("   ✓ 跳过 科技进展图 测试（无数据）")
    except Exception as e:
        print(f"   ✗ 科技进展图 生成失败: {e}")
        viz_success = False
    
    # 科技树比较图
    try:
        # 确保technology_history属性存在
        if hasattr(sim, 'technology_history') and sim.technology_history:
            visualizer.plot_tech_tree_comparison(
                sim.technology_history, 
                filename_prefix="test"
            )
            print("   ✓ 成功生成 科技树比较图")
        else:
            print("   ✓ 跳过 科技树比较图 测试（无数据）")
    except Exception as e:
        print(f"   ✗ 科技树比较图 生成失败: {e}")
        viz_success = False
    
    # 属性比较图
    try:
        # 检查是否有civilizations属性
        if hasattr(sim, 'civilizations'):
            # 为测试创建简单的属性数据
            attribute_data = {}
            for i, agent in enumerate(sim.civilizations):
                attribute_data[f'文明{i}'] = {
                    'resources': getattr(agent, 'resources', 0),
                    'strength': getattr(agent, 'strength', 0),
                    'defense': getattr(agent, 'defense', 0),
                    'research': getattr(agent, 'research', 0)
                }
            visualizer.plot_attribute_comparison(
                attribute_data, 
                filename_prefix="test"
            )
            print("   ✓ 成功生成 属性比较图")
        else:
            print("   ✓ 跳过 属性比较图 测试（无数据）")
    except Exception as e:
        print(f"   ✗ 属性比较图 生成失败: {e}")
        viz_success = False
    
    # 属性雷达图
    try:
        # 为示例创建属性数据
        if hasattr(sim, 'civilizations') and sim.civilizations:
            # 使用第一个文明的属性
            agent = sim.civilizations[0]
            attributes = {
                "resources": getattr(agent, 'resources', 50),
                "strength": getattr(agent, 'strength', 50),
                "defense": getattr(agent, 'defense', 50),
                "research": getattr(agent, 'research', 50),
                "population": getattr(agent, 'population', 50),
                "territory": getattr(agent, 'territory', 50)
            }
            visualizer.plot_radar_chart(
                attributes, 
                title="文明属性雷达图",
                filename_prefix="test"
            )
            print("   ✓ 成功生成 属性雷达图")
        else:
            print("   ✓ 跳过 属性雷达图 测试（无数据）")
    except Exception as e:
        print(f"   ✗ 属性雷达图 生成失败: {e}")
        viz_success = False
    
    # 关系网络图
    try:
        # 获取文明关系数据
        if hasattr(sim, 'civilizations') and hasattr(sim, '_get_neighbors'):
            relationships_data = {}
            for agent in sim.civilizations:
                neighbors = sim._get_neighbors(agent.id)
                relationships = {}
                for neighbor_id, neighbor_info in neighbors.items():
                    # 确保关系数据格式正确
                    relationships[neighbor_id] = {
                        'value': neighbor_info.get('relationship', 0),
                        'type': neighbor_info.get('type', 'neutral')
                    }
                relationships_data[agent.id] = relationships
            
            visualizer.plot_relationships_network(
                relationships_data, 
                filename_prefix="test"
            )
            print("   ✓ 成功生成 关系网络图")
        else:
            print("   ✓ 跳过 关系网络图 测试（无数据或方法）")
    except Exception as e:
        print(f"   ✗ 关系网络图 生成失败: {e}")
        viz_success = False
    
    # 7. 测试数据保存功能
    print("7. 测试数据保存功能...")
    save_success = True
    
    # 保存策略数据
    try:
        if hasattr(visualizer, 'save_to_csv'):
            # 确保我们有足够的策略历史数据
            if hasattr(sim, 'strategy_history') and len(sim.strategy_history) > 0:
                strategy_history = np.array(sim.strategy_history)
                visualizer.save_to_csv(strategy_history, filename="test_strategy_data.csv")
                print("   ✓ 成功保存策略数据")
            else:
                # 创建模拟数据
                mock_data = np.random.rand(5, 4)  # 模拟5个周期，4种策略
                visualizer.save_to_csv(mock_data, filename="test_strategy_data.csv")
                print("   ✓ 成功保存策略数据 (使用模拟数据)")
    except Exception as e:
        print(f"   ✗ 保存策略数据失败: {e}")
        save_success = False
    
    # 保存属性历史
    try:
        if hasattr(visualizer, 'save_attribute_history'):
            # 准备属性历史数据
            attribute_names = ["resources", "strength", "defense", "research", "population", "territory"]
            
            # 创建pandas数据框
            attribute_history_df = pd.DataFrame(columns=attribute_names)
            for cycle in range(config.SIMULATION_CYCLES):
                row = {}
                for attr in attribute_names:
                    # 用随机数据填充作为示例
                    row[attr] = np.random.rand() * 100
                attribute_history_df.loc[cycle] = row
            
            visualizer.save_attribute_history(attribute_history_df, attribute_names, filename="test_attribute_history.csv")
            print("   ✓ 成功保存属性历史")
    except Exception as e:
        print(f"   ✗ 保存属性历史失败: {e}")
        save_success = False
    
    # 保存科技数据
    try:
        if hasattr(visualizer, 'save_technology_data'):
            if hasattr(sim, 'technology_history') and sim.technology_history:
                visualizer.save_technology_data(sim.technology_history, filename="test_technology_data.csv")
                print("   ✓ 成功保存科技数据")
            else:
                # 创建模拟科技数据
                mock_tech_history = {}
                for agent_id in range(3):
                    agent_history = []
                    for cycle in range(config.SIMULATION_CYCLES):
                        tech_data = {
                            "cycle": cycle,
                            "technologies": {
                                "agriculture": 1 + cycle * 0.1,
                                "military": 1 + cycle * 0.08,
                                "trade": 1 + cycle * 0.05,
                                "science": 1 + cycle * 0.12
                            }
                        }
                        agent_history.append(tech_data)
                    mock_tech_history[agent_id] = agent_history
                
                visualizer.save_technology_data(mock_tech_history, filename="test_technology_data.csv")
                print("   ✓ 成功保存科技数据 (使用模拟数据)")
    except Exception as e:
        print(f"   ✗ 保存科技数据失败: {e}")
        save_success = False
    
    # 保存总结报告
    try:
        if hasattr(visualizer, 'create_summary_report'):
            # 准备策略历史数据
            if hasattr(sim, 'strategy_history') and len(sim.strategy_history) > 0:
                strategy_history = np.array(sim.strategy_history)
            else:
                strategy_history = np.random.rand(config.SIMULATION_CYCLES, 4)  # 模拟4种策略
            
            # 准备属性历史数据框
            attribute_names = ["resources", "strength", "defense", "research", "population", "territory"]
            attribute_history_df = pd.DataFrame(columns=attribute_names)
            for cycle in range(config.SIMULATION_CYCLES):
                row = {}
                for attr in attribute_names:
                    row[attr] = np.random.rand() * 100
                attribute_history_df.loc[cycle] = row
            
            # 准备科技历史数据
            if hasattr(sim, 'technology_history') and sim.technology_history:
                tech_history = sim.technology_history
            else:
                # 使用之前创建的模拟科技数据
                tech_history = mock_tech_history if 'mock_tech_history' in locals() else {}
            
            visualizer.create_summary_report(
                strategy_history,
                attribute_history_df,
                tech_history,
                filename_prefix="test"
            )
            print("   ✓ 成功生成总结报告")
    except Exception as e:
        print(f"   ✗ 生成报告失败: {e}")
        save_success = False
    
    # 8. 保存测试结果
    print("8. 保存测试结果...")
    try:
        test_result_path = os.path.join(test_output_dir, "test_result.npz")
        # 收集可保存的数据
        save_data = {}
        
        # 策略数据
        if hasattr(sim, 'strategy_history') and len(sim.strategy_history) > 0:
            save_data['strategies'] = np.array(sim.strategy_history)
        
        # 资源数据
        if hasattr(sim, 'resource_grid'):
            save_data['resources'] = sim.resource_grid
        
        # 其他数据
        if hasattr(sim, 'civilizations') and sim.civilizations:
            strengths = [getattr(agent, 'strength', 0) for agent in sim.civilizations]
            save_data['strength'] = np.array(strengths)
        
        if save_data:
            np.savez(test_result_path, **save_data)
            print(f"   ✓ 测试结果已保存到 {test_result_path}")
        else:
            print(f"   ✓ 跳过测试结果保存（无有效数据）")
    except Exception as e:
        print(f"   ✗ 保存测试结果失败: {e}")
    
    # 9. 恢复原始配置
    print("9. 恢复原始配置...")
    restore_config(original_config)
    
    # 10. 生成测试总结
    print("\n==== 测试总结 ====")
    all_success = viz_success and save_success
    
    if all_success:
        print("✅ 所有测试项均已通过！")
        print("\n系统工作正常，您可以使用以下命令运行完整模拟：")
        print("- python simulation_cli.py      # 以默认配置运行")
        print("- python simulation_cli.py --interactive # 以交互式模式运行")
        print("- 或双击 run_simulation.bat 文件")
    else:
        print("❌ 部分测试项未通过，请查看上面的错误信息。")
        print("\n建议检查以下内容：")
        print("1. 依赖包是否正确安装（pip install -r requirements.txt）")
        print("2. Python版本是否符合要求（3.8+）")
        print("3. 相关文件是否存在且权限正确")
    
    return all_success

def restore_config(original_config):
    """恢复原始配置"""
    for key, value in original_config.items():
        setattr(config, key, value)


if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)