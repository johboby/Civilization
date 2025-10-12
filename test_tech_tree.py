import numpy as np
import json
from multi_agent_simulation import MultiAgentSimulation
from civilization_visualizer import CivilizationVisualizer
from tech_tree import TechTree

"""测试科技树功能的脚本"""

def main():
    print("开始测试科技树功能...")
    
    # 1. 初始化科技树
    tech_tree = TechTree()
    print("科技树初始化完成")
    print(tech_tree.get_tech_tree_summary())
    
    # 2. 初始化多智能体模拟（配置一个小型环境）
    simulation = MultiAgentSimulation(
        num_agents=3,      # 3个文明
        grid_size=64       # 8x8网格大小
    )
    print(f"多智能体模拟初始化完成，包含{len(simulation.agents)}个文明")
    
    # 3. 运行一个简短的模拟（20个周期）
    num_cycles = 20
    print(f"开始运行{num_cycles}个周期的模拟...")
    strategy_history = []
    resource_history = []
    technology_history = {agent.agent_id: [] for agent in simulation.agents}
    
    for cycle in range(num_cycles):
        # 运行一个周期的模拟
        strategies = simulation.step()
        
        # 收集策略数据
        cycle_strategies = np.array(list(strategies.values()))
        strategy_history.append(np.mean(cycle_strategies, axis=0))
        
        # 收集资源数据
        cycle_resources = np.array([agent.resources for agent in simulation.agents])
        resource_history.append(np.mean(cycle_resources))
        
        # 收集科技数据
        for agent in simulation.agents:
            tech_data = {
                "cycle": cycle,
                "technologies": agent.technology,  # 使用正确的属性名 technology
                "current_research": agent.current_research,
                "research_progress": agent.research_progress,
                "tech_bonuses": agent.tech_bonuses
            }
            technology_history[agent.agent_id].append(tech_data)
            
        if cycle % 5 == 0 or cycle == num_cycles - 1:
            print(f"周期 {cycle}/{num_cycles} 完成")
            # 打印一些进度信息
            for agent in simulation.agents:
                resources = agent.resources
                tech_level = sum(agent.technology.values())  # 使用正确的属性名 technology
                # 简化策略名称显示
                cycle_strategy = strategies.get(agent.agent_id, [0, 0, 0, 0])
                strategy_index = np.argmax(cycle_strategy)
                strategy_names = ["扩张", "防御", "贸易", "研发"]
                current_strategy = strategy_names[strategy_index] if strategy_index < len(strategy_names) else "未知"
                
                research_status = "无" if agent.current_research is None else \
                                f"{agent.current_research} ({agent.research_progress:.1f}%)"
                
                print(f"  文明{agent.agent_id}: 资源={resources:.1f}, 科技等级={tech_level}, " \
                      f"策略={current_strategy}, 研究={research_status}")
    
    # 4. 转换历史数据为numpy数组
    strategy_history = np.array(strategy_history)
    resource_util_history = np.array(resource_history) / 100.0  # 归一化资源
    combined_data = np.column_stack((strategy_history, resource_util_history))
    
    # 5. 使用可视化工具生成结果
    print("生成可视化结果...")
    visualizer = CivilizationVisualizer()
    
    # 绘制传统的策略热力图和演化曲线
    visualizer.plot_strategy_heatmap(strategy_history)
    visualizer.plot_evolution_curve(strategy_history, resource_history)
    
    # 绘制科技相关的可视化结果
    visualizer.plot_technology_progress(technology_history)
    visualizer.plot_tech_tree_comparison(technology_history)
    
    # 6. 保存数据
    print("保存模拟数据...")
    np.savez("test_simulation_results.npz", 
             strategy_history=strategy_history, 
             resource_history=resource_history)
    
    # 保存科技历史数据到JSON文件
    with open("test_technology_history.json", "w", encoding="utf-8") as f:
        json.dump(technology_history, f, ensure_ascii=False, indent=2)
    
    # 保存CSV数据
    visualizer.save_to_csv(combined_data, "test_civilization_data.csv")
    
    print("\n测试完成！所有结果已保存到当前目录。")
    print("生成的文件：")
    print("- strategy_heatmap.png: 资源分布热力图")
    print("- evolution_curve.png: 策略演化曲线")
    print("- technology_progress.png: 科技发展趋势图")
    print("- tech_tree_comparison.png: 文明科技树比较图")
    print("- test_simulation_results.npz: 模拟结果数据")
    print("- test_technology_history.json: 科技发展历史数据")
    print("- test_civilization_data.csv: 文明数据CSV")

if __name__ == "__main__":
    main()