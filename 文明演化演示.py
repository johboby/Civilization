import numpy as np
import torch
import matplotlib.pyplot as plt
from civilization_visualizer import CivilizationVisualizer
from multi_agent_simulation import MultiAgentSimulation
from tech_tree import TechTree

class CivilizationDemo:
    """文明演化演示类"""
    def __init__(self, num_agents=5, grid_size=10, output_dir="."):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.output_dir = output_dir
        self.simulation = MultiAgentSimulation(num_agents, grid_size)
        self.visualizer = CivilizationVisualizer(output_dir)
        self.strategy_history = {}
        self.resource_history = {}
        self.relationship_history = {}
        self.technology_history = {}
        self.cycle_data = []
        
        # 初始化科技树历史记录
        for agent_id in range(num_agents):
            self.technology_history[agent_id] = []
        
    def run_simulation(self, num_cycles=None):
        """运行文明演化模拟"""
        if num_cycles is None:
            num_cycles = getattr(self, 'epochs', 30)
            
        print(f"开始文明演化演示，{self.num_agents}个文明，{self.grid_size}个网格，共{num_cycles}个周期")
        
        # 执行模拟循环
        for cycle in range(num_cycles):
            # 执行一个周期的模拟
            strategies, resources, relationships = self.simulation.run_cycle()
            
            # 记录数据
            self._record_cycle_data(cycle, strategies, resources, relationships)
            
            # 记录科技发展数据
            self._record_technology_data(cycle)
            
            # 打印周期信息
            if cycle % 5 == 0:
                print(f"周期{cycle}完成")
                print(f"  文明资源分布: { {k: int(v) for k, v in resources.items()} }")
                
        print("文明演化演示完成")
        return self.strategy_history, self.resource_history
    
    def visualize_results(self):
        """可视化模拟结果"""
        # 1. 将策略历史转换为矩阵格式
        num_agents = self.num_agents
        strategy_matrix = np.zeros((self.epochs, num_agents, 3))
        
        for epoch_idx, strategies in enumerate(self.strategy_history):
            for agent_id, strategy in strategies.items():
                strategy_matrix[epoch_idx, agent_id] = strategy
        
        # 2. 创建平均策略演化曲线
        avg_strategy = np.mean(strategy_matrix, axis=1)
        
        # 3. 绘制策略演化曲线
        plt.figure(figsize=(12, 6))
        epochs = range(self.epochs)
        plt.plot(epochs, avg_strategy[:, 0], label="扩张策略", color="#0868ac")
        plt.plot(epochs, avg_strategy[:, 1], label="防御策略", color="#d95f0e")
        plt.plot(epochs, avg_strategy[:, 2], label="贸易策略", color="#3f007d")
        plt.xlabel("演化周期")
        plt.ylabel("平均策略概率")
        plt.title("文明策略演化趋势", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("demo_strategy_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 4. 绘制资源演化曲线
        plt.figure(figsize=(12, 6))
        resource_matrix = np.zeros((self.epochs, num_agents))
        
        for epoch_idx, resources in enumerate(self.resource_history):
            for agent_id, resource in resources.items():
                resource_matrix[epoch_idx, agent_id] = resource
        
        for agent_id in range(num_agents):
            plt.plot(epochs, resource_matrix[:, agent_id], label=f"文明{agent_id}")
        
        plt.xlabel("演化周期")
        plt.ylabel("资源储备")
        plt.title("各文明资源演化曲线", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("demo_resource_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 5. 创建文明关系热力图
        plt.figure(figsize=(10, 8))
        relationship_matrix = np.zeros((num_agents, num_agents))
        
        for i, agent_i in enumerate(self.sim.agents):
            for j, agent_j in enumerate(self.sim.agents):
                if i == j:
                    relationship_matrix[i, j] = 0
                elif j in agent_i.allies:
                    relationship_matrix[i, j] = 1  # 盟友
                elif j in agent_i.enemies:
                    relationship_matrix[i, j] = -1  # 敌人
                else:
                    relationship_matrix[i, j] = 0.5  # 中立
        
        plt.imshow(relationship_matrix, cmap="RdYlBu", vmin=-1, vmax=1)
        plt.colorbar(label="关系类型 (-1=敌对, 0=中立, 1=盟友)")
        plt.title("文明关系热力图", fontsize=14)
        plt.xlabel("文明ID")
        plt.ylabel("文明ID")
        plt.savefig("demo_relationship_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("可视化结果已保存：")
        print("- demo_strategy_evolution.png：策略演化趋势图")
        print("- demo_resource_evolution.png：资源演化曲线")
        print("- demo_relationship_heatmap.png：文明关系热力图")
    
    def _record_cycle_data(self, cycle, strategies, resources, relationships):
        """记录每个周期的数据"""
        for agent_id in range(self.num_agents):
            if agent_id not in self.strategy_history:
                self.strategy_history[agent_id] = []
                self.resource_history[agent_id] = []
                self.relationship_history[agent_id] = []
            
            # 记录策略和资源数据
            self.strategy_history[agent_id].append(strategies[agent_id].tolist())
            self.resource_history[agent_id].append(float(resources[agent_id]))
            
            # 记录关系数据
            agent_relationships = {}
            for other_id in range(self.num_agents):
                if other_id != agent_id:
                    agent_relationships[other_id] = relationships[agent_id, other_id]
            self.relationship_history[agent_id].append(agent_relationships)
            
    def _record_technology_data(self, cycle):
        """记录每个周期的科技发展数据"""
        for agent_id in range(self.num_agents):
            agent = self.simulation.agents[agent_id]
            tech_data = {
                "cycle": cycle,
                "technologies": agent.technology.copy(),
                "current_research": agent.current_research,
                "research_progress": float(agent.research_progress),
                "research_cost": getattr(agent, "research_cost", 0),
                "tech_bonuses": agent.tech_bonuses.copy()
            }
            self.technology_history[agent_id].append(tech_data)
    
    def save_results(self):
        """保存模拟结果到文件"""
        # 保存策略历史
        np.savez(f"{self.output_dir}\文明演化演示策略历史.npz", **self.strategy_history)
        
        # 保存资源历史
        np.savez(f"{self.output_dir}\文明演化演示资源历史.npz", **self.resource_history)
        
        # 保存文明状态
        civilization_state = {
            "cycle_data": self.cycle_data,
            "agent_states": {}
        }
        
        # 收集各文明的最终状态
        for agent_id, agent in enumerate(self.simulation.agents):
            civilization_state["agent_states"][agent_id] = {
                "strength": float(agent.strength),
                "resources": float(agent.resources),
                "territory": list(agent.territory),
                "allies": list(agent.allies),
                "enemies": list(agent.enemies),
                "technology": agent.technology
            }
        
        # 保存为JSON文件
        import json
        with open(f"{self.output_dir}\文明演化演示文明状态.json", "w", encoding="utf-8") as f:
            json.dump(civilization_state, f, ensure_ascii=False, indent=2)
            
        # 保存科技历史数据
        with open(f"{self.output_dir}\文明演化演示科技历史.json", "w", encoding="utf-8") as f:
            json.dump(self.technology_history, f, ensure_ascii=False, indent=2)
        
        print("模拟数据已保存：")
        print(f"- {self.output_dir}\文明演化演示策略历史.npz")
        print(f"- {self.output_dir}\文明演化演示资源历史.npz")
        print(f"- {self.output_dir}\文明演化演示文明状态.json")
        print(f"- {self.output_dir}\文明演化演示科技历史.json")

if __name__ == "__main__":
    # 创建并运行演示
    demo = CivilizationDemo(num_agents=5, grid_size=200, epochs=30)
    demo.run_simulation()
    demo.visualize_results()
    demo.save_results()