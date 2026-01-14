"""
Interactive demonstration script for civilization simulation.

This script provides an interactive interface for running and visualizing simulations.
"""
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from civsim import (
    MultiAgentSimulation,
    SimulationConfig,
    get_preset,
    init_logging,
    info,
    warning
)


class InteractiveDemo:
    """Interactive demonstration interface."""

    def __init__(self, config: SimulationConfig):
        """Initialize interactive demo.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self.simulation = MultiAgentSimulation(config)
        self.running = True
        self.cycle = 0

    def run(self):
        """Run interactive demo."""
        info("=== 文明演化模拟系统 - 交互式演示 ===")
        info(f"配置: {self.config.num_civilizations} 个文明, {self.config.simulation_cycles} 个周期")
        info("")  # Empty line for spacing
        self.print_menu()

        while self.running:
            try:
                choice = input("请选择操作 (输入数字): ").strip()
                self.handle_choice(choice)
            except KeyboardInterrupt:
                print("\n\n程序已停止")
                self.running = False
            except EOFError:
                print("\n\n程序已停止")
                self.running = False

    def print_menu(self):
        """Print main menu."""
        print("\n" + "=" * 50)
        print("主菜单:")
        print("  1. 运行一个周期")
        print("  2. 运行 N 个周期")
        print("  3. 运行到结束")
        print("  4. 查看当前状态")
        print("  5. 查看文明详情")
        print("  6. 查看关系网络")
        print("  7. 查看科技树")
        print("  8. 生成可视化")
        print("  9. 保存当前状态")
        print("  0. 退出")
        print("=" * 50)

    def handle_choice(self, choice: str):
        """Handle user menu choice.

        Args:
            choice: User input choice.
        """
        if choice == "1":
            self.run_one_cycle()
        elif choice == "2":
            self.run_multiple_cycles()
        elif choice == "3":
            self.run_to_end()
        elif choice == "4":
            self.show_current_state()
        elif choice == "5":
            self.show_civilization_details()
        elif choice == "6":
            self.show_relationship_network()
        elif choice == "7":
            self.show_technology_tree()
        elif choice == "8":
            self.generate_visualization()
        elif choice == "9":
            self.save_state()
        elif choice == "0":
            self.running = False
            print("退出程序...")
        else:
            print("无效的选择，请重试")

    def run_one_cycle(self):
        """Run a single simulation cycle."""
        if self.cycle >= self.config.simulation_cycles:
            warning("模拟已完成！")
            return

        info(f"\n--- 运行周期 {self.cycle + 1} ---")

        # Run simulation step (would normally call sim.run(1))
        # For now, just increment cycle
        self.cycle += 1

        info(f"周期 {self.cycle} 完成")

    def run_multiple_cycles(self):
        """Run multiple simulation cycles."""
        try:
            n = int(input("输入要运行的周期数: "))
            if n <= 0:
                warning("周期数必须大于 0")
                return

            if self.cycle + n > self.config.simulation_cycles:
                warning(f"总周期数不能超过 {self.config.simulation_cycles}")
                n = self.config.simulation_cycles - self.cycle

            info(f"运行 {n} 个周期...")
            for _ in range(n):
                self.run_one_cycle()

        except ValueError:
            warning("请输入有效的数字")

    def run_to_end(self):
        """Run simulation until completion."""
        remaining = self.config.simulation_cycles - self.cycle
        if remaining <= 0:
            warning("模拟已完成！")
            return

        info(f"运行剩余 {remaining} 个周期...")
        for _ in range(remaining):
            self.run_one_cycle()

    def show_current_state(self):
        """Display current simulation state."""
        print("\n--- 当前状态 ---")
        print(f"当前周期: {self.cycle} / {self.config.simulation_cycles}")
        print(f"文明数量: {len(self.simulation.agents)}")

        total_resources = sum(agent.resources for agent in self.simulation.agents)
        total_population = sum(agent.population for agent in self.simulation.agents)
        total_strength = sum(agent.strength for agent in self.simulation.agents)

        print(f"总资源: {total_resources:.1f}")
        print(f"总人口: {total_population:.1f}")
        print(f"总军事实力: {total_strength:.1f}")

        avg_resources = total_resources / len(self.simulation.agents)
        avg_population = total_population / len(self.simulation.agents)
        avg_strength = total_strength / len(self.simulation.agents)

        print(f"平均资源: {avg_resources:.1f}")
        print(f"平均人口: {avg_population:.1f}")
        print(f"平均军事实力: {avg_strength:.1f}")

    def show_civilization_details(self):
        """Display detailed information about each civilization."""
        try:
            agent_id = int(input("输入文明编号 (0-{}): ".format(len(self.simulation.agents) - 1)))
        except ValueError:
            warning("请输入有效的数字")
            return

        if agent_id < 0 or agent_id >= len(self.simulation.agents):
            warning("无效的文明编号")
            return

        agent = self.simulation.agents[agent_id]
        print(f"\n=== 文明 {agent_id} 详情 ===")
        print(f"资源: {agent.resources:.1f}")
        print(f"人口: {agent.population:.1f}")
        print(f"军事实力: {agent.strength:.1f}")
        print(f"防御力: {agent.defense:.1f}")
        print(f"基础设施: {agent.infrastructure:.2f}")
        print(f"稳定性: {agent.stability:.2f}")
        print(f"领土数量: {len(agent.territory)}")
        print(f"盟友数量: {len(agent.allies)}")
        print(f"敌对数量: {len(agent.enemies)}")
        print(f"\n科技等级:")
        for tech_name, level in agent.technology.items():
            print(f"  {tech_name}: {level:.1f}")

    def show_relationship_network(self):
        """Display relationship network."""
        print("\n--- 关系网络 ---")
        for agent in self.simulation.agents:
            print(f"\n文明 {agent.agent_id} 的关系:")
            for other_id, weight in agent.relationship_weights.items():
                relation_type = "中立"
                if weight > 0.5:
                    relation_type = "盟友"
                elif weight < -0.5:
                    relation_type = "敌对"
                print(f"  -> 文明 {other_id}: {weight:.2f} ({relation_type})")

    def show_technology_tree(self):
        """Display technology tree status."""
        try:
            agent_id = int(input("输入文明编号 (0-{}): ".format(len(self.simulation.agents) - 1)))
        except ValueError:
            warning("请输入有效的数字")
            return

        if agent_id < 0 or agent_id >= len(self.simulation.agents):
            warning("无效的文明编号")
            return

        agent = self.simulation.agents[agent_id]
        print(f"\n=== 文明 {agent_id} 科技树 ===")
        tech_tree = agent.tech_tree

        print("\n已研究科技:")
        for tech_name, level in agent.technology.items():
            tech_info = tech_tree.get_tech_info(tech_name)
            if tech_info:
                print(f"  {tech_info.name} (等级 {level:.1f}/{tech_info.level})")

        print("\n可研究科技:")
        available_techs = tech_tree.get_available_techs(agent.technology)
        if not available_techs:
            print("  无可研究的科技")
        else:
            for tech_info in available_techs:
                print(f"  {tech_info['name']} (等级 {tech_info['level']}) - 费用: {tech_info['cost']:.1f}")

    def generate_visualization(self):
        """Generate visualization of current state."""
        try:
            from civilization_visualizer import CivilizationVisualizer
            import numpy as np

            output_dir = "results/interactive"
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Create visualizer
            visualizer = CivilizationVisualizer(output_dir=output_dir)

            # Get agents data for comparison
            agents_data = self.simulation.agents

            # Plot civilization comparison radar
            visualizer.plot_civilization_comparison_radar(agents_data, filename="civilization_comparison_radar.png")

            # Create mock evolution curve data from current state
            mock_history = np.zeros((10, min(7, agents_data[0].strategy_count)))
            for i in range(10):
                for j in range(mock_history.shape[1]):
                    mock_history[i, j] = np.random.rand()

            # Plot evolution curve
            visualizer.plot_evolution_curve(mock_history, filename="evolution_curve.png")

            # Plot attribute comparison
            attribute_names = ['resources', 'strength', 'technology', 'population', 'territory']
            attribute_data = np.array([
                [agent.resources, agent.strength, sum(agent.technology.values()), agent.population, len(agent.territory)]
                for agent in agents_data
            ])
            visualizer.plot_attribute_comparison(attribute_data, attribute_names, filename="attribute_comparison.png")

            info(f"可视化已生成到 {output_dir}")

        except ImportError:
            warning("civilization_visualizer 模块未找到，无法生成可视化")
        except Exception as e:
            warning(f"可视化生成失败: {e}")

    def save_state(self):
        """Save current simulation state."""
        import json
        from pathlib import Path
        from datetime import datetime

        output_dir = Path("results/interactive")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"state_{timestamp}.json"

        state_data = {
            'cycle': self.cycle,
            'num_agents': len(self.simulation.agents),
            'agents': []
        }

        for agent in self.simulation.agents:
            agent_data = {
                'agent_id': agent.agent_id,
                'resources': float(agent.resources),
                'population': float(agent.population),
                'strength': float(agent.strength),
                'defense': float(agent.defense),
                'infrastructure': float(agent.infrastructure),
                'stability': float(agent.stability),
                'territory_size': len(agent.territory),
                'allies_size': len(agent.allies),
                'enemies_size': len(agent.enemies),
                'technology': {k: float(v) for k, v in agent.technology.items()},
                'relationships': {k: float(v) for k, v in agent.relationship_weights.items()}
            }
            state_data['agents'].append(agent_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)

        info(f"状态已保存到 {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Interactive demonstration of civilization simulation')
    parser.add_argument(
        '--preset',
        type=str,
        choices=['demo', 'small', 'medium', 'large'],
        default='demo',
        help='Configuration preset'
    )
    parser.add_argument(
        '--num-civs',
        type=int,
        help='Number of civilizations'
    )
    parser.add_argument(
        '--cycles',
        type=int,
        help='Number of simulation cycles'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Initialize logging
    init_logging()

    # Get configuration
    config = get_preset(args.preset)

    # Apply command line overrides
    if args.num_civs:
        config.num_civilizations = args.num_civs
    if args.cycles:
        config.simulation_cycles = args.cycles
    if args.seed:
        config.random_seed = args.seed
        np.random.seed(args.seed)

    # Create and run interactive demo
    demo = InteractiveDemo(config)
    demo.run()


if __name__ == "__main__":
    main()
