import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import networkx as nx
import matplotlib.animation as animation
from matplotlib.patches import Wedge
import matplotlib.cm as cm

# Simplify font settings, only use the most common Chinese fonts on Windows systems
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # Solve negative sign display issue

class CivilizationVisualizer:
    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        # Custom color mapping (civilization expansion/defense/trade/research strategies)
        self.cmaps = {
            "expansion": LinearSegmentedColormap.from_list("expansion", ["#f0f9e8", "#7bccc4", "#0868ac"]),
            "defense": LinearSegmentedColormap.from_list("defense", ["#fff7bc", "#fec44f", "#d95f0e"]),
            "trade": LinearSegmentedColormap.from_list("trade", ["#f1eef6", "#9e9ac8", "#3f007d"]),
            "research": LinearSegmentedColormap.from_list("research", ["#f7f7f7", "#96CEB4", "#4E9CAF"]),
            "diplomacy": LinearSegmentedColormap.from_list("diplomacy", ["#fde0dd", "#fa9fb5", "#c51b8a"]),
            "population": LinearSegmentedColormap.from_list("population", ["#f2f0f7", "#cbc9e2", "#807dba"]),
            "health": LinearSegmentedColormap.from_list("health", ["#edf8e9", "#bae4b3", "#74c476"]),
            "resources": LinearSegmentedColormap.from_list("resources", ["#fff9e6", "#ffd670", "#ffb72b"])
        }
        # Base colors
        self.colors = {
            'expansion': '#0868ac',  # 扩张策略 - 蓝色
            'defense': '#d95f0e',    # 防御策略 - 橙色
            'trade': '#3f007d',      # 贸易策略 - 紫色
            'research': '#4E9CAF',   # 研发策略 - 青色
            'diplomacy': '#c51b8a',  # 外交策略 - 粉色
            'population': '#807dba', # 人口发展 - 紫色系
            'health': '#74c476',     # 健康水平 - 绿色
            'resources': '#ffb72b',  # 资源丰富度 - 黄色
            'infrastructure': '#54278f', # 基础设施 - 深紫色
            'stability': '#a63603',  # 社会稳定 - 深橙色
            'unknown': '#999999'     # 未知策略 - 灰色
        }
        # Top technology colors
        self.top_tech_colors = {
            'genetic_engineering': '#4DAF4A',  # 基因工程 - 绿色
            'nuclear_technology': '#E41A1C',   # 核技术 - 红色
            'space_colonization': '#377EB8',   # 太空殖民 - 蓝色
            'artificial_intelligence': '#984EA3' # 人工智能 - 紫色
        }

    def plot_strategy_heatmap(self, strategy_matrix, title="Civilization Strategy Heatmap", filename="strategy_heatmap.png", cmap_type="expansion", colorbar_label="Intensity"):
        """Draw strategy selection heatmap"""
        plt.figure(figsize=(10, 8))
        cmap = self.cmaps.get(cmap_type, self.cmaps["expansion"])
        plt.imshow(strategy_matrix, cmap=cmap)
        plt.colorbar(label=colorbar_label)
        plt.title(title, fontsize=14)
        plt.xlabel("网格X坐标")
        plt.ylabel("网格Y坐标")
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches="tight")
        plt.close()
        
    def plot_attribute_heatmap(self, attribute_matrix, attribute_name, title=None, filename=None):
        """Draw heatmap for specific attributes"""
        if title is None:
            title = f"文明{attribute_name}热力图"
        if filename is None:
            filename = f"{attribute_name}_heatmap.png"
        
        # Select appropriate color mapping based on attribute name
        cmap_type = "resources"  # 默认使用资源的颜色映射
        if attribute_name in self.cmaps:
            cmap_type = attribute_name
        
        self.plot_strategy_heatmap(
            attribute_matrix, 
            title=title, 
            filename=filename, 
            cmap_type=cmap_type, 
            colorbar_label=f"{attribute_name}值"
        )

    def plot_evolution_curve(self, history_data, resource_history=None, attribute_history=None, attribute_names=None, filename="evolution_curve.png"):
        """Draw civilization evolution curve (strategy change trend)"""
        plt.figure(figsize=(12, 6))
        epochs = range(len(history_data))
        
        # Draw strategy curves
        if history_data.shape[1] >= 7:
            # 7 strategies (including religion)
            plt.plot(epochs, history_data[:, 0], label="扩张策略", color=self.colors['expansion'])
            plt.plot(epochs, history_data[:, 1], label="防御策略", color=self.colors['defense'])
            plt.plot(epochs, history_data[:, 2], label="贸易策略", color=self.colors['trade'])
            plt.plot(epochs, history_data[:, 3], label="研发策略", color=self.colors['research'])
            plt.plot(epochs, history_data[:, 4], label="外交策略", color=self.colors['diplomacy'])
            plt.plot(epochs, history_data[:, 5], label="文化策略", color=self.colors['population'])
            plt.plot(epochs, history_data[:, 6], label="宗教策略", color=self.colors['health'])
        elif history_data.shape[1] >= 5:
            # New strategy + research + diplomacy data
            plt.plot(epochs, history_data[:, 0], label="扩张策略", color=self.colors['expansion'])
            plt.plot(epochs, history_data[:, 1], label="防御策略", color=self.colors['defense'])
            plt.plot(epochs, history_data[:, 2], label="贸易策略", color=self.colors['trade'])
            plt.plot(epochs, history_data[:, 3], label="研发策略", color=self.colors['research'])
            plt.plot(epochs, history_data[:, 4], label="外交策略", color=self.colors['diplomacy'])
        elif history_data.shape[1] == 4:
            # 4 strategies (including research)
            plt.plot(epochs, history_data[:, 0], label="扩张策略", color=self.colors['expansion'])
            plt.plot(epochs, history_data[:, 1], label="防御策略", color=self.colors['defense'])
            plt.plot(epochs, history_data[:, 2], label="贸易策略", color=self.colors['trade'])
            plt.plot(epochs, history_data[:, 3], label="研发策略", color=self.colors['research'])
        else:
            # Traditional 3 strategies
            plt.plot(epochs, history_data[:, 0], label="扩张策略", color=self.colors['expansion'])
            plt.plot(epochs, history_data[:, 1], label="防御策略", color=self.colors['defense'])
            plt.plot(epochs, history_data[:, 2], label="贸易策略", color=self.colors['trade'])
        
        # 如果提供了资源历史数据，也绘制资源曲线
        if resource_history is not None:
            ax2 = plt.twinx()
            ax2.plot(epochs, resource_history, label="平均资源", color='gray', linestyle='--')
            ax2.set_ylabel("资源值")
            ax2.legend(loc='upper right')
        
        plt.xlabel("演化周期")
        plt.ylabel("策略选择概率")
        plt.title("文明策略演化趋势", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches="tight")
        plt.close()
        
    def plot_attribute_comparison(self, attribute_data, attribute_names, title="文明属性比较", filename="attribute_comparison.png"):
        """绘制多个文明之间的属性比较图"""
        n_agents = attribute_data.shape[0]  # 文明数量
        n_attributes = attribute_data.shape[1]  # 属性数量
        
        plt.figure(figsize=(15, 8))
        bar_width = 0.8 / n_attributes
        x_pos = np.arange(n_agents)
        
        for i in range(n_attributes):
            attribute_name = attribute_names[i] if i < len(attribute_names) else f"属性{i+1}"
            color = self.colors.get(attribute_name, self.colors['unknown'])
            plt.bar(x_pos + i * bar_width, attribute_data[:, i], width=bar_width, label=attribute_name, color=color)
        
        plt.title(title, fontsize=14)
        plt.xlabel("文明ID")
        plt.ylabel("属性值")
        plt.xticks(x_pos + bar_width * (n_attributes - 1) / 2, [f"文明{i}" for i in range(n_agents)])
        plt.legend()
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches="tight")
        plt.close()
        
    def plot_radar_chart(self, attribute_data, attribute_names, agent_names=None, title="文明属性雷达图", filename="radar_chart.png"):
        """绘制文明属性雷达图"""
        # 计算角度
        n_attributes = len(attribute_names)
        angles = np.linspace(0, 2 * np.pi, n_attributes, endpoint=False).tolist()
        
        # 闭合雷达图
        angles += angles[:1]
        attribute_data = np.concatenate((attribute_data, attribute_data[:, [0]]), axis=1)
        attribute_names = attribute_names + [attribute_names[0]]
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # 为每个文明绘制雷达图
        for i in range(len(attribute_data)):
            agent_name = agent_names[i] if agent_names and i < len(agent_names) else f"文明{i}"
            # 从colors中选择颜色，如果不够则使用自动颜色
            if i < len(self.colors):
                color = list(self.colors.values())[i]
            else:
                color = None  # 使用默认颜色
            ax.plot(angles, attribute_data[i], 'o-', linewidth=2, label=agent_name, color=color)
            ax.fill(angles, attribute_data[i], alpha=0.25)
        
        # 设置标签
        ax.set_thetagrids(np.degrees(angles[:-1]), attribute_names[:-1])
        ax.set_ylim(0, 1.1 * np.max(attribute_data))
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches="tight")
        plt.close()
        
    def plot_technology_progress(self, technology_history, num_cycles=None, filename_prefix="demo"):
        """绘制科技发展进度图，支持显示顶级科技突破点"""
        if not technology_history:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Get all technology types
        all_techs = set()
        for agent_id, history in technology_history.items():
            if history:
                for tech_data in history:
                    all_techs.update(tech_data["technologies"].keys())
        
        all_techs = sorted(list(all_techs))
        
        # Draw technology development curves for each civilization
        for agent_id in technology_history.keys():
            history = technology_history[agent_id]
            if num_cycles is not None:
                history = history[:num_cycles]
            
            cycles = [data["cycle"] for data in history]
            
            # 计算总科技水平
            total_tech_levels = []
            # 记录顶级科技突破点
            top_tech_breakthroughs = {}
            
            for data in history:
                total_level = sum(data["technologies"].values())
                total_tech_levels.append(total_level)
                
                # 检查顶级科技突破
                for tech, level in data["technologies"].items():
                    if tech in self.top_tech_colors and level >= 3:  # 假设3级是顶级科技的标志
                        if tech not in top_tech_breakthroughs:
                            top_tech_breakthroughs[tech] = data["cycle"]
            
            # 绘制总科技水平曲线
            color = self.colors.get(f"civilization_{agent_id}", None)
            line, = plt.plot(cycles, total_tech_levels, label=f'文明{agent_id}总科技水平', linewidth=2, color=color)
            
            # 标记顶级科技突破点
            for tech, cycle in top_tech_breakthroughs.items():
                idx = cycles.index(cycle)
                tech_color = self.top_tech_colors.get(tech, 'red')
                plt.plot(cycle, total_tech_levels[idx], 'o', color=tech_color, markersize=10)
                plt.annotate(
                    f'突破: {tech}', 
                    xy=(cycle, total_tech_levels[idx]), 
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=tech_color)
                )
        
        plt.title('各文明科技发展趋势', fontsize=16)
        plt.xlabel('周期', fontsize=14)
        plt.ylabel('总科技等级', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(f"{self.output_dir}/{filename_prefix}_technology_progress.png", dpi=300)
        plt.close()
        
    def plot_top_tech_comparison(self, technology_history, filename_prefix="demo"):
        """专门比较各文明的顶级科技发展情况"""
        if not technology_history:
            return
            
        plt.figure(figsize=(15, 8))
        
        # 收集各文明的顶级科技数据
        top_tech_data = {}
        for agent_id, history in technology_history.items():
            if not history:
                continue
                
            last_tech = history[-1]["technologies"]
            top_techs = {tech: level for tech, level in last_tech.items() if tech in self.top_tech_colors}
            top_tech_data[agent_id] = top_techs
        
        # 绘制顶级科技比较图
        n_agents = len(top_tech_data)
        x = np.arange(n_agents)
        bar_width = 0.15
        
        # 为每种顶级科技创建一个分组的柱状图
        for i, tech_name in enumerate(self.top_tech_colors.keys()):
            tech_levels = []
            for agent_id in top_tech_data.keys():
                tech_levels.append(top_tech_data[agent_id].get(tech_name, 0))
                
            plt.bar(x + i * bar_width, tech_levels, width=bar_width, label=tech_name, color=self.top_tech_colors[tech_name])
        
        plt.title('各文明顶级科技比较', fontsize=16)
        plt.xlabel('文明ID', fontsize=14)
        plt.ylabel('科技等级', fontsize=14)
        plt.xticks(x + bar_width * (len(self.top_tech_colors) - 1) / 2, [f'文明{id}' for id in top_tech_data.keys()])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(f"{self.output_dir}/{filename_prefix}_top_tech_comparison.png", dpi=300)
        plt.close()
        
    def plot_tech_tree_comparison(self, technology_history, filename_prefix="demo"):
        """绘制不同文明科技树比较图，区分普通科技和顶级科技"""
        if not technology_history:
            return
            
        # 创建画布
        fig, axes = plt.subplots(len(technology_history), 1, figsize=(14, 5 * len(technology_history)))
        
        if len(technology_history) == 1:
            axes = [axes]  # 确保axes是列表
            
        # 为每个文明绘制科技树状态
        for idx, agent_id in enumerate(technology_history.keys()):
            history = technology_history[agent_id]
            if not history:
                continue
                
            # 获取最后一个周期的科技状态
            last_tech_data = history[-1]
            tech_levels = last_tech_data["technologies"]
            
            # 创建科技树图
            ax = axes[idx]
            
            # 区分普通科技和顶级科技
            regular_techs = {tech: level for tech, level in tech_levels.items() if tech not in self.top_tech_colors}
            top_techs = {tech: level for tech, level in tech_levels.items() if tech in self.top_tech_colors}
            
            # 合并排序，确保顶级科技在右侧
            all_techs = {**regular_techs, **top_techs}
            tech_names = list(all_techs.keys())
            tech_values = list(all_techs.values())
            
            # 为每个科技选择颜色
            colors = []
            for tech in tech_names:
                if tech in self.top_tech_colors:
                    colors.append(self.top_tech_colors[tech])
                else:
                    colors.append(self.colors["research"])
            
            bars = ax.bar(tech_names, tech_values, color=colors, alpha=0.8)
            ax.set_title(f'文明{agent_id}科技树状态', fontsize=14)
            ax.set_xlabel('科技领域', fontsize=12)
            ax.set_ylabel('科技等级', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 在柱状图上显示数值
            for bar, value in zip(bars, tech_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, 
                        f'{value}', ha='center', va='bottom')
                        
            # 添加图例说明顶级科技
            if top_techs:
                handles = [plt.Rectangle((0,0),1,1, color=self.top_tech_colors[tech]) for tech in top_techs]
                labels = list(top_techs.keys())
                ax.legend(handles, labels, title='顶级科技', loc='upper right')
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(f"{self.output_dir}/{filename_prefix}_tech_tree_comparison.png", dpi=300)
        plt.close()
        
    def create_evolution_animation(self, grid_history, strategy_history, filename_prefix="evolution"):
        """创建文明演化动画"""
        if not grid_history or len(grid_history) < 2:
            print("没有足够的历史数据来创建动画")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 初始化热力图
        cmap = self.cmaps["resources"]
        im1 = ax1.imshow(grid_history[0], cmap=cmap)
        ax1.set_title('文明领土与资源分布')
        fig.colorbar(im1, ax=ax1, label='资源值')
        
        # 初始化策略分布图
        im2 = ax2.imshow(strategy_history[0], cmap=self.cmaps["expansion"])
        ax2.set_title('文明策略分布')
        fig.colorbar(im2, ax=ax2, label='扩张策略强度')
        
        # 添加时间标签
        text = fig.text(0.5, 0.01, f'周期: 0', ha='center', fontsize=12)
        
        def update(frame):
            # 更新热力图
            im1.set_array(grid_history[frame])
            im2.set_array(strategy_history[frame])
            text.set_text(f'周期: {frame}')
            return [im1, im2, text]
        
        # 创建动画
        ani = animation.FuncAnimation(
            fig, update, frames=len(grid_history), interval=500, blit=True
        )
        
        # 保存动画为GIF
        ani.save(f"{self.output_dir}/{filename_prefix}_animation.gif", writer='pillow')
        plt.close()
        
        print(f"动画已保存为: {self.output_dir}/{filename_prefix}_animation.gif")
        
    def plot_relationships_network(self, relationships_data, filename_prefix="demo"):
        """绘制文明关系网络图"""
        if not relationships_data:
            return
            
        # 创建图
        G = nx.DiGraph()
        
        # 添加节点（文明）
        for agent_id in relationships_data:
            G.add_node(agent_id)
        
        # 添加边（关系）
        for source, relations in relationships_data.items():
            for target, relation in relations.items():
                if source != target:
                    # 根据关系值设置边的颜色和宽度
                    relation_value = relation.get('value', 0)
                    width = max(0.5, min(3, abs(relation_value) * 2))
                    color = 'green' if relation_value > 0 else 'red' if relation_value < 0 else 'gray'
                    
                    G.add_edge(source, target, weight=relation_value, color=color, width=width)
        
        # 绘制网络图
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # 获取边的属性
        edges = G.edges()
        colors = [G[u][v]['color'] for u, v in edges]
        widths = [G[u][v]['width'] for u, v in edges]
        
        # 绘制节点和边
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')
        nx.draw_networkx_labels(G, pos, font_size=12)
        nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, alpha=0.7, arrows=True)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='友好关系'),
            Line2D([0], [0], color='red', lw=2, label='敌对关系'),
            Line2D([0], [0], color='gray', lw=2, label='中立关系')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('文明关系网络图', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename_prefix}_relationships_network.png", dpi=300)
        plt.close()

    def plot_civilization_comparison_radar(self, agents_data, filename="civilization_comparison_radar.png"):
        """绘制文明间属性对比雷达图，用于比较不同文明的综合能力"""
        # 定义要比较的属性
        attributes = ['strength', 'resources', 'population', 'technology', 'territory']
        attribute_names = ['军事力量', '资源储备', '人口数量', '科技水平', '领土面积']
        
        # 提取每个文明的数据并标准化
        data = []
        agent_names = []
        
        for agent in agents_data:
            agent_names.append(f'文明{agent.agent_id}')
            
            # 收集数据并标准化到0-1范围
            agent_data = [
                agent.strength / 100,  # 假设100是最大可能值
                agent.resources / 1000,  # 假设1000是最大可能值
                agent.population / 1000,  # 假设1000是最大可能值
                sum(agent.technology.values()) / 20,  # 假设20是最大可能科技总和
                len(agent.territory) / 50  # 假设50是最大可能领土数
            ]
            
            # 确保数据在0-1范围内
            agent_data = [max(0, min(1, val)) for val in agent_data]
            data.append(agent_data)
        
        # 绘制雷达图
        plt.figure(figsize=(12, 10))
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 为每个文明绘制数据
        for i, (agent_name, agent_data) in enumerate(zip(agent_names, data)):
            # 闭合数据
            agent_data += agent_data[:1]
            
            # 绘制线条
            plt.plot(angles, agent_data, 'o-', linewidth=2, label=agent_name)
            
            # 填充区域
            plt.fill(angles, agent_data, alpha=0.25)
        
        # 添加属性标签
        plt.xticks(angles[:-1], attribute_names)
        
        # 添加标题和图例
        plt.title('文明综合能力对比雷达图', size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 保存图像
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

    def save_to_csv(self, data, filename="civilization_data.csv", headers=None):
        """保存策略数据到CSV"""
        if headers is not None:
            # 使用提供的自定义表头
            header_str = ",".join(headers)
        elif data.shape[1] == 5:
            # 5种策略（扩张、防御、贸易、研发、外交）
            header_str = "expansion_prob,defense_prob,trade_prob,research_prob,diplomacy_prob,resource_utilization"
        elif data.shape[1] == 4:
            # 4种策略（扩张、防御、贸易、研发）
            header_str = "expansion_prob,defense_prob,trade_prob,research_prob,resource_utilization"
        else:
            # 传统的3种策略
            header_str = "expansion_prob,defense_prob,trade_prob,resource_utilization"
        
        np.savetxt(
            f"{self.output_dir}/{filename}",
            data,
            delimiter=",",
            header=header_str,
            comments=""
        )
        
    def save_attribute_history(self, attribute_history, attribute_names, filename="attribute_history.csv"):
        """保存属性历史数据到CSV"""
        df = pd.DataFrame(attribute_history, columns=attribute_names)
        df.to_csv(f"{self.output_dir}/{filename}", index_label="cycle")
        
    def save_technology_data(self, technology_history, filename="technology_data.csv"):
        """保存科技发展数据到CSV"""
        all_data = []
        
        for agent_id, history in technology_history.items():
            for data in history:
                row = {
                    'agent_id': agent_id,
                    'cycle': data['cycle']
                }
                
                # 添加科技数据
                for tech, level in data['technologies'].items():
                    row[f'tech_{tech}'] = level
                
                # 添加当前研发信息
                if 'current_research' in data and data['current_research'] is not None:
                    row['current_research'] = data['current_research']
                    row['research_progress'] = data.get('research_progress', 0)
                    row['research_cost'] = data.get('research_cost', 0)
                
                # 添加科技加成
                if 'tech_bonuses' in data:
                    for bonus, value in data['tech_bonuses'].items():
                        row[f'bonus_{bonus}'] = value
                
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/{filename}", index=False)
        
    def create_summary_report(self, history_data, attribute_history=None, technology_history=None, filename_prefix="summary"):
        """创建模拟结果的综合报告"""
        report_path = f"{self.output_dir}/{filename_prefix}_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("===== 文明演化模拟结果报告 =====\n\n")
            
            # 基本统计信息
            f.write("1. 基本统计信息\n")
            f.write(f"   模拟周期数: {len(history_data)}\n")
            
            # 策略选择统计
            if history_data.shape[1] >= 5:
                f.write("\n2. 策略选择统计\n")
                f.write(f"   平均扩张策略概率: {np.mean(history_data[:, 0]):.3f}\n")
                f.write(f"   平均防御策略概率: {np.mean(history_data[:, 1]):.3f}\n")
                f.write(f"   平均贸易策略概率: {np.mean(history_data[:, 2]):.3f}\n")
                f.write(f"   平均研发策略概率: {np.mean(history_data[:, 3]):.3f}\n")
                f.write(f"   平均外交策略概率: {np.mean(history_data[:, 4]):.3f}\n")
            elif history_data.shape[1] == 4:
                f.write("\n2. 策略选择统计\n")
                f.write(f"   平均扩张策略概率: {np.mean(history_data[:, 0]):.3f}\n")
                f.write(f"   平均防御策略概率: {np.mean(history_data[:, 1]):.3f}\n")
                f.write(f"   平均贸易策略概率: {np.mean(history_data[:, 2]):.3f}\n")
                f.write(f"   平均研发策略概率: {np.mean(history_data[:, 3]):.3f}\n")
            
            # 属性统计
            if attribute_history is not None:
                f.write("\n3. 属性发展统计\n")
                for i, attr in enumerate(attribute_history.columns):
                    if hasattr(attribute_history[attr], 'mean'):
                        f.write(f"   {attr}: 平均值={attribute_history[attr].mean():.3f}, 最大值={attribute_history[attr].max():.3f}\n")
            
            # 科技发展统计
            if technology_history is not None:
                f.write("\n4. 科技发展统计\n")
                for agent_id, history in technology_history.items():
                    if not history:
                        continue
                        
                    last_tech = history[-1]['technologies']
                    total_tech = sum(last_tech.values())
                    top_tech_count = sum(1 for tech in last_tech if tech in self.top_tech_colors and last_tech[tech] >= 3)
                    
                    f.write(f"   文明{agent_id}: 总科技水平={total_tech:.1f}, 顶级科技数量={top_tech_count}\n")
            
            f.write("\n==========================\n")
            f.write(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"综合报告已保存为: {report_path}")

# 测试代码
if __name__ == "__main__":
    # 加载简化版模型输出
    try:
        # 创建模拟数据
        input_features = np.random.rand(1, 1, 10, 10)  # 模拟输入特征
        evolution_history = np.random.rand(100, 4)  # 模拟4种策略概率
        resource_util = np.random.uniform(0.2, 0.8, size=100)  # 模拟资源利用率
        
        # 创建模拟科技历史数据
        tech_history = {}
        for agent_id in range(3):  # 模拟3个文明
            agent_tech = []
            for cycle in range(100):
                tech_levels = {
                    "agriculture": 1 + int(cycle / 20),
                    "military": 1 + int(cycle / 25),
                    "trade": 1 + int(cycle / 30),
                    "science": 1 + int(cycle / 15) + (agent_id * 2)  # 不同文明有差异
                }
                
                # 增加一些随机性
                for tech in tech_levels:
                    tech_levels[tech] += np.random.choice([0, 0, 1], p=[0.8, 0.15, 0.05])
                
                tech_data = {
                    "cycle": cycle,
                    "technologies": tech_levels,
                    "current_research": "science" if cycle % 10 < 5 else None,
                    "research_progress": float(np.random.rand() * 100) if cycle % 10 < 5 else 0,
                    "research_cost": 100.0 if cycle % 10 < 5 else 0,
                    "tech_bonuses": {
                        "resources": 1.0 + (sum(tech_levels.values()) / 100),
                        "strength": 1.0 + (tech_levels["military"] / 10),
                        "research_speed": 1.0 + (tech_levels["science"] / 5)
                    }
                }
                agent_tech.append(tech_data)
            tech_history[agent_id] = agent_tech
        
        # 初始化可视化工具
        visualizer = CivilizationVisualizer()
        
        # 生成可视化结果
        visualizer.plot_strategy_heatmap(
            input_features[0, 0],  # 取第一个特征通道
            title="初始资源分布热力图"
        )
        visualizer.plot_evolution_curve(evolution_history)
        
        # 生成科技可视化结果
        visualizer.plot_technology_progress(tech_history)
        visualizer.plot_tech_tree_comparison(tech_history)
        
        # 保存数据
        combined_data = np.column_stack((evolution_history, resource_util))
        visualizer.save_to_csv(combined_data)
        
        print("可视化结果已保存：strategy_heatmap.png, evolution_curve.png, technology_progress.png, tech_tree_comparison.png, civilization_data.csv")
    except Exception as e:
        print(f"可视化工具测试失败：{str(e)}")