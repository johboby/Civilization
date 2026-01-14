"""
Real-time animation visualizer for civilization simulation.

This module provides animation capabilities for visualizing simulation progress in real-time.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Any, Dict, List, Optional, Callable
import io


class AnimationVisualizer:
    """Visualizer for real-time animation of simulation progress."""

    def __init__(self, num_agents: int, grid_size: int = 20):
        """Initialize animation visualizer.

        Args:
            num_agents: Number of civilization agents.
            grid_size: Size of the simulation grid.
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.fig = None
        self.axes = None
        self.agent_colors = cm.get_cmap('tab10')(np.linspace(0, 1, num_agents))

        # Data storage
        self.history = {
            'resources': [[] for _ in range(num_agents)],
            'strength': [[] for _ in range(num_agents)],
            'population': [[] for _ in range(num_agents)],
            'territory': [[] for _ in range(num_agents)]
        }

    def create_animation(
        self,
        simulation_update: Callable,
        num_frames: int = 100,
        interval: int = 100
    ) -> animation.FuncAnimation:
        """Create animation object.

        Args:
            simulation_update: Callable that advances simulation by one step.
            num_frames: Number of animation frames.
            interval: Interval between frames in milliseconds.

        Returns:
            Matplotlib animation object.
        """
        self.fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=self.fig)

        # Territory map subplot
        self.ax_territory = self.fig.add_subplot(gs[0, 0])
        self.ax_territory.set_title('Territory Distribution')
        self.ax_territory.set_xlim(0, self.grid_size)
        self.ax_territory.set_ylim(0, self.grid_size)

        # Resources subplot
        self.ax_resources = self.fig.add_subplot(gs[0, 1])
        self.ax_resources.set_title('Resources Over Time')
        self.ax_resources.set_xlim(0, num_frames)
        self.ax_resources.set_xlabel('Cycle')
        self.ax_resources.set_ylabel('Resources')

        # Strength subplot
        self.ax_strength = self.fig.add_subplot(gs[0, 2])
        self.ax_strength.set_title('Strength Over Time')
        self.ax_strength.set_xlim(0, num_frames)
        self.ax_strength.set_xlabel('Cycle')
        self.ax_strength.set_ylabel('Strength')

        # Population subplot
        self.ax_population = self.fig.add_subplot(gs[1, 0])
        self.ax_population.set_title('Population Over Time')
        self.ax_population.set_xlim(0, num_frames)
        self.ax_population.set_xlabel('Cycle')
        self.ax_population.set_ylabel('Population')

        # Strategy distribution subplot
        self.ax_strategy = self.fig.add_subplot(gs[1, 1])
        self.ax_strategy.set_title('Strategy Distribution')
        self.ax_strategy.set_ylim(0, 1)

        # Events log subplot
        self.ax_events = self.fig.add_subplot(gs[1, 2])
        self.ax_events.set_title('Recent Events')
        self.ax_events.axis('off')

        self.fig.tight_layout()

        # Initialize plots
        self.territory_scatter = self.ax_territory.scatter([], [], s=100, alpha=0.7)
        self.resource_lines = []
        self.strength_lines = []
        self.population_lines = []

        for i in range(self.num_agents):
            line, = self.ax_resources.plot([], [], color=self.agent_colors[i], label=f'Agent {i}')
            self.resource_lines.append(line)
            line, = self.ax_strength.plot([], [], color=self.agent_colors[i], label=f'Agent {i}')
            self.strength_lines.append(line)
            line, = self.ax_population.plot([], [], color=self.agent_colors[i], label=f'Agent {i}')
            self.population_lines.append(line)

        self.ax_resources.legend(loc='upper right', fontsize='small')
        self.ax_strength.legend(loc='upper right', fontsize='small')
        self.ax_population.legend(loc='upper right', fontsize='small')

        # Initialize strategy bar chart
        self.strategy_labels = ['expansion', 'defense', 'trade', 'research']
        self.strategy_bars = self.ax_strategy.bar(
            self.strategy_labels,
            [0.25, 0.25, 0.25, 0.25],
            color=['green', 'blue', 'orange', 'purple']
        )

        # Event text
        self.event_text = self.ax_events.text(
            0.05, 0.95, '', transform=self.ax_events.transAxes,
            fontsize=9, verticalalignment='top'
        )

        def update(frame):
            # Advance simulation
            data = simulation_update()

            # Update territory
            if 'territory_data' in data:
                territories = data['territory_data']
                x_coords = [t[0] for t in territories]
                y_coords = [t[1] for t in territories]
                colors = []
                for i in range(len(territories)):
                    agent_id = territories[i][2] if len(territories[i]) > 2 else 0
                    colors.append(self.agent_colors[agent_id % self.num_agents])
                self.territory_scatter.set_offsets(np.column_stack([x_coords, y_coords]))
                self.territory_scatter.set_color(colors)

            # Update time series
            for i in range(self.num_agents):
                self.history['resources'][i].append(data.get('resources', [])[i] if 'resources' in data else 0)
                self.history['strength'][i].append(data.get('strength', [])[i] if 'strength' in data else 0)
                self.history['population'][i].append(data.get('population', [])[i] if 'population' in data else 0)

                x = range(len(self.history['resources'][i]))
                self.resource_lines[i].set_data(x, self.history['resources'][i])
                self.strength_lines[i].set_data(x, self.history['strength'][i])
                self.population_lines[i].set_data(x, self.history['population'][i])

            # Update strategy distribution
            if 'strategy_distribution' in data:
                strategies = data['strategy_distribution']
                for bar, value in zip(self.strategy_bars, strategies):
                    bar.set_height(value)

            # Update events
            if 'events' in data:
                events_text = '\n'.join(data['events'][-5:])
                self.event_text.set_text(events_text)

            return [self.territory_scatter] + self.resource_lines + self.strength_lines + self.population_lines + list(self.strategy_bars)

        ani = animation.FuncAnimation(
            self.fig, update, frames=num_frames, interval=interval, blit=False
        )

        return ani

    def save_animation(self, ani: animation.FuncAnimation, filename: str = 'simulation_animation.mp4'):
        """Save animation to file.

        Args:
            ani: Animation object to save.
            filename: Output filename.
        """
        try:
            ani.save(filename, writer='ffmpeg', fps=10, dpi=100)
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            print("Try saving as GIF instead...")
            ani.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=10)

    def show(self, ani: animation.FuncAnimation):
        """Show animation.

        Args:
            ani: Animation object to display.
        """
        plt.show()

    def close(self):
        """Close the figure and release resources."""
        if self.fig is not None:
            plt.close(self.fig)


class DashboardVisualizer:
    """Real-time dashboard for simulation monitoring."""

    def __init__(self, num_agents: int):
        """Initialize dashboard visualizer.

        Args:
            num_agents: Number of civilization agents.
        """
        self.num_agents = num_agents
        self.fig = plt.figure(figsize=(12, 8))
        self.agent_colors = cm.get_cmap('tab10')(np.linspace(0, 1, num_agents))

    def create_dashboard(self) -> Dict[str, plt.Axes]:
        """Create dashboard layout.

        Returns:
            Dictionary of subplot axes.
        """
        gs = GridSpec(2, 3, figure=self.fig)

        axes = {
            'leaderboard': self.fig.add_subplot(gs[0, 0]),
            'resources_pie': self.fig.add_subplot(gs[0, 1]),
            'strength_bar': self.fig.add_subplot(gs[0, 2]),
            'trend': self.fig.add_subplot(gs[1, :]),
        }

        # Setup leaderboard
        axes['leaderboard'].set_title('Leaderboard')
        axes['leaderboard'].axis('off')

        # Setup resources pie chart
        axes['resources_pie'].set_title('Resource Distribution')

        # Setup strength bar chart
        axes['strength_bar'].set_title('Military Strength')
        axes['strength_bar'].set_ylim(0, 100)

        # Setup trend plot
        axes['trend'].set_title('Overall Trends')
        axes['trend'].set_xlabel('Cycle')

        self.fig.tight_layout()

        return axes

    def update_dashboard(
        self,
        axes: Dict[str, plt.Axes],
        resources: List[float],
        strengths: List[float],
        populations: List[float]
    ) -> None:
        """Update dashboard with current data.

        Args:
            axes: Dictionary of subplot axes.
            resources: Current resource values.
            strengths: Current strength values.
            populations: Current population values.
        """
        # Update leaderboard
        axes['leaderboard'].clear()
        axes['leaderboard'].axis('off')

        sorted_agents = sorted(
            range(self.num_agents),
            key=lambda i: populations[i] if i < len(populations) else 0,
            reverse=True
        )

        leaderboard_text = "Leaderboard (by population):\n"
        for i, agent_id in enumerate(sorted_agents):
            if agent_id < len(populations):
                leaderboard_text += f"{i+1}. Agent {agent_id}: {int(populations[agent_id])}\n"

        axes['leaderboard'].text(
            0.1, 0.9, leaderboard_text,
            transform=axes['leaderboard'].transAxes,
            fontsize=10, verticalalignment='top',
            family='monospace'
        )

        # Update resources pie chart
        axes['resources_pie'].clear()
        if len(resources) > 0:
            wedges, texts, autotexts = axes['resources_pie'].pie(
                resources,
                labels=[f'Agent {i}' for i in range(len(resources))],
                colors=[self.agent_colors[i % self.num_agents] for i in range(len(resources))],
                autopct='%1.1f%%',
                startangle=90
            )

        # Update strength bar chart
        axes['strength_bar'].clear()
        if len(strengths) > 0:
            x_pos = np.arange(len(strengths))
            bars = axes['strength_bar'].bar(
                x_pos, strengths, color=[self.agent_colors[i % self.num_agents] for i in range(len(strengths))]
            )
            axes['strength_bar'].set_xticks(x_pos)
            axes['strength_bar'].set_xticklabels([f'Agent {i}' for i in range(len(strengths))])
            axes['strength_bar'].set_ylabel('Strength')

    def refresh(self):
        """Refresh the display."""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


__all__ = ['AnimationVisualizer', 'DashboardVisualizer']
