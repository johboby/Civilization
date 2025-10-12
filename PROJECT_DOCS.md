# 文明演化模拟系统 - 详细文档

## 1. 项目概述

文明演化模拟系统是一个基于多智能体系统和深度学习的模拟平台，用于研究文明的演化过程。该系统通过模拟多个文明在虚拟环境中的互动、发展和竞争，探索文明演化的规律和可能的路径。

### 1.1 主要功能

- **多智能体模拟**：支持多个独立文明在共享环境中的互动
- **科技树系统**：实现了复杂的科技研发和传承机制
- **资源管理**：模拟资源的分布、消耗和再生
- **策略决策**：文明可以根据环境和自身状态做出策略选择
- **可视化展示**：提供多种图表和可视化方式展示模拟结果
- **命令行界面**：支持交互式和批处理模式运行
- **数据导出**：支持将模拟结果导出为多种格式进行后续分析

## 2. 系统架构

系统由以下几个核心模块组成：

### 2.1 核心模块

- **`multi_agent_simulation.py`**：实现了多智能体模拟的核心逻辑，包括文明智能体的定义、环境交互和演化过程
- **`civilization_visualizer.py`**：提供了多种可视化工具，用于展示模拟结果和分析数据
- **`tech_tree.py`**：实现了科技树系统，定义了科技之间的依赖关系和对文明的影响
- **`simulation_config.py`**：集中管理模拟的各种配置参数

### 2.2 辅助模块

- **`simulation_cli.py`**：命令行界面实现，支持参数配置和交互操作
- **`demo_simulation.py`**：演示脚本，展示系统的主要功能
- **`test_system.py`**：测试脚本，用于验证系统功能是否正常
- **`example_config.py`**：示例配置文件，展示如何自定义模拟参数
- **`run_simulation.bat`**：Windows启动脚本，简化用户操作流程

## 3. 安装指南

### 3.1 环境要求

- Python 3.8 或更高版本
- 依赖包：torch, numpy, matplotlib, pandas, networkx

### 3.2 安装步骤

1. 克隆或下载项目代码到本地
2. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```
3. 运行测试脚本验证安装：
   ```
   python test_system.py
   ```

### 3.3 快速启动

在Windows系统上，可以直接双击 `run_simulation.bat` 文件启动系统，该脚本会自动创建虚拟环境、安装依赖并启动命令行界面。

## 4. 使用说明

### 4.1 命令行界面

系统提供了功能丰富的命令行界面，可以通过以下命令启动：

```
python simulation_cli.py [选项]
```

主要选项包括：

- **基本选项**：
  - `--cycles`：模拟周期数
  - `--civs`：文明数量
  - `--grid`：网格大小
  - `--seed`：随机种子

- **配置选项**：
  - `--config`：自定义配置文件路径

- **输出选项**：
  - `--output`：结果保存目录
  - `--no-save`：不保存结果
  - `--no-visualize`：不生成可视化图表
  - `--save-every`：中间结果保存间隔

- **运行模式**：
  - `--interactive`：交互式模式
  - `--fast-mode`：快速模式
  - `--resume`：从保存的结果恢复模拟

- **高级选项**：
  - `--verbose`：详细日志模式
  - `--quiet`：静默模式
  - `--debug`：调试模式

### 4.2 交互式模式

在交互式模式下，您可以通过键盘命令控制模拟过程：

- `q`：退出模拟
- `v`：立即生成可视化
- `s`：立即保存结果
- `p`：打印当前状态
- `h` 或 `help`：显示帮助信息

### 4.3 使用自定义配置

您可以通过创建自定义配置文件来调整模拟的各种参数。配置文件可以是Python文件（推荐）或JSON文件。

示例（Python配置文件）：

```python
class SimulationConfig:
    # 文明数量
    NUM_CIVILIZATIONS = 6
    # 网格大小
    GRID_SIZE = 150
    # 模拟周期数
    SIMULATION_CYCLES = 300
    # 是否打印详细日志
    PRINT_LOGS = True

# 创建配置实例
example_config = SimulationConfig()
```

使用自定义配置文件：

```
python simulation_cli.py --config your_config.py
```

## 5. 配置参数详解

系统的主要配置参数如下：

### 5.1 基础模拟参数

- **`NUM_CIVILIZATIONS`**：模拟中的文明数量
- **`GRID_SIZE`**：模拟世界的网格大小
- **`SIMULATION_CYCLES`**：模拟运行的周期数
- **`PRINT_LOGS`**：是否打印模拟日志
- **`LOG_INTERVAL`**：日志打印间隔周期数
- **`VISUALIZE_EACH_STEP`**：是否在每一步后生成可视化结果

### 5.2 资源系统参数

- **`RESOURCE_DISTRIBUTION_VARIANCE`**：初始资源分布的随机度（0-1）
- **`RESOURCE_CONSUMPTION_RATE`**：文明对资源的消耗率
- **`RESOURCE_REGENERATION_RATE`**：资源的自然再生率
- **`MAX_RESOURCE_AMOUNT`**：单个格子的最大资源量
- **`MIN_RESOURCE_AMOUNT`**：单个格子的最小资源量

### 5.3 科技研发参数

- **`BASE_RESEARCH_SPEED`**：文明的基础研发速度
- **`RESEARCH_RESOURCE_EFFICIENCY`**：资源对研发的效率加成
- **`RESEARCH_DIFFICULTY_FACTOR`**：科技研发的难度系数
- **`TECH_TREE_EXPANSION_FACTOR`**：科技树扩展的复杂度因子

### 5.4 领土扩张参数

- **`BASE_EXPANSION_PROBABILITY`**：文明的基础扩张概率
- **`EXPANSION_FAILURE_RATE`**：扩张失败的概率
- **`EXPANSION_COST_FACTOR`**：扩张的资源成本系数
- **`INITIAL_TERRITORY_SIZE`**：文明的初始领土大小

### 5.5 文明属性参数

- **`ATTRIBUTE_INITIAL_RANGE`**：属性初始化的随机范围（0-1）
- **`MAX_ATTRIBUTE_DECAY`**：属性的最大衰减率
- **`ATTRIBUTE_IMPROVEMENT_COST`**：提升属性的资源成本
- **`INTER_ATTRIBUTE_INFLUENCE`**：属性间的相互影响系数

### 5.6 策略决策参数

- **`STRATEGY_TRANSITION_SMOOTHNESS`**：策略转换的平滑因子
- **`STRATEGY_EXPLORATION_PROBABILITY`**：策略探索的概率
- **`STRATEGY_EVALUATION_PERIOD`**：策略评估的周期
- **`STRATEGY_ADJUSTMENT_FACTOR`**：策略调整的系数

### 5.7 文明交互参数

- **`BASE_ATTACK_PROBABILITY`**：文明间的基础攻击概率
- **`BASE_DEFENSE_SUCCESS_RATE`**：基础防御成功率
- **`TRADE_SUCCESS_RATE`**：贸易成功的概率
- **`CULTURAL_EXCHANGE_PROBABILITY`**：文化交流的概率

### 5.8 结果输出参数

- **`SAVE_VISUALIZATION`**：是否保存可视化结果
- **`VISUALIZATION_INTERVAL`**：可视化结果的保存间隔
- **`RESULTS_DIR`**：结果保存的目录
- **`SAVE_DETAILED_HISTORY`**：是否保存详细的历史数据

### 5.9 高级参数

- **`RANDOM_SEED`**：随机数生成种子（用于复现结果）
- **`MULTITHREADING`**：是否启用多线程模拟
- **`NUM_THREADS`**：多线程模式下的线程数
- **`PERFORMANCE_OPTIMIZATION`**：性能优化级别（0-3）

## 6. 可视化功能

系统提供了多种可视化方式来展示模拟结果：

### 6.1 图表类型

- **策略热力图**：展示各文明在不同策略上的分布
- **演化趋势图**：展示各种属性随时间的变化趋势
- **科技进展图**：展示各文明的科技研发进度
- **科技树比较图**：比较不同文明的科技树发展路径
- **属性雷达图**：直观展示各文明的属性对比
- **关系网络图**：展示文明之间的互动关系
- **资源分布热力图**：展示资源在地图上的分布情况

### 6.2 自定义可视化

您可以在代码中使用 `CivilizationVisualizer` 类来自定义可视化：

```python
from civilization_visualizer import CivilizationVisualizer

# 初始化可视化器
visualizer = CivilizationVisualizer()

# 生成策略热力图
visualizer.plot_strategy_heatmap(simulation_history)

# 生成演化曲线
visualizer.plot_evolution_curve(simulation_history)

# 生成科技进展图
visualizer.plot_technology_progress(civilization_agents)
```

## 7. 数据导出与分析

系统支持将模拟结果导出为多种格式，便于后续分析：

### 7.1 导出格式

- **NPZ格式**：保存完整的模拟历史数据
- **CSV格式**：保存文明策略和属性数据
- **JSON格式**：保存科技研发历史数据
- **Markdown格式**：生成模拟总结报告

### 7.2 数据导出方法

```python
# 保存完整结果（NPZ格式）
simulation.save_results("results.npz")

# 保存策略数据（CSV格式）
visualizer.save_to_csv(strategy_data, "strategy_data.csv")

# 保存属性历史（CSV格式）
visualizer.save_attribute_history(attribute_data, attribute_names, "attribute_history.csv")

# 保存科技数据（JSON格式）
visualizer.save_technology_data(tech_history, "tech_history.json")

# 创建总结报告（Markdown格式）
visualizer.create_summary_report(agents, history, "simulation_report.md")
```

## 8. 示例用例

### 8.1 基本模拟

运行一个基本的文明演化模拟：

```
python simulation_cli.py --cycles 500 --civs 5 --output basic_simulation
```

### 8.2 自定义参数模拟

使用自定义参数运行模拟：

```
python simulation_cli.py --cycles 300 --civs 8 --grid 150 --seed 42
```

### 8.3 交互式模拟

启动交互式模拟，实时观察和控制模拟过程：

```
python simulation_cli.py --interactive --verbose
```

### 8.4 使用自定义配置

通过配置文件自定义模拟参数：

```
python simulation_cli.py --config example_config.py
```

### 8.5 运行演示脚本

运行演示脚本，快速了解系统功能：

```
python demo_simulation.py
```

## 9. 常见问题与解决方案

### 9.1 运行错误

**问题**：运行时出现 `ModuleNotFoundError`
**解决方案**：确保所有依赖包已正确安装：`pip install -r requirements.txt`

**问题**：可视化图表不显示或保存失败
**解决方案**：检查matplotlib是否正确安装，以及是否有写入权限

**问题**：模拟速度过慢
**解决方案**：使用 `--fast-mode` 参数，或减少文明数量和网格大小

### 9.2 结果分析

**问题**：如何理解模拟结果？
**解决方案**：查看生成的可视化图表和总结报告，重点关注各文明的策略选择、科技发展路径和最终状态

**问题**：如何复现特定的模拟结果？
**解决方案**：使用 `--seed` 参数设置相同的随机种子

### 9.3 自定义扩展

**问题**：如何添加新的科技或策略？
**解决方案**：修改 `tech_tree.py` 文件中的科技树定义，以及 `multi_agent_simulation.py` 中的策略决策逻辑

**问题**：如何自定义可视化效果？
**解决方案**：扩展 `CivilizationVisualizer` 类，添加新的可视化方法

## 10. 扩展方向

文明演化模拟系统有许多可能的扩展方向：

1. **更复杂的经济系统**：引入货币、贸易网络和市场机制
2. **文化和宗教系统**：模拟文化传播、宗教形成和冲突
3. **政治制度模拟**：实现不同政治制度对文明发展的影响
4. **气候变化模型**：引入环境变化对文明发展的挑战
5. **天灾人祸模拟**：添加随机事件如自然灾害、疾病等
6. **深度学习增强**：使用神经网络改进文明的决策能力
7. **多人交互模式**：支持人类玩家参与模拟过程
8. **3D可视化**：提供更直观的三维可视化效果

## 11. 许可证信息

[MIT License](https://opensource.org/licenses/MIT)

## 12. 联系与反馈

如有任何问题或建议，请联系项目维护者。

---

版本：1.0.0
更新日期：2024-01-15


*和我聊天微：cy321one*

*反馈邮箱：samhoclub@163.com

*公众号：尘渊文化*

*官网：www.cycu.top*


【腾讯文档】留言板
https://docs.qq.com/aio/DQVVjemFqaUVFck5H