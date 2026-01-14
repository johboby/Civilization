# 文明演化模拟系统

<div align="center">
  <img src="docs/assets/logo.png" alt="Civilization Evolution Simulation" width="200">
</div>

一个基于多智能体系统的文明演化模拟平台，结合了人工智能、复杂系统理论和博弈论。

## 功能特点

### 核心特性
- 🤖 **多智能体交互**: 模拟多个独立决策的文明，每个文明具有独特的属性和行为模式
- 🌳 **完整科技树系统**: 包含基础、中级、高级、顶级四个等级，共16种科技
- 🎯 **复杂策略决策**: 7种策略类型（扩张、防御、贸易、研发、外交、文化、宗教），基于资源、军事、科技、外交动态调整
- 🗺️ **领土与资源管理**: 模拟资源分布和领土扩张，支持基于地形和气候的动态资源生成

### 高级功能
- 🤝 **文明关系网络**: 盟友和敌对关系的建立与演化，基于行为历史和文化相似度
- 📊 **丰富的可视化**: 热力图、演化曲线、雷达图、关系网络图等
- 💾 **完整数据导出**: CSV、JSON、NPZ格式支持
- 🖥️ **命令行界面**: 交互式和批处理模式，支持参数自定义
- 🎲 **随机事件系统**: 11种事件类型（自然灾害、技术突破、疫情、贸易协定等）

### 智能演化
- 🧬 **高级演化引擎**: 基于博弈论和元认知的智能决策
- 🌍 **文化影响系统**: 模拟文明间的文化传播和影响
- 💎 **复杂资源管理**: 基于地形和气候的动态资源分布
- ⚡ **性能优化**: 向量化操作，支持大规模模拟
- 🎨 **实时动画**: 支持模拟过程的动态可视化

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行演示

#### 基本演示
```bash
python run_demo.py basic
```

#### 高级演化演示
```bash
python run_demo.py advanced
```

#### 新功能演示
```bash
python run_demo.py new-features --save --output-dir my_results
```

### 使用命令行界面

#### 基本使用
```bash
python simulation_cli.py --cycles 100 --num-civs 5
```

#### 交互式模式
```bash
python simulation_cli.py --interactive
```

#### 启用随机事件
```bash
python simulation_cli.py --enable-events --event-modifier 1.5
```

### 使用交互式演示

```bash
# 基本交互式演示
python interactive_demo.py

# 使用自定义配置
python interactive_demo.py --preset medium --num-civs 5 --cycles 200

# 使用特定随机种子
python interactive_demo.py --preset large --seed 12345
```

## 项目结构

```
Civilization/
├── civsim/                    # 核心包
│   ├── __init__.py           # 包初始化和导出
│   ├── config.py             # 配置管理
│   ├── logger.py            # 日志系统
│   ├── strategy.py          # 策略引擎
│   ├── technology.py        # 科技管理
│   ├── events.py           # 随机事件（11种事件）
│   ├── evolution.py         # 高级演化
│   ├── performance.py      # 性能优化
│   ├── animation.py        # 实时动画
│   ├── simulation.py       # 模拟核心
│   ├── multi_agent.py      # 多智能体
│   ├── relationship_manager.py  # 关系管理
│   ├── strategy_executor.py     # 策略执行
│   └── constants.py       # 常量定义
├── results/                  # 模拟结果
├── run_demo.py             # 演示入口
├── simulation_cli.py       # 命令行界面
├── interactive_demo.py     # 交互式演示
├── civilization_visualizer.py  # 可视化工具
├── example_config.py       # 配置示例
├── requirements.txt        # 依赖列表
├── pyproject.toml        # 项目配置
└── README_cn.md          # 本文档
```

## 核心模块

### 配置管理 (`civsim.config`)
- 类型安全的配置类
- 预设配置（demo, small, medium, large, resource_scarcity, tech_focus, advanced_evolution）
- 配置验证

### 策略引擎 (`civsim.strategy`)
- 可插拔决策引擎
- 多种策略类型
- 策略权重计算

### 科技管理 (`civsim.technology`)
- 完整科技树（16种科技，4个等级）
- 科技效果系统（19种属性加成）
- 科技依赖关系

### 随机事件 (`civsim.events`)
11种随机事件类型：
- 自然灾害：小规模自然灾害、重大自然灾害
- 技术突破：意外的技术发现
- 社会改革：提高组织效率
- 资源发现：增加资源储备
- 疫情爆发：人口和稳定性大幅下降
- 移民浪潮：人口增长，带来机遇与挑战
- 贸易协定：促进经济繁荣
- 文化交流：促进技术进步和社会稳定
- 资源枯竭：影响文明发展
- 外交危机：关系恶化，可能引发冲突

### 高级演化 (`civsim.evolution`)
- 博弈论决策
- 元认知学习
- 复杂资源管理
- 文化影响系统

### 性能优化 (`civsim.performance`)
- 向量化操作
- 批量计算
- 优化的模拟循环

### 实时动画 (`civsim.animation`)
- 动画可视化
- 实时更新
- 多种图表类型

## 科技树

科技分为四个等级：

### 基础科技 (等级1)
- **农业** - 提高资源产出和领土增长
- **军事** - 提升军事实力和防御能力
- **贸易** - 增加资源和外交能力
- **科学** - 提高研发速度和科技发现

### 中级科技 (等级2)
- **灌溉系统** - 提高资源产出和领土价值
- **防御工事** - 增强防御能力和稳定性
- **货币系统** - 提高资源和贸易效率
- **工程学** - 提高研发速度和基础设施

### 高级科技 (等级3)
- **工业化农业** - 大幅提高资源和人口增长
- **高级战术** - 提升军事实力和战术优势
- **全球贸易** - 提高资源和外交能力
- **高级科学** - 显著提高研发速度和创新能力

### 顶级科技 (等级4)
- **基因工程** - 提高资源、人口增长和健康水平
- **核技术** - 大幅提升军事实力、防御和能效
- **太空殖民** - 提升领土增长、资源获取和全球影响力
- **人工智能** - 显著提高研发速度、创新和决策质量

## 策略类型

文明可以采用7种策略：

1. **扩张策略** - 获取更多领土和资源
2. **防御策略** - 增强军事实力和防御能力
3. **贸易策略** - 与盟友进行资源交换
4. **研发策略** - 投入资源进行科技研发
5. **外交策略** - 与其他文明建立和维持关系
6. **文化策略** - 推广文化影响力
7. **宗教策略** - 传播宗教信仰

## 性能

### 优化特性
- ✅ NumPy 向量化计算
- ✅ 批量操作优化
- ✅ 内存高效的数据结构
- ✅ 缓存常用计算结果

### 性能指标

| 场景 | 文明数 | 周期数 | 模拟时间 | 峰值内存 |
|--------|--------|--------|---------|----------|
| 小规模 | 3 | 100 | ~0.3s | ~30MB |
| 中规模 | 5 | 200 | ~0.8s | ~50MB |
| 大规模 | 10 | 500 | ~3.5s | ~150MB |
| 超大规模 | 20 | 1000 | ~12s | ~350MB |

## 配置预设

系统提供多种预设配置：

- `demo` - 演示配置（4文明，100周期）
- `small` - 小规模配置（5文明，200周期）
- `medium` - 中等规模配置（8文明，300周期）
- `large` - 大规模配置（10文明，500周期）
- `resource_scarcity` - 资源稀缺配置
- `tech_focus` - 科技优先配置
- `advanced_evolution` - 高级演化配置

## 可视化功能

### 支持的图表类型
- 策略热力图
- 演化趋势曲线
- 科技进展图
- 科技树比较图
- 属性比较图
- 文明综合能力雷达图
- 关系网络图

### 数据导出格式
- CSV - 表格数据
- JSON - 结构化数据
- NPZ - NumPy压缩数据

## 开发

### 设置开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_core.py::test_config_creation -v

# 查看测试覆盖率
pytest tests/ --cov=civsim --cov-report=html
```

### 代码质量

```bash
# 运行类型检查
mypy civsim

# 运行代码格式化
black civsim

# 运行 linting
flake8 civsim
```

## 常见问题

### Q: 如何调整模拟参数？
A: 可以通过命令行参数或自定义配置文件调整参数。参考 `example_config.py` 了解所有可用参数。

### Q: 如何保存和加载模拟状态？
A: 使用 `--save` 参数保存结果，或使用 `--resume <file>` 参数从保存的文件恢复模拟。

### Q: 支持哪些可视化格式？
A: 支持PNG格式的静态图像和GIF格式的动画。

### Q: 如何添加新的随机事件？
A: 在 `civsim/events.py` 中添加新的事件定义和效果函数。

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 致谢

感谢所有贡献者的支持和帮助！

---

<div align="center">
  <sub>Build with ❤️ for simulation research</sub>
</div>

**版本**: v0.2.0
**最后更新**: 2025-12-30
