"""
性能基准测试脚本

用于测试和比较不同配置下模拟系统的性能。
"""

import time
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from civsim.optimized_multi_agent import OptimizedMultiAgentSimulation
from civsim.multi_agent import MultiAgentSimulation
from civsim.config import SimulationConfig
import numpy as np


class PerformanceBenchmark:
    """性能基准测试工具。"""

    def __init__(self):
        self.results = []

    def run_benchmark(
        self,
        name: str,
        config: SimulationConfig,
        simulation_class,
        enable_vectorization: bool = True,
        enable_compression: bool = True,
        iterations: int = 1,
    ) -> dict:
        """运行基准测试。

        Args:
            name: 测试名称
            config: 模拟配置
            simulation_class: 模拟器类
            enable_vectorization: 是否启用向量化
            enable_compression: 是否启用压缩
            iterations: 迭代次数

        Returns:
            测试结果字典
        """
        times = []
        memory_usage = []

        for i in range(iterations):
            # 固定随机种子确保可重复
            seed = 42 + i

            # 创建模拟器
            start_mem = self._get_memory_usage()

            if simulation_class == OptimizedMultiAgentSimulation:
                config.random_seed = seed
                sim = simulation_class(
                    config=config,
                    enable_vectorization=enable_vectorization,
                    enable_history_compression=enable_compression,
                )
            else:
                config.random_seed = seed
                sim = simulation_class(config=config)

            # 运行模拟
            start_time = time.perf_counter()
            sim.run(config.simulation_cycles)
            elapsed = time.perf_counter() - start_time

            # 记录结果
            times.append(elapsed)
            memory_usage.append(self._get_memory_usage() - start_mem)

        # 计算统计量
        result = {
            "name": name,
            "num_civilizations": config.num_civilizations,
            "simulation_cycles": config.simulation_cycles,
            "grid_size": config.grid_size,
            "class": simulation_class.__name__,
            "vectorization": enable_vectorization,
            "compression": enable_compression,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "mean_memory": np.mean(memory_usage),
            "iterations": iterations,
        }

        self.results.append(result)
        return result

    def _get_memory_usage(self) -> float:
        """获取当前内存使用量 (MB)。

        Returns:
            内存使用量 (MB)
        """
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def print_results(self):
        """打印测试结果。"""
        print("\n" + "=" * 80)
        print("性能基准测试结果")
        print("=" * 80)

        for result in self.results:
            print(f"\n测试: {result['name']}")
            print(f"  类: {result['class']}")
            print(f"  文明数: {result['num_civilizations']}")
            print(f"  周期数: {result['simulation_cycles']}")
            print(f"  网格大小: {result['grid_size']}")
            print(f"  向量化: {result['vectorization']}")
            print(f"  压缩: {result['compression']}")
            print(f"  平均耗时: {result['mean_time']:.2f} ± {result['std_time']:.2f} 秒")
            print(f"  最小耗时: {result['min_time']:.2f} 秒")
            print(f"  最大耗时: {result['max_time']:.2f} 秒")
            print(f"  平均内存: {result['mean_memory']:.2f} MB")

    def compare_implementations(self):
        """比较不同实现的性能。"""
        print("\n" + "=" * 80)
        print("实现对比")
        print("=" * 80)

        # 按配置分组
        configs = {}
        for result in self.results:
            key = (result["num_civilizations"], result["simulation_cycles"])
            if key not in configs:
                configs[key] = []
            configs[key].append(result)

        # 打印对比
        for (num_civs, cycles), results in configs.items():
            print(f"\n配置: {num_civs} 文明, {cycles} 周期")
            print("-" * 80)

            original = next((r for r in results if r["class"] == "MultiAgentSimulation"), None)
            optimized = next(
                (r for r in results if r["class"] == "OptimizedMultiAgentSimulation"), None
            )

            if original and optimized:
                speedup = original["mean_time"] / optimized["mean_time"]
                time_saved = original["mean_time"] - optimized["mean_time"]
                memory_saved = original["mean_memory"] - optimized["mean_memory"]

                print(f"原始版本:")
                print(f"  耗时: {original['mean_time']:.2f} 秒")
                print(f"  内存: {original['mean_memory']:.2f} MB")

                print(f"\n优化版本:")
                print(f"  耗时: {optimized['mean_time']:.2f} 秒")
                print(f"  内存: {optimized['mean_memory']:.2f} MB")

                print(f"\n性能提升:")
                print(f"  加速比: {speedup:.2f}x")
                print(f"  时间节省: {time_saved:.2f} 秒 ({time_saved / original['mean_time']:.1%})")
                print(
                    f"  内存节省: {memory_saved:.2f} MB ({memory_saved / original['mean_memory']:.1%})"
                )

    def save_results(self, filename: str = "benchmark_results.json"):
        """保存测试结果到文件。

        Args:
            filename: 输出文件名
        """
        import json

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n结果已保存到: {filename}")


def run_comprehensive_benchmark():
    """运行全面的性能基准测试。"""
    benchmark = PerformanceBenchmark()

    # 测试配置
    test_configs = [
        ("小规模", SimulationConfig(num_civilizations=4, grid_size=20, simulation_cycles=100)),
        ("中规模", SimulationConfig(num_civilizations=8, grid_size=50, simulation_cycles=500)),
        ("大规模", SimulationConfig(num_civilizations=12, grid_size=100, simulation_cycles=1000)),
    ]

    print("=" * 80)
    print("开始性能基准测试")
    print("=" * 80)

    for name, config in test_configs:
        print(f"\n正在测试: {name}")

        # 原始版本
        print("  运行原始版本...")
        benchmark.run_benchmark(
            name=f"{name}-原始", config=config, simulation_class=MultiAgentSimulation, iterations=3
        )

        # 优化版本 - 所有优化
        print("  运行优化版本...")
        benchmark.run_benchmark(
            name=f"{name}-优化",
            config=config,
            simulation_class=OptimizedMultiAgentSimulation,
            enable_vectorization=True,
            enable_compression=True,
            iterations=3,
        )

    # 打印结果
    benchmark.print_results()
    benchmark.compare_implementations()

    # 保存结果
    benchmark.save_results()


def run_quick_benchmark():
    """运行快速基准测试。"""
    benchmark = PerformanceBenchmark()

    config = SimulationConfig(num_civilizations=6, grid_size=30, simulation_cycles=200)

    print("=" * 80)
    print("快速性能基准测试")
    print("=" * 80)
    print(f"配置: {config.num_civilizations} 文明, {config.simulation_cycles} 周期")

    # 原始版本
    print("\n运行原始版本...")
    benchmark.run_benchmark(
        name="快速-原始", config=config, simulation_class=MultiAgentSimulation, iterations=3
    )

    # 优化版本
    print("\n运行优化版本...")
    benchmark.run_benchmark(
        name="快速-优化",
        config=config,
        simulation_class=OptimizedMultiAgentSimulation,
        enable_vectorization=True,
        enable_compression=True,
        iterations=3,
    )

    # 打印结果
    benchmark.print_results()
    benchmark.compare_implementations()

    # 保存结果
    benchmark.save_results("quick_benchmark_results.json")


def run_optimization_comparison():
    """比较不同优化选项的效果。"""
    benchmark = PerformanceBenchmark()

    config = SimulationConfig(num_civilizations=8, grid_size=50, simulation_cycles=500)

    print("=" * 80)
    print("优化选项对比")
    print("=" * 80)
    print(f"配置: {config.num_civilizations} 文明, {config.simulation_cycles} 周期")

    scenarios = [
        ("无优化", False, False),
        ("仅向量化", True, False),
        ("仅压缩", False, True),
        ("全部优化", True, True),
    ]

    for name, vectorize, compress in scenarios:
        print(f"\n测试: {name}")
        benchmark.run_benchmark(
            name=f"优化-{name}",
            config=config,
            simulation_class=OptimizedMultiAgentSimulation,
            enable_vectorization=vectorize,
            enable_compression=compress,
            iterations=3,
        )

    # 打印结果
    benchmark.print_results()

    # 计算加速比
    baseline = next((r for r in benchmark.results if r["name"] == "优化-无优化"), None)

    if baseline:
        print("\n" + "=" * 80)
        print("优化效果对比 (相对于无优化)")
        print("=" * 80)

        for result in benchmark.results:
            if result["name"] != "优化-无优化":
                speedup = baseline["mean_time"] / result["mean_time"]
                time_saved = baseline["mean_time"] - result["mean_time"]

                print(f"\n{result['name']}:")
                print(f"  加速比: {speedup:.2f}x")
                print(f"  时间节省: {time_saved:.2f} 秒 ({time_saved / baseline['mean_time']:.1%})")

    # 保存结果
    benchmark.save_results("optimization_comparison.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "comprehensive"

    if mode == "comprehensive":
        run_comprehensive_benchmark()
    elif mode == "quick":
        run_quick_benchmark()
    elif mode == "optimization":
        run_optimization_comparison()
    else:
        print(f"未知模式: {mode}")
        print("可用模式:")
        print("  comprehensive - 全面基准测试")
        print("  quick        - 快速基准测试")
        print("  optimization  - 优化选项对比")
