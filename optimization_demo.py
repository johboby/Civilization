"""
示例：使用优化后的文明演化模拟系统

本示例展示了如何使用新的优化功能来运行模拟。
"""

import time
import numpy as np
from pathlib import Path

from civsim.optimized_multi_agent import OptimizedMultiAgentSimulation
from civsim.config import get_preset, SimulationConfig
from civsim.multi_agent import MultiAgentSimulation
from civsim.logger import init_logging, get_logger


def run_comparison():
    """对比优化前后的性能。"""
    init_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("文明演化模拟系统 - 性能对比测试")
    logger.info("=" * 60)

    # 测试配置
    configs = [
        ("小规模", SimulationConfig(num_civilizations=4, grid_size=20, simulation_cycles=100)),
        ("中规模", SimulationConfig(num_civilizations=8, grid_size=50, simulation_cycles=500)),
    ]

    for scale_name, config in configs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"测试规模: {scale_name}")
        logger.info(f"{'=' * 60}")
        logger.info(f"文明数量: {config.num_civilizations}")
        logger.info(f"模拟周期: {config.simulation_cycles}")
        logger.info(f"网格大小: {config.grid_size}")

        # 原始版本
        logger.info("\n--- 运行原始版本 ---")
        np.random.seed(42)  # 固定种子确保可重复
        start_time = time.perf_counter()
        sim_original = MultiAgentSimulation(config)
        history_original = sim_original.run(config.simulation_cycles)
        original_time = time.perf_counter() - start_time
        logger.info(f"原始版本耗时: {original_time:.2f} 秒")

        # 优化版本 - 所有优化
        logger.info("\n--- 运行优化版本 (所有优化) ---")
        np.random.seed(42)  # 相同种子
        start_time = time.perf_counter()
        sim_optimized = OptimizedMultiAgentSimulation(
            config=config, enable_vectorization=True, enable_history_compression=True
        )
        history_optimized = sim_optimized.run(config.simulation_cycles)
        optimized_time = time.perf_counter() - start_time
        logger.info(f"优化版本耗时: {optimized_time:.2f} 秒")

        # 获取性能指标
        metrics = sim_optimized.get_performance_metrics()
        logger.info(f"\n性能指标:")
        logger.info(f"  缓存命中率: {metrics.cache_hit_rate():.2%}")
        logger.info(f"  向量化操作: {metrics.vectorized_operations}")

        # 计算性能提升
        speedup = original_time / optimized_time
        time_saved = original_time - optimized_time
        logger.info(f"\n性能提升:")
        logger.info(f"  加速比: {speedup:.2f}x")
        logger.info(f"  时间节省: {time_saved:.2f} 秒 ({time_saved / original_time:.1%})")

        # 保存优化版本结果
        output_dir = Path("results") / scale_name.replace(" ", "_")
        sim_optimized.save_results(str(output_dir))
        logger.info(f"\n结果已保存到: {output_dir}")


def run_optimization_demo():
    """演示各种优化选项。"""
    init_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("优化选项演示")
    logger.info("=" * 60)

    config = SimulationConfig(num_civilizations=6, grid_size=30, simulation_cycles=200)

    scenarios = [
        ("无优化", False, False),
        ("仅缓存", False, False),
        ("缓存 + 向量化", True, False),
        ("所有优化", True, True),
    ]

    results = []

    for name, vectorize, compress in scenarios:
        logger.info(f"\n--- {name} ---")

        np.random.seed(42)
        start_time = time.perf_counter()

        sim = OptimizedMultiAgentSimulation(
            config=config, enable_vectorization=vectorize, enable_history_compression=compress
        )

        sim.run(config.simulation_cycles)

        elapsed = time.perf_counter() - start_time
        metrics = sim.get_performance_metrics()

        results.append(
            {
                "name": name,
                "time": elapsed,
                "vectorized_ops": metrics.vectorized_operations,
                "cache_hit_rate": metrics.cache_hit_rate(),
            }
        )

        logger.info(f"耗时: {elapsed:.2f} 秒")
        logger.info(f"向量化操作: {metrics.vectorized_operations}")
        logger.info(f"缓存命中率: {metrics.cache_hit_rate():.2%}")

    # 打印对比表
    logger.info("\n" + "=" * 60)
    logger.info("性能对比表")
    logger.info("=" * 60)
    logger.info(f"{'方案':<20} {'耗时(秒)':<12} {'向量化操作':<12} {'缓存命中率':<12}")
    logger.info("-" * 60)

    for r in results:
        logger.info(
            f"{r['name']:<20} "
            f"{r['time']:<12.2f} "
            f"{r['vectorized_ops']:<12} "
            f"{r['cache_hit_rate']:<12.2%}"
        )


def run_basic_example():
    """运行基本的优化示例。"""
    init_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("基本优化示例")
    logger.info("=" * 60)

    # 使用预设配置
    config = get_preset("medium")
    logger.info(f"配置: 中等规模")
    logger.info(f"文明数量: {config.num_civilizations}")
    logger.info(f"模拟周期: {config.simulation_cycles}")

    # 创建优化模拟器
    sim = OptimizedMultiAgentSimulation(
        config=config, enable_vectorization=True, enable_history_compression=True
    )

    # 运行模拟
    logger.info("\n开始模拟...")
    start_time = time.perf_counter()
    sim.run(config.simulation_cycles)
    elapsed = time.perf_counter() - start_time

    logger.info(f"\n模拟完成! 耗时: {elapsed:.2f} 秒")

    # 显示性能指标
    metrics = sim.get_performance_metrics()
    logger.info(f"\n性能指标:")
    logger.info(f"  缓存命中率: {metrics.cache_hit_rate():.2%}")
    logger.info(f"  向量化操作: {metrics.vectorized_operations}")

    # 保存结果
    logger.info("\n保存结果...")
    sim.save_results(output_dir="results/demo")
    logger.info("结果已保存到 results/demo/")


def run_validation_demo():
    """演示输入验证功能。"""
    from civsim.validation import (
        validate_positive,
        validate_probability,
        validate_range,
        validate_dict,
        ValidationError,
    )

    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("输入验证演示")
    logger.info("=" * 60)

    # 验证正数
    try:
        value = validate_positive(100, "strength")
        logger.info(f"✓ strength 验证通过: {value}")
    except ValidationError as e:
        logger.warning(f"✗ strength 验证失败: {e}")

    try:
        validate_positive(-10, "strength")
    except ValidationError as e:
        logger.warning(f"✗ strength 验证失败 (预期): {e}")

    # 验证概率
    try:
        value = validate_probability(0.75, "probability")
        logger.info(f"✓ probability 验证通过: {value}")
    except ValidationError as e:
        logger.warning(f"✗ probability 验证失败: {e}")

    try:
        validate_probability(1.5, "probability")
    except ValidationError as e:
        logger.warning(f"✗ probability 验证失败 (预期): {e}")

    # 验证范围
    try:
        value = validate_range(0.5, "stability", 0.0, 1.0)
        logger.info(f"✓ stability 验证通过: {value}")
    except ValidationError as e:
        logger.warning(f"✗ stability 验证失败: {e}")

    try:
        validate_range(1.5, "stability", 0.0, 1.0)
    except ValidationError as e:
        logger.warning(f"✗ stability 验证失败 (预期): {e}")

    # 验证字典
    try:
        tech_dict = validate_dict(
            {"agriculture": 1, "military": 1}, "technology", key_type=str, value_type=int
        )
        logger.info(f"✓ technology 字典验证通过: {tech_dict}")
    except ValidationError as e:
        logger.warning(f"✗ technology 字典验证失败: {e}")


def run_utils_demo():
    """演示工具函数。"""
    from civsim.utils import (
        calculate_bonus_multiplier,
        normalize_probabilities,
        weighted_choice,
        softmax,
        sigmoid,
        format_number,
    )

    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("工具函数演示")
    logger.info("=" * 60)

    # 计算加成倍率
    base_value = 100.0
    bonuses = {"resources": 1.5, "strength": 1.2}
    final_value = calculate_bonus_multiplier(base_value, bonuses, "resources")
    logger.info(f"加成倍率: {base_value} * 1.5 = {final_value}")

    # 归一化概率
    probs = np.array([2.0, 3.0, 5.0])
    normalized = normalize_probabilities(probs)
    logger.info(f"归一化概率: {probs} -> {normalized}")
    logger.info(f"总和: {np.sum(normalized):.6f}")

    # 加权随机选择
    items = ["扩张", "防御", "贸易", "研发"]
    weights = [0.3, 0.2, 0.3, 0.2]
    np.random.seed(42)
    selected = weighted_choice(items, weights)
    logger.info(f"加权选择: {selected}")

    # Softmax
    scores = np.array([1.0, 2.0, 3.0])
    softmax_probs = softmax(scores)
    logger.info(f"Softmax: {softmax_probs}")
    logger.info(f"总和: {np.sum(softmax_probs):.6f}")

    # Sigmoid
    x = 0.0
    sigmoid_value = sigmoid(x, center=0.0, steepness=2.0)
    logger.info(f"Sigmoid(0): {sigmoid_value:.6f}")

    # 格式化数字
    logger.info(f"格式化: {format_number(1234567.89)}")
    logger.info(f"格式化: {format_number(1234.56)}")
    logger.info(f"格式化: {format_number(12.34)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        demo_type = sys.argv[1]
    else:
        demo_type = "basic"

    if demo_type == "basic":
        run_basic_example()
    elif demo_type == "comparison":
        run_comparison()
    elif demo_type == "optimization":
        run_optimization_demo()
    elif demo_type == "validation":
        run_validation_demo()
    elif demo_type == "utils":
        run_utils_demo()
    else:
        print(f"未知演示类型: {demo_type}")
        print("可用演示类型:")
        print("  basic      - 基本优化示例")
        print("  comparison - 性能对比测试")
        print("  optimization - 优化选项演示")
        print("  validation  - 输入验证演示")
        print("  utils      - 工具函数演示")
