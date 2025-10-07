"""Tests for the heuristic optimizer."""

import torch
from pytest_mock import MockerFixture

from sire.core.optimizer.heuristic import HeuristicOptimizer
from sire.core.profile_tool import AverageModuleStats, AverageProfilingStats, ProfilingData


def _create_mock_profiling_data(module_stats: dict[str, dict[str, float]], execution_order: list[str]) -> ProfilingData:
    """Create a mock ProfilingData object for testing."""
    profiling_data = ProfilingData()
    avg_module_stats: dict[str, AverageModuleStats] = {}
    for name, stats in module_stats.items():
        avg_module_stats[name] = AverageModuleStats(
            avg_exec_time=stats["time"],
            max_peak_vram_delta=int(stats["vram"]),
            weight_size=int(stats["size"]),
        )

    profiling_data.get_avg_stats = lambda: AverageProfilingStats(  # type: ignore
        avg_module_stats=avg_module_stats,
        avg_move_times={},  # Use estimated move times for simplicity
        execution_order=execution_order,
        module_vram_footprint={},
    )
    return profiling_data


def test_heuristic_optimizer_simple_placement(mocker: MockerFixture):
    """Tests a simple case where all modules fit on a single GPU and no prefetching is needed."""
    module_stats = {
        "A": {"size": 100 * 1024**2, "vram": 50 * 1024**2, "time": 0.1},
        "B": {"size": 200 * 1024**2, "vram": 100 * 1024**2, "time": 0.2},
        "C": {"size": 150 * 1024**2, "vram": 70 * 1024**2, "time": 0.15},
    }
    execution_order = ["A", "B", "C"]
    profiling_data = _create_mock_profiling_data(module_stats, execution_order)

    # 1GB GPU, plenty of space
    max_memory_bytes = {"0": 1 * 1024**3, "cpu": 8 * 1024**3}

    optimizer = HeuristicOptimizer(profiling_data, max_memory_bytes)
    # Mock move time to be very high to prevent prefetching
    mocker.patch.object(optimizer, "_estimate_move_time", return_value=100.0)
    plan = optimizer.optimize()

    # All modules should be placed on the GPU
    expected_map = {
        "A": torch.device("cuda:0"),
        "B": torch.device("cuda:0"),
        "C": torch.device("cuda:0"),
    }

    assert not plan.prefetch_schedule, "Should be no prefetching in this simple case"
    # Convert device objects to strings for comparison to avoid device index issues
    plan_map_str = {k: str(v) for k, v in plan.optimized_device_map.items()}
    expected_map_str = {k: str(v) for k, v in expected_map.items()}
    assert plan_map_str == expected_map_str


def test_heuristic_optimizer_with_prefetching(mocker: MockerFixture):
    """Tests a more complex scenario where prefetching is beneficial."""
    mb = 1024**2
    module_stats = {
        "A": {"size": 10 * mb, "vram": 5 * mb, "time": 0.1},  # Small, short
        "B": {"size": 500 * mb, "vram": 100 * mb, "time": 1.0},  # Large, long
        "C": {"size": 100 * mb, "vram": 50 * mb, "time": 0.5},  # Medium, medium
        "D": {"size": 600 * mb, "vram": 200 * mb, "time": 0.2},  # Large, short
    }
    execution_order = ["A", "B", "C", "D"]
    profiling_data = _create_mock_profiling_data(module_stats, execution_order)

    # Two 1GB GPUs
    max_memory_bytes = {"0": 1000 * mb, "1": 1000 * mb, "cpu": 8192 * mb}

    optimizer = HeuristicOptimizer(profiling_data, max_memory_bytes)

    def mock_move_time(size_bytes: int, _src: str, _tgt: str) -> float:
        return (size_bytes / mb) / 5000.0

    # Mock estimate_move_time to return predictable values
    mocker.patch.object(optimizer, "_estimate_move_time", side_effect=mock_move_time)
    plan = optimizer.optimize()

    expected_map = {
        "A": torch.device("cuda:0"),
        "B": torch.device("cuda:0"),
        "C": torch.device("cpu"),
        "D": torch.device("cuda:0"),
    }
    plan_map_str = {k: str(v) for k, v in plan.optimized_device_map.items()}
    expected_map_str = {k: str(v) for k, v in expected_map.items()}
    assert plan_map_str == expected_map_str

    assert len(plan.prefetch_schedule) == 1, "Should have one prefetch instruction"

    pf_instr = plan.prefetch_schedule[0]
    assert pf_instr.module_to_prefetch == "C"
    assert pf_instr.target_device == torch.device("cuda:0")
    assert pf_instr.trigger_module == "A"


def test_heuristic_optimizer_no_gpu():
    """Tests the case where no GPUs are available."""
    module_stats = {
        "A": {"size": 100 * 1024**2, "vram": 0, "time": 0.1},
        "B": {"size": 200 * 1024**2, "vram": 0, "time": 0.2},
    }
    execution_order = ["A", "B"]
    profiling_data = _create_mock_profiling_data(module_stats, execution_order)

    max_memory_bytes = {"cpu": 8 * 1024**3}

    optimizer = HeuristicOptimizer(profiling_data, max_memory_bytes)
    plan = optimizer.optimize()

    # All modules should be placed on the CPU
    expected_map = {
        "A": torch.device("cpu"),
        "B": torch.device("cpu"),
    }

    assert not plan.prefetch_schedule, "Should be no prefetching with no GPUs"
    assert plan.optimized_device_map == expected_map
