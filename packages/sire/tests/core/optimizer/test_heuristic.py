
import torch

from sire.core.optimizer.heuristic import HeuristicOptimizer
from sire.core.profile_tool import AverageModuleStats, AverageProfilingStats, ProfilingData


def _create_mock_profiling_data(module_stats: dict[str, dict[str, float]], execution_order: list[str]) -> ProfilingData:
    """Creates a mock ProfilingData object for testing."""
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


def test_heuristic_optimizer_simple_placement():
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
    optimizer._estimate_move_time = lambda size_bytes, src, tgt: 100.0  # type: ignore
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


def test_heuristic_optimizer_with_prefetching():
    """Tests a more complex scenario where prefetching is beneficial."""
    MB = 1024**2
    module_stats = {
        "A": {"size": 10 * MB, "vram": 5 * MB, "time": 0.1},  # Small, short
        "B": {"size": 500 * MB, "vram": 100 * MB, "time": 1.0},  # Large, long
        "C": {"size": 100 * MB, "vram": 50 * MB, "time": 0.5},  # Medium, medium
        "D": {"size": 600 * MB, "vram": 200 * MB, "time": 0.2},  # Large, short
    }
    execution_order = ["A", "B", "C", "D"]
    profiling_data = _create_mock_profiling_data(module_stats, execution_order)

    # Two 1GB GPUs
    max_memory_bytes = {"0": 1000 * MB, "1": 1000 * MB, "cpu": 8192 * MB}

    optimizer = HeuristicOptimizer(profiling_data, max_memory_bytes)
    # Mock estimate_move_time to return predictable values
    optimizer._estimate_move_time = lambda size_bytes, src, tgt: (size_bytes / MB) / 5000.0  # type: ignore
    plan = optimizer.optimize()

    # Expected logic:
    # 1. A is small, runs on GPU 0. Its exec time (0.1s) is enough to hide B's move time.
    #    So, B is prefetched.
    #    - A -> cuda:0
    #    - B -> cpu (prefetched to cuda:0)
    #    - D -> cpu (prefetched to cuda:1)  <- This was the initial flawed assumption.
    # Correct trace:
    # 1. Window [A], accum 0.1s.
    # 2. Candidate B, move time 0.1s. 0.1 > 0.1 is false. Window [A, B], accum 1.1s
    # 3. Candidate C, move time 0.02s. 1.1 > 0.02 is true. Prefetch C is viable.
    #    - Place window [A, B] on GPU 0.
    #    - Place C on CPU.
    #    - Schedule prefetch for C to GPU 0, triggered by A.
    #    - gpu_load["0"] becomes 100MB (weight of C).
    # 4. cursor advances past C.
    # 5. Last window is [D]. Place D.
    #    - footprint(D) = 800MB. gpu_load["0"] = 100MB. 100+800 < 1000. Fits on GPU 0.
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
