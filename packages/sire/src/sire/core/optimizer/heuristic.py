"""A heuristic-based optimizer for device placement and prefetching."""

import logging
from collections import defaultdict

import torch

from ..profile_tool import ProfilingData
from .plan import OptimizationPlan, PrefetchInstruction

logger = logging.getLogger(__name__)

MIN_PREFETCH_MOVE_TIME = 0.003


def _parse_device_str(device_str: str | int | torch.device) -> torch.device:
    if isinstance(device_str, torch.device):
        return device_str
    if isinstance(device_str, int):
        return torch.device("cuda", device_str)

    s = str(device_str).lower()
    if s in {"cpu", "meta"}:
        return torch.device(s)
    if ":" in s:
        return torch.device(s)
    if s.isdigit():
        return torch.device("cuda", int(s))
    raise ValueError(f"Invalid device string: {device_str}")


class HeuristicOptimizer:
    """
    Analyze profiling data to create an optimized execution plan.

    The plan includes device placements and a prefetching schedule.
    """

    def __init__(self, profiling_data: ProfilingData, max_memory_bytes: dict[str, int]):
        """
        Initialize the optimizer.

        Args:
            profiling_data: The profiling data to use for optimization.
            max_memory_bytes: The maximum memory available on each device.

        """
        self.profiling_data = profiling_data
        self.max_memory_bytes = {str(k): int(v) for k, v in max_memory_bytes.items()}
        avg_stats_obj = self.profiling_data.get_avg_stats()
        self.avg_stats = avg_stats_obj.avg_module_stats
        self.avg_move_times = avg_stats_obj.avg_move_times
        self.execution_order = avg_stats_obj.execution_order
        self.bandwidth_cache: dict[tuple[str, str], float] = {}
        self._init_bandwidth_info()

    def _init_bandwidth_info(self):
        gpus = sorted([k for k in self.max_memory_bytes if k != "cpu"])
        devs = ["cpu", *gpus]
        for s in devs:
            for t in devs:
                if s == t:
                    continue
                self.bandwidth_cache[(s, t)] = 5 if "cpu" in [s, t] else 50  # GB/s
        logger.debug("Initialized bandwidth cache: %s", self.bandwidth_cache)

    def _estimate_move_time(self, size_bytes: int, src: str, tgt: str) -> float:
        if src == tgt or size_bytes == 0:
            return 0.0
        key = str((src, tgt, size_bytes))
        if (
            key in self.avg_move_times
            and self.avg_move_times[key] > 0
            and len(self.profiling_data.move_times.get(key, [])) >= 1
        ):
            logger.debug("Using profiled move time for %s: %.6fs", key, self.avg_move_times[key])
            return self.avg_move_times[key]
        bw = self.bandwidth_cache.get((src, tgt)) or self.bandwidth_cache.get((tgt, src)) or 10
        est = max(size_bytes / (1024**3) / bw + 0.0001, 0.00001)
        logger.debug(
            "Estimated move %.2fMB %s->%s (BW %dGB/s): %.6fs",
            size_bytes / (1024**2),
            src,
            tgt,
            bw,
            est,
        )
        return est

    def optimize(self) -> OptimizationPlan:  # noqa: PLR0912, PLR0915
        """Run the optimization algorithm to generate a plan."""
        logger.info("Starting heuristic optimization (prefetch-focused)...")
        opt_map_str: dict[str, str] = {}
        prefetch_sched: list[PrefetchInstruction] = []
        cpu, gpus = (
            "cpu",
            sorted(
                [k for k in self.max_memory_bytes if k != "cpu" and self.max_memory_bytes[k] > 0],
                key=lambda x: self.max_memory_bytes[x],
                reverse=True,
            ),
        )
        if not gpus:
            logger.warning("No GPUs available/configured. All modules on CPU.")
            return OptimizationPlan({name: torch.device(cpu) for name in self.execution_order}, [])

        gpu_load: dict[str, int] = defaultdict(int)
        benefit_ratio = 1.0
        cursor, exec_len = 0, len(self.execution_order)

        while cursor < exec_len:
            if self.execution_order[cursor] in opt_map_str:
                cursor += 1
                continue
            accum_time = 0.0
            window: list[str] = []
            last_processed_idx = cursor

            for pf_cand_idx in range(cursor, exec_len):
                last_processed_idx = pf_cand_idx
                pf_mod_name = self.execution_order[pf_cand_idx]
                if pf_mod_name in opt_map_str:
                    break  # Window ends if module already placed

                pf_stats = self.avg_stats.get(pf_mod_name)
                if not pf_stats:
                    logger.warning("Stats not found for %s, skipping.", pf_mod_name)
                    if pf_cand_idx == cursor:
                        opt_map_str[pf_mod_name] = cpu
                        break  # Place current on CPU
                    continue  # Skip as prefetch candidate

                tgt_gpu_pf = next(
                    (
                        gpu_id
                        for gpu_id in gpus
                        if gpu_load[gpu_id] + pf_stats.get_runtime_footprint() <= self.max_memory_bytes[gpu_id]
                    ),
                    None,
                )

                if not tgt_gpu_pf:  # No GPU can fit this module for execution
                    if pf_cand_idx == cursor:
                        opt_map_str[pf_mod_name] = cpu
                        break  # Must place on CPU
                    window.append(pf_mod_name)
                    accum_time += pf_stats.avg_exec_time
                    continue  # Add to window, try next

                move_time = self._estimate_move_time(pf_stats.weight_size, cpu, tgt_gpu_pf)
                if accum_time * benefit_ratio > move_time and window and move_time > MIN_PREFETCH_MOVE_TIME:
                    logger.info(
                        "Prefetch %s to %s. Hide: %.4fs, Move: %.4fs. Window: %s",
                        pf_mod_name,
                        tgt_gpu_pf,
                        accum_time,
                        move_time,
                        window,
                    )
                    for mod in window:
                        opt_map_str[mod] = tgt_gpu_pf  # Place window mods on target GPU
                    opt_map_str[pf_mod_name] = cpu  # Prefetched mod initially on CPU
                    prefetch_sched.append(PrefetchInstruction(pf_mod_name, _parse_device_str(tgt_gpu_pf), window[0]))
                    gpu_load[tgt_gpu_pf] += pf_stats.weight_size  # Add weight to static load
                    cursor = pf_cand_idx + 1
                    window = []
                    accum_time = 0.0  # Advance main cursor, reset window
                    break  # From inner prefetch candidate loop
                # Prefetch not viable yet or first item
                window.append(pf_mod_name)
                accum_time += pf_stats.avg_exec_time
            else:  # Inner loop exhausted without break
                cursor = last_processed_idx + 1

            for mod_name in window:  # Place any remaining modules in the last window
                if mod_name in opt_map_str:
                    continue
                stats = self.avg_stats.get(mod_name)
                if not stats:
                    opt_map_str[mod_name] = cpu
                    continue
                tgt_gpu = next(
                    (
                        gid
                        for gid in gpus
                        if gpu_load[gid] + stats.get_runtime_footprint() <= self.max_memory_bytes[gid]
                    ),
                    None,
                )
                if tgt_gpu:
                    opt_map_str[mod_name] = tgt_gpu
                    gpu_load[tgt_gpu] += stats.weight_size
                else:
                    opt_map_str[mod_name] = cpu

        for name in self.execution_order:  # Final check
            if name not in opt_map_str:
                logger.warning("Module '%s' missed. Fallback to CPU.", name)
                opt_map_str[name] = cpu

        final_map = {name: _parse_device_str(dev_str) for name, dev_str in opt_map_str.items()}
        footprints = {name: stats.get_runtime_footprint() for name, stats in self.avg_stats.items()}

        logger.info(
            "Heuristic optimization complete. Map: %d entries. Prefetch: %d instr.",
            len(final_map),
            len(prefetch_sched),
        )
        return OptimizationPlan(final_map, prefetch_sched, footprints)
