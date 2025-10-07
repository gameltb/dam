# type: ignore
"""Tools for profiling PyTorch models."""

from __future__ import annotations

import contextlib
import copy
import csv
import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import dispatch_model
from accelerate.hooks import ModelHook, add_hook_to_module, clear_device_cache
from accelerate.utils import find_device
from torch import nn

from ..utils import human_readable_filesize
from ..utils.hook_manager import HookManager
from ..utils.json_helpers import load_from_json_file, save_to_json_file

_logger = logging.getLogger(__name__)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
PROF_OUT_DIR = Path("/tmp/prof/")

PROF_OUT_DIR.mkdir(exist_ok=True)


@dataclass
class InferenceMemorySizeCSVPoint:
    """A data point for inference memory size."""

    model_cls: str = ""
    batch_size: int = 0
    width: int = 0
    height: int = 0
    embedding_size: int = 0
    inference_memory_size: int = 0
    memory_history_snapshot: str = ""
    model_dtype: str = ""


def csv_dump(objects: list[Any], filename: str) -> None:
    """Dump a list of dataclass objects to a CSV file."""
    with Path(filename).open("w") as f:
        flds = [fld.name for fld in fields(objects[0])]
        w = csv.DictWriter(f, flds)
        w.writeheader()
        w.writerows([asdict(obj) for obj in objects])


def csv_load(object_cls: type, filename: str) -> list[Any]:
    """Load a list of dataclass objects from a CSV file."""
    with Path(filename).open() as f:
        results = csv.DictReader(f)
        return [object_cls(**result) for result in results]


@contextlib.contextmanager
def profile_torch():
    """Profile a block of code with the PyTorch profiler."""
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield

    prof.export_chrome_trace(str(PROF_OUT_DIR / f"{timestamp}.json"))


@contextlib.contextmanager
def record_cuda_memory_history():  # type: ignore
    """Record CUDA memory history for a block of code."""
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    max_num_of_mem_events_per_snapshot: int = 100000

    torch.cuda.memory._record_memory_history(max_entries=max_num_of_mem_events_per_snapshot)  # type: ignore

    try:
        yield
    finally:
        try:
            torch.cuda.memory._dump_snapshot(str(PROF_OUT_DIR / f"{timestamp}.pickle"))  # type: ignore
        except Exception:
            _logger.exception("Failed to capture memory snapshot")

        # Stop recording memory snapshot history.
        torch.cuda.memory._record_memory_history(enabled=None)  # type: ignore


@contextlib.contextmanager
def memory_stats(kwargs: dict[str, Any] | None = None):
    """Record memory statistics for a block of code."""
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    prestats = torch.cuda.memory_stats()
    _logger.info("\n%s", torch.cuda.memory_summary())

    prestats_alloc = prestats["requested_bytes.all.current"]

    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        stats = torch.cuda.memory_stats()
        _logger.info("\n%s", torch.cuda.memory_summary())

        stats_alloc_peak = stats["requested_bytes.all.peak"]

        inference_memory_size = stats_alloc_peak - prestats_alloc
        _logger.info(
            "inference_memory_size : %d (%s)", inference_memory_size, human_readable_filesize(inference_memory_size)
        )

        point = InferenceMemorySizeCSVPoint()
        point.inference_memory_size = inference_memory_size
        point.memory_history_snapshot = timestamp

        if kwargs is not None:
            model = kwargs.get("model")
            latent_image = kwargs.get("latent_image")

            if latent_image is not None:
                batch_size, _, height, width = latent_image["samples"].shape
                point.batch_size = batch_size
                point.width = width
                point.height = height
            if model is not None:
                point.model_cls = model.model.model_config.__class__.__name__
                point.model_dtype = str(model.model_dtype())

        csv_path = PROF_OUT_DIR / "inference_memory_size.csv"
        points = csv_load(InferenceMemorySizeCSVPoint, str(csv_path)) if csv_path.exists() else []
        points.append(point)
        csv_dump(points, str(csv_path))


@dataclass
class ModuleStats:
    """Statistics for a single module."""

    exec_times: list[float] = field(default_factory=list)
    peak_vram_usages: list[int] = field(default_factory=list)
    weight_size: int = 0


@dataclass
class AverageModuleStats:
    """Average statistics for a single module."""

    avg_exec_time: float = 0.0
    max_peak_vram_delta: int = 0
    weight_size: int = 0

    def get_runtime_footprint(self) -> int:
        """Get the total runtime memory footprint of the module."""
        return self.weight_size + self.max_peak_vram_delta


@dataclass
class AverageProfilingStats:
    """Average statistics for all modules in a model."""

    avg_module_stats: dict[str, AverageModuleStats] = field(default_factory=dict)
    avg_move_times: dict[str, float] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    module_vram_footprint: dict[str, int] = field(default_factory=dict)


@dataclass
class ProfilingData:
    """Profiling data for a model."""

    module_stats: defaultdict[str, ModuleStats] = field(default_factory=lambda: defaultdict(ModuleStats))
    move_times: defaultdict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    execution_order: list[str] = field(default_factory=list)
    module_vram_footprint: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the profiling data."""
        self.module_stats = defaultdict(ModuleStats, self.module_stats or {})
        self.move_times = defaultdict(list, self.move_times or {})

    def record_execution(self, name: str, exec_time: float | None, peak_vram_delta: int | None) -> None:
        """Record the execution of a module."""
        if name not in self.execution_order:
            self.execution_order.append(name)
        stats_entry = self.module_stats[name]
        if exec_time is not None:
            stats_entry.exec_times.append(exec_time)
        if peak_vram_delta is not None:
            stats_entry.peak_vram_usages.append(peak_vram_delta)

    def record_weight_size(self, name: str, size: int) -> None:
        """Record the weight size of a module."""
        stats_entry = self.module_stats[name]
        if stats_entry.weight_size == 0 and size > 0:
            stats_entry.weight_size = size
        if name not in self.execution_order:
            self.execution_order.append(name)

    def record_move_time(self, src_dev: torch.device, tgt_dev: torch.device, size: int, move_time: float) -> None:
        """Record the time it takes to move a tensor between devices."""
        key_str = str((str(src_dev), str(tgt_dev), size))
        self.move_times[key_str].append(move_time)

    def calculate_footprints(self) -> None:
        """Calculate the memory footprint of each module."""
        self.module_vram_footprint = {}
        for name, stats_data in self.module_stats.items():
            peak_vram_delta = max(stats_data.peak_vram_usages) if stats_data.peak_vram_usages else 0
            self.module_vram_footprint[name] = stats_data.weight_size + peak_vram_delta

    def get_avg_stats(self) -> AverageProfilingStats:
        """Get the average statistics for all modules."""
        avg_stats_map = {
            name: AverageModuleStats(
                avg_exec_time=sum(data.exec_times) / len(data.exec_times) if data.exec_times else 0.0,
                max_peak_vram_delta=max(data.peak_vram_usages) if data.peak_vram_usages else 0,
                weight_size=data.weight_size,
            )
            for name, data in self.module_stats.items()
        }
        avg_move_times_map = {k: sum(v) / len(v) if v else 0.0 for k, v in self.move_times.items()}
        if not self.module_vram_footprint and self.module_stats:
            self.calculate_footprints()
        return AverageProfilingStats(
            avg_stats_map, avg_move_times_map, list(self.execution_order), dict(self.module_vram_footprint)
        )

    def save(self, filepath: str) -> None:
        """Save the profiling data to a file."""
        self.calculate_footprints()
        save_to_json_file(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> ProfilingData | None:
        """Load profiling data from a file."""
        instance = load_from_json_file(filepath, cls)
        if isinstance(instance, cls):
            instance.__post_init__()
            return instance
        return None


# --- Profiler Internals ---
_current_profiling_data_global: ProfilingData | None = None
_profiling_enabled_global: bool = False


@contextmanager
def _profile_run_context(data_store: ProfilingData):
    global _profiling_enabled_global, _current_profiling_data_global
    if _profiling_enabled_global:
        _logger.warning("Profiling already enabled (re-entrant call).")
    _profiling_enabled_global, _current_profiling_data_global = True, data_store
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except Exception as e:
                _logger.warning("Could not reset peak memory stats for device %d: %s", i, e)
    _logger.info("Profiling run context entered.")
    try:
        yield
    finally:
        _profiling_enabled_global, _current_profiling_data_global = False, None  # Reset to initial state
        _logger.info("Profiling run context exited.")


def get_module_size(module: nn.Module, include_children: bool = True) -> int:
    """Get the size of a module in bytes."""
    s = sum(p.numel() * p.element_size() for p in module.parameters(False) if p.device.type != "meta")
    s += sum(b.numel() * b.element_size() for b in module.buffers(False) if b.device.type != "meta")
    if include_children:
        s += sum(get_module_size(c, True) for c in module.children())
    return s


def infer_fine_grained_device_map(
    model: nn.Module,
    max_memory: dict[str, int] | None,  # keys are str
    no_split: list[str] | None,
) -> dict[str, str]:
    """Infer a fine-grained device map for a model."""
    no_split = no_split or []
    dev_map: dict[str, str] = {}
    frozen: set[str] = set()
    default_dev = "cpu"
    if max_memory:
        gpus = [k for k, v in max_memory.items() if k != "cpu" and v > 0]
        if gpus:
            default_dev = min(gpus)
            _logger.debug("Initial map using %s for no_split.", default_dev)

    def _traverse(mod: nn.Module, path: str = ""):
        nonlocal dev_map, frozen
        cls_name = mod.__class__.__name__
        is_frozen = any(path.startswith(p + ".") for p in frozen if p)
        if is_frozen:
            pass
        elif cls_name in (no_split or []):
            if path:
                dev_map[path] = default_dev
                frozen.add(path)
            for k_rem in [k for k in dev_map if k.startswith(path + ".")]:
                del dev_map[k_rem]
        elif path and (any(True for _ in mod.parameters(False)) or any(True for _ in mod.buffers(False))):
            dev_map[path] = default_dev
        for name, child in mod.named_children():
            child_path = f"{path}.{name}" if path else name
            if not any(child_path.startswith(p + ".") for p in frozen if p) and child_path not in frozen:
                _traverse(child, child_path)

    _traverse(model)
    if not dev_map and (any(True for _ in model.parameters(False)) or any(True for _ in model.buffers(False))):
        dev_map[""] = default_dev
    return dev_map


class Profiler:
    """Encapsulates the logic for running a profiling pass on a torch.nn.Module."""

    def __init__(self, module: nn.Module):
        """Initialize the profiler."""
        self.module = module
        self.hook_manager = HookManager(self.module)

    def run(
        self,
        *args: Any,
        no_split_module_classes: list[str] | None = None,
        max_memory: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> ProfilingData:
        """
        Run a profiling pass on the module.

        This involves temporarily replacing the module's hooks with ProfilerHooks,
        running a warmup and a main inference pass, and collecting data.

        Args:
            *args: Positional arguments for the module's forward pass.
            no_split_module_classes: A list of module class names that should not be split during device mapping.
            max_memory: A dictionary mapping device identifiers to the maximum memory available.
            **kwargs: Keyword arguments for the module's forward pass.

        Returns:
            A ProfilingData object containing the collected statistics.

        """
        _logger.info("%s Profiling Session Start %s", "=" * 20, "=" * 20)
        prof_data = ProfilingData()

        with self.hook_manager.scope():
            _logger.info("Preparing '%s' for profiling.", self.module.__class__.__name__)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            init_map = infer_fine_grained_device_map(self.module, None, no_split_module_classes or [])
            if not init_map and any(self.module.parameters()):
                raise RuntimeError("Failed to create initial device map for profiling.")

            main_dev_prof = (
                torch.device("cuda:0")
                if torch.cuda.is_available() and max_memory and "0" in max_memory
                else torch.device("cpu")
            )
            dispatch_model(self.module, device_map=cast(Any, init_map), main_device=main_dev_prof, force_hooks=True)

            ph_count = 0
            for name, sub_mod in self.module.named_modules():
                if hasattr(sub_mod, "_hf_hook"):
                    add_hook_to_module(sub_mod, ProfilerHook(name), append=True)
                    ph_count += 1
            _logger.info("Registered %d ProfilerHooks.", ph_count)

            p_args, p_kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
            _logger.info("Warm-up profiling inference...")
            with torch.no_grad():
                _ = self.module(*p_args, **p_kwargs)

            clear_device_cache()
            _logger.info("Main profiling inference...")
            with _profile_run_context(prof_data), torch.no_grad():
                _ = self.module(*p_args, **p_kwargs)

        _logger.info("%s Profiling Session End %s", "=" * 20, "=" * 20)
        return prof_data


class ProfilerHook(ModelHook):
    """A hook that records performance and memory statistics for a module."""

    def __init__(self, module_name: str):
        """Initialize the hook."""
        super().__init__()
        self.module_timing_events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.module_start_vram_max: dict[int, int] = {}
        self.module_name = module_name
        self.module: nn.Module | None = None

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Record pre-forward statistics."""
        if not (_profiling_enabled_global and _current_profiling_data_global):
            return args, kwargs
        name, module_id = self.module_name, id(module)
        try:
            if name and _current_profiling_data_global.module_stats[name].weight_size == 0:
                size = get_module_size(module, include_children=False)
                if size > 0:
                    _current_profiling_data_global.record_weight_size(name, size)
                elif name not in _current_profiling_data_global.execution_order:
                    _current_profiling_data_global.record_execution(name, None, None)
            dev = find_device(module.state_dict())
            if dev and dev.type == "cuda":
                try:
                    self.module_start_vram_max[module_id] = torch.cuda.max_memory_allocated(dev)
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    self.module_timing_events[module_id] = (s, e)
                except Exception as e_cuda:
                    _logger.warning("ProfilerHook: VRAM/event pre_fwd fail %s on %s: %s", name, dev, e_cuda)
                    if module_id in self.module_timing_events:
                        del self.module_timing_events[module_id]
                    if module_id in self.module_start_vram_max:
                        del self.module_start_vram_max[module_id]
        except Exception:
            _logger.exception("ProfilerHook: pre_fwd error [%s]", name)
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:  # noqa: PLR0912
        """Record post-forward statistics."""
        if not (_profiling_enabled_global and _current_profiling_data_global):
            return output
        name, module_id = self.module_name, id(module)
        if name and name not in _current_profiling_data_global.execution_order:
            _current_profiling_data_global.record_execution(name, None, None)
        if module_id not in self.module_timing_events and module_id not in self.module_start_vram_max:
            return output

        time_ms, vram_delta = None, None
        dev = find_device(output if output is not None else module.state_dict())
        if dev and dev.type == "cuda":
            try:
                if module_id in self.module_timing_events:
                    s, e = self.module_timing_events[module_id]
                    e.record()
                    torch.cuda.synchronize(dev)
                    time_ms = s.elapsed_time(e)
                if module_id in self.module_start_vram_max:
                    vram_before = self.module_start_vram_max[module_id]
                    vram_after = torch.cuda.max_memory_allocated(dev)
                    vram_delta = max(0, vram_after - vram_before)
            except Exception:
                _logger.exception("ProfilerHook: post_fwd CUDA error [%s on %s]", name, dev)
            finally:
                if module_id in self.module_timing_events:
                    del self.module_timing_events[module_id]
                if module_id in self.module_start_vram_max:
                    del self.module_start_vram_max[module_id]
        if name:
            if _current_profiling_data_global:
                _current_profiling_data_global.record_execution(name, time_ms / 1000.0 if time_ms else None, vram_delta)
        else:
            _logger.warning("ProfilerHook: Skipping recording, missing name for module ID %d.", module_id)
        return output
