from __future__ import annotations

import contextlib
import copy
import csv
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

import torch
import torch.nn as nn
from accelerate import dispatch_model
from accelerate.hooks import (
    ModelHook,
    add_hook_to_module,
    clear_device_cache,
)
from accelerate.utils import (
    find_device,
)

from ..utils import human_readable_filesize
from ..utils.hook_manager import HookManager
from ..utils.json_helpers import load_from_json_file, save_to_json_file

_logger = logging.getLogger(__name__)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
PROF_OUT_DIR = "/tmp/prof/"

os.makedirs(PROF_OUT_DIR, exist_ok=True)


@dataclass
class InferenceMemorySizeCSVPoint:
    model_cls: str = ""
    batch_size: int = 0
    width: int = 0
    height: int = 0
    embedding_size: int = 0
    inference_memory_size: int = 0
    memory_history_snapshot: str = ""
    model_dtype: str = ""


def csv_dump(objects: List[Any], filename: str):
    with open(filename, "w") as f:
        flds = [fld.name for fld in fields(objects[0])]
        w = csv.DictWriter(f, flds)
        w.writeheader()
        w.writerows([asdict(object) for object in objects])


def csv_load(object_cls: type, filename: str) -> List[Any]:
    with open(filename, "r") as f:
        results = csv.DictReader(f)
        return [object_cls(**result) for result in results]


@contextlib.contextmanager
def profile_torch():
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

    prof.export_chrome_trace(os.path.join(PROF_OUT_DIR, f"{timestamp}.json"))


@contextlib.contextmanager
def record_cuda_memory_history():  # type: ignore
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)  # type: ignore

    try:
        yield
    finally:
        try:
            torch.cuda.memory._dump_snapshot(os.path.join(PROF_OUT_DIR, f"{timestamp}.pickle"))  # type: ignore
        except Exception as e:
            _logger.error(f"Failed to capture memory snapshot {e}")

        # Stop recording memory snapshot history.
        torch.cuda.memory._record_memory_history(enabled=None)  # type: ignore


@contextlib.contextmanager
def memory_stats(kwargs: Optional[Dict[str, Any]] = None):
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    # When dynamic weights are not involved, we can use this simple method to determine how much memory we need.
    # When the weights are only transferred when needed,
    # we can run them once with a very small input when we don't store the weights at all on the device,
    # determine a minimum value z, and then simply add the curve under normal conditions

    prestats = torch.cuda.memory_stats()
    _logger.info(torch.cuda.memory_summary())

    prestats_alloc = prestats["requested_bytes.all.current"]

    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        stats = torch.cuda.memory_stats()
        _logger.info("\n" + torch.cuda.memory_summary())

        stats_alloc_peak = stats["requested_bytes.all.peak"]

        inference_memory_size = stats_alloc_peak - prestats_alloc
        _logger.info(
            f"inference_memory_size : {inference_memory_size} ({human_readable_filesize(inference_memory_size)})"
        )

        point = InferenceMemorySizeCSVPoint()
        point.inference_memory_size = inference_memory_size
        point.memory_history_snapshot = timestamp

        if kwargs is not None:
            # for SD
            # inference_memory_size = dtype_size * batch_size * width * height * X
            # TODO: find X and where embedding_size

            model = kwargs.get("model", None)
            latent_image = kwargs.get("latent_image", None)

            if latent_image is not None:
                B, _, H, W = latent_image["samples"].shape
                point.batch_size = B
                point.width = W
                point.height = H
            if model is not None:
                point.model_cls = model.model.model_config.__class__.__name__
                point.model_dtype = str(model.model_dtype())

        csv_path = os.path.join(PROF_OUT_DIR, "inference_memory_size.csv")
        if os.path.exists(csv_path):
            points = csv_load(InferenceMemorySizeCSVPoint, csv_path)
        else:
            points = []
        points.append(point)
        csv_dump(points, csv_path)


@dataclass
class ModuleStats:
    exec_times: List[float] = field(default_factory=list)  # type: ignore
    peak_vram_usages: List[int] = field(default_factory=list)  # type: ignore
    weight_size: int = 0


@dataclass
class AverageModuleStats:
    avg_exec_time: float = 0.0
    max_peak_vram_delta: int = 0
    weight_size: int = 0

    def get_runtime_footprint(self) -> int:
        return self.weight_size + self.max_peak_vram_delta


@dataclass
class AverageProfilingStats:
    avg_module_stats: Dict[str, AverageModuleStats] = field(default_factory=dict)  # type: ignore
    avg_move_times: Dict[str, float] = field(default_factory=dict)  # type: ignore
    execution_order: List[str] = field(default_factory=list)  # type: ignore
    module_vram_footprint: Dict[str, int] = field(default_factory=dict)  # type: ignore


@dataclass
class ProfilingData:
    module_stats: DefaultDict[str, ModuleStats] = field(default_factory=lambda: defaultdict(ModuleStats))
    move_times: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    execution_order: List[str] = field(default_factory=list)
    module_VRAM_footprint: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.module_stats = defaultdict(ModuleStats, self.module_stats or {})
        self.move_times = defaultdict(list, self.move_times or {})

    def record_execution(self, name: str, exec_time: Optional[float], peak_vram_delta: Optional[int]):
        if name not in self.execution_order:
            self.execution_order.append(name)
        stats_entry = self.module_stats[name]
        if exec_time is not None:
            stats_entry.exec_times.append(exec_time)
        if peak_vram_delta is not None:
            stats_entry.peak_vram_usages.append(peak_vram_delta)

    def record_weight_size(self, name: str, size: int):
        stats_entry = self.module_stats[name]
        if stats_entry.weight_size == 0 and size > 0:
            stats_entry.weight_size = size
        if name not in self.execution_order:
            self.execution_order.append(name)

    def record_move_time(self, src_dev: torch.device, tgt_dev: torch.device, size: int, move_time: float):
        key_str = str((str(src_dev), str(tgt_dev), size))
        self.move_times[key_str].append(move_time)

    def calculate_footprints(self):
        self.module_VRAM_footprint = {}
        for name, stats_data in self.module_stats.items():
            peak_vram_delta = max(stats_data.peak_vram_usages) if stats_data.peak_vram_usages else 0
            self.module_VRAM_footprint[name] = stats_data.weight_size + peak_vram_delta

    def get_avg_stats(self) -> AverageProfilingStats:
        avg_stats_map = {
            name: AverageModuleStats(
                avg_exec_time=sum(data.exec_times) / len(data.exec_times) if data.exec_times else 0.0,
                max_peak_vram_delta=max(data.peak_vram_usages) if data.peak_vram_usages else 0,
                weight_size=data.weight_size,
            )
            for name, data in self.module_stats.items()
        }
        avg_move_times_map = {k: sum(v) / len(v) if v else 0.0 for k, v in self.move_times.items()}
        if not self.module_VRAM_footprint and self.module_stats:
            self.calculate_footprints()
        return AverageProfilingStats(
            avg_stats_map, avg_move_times_map, list(self.execution_order), dict(self.module_VRAM_footprint)
        )

    def save(self, filepath: str):
        self.calculate_footprints()
        save_to_json_file(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> ProfilingData | None:
        instance = load_from_json_file(filepath, cls)
        if isinstance(instance, cls):
            instance.__post_init__()
            return instance
        return None


# --- Profiler Internals ---
_current_profiling_data_global: Optional[ProfilingData] = None
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
                _logger.warning(f"Could not reset peak memory stats for device {i}: {e}")
    _logger.info("Profiling run context entered.")
    try:
        yield
    finally:
        _profiling_enabled_global, _current_profiling_data_global = False, None  # Reset to initial state
        _logger.info("Profiling run context exited.")


def get_module_size(module: nn.Module, include_children: bool = True) -> int:
    s = sum(p.numel() * p.element_size() for p in module.parameters(False) if p.device.type != "meta")
    s += sum(b.numel() * b.element_size() for b in module.buffers(False) if b.device.type != "meta")
    if include_children:
        s += sum(get_module_size(c, True) for c in module.children())
    return s


def infer_fine_grained_device_map(
    model: nn.Module,
    max_memory: Optional[Dict[str, int]],  # keys are str
    no_split: Optional[List[str]],
    verbose: bool,
) -> Dict[str, str]:
    no_split = no_split or []
    dev_map: Dict[str, str] = {}
    frozen: set[str] = set()
    default_dev = "cpu"
    if max_memory:
        gpus = [k for k, v in max_memory.items() if k != "cpu" and v > 0]
        if gpus:
            default_dev = min(gpus)
            _logger.debug(f"Initial map using {default_dev} for no_split.")

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
    """
    Encapsulates the logic for running a profiling pass on a torch.nn.Module
    to gather performance and memory statistics.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.hook_manager = HookManager(self.module)

    def run(
        self,
        *args: Any,
        no_split_module_classes: Optional[List[str]] = None,
        max_memory: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> ProfilingData:
        """
        Runs a profiling pass on the module.

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
        _logger.info("=" * 20 + " Profiling Session Start " + "=" * 20)
        prof_data = ProfilingData()

        with self.hook_manager.scope():
            # Inside this scope, the module has no hooks. The original hooks will be
            # restored automatically on exit.
            _logger.info(f"Preparing '{self.module.__class__.__name__}' for profiling.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            init_map = infer_fine_grained_device_map(
                self.module, None, no_split_module_classes or [], _logger.isEnabledFor(logging.DEBUG)
            )
            if not init_map and any(self.module.parameters()):
                raise RuntimeError("Failed to create initial device map for profiling.")

            main_dev_prof = (
                torch.device("cuda:0")
                if torch.cuda.is_available() and max_memory and "0" in max_memory
                else torch.device("cpu")
            )
            # dispatch_model will add AlignDevicesHook where needed.
            dispatch_model(self.module, device_map=cast(Any, init_map), main_device=main_dev_prof, force_hooks=True)  # type: ignore

            # Now, append ProfilerHook to the hooks created by dispatch_model.
            ph_count = 0
            for name, sub_mod in self.module.named_modules():
                if hasattr(sub_mod, "_hf_hook"):
                    add_hook_to_module(sub_mod, ProfilerHook(name), append=True)  # type: ignore
                    ph_count += 1
            _logger.info(f"Registered {ph_count} ProfilerHooks.")

            p_args, p_kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
            _logger.info("Warm-up profiling inference...")
            with torch.no_grad():
                _ = self.module(*p_args, **p_kwargs)

            clear_device_cache()
            _logger.info("Main profiling inference...")
            with _profile_run_context(prof_data), torch.no_grad():
                _ = self.module(*p_args, **p_kwargs)

        _logger.info("=" * 20 + " Profiling Session End " + "=" * 20)
        return prof_data


class ProfilerHook(ModelHook):
    def __init__(self, module_name: str):
        super().__init__()
        self.module_timing_events: Dict[int, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.module_start_vram_max: Dict[int, int] = {}
        self.module_name = module_name
        self.module: Optional[nn.Module] = None

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
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
            dev = find_device(module.state_dict())  # type: ignore
            if dev and dev.type == "cuda":
                try:
                    self.module_start_vram_max[module_id] = torch.cuda.max_memory_allocated(dev)  # type: ignore
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    self.module_timing_events[module_id] = (s, e)
                except Exception as e_cuda:
                    _logger.warning(f"ProfilerHook: VRAM/event pre_fwd fail {name} on {dev}: {e_cuda}")
                    if module_id in self.module_timing_events:
                        del self.module_timing_events[module_id]
                    if module_id in self.module_start_vram_max:
                        del self.module_start_vram_max[module_id]
        except Exception as e:
            _logger.error(f"ProfilerHook: pre_fwd error [{name}]: {e}", exc_info=True)
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        if not (_profiling_enabled_global and _current_profiling_data_global):
            return output
        name, module_id = self.module_name, id(module)
        if name and name not in _current_profiling_data_global.execution_order:
            _current_profiling_data_global.record_execution(name, None, None)
        if module_id not in self.module_timing_events and module_id not in self.module_start_vram_max:
            return output

        time_ms, vram_delta = None, None
        dev = find_device(output if output is not None else module.state_dict())  # type: ignore
        if dev and dev.type == "cuda":
            try:
                if module_id in self.module_timing_events:
                    s, e = self.module_timing_events[module_id]
                    e.record()
                    torch.cuda.synchronize(dev)  # type: ignore
                    time_ms = s.elapsed_time(e)  # type: ignore
                if module_id in self.module_start_vram_max:
                    vram_before = self.module_start_vram_max[module_id]
                    vram_after = torch.cuda.max_memory_allocated(dev)  # type: ignore
                    vram_delta = max(0, vram_after - vram_before)
            except Exception as e_post:
                _logger.error(f"ProfilerHook: post_fwd CUDA error [{name} on {dev}]: {e_post}", exc_info=True)
            finally:
                if module_id in self.module_timing_events:
                    del self.module_timing_events[module_id]
                if module_id in self.module_start_vram_max:
                    del self.module_start_vram_max[module_id]
        if name:
            if _current_profiling_data_global:
                _current_profiling_data_global.record_execution(name, time_ms / 1000.0 if time_ms else None, vram_delta)
        else:
            _logger.warning(f"ProfilerHook: Skipping recording, missing name for module ID {module_id}.")
        return output
