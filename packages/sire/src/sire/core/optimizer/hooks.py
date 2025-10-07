# type: ignore
"""Hooks for optimizing model inference with device placement and prefetching."""

import functools
import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import accelerate.hooks
import accelerate.utils.modeling
import torch
from accelerate import dispatch_model
from accelerate.hooks import (
    AlignDevicesHook,
    ModelHook,
    SequentialHook,
    add_hook_to_module,
    find_device,
    named_module_tensors,
    remove_hook_from_module,
)
from accelerate.utils import get_balanced_memory
from torch import nn
from torch.cuda import nvtx

from ...utils.json_helpers import save_to_json_file
from ..profile_tool import Profiler, ProfilingData
from .heuristic import HeuristicOptimizer
from .plan import OptimizationPlan
from .signature import ConfigSignatureGenerator, SignatureType

logger = logging.getLogger(__name__)


# hack
accelerate.hooks.set_module_tensor_to_device = functools.partial(
    accelerate.utils.modeling.set_module_tensor_to_device, non_blocking=True, clear_cache=False
)


class PrefetchContext:
    """A context for managing prefetching operations."""

    def __init__(self, plan: OptimizationPlan, model: nn.Module, num_streams: int, offload_policy: str) -> None:
        """
        Initialize the prefetch context.

        Args:
            plan: The optimization plan.
            model: The model being optimized.
            num_streams: The number of CUDA streams to use for prefetching.
            offload_policy: The policy for offloading modules.

        """
        self.plan, self.model = plan, model
        self.module_map: dict[str, nn.Module] = {name: mod for name, mod in model.named_modules()}
        self.num_streams, self.offload_policy = num_streams, offload_policy
        self.stream_mgr = self._init_stream_mgr()
        self.module_pf_streams: dict[str, torch.cuda.Stream] = {}  # module_name -> stream for its prefetch

    def _init_stream_mgr(self) -> dict[str, Any]:
        streams: defaultdict[int, list[torch.cuda.Stream]] = defaultdict(list)
        s_idx: defaultdict[int, int] = defaultdict(int)
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    streams[i] = [torch.cuda.Stream(device=i) for _ in range(self.num_streams)]
            except Exception as e:
                logger.warning("Error initializing CUDA prefetch streams: %s", e)
        return {"streams": streams, "stream_idx": s_idx}

    def get_stream(self, device: torch.device) -> torch.cuda.Stream | None:
        """Get a CUDA stream for a given device."""
        if not (device.type == "cuda" and torch.cuda.is_available() and device.index < torch.cuda.device_count()):
            return None
        pool = self.stream_mgr["streams"].get(device.index, [])
        if not pool:
            return None
        idx = self.stream_mgr["stream_idx"][device.index]
        stream = pool[idx % len(pool)]
        self.stream_mgr["stream_idx"][device.index] = (idx + 1) % len(pool)
        return stream

    def set_module_prefetch_stream(self, name: str, stream: torch.cuda.Stream) -> None:
        """Set the prefetch stream for a module."""
        self.module_pf_streams[name] = stream

    def get_module_prefetch_stream(self, name: str) -> torch.cuda.Stream | None:
        """Get the prefetch stream for a module."""
        return self.module_pf_streams.get(name)

    def clear_all_module_prefetch_streams(self) -> None:
        """Clear all module prefetch streams."""
        self.module_pf_streams.clear()


class PrefetchingWaitHook(ModelHook):
    """A hook that waits for a prefetch operation to complete."""

    def __init__(self, ctx: PrefetchContext, name: str, mod_inst: nn.Module, exec_dev: torch.device) -> None:
        """
        Initialize the hook.

        Args:
            ctx: The prefetch context.
            name: The name of the module.
            mod_inst: The module instance.
            exec_dev: The execution device.

        """
        super().__init__()
        self.ctx, self.name, self.mod_inst, self.exec_dev = ctx, name, mod_inst, exec_dev
        self.tied_ptrs_to_rm: set[tuple[int, torch.device]] = set()
        self.pf_submod_hf_hook: AlignDevicesHook | None = None
        logger.debug("PrefetchingWaitHook for %s on %s", self.name, self.exec_dev)

    @torch.compiler.disable()
    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Wait for the prefetch stream before the forward pass."""
        if module is not self.mod_inst:
            logger.warning("WaitHook for %s called on wrong mod", self.name)
            return args, kwargs
        pf_stream = self.ctx.get_module_prefetch_stream(self.name)
        if pf_stream:
            comp_stream = torch.cuda.current_stream(self.exec_dev)
            logger.debug(
                "Mod %s on dev %s (stream %d) waiting for pf stream %d",
                self.name,
                self.exec_dev,
                comp_stream.stream_id,
                pf_stream.stream_id,
            )
            comp_stream.wait_stream(pf_stream)
            if self.pf_submod_hf_hook:
                self.pf_submod_hf_hook.tied_pointers_to_remove = self.tied_ptrs_to_rm
        return args, kwargs


class AlignDevicesHookTorchCompilerDisable(AlignDevicesHook):
    """An AlignDevicesHook that disables the torch compiler."""

    @classmethod
    def from_align_devices_hook(cls, align_devices_hook: AlignDevicesHook) -> "AlignDevicesHookTorchCompilerDisable":
        """Create a new hook from an existing AlignDevicesHook."""
        align_devices_hook.__class__ = cls
        return align_devices_hook

    @torch.compiler.disable()
    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Execute the pre_forward hook with the torch compiler disabled."""
        return super().pre_forward(module, *args, **kwargs)

    @torch.compiler.disable()
    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Execute the post_forward hook with the torch compiler disabled."""
        return super().post_forward(module, output)


class PrefetchingHook(ModelHook):  # Placed on trigger module
    """A hook that triggers prefetching operations."""

    def __init__(self, ctx: PrefetchContext, name: str, mod_inst: nn.Module) -> None:
        """
        Initialize the hook.

        Args:
            ctx: The prefetch context.
            name: The name of the module.
            mod_inst: The module instance.

        """
        super().__init__()
        self.ctx, self.name, self.mod_inst = ctx, name, mod_inst
        logger.debug("PrefetchingHook (trigger) for %s", self.name)

    @torch.compiler.disable()
    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Trigger prefetching before the forward pass."""
        if module is not self.mod_inst:
            logger.warning("TriggerHook for %s called on wrong mod", self.name)
            return args, kwargs
        logger.debug("TriggerHook pre_fwd for %s (id %d)", self.name, id(module))
        nvtx.range_push(f"pf_trigger_{self.name}")
        trigger_dev = find_device(module.state_dict())

        for instr in self.ctx.plan.trigger_index.get(self.name, []):
            pf_mod_name, pf_tgt_dev = instr.module_to_prefetch, instr.target_device
            mod_to_pf = self.ctx.module_map.get(pf_mod_name)
            if not mod_to_pf:
                logger.warning("Mod '%s' for prefetch not found", pf_mod_name)
                continue
            pf_stream = self.ctx.get_stream(pf_tgt_dev)
            if not pf_stream:
                logger.warning("No pf stream for %s. Cannot prefetch %s.", pf_tgt_dev, pf_mod_name)
                continue

            if trigger_dev and trigger_dev.type == "cuda" and pf_tgt_dev.type == "cuda":
                pf_stream.wait_stream(torch.cuda.current_stream(trigger_dev))
            self.do_prefetch(pf_mod_name, pf_tgt_dev, mod_to_pf, pf_stream)
        nvtx.range_pop()
        return args, kwargs

    def do_prefetch(  # noqa: PLR0912, PLR0915
        self, pf_name: str, pf_dev: torch.device, pf_mod: nn.Module, pf_stream: torch.cuda.Stream
    ) -> None:
        """Perform the prefetch operation."""
        nvtx.range_push(f"pf_task_{pf_name}_on_{pf_stream.stream_id}")
        try:
            pf_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(pf_stream):
                hook = getattr(pf_mod, "_hf_hook", None)
                align_hook, wait_hook = None, None
                if isinstance(hook, SequentialHook):
                    for h_ in hook.hooks:
                        if isinstance(h_, AlignDevicesHook):
                            align_hook = h_
                        elif isinstance(h_, PrefetchingWaitHook):
                            wait_hook = h_
                elif isinstance(hook, AlignDevicesHook):
                    align_hook = hook

                if not align_hook:
                    logger.error("No AlignHook on %s. Abort prefetch.", pf_name)
                    nvtx.range_pop()
                    return
                if not wait_hook and self.ctx.plan.prefetch_schedule:
                    logger.error("No WaitHook on %s with active prefetch. Abort.", pf_name)
                    nvtx.range_pop()
                    return

                align_hook = AlignDevicesHookTorchCompilerDisable.from_align_devices_hook(align_hook)

                if wait_hook:
                    wait_hook.tied_ptrs_to_rm.clear()
                    wait_hook.pf_submod_hf_hook = align_hook
                    wait_hook.exec_dev = pf_dev

                w_map, tied_map = (
                    getattr(align_hook, "weights_map", None),
                    getattr(align_hook, "tied_params_map", None),
                )
                if w_map is None or tied_map is None:
                    logger.error("AlignHook on %s not init. No prefetch.", pf_name)
                    nvtx.range_pop()
                    return
                align_hook.execution_device = pf_dev

                for name, val in named_module_tensors(pf_mod, True, False, True):
                    if val.device.type == "meta":
                        map_val = w_map.get(name)
                        if map_val is None:
                            logger.warning("Meta tensor '%s' in '%s' no value in weights_map.", name, pf_name)
                            continue
                        if map_val.device.type == "cpu" and not map_val.is_pinned():
                            try:
                                map_val = map_val.pin_memory()
                                w_map.dataset.state_dict[w_map.prefix + name] = map_val
                            except RuntimeError as e_pin:
                                logger.warning("Could not pin %s for prefetch: %s. Using unpinned.", name, e_pin)

                        if map_val.data_ptr() not in tied_map:
                            tied_map[map_val.data_ptr()] = {}

                        tgt_tensor = torch.empty_like(map_val, device=pf_dev)
                        tgt_tensor.copy_(map_val, non_blocking=True)
                        param_tgt = torch.nn.Parameter(tgt_tensor, requires_grad=map_val.requires_grad)
                        tied_map[map_val.data_ptr()][pf_dev] = param_tgt
                        if wait_hook:
                            wait_hook.tied_ptrs_to_rm.add((map_val.data_ptr(), pf_dev))
                    elif val.device != pf_dev:
                        logger.debug("Prefetching existing tensor %s from %s to %s", name, val.device, pf_dev)
                        val.data = val.data.to(pf_dev, non_blocking=True)
                logger.debug("Prefetch task for %s to %s submitted on stream %d.", pf_name, pf_dev, pf_stream.stream_id)
            self.ctx.set_module_prefetch_stream(pf_name, pf_stream)
        except Exception:
            logger.exception("Error in do_prefetch for %s", pf_name)
        finally:
            nvtx.range_pop()


class InferenceOptimizerHook(ModelHook):
    """A hook that optimizes model inference."""

    def __init__(  # noqa: PLR0913
        self,
        cache_dir: str = "opt_cache",
        num_prefetch_streams: int = 1,
        no_split_module_classes: list[str] | None = None,
        custom_signature_callback: Callable[..., Any] | None = None,
        default_offload_policy: str = "cpu",
        force_profiling: bool = False,
        run_profiling_if_needed: bool = True,
        max_memory_gb: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize the hook.

        Args:
            cache_dir: The directory to cache optimization plans.
            num_prefetch_streams: The number of CUDA streams for prefetching.
            no_split_module_classes: A list of module classes that should not be split across devices.
            custom_signature_callback: A callback to generate a custom signature for the model.
            default_offload_policy: The default policy for offloading modules.
            force_profiling: Whether to force profiling even if a cached plan exists.
            run_profiling_if_needed: Whether to run profiling if no cached plan is found.
            max_memory_gb: A dictionary mapping device IDs to the maximum memory in GB.

        """
        super().__init__()
        self.base_cache_dir = Path(cache_dir)
        self.base_cache_dir.mkdir(exist_ok=True)
        self.num_prefetch_streams = max(1, num_prefetch_streams)
        self.no_split_module_classes = no_split_module_classes or []
        self.custom_signature_callback = custom_signature_callback
        self.default_offload_policy = default_offload_policy
        self.force_profiling = force_profiling  # User's initial setting
        self._force_profiling_active = force_profiling  # Internal flag, reset after use
        self.run_profiling_if_needed = run_profiling_if_needed
        self.user_max_memory_gb = max_memory_gb

        self.sig_gen = ConfigSignatureGenerator()
        self.current_plan: OptimizationPlan | None = None
        self.current_config_sig_hash: str | None = None
        self.current_input_sig_hash: str | None = None
        self.current_max_memory_bytes: dict[str, int] | None = None
        self.current_plan_id: str | None = None
        self.last_module_id_processed: int | None = None
        self.active_pf_ctx: PrefetchContext | None = None
        self.hooked_module_instance: nn.Module | None = None
        self.current_module_dtype: torch.dtype | None = None
        self.module: nn.Module | None = None
        self.cpu_state_dict: dict[str, torch.Tensor] = {}
        logger.info("IOHook init. Cache: %s, Prefetch: %d", self.base_cache_dir, self.num_prefetch_streams)

    def _get_max_mem_bytes(self) -> dict[str, int]:  # keys are str
        mem_map: dict[str, int] = {}
        if self.user_max_memory_gb:
            for k, v in self.user_max_memory_gb.items():
                mem_map[str(k)] = int(v * (1024**3))
            mem_str = {k: f"{val / (1024**3):.1f}GB" for k, val in mem_map.items()}
            logger.info("User max_mem: %s", mem_str)
        else:
            logger.info("Auto-balancing memory.")
            if not self.hooked_module_instance:
                logger.error("Cannot auto-balance: no module.")
                return {"cpu": 64 * (1024**3)}
            try:
                balanced = get_balanced_memory(
                    self.hooked_module_instance,
                    dtype=self.current_module_dtype,
                    no_split_module_classes=self.no_split_module_classes,
                )
                if balanced:
                    mem_map = {str(k): v for k, v in balanced.items()}
                mem_str = {k: f"{val / (1024**3):.1f}GB" for k, val in mem_map.items()}
                logger.info("Auto-balanced max_mem: %s", mem_str)
            except Exception:
                logger.exception("Auto-balance fail")
                mem_map = {"cpu": 64 * (1024**3)}
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        if str(i) not in mem_map:
                            mem_map[str(i)] = 1 * (1024**3)
        return mem_map

    def _get_sig_dir_path(self) -> Path:  # For current_config_sig_hash
        if not self.current_config_sig_hash:
            logger.error("Config sig hash None. Cannot get dir.")
            return self.base_cache_dir / "_ERR_NO_CONF_SIG"
        d = self.base_cache_dir / self.current_config_sig_hash
        d.mkdir(exist_ok=True)
        return d

    def _get_plan_file_path(self) -> Path | None:  # For current_plan_id
        if not self.current_plan_id:
            logger.error("Plan ID None. Cannot get path.")
            return None
        d = self._get_sig_dir_path() / "plans"
        d.mkdir(exist_ok=True)
        return d / f"{self.current_plan_id}.json"

    def _gen_opt_plan(self, prof_data: ProfilingData, max_mem_plan: dict[str, int]) -> OptimizationPlan | None:
        logger.info("Generating optimization plan...")
        if not self.hooked_module_instance:
            logger.error("Cannot gen plan: no module.")
            return None
        opt = HeuristicOptimizer(prof_data, max_mem_plan)
        return opt.optimize()

    def _setup_module_with_plan(self, mod_to_opt: nn.Module, plan: OptimizationPlan) -> PrefetchContext | None:
        logger.info("Preparing/dispatching '%s' with plan...", mod_to_opt.__class__.__name__)
        offload = self.default_offload_policy
        if any(d.type == "cpu" for d in plan.optimized_device_map.values()) and offload != "cpu":
            offload = "cpu"

        pf_ctx = PrefetchContext(plan, mod_to_opt, self.num_prefetch_streams, offload)
        remove_hook_from_module(mod_to_opt, recurse=True)  # Remove IOH and any prior hooks

        self.cpu_state_dict = mod_to_opt.state_dict()

        main_dev_dispatch = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        map_for_dispatch: dict[str, Any] = {
            k: ("cpu" if v.type == "cpu" else f"{v.type}:{v.index}") for k, v in plan.optimized_device_map.items()
        }
        dispatch_model(
            mod_to_opt,
            device_map=map_for_dispatch,
            main_device=main_dev_dispatch,
            force_hooks=True,
            state_dict=self.cpu_state_dict,
        )

        trig_mods = {i.trigger_module for i in plan.prefetch_schedule}
        pf_mods = {i.module_to_prefetch for i in plan.prefetch_schedule}
        waits, trigs = 0, 0
        for name, sub_mod in mod_to_opt.named_modules():
            if not hasattr(sub_mod, "_hf_hook"):
                continue
            sub_mod_exec_dev = plan.optimized_device_map.get(name, main_dev_dispatch)
            if name in pf_mods:
                add_hook_to_module(sub_mod, PrefetchingWaitHook(pf_ctx, name, sub_mod, sub_mod_exec_dev), True)
                waits += 1
            if name in trig_mods:
                add_hook_to_module(sub_mod, PrefetchingHook(pf_ctx, name, sub_mod), True)
                trigs += 1

        add_hook_to_module(mod_to_opt, self, append=True)
        logger.info("Registered %d WaitHooks and %d TriggerHooks.", waits, trigs)
        return pf_ctx

    @torch.compiler.disable()
    def pre_forward(  # noqa: PLR0911, PLR0912, PLR0915
        self, module: nn.Module, *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Prepare for the forward pass, handling profiling and planning."""
        nvtx.range_push("IOHook.full_forward")
        nvtx.range_push("IOHook.pre_forward")
        self.hooked_module_instance = module

        # One-time setup
        if self.current_module_dtype is None:
            logger.info("Hook first fwd: %s (id:%d)", module.__class__.__name__, id(module))
            try:
                p_dt = next((p.dtype for p in module.parameters(False)), None)
                b_dt = next((b.dtype for b in module.buffers(False)), None)
                self.current_module_dtype = getattr(module, "dtype", p_dt or b_dt or torch.get_default_dtype())
            except Exception as e:
                self.current_module_dtype = torch.get_default_dtype()
                logger.warning("Dtype infer fail: %s. Using %s", e, self.current_module_dtype)
            logger.info("Module dtype: %s", self.current_module_dtype)
            self.current_max_memory_bytes = self._get_max_mem_bytes()

        if not all([self.hooked_module_instance, self.current_module_dtype, self.current_max_memory_bytes]):
            logger.error("Hook state not initialized.")
            nvtx.range_pop()
            return args, kwargs

        # Determine if a plan update is needed
        needs_plan_update = (
            id(module) != self.last_module_id_processed or self.current_plan is None or self._force_profiling_active
        )
        input_sig_changed = False
        if not needs_plan_update:
            if not self.current_module_dtype:
                logger.error("Module dtype not set, cannot generate signature.")
                nvtx.range_pop()
                return args, kwargs
            new_input_sig, _ = self.sig_gen.generate_config_signature(
                module,
                args,
                kwargs,
                self.current_module_dtype,
                self.custom_signature_callback,
                level=SignatureType.INPUT_ONLY,
            )
            if new_input_sig != self.current_input_sig_hash:
                logger.info("Input signature changed, triggering plan update.")
                needs_plan_update = True
                input_sig_changed = True

        if needs_plan_update:
            logger.info(
                "Plan update triggered. ModID_changed:%s, NoPlan:%s, ForceProf:%s, InputSig_changed:%s",
                id(module) != self.last_module_id_processed,
                self.current_plan is None,
                self._force_profiling_active,
                input_sig_changed,
            )

            if not self.current_module_dtype or not self.hooked_module_instance:
                logger.error("Module dtype not set, cannot update plan.")
                nvtx.range_pop()
                return args, kwargs

            # Generate model config signature (with weight shapes) to find matching profiling data
            new_conf_hash, raw_details = self.sig_gen.generate_config_signature(
                self.hooked_module_instance,
                args,
                kwargs,
                self.current_module_dtype,
                self.custom_signature_callback,
                level=SignatureType.WITH_WEIGHT_SHAPES,
            )
            if not self.current_max_memory_bytes:
                logger.error("Max memory not set, cannot generate plan identifier.")
                nvtx.range_pop()
                return args, kwargs
            new_plan_id = self.sig_gen.generate_plan_identifier(self.current_max_memory_bytes)
            self.current_config_sig_hash, self.current_plan_id = new_conf_hash, new_plan_id

            sig_dir = self._get_sig_dir_path()
            prof_data_path = sig_dir / "profiling_data.json"
            plan_path = self._get_plan_file_path()

            prof_data: ProfilingData | None = None
            just_profiled = False
            if not self._force_profiling_active:
                prof_data = ProfilingData.load(str(prof_data_path))

            if prof_data is None:
                if self.run_profiling_if_needed or self._force_profiling_active:
                    logger.info(
                        "Profiling for conf %s. Force=%s", self.current_config_sig_hash, self._force_profiling_active
                    )
                    profiler = Profiler(self.hooked_module_instance)
                    prof_data = profiler.run(
                        *args,
                        no_split_module_classes=self.no_split_module_classes,
                        max_memory=self.current_max_memory_bytes,
                        **kwargs,
                    )
                    just_profiled = True
                    if prof_data:
                        prof_data.save(str(prof_data_path))
                        raw_details_path = sig_dir / "raw_signature_details.json"
                        try:
                            save_to_json_file(raw_details, str(raw_details_path))
                        except Exception:
                            logger.exception("Failed to save raw signature details")
                    else:
                        logger.error("Profiling failed, no data returned.")
                        self.current_plan = None
                        self.active_pf_ctx = None
                        nvtx.range_pop()
                        return args, kwargs
                else:
                    logger.error(
                        "Profiling data not found for %s and profiling is disabled.", self.current_config_sig_hash
                    )
                    self.current_plan = None
                    self.active_pf_ctx = None
                    nvtx.range_pop()
                    return args, kwargs
            else:
                logger.info("Using existing profiling data from %s.", prof_data_path)
                raw_details_path = sig_dir / "raw_signature_details.json"
                if not raw_details_path.exists():
                    try:
                        save_to_json_file(raw_details, str(raw_details_path))
                    except Exception:
                        logger.exception("Failed to save raw signature details")

            self.current_plan = None
            if not just_profiled and plan_path:
                self.current_plan = OptimizationPlan.load(str(plan_path))

            if self.current_plan is None:
                if not prof_data or not self.current_max_memory_bytes:
                    logger.error("Plan gen fail: no prof_data or max_memory.")
                    self.active_pf_ctx = None
                    nvtx.range_pop()
                    return args, kwargs
                logger.info(
                    "Generating plan for plan_id %s (config %s)", self.current_plan_id, self.current_config_sig_hash
                )
                self.current_plan = self._gen_opt_plan(prof_data, self.current_max_memory_bytes)
                if self.current_plan and plan_path:
                    self.current_plan.save(str(plan_path))
                elif not self.current_plan:
                    logger.error("Plan gen fail.")
                    self.active_pf_ctx = None
                    nvtx.range_pop()
                    return args, kwargs
            else:
                logger.info("Loaded existing plan from %s", plan_path)

            if self.active_pf_ctx:
                self.active_pf_ctx.clear_all_module_prefetch_streams()
            self.active_pf_ctx = self._setup_module_with_plan(self.hooked_module_instance, self.current_plan)
            if not self.active_pf_ctx:
                logger.error("Failed to prep/dispatch module.")
                self.current_plan = None
                nvtx.range_pop()
                return args, kwargs

            # After a successful plan setup, store the input signature for future checks
            if not self.current_module_dtype:
                logger.error("Module dtype not set, cannot generate signature.")
                nvtx.range_pop()
                return args, kwargs
            self.current_input_sig_hash, _ = self.sig_gen.generate_config_signature(
                module,
                args,
                kwargs,
                self.current_module_dtype,
                self.custom_signature_callback,
                level=SignatureType.INPUT_ONLY,
            )
            self.last_module_id_processed = id(self.hooked_module_instance)
            self._force_profiling_active = False

            # Arg alignment by the module's current hook (set up by _setup_module_with_plan).
            hf_hook = getattr(self.hooked_module_instance, "_hf_hook", None)
            if isinstance(hf_hook, SequentialHook):
                align_hook = next(
                    (h for h in hf_hook.hooks if isinstance(h, AlignDevicesHook) and not isinstance(h, type(self))),
                    None,
                )
                if align_hook:
                    logger.debug("Manually calling pre_forward of AlignDevicesHook: %s", type(align_hook))
                    result = align_hook.pre_forward(self.hooked_module_instance, *args, **kwargs)
                    args, kwargs = result[0], result[1]
                else:
                    logger.warning("No AlignDevicesHook found in SequentialHook post-setup.")
            elif hf_hook is not self and isinstance(hf_hook, AlignDevicesHook):
                logger.debug("Manually calling pre_forward of sole AlignDevicesHook: %s", type(hf_hook))
                result = hf_hook.pre_forward(self.hooked_module_instance, *args, **kwargs)
                args, kwargs = result[0], result[1]
            else:
                logger.warning(
                    "No applicable hook found for arg alignment on %s post-setup. Hook is: %s",
                    self.hooked_module_instance.__class__.__name__,
                    type(hf_hook),
                )

        nvtx.range_pop()
        return args, kwargs

    def post_forward(self, _module: nn.Module, output: Any) -> Any:
        """Clean up after the forward pass."""
        nvtx.range_pop()
        return output
