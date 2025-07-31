import copy
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from accelerate import dispatch_model
from accelerate.hooks import (
    AlignDevicesHook,
    ModelHook,
    SequentialHook,
    add_hook_to_module,
    clear_device_cache,
    find_device,
    named_module_tensors,
    remove_hook_from_module,
)
from accelerate.utils import get_balanced_memory
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from torch.cuda import nvtx

from ...utils.json_helpers import save_to_json_file
from ..profile_tool import ProfilerHook, ProfilingData, _profile_run_context
from .heuristic import HeuristicOptimizer
from .plan import OptimizationPlan
from .signature import ConfigSignatureGenerator

logger = logging.getLogger(__name__)


class PrefetchContext:
    def __init__(self, plan: OptimizationPlan, model: nn.Module, num_streams: int, offload_policy: str):
        self.plan, self.model = plan, model
        self.module_map: Dict[str, nn.Module] = {name: mod for name, mod in model.named_modules()}
        self.num_streams, self.offload_policy = num_streams, offload_policy
        self.stream_mgr = self._init_stream_mgr()
        self.module_pf_streams: Dict[str, torch.cuda.Stream] = {}  # module_name -> stream for its prefetch

    def _init_stream_mgr(self) -> Dict[str, Any]:
        streams, s_idx = defaultdict(list), defaultdict(int)
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    streams[i] = [torch.cuda.Stream(device=i) for _ in range(self.num_streams)]
            except Exception as e:
                logger.warning(f"Error initializing CUDA prefetch streams: {e}")
        return {"streams": streams, "stream_idx": s_idx}

    def get_stream(self, device: torch.device) -> Optional[torch.cuda.Stream]:
        if not (
            device.type == "cuda"
            and torch.cuda.is_available()
            and device.index is not None
            and device.index < torch.cuda.device_count()
        ):
            return None
        pool = self.stream_mgr["streams"].get(device.index, [])
        if not pool:
            return None
        idx = self.stream_mgr["stream_idx"][device.index]
        stream = pool[idx % len(pool)]
        self.stream_mgr["stream_idx"][device.index] = (idx + 1) % len(pool)
        return stream

    def set_module_prefetch_stream(self, name: str, stream: torch.cuda.Stream):
        self.module_pf_streams[name] = stream

    def get_module_prefetch_stream(self, name: str) -> Optional[torch.cuda.Stream]:
        return self.module_pf_streams.get(name)

    def clear_all_module_prefetch_streams(self):
        self.module_pf_streams.clear()


class PrefetchingWaitHook(ModelHook):
    def __init__(self, ctx: PrefetchContext, name: str, mod_inst: nn.Module, exec_dev: torch.device):
        super().__init__()
        self.ctx, self.name, self.mod_inst, self.exec_dev = ctx, name, mod_inst, exec_dev
        self.tied_ptrs_to_rm: Set[Tuple[int, torch.device]] = set()
        self.pf_submod_hf_hook: Optional[AlignDevicesHook] = None
        logger.debug(f"PrefetchingWaitHook for {self.name} on {self.exec_dev}")

    @torch.compiler.disable()
    def pre_forward(self, module: nn.Module, *args, **kwargs):
        if module is not self.mod_inst:
            logger.warning(f"WaitHook for {self.name} called on wrong mod")
            return args, kwargs
        pf_stream = self.ctx.get_module_prefetch_stream(self.name)
        if pf_stream:
            comp_stream = torch.cuda.current_stream(self.exec_dev)
            logger.debug(
                f"Mod {self.name} on dev {self.exec_dev} (stream {comp_stream.stream_id}) waiting for pf stream {pf_stream.stream_id}"
            )
            comp_stream.wait_stream(pf_stream)
            if self.pf_submod_hf_hook:
                self.pf_submod_hf_hook.tied_pointers_to_remove = self.tied_ptrs_to_rm
        return args, kwargs


class AlignDevicesHookTorchCompilerDisable(AlignDevicesHook):
    @classmethod
    def from_align_devices_hook(cls, align_devices_hook):
        align_devices_hook.__class__ = cls
        return align_devices_hook

    @torch.compiler.disable()
    def pre_forward(self, module, *args, **kwargs):
        return super().pre_forward(module, *args, **kwargs)

    @torch.compiler.disable()
    def post_forward(self, module, output):
        return super().post_forward(module, output)


class PrefetchingHook(ModelHook):  # Placed on trigger module
    def __init__(self, ctx: PrefetchContext, name: str, mod_inst: nn.Module):
        super().__init__()
        self.ctx, self.name, self.mod_inst = ctx, name, mod_inst
        logger.debug(f"PrefetchingHook (trigger) for {self.name}")

    @torch.compiler.disable()
    def pre_forward(self, module: nn.Module, *args, **kwargs):
        if module is not self.mod_inst:
            logger.warning(f"TriggerHook for {self.name} called on wrong mod")
            return args, kwargs
        logger.debug(f"TriggerHook pre_fwd for {self.name} (id {id(module)})")
        nvtx.range_push(f"pf_trigger_{self.name}")
        trigger_dev = find_device(module.state_dict())

        for instr in self.ctx.plan.trigger_index.get(self.name, []):
            pf_mod_name, pf_tgt_dev = instr.module_to_prefetch, instr.target_device
            mod_to_pf = self.ctx.module_map.get(pf_mod_name)
            if not mod_to_pf:
                logger.warning(f"Mod '{pf_mod_name}' for prefetch not found")
                continue
            pf_stream = self.ctx.get_stream(pf_tgt_dev)
            if not pf_stream:
                logger.warning(f"No pf stream for {pf_tgt_dev}. Cannot prefetch {pf_mod_name}.")
                continue

            if trigger_dev and trigger_dev.type == "cuda" and pf_tgt_dev.type == "cuda":
                pf_stream.wait_stream(torch.cuda.current_stream(trigger_dev))
            self.do_prefetch(pf_mod_name, pf_tgt_dev, mod_to_pf, pf_stream)
        nvtx.range_pop()
        return args, kwargs

    def do_prefetch(self, pf_name: str, pf_dev: torch.device, pf_mod: nn.Module, pf_stream: torch.cuda.Stream):
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
                    logger.error(f"No AlignHook on {pf_name}. Abort prefetch.")
                    nvtx.range_pop()
                    return
                if not wait_hook and self.ctx.plan.prefetch_schedule:
                    logger.error(f"No WaitHook on {pf_name} with active prefetch. Abort.")
                    nvtx.range_pop()
                    return

                align_hook = AlignDevicesHookTorchCompilerDisable.from_align_devices_hook(align_hook)

                wait_hook.tied_ptrs_to_rm.clear()
                wait_hook.pf_submod_hf_hook = align_hook
                wait_hook.exec_dev = pf_dev

                w_map, tied_map = getattr(align_hook, "weights_map", None), getattr(align_hook, "tied_params_map", None)
                if w_map is None or tied_map is None:
                    logger.error(f"AlignHook on {pf_name} not init. No prefetch.")
                    nvtx.range_pop()
                    return
                align_hook.execution_device = pf_dev

                for name, val in named_module_tensors(pf_mod, True, False, True):
                    if val.device.type == "meta":
                        map_val = w_map.get(name)
                        if map_val is None:
                            logger.warning(f"Meta tensor '{name}' in '{pf_name}' no value in weights_map.")
                            continue
                        if map_val.device.type == "cpu" and not map_val.is_pinned():
                            try:
                                map_val = map_val.pin_memory()
                                w_map.dataset.state_dict[w_map.prefix + name] = map_val
                            except RuntimeError as e_pin:
                                logger.warning(f"Could not pin {name} for prefetch: {e_pin}. Using unpinned.")

                        if map_val.data_ptr() not in tied_map:
                            tied_map[map_val.data_ptr()] = {}

                        tgt_tensor = torch.empty_like(map_val, device=pf_dev)
                        tgt_tensor.copy_(map_val, non_blocking=True)
                        param_tgt = torch.nn.Parameter(tgt_tensor, requires_grad=map_val.requires_grad)
                        tied_map[map_val.data_ptr()][pf_dev] = param_tgt
                        if wait_hook:
                            wait_hook.tied_ptrs_to_rm.add((map_val.data_ptr(), pf_dev))
                    elif val.device != pf_dev:
                        logger.debug(f"Prefetching existing tensor {name} from {val.device} to {pf_dev}")
                        val.data = val.data.to(pf_dev, non_blocking=True)
                logger.info(f"Prefetch task for {pf_name} to {pf_dev} submitted on stream {pf_stream.stream_id}.")
            self.ctx.set_module_prefetch_stream(pf_name, pf_stream)
        except Exception as e_pf:
            logger.error(f"Error in do_prefetch for {pf_name}: {e_pf}", exc_info=True)
        finally:
            nvtx.range_pop()


def infer_fine_grained_device_map(
    model: nn.Module,
    max_memory: Optional[Dict[str, int]],  # keys are str
    no_split: Optional[List[str]],
    verbose: bool,
) -> Dict[str, str]:
    no_split = no_split or []
    dev_map: Dict[str, str] = {}
    frozen: Set[str] = set()
    default_dev = "cpu"
    if max_memory:
        gpus = [k for k, v in max_memory.items() if k != "cpu" and v > 0]
        if gpus:
            default_dev = min(gpus)
            logger.debug(f"Initial map using {default_dev} for no_split.")

    def _traverse(mod: nn.Module, path: str = ""):
        nonlocal dev_map, frozen
        cls_name = mod.__class__.__name__
        is_frozen = any(path.startswith(p + ".") for p in frozen if p)
        if is_frozen:
            pass
        elif cls_name in no_split:
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


class InferenceOptimizerHook(ModelHook):
    def __init__(
        self,
        cache_dir="opt_cache",
        num_prefetch_streams=1,
        no_split_module_classes=None,
        custom_signature_callback=None,
        default_offload_policy="cpu",
        force_profiling=False,
        run_profiling_if_needed=True,
        max_memory_gb=None,
    ):
        super().__init__()
        self.base_cache_dir = cache_dir
        os.makedirs(self.base_cache_dir, exist_ok=True)
        self.num_prefetch_streams = max(1, num_prefetch_streams)
        self.no_split_module_classes = no_split_module_classes or []
        self.custom_signature_callback = custom_signature_callback
        self.default_offload_policy = default_offload_policy
        self.force_profiling = force_profiling  # User's initial setting
        self._force_profiling_active = force_profiling  # Internal flag, reset after use
        self.run_profiling_if_needed = run_profiling_if_needed
        self.user_max_memory_gb = max_memory_gb

        self.sig_gen = ConfigSignatureGenerator()
        self.current_plan: Optional[OptimizationPlan] = None
        self.current_config_sig_hash: Optional[str] = None
        self.current_max_memory_bytes: Optional[Dict[str, int]] = None
        self.current_plan_id: Optional[str] = None
        self.last_module_id_processed: Optional[int] = None
        self.active_pf_ctx: Optional[PrefetchContext] = None
        self.hooked_module_instance: Optional[nn.Module] = None
        self.current_module_dtype: Optional[torch.dtype] = None
        self.is_first_forward = True
        self.module: Optional[nn.Module] = None
        logger.info(f"IOHook init. Cache: {self.base_cache_dir}, Prefetch: {self.num_prefetch_streams}")

    def _get_max_mem_bytes(self) -> Dict[str, int]:  # keys are str
        mem_map: Dict[str, int] = {}
        if self.user_max_memory_gb:
            for k, v in self.user_max_memory_gb.items():
                mem_map[str(k)] = int(v * (1024**3))
            logger.info(f"User max_mem: {{k:f'{v / (1024**3):.1f}GB' for k,v in mem_map.items()}}")
        else:
            logger.info("Auto-balancing memory.")
            if not self.hooked_module_instance:
                logger.error("Cannot auto-balance: no module.")
                return {"cpu": 64 * (1024**3)}
            try:
                balanced = get_balanced_memory(
                    self.hooked_module_instance, self.current_module_dtype, False, self.no_split_module_classes
                )
                mem_map = {str(k): v for k, v in balanced.items()}
                mem_str = {k: f"{val / (1024**3):.1f}GB" for k, val in mem_map.items()}
                logger.info(f"Auto-balanced max_mem: {mem_str}")
            except Exception as e:
                logger.error(f"Auto-balance fail: {e}", exc_info=True)
                mem_map = {"cpu": 64 * (1024**3)}
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        if str(i) not in mem_map:
                            mem_map[str(i)] = 1 * (1024**3)
        return mem_map

    def _get_sig_dir_path(self) -> str:  # For current_config_sig_hash
        if not self.current_config_sig_hash:
            logger.error("Config sig hash None. Cannot get dir.")
            return os.path.join(self.base_cache_dir, "_ERR_NO_CONF_SIG")
        d = os.path.join(self.base_cache_dir, self.current_config_sig_hash)
        os.makedirs(d, exist_ok=True)
        return d

    def _get_plan_file_path(self) -> Optional[str]:  # For current_plan_id
        if not self.current_plan_id:
            logger.error("Plan ID None. Cannot get path.")
            return None
        d = os.path.join(self._get_sig_dir_path(), "plans")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.current_plan_id}.json")

    def _run_profiling(self, args, kwargs, max_mem_prof, conf_sig_hash, raw_sig_details) -> Optional[ProfilingData]:
        logger.info("=" * 20 + " Profiling Session Start " + "=" * 20)
        if not self.hooked_module_instance or not self.current_module_dtype:
            logger.error("Cannot profile: state missing.")
            return None

        mod_to_prof = self.hooked_module_instance
        prof_data = ProfilingData()
        sig_dir_for_save = os.path.join(self.base_cache_dir, conf_sig_hash)
        os.makedirs(sig_dir_for_save, exist_ok=True)
        prof_data_path = os.path.join(sig_dir_for_save, "profiling_data.json")
        raw_details_path = os.path.join(sig_dir_for_save, "raw_signature_details.json")
        try:
            save_to_json_file(raw_sig_details, raw_details_path)
            logger.info(f"Raw sig details saved to {raw_details_path}")
        except Exception as e:
            logger.error(f"Failed to save raw sig details: {e}")

        remove_hook_from_module(mod_to_prof, recurse=True)
        try:
            logger.info(f"Preparing '{mod_to_prof.__class__.__name__}' for profiling.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            init_map = infer_fine_grained_device_map(
                mod_to_prof, None, self.no_split_module_classes, logger.isEnabledFor(logging.DEBUG)
            )
            if not init_map and any(mod_to_prof.parameters()):
                raise RuntimeError("Failed to create init map for profiling.")

            main_dev_prof = (
                torch.device("cuda:0") if torch.cuda.is_available() and "0" in max_mem_prof else torch.device("cpu")
            )
            dispatch_model(mod_to_prof, init_map, main_dev_prof, force_hooks=True)

            ph_count = 0
            for name, sub_mod in mod_to_prof.named_modules():
                if hasattr(sub_mod, "_hf_hook"):
                    add_hook_to_module(sub_mod, ProfilerHook(name), True)
                    ph_count += 1
            logger.info(f"Registered {ph_count} ProfilerHooks.")

            p_args, p_kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
            logger.info("Warm-up profiling inference...")
            with torch.no_grad():
                _ = mod_to_prof(*p_args, **p_kwargs)
            clear_device_cache()
            with _profile_run_context(prof_data), torch.no_grad():
                logger.info("Main profiling inference...")
                _ = mod_to_prof(*p_args, **p_kwargs)
            prof_data.save(prof_data_path)
        except Exception as e:
            logger.error(f"Error in profiling: {e}", exc_info=True)
            if prof_data.module_stats:
                prof_data.save(os.path.join(sig_dir_for_save, "profiling_data.error.json"))
            return None
        finally:
            logger.info("Cleaning up post-profiling...")
            remove_hook_from_module(mod_to_prof, recurse=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("=" * 20 + " Profiling Session End " + "=" * 20)
        return prof_data

    def _gen_opt_plan(self, prof_data: ProfilingData, max_mem_plan: Dict[str, int]) -> Optional[OptimizationPlan]:
        logger.info("Generating optimization plan...")
        if not self.hooked_module_instance:
            logger.error("Cannot gen plan: no module.")
            return None
        opt = HeuristicOptimizer(prof_data, max_mem_plan)
        return opt.optimize()

    def _setup_module_with_plan(self, mod_to_opt: nn.Module, plan: OptimizationPlan) -> Optional[PrefetchContext]:
        logger.info(f"Preparing/dispatching '{mod_to_opt.__class__.__name__}' with plan...")
        offload = self.default_offload_policy
        if any(d.type == "cpu" for d in plan.optimized_device_map.values()) and offload != "cpu":
            offload = "cpu"

        pf_ctx = PrefetchContext(plan, mod_to_opt, self.num_prefetch_streams, offload)
        remove_hook_from_module(mod_to_opt, recurse=True)  # Remove IOH and any prior hooks

        self.cpu_state_dict = mod_to_opt.state_dict()

        main_dev_dispatch = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        map_for_dispatch = {
            k: ("cpu" if v.type == "cpu" else f"{v.type}:{v.index}" if v.index is not None else v.type)
            for k, v in plan.optimized_device_map.items()
        }
        dispatch_model(
            mod_to_opt, map_for_dispatch, main_dev_dispatch, force_hooks=True, state_dict=self.cpu_state_dict
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
        logger.info(f"Registered {waits} WaitHooks and {trigs} TriggerHooks.")
        return pf_ctx

    @torch.compiler.disable()
    def pre_forward(self, module: nn.Module, *args, **kwargs):
        nvtx.range_push("IOHook.pre_forward")
        if self.is_first_forward:
            self.hooked_module_instance = module
            logger.info(f"Hook first fwd: {module.__class__.__name__} (id:{id(module)})")
            try:
                p_dt = next((p.dtype for p in module.parameters(False)), None)
                b_dt = next((b.dtype for b in module.buffers(False)), None)
                self.current_module_dtype = getattr(module, "dtype", p_dt or b_dt or torch.get_default_dtype())
            except Exception as e:
                self.current_module_dtype = torch.get_default_dtype()
                logger.warning(f"Dtype infer fail: {e}. Using {self.current_module_dtype}")
            logger.info(f"Module dtype: {self.current_module_dtype}")
            self.current_max_memory_bytes = self._get_max_mem_bytes()
            self.is_first_forward = False

        if not all([self.hooked_module_instance, self.current_module_dtype, self.current_max_memory_bytes]):
            logger.error("Hook state not init.")
            nvtx.range_pop()
            return args, kwargs

        new_conf_hash, raw_details = self.sig_gen.generate_config_signature(
            self.hooked_module_instance, args, kwargs, self.current_module_dtype, self.custom_signature_callback
        )
        new_plan_id = self.sig_gen.generate_plan_identifier(self.current_max_memory_bytes)

        needs_update = (
            id(self.hooked_module_instance) != self.last_module_id_processed
            or new_conf_hash != self.current_config_sig_hash
            or new_plan_id != self.current_plan_id
            or self.current_plan is None
            or self._force_profiling_active
        )

        if needs_update:
            logger.info(
                f"Plan update triggered. ModID:{id(self.hooked_module_instance) != self.last_module_id_processed}, "
                f"ConfHash:{new_conf_hash != self.current_config_sig_hash}, PlanID:{new_plan_id != self.current_plan_id}, "
                f"NoPlan:{self.current_plan is None}, ForceProf:{self._force_profiling_active}"
            )
            self.current_config_sig_hash, self.current_plan_id = new_conf_hash, new_plan_id

            sig_dir = self._get_sig_dir_path()
            prof_data_path = os.path.join(sig_dir, "profiling_data.json")
            plan_path = self._get_plan_file_path()

            prof_data: Optional[ProfilingData] = None
            just_profiled = False
            if not self._force_profiling_active:
                prof_data = ProfilingData.load(prof_data_path)

            if prof_data is None:
                if self.run_profiling_if_needed or self._force_profiling_active:
                    logger.info(
                        f"Profiling for conf {self.current_config_sig_hash}. Force={self._force_profiling_active}"
                    )
                    prof_data = self._run_profiling(
                        args, kwargs, self.current_max_memory_bytes, self.current_config_sig_hash, raw_details
                    )
                    just_profiled = True  # Mark that we just ran profiling
                    if not prof_data:
                        logger.error("Profiling fail.")
                        self.current_plan = None
                        self.active_pf_ctx = None
                        nvtx.range_pop()
                        return args, kwargs
                else:
                    logger.error(f"Prof data missing for {self.current_config_sig_hash}, not run.")
                    self.current_plan = None
                    self.active_pf_ctx = None
                    nvtx.range_pop()
                    return args, kwargs
            else:
                logger.info(f"Using existing prof data from {prof_data_path}")
                raw_det_path = os.path.join(sig_dir, "raw_signature_details.json")
                if not os.path.exists(raw_det_path):
                    try:
                        save_to_json_file(raw_details, raw_det_path)
                        logger.info(f"Raw sig details {raw_det_path} (prof skipped).")
                    except Exception as e:
                        logger.error(f"Fail save raw sig details (prof skipped): {e}")

            self.current_plan = None
            if not just_profiled and plan_path:
                self.current_plan = OptimizationPlan.load(plan_path)

            if self.current_plan is None:
                logger.info(
                    f"Generating plan for plan_id {self.current_plan_id} (config {self.current_config_sig_hash})"
                )
                self.current_plan = self._gen_opt_plan(prof_data, self.current_max_memory_bytes)
                if self.current_plan and plan_path:
                    self.current_plan.save(plan_path)
                elif not self.current_plan:
                    logger.error("Plan gen fail.")
                    self.active_pf_ctx = None
                    nvtx.range_pop()
                    return args, kwargs
            else:
                logger.info(f"Loaded existing plan from {plan_path}")

            if self.active_pf_ctx:
                self.active_pf_ctx.clear_all_module_prefetch_streams()
            self.active_pf_ctx = self._setup_module_with_plan(self.hooked_module_instance, self.current_plan)
            if not self.active_pf_ctx:
                logger.error("Failed to prep/dispatch module.")
                self.current_plan = None
                nvtx.range_pop()
                return args, kwargs

            self.last_module_id_processed = id(self.hooked_module_instance)
            self._force_profiling_active = False

            # Arg alignment by the module's current hook (set up by _setup_module_with_plan).
            # This is complex because our own hook is in the sequence. We need to find the
            # real AlignDevicesHook and call it, without causing infinite recursion.
            hf_hook = getattr(self.hooked_module_instance, "_hf_hook", None)
            if isinstance(hf_hook, SequentialHook):
                # Find the AlignDevicesHook, but avoid the InferenceOptimizerHook itself
                align_hook = next((h for h in hf_hook.hooks if isinstance(h, AlignDevicesHook) and h is not self), None)
                if align_hook:
                    logger.debug(f"Manually calling pre_forward of AlignDevicesHook: {type(align_hook)}")
                    args, kwargs = align_hook.pre_forward(self.hooked_module_instance, *args, **kwargs)
                else:
                    logger.warning("No AlignDevicesHook found in SequentialHook post-setup.")
            elif hf_hook is not self and isinstance(hf_hook, AlignDevicesHook):
                # The only hook is the AlignDevicesHook
                logger.debug(f"Manually calling pre_forward of sole AlignDevicesHook: {type(hf_hook)}")
                args, kwargs = hf_hook.pre_forward(self.hooked_module_instance, *args, **kwargs)
            else:
                logger.warning(
                    f"No applicable hook found for arg alignment on {self.hooked_module_instance.__class__.__name__} post-setup. Hook is: {type(hf_hook)}"
                )

        nvtx.range_pop()
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any):
        return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )
    logger.setLevel(logging.DEBUG)

    example_logger = logging.getLogger("SDXL_Example")
    example_logger.setLevel(logging.INFO)

    optimizer_hook = InferenceOptimizerHook(
        cache_dir="example_optim_cache_sdxl",
        max_memory_gb={0: 8, "cpu": 24} if torch.cuda.is_available() and torch.cuda.device_count() > 0 else {"cpu": 24},
        force_profiling=False,
        num_prefetch_streams=1,
    )

    model_path = os.getenv("CI_TEST_MODEL_PATH", "playground-v2.5-1024px-aesthetic.fp16.safetensors")
    if not os.path.exists(model_path):
        example_logger.warning(f"Model '{model_path}' not found. Skipping SDXL example.")
    else:
        example_logger.info(f"Loading SDXL pipeline from: {model_path}")
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=torch.float16, local_files_only=True, use_safetensors=True
            )
            example_logger.info(f"Pipeline loaded. UNet dtype: {pipe.unet.dtype}")
        except Exception as e:
            example_logger.error(f"Failed to load pipeline: {e}", exc_info=True)
            sys.exit(1)

        add_hook_to_module(pipe.unet, optimizer_hook, append=False)
        example_logger.info(f"IOHook attached to UNet ({pipe.unet.__class__.__name__}).")

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            example_logger.info("Moving VAE & TextEncoders to CPU for UNet GPU memory.")
            pipe.vae.to("cpu")
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
        else:
            example_logger.info("No CUDA GPUs. Running on CPU.")

        prompt = "A majestic dragon soaring through a vibrant sunset sky, fantasy art, highly detailed"
        num_steps = 3
        for i in range(2):
            example_logger.info(f"\n--- Inference pass {i + 1} --- (Prompt: '{prompt}', Steps: {num_steps})")
            try:
                with torch.no_grad():
                    latents = pipe(prompt, num_inference_steps=num_steps, output_type="latent").images
                example_logger.info(
                    f"Pass {i + 1} complete. Latents shape: {latents.shape if latents is not None else 'None'}"
                )
            except Exception as e:
                example_logger.error(f"Pass {i + 1} error: {e}", exc_info=True)
        example_logger.info("\nExample finished.")
