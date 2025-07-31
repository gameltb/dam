from __future__ import annotations

from typing import Callable, Union

import accelerate
import accelerate.hooks
import torch

from ...utils import human_readable_filesize
from ..optimizer.plan import OptimizationPlan
from ..profile_tool import Profiler
from . import WeakRefResourcePoolUser


class TorchModuleWrapper(WeakRefResourcePoolUser[torch.nn.Module]):
    def __init__(
        self,
        torch_model: torch.nn.Module,
        *,
        inference_memory_estimator: Union[int, Callable, Profiler, OptimizationPlan, None] = None,
    ) -> None:
        super().__init__(torch_model)
        self.inference_memory_estimator = inference_memory_estimator
        self.use_accelerate = False
        self.accelerate_state_dict = None
        self.accelerate_state_dict_pin_memory = False

    def _estimate_inference_memory(self, user_context=None) -> int:
        estimator = self.inference_memory_estimator
        if isinstance(estimator, int):
            return estimator
        if callable(estimator):
            # Assumes the callable has a signature of (model, user_context) -> int
            return estimator(self.manage_object, user_context)

        args = user_context.get("args", []) if user_context else []
        kwargs = user_context.get("kwargs", {}) if user_context else {}

        if isinstance(estimator, Profiler):
            profiling_data = estimator.run(*args, **kwargs)
            avg_stats = profiling_data.get_avg_stats()
            # Sum of max VRAM delta over all modules
            total_vram_delta = sum(s.max_peak_vram_delta for s in avg_stats.avg_module_stats.values())
            return total_vram_delta
        if isinstance(estimator, OptimizationPlan):
            # An optimization plan should know its memory requirements.
            # This part needs to be implemented in OptimizationPlan.
            # For now, let's assume a method exists.
            return estimator.get_total_runtime_vram()

        # Fallback to old logic if no estimator is provided
        # TODO: Remove this hardcoded logic eventually
        inference_memory_size = 1024 * 1024 * 1024 * 2  # Default 2GB
        module_cls_name = self.manage_object.__class__.__name__
        if "AutoencoderKL" in module_cls_name and args:
            x_input_shape = args[0].shape
            area = x_input_shape[0] * x_input_shape[2] * x_input_shape[3]
            inference_memory_size = int(2178 * area * 64) * 2
        # ... other hardcoded cases ...
        return inference_memory_size

    def on_setup(self, manager):
        self.offload_resource_pool = manager.get_resource_pool(torch.device("cpu"))

        if torch.cuda.is_available():
            self.runtime_resource_pool = manager.get_resource_pool(torch.device("cuda"))
        else:
            self.runtime_resource_pool = self.offload_resource_pool

        # Collect existing pools
        pools = []
        if self.runtime_resource_pool:
            pools.append(self.runtime_resource_pool)

        # Avoid adding the same pool twice
        if self.offload_resource_pool and self.offload_resource_pool not in pools:
            pools.append(self.offload_resource_pool)

        return pools

    def on_load(self, user_context=None):
        if self.loaded:
            return

        torch_model = self.manage_object
        runtime_device = self.get_runtime_device()

        # Step 1: Estimate the required memory for this inference operation.
        inference_memory_size = self._estimate_inference_memory(user_context)

        # Step 2: Calculate the memory needed for the model's weights.
        # This only includes weights currently offloaded that need to be moved.
        module_memory_size = get_module_size(torch_model) - self.get_used_resource_size(runtime_device)

        # Step 3: Request the total required memory from the pool.
        total_request_size = module_memory_size + inference_memory_size
        self.runtime_resource_pool.request_resource(total_request_size)
        self.logger.info(
            f"Requesting {human_readable_filesize(total_request_size)} "
            f"(weights: {human_readable_filesize(module_memory_size)}, "
            f"inference: {human_readable_filesize(inference_memory_size)}). "
            f"Pool free: {human_readable_filesize(self.runtime_resource_pool.get_pool_free_size())}"
        )

        # Step 4: Move the model to the runtime device.
        # If the model has hooks (e.g., InferenceOptimizerHook), it will not be
        # moved monolithically. The hooks themselves will manage moving sub-parts
        # of the model during the actual forward pass. Calling `.to()` here ensures
        # that at least the top-level module object is on the right device and
        # that non-parameter/buffer tensors are moved. For models with advanced
        # hooks, this might be a no-op for the parameters themselves.
        torch_model.to(device=self.get_runtime_device())

        # The wrapper is no longer responsible for dispatching or optimization.
        # If the user wants optimization, they must apply the hook before wrapping.
        self.use_accelerate = hasattr(torch_model, "_hf_hook")

        super().on_load()

    def on_resource_request(self, device, size):
        if self.manage_object is None:
            return

        self.logger.info(
            f"Offloading model in response to resource request on {device} for {human_readable_filesize(int(size))}"
        )
        pre_free_size = self.runtime_resource_pool.get_pool_free_size()

        # The wrapper's only job is to move the object to the offload device.
        # Any hooks on the object (from accelerate or InferenceOptimizerHook) are
        # responsible for managing their own state during this move.
        self.manage_object.to(device=self.offload_resource_pool.get_pool_device())

        accelerate.utils.memory.clear_device_cache(garbage_collection=True)

        post_free_size = self.runtime_resource_pool.get_pool_free_size()
        self.logger.info(
            f"Freed {human_readable_filesize(post_free_size - pre_free_size)}. "
            f"Pool free size now: {human_readable_filesize(post_free_size)}"
        )
        super().on_resource_request(device, size)

    def get_runtime_device(self):
        return self.runtime_resource_pool.get_pool_device()

    def get_used_resource_size(self, device):
        return get_module_size(self.manage_object, device)

    def lock(self):
        super().lock()
        # self.logger.info("lock")

    def unlock(self):
        super().unlock()
        # self.logger.info("unlock")


def get_module_size(module: torch.nn.Module, device=None):
    module_mem = 0
    sd: dict[str, torch.Tensor] = module.state_dict()
    for k, t in sd.items():
        if device is None or t.device == device:
            module_mem += t.nelement() * t.element_size()
    return module_mem
