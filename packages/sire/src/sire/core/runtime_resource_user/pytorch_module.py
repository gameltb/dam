"""A resource pool user for PyTorch modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import accelerate
import torch

from ...utils import human_readable_filesize
from ..context import get_sire_inference_context
from ..optimizer.plan import OptimizationPlan
from ..profile_tool import Profiler
from ..runtime_resource_management import ResourcePoolManagement
from ..runtime_resource_pool import ResourcePool, resources_device
from . import WeakRefResourcePoolUser


class TorchModuleWrapper(WeakRefResourcePoolUser[torch.nn.Module]):
    """A resource pool user for a PyTorch module."""

    def __init__(
        self,
        torch_model: torch.nn.Module,
        *,
        inference_memory_estimator: int | Callable[..., int] | Profiler | OptimizationPlan | None = None,
    ) -> None:
        """
        Initialize the wrapper.

        Args:
            torch_model: The PyTorch module to manage.
            inference_memory_estimator: An estimator for the inference memory.

        """
        super().__init__(torch_model)
        self.inference_memory_estimator = inference_memory_estimator
        self.use_accelerate = False
        self.accelerate_state_dict = None
        self.accelerate_state_dict_pin_memory = False

    def _estimate_inference_memory(self) -> int:
        estimator = self.inference_memory_estimator
        if isinstance(estimator, int):
            return estimator

        context = get_sire_inference_context()
        args: list[Any] = context.get("args", []) if context else []
        kwargs: dict[str, Any] = context.get("kwargs", {}) if context else {}

        if callable(estimator):
            return estimator(self.manage_object, *args, **kwargs)

        if isinstance(estimator, Profiler):
            profiling_data = estimator.run(*args, **kwargs)
            avg_stats = profiling_data.get_avg_stats()
            return sum(s.max_peak_vram_delta for s in avg_stats.avg_module_stats.values())
        if isinstance(estimator, OptimizationPlan):
            return estimator.get_total_runtime_vram()

        inference_memory_size = 1024 * 1024 * 1024 * 2  # Default 2GB
        if self.manage_object:
            module_cls_name = self.manage_object.__class__.__name__
            if "AutoencoderKL" in module_cls_name and args:
                x_input_shape = args[0].shape
                area = x_input_shape[0] * x_input_shape[2] * x_input_shape[3]
                inference_memory_size = int(2178 * area * 64) * 2
        return inference_memory_size

    def on_setup(self, manager: ResourcePoolManagement) -> list[ResourcePool]:
        """Set up the resource pools for the module."""
        self.offload_resource_pool = manager.get_resource_pool(torch.device("cpu"))

        if torch.cuda.is_available():
            self.runtime_resource_pool = manager.get_resource_pool(torch.device("cuda"))
        else:
            self.runtime_resource_pool = self.offload_resource_pool

        pools: list[ResourcePool] = []
        if self.runtime_resource_pool:
            pools.append(self.runtime_resource_pool)

        if self.offload_resource_pool and self.offload_resource_pool not in pools:
            pools.append(self.offload_resource_pool)

        return [p for p in pools if p]

    def on_load(self) -> None:
        """Load the module's resources onto the runtime device."""
        if self.loaded:
            return

        torch_model = self.manage_object
        if not torch_model:
            return
        runtime_device = self.get_runtime_device()
        if not runtime_device:
            return

        inference_memory_size = self._estimate_inference_memory()
        module_memory_size = get_module_size(torch_model) - self.get_used_resource_size(runtime_device)
        total_request_size = module_memory_size + inference_memory_size

        if self.runtime_resource_pool:
            self.runtime_resource_pool.request_resource(total_request_size)
            self.logger.info(
                "Requesting %s (weights: %s, inference: %s). Pool free: %s",
                human_readable_filesize(total_request_size),
                human_readable_filesize(module_memory_size),
                human_readable_filesize(inference_memory_size),
                human_readable_filesize(self.runtime_resource_pool.get_pool_free_size()),
            )

        torch_model.to(device=self.get_runtime_device())
        self.use_accelerate = hasattr(torch_model, "_hf_hook")

        super().on_load()

    def on_resource_request(self, device: resources_device, size: int) -> None:
        """Handle a resource request from the pool by offloading the module."""
        if self.manage_object is None:
            return

        self.logger.info(
            "Offloading model in response to resource request on %s for %s",
            device,
            human_readable_filesize(int(size)),
        )
        if self.runtime_resource_pool:
            pre_free_size = self.runtime_resource_pool.get_pool_free_size()

            if self.offload_resource_pool:
                self.manage_object.to(device=self.offload_resource_pool.get_pool_device())

            accelerate.utils.memory.clear_device_cache(garbage_collection=True)

            post_free_size = self.runtime_resource_pool.get_pool_free_size()
            self.logger.info(
                "Freed %s. Pool free size now: %s",
                human_readable_filesize(post_free_size - pre_free_size),
                human_readable_filesize(post_free_size),
            )
        super().on_resource_request(device, size)

    def get_runtime_device(self) -> resources_device | None:
        """Get the runtime device for the module."""
        if self.runtime_resource_pool:
            return self.runtime_resource_pool.get_pool_device()
        return None

    def get_used_resource_size(self, device: resources_device) -> int:
        """Get the size of the resources used by the module on a specific device."""
        if self.manage_object:
            return get_module_size(self.manage_object, device)
        return 0

    def get_used_resource_devices(self) -> set[torch.device]:
        """Get the set of devices the module is using resources on."""
        devices: set[torch.device] = set()
        if self.manage_object:
            for param in self.manage_object.parameters():
                devices.add(param.device)
        return devices

    def lock(self) -> None:
        """Lock the module, preventing it from being offloaded."""
        super().lock()

    def unlock(self) -> None:
        """Unlock the module, allowing it to be offloaded."""
        super().unlock()


def get_module_size(module: torch.nn.Module, device: torch.device | None = None) -> int:
    """
    Get the size of a module in bytes on a specific device.

    Args:
        module: The module to measure.
        device: The device to measure the size on. If None, all devices are measured.

    Returns:
        The size of the module in bytes.

    """
    module_mem = 0
    sd: dict[str, torch.Tensor] = module.state_dict()
    for _, t in sd.items():
        if device is None or t.device == device:
            module_mem += t.nelement() * t.element_size()
    return module_mem
