"""A resource pool user for Hugging Face Diffusers pipelines."""

from __future__ import annotations

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ..runtime_resource_management import AutoManageHook, ResourcePoolManagement
from ..runtime_resource_pool import ResourcePool, resources_device
from . import WeakRefResourcePoolUser
from .pytorch_module import get_module_size


class DiffusersPipelineWrapper(WeakRefResourcePoolUser[DiffusionPipeline]):
    """A resource pool user for a Hugging Face Diffusers pipeline."""

    def __init__(self, manage_object: DiffusionPipeline):
        """
        Initialize the wrapper.

        Args:
            manage_object: The DiffusionPipeline to manage.

        """
        super().__init__(manage_object)
        self.all_module_hooks_map: dict[str, AutoManageHook] = {}

    def apply_components_hook(self) -> None:
        """Apply AutoManageHooks to all model components in the pipeline."""
        pipeline = self.manage_object
        if pipeline is None:
            return

        self.all_module_hooks_map = {}

        all_model_components = {k: v for k, v in pipeline.components.items() if isinstance(v, torch.nn.Module)}

        for name, model in all_model_components.items():
            if name in self.all_module_hooks_map:
                continue
            if not isinstance(model, torch.nn.Module):
                continue

            hook = AutoManageHook.manage_module(model)
            self.all_module_hooks_map[name] = hook

    def on_setup(self, manager: ResourcePoolManagement) -> list[ResourcePool]:
        """Set up the resource pool user."""
        self.resource_pool = manager.get_resource_pool(torch.device("cpu"))
        if self.resource_pool:
            return [self.resource_pool]
        return []

    def on_load(self) -> None:
        """Load the pipeline's resources."""
        self.apply_components_hook()
        super().on_load()

    def on_resource_request(self, device: resources_device, size: int) -> None:
        """Handle a resource request from the pool."""
        super().on_resource_request(device, size)

    def get_runtime_device(self) -> resources_device | None:
        """Get the runtime device of the pipeline."""
        for hook in self.all_module_hooks_map.values():
            return hook.am.get_execution_device()
        return None

    def unlock(self) -> None:
        """Unlock the user and offload the pipeline's resources."""
        super().unlock()
        for _, hook in self.all_module_hooks_map.items():
            hook.post_forward(None, None)  # type: ignore


def get_pipeline_size(pipeline: DiffusionPipeline) -> int:
    """
    Get the total size of a pipeline's components in bytes.

    Args:
        pipeline: The pipeline to measure.

    Returns:
        The total size of the pipeline's components in bytes.

    """
    pipe_mem = 0
    for _, comp in pipeline.components.items():
        if isinstance(comp, torch.nn.Module):
            pipe_mem += get_module_size(comp)
    return pipe_mem
