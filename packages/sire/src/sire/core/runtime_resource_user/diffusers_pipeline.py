from __future__ import annotations

from typing import Any

import diffusers
import torch

from ..runtime_resource_management import AutoManageHook, ResourcePoolManagement
from ..runtime_resource_pool import ResourcePool, resources_device
from . import WeakRefResourcePoolUser
from .pytorch_module import get_module_size


class DiffusersPipelineWrapper(WeakRefResourcePoolUser[diffusers.DiffusionPipeline]):  # type: ignore
    def __init__(self, manage_object: diffusers.DiffusionPipeline):  # type: ignore
        super().__init__(manage_object)
        self.all_module_hooks_map: dict[str, AutoManageHook] = {}

    def apply_components_hook(self):
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
        self.resource_pool = manager.get_resource_pool(torch.device("cpu"))
        if self.resource_pool:
            return [self.resource_pool]
        return []

    def on_load(self, user_context: Any = None):
        self.apply_components_hook()
        super().on_load()

    def on_resource_request(self, device: resources_device, size: int):
        super().on_resource_request(device, size)

    def get_runtime_device(self):
        for hook in self.all_module_hooks_map.values():
            return hook.am.get_execution_device()

    def unlock(self) -> None:
        super().unlock()
        for _, hook in self.all_module_hooks_map.items():
            hook.post_forward(None, None)  # type: ignore


def get_pipeline_size(pipeline: diffusers.DiffusionPipeline):
    pipe_mem = 0
    for _, comp in pipeline.components.items():
        if isinstance(comp, torch.nn.Module):
            pipe_mem += get_module_size(comp)
    return pipe_mem
