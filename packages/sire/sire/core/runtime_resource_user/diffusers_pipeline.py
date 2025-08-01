from __future__ import annotations

import diffusers
import torch

from ..commit_object import CommitObjectProxy
from ..runtime_resource_management import AutoManageHook
from . import WeakRefResourcePoolUser
from .pytorch_module import get_module_size


class DiffusersPipelineWrapper(WeakRefResourcePoolUser[diffusers.DiffusionPipeline]):
    def __init__(self, manage_object):
        super().__init__(manage_object)
        self.all_module_hooks_map: dict[str, AutoManageHook] = {}

    def apply_components_hook(self):
        pipeline = self.manage_object

        self.all_module_hooks_map = {}

        all_model_components = {k: v for k, v in pipeline.components.items()}

        for name, model in all_model_components.items():
            if name in self.all_module_hooks_map:
                continue

            # If a component is managed by a CommitObjectProxy, skip it.
            # The user is responsible for managing its state.
            if isinstance(model, CommitObjectProxy):
                self.logger.info(f"Skipping AutoManageHook for '{name}' as it is managed by a CommitObjectProxy.")
                continue

            if not isinstance(model, torch.nn.Module):
                continue

            hook = AutoManageHook.manage_module(model)
            self.all_module_hooks_map[name] = hook

    def on_setup(self, manager):
        self.resource_pool = manager.get_resource_pool(torch.device("cpu"))
        return [self.resource_pool]

    def on_load(self, user_context=None):
        self.apply_components_hook()
        super().on_load()

    def on_resource_request(self, device, size):
        super().on_resource_request(device, size)

    def get_runtime_device(self):
        for hook in self.all_module_hooks_map.values():
            return hook.am.get_execution_device()

    # def get_used_resource_devices(self):
    #     return super().get_used_resource_devices()

    # def get_used_resource_size(self, device):
    #     return 0

    def unlock(self):
        super().unlock()
        for name, hook in self.all_module_hooks_map.items():
            hook.post_forward(None, None)


def get_pipeline_size(pipeline: diffusers.DiffusionPipeline):
    pipe_mem = 0
    for comp_name, comp in pipeline.components.items():
        if isinstance(comp, torch.nn.Module):
            pipe_mem += get_module_size(comp)
    return pipe_mem
