from __future__ import annotations

import accelerate
import accelerate.hooks
import torch

from ...utils import human_readable_filesize
from . import WeakRefResourcePoolUser


class TorchModuleWrapper(WeakRefResourcePoolUser[torch.nn.Module]):
    def __init__(self, torch_model: torch.nn.Module) -> None:
        super().__init__(torch_model)
        # TODO:measures and cache inference memory size
        self.inference_memory_size = 1024 * 1024 * 1024 * 2
        self.use_accelerate = False
        self.accelerate_state_dict = None
        self.accelerate_state_dict_pin_memory = False

    def on_setup(self, manager):
        self.runtime_resource_pool = manager.get_resource_pool(torch.device("cuda"))
        self.offload_resource_pool = manager.get_resource_pool(torch.device("cpu"))
        return [self.runtime_resource_pool, self.offload_resource_pool]

    def on_load(self, user_context=None):
        user_context_changed = False
        if self.loaded and not user_context_changed:
            return

        torch_model = self.manage_object
        runtime_device = self.get_runtime_device()

        module_cls_name = torch_model.__class__.__name__
        if isinstance(user_context, dict):
            if "AutoencoderKL" in module_cls_name:
                if args := user_context.get("args", None):
                    x_input_shape = args[0].shape
                    area = x_input_shape[0] * x_input_shape[2] * x_input_shape[3]
                    self.inference_memory_size = int(2178 * area * 64) * 2
            elif "UNet2DConditionModel" in module_cls_name:
                if args := user_context.get("args", None):
                    x_input_shape = args[0].shape
                    area = x_input_shape[0] * x_input_shape[2] * x_input_shape[3]
                    self.inference_memory_size = int((area * torch_model.dtype.itemsize / 50) * (1024 * 1024))
            elif "SanaTransformer2DModel" in module_cls_name:
                self.inference_memory_size = 1024 * 1024 * 1024 * 5
            elif "AutoencoderDC" in module_cls_name:
                self.inference_memory_size = 1024 * 1024 * 1024 * 7
                torch_model.enable_tiling()
            elif "FluxTransformer2DModel" in module_cls_name:
                self.inference_memory_size = 1024 * 1024 * 768

        is_vae = "Autoencoder" in module_cls_name
        if is_vae:
            torch_model.enable_slicing()

        module_memory_size = get_module_size(torch_model) - self.get_used_resource_size(runtime_device)
        self.runtime_resource_pool.request_resource(module_memory_size + self.inference_memory_size)
        free_size = self.runtime_resource_pool.get_pool_free_size()
        self.logger.info(
            f"request_size = {human_readable_filesize(module_memory_size + self.inference_memory_size)} free_size = {human_readable_filesize(free_size)}"
        )
        if is_vae or free_size > module_memory_size + self.inference_memory_size:
            torch_model.to(device=self.get_runtime_device())
        else:
            accelerate_free_size = free_size - self.inference_memory_size
            accelerate_minimum_requirement = 0
            if accelerate_free_size < accelerate_minimum_requirement:
                accelerate_free_size = min(1024 * 1024 * 1024, int(free_size / 2))
            no_split_module_classes = ["NunchakuJointTransformerBlock", "NunchakuFluxSingleTransformerBlock"]
            # device_map = accelerate.infer_auto_device_map(
            #     torch_model,
            #     max_memory={0: accelerate_free_size, "cpu": "26GiB"},
            #     no_split_module_classes=no_split_module_classes,
            # )
            # self.logger.warning(f"accelerate.infer_auto_device_map {device_map}")

            # if False and not self.accelerate_state_dict_pin_memory:
            #     state_dict = torch_model.state_dict()

            #     # while len(state_dict) > 0:
            #     #     k, v = state_dict.popitem()
            #     #     torch_model.load_state_dict({k: v.pin_memory()}, strict=False, assign=True)

            #     for k in state_dict:
            #         state_dict[k] = state_dict[k].pin_memory()
            #     torch_model.load_state_dict(state_dict, strict=False, assign=True)

            #     self.accelerate_state_dict_pin_memory = True

            # self.accelerate_state_dict: dict[str, torch.Tensor] = torch_model.state_dict()

            # old_hook = None
            # if getattr(torch_model, "_hf_hook", None) is not None:
            #     old_hook = torch_model._hf_hook
            # accelerate.dispatch_model(
            #     torch_model,
            #     device_map=device_map,
            #     main_device=runtime_device,
            #     preload_module_classes=no_split_module_classes,
            #     state_dict=self.accelerate_state_dict,
            # )
            # if old_hook is not None:
            #     accelerate.hooks.add_hook_to_module(torch_model, old_hook, append=True)
            old_hook = None
            if getattr(torch_model, "_hf_hook", None) is not None:
                old_hook = torch_model._hf_hook
            from ..accelerate.hook import InferenceOptimizerHook

            self.optimizer_hook = InferenceOptimizerHook(
                cache_dir="example_optim_cache_sdxl",
                max_memory_gb={0: 4, "cpu": 16}
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else {"cpu": 16},
            )
            accelerate.hooks.add_hook_to_module(torch_model, self.optimizer_hook, append=False)
            self.optimizer_hook.pre_forward(torch_model, *user_context.get("args"), **user_context.get("kwargs"))
            if old_hook is not None:
                accelerate.hooks.add_hook_to_module(torch_model, old_hook, append=False)
            accelerate.hooks.add_hook_to_module(torch_model, self.optimizer_hook, append=True)

            self.use_accelerate = True

        super().on_load()

    def on_resource_request(self, device, size):
        if self.manage_object is None:
            return

        self.logger.info(f"on_resource_request {device} {human_readable_filesize(int(size))}")
        pre_free_size = self.runtime_resource_pool.get_pool_free_size()

        if self.use_accelerate:
            accelerate.hooks.remove_hook_from_module(self.manage_object, recurse=True)
            if self.optimizer_hook.cpu_state_dict:
                self.manage_object.load_state_dict(self.optimizer_hook.cpu_state_dict, assign=True)
            self.accelerate_state_dict = None
            self.optimizer_hook = None
            self.use_accelerate = False
        else:
            self.manage_object.to(device=self.offload_resource_pool.get_pool_device())

        accelerate.utils.memory.clear_device_cache(garbage_collection=True)

        post_free_size = self.runtime_resource_pool.get_pool_free_size()
        self.logger.info(
            f"on_resource_request free {human_readable_filesize(post_free_size - pre_free_size)} free_size_now = {human_readable_filesize(post_free_size)}"
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
