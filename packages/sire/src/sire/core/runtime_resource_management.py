# type: ignore
import contextlib
import weakref
from typing import Any, Generic, Iterator, Optional, TypeVar

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device

from ..utils.runtime_resource_util import clear_device_cache_and_libc_alloc
from .context import sire_inference_context
from .runtime_resource_pool import ResourcePool, resources_device
from .runtime_resource_user import ResourcePoolUserABC

T = TypeVar("T")


class AutoManageWrapper(Generic[T]):
    type_wrapper_map: dict[type, type] = {}
    wrapper_obj_map: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()

    def __init__(self, obj: Any, **kwargs: Any) -> None:
        if not isinstance(obj, ResourcePoolUserABC):
            user = self.wrapper_obj_map.get(obj, None)
            if user is None:
                for tp, wrapper_cls in self.type_wrapper_map.items():
                    if isinstance(obj, tp):
                        user = wrapper_cls(obj, **kwargs)
                        self.wrapper_obj_map[obj] = user
                        break
            if user is None:
                raise NotImplementedError()
        else:
            user = obj
        assert user is not None
        self.user: ResourcePoolUserABC[T] = user
        get_management().clean_user()
        get_management().setup_user(self.user)

    def load(self):
        self.user.lock()
        self.user.on_load()

    def offload(self):
        self.user.unlock()

    def get_execution_device(self):
        return self.user.get_runtime_device()

    def get_manage_object(self):
        return self.user.manage_object

    @classmethod
    def register_type_wrapper(cls, tp: type, wrapper_cls: type) -> None:
        if tp in cls.type_wrapper_map:
            raise RuntimeError(f"type {tp} registed with {cls.type_wrapper_map[tp]}")
        cls.type_wrapper_map[tp] = wrapper_cls


@contextlib.contextmanager
def auto_manage(obj: T, **kwargs: Any) -> Iterator[AutoManageWrapper[T]]:
    """
    A context manager to automatically manage the memory of a resource,
    such as a PyTorch model.

    Args:
        obj: The object to manage.
        **kwargs: Configuration options passed to the underlying resource wrapper
                  (e.g., `inference_memory_estimator` for a `TorchModuleWrapper`).
    """
    if isinstance(obj, AutoManageWrapper):
        am = obj
    else:
        am = AutoManageWrapper(obj, **kwargs)
    try:
        # Note: Loading is no longer done automatically on entry.
        # The user should call am.load() explicitly if not using a hook.
        yield am
    finally:
        am.offload()


class AutoManageHook(ModelHook):
    cache_hook_map: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()

    def __init__(self, am: AutoManageWrapper[Any]):
        self.am = am

    @property
    def execution_device(self):
        return self.am.get_execution_device()

    def pre_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        self.context_token = sire_inference_context.set({"args": args, "kwargs": kwargs})
        self.am.load()
        return send_to_device(args, self.am.get_execution_device()), send_to_device(
            kwargs, self.am.get_execution_device()
        )

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
        self.am.offload()
        if hasattr(self, "context_token"):
            sire_inference_context.reset(self.context_token)
            del self.context_token
        return output

    @classmethod
    def manage_module(cls, module: torch.nn.Module) -> "AutoManageHook":
        # If a user for this module already exists in the central map, do nothing.
        if module in AutoManageWrapper.wrapper_obj_map:
            # Return the existing hook if possible, or None
            return cls.cache_hook_map.get(module, None)

        hook = cls.cache_hook_map.get(module, None)
        if hook is None or getattr(module, "_hf_hook", None) is None:
            hook = cls(AutoManageWrapper(module))
            add_hook_to_module(module, hook, append=True)
            cls.cache_hook_map[module] = hook
        return hook


class ResourcePoolManagement:
    def __init__(self) -> None:
        self.resource_pools: dict[resources_device, ResourcePool] = {}
        self.user_pool_map: dict[ResourcePoolUserABC[Any], list[ResourcePool]] = {}

    def get_resource_pool(self, device: resources_device) -> Optional[ResourcePool]:
        if device.type != "cpu" and device.index is None:
            device = resources_device(device.type, 0)
        return self.resource_pools.get(device, None)

    def set_resource_pool(self, device: resources_device, resource_pool: ResourcePool) -> None:
        self.resource_pools[device] = resource_pool

    def get_devices(self) -> list[resources_device]:
        return list(self.resource_pools.keys())

    def get_user_pools(self, user: ResourcePoolUserABC[Any]) -> list[ResourcePool]:
        return self.user_pool_map.get(user, [])

    def setup_user(self, user: ResourcePoolUserABC[Any]) -> None:
        if len(self.get_user_pools(user)) != 0:
            return
        pools = user.on_setup(self)
        if pools:
            for pool in pools:
                if pool:
                    pool.register_resource_pool_user(user)
            self.user_pool_map[user] = pools

    def forget_user(self, user: ResourcePoolUserABC[Any]) -> None:
        pools = self.user_pool_map.pop(user, [])
        if pools:
            for pool in pools:
                if pool and pool.have_resource_pool_user(user):
                    pool.remove_resource_pool_user(user)

    def clean_user(self):
        for user in list(self.user_pool_map):
            if user.should_clear():
                self.user_pool_map.pop(user)
        clear_device_cache_and_libc_alloc()


MANAGEMENT_INSTANCE = ResourcePoolManagement()


def get_management() -> ResourcePoolManagement:
    return MANAGEMENT_INSTANCE
