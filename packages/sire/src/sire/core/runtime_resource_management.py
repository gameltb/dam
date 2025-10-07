"""Runtime resource management for Sire."""

import contextlib
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any, ClassVar, TypeVar, cast, overload

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device  # type: ignore

from ..utils.runtime_resource_util import clear_device_cache_and_libc_alloc
from .context import sire_inference_context
from .runtime_resource_pool import ResourcePool, resources_device
from .runtime_resource_user import ResourcePoolUserABC

T = TypeVar("T")


class AutoManageWrapper[T]:
    """A wrapper to automatically manage the resources of an object."""

    type_wrapper_map: ClassVar[dict[type, type]] = {}
    wrapper_obj_map: ClassVar[weakref.WeakKeyDictionary[Any, ResourcePoolUserABC[Any]]] = weakref.WeakKeyDictionary()

    def __init__(self, obj: Any, **kwargs: Any) -> None:
        """
        Initialize the wrapper.

        Args:
            obj: The object to manage.
            **kwargs: Additional arguments for the resource pool user.

        """
        if not isinstance(obj, ResourcePoolUserABC):
            user = self.wrapper_obj_map.get(obj)
            if user is None:
                for tp, wrapper_cls in self.type_wrapper_map.items():
                    if isinstance(obj, tp):
                        user = wrapper_cls(obj, **kwargs)
                        self.wrapper_obj_map[obj] = user
                        break
            if user is None:
                raise NotImplementedError()
        else:
            user = cast(ResourcePoolUserABC[Any], obj)
        self.user: ResourcePoolUserABC[T] = user
        get_management().clean_user()
        get_management().setup_user(self.user)

    def load(self) -> None:
        """Load the managed object's resources."""
        self.user.lock()
        self.user.on_load()

    def offload(self) -> None:
        """Offload the managed object's resources."""
        self.user.unlock()

    def get_execution_device(self) -> resources_device | None:
        """Get the execution device of the managed object."""
        return self.user.get_runtime_device()

    def get_manage_object(self) -> T | None:
        """Get the managed object."""
        return self.user.manage_object

    @classmethod
    def register_type_wrapper(cls, tp: type, wrapper_cls: type) -> None:
        """Register a wrapper for a given type."""
        if tp in cls.type_wrapper_map:
            raise RuntimeError(f"type {tp} registed with {cls.type_wrapper_map[tp]}")
        cls.type_wrapper_map[tp] = wrapper_cls


@overload
def auto_manage[T](obj: AutoManageWrapper[T], **kwargs: Any) -> AbstractContextManager[AutoManageWrapper[T]]: ...


@overload
def auto_manage[T](obj: T, **kwargs: Any) -> AbstractContextManager[AutoManageWrapper[T]]: ...


@contextlib.contextmanager
def auto_manage(obj: Any, **kwargs: Any) -> Iterator[AutoManageWrapper[Any]]:
    """
    Automatically manage the memory of a resource.

    This is a context manager for resources like PyTorch models.

    Args:
        obj: The object to manage.
        **kwargs: Configuration options passed to the underlying resource wrapper
                  (e.g., `inference_memory_estimator` for a `TorchModuleWrapper`).

    """
    am: AutoManageWrapper[Any] = (
        cast(AutoManageWrapper[Any], obj) if isinstance(obj, AutoManageWrapper) else AutoManageWrapper(obj, **kwargs)
    )
    try:
        # Note: Loading is no longer done automatically on entry.
        # The user should call am.load() explicitly if not using a hook.
        yield am
    finally:
        am.offload()


class AutoManageHook(ModelHook):
    """A hook to automatically manage the resources of a module."""

    cache_hook_map: ClassVar[weakref.WeakKeyDictionary[Any, "AutoManageHook"]] = weakref.WeakKeyDictionary()

    def __init__(self, am: AutoManageWrapper[Any]):
        """
        Initialize the hook.

        Args:
            am: The AutoManageWrapper for the module.

        """
        self.am = am

    @property
    def execution_device(self) -> resources_device | None:
        """The execution device of the managed module."""
        return self.am.get_execution_device()

    def pre_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:  # noqa: ARG002
        """Load resources before the forward pass."""
        self.context_token = sire_inference_context.set({"args": args, "kwargs": kwargs})
        self.am.load()
        new_args = cast(tuple[Any, ...], send_to_device(args, self.am.get_execution_device()))
        new_kwargs = cast(dict[str, Any], send_to_device(kwargs, self.am.get_execution_device()))
        return new_args, new_kwargs

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:  # noqa: ARG002
        """Offload resources after the forward pass."""
        self.am.offload()
        if hasattr(self, "context_token"):
            sire_inference_context.reset(self.context_token)
            del self.context_token
        return output

    @classmethod
    def manage_module(cls, module: torch.nn.Module) -> "AutoManageHook":
        """
        Manage a module with an AutoManageHook.

        If a hook for this module already exists, it is returned. Otherwise, a new
        hook is created and attached to the module.
        """
        hook = cls.cache_hook_map.get(module)
        if hook is not None:
            return hook

        if hook is None or getattr(module, "_hf_hook", None) is None:
            hook = cls(AutoManageWrapper(module))
            add_hook_to_module(module, hook, append=True)
            cls.cache_hook_map[module] = hook
        return hook


class ResourcePoolManagement:
    """Manages resource pools and users."""

    def __init__(self) -> None:
        """Initialize the resource pool management."""
        self.resource_pools: dict[resources_device, ResourcePool] = {}
        self.user_pool_map: dict[ResourcePoolUserABC[Any], list[ResourcePool]] = {}

    def get_resource_pool(self, device: resources_device) -> ResourcePool | None:
        """Get the resource pool for a given device."""
        if device.type != "cpu":
            device = resources_device(device.type, device.index or 0)
        return self.resource_pools.get(device)

    def set_resource_pool(self, device: resources_device, resource_pool: ResourcePool) -> None:
        """Set the resource pool for a given device."""
        self.resource_pools[device] = resource_pool

    def get_devices(self) -> list[resources_device]:
        """Get a list of all managed devices."""
        return list(self.resource_pools.keys())

    def get_user_pools(self, user: ResourcePoolUserABC[Any]) -> list[ResourcePool]:
        """Get the resource pools for a given user."""
        return self.user_pool_map.get(user, [])

    def setup_user(self, user: ResourcePoolUserABC[Any]) -> None:
        """Set up a new resource user."""
        if len(self.get_user_pools(user)) != 0:
            return
        pools = user.on_setup(self)
        if pools:
            for pool in pools:
                if pool:
                    pool.register_resource_pool_user(user)
            self.user_pool_map[user] = pools

    def forget_user(self, user: ResourcePoolUserABC[Any]) -> None:
        """Remove a user from the management system."""
        pools = self.user_pool_map.pop(user, [])
        if pools:
            for pool in pools:
                if pool and pool.have_resource_pool_user(user):
                    pool.remove_resource_pool_user(user)

    def clean_user(self) -> None:
        """Clean up unused users."""
        for user in list(self.user_pool_map):
            if user.should_clear():
                self.user_pool_map.pop(user)
        clear_device_cache_and_libc_alloc()


MANAGEMENT_INSTANCE = ResourcePoolManagement()


def get_management() -> ResourcePoolManagement:
    """Get the global resource pool management instance."""
    return MANAGEMENT_INSTANCE
