from __future__ import annotations

import weakref
from typing import Any, TypeVar

from torch import device

from ..commit_object import CommitObjectProxy
from ..runtime_resource_management import (
    AutoManageWrapper,
    ResourcePool,
    ResourcePoolManagement,
)
from . import WeakRefResourcePoolUser

T = TypeVar("T")


class CommitObjectProxyWrapper(WeakRefResourcePoolUser[CommitObjectProxy[T]]):
    def __init__(self, obj: CommitObjectProxy[T], **kwargs: Any) -> None:
        super().__init__(obj)
        self.wrapper_kwargs = kwargs
        self.base_object_am: AutoManageWrapper[Any] | None = None

        # Link this manager to the proxy
        if (proxy := self.manage_object) is not None:
            proxy.am_ref = weakref.ref(self)

    def on_setup(self, manager: ResourcePoolManagement) -> list[ResourcePool]:
        if (proxy := self.manage_object) is None:
            return []

        # The AMW for the base object handles the actual resource management
        self.base_object_am = AutoManageWrapper(proxy.base_object, **self.wrapper_kwargs)
        # The wrapper for the proxy itself doesn't need a separate pool,
        # it delegates to the base object's wrapper.
        if self.base_object_am and self.base_object_am.user:
            return manager.get_user_pools(self.base_object_am.user)
        return []

    def on_load(self) -> None:
        if self.base_object_am:
            self.base_object_am.load()
        super().on_load()

    def on_resource_request(self, device: device, size: int) -> None:
        # Handled by the base object's AutoManageWrapper
        super().on_resource_request(device, size)

    def get_runtime_device(self) -> device | None:
        if self.base_object_am:
            return self.base_object_am.get_execution_device()
        return None

    def unlock(self) -> None:
        super().unlock()
        if self.base_object_am:
            self.base_object_am.offload()
