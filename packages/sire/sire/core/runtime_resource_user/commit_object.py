from __future__ import annotations

import weakref

from ..commit_object import CommitObjectProxy
from ..runtime_resource_management import AutoManageWrapper
from . import WeakRefResourcePoolUser


class CommitObjectProxyWrapper(WeakRefResourcePoolUser[CommitObjectProxy]):
    def __init__(self, obj: CommitObjectProxy, **kwargs) -> None:
        super().__init__(obj)
        self.wrapper_kwargs = kwargs
        self.base_object_am: AutoManageWrapper | None = None

        # Link this manager to the proxy
        if (proxy := self.manage_object) is not None:
            proxy.am_ref = weakref.ref(self)

    def on_setup(self, manager):
        if (proxy := self.manage_object) is None:
            return []

        # The AMW for the base object handles the actual resource management
        self.base_object_am = AutoManageWrapper(proxy.base_object, **self.wrapper_kwargs)
        # The wrapper for the proxy itself doesn't need a separate pool,
        # it delegates to the base object's wrapper.
        return manager.get_user_pools(self.base_object_am.user)

    def on_load(self):
        if self.base_object_am:
            self.base_object_am.load()
        super().on_load()

    def on_resource_request(self, device, size):
        # Handled by the base object's AutoManageWrapper
        super().on_resource_request(device, size)

    def get_runtime_device(self):
        if self.base_object_am:
            return self.base_object_am.get_execution_device()
        # Fallback or error
        raise RuntimeError("CommitObjectProxyWrapper has no execution device without a base object manager.")

    def unlock(self):
        super().unlock()
        if self.base_object_am:
            self.base_object_am.offload()
