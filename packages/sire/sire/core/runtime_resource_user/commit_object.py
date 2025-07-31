from __future__ import annotations

import logging
from typing import TypeVar

from ..commit_object import CommitObjectProxy
from ..commit_object_auto_device_manage import AutoManageCommitObjectProxy
from ..runtime_resource_management import AutoManageWrapper
from . import WeakRefResourcePoolUser

_logger = logging.getLogger(__name__)
T = TypeVar("T")


class CommitObjectProxyWrapper(WeakRefResourcePoolUser[CommitObjectProxy[T]]):
    def __init__(self, obj, **kwargs) -> None:
        super().__init__(obj)
        self.wrapper_kwargs = kwargs

    def on_setup(self, manager):
        self.base_object_am = AutoManageWrapper(self.manage_object.base_object, **self.wrapper_kwargs)
        self.resource_pool = manager.get_user_pools(self.base_object_am)
        return self.resource_pool

    def on_load(self, user_context=None):
        if isinstance(self.manage_object, AutoManageCommitObjectProxy):
            self.manage_object.base_object_ref.am = self.base_object_am
            self.manage_object.base_object_ref.am_user_context = user_context
        else:
            self.manage_object.get_current_object()
            self.base_object_am.load(user_context=user_context)

        super().on_load()

    def on_resource_request(self, device, size):
        # handle by self.base_object_am
        super().on_resource_request(device, size)

    def get_runtime_device(self):
        return self.base_object_am.get_execution_device()

    def unlock(self):
        super().unlock()
        self.base_object_am.user.unlock()
