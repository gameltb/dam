from __future__ import annotations

import logging
import sys
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, List, Optional, Set, TypeVar

import torch

if TYPE_CHECKING:
    from ..runtime_resource_management import ResourcePoolManagement
    from ..runtime_resource_pool import ResourcePool

T = TypeVar("T")

resources_device = torch.device


class ResourcePoolUserABC(ABC, Generic[T]):
    def __init__(self) -> None:
        self._locked = False
        self.loaded = False

    @property
    @abstractmethod
    def manage_object(self) -> Optional[T]:
        ...

    @abstractmethod
    def on_setup(self, manager: "ResourcePoolManagement") -> list["ResourcePool"]:
        ...

    @abstractmethod
    def on_load(self) -> None:
        self.loaded = True

    @abstractmethod
    def on_resource_request(self, device: resources_device, size: int) -> None:
        self.loaded = False

    @abstractmethod
    def get_used_resource_size(self, device: resources_device) -> int:
        ...

    @abstractmethod
    def get_used_resource_devices(self) -> Set[resources_device]:
        ...

    @abstractmethod
    def get_runtime_device(self) -> resources_device:
        """get execution device for manage_object, mainly for torch module.

        Returns:
            resources_device: execution device
        """
        ...

    def should_clear(self) -> bool:
        return True

    def lock(self) -> None:
        self._locked = True

    def unlock(self) -> None:
        self._locked = False

    @property
    def locked(self) -> bool:
        return self._locked


class BaseResourcePoolUser(ResourcePoolUserABC[T]):
    def __init__(self, manage_object: T) -> None:
        super().__init__()
        self._manage_object = manage_object

    @property
    def manage_object(self) -> T:
        return self._manage_object

    def should_clear(self) -> bool:
        return sys.getrefcount(self.manage_object) <= 2


class WeakRefResourcePoolUser(ResourcePoolUserABC[T]):
    def __init__(self, manage_object: T) -> None:
        super().__init__()
        self._manage_object_weak_ref = weakref.ref(manage_object)
        if hasattr(manage_object, "__class__"):
            self.logger = logging.getLogger(
                f"{__name__}_{self.__class__.__name__}_{manage_object.__class__.__name__}"
            )
        else:
            self.logger = logging.getLogger(f"{__name__}_{self.__class__.__name__}")

    @property
    def manage_object(self) -> Optional[T]:
        return self._manage_object_weak_ref()

    def should_clear(self) -> bool:
        return self.manage_object is None
