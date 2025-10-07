"""Abstract base classes and concrete implementations for resource pool users."""

from __future__ import annotations

import logging
import sys
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import torch

if TYPE_CHECKING:
    from ..runtime_resource_management import ResourcePoolManagement
    from ..runtime_resource_pool import ResourcePool

T = TypeVar("T")

resources_device = torch.device

# The number of references to an object that are considered "local" to the
# object itself. This is used to determine if an object has any external
# references.
LOCAL_REFERENCES_COUNT = 2


class ResourcePoolUserABC[T](ABC):
    """Abstract base class for a user of a resource pool."""

    def __init__(self) -> None:
        """Initialize the resource pool user."""
        self._locked = False
        self.loaded = False

    @property
    @abstractmethod
    def manage_object(self) -> T | None:
        """The object being managed."""

    @abstractmethod
    def on_setup(self, manager: ResourcePoolManagement) -> list[ResourcePool]:
        """Set up the user when it is first registered."""

    @abstractmethod
    def on_load(self) -> None:
        """Load the user's resources."""
        self.loaded = True

    @abstractmethod
    def on_resource_request(self, device: resources_device, size: int) -> None:
        """Handle a resource request from the pool."""
        self.loaded = False

    @abstractmethod
    def get_used_resource_size(self, device: resources_device) -> int:
        """Get the size of the resources used by this user on a specific device."""

    @abstractmethod
    def get_used_resource_devices(self) -> set[resources_device]:
        """Get the set of devices this user is using resources on."""

    @abstractmethod
    def get_runtime_device(self) -> resources_device | None:
        """Get the execution device for the managed object."""

    def should_clear(self) -> bool:
        """Whether this user should be cleared from the resource pool."""
        return True

    def lock(self) -> None:
        """Lock the user, preventing its resources from being offloaded."""
        self._locked = True

    def unlock(self) -> None:
        """Unlock the user, allowing its resources to be offloaded."""
        self._locked = False

    @property
    def locked(self) -> bool:
        """Whether the user is locked."""
        return self._locked


class BaseResourcePoolUser(ResourcePoolUserABC[T]):
    """A basic resource pool user that holds a strong reference to the managed object."""

    def __init__(self, manage_object: T) -> None:
        """
        Initialize the user.

        Args:
            manage_object: The object to manage.

        """
        super().__init__()
        self._manage_object = manage_object

    @property
    def manage_object(self) -> T:
        """The object being managed."""
        return self._manage_object

    def should_clear(self) -> bool:
        """Clear the user if the managed object has no other references."""
        return sys.getrefcount(self.manage_object) <= LOCAL_REFERENCES_COUNT


class WeakRefResourcePoolUser(ResourcePoolUserABC[T]):
    """A resource pool user that holds a weak reference to the managed object."""

    def __init__(self, manage_object: T) -> None:
        """
        Initialize the user.

        Args:
            manage_object: The object to manage.

        """
        super().__init__()
        self._manage_object_weak_ref = weakref.ref(manage_object)
        if hasattr(manage_object, "__class__"):
            self.logger = logging.getLogger(f"{__name__}_{self.__class__.__name__}_{manage_object.__class__.__name__}")
        else:
            self.logger = logging.getLogger(f"{__name__}_{self.__class__.__name__}")

    @property
    def manage_object(self) -> T | None:
        """The object being managed (or None if it has been garbage collected)."""
        return self._manage_object_weak_ref()

    def should_clear(self) -> bool:
        """Clear the user if the managed object has been garbage collected."""
        return self.manage_object is None
