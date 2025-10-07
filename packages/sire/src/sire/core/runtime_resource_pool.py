"""Runtime resource pools for managing device memory."""

import logging
import weakref
from typing import Any

from ..utils.runtime_resource_util import get_free_mem_size_cpu, get_free_mem_size_cuda_pytorch
from .runtime_resource_user import ResourcePoolUserABC, resources_device

_logger = logging.getLogger(__name__)


class ResourcePool:
    """A resource pool for managing device memory."""

    def __init__(self, runtime_device: resources_device | None = None) -> None:
        """
        Initialize the resource pool.

        Args:
            runtime_device: The device this pool manages resources for.

        """
        if runtime_device is None:
            runtime_device = resources_device("cpu", 0)
        self.users: weakref.WeakSet[ResourcePoolUserABC[Any]] = weakref.WeakSet()
        self.resource_device: resources_device = runtime_device
        self.pool_size = 0
        self.shared_pool_size = 0

    def request_resource(self, size: int) -> None:
        """
        Request a certain amount of resource from the pool.

        This may trigger offloading of other resources.

        Args:
            size: The amount of resource to request in bytes.

        """
        for user in self.users:
            if user.locked:
                _logger.warning(
                    "request_resource skip user of obj %s becuse locked.", user.manage_object.__class__.__name__
                )
                continue
            pool_free_size = self.get_pool_free_size()
            free_size_need = size - pool_free_size
            if free_size_need > 0:
                user.on_resource_request(self.resource_device, free_size_need)
            else:
                break

    def get_pool_device(self) -> resources_device:
        """Get the device this pool is associated with."""
        return self.resource_device

    def get_pool_free_size(self) -> int:
        """Get the free size of the pool in bytes."""
        raise NotImplementedError()

    def register_resource_pool_user(self, user: ResourcePoolUserABC[Any]) -> None:
        """Register a new user with the resource pool."""
        self.users.add(user)

    def remove_resource_pool_user(self, user: ResourcePoolUserABC[Any]) -> None:
        """Remove a user from the resource pool."""
        self.users.remove(user)

    def have_resource_pool_user(self, user: ResourcePoolUserABC[Any]) -> bool:
        """Check if a user is registered with the resource pool."""
        return user in self.users


class ResourcePoolCUDA(ResourcePool):
    """A resource pool for CUDA devices."""

    def get_pool_free_size(self) -> int:
        """Get the free size of the CUDA device memory."""
        return get_free_mem_size_cuda_pytorch(self.resource_device)


class ResourcePoolCPU(ResourcePool):
    """A resource pool for CPU devices."""

    def get_pool_free_size(self) -> int:
        """Get the free size of the CPU memory."""
        return get_free_mem_size_cpu()
