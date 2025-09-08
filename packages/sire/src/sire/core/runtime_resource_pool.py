import logging
import weakref

from ..utils.runtime_resource_util import get_free_mem_size_cpu, get_free_mem_size_cuda_pytorch
from .runtime_resource_user import ResourcePoolUserABC, resources_device

_logger = logging.getLogger(__name__)


class ResourcePool:
    def __init__(self, runtime_device: resources_device = resources_device("cpu", 0)) -> None:
        self.users: weakref.WeakSet[ResourcePoolUserABC] = weakref.WeakSet()
        self.resource_device: resources_device = runtime_device
        self.pool_size = 0
        self.shared_pool_size = 0

    def request_resource(self, size: int):
        for user in self.users:
            if user.locked:
                _logger.warning(
                    f"request_resource skip user of obj {user.manage_object.__class__.__name__} becuse locked."
                )
                continue
            pool_free_size = self.get_pool_free_size()
            free_size_need = size - pool_free_size
            if free_size_need > 0:
                user.on_resource_request(self.resource_device, free_size_need)
            else:
                break

    def get_pool_device(self) -> resources_device:
        return self.resource_device

    def get_pool_free_size(self) -> int:
        raise NotImplementedError()

    def register_resource_pool_user(self, user: ResourcePoolUserABC):
        self.users.add(user)

    def remove_resource_pool_user(self, user: ResourcePoolUserABC):
        self.users.remove(user)

    def have_resource_pool_user(self, user: ResourcePoolUserABC) -> bool:
        return user in self.users


class ResourcePoolCUDA(ResourcePool):
    def get_pool_free_size(self):
        return get_free_mem_size_cuda_pytorch(self.resource_device)


class ResourcePoolCPU(ResourcePool):
    def get_pool_free_size(self):
        return get_free_mem_size_cpu()
