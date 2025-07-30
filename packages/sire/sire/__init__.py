__version__ = "0.2.0"

import torch
from torch import device as torch_device

from .core.commit_object import CallableCommit, CommitABC, CommitObjectProxy
from .core.runtime_resource_management import (
    AutoManageHook,
    AutoManageWrapper,
    auto_manage,
)
from .core.runtime_resource_management import get_management as get_resource_management
from .core.runtime_resource_pool import ResourcePoolCPU, ResourcePoolCUDA
from .core.runtime_resource_user.pytorch_module import TorchModuleWrapper


def setup_default_pools():
    """
    Sets up default CPU and CUDA resource pools.
    """
    management = get_resource_management()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch_device("cuda", i)
            management.set_resource_pool(device, ResourcePoolCUDA(device))
    management.set_resource_pool(torch_device("cpu"), ResourcePoolCPU(torch_device("cpu")))


def manage(model_object):
    """
    Wraps a model or object to be managed by Sire's resource manager.
    This is the main entry point for making an object "Sire-aware".
    """
    return AutoManageWrapper(model_object)


# Expose the core components for advanced users
__all__ = [
    "__version__",
    "auto_manage",
    "AutoManageHook",
    "manage",
    "CommitObjectProxy",
    "CommitABC",
    "CallableCommit",
    "get_resource_management",
    "setup_default_pools",
    "TorchModuleWrapper",
]
