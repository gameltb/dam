__version__ = "0.2.0"

import diffusers
import torch
from torch import device as torch_device

from .core.commit_object import CallableCommit, CommitABC, CommitObjectProxy
from .core.optimizer import InferenceOptimizerCommit
from .core.runtime_resource_management import (
    AutoManageHook,
    AutoManageWrapper,
    auto_manage,
)
from .core.runtime_resource_management import get_management as get_resource_management
from .core.runtime_resource_pool import ResourcePoolCPU, ResourcePoolCUDA
from .core.runtime_resource_user.diffusers_pipeline import DiffusersPipelineWrapper
from .core.runtime_resource_user.pytorch_module import TorchModuleWrapper

_initialized = False


def initialize():
    """
    Initializes Sire's environment. Sets up default CPU and CUDA resource pools
    and registers the default type wrappers for common libraries like PyTorch
    and Diffusers. This function is idempotent.
    """
    global _initialized
    if _initialized:
        return

    # Set up CPU and CUDA resource pools
    management = get_resource_management()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch_device("cuda", i)
            if management.get_resource_pool(device) is None:
                management.set_resource_pool(device, ResourcePoolCUDA(device))
    if management.get_resource_pool(torch.device("cpu")) is None:
        management.set_resource_pool(torch_device("cpu"), ResourcePoolCPU(torch_device("cpu")))

    # Register default type wrappers
    AutoManageWrapper.registe_type_wrapper(torch.nn.Module, TorchModuleWrapper)
    AutoManageWrapper.registe_type_wrapper(diffusers.DiffusionPipeline, DiffusersPipelineWrapper)
    _initialized = True


def manage(model_object):
    """
    Wraps a model or object to be managed by Sire's resource manager.
    This is the main entry point for making an object "Sire-aware".
    """
    return AutoManageWrapper(model_object)


# Expose the core components for advanced users
__all__ = [
    "__version__",
    "initialize",
    "auto_manage",
    "AutoManageHook",
    "manage",
    "CommitObjectProxy",
    "CommitABC",
    "CallableCommit",
    "get_resource_management",
    "InferenceOptimizerCommit",
    "TorchModuleWrapper",
    "DiffusersPipelineWrapper",
]
