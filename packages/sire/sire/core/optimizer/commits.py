from __future__ import annotations

import torch.nn as nn
from accelerate.hooks import add_hook_to_module, remove_hook_from_module

from ..commit_object import CommitABC
from .hooks import InferenceOptimizerHook


class InferenceOptimizerCommit(CommitABC[nn.Module]):
    """
    A commit that applies the InferenceOptimizerHook to a torch.nn.Module.
    """

    def __init__(self, **kwargs):
        """
        Initializes the commit with arguments for the InferenceOptimizerHook.
        """
        super().__init__()
        self.hook_kwargs = kwargs
        self.hook_instance: InferenceOptimizerHook | None = None

    def apply(self, base_object: nn.Module, **kwargs):
        """
        Applies the InferenceOptimizerHook to the base module.
        If a hook is already applied, it's first removed.
        """
        if self.hook_instance:
            self.revert(base_object)

        self.hook_instance = InferenceOptimizerHook(**self.hook_kwargs)
        add_hook_to_module(base_object, self.hook_instance, append=False)

    def revert(self, base_object: nn.Module):
        """
        Removes the InferenceOptimizerHook from the base module.
        """
        if self.hook_instance:
            remove_hook_from_module(self.hook_instance)
            self.hook_instance = None
