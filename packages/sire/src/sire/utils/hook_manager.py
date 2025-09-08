import contextlib
import logging
from typing import Dict, Optional

import torch.nn as nn
from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module

logger = logging.getLogger(__name__)


class HookManager:
    """
        A utility to manage hooks on a torch.nn.Module, particularly for temporarily
        replacing hooks for operations like profiling.

        This manager helps to solve the problem of multiple components wanting to
        control the hooks on a module by providing a centralized, predictable way
    to
        manipulate them.

        Example:
            >>> model = MyModel()
            >>> hook_manager = HookManager(model)
            >>>
            >>> # To run a profiling pass with temporary hooks:
            >>> with hook_manager.temporary_hooks([MyProfilerHook()]):
            ...     # Inside this block, only MyProfilerHook is active.
            ...     model(data)
            >>> # Outside the block, the original hooks are restored.
    """

    def __init__(self, module: nn.Module):
        """
        Initializes the HookManager.

        Args:
            module: The top-level module to manage hooks for.
        """
        self.module = module
        self._original_hooks: Optional[Dict[str, ModelHook]] = None

    def _get_all_hooks(self) -> Dict[str, ModelHook]:
        """
        Recursively finds and returns all hooks attached to the module and its submodules.
        Hooks are expected to be stored in the `_hf_hook` attribute by `accelerate`.
        """
        all_hooks: Dict[str, ModelHook] = {}
        for name, sub_mod in self.module.named_modules():  # type: ignore
            hook = getattr(sub_mod, "_hf_hook", None)  # type: ignore
            if hook is not None:
                all_hooks[name] = hook
        return all_hooks

    def _add_hooks(self, hooks: Dict[str, ModelHook], append: bool = True):
        """Adds a dictionary of hooks to the corresponding submodules."""
        module_map = {name: mod for name, mod in self.module.named_modules()}  # type: ignore
        for name, hook in hooks.items():
            if name in module_map:
                add_hook_to_module(module_map[name], hook, append=append)  # type: ignore
            else:
                logger.warning(f"Could not find submodule '{name}' to attach hook.")

    def _clear_all_hooks(self):
        """Removes all hooks from the module and its submodules."""
        remove_hook_from_module(self.module, recurse=True)

    @contextlib.contextmanager
    def scope(self):
        """
        A context manager that creates a scope for temporary hook modifications.

        It saves the original hooks on entry, clears all hooks, and then restores
        the original hooks on exit. This allows for complex, temporary hook
        setups (like for profiling) within the `with` block.

        Example:
            >>> with hook_manager.scope():
            ...     # The model starts with no hooks here.
            ...     dispatch_model(model, ...)  # This adds its own hooks.
            ...     add_hook_to_module(model, MyProfilerHook(), append=True)
            ...     model(data)
            >>> # Outside the block, the original hooks are back.
        """
        self._original_hooks = self._get_all_hooks()
        logger.debug(f"Storing {len(self._original_hooks)} original hooks for temporary scope.")
        self._clear_all_hooks()

        try:
            yield
        finally:
            self._clear_all_hooks()
            if self._original_hooks:
                logger.debug(f"Restoring {len(self._original_hooks)} original hooks.")
                self._add_hooks(self._original_hooks, append=False)
            self._original_hooks = None
            logger.debug("Restored original hooks from temporary scope.")
