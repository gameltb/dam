"""A utility to manage hooks on a torch.nn.Module."""

import contextlib
import logging

from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module
from torch import nn

logger = logging.getLogger(__name__)


class HookManager:
    """
    A utility to manage hooks on a torch.nn.Module.

    This is particularly for temporarily replacing hooks for operations like
    profiling.

    This manager helps to solve the problem of multiple components wanting to
    control the hooks on a module by providing a centralized, predictable way
    to manipulate them.

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
        Initialize the HookManager.

        Args:
            module: The top-level module to manage hooks for.

        """
        self.module = module
        self._original_hooks: dict[str, ModelHook] | None = None

    def _get_all_hooks(self) -> dict[str, ModelHook]:
        """
        Recursively find and return all hooks attached to the module.

        Hooks are expected to be stored in the `_hf_hook` attribute by `accelerate`.
        """
        all_hooks: dict[str, ModelHook] = {}
        for name, sub_mod in self.module.named_modules():  # type: ignore
            hook = getattr(sub_mod, "_hf_hook", None)  # type: ignore
            if hook is not None:
                all_hooks[name] = hook
        return all_hooks

    def _add_hooks(self, hooks: dict[str, ModelHook], append: bool = True) -> None:
        """Add a dictionary of hooks to the corresponding submodules."""
        module_map = {name: mod for name, mod in self.module.named_modules()}  # type: ignore
        for name, hook in hooks.items():
            if name in module_map:
                add_hook_to_module(module_map[name], hook, append=append)  # type: ignore
            else:
                logger.warning("Could not find submodule '%s' to attach hook.", name)

    def _clear_all_hooks(self) -> None:
        """Remove all hooks from the module and its submodules."""
        remove_hook_from_module(self.module, recurse=True)

    @contextlib.contextmanager
    def scope(self):
        """
        Create a scope for temporary hook modifications.

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
        logger.debug("Storing %d original hooks for temporary scope.", len(self._original_hooks))
        self._clear_all_hooks()

        try:
            yield
        finally:
            self._clear_all_hooks()
            if self._original_hooks:
                logger.debug("Restoring %d original hooks.", len(self._original_hooks))
                self._add_hooks(self._original_hooks, append=False)
            self._original_hooks = None
            logger.debug("Restored original hooks from temporary scope.")
