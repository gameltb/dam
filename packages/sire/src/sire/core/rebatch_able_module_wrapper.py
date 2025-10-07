"""A wrapper for modules that allows for re-batching."""

# type: ignore
from collections import OrderedDict
from typing import Any

import torch


class PatchModuleKwargsHook:
    """A hook to inject keyword arguments into a module's forward pass."""

    def __init__(self) -> None:
        """Initialize the hook."""
        self.ext_kwargs: dict[str, Any] = {}

    def __call__(
        self, _module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Inject the external keyword arguments."""
        kwargs.update(self.ext_kwargs)
        return (args, kwargs)


class RebatchAbleModuleWrapper(torch.nn.Module):
    """A wrapper for modules that allows for re-batching."""

    def __init__(self, module: torch.nn.Module) -> None:
        """
        Initialize the wrapper.

        Args:
            module: The module to wrap.

        """
        super().__init__()  # type: ignore
        self.module = module
        self.replaced_module_kwargs_hook_map: OrderedDict[str, PatchModuleKwargsHook] = OrderedDict()

    def forward(self, /, **kwargs: Any) -> Any:
        """
        Forward pass for the wrapper.

        This method extracts hook-specific keyword arguments and passes them to the
        appropriate hooks before calling the wrapped module.
        """
        for hook_id, v in self.replaced_module_kwargs_hook_map.items():
            hook_kwarg_prefix = f"{hook_id}_"
            for arg_name in list(kwargs.keys()):
                if arg_name.startswith(hook_kwarg_prefix):
                    v.ext_kwargs[arg_name.removeprefix(hook_kwarg_prefix)] = kwargs.pop(arg_name)

        return self.module(**kwargs)

    def register_forward_ext_kwargs_hook(self, hook_id: str) -> None:
        """
        Register a hook to extend the forward pass with extra keyword arguments.

        Args:
            hook_id: The unique identifier for the hook.

        Raises:
            Exception: If the hook_id is already registered.

        """
        if hook_id in self.replaced_module_kwargs_hook_map:
            raise Exception(f"hook_id {hook_id} already registered")
        hook = PatchModuleKwargsHook()
        self.replaced_module_kwargs_hook_map[hook_id] = hook

    def apply_forward_ext_kwargs_hook(self, module: torch.nn.Module, hook_id: str) -> None:
        """
        Apply a registered forward hook to a specific module.

        Args:
            module: The module to apply the hook to.
            hook_id: The ID of the hook to apply.

        Raises:
            Exception: If the hook_id is not registered.

        """
        ext_kwargs_hook = self.replaced_module_kwargs_hook_map.get(hook_id)
        if ext_kwargs_hook is None:
            raise Exception(f"module_ext_kwargs_hook_id {hook_id} not registered")
        module.register_forward_pre_hook(ext_kwargs_hook, with_kwargs=True)
