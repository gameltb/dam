# wrapper to module have pacth module, make it run and can be export to onnx.
from collections import OrderedDict
from functools import reduce
from typing import Annotated, Any, Dict, Tuple, Union

import torch


# module can be inject to other network and get ext input it self.
class PatchModuleKwargsHook:
    """"""

    def __init__(self) -> None:
        self.ext_kwargs = {}

    def __call__(self, module, args, kwargs: Annotated[str, "x":3]) -> Tuple[Any, Dict[str, torch.Tensor]]:
        kwargs.update(self.ext_kwargs)
        return (args, kwargs)


class RebatchAbleModuleWrapper(torch.nn.Module):
    """
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.replaced_module_kwargs_hook_map: OrderedDict[str, PatchModuleKwargsHook] = OrderedDict()

    def forward(self, /, **kwargs):
        for hook_id, v in self.replaced_module_kwargs_hook_map.items():
            hook_kwarg_prefix = f"{hook_id}_"
            for arg_name in list(kwargs.keys()):
                if arg_name.startswith(hook_kwarg_prefix):
                    v.ext_kwargs[arg_name.removeprefix(hook_kwarg_prefix)] = kwargs.pop(arg_name)

        return self.module(**kwargs)

    def register_forward_ext_kwargs_hook(self, hook_id):
        if hook_id in self.replaced_module_kwargs_hook_map:
            raise Exception(f"hook_id {hook_id} already registered")
        hook = PatchModuleKwargsHook()
        self.replaced_module_kwargs_hook_map[hook_id] = hook

    def apply_forward_ext_kwargs_hook(self, module: torch.nn.Module, hook_id):
        ext_kwargs_hook = self.replaced_module_kwargs_hook_map.get(hook_id, None)
        if ext_kwargs_hook is None:
            raise Exception(f"module_ext_kwargs_hook_id {hook_id} not registered")
        module.register_forward_pre_hook(ext_kwargs_hook, with_kwargs=True)


# https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797
def get_module_by_name(module: Union[torch.Tensor, torch.nn.Module], access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
