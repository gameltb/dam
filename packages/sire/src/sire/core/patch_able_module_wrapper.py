# type: ignore
# wrapper to module have pacth module, make it run and can be export to onnx.
from collections import OrderedDict
from functools import reduce
from typing import Any, Dict, Tuple, Union

import torch


# module can be inject to other network and get ext input it self.
class PatchModuleKwargsHook:
    """"""

    def __init__(self) -> None:
        self.ext_kwargs: Dict[str, Any] = {}

    def __call__(
        self, module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        kwargs.update(self.ext_kwargs)
        return (args, kwargs)


class PatchAbleModuleWrapper(torch.nn.Module):
    """需要两种补丁模式,
    1. 直接替换模块,此模块本身即为一个正常nn.Module,只不过可以添加额外的输入参数,Wrapper接受参数字典,然后将对应模块的额外参数通过hook发送到指定的模块.
    2. 控制流补丁,这种需要对应的模块为专门编写的可以接受和应用补丁,补丁需要可以提供可能静态字典, 通过字符串识别补丁类型
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.replaced_module_kwargs_hook_map: "OrderedDict[str, PatchModuleKwargsHook]" = OrderedDict()

    def forward(self, /, **kwargs: Any) -> Any:
        for hook_id, v in self.replaced_module_kwargs_hook_map.items():
            hook_kwarg_prefix = f"{hook_id}_"
            for arg_name in list(kwargs.keys()):
                if arg_name.startswith(hook_kwarg_prefix):
                    v.ext_kwargs[arg_name.removeprefix(hook_kwarg_prefix)] = kwargs.pop(arg_name)

        return self.module(**kwargs)

    def register_forward_ext_kwargs_hook(self, hook_id: str):
        if hook_id in self.replaced_module_kwargs_hook_map:
            raise Exception(f"hook_id {hook_id} already registered")
        hook = PatchModuleKwargsHook()
        self.replaced_module_kwargs_hook_map[hook_id] = hook

    def apply_forward_ext_kwargs_hook(self, module: torch.nn.Module, hook_id: str):
        ext_kwargs_hook = self.replaced_module_kwargs_hook_map.get(hook_id, None)
        if ext_kwargs_hook is None:
            raise Exception(f"module_ext_kwargs_hook_id {hook_id} not registered")
        module.register_forward_pre_hook(ext_kwargs_hook, with_kwargs=True)


# https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797
def get_module_by_name(module: Union[torch.Tensor, torch.nn.Module], access_string: str) -> Any:
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
