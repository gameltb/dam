import collections
import logging
from typing import Any

import safetensors.torch
import torch

_logger = logging.getLogger(__name__)


class TensorCheckPoint:
    def __init__(self) -> None:
        self.state_dict_name_counter: collections.defaultdict[str, int] = collections.defaultdict(lambda: 0)
        self.state_dict: dict[str, torch.Tensor] = {}
        self.check = False
        self.enable = False

    def __call__(self, /, *_: Any, **kwds: torch.Tensor) -> Any:
        if not self.enable:
            return
        for name, state in kwds.items():
            name_count = self.state_dict_name_counter[name]

            state_dict_name = f"{name}_{name_count}"
            state: torch.Tensor

            if self.check:
                _logger.warning(f"check_tensor ==> {name}")
                check_tensor(state, self.state_dict.get(state_dict_name, None))
            elif state.device.type != "cpu":
                self.state_dict[state_dict_name] = state.cpu()
            else:
                self.state_dict[state_dict_name] = state.clone()

            self.state_dict_name_counter[name] = name_count + 1

    def save(self, path: str, tiny: bool = False):
        safetensors.torch.save_file(self.state_dict, path)  # type: ignore

    def load(self, path: str):
        self.state_dict = safetensors.torch.load_file(path)  # type: ignore
        self.check = True
        self.state_dict_name_counter = collections.defaultdict(lambda: 0)


def check_tensor(tensor: torch.Tensor, other_tensor: torch.Tensor | None = None):
    tensor_is_nan = torch.isnan(tensor)
    if torch.any(tensor_is_nan):
        _logger.warning(torch.nonzero(tensor_is_nan))
        breakpoint()
        pass

    tensor_is_inf = torch.isinf(tensor)
    if torch.any(tensor_is_inf):
        _logger.warning(torch.nonzero(tensor_is_inf))
        breakpoint()
        pass

    if not tensor.is_contiguous():
        breakpoint()
        pass

    if other_tensor is not None:
        try:
            torch.testing.assert_close(tensor, other_tensor.to(device=tensor.device))
        except Exception as e:
            _logger.warning(e)
            breakpoint()
            pass


TENSOR_CHECKPOINT = TensorCheckPoint()
