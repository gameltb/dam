"""A tool for checkpointing and debugging tensors."""

import collections
import logging
from typing import Any

import safetensors.torch
import torch

_logger = logging.getLogger(__name__)


class TensorCheckPoint:
    """A class for checkpointing and debugging tensors."""

    def __init__(self) -> None:
        """Initialize the tensor checkpoint."""
        self.state_dict_name_counter: collections.defaultdict[str, int] = collections.defaultdict(lambda: 0)
        self.state_dict: dict[str, torch.Tensor] = {}
        self.check = False
        self.enable = False

    def __call__(self, /, *_: Any, **kwds: torch.Tensor) -> Any:
        """Checkpoint the given tensors."""
        if not self.enable:
            return
        for name, state in kwds.items():
            name_count = self.state_dict_name_counter[name]

            state_dict_name = f"{name}_{name_count}"
            state: torch.Tensor

            if self.check:
                _logger.warning("check_tensor ==> %s", name)
                check_tensor(state, self.state_dict.get(state_dict_name))
            elif state.device.type != "cpu":
                self.state_dict[state_dict_name] = state.cpu()
            else:
                self.state_dict[state_dict_name] = state.clone()

            self.state_dict_name_counter[name] = name_count + 1

    def save(self, path: str) -> None:
        """Save the checkpoint to a file."""
        safetensors.torch.save_file(self.state_dict, path)  # type: ignore

    def load(self, path: str) -> None:
        """Load a checkpoint from a file."""
        self.state_dict = safetensors.torch.load_file(path)  # type: ignore
        self.check = True
        self.state_dict_name_counter = collections.defaultdict(lambda: 0)


def check_tensor(tensor: torch.Tensor, other_tensor: torch.Tensor | None = None) -> None:
    """
    Check a tensor for NaNs, infs, and contiguity.

    Optionally, compare it to another tensor.

    Args:
        tensor: The tensor to check.
        other_tensor: An optional tensor to compare against.

    """
    tensor_is_nan = torch.isnan(tensor)
    if torch.any(tensor_is_nan):
        _logger.warning(torch.nonzero(tensor_is_nan))
        breakpoint()

    tensor_is_inf = torch.isinf(tensor)
    if torch.any(tensor_is_inf):
        _logger.warning(torch.nonzero(tensor_is_inf))
        breakpoint()

    if not tensor.is_contiguous():
        breakpoint()

    if other_tensor is not None:
        try:
            torch.testing.assert_close(tensor, other_tensor.to(device=tensor.device))
        except Exception as e:
            _logger.warning(e)
            breakpoint()


TENSOR_CHECKPOINT = TensorCheckPoint()
