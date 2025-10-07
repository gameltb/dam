"""Utilities for loading static dictionaries."""

from pathlib import Path

import safetensors
import safetensors.torch
import torch


def load_state_dict(file_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """
    Load a state dictionary from a file.

    This function supports both `.safetensors` and PyTorch's native `.pth` format.

    Args:
        file_path: The path to the file.
        device: The device to load the tensors onto.

    Returns:
        The loaded state dictionary.

    """
    model = {}

    ext = Path(file_path).suffix

    if ext.lower() in (".safetensors", ".st"):
        model = safetensors.torch.load_file(file_path, device=device)  # type: ignore
    else:
        model = torch.load(file_path, map_location=device, mmap=True, weights_only=True)

    return model
