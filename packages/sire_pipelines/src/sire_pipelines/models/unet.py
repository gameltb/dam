"""A dummy UNet model."""

import torch
from torch import nn


class UNet(nn.Module):
    """A dummy UNet model to simulate a large component."""

    def __init__(self):
        """Initialize the UNet model."""
        super().__init__()  # type: ignore
        self.layer1 = nn.Linear(128, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the UNet model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.

        """
        return self.layer3(self.layer2(self.layer1(x)))
