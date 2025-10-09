"""A dummy VAE model."""

import torch
from torch import nn


class VAE(nn.Module):
    """A dummy VAE model to simulate a smaller component."""

    def __init__(self):
        """Initialize the VAE model."""
        super().__init__()  # type: ignore
        self.encoder = nn.Linear(256, 64)
        self.decoder = nn.Linear(64, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the VAE model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.

        """
        return self.decoder(self.encoder(x))
