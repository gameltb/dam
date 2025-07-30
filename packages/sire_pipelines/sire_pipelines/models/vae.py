import torch
from torch import nn

class VAE(nn.Module):
    """A dummy VAE model to simulate a smaller component."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
        self.decoder = nn.Linear(64, 256)

    def forward(self, x):
        return self.decoder(self.encoder(x))
