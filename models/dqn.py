"""The four outputs are unrestricted action values, not probabilities."""
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
