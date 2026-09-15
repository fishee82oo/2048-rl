"""Shared 16 -> 128 -> 128 MLP; action order UP, DOWN, LEFT, RIGHT."""
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(16, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU())
        self.policy = nn.Linear(128, 4)
        self.value = nn.Linear(128, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 2 ** 0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy.weight, 0.01)
        nn.init.orthogonal_(self.value.weight, 1.0)

    def forward(self, states):
        features = self.features(states)
        return self.policy(features), self.value(features).squeeze(-1)

    def distribution(self, states, masks):
        logits, values = self(states)
        masks = masks.bool()
        if masks.shape != logits.shape or not masks.any(dim=-1).all():
            raise ValueError('Each sampled state requires a nonempty four-action mask')
        return Categorical(logits=logits.masked_fill(~masks, -torch.inf)), values
