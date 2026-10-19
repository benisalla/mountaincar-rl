"""Small MLPs used by the neural methods (inputs are normalised (position, velocity) in [-1, 1]^2)."""

from __future__ import annotations

import torch
from torch import nn


def mlp(n_in: int, n_out: int, hidden: int = 64, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


class QNet(nn.Module):
    """Q(s, .) for the 3 discrete actions."""

    def __init__(self, hidden: int = 64, n_actions: int = 3):
        super().__init__()
        self.net = mlp(2, n_actions, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling head (Wang et al. 2016): Q(s, a) = V(s) + A(s, a) - mean_b A(s, b).

    V learns "how good is this state" from every transition, whatever the action, which helps when
    most actions have similar values (true almost everywhere in MountainCar).
    """

    def __init__(self, hidden: int = 64, n_actions: int = 3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean(dim=-1, keepdim=True)
