"""The one interface every algorithm implements, plus epsilon schedules.

The training loop (train.py) only ever calls these five methods, so a tabular Q-table, a DQN and
the cross-entropy method all run through the same loop, the same logging and the same evaluation:

    a = agent.act(obs, explore=True)                  # choose an action
    agent.observe(obs, a, r, next_obs, term, trunc)   # learn from one transition
    agent.end_episode()                               # per-episode bookkeeping (CEM, REINFORCE)
    agent.policy_grid(states) / agent.value_grid(states)  # for heatmaps over (position, velocity)
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch


def epsilon(schedule: str, progress: float, start: float, end: float, decay: float) -> float:
    """Exploration rate after `progress` (0..1) of training.

    const:  always `start`
    linear: start -> end over the first `decay` fraction of training, then end
    exp:    end + (start - end) * exp(-5 progress / decay)
    """
    if schedule == "const":
        return start
    if schedule == "linear":
        frac = min(1.0, progress / max(decay, 1e-8))
        return start + frac * (end - start)
    if schedule == "exp":
        return end + (start - end) * math.exp(-5.0 * progress / max(decay, 1e-8))
    raise ValueError(schedule)


class Agent:
    """Base class. `progress` (0..1) is set by the training loop at the start of every episode."""

    name = "agent"
    continuous = False

    def __init__(self, cfg, n_actions: int = 3):
        self.cfg = cfg
        self.n_actions = n_actions
        self.progress = 0.0
        self.rng = np.random.default_rng(cfg.seed)
        self.pyrng = random.Random(cfg.seed)   # plain-Python RNG: faster than numpy for single draws

    @property
    def epsilon(self) -> float:
        c = self.cfg
        return epsilon(c.eps_schedule, self.progress, c.eps_start, c.eps_end, c.eps_decay)

    def greedy_with_ties(self, q) -> int:
        """argmax with a random choice among equal values (matters for tables full of zeros)."""
        q = list(q)
        m = max(q)
        best = [a for a, v in enumerate(q) if v == m]
        return best[0] if len(best) == 1 else self.pyrng.choice(best)

    def act(self, obs: np.ndarray, explore: bool = True):
        raise NotImplementedError

    def observe(self, obs, action, reward: float, next_obs, terminated: bool, truncated: bool) -> None:
        pass

    def end_episode(self) -> None:
        pass

    def start_episode(self) -> None:
        pass

    def policy_grid(self, states: np.ndarray) -> np.ndarray:
        """Greedy action for each state (int for discrete, float force for continuous)."""
        return np.array([self.act(s, explore=False) for s in states])

    def value_grid(self, states: np.ndarray) -> np.ndarray | None:
        """State value V(s) (or max_a Q(s, a)) for each state, if the method has one."""
        return None

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
