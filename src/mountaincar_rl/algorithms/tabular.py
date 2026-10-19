"""Tabular Q-learning and SARSA on a discretized (position, velocity) grid.

The continuous state is cut into `bins x bins` cells and each cell gets its own row of Q-values.
One update per step:

    Q-learning (off-policy): Q(s,a) += alpha * (r + gamma * max_b Q(s',b) - Q(s,a))
    SARSA      (on-policy):  Q(s,a) += alpha * (r + gamma * Q(s',a')    - Q(s,a)),  a' = the action taken next

Why it can work on a sparse reward: every real return is negative (-1 per step), so a table
initialised at 0 is *optimistic*. Every untried (cell, action) looks better than anything tried,
and a greedy agent systematically visits new cells. That is exploration for free, and the reason
these two learn MountainCar in a few hundred episodes without any shaping.
"""

from __future__ import annotations

import numpy as np

from mountaincar_rl.algorithms.base import Agent
from mountaincar_rl.envs import discretize


class TabularAgent(Agent):
    def __init__(self, cfg, n_actions: int = 3, method: str = "q-learning"):
        super().__init__(cfg, n_actions)
        self.name, self.method = method, method
        self.q = np.full((cfg.bins, cfg.bins, n_actions), cfg.q_init, dtype=np.float64)
        self._next_action: int | None = None   # SARSA: a' chosen during the update, used by the next act()

    def _greedy(self, cell) -> int:
        return self.greedy_with_ties(self.q[cell].tolist())   # random tie-break: the table starts full of zeros

    def _eps_greedy(self, cell) -> int:
        if self.pyrng.random() < self.epsilon:
            return self.pyrng.randrange(self.n_actions)
        return self._greedy(cell)

    def start_episode(self) -> None:
        self._next_action = None

    def act(self, obs, explore: bool = True) -> int:
        cell = discretize(obs, self.cfg.bins)
        if not explore:
            return self._greedy(cell)
        if self._next_action is not None:
            a, self._next_action = self._next_action, None
            return a
        return self._eps_greedy(cell)

    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        s, s2 = discretize(obs, self.cfg.bins), discretize(next_obs, self.cfg.bins)
        if terminated:
            target = reward                                   # the flag: no future
        elif self.method == "q-learning":
            target = reward + self.cfg.gamma * self.q[s2].max()
        else:                                                 # SARSA: bootstrap on the action we will take
            a2 = self._eps_greedy(s2)
            target = reward + self.cfg.gamma * self.q[s2 + (a2,)]
            if not truncated:
                self._next_action = a2
        self.q[s + (action,)] += self.cfg.alpha * (target - self.q[s + (action,)])

    def _cells(self, states):
        return [discretize(s, self.cfg.bins) for s in states]

    def policy_grid(self, states):
        return np.array([int(np.argmax(self.q[c])) for c in self._cells(states)])

    def value_grid(self, states):
        return np.array([self.q[c].max() for c in self._cells(states)])

    def state_dict(self):
        return {"q": self.q}

    def load_state_dict(self, state):
        self.q = np.asarray(state["q"])
