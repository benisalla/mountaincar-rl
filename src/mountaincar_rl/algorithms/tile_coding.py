"""Linear function approximation with tile coding (Sutton & Barto, chapter 10.1).

Instead of one grid, `tilings` grids of `tiles x tiles` cells are laid over the state space, each
shifted by a fraction of a cell. A state activates exactly one tile per tiling, so its feature
vector x(s) is binary with `tilings` ones. Q(s, a) = sum of the weights of the active tiles:

    Q(s, a) = w_a . x(s)
    semi-gradient SARSA:      w_a += alpha / tilings * (r + gamma * Q(s', a') - Q(s, a)) * x(s)
    semi-gradient Q-learning: same with max_b Q(s', b)

Why this beats a plain grid: neighbouring states share most of their tiles, so one update
generalises to the neighbourhood (smooth Q), while the overlap of shifted grids still resolves
fine detail. Weights at 0 are optimistic (as in tabular.py), which is all the exploration needed.
"""

from __future__ import annotations

import numpy as np

from mountaincar_rl.algorithms.base import Agent
from mountaincar_rl.envs import OBS_HIGH, OBS_LOW


class TileCoder:
    """Uniformly offset grid tilings over the 2-D box; `active(s)` returns `tilings` feature indices."""

    def __init__(self, tilings: int = 8, tiles: int = 8, low=OBS_LOW, high=OBS_HIGH):
        self.tilings, self.tiles = tilings, tiles
        self.low, self.high = np.asarray(low, np.float64), np.asarray(high, np.float64)
        # each tiling covers (tiles + 1) cells so that the shifted grid still covers the whole box
        self.per_tiling = (tiles + 1) ** 2
        self.n_features = tilings * self.per_tiling
        # asymmetric offsets (1, 3) * k / tilings, as recommended in Sutton & Barto
        self.offsets = np.array([[(1 * k) % tilings, (3 * k) % tilings] for k in range(tilings)]) / tilings

    def active(self, obs) -> np.ndarray:
        scaled = (np.asarray(obs, np.float64) - self.low) / (self.high - self.low) * self.tiles
        coords = np.floor(scaled[None, :] + self.offsets).astype(int)
        coords = np.clip(coords, 0, self.tiles)
        return np.arange(self.tilings) * self.per_tiling + coords[:, 0] * (self.tiles + 1) + coords[:, 1]


class TileCodingAgent(Agent):
    name = "tile-coding"

    def __init__(self, cfg, n_actions: int = 3):
        super().__init__(cfg, n_actions)
        self.coder = TileCoder(cfg.tilings, cfg.tiles)
        self.w = np.full((n_actions, self.coder.n_features), cfg.q_init / cfg.tilings)
        self.step_size = cfg.alpha / cfg.tilings
        self._next_action: int | None = None

    def q_values(self, obs) -> np.ndarray:
        return self.w[:, self.coder.active(obs)].sum(axis=1)

    def _eps_greedy(self, obs) -> int:
        if self.pyrng.random() < self.epsilon:
            return self.pyrng.randrange(self.n_actions)
        return self.greedy_with_ties(self.q_values(obs).tolist())

    def start_episode(self) -> None:
        self._next_action = None

    def act(self, obs, explore: bool = True) -> int:
        if not explore:
            return self.greedy_with_ties(self.q_values(obs).tolist())
        if self._next_action is not None:
            a, self._next_action = self._next_action, None
            return a
        return self._eps_greedy(obs)

    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        feats = self.coder.active(obs)
        q_sa = self.w[action, feats].sum()
        if terminated:
            target = reward
        elif self.cfg.td_target == "q":
            target = reward + self.cfg.gamma * self.q_values(next_obs).max()
        else:
            a2 = self._eps_greedy(next_obs)
            target = reward + self.cfg.gamma * self.w[a2, self.coder.active(next_obs)].sum()
            if not truncated:
                self._next_action = a2
        # gradient of w_a . x(s) w.r.t. w_a is x(s): only the active tiles move
        self.w[action, feats] += self.step_size * (target - q_sa)

    def policy_grid(self, states):
        return np.array([int(np.argmax(self.q_values(s))) for s in states])

    def value_grid(self, states):
        return np.array([self.q_values(s).max() for s in states])

    def state_dict(self):
        return {"w": self.w}

    def load_state_dict(self, state):
        self.w = np.asarray(state["w"])
