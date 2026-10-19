"""Replay buffers: uniform, prioritized (sum tree), and an n-step return accumulator.

Prioritized replay (Schaul et al. 2016): transition i is sampled with probability
p_i^alpha / sum_k p_k^alpha, where p_i = |TD error| + eps, and the loss is re-weighted by the
importance weight (N P(i))^-beta / max_j w_j, with beta annealed to 1. On MountainCar the rare
transitions near the flag have large TD errors, so they get replayed far more often.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int = 2, act_dim: int = 1, seed: int = 0):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros((capacity, act_dim), np.float32)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)      # 1 only when the flag was reached (terminated)
        self.discounts = np.ones(capacity, np.float32)   # gamma^n for n-step transitions
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done, discount=1.0) -> int:
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i], self.discounts[i] = next_obs, done, discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _batch(self, idx):
        return {"obs": self.obs[idx], "actions": self.actions[idx], "rewards": self.rewards[idx],
                "next_obs": self.next_obs[idx], "dones": self.dones[idx], "discounts": self.discounts[idx],
                "idx": idx, "weights": np.ones(len(idx), np.float32)}

    def sample(self, batch_size: int) -> dict:
        return self._batch(self.rng.integers(0, self.size, batch_size))

    def update_priorities(self, idx, td_errors) -> None:
        pass

    def __len__(self) -> int:
        return self.size


class SumTree:
    """Binary tree whose leaves hold priorities and whose nodes hold sums: O(log N) sample and update."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, np.float64)

    def update(self, i: int, value: float) -> None:
        j = i + self.capacity
        self.tree[j] = value
        j //= 2
        while j >= 1:
            self.tree[j] = self.tree[2 * j] + self.tree[2 * j + 1]
            j //= 2

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def find(self, mass: float) -> int:
        """Leaf index i such that the prefix sum of priorities first exceeds `mass`."""
        j = 1
        while j < self.capacity:
            left = 2 * j
            if mass <= self.tree[left]:
                j = left
            else:
                mass -= self.tree[left]
                j = left + 1
        return j - self.capacity

    def __getitem__(self, i: int) -> float:
        return float(self.tree[i + self.capacity])


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, obs_dim: int = 2, act_dim: int = 1, alpha: float = 0.6,
                 beta: float = 0.4, eps: float = 1e-3, seed: int = 0):
        super().__init__(capacity, obs_dim, act_dim, seed)
        self.tree = SumTree(capacity)
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.max_priority = 1.0

    def add(self, *args, **kwargs) -> int:
        i = super().add(*args, **kwargs)
        self.tree.update(i, self.max_priority ** self.alpha)   # new transitions: replay at least once
        return i

    def sample(self, batch_size: int) -> dict:
        total = self.tree.total
        # stratified: one sample from each of batch_size equal slices of the total mass
        bounds = np.linspace(0, total, batch_size + 1)
        masses = self.rng.uniform(bounds[:-1], bounds[1:])
        idx = np.array([min(self.tree.find(m), self.size - 1) for m in masses])
        probs = np.array([self.tree[i] for i in idx]) / total
        weights = (self.size * probs) ** (-self.beta)
        batch = self._batch(idx)
        batch["weights"] = (weights / weights.max()).astype(np.float32)
        return batch

    def update_priorities(self, idx, td_errors) -> None:
        prios = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, float(prios.max()))
        for i, p in zip(idx, prios):
            self.tree.update(int(i), float(p) ** self.alpha)


class NStepAccumulator:
    """Turns 1-step transitions into n-step ones: (s_t, a_t, r_t + g r_t+1 + ... + g^(n-1) r_t+n-1, s_t+n).

    The reward of reaching the flag propagates n states back in a single update instead of one.
    """

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.queue: deque = deque()

    def push(self, obs, action, reward, next_obs, terminated, truncated) -> list[tuple]:
        """Add one step; returns the n-step transitions now complete (several at an episode end)."""
        self.queue.append((obs, action, reward))
        out = []
        if len(self.queue) == self.n:
            out.append(self._emit(next_obs, terminated))
            self.queue.popleft()
        if terminated or truncated:
            while self.queue:
                out.append(self._emit(next_obs, terminated))
                self.queue.popleft()
        return out

    def _emit(self, next_obs, terminated) -> tuple:
        ret, g = 0.0, 1.0
        for _, _, r in self.queue:
            ret += g * r
            g *= self.gamma
        obs, action, _ = self.queue[0]
        return obs, action, ret, next_obs, float(terminated), g
