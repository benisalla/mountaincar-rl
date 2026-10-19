"""Cross-entropy method: black-box search over the weights of a tiny linear policy.

The policy is linear in the features phi(s) = (position, velocity, 1) (normalised):
discrete: a = argmax(W phi(s)) with W of shape 3 x 3; continuous: a = tanh(w . phi(s)).
CEM keeps a Gaussian N(mean, sigma^2) over the weights and repeats:

    1. sample `cem_population` weight vectors, run ONE episode with each (an episode is one
       training episode, so the budget is directly comparable with the other methods)
    2. keep the best `cem_elite` fraction by return
    3. mean, sigma <- mean and std of the elites (+ a little extra noise that decays)

No gradients, no value function, no TD. It works on MountainCar because a very simple policy,
"push in the direction you are already moving", solves it, and a random linear policy has a fair
chance of being close to that. It is the baseline any RL method should beat.
"""

from __future__ import annotations

import numpy as np

from mountaincar_rl.algorithms.base import Agent
from mountaincar_rl.envs import normalize_obs


def features(obs) -> np.ndarray:
    x = normalize_obs(obs)
    return np.array([x[0], x[1], 1.0])


class CEMAgent(Agent):
    name = "cem"

    def __init__(self, cfg, n_actions: int = 3):
        super().__init__(cfg, n_actions)
        self.continuous = cfg.continuous
        self.dim = 3 if cfg.continuous else 3 * n_actions
        self.mean = np.zeros(self.dim)
        self.sigma = np.ones(self.dim)
        self.n_elite = max(2, int(round(cfg.cem_population * cfg.cem_elite)))
        self._new_population()
        self._episode_return = 0.0

    def _new_population(self) -> None:
        self.population = self.mean + self.sigma * self.rng.standard_normal((self.cfg.cem_population, self.dim))
        self.fitness: list[float] = []

    def _action(self, weights, obs):
        phi = features(obs)
        if self.continuous:
            return np.array([np.tanh(weights @ phi)], dtype=np.float32)
        return int(np.argmax(weights.reshape(self.n_actions, 3) @ phi))

    def act(self, obs, explore: bool = True):
        weights = self.population[len(self.fitness)] if explore else self.mean
        return self._action(weights, obs)

    def start_episode(self) -> None:
        self._episode_return = 0.0

    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        self._episode_return += reward

    def end_episode(self) -> None:
        self.fitness.append(self._episode_return)
        if len(self.fitness) == self.cfg.cem_population:
            elite = self.population[np.argsort(self.fitness)[-self.n_elite:]]
            extra = max(0.0, 0.5 * (1.0 - 2.0 * self.progress))   # decaying extra noise, avoids early collapse
            self.mean, self.sigma = elite.mean(axis=0), elite.std(axis=0) + extra
            self._new_population()

    def policy_grid(self, states):
        return np.array([self._action(self.mean, s) for s in states]).reshape(len(states))

    def state_dict(self):
        return {"mean": self.mean, "sigma": self.sigma}

    def load_state_dict(self, state):
        self.mean, self.sigma = np.asarray(state["mean"]), np.asarray(state["sigma"])
