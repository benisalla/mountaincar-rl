"""The two MountainCar environments, reward shaping, intrinsic bonuses and state helpers.

MountainCar is the textbook sparse-reward problem: the reward is -1 every step (or, in the
continuous version, -0.1 a^2 per step and +100 at the flag), and the engine is too weak to drive
straight up the hill. The car must rock back and forth to build energy, and nothing in the reward
says so until the flag is reached once. Exploration aids live here as wrappers, so the learning
algorithms never need to know about them:

* `ShapingWrapper`: potential-based reward shaping F = gamma * Phi(s') - Phi(s), with Phi the
  car's mechanical energy or its height. Potential-based shaping keeps the optimal policy
  unchanged (Ng et al. 1999), but it IS extra knowledge given to the agent: results that use it
  are marked "shaped" everywhere.
* `IntrinsicWrapper`: a novelty bonus, either count based (1 / sqrt(N(cell))) or RND (the error
  of a small network trained to predict a fixed random network's output).

Every wrapper keeps the true environment reward in `info["env_reward"]`, which is what is logged
and evaluated.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch
from torch import nn

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless rgb_array rendering

POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07
GOAL_POS = 0.5
OBS_LOW = np.array([POS_MIN, VEL_MIN], dtype=np.float32)
OBS_HIGH = np.array([POS_MAX, VEL_MAX], dtype=np.float32)
GRAVITY = 0.0025


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """(position, velocity) -> [-1, 1]^2, the input of every neural network."""
    return (2.0 * (np.asarray(obs, dtype=np.float32) - OBS_LOW) / (OBS_HIGH - OBS_LOW) - 1.0).astype(np.float32)


def height(pos):
    """Height of the track: the hill is y = sin(3x)."""
    return np.sin(3.0 * np.asarray(pos))


def energy(obs: np.ndarray) -> np.ndarray:
    """Mechanical energy of the car, scaled so that standing at the flag = 1 and the valley floor ~ -1.

    The dynamics are v' = v + force * a - 0.0025 cos(3x), so the potential energy is
    U(x) = 0.0025 / 3 * sin(3x) and the kinetic energy is v^2 / 2.
    """
    obs = np.asarray(obs, dtype=np.float64)
    pos, vel = obs[..., 0], obs[..., 1]
    potential = GRAVITY / 3.0 * np.sin(3.0 * pos)
    kinetic = 0.5 * vel**2
    return (potential + kinetic) / (GRAVITY / 3.0 * np.sin(3.0 * GOAL_POS))


class ShapingWrapper(gym.Wrapper):
    """Potential-based reward shaping: r + gamma * Phi(s') - Phi(s) (Phi(terminal) = 0)."""

    def __init__(self, env: gym.Env, kind: str = "energy", scale: float = 10.0, gamma: float = 0.99):
        super().__init__(env)
        if kind not in ("energy", "height"):
            raise ValueError(kind)
        self.kind, self.scale, self.gamma = kind, scale, gamma
        self._phi = 0.0

    def potential(self, obs: np.ndarray) -> float:
        value = energy(obs) if self.kind == "energy" else height(np.asarray(obs)[0])
        return float(self.scale * value)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.setdefault("env_reward", float(reward))
        phi_next = 0.0 if terminated else self.potential(obs)
        shaped = reward + self.gamma * phi_next - self._phi
        self._phi = phi_next
        return obs, float(shaped), terminated, truncated, info


class CountBonus:
    """Count-based novelty: bonus = 1 / sqrt(N(cell)) on a fine (position, velocity) grid."""

    def __init__(self, bins: int = 40):
        self.bins = bins
        self.counts = np.zeros((bins, bins), dtype=np.int64)

    def __call__(self, obs: np.ndarray) -> float:
        i, j = discretize(obs, self.bins)
        self.counts[i, j] += 1
        return float(1.0 / np.sqrt(self.counts[i, j]))


class RND(nn.Module):
    """Random Network Distillation: novelty = prediction error of a trained net vs a frozen random net.

    States seen often are predicted well (small bonus); new states are not (large bonus). The
    predictor is trained online, one small Adam step per `batch` states.
    """

    def __init__(self, hidden: int = 64, out: int = 16, lr: float = 1e-3, batch: int = 32, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed + 12345)

        def mlp():
            return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, out))

        self.target, self.predictor = mlp(), mlp()
        with torch.no_grad():
            for p in self.target.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 1.0)
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.batch, self._buffer = batch, []
        self.running = 1.0  # running scale of the error, keeps the bonus around 1

    def __call__(self, obs: np.ndarray) -> float:
        x = torch.as_tensor(normalize_obs(obs)).unsqueeze(0)
        with torch.no_grad():
            err = float(((self.predictor(x) - self.target(x)) ** 2).mean())
        self.running = 0.99 * self.running + 0.01 * err
        self._buffer.append(x)
        if len(self._buffer) >= self.batch:
            xb = torch.cat(self._buffer)
            self._buffer = []
            loss = ((self.predictor(xb) - self.target(xb)) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return err / (self.running + 1e-8)


class IntrinsicWrapper(gym.Wrapper):
    """Adds `scale * bonus(s')` to the reward the agent learns from (never to the logged return)."""

    def __init__(self, env: gym.Env, kind: str = "count", scale: float = 0.1, seed: int = 0):
        super().__init__(env)
        self.bonus = CountBonus() if kind == "count" else RND(seed=seed)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.setdefault("env_reward", float(reward))
        return obs, float(reward + self.scale * self.bonus(obs)), terminated, truncated, info


def make_env(
    continuous: bool = False,
    shaping: str = "none",
    shaping_scale: float = 10.0,
    intrinsic: str = "none",
    intrinsic_scale: float = 0.1,
    gamma: float = 0.99,
    seed: int = 0,
    render_mode: str | None = None,
    max_steps: int = 0,
) -> gym.Env:
    """MountainCar-v0 (3 actions: left, none, right) or MountainCarContinuous-v0 (force in [-1, 1])."""
    kwargs = {"max_episode_steps": max_steps} if max_steps else {}
    env = gym.make("MountainCarContinuous-v0" if continuous else "MountainCar-v0", render_mode=render_mode, **kwargs)
    if shaping != "none":
        env = ShapingWrapper(env, shaping, shaping_scale, gamma)
    if intrinsic != "none":
        env = IntrinsicWrapper(env, intrinsic, intrinsic_scale, seed)
    env.action_space.seed(seed)
    return env


def make_env_from_config(cfg, render_mode: str | None = None) -> gym.Env:
    return make_env(cfg.continuous, cfg.shaping, cfg.shaping_scale, cfg.intrinsic, cfg.intrinsic_scale,
                    cfg.gamma, cfg.seed, render_mode, cfg.max_steps)


def discretize(obs, bins: int) -> tuple[int, int]:
    """(position, velocity) -> grid cell (i, j) with `bins` cells per dimension.

    Plain Python on two floats: it runs at every step of every tabular update, and numpy's
    per-call overhead would dominate.
    """
    i = int((float(obs[0]) - POS_MIN) / (POS_MAX - POS_MIN) * bins)
    j = int((float(obs[1]) - VEL_MIN) / (VEL_MAX - VEL_MIN) * bins)
    return min(max(i, 0), bins - 1), min(max(j, 0), bins - 1)


def state_grid(n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A regular n x n grid of states, for heatmaps: returns (positions, velocities, states[n*n, 2])."""
    pos = np.linspace(POS_MIN, POS_MAX, n)
    vel = np.linspace(VEL_MIN, VEL_MAX, n)
    pp, vv = np.meshgrid(pos, vel, indexing="ij")
    return pos, vel, np.stack([pp.ravel(), vv.ravel()], axis=1).astype(np.float32)
