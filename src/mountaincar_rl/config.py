"""Run configuration: one dataclass shared by every algorithm, the CLIs and the notebook.

Every field has a small default that trains on a CPU in seconds to minutes. Algorithm-specific
fields are simply ignored by the algorithms that do not use them, so one `TrainConfig` describes
any run and is saved inside every checkpoint.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

DISCRETE_ENV = "MountainCar-v0"
CONTINUOUS_ENV = "MountainCarContinuous-v0"

# algorithm name -> which action spaces it supports
ALGORITHMS: dict[str, tuple[str, ...]] = {
    "q-learning": ("discrete",),
    "sarsa": ("discrete",),
    "tile-coding": ("discrete",),
    "dqn": ("discrete",),
    "double-dqn": ("discrete",),
    "dueling-dqn": ("discrete",),
    "reinforce": ("discrete", "continuous"),
    "a2c": ("discrete", "continuous"),
    "ppo": ("discrete", "continuous"),
    "ddpg": ("continuous",),
    "td3": ("continuous",),
    "sac": ("continuous",),
    "cem": ("discrete", "continuous"),
}


@dataclass
class TrainConfig:
    # what to train
    algo: str = "q-learning"
    continuous: bool = False          # False: MountainCar-v0 (3 actions), True: MountainCarContinuous-v0
    episodes: int = 500
    max_steps: int = 0                # 0 = the environment's limit (200 discrete, 999 continuous)
    seed: int = 0
    gamma: float = 0.99
    lr: float = 1e-3                  # neural nets (tabular / tile coding use `alpha`)

    # exploration aids (all off by default: the plain sparse-reward problem)
    shaping: str = "none"             # none | energy | height   (potential-based REWARD SHAPING)
    shaping_scale: float = 10.0
    intrinsic: str = "none"           # none | count | rnd       (intrinsic novelty bonus)
    intrinsic_scale: float = 0.1
    eps_schedule: str = "linear"      # const | linear | exp      (epsilon-greedy methods)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.5            # fraction of training over which epsilon decays

    # tabular and tile coding
    alpha: float = 0.1                # step size (tile coding divides it by the number of tilings)
    bins: int = 20                    # grid cells per dimension for the tabular methods
    tilings: int = 8
    tiles: int = 8                    # tiles per dimension in each tiling
    td_target: str = "sarsa"          # tile-coding only: sarsa | q
    q_init: float = 0.0               # 0 is optimistic here (all real returns are negative)

    # neural networks
    hidden: int = 64
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup_steps: int = 1_000
    target_update: int = 500          # DQN: hard target copy period (steps)
    tau: float = 0.005                # DDPG / TD3 / SAC: soft target update
    prioritized: bool = False         # DQN family: prioritized experience replay
    n_step: int = 1                   # DQN family: n-step returns

    # policy gradient
    rollout_steps: int = 2048         # PPO rollout length (A2C uses `a2c_steps`)
    a2c_steps: int = 16
    ppo_epochs: int = 10
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01

    # continuous control
    action_noise: float = 0.5         # DDPG / TD3 Gaussian exploration noise (std, actions in [-1, 1])
    policy_delay: int = 2             # TD3

    # cross-entropy method
    cem_population: int = 20
    cem_elite: float = 0.2

    # bookkeeping
    eval_episodes: int = 20
    threads: int = 1
    output_dir: str = "checkpoints/run"
    log_every: int = 50

    @property
    def env_id(self) -> str:
        return CONTINUOUS_ENV if self.continuous else DISCRETE_ENV

    @property
    def run_name(self) -> str:
        parts = [self.algo, "cont" if self.continuous else "disc"]
        if self.shaping != "none":
            parts.append(f"shape-{self.shaping}")
        if self.intrinsic != "none":
            parts.append(self.intrinsic)
        if self.prioritized:
            parts.append("per")
        if self.n_step > 1:
            parts.append(f"n{self.n_step}")
        return "_".join(parts)

    def validate(self) -> TrainConfig:
        if self.algo not in ALGORITHMS:
            raise ValueError(f"unknown algo {self.algo!r}; choose from {sorted(ALGORITHMS)}")
        space = "continuous" if self.continuous else "discrete"
        if space not in ALGORITHMS[self.algo]:
            raise ValueError(f"{self.algo} does not support the {space} MountainCar")
        if self.shaping not in ("none", "energy", "height"):
            raise ValueError(f"unknown shaping {self.shaping!r}")
        if self.intrinsic not in ("none", "count", "rnd"):
            raise ValueError(f"unknown intrinsic bonus {self.intrinsic!r}")
        if self.eps_schedule not in ("const", "linear", "exp"):
            raise ValueError(f"unknown epsilon schedule {self.eps_schedule!r}")
        if self.td_target not in ("sarsa", "q"):
            raise ValueError("td_target must be 'sarsa' or 'q'")
        if self.episodes < 1:
            raise ValueError("episodes must be >= 1")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Recommended small settings per method: what `mountaincar-train --algo X` uses unless a flag
# overrides it. They are chosen to train in seconds to a few minutes on one CPU thread.
RECOMMENDED: dict[str, dict] = {
    "q-learning": dict(episodes=3000, alpha=0.5, bins=16, eps_schedule="const", eps_start=0.0),
    "sarsa": dict(episodes=3000, alpha=0.5, bins=16, eps_schedule="const", eps_start=0.0),
    "tile-coding": dict(episodes=300, alpha=0.5, eps_schedule="const", eps_start=0.0),
    "dqn": dict(episodes=300, lr=1e-3),
    "double-dqn": dict(episodes=300, lr=1e-3),
    "dueling-dqn": dict(episodes=300, lr=1e-3),
    "reinforce": dict(episodes=500, lr=1e-3, gamma=0.9),   # see docs/architecture.md: MC returns + shaping
    "a2c": dict(episodes=500, lr=7e-4),
    "ppo": dict(episodes=500, lr=3e-4),
    "cem": dict(episodes=300),
}
RECOMMENDED_CONTINUOUS: dict[str, dict] = {
    "reinforce": dict(episodes=60, lr=1e-3),
    "a2c": dict(episodes=60, lr=7e-4),
    "ppo": dict(episodes=60, lr=3e-4),
    "ddpg": dict(episodes=30, lr=1e-3, warmup_steps=5000),
    "td3": dict(episodes=30, lr=1e-3, warmup_steps=5000),
    "sac": dict(episodes=30, lr=1e-3, warmup_steps=5000),
    "cem": dict(episodes=100),
}


def recommended_config(algo: str, continuous: bool = False, **overrides) -> TrainConfig:
    """TrainConfig with the method's recommended settings, then any explicit overrides."""
    table = RECOMMENDED_CONTINUOUS if continuous else RECOMMENDED
    params = {"algo": algo, "continuous": continuous, **table.get(algo, {}), **overrides}
    return TrainConfig(**params).validate()
