"""Every algorithm behind one factory: `make_agent(cfg)`."""

from __future__ import annotations

from mountaincar_rl.algorithms.base import Agent, epsilon
from mountaincar_rl.algorithms.tabular import TabularAgent
from mountaincar_rl.algorithms.tile_coding import TileCoder, TileCodingAgent


def make_agent(cfg) -> Agent:
    cfg.validate()
    algo = cfg.algo
    if algo in ("q-learning", "sarsa"):
        return TabularAgent(cfg, 3, method=algo)
    if algo == "tile-coding":
        return TileCodingAgent(cfg, 3)
    if algo in ("dqn", "double-dqn", "dueling-dqn"):
        from mountaincar_rl.algorithms.dqn import DQNAgent
        return DQNAgent(cfg)
    if algo in ("reinforce", "a2c", "ppo"):
        from mountaincar_rl.algorithms.policy_gradient import PolicyGradientAgent
        return PolicyGradientAgent(cfg)
    if algo in ("ddpg", "td3", "sac"):
        from mountaincar_rl.algorithms.actor_critic import OffPolicyActorCritic
        return OffPolicyActorCritic(cfg)
    if algo == "cem":
        from mountaincar_rl.algorithms.cem import CEMAgent
        return CEMAgent(cfg)
    raise ValueError(algo)


__all__ = ["Agent", "TabularAgent", "TileCoder", "TileCodingAgent", "epsilon", "make_agent"]
