"""Greedy evaluation, identical for every method: success rate, mean return, mean steps.

The agent acts with `explore=False` (no epsilon, no noise, the mean action for stochastic
policies) on `episodes` fresh episodes whose start states come from fixed seeds (1000, 1001, ...),
so every method is tested on exactly the same start positions. Returns are the true environment
returns: shaping and intrinsic bonuses are never used here.

CLI:  mountaincar-eval checkpoints/q-learning/agent.pt --episodes 100
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mountaincar_rl.envs import make_env

EVAL_SEED = 1000


def run_episode(agent, env, seed: int | None = None, explore: bool = False, frames: list | None = None):
    """One episode; returns (true return, steps, reached the flag). Appends rgb frames if asked."""
    obs, _ = env.reset(seed=seed)
    agent.start_episode()
    total, steps, done, success = 0.0, 0, False, False
    while not done:
        if frames is not None:
            frames.append(env.render())
        action = agent.act(obs, explore=explore)
        obs, reward, terminated, truncated, info = env.step(action)
        total += info.get("env_reward", reward)
        steps += 1
        success = bool(terminated)
        done = terminated or truncated
    return total, steps, success


def evaluate(agent, continuous: bool = False, episodes: int = 20, seed: int = EVAL_SEED) -> dict:
    env = make_env(continuous=continuous, seed=seed)
    results = [run_episode(agent, env, seed=seed + i) for i in range(episodes)]
    env.close()
    rets, steps, succ = (np.array(x, dtype=np.float64) for x in zip(*results))
    return {
        "success_rate": float(succ.mean()),
        "mean_return": float(rets.mean()),
        "std_return": float(rets.std()),
        "mean_steps": float(steps.mean()),
        "episodes": episodes,
    }


def main(argv=None) -> None:
    from mountaincar_rl.train import load_agent

    p = argparse.ArgumentParser(description="Evaluate a trained MountainCar agent (greedy policy).")
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    args = p.parse_args(argv)
    agent, cfg = load_agent(args.checkpoint)
    result = evaluate(agent, cfg.continuous, args.episodes, args.seed)
    print(json.dumps({"run": cfg.run_name, **result}, indent=2))


if __name__ == "__main__":
    main()
