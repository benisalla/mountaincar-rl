"""The single training loop shared by every algorithm, checkpoints, and the `mountaincar-train` CLI.

    for each episode:
        obs = env.reset(seed)
        until terminated (flag) or truncated (time limit):
            a = agent.act(obs)                    # exploration is the agent's business
            obs', r = env.step(a)                 # r may include shaping / intrinsic bonus
            agent.observe(obs, a, r, obs', ...)   # learning happens here (or is buffered)
        agent.end_episode()

The loop also records the TRUE return, the steps, success, and a visitation histogram over
(position, velocity) for the heatmaps. One loop for every method is what makes the comparison fair.

CLI:  mountaincar-train --algo tile-coding --episodes 300
      mountaincar-train --algo dqn --episodes 400 --shaping energy
      mountaincar-train --algo sac --continuous --episodes 40
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from mountaincar_rl.algorithms import make_agent
from mountaincar_rl.config import ALGORITHMS, TrainConfig, recommended_config
from mountaincar_rl.envs import discretize, make_env_from_config
from mountaincar_rl.evaluate import evaluate
from mountaincar_rl.metrics import History

VISIT_BINS = 50


def train(cfg: TrainConfig, verbose: bool = True, save: bool = True):
    """Train `cfg.algo`; returns (agent, history, visits[VISIT_BINS, VISIT_BINS], evaluation dict)."""
    cfg.validate()
    torch.set_num_threads(cfg.threads)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = make_env_from_config(cfg)
    agent = make_agent(cfg)
    history = History()
    visits = np.zeros((VISIT_BINS, VISIT_BINS), dtype=np.int64)
    start = time.time()
    for ep in range(cfg.episodes):
        agent.progress = ep / cfg.episodes
        agent.start_episode()
        obs, _ = env.reset(seed=cfg.seed * 100_000 + ep)
        done, ret, learner_ret, steps, success = False, 0.0, 0.0, 0, False
        while not done:
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.observe(obs, action, reward, next_obs, terminated, truncated)
            visits[discretize(obs, VISIT_BINS)] += 1
            ret += info.get("env_reward", reward)
            learner_ret += reward
            steps += 1
            obs, done, success = next_obs, terminated or truncated, bool(terminated)
        agent.end_episode()
        history.add(ret, steps, success, learner_ret)
        if verbose and ((ep + 1) % cfg.log_every == 0 or ep == cfg.episodes - 1):
            print(f"[{cfg.run_name}] episode {ep + 1:5d} | return {ret:8.1f} | steps {steps:4d} | "
                  f"success(last 50) {history.success_rate(50):5.0%} | first success "
                  f"{history.first_success} | {time.time() - start:6.1f} s", flush=True)
    history.wall_time = time.time() - start
    env.close()
    evaluation = evaluate(agent, cfg.continuous, cfg.eval_episodes)
    if verbose:
        print(f"[{cfg.run_name}] greedy eval: success {evaluation['success_rate']:.0%} | "
              f"return {evaluation['mean_return']:.1f} | steps {evaluation['mean_steps']:.0f}", flush=True)
    if save:
        save_run(cfg, agent, history, visits, evaluation)
    return agent, history, visits, evaluation


def save_run(cfg, agent, history, visits, evaluation) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "agent": agent.state_dict()}, out / "agent.pt")
    history.save(out / "history.json")
    np.save(out / "visits.npy", visits)
    (out / "eval.json").write_text(json.dumps(evaluation, indent=2))
    return out


def load_agent(path: str | Path):
    """Checkpoint (agent.pt or its folder) -> (agent, config)."""
    path = Path(path)
    if path.is_dir():
        path = path / "agent.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(ckpt["config"])
    agent = make_agent(cfg)
    agent.load_state_dict(ckpt["agent"])
    return agent, cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one RL method on MountainCar. Unset flags take the "
                                "method's recommended value (config.RECOMMENDED), then the TrainConfig default.")
    defaults = TrainConfig()
    for f in dataclasses.fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        # default=SUPPRESS: only flags given on the command line override the recommended settings
        if isinstance(default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        elif f.name == "algo":
            p.add_argument(flag, choices=sorted(ALGORITHMS), default="tile-coding")
        else:
            p.add_argument(flag, type=type(default), default=argparse.SUPPRESS, help=f"(base default {default})")
    return p


def main(argv=None) -> None:
    args = vars(build_parser().parse_args(argv))
    cfg = recommended_config(args.pop("algo"), args.pop("continuous", False), **args)
    if cfg.output_dir == TrainConfig.output_dir:
        cfg.output_dir = f"checkpoints/{cfg.run_name}"
    if cfg.threads == 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    train(cfg)
    print(f"saved to {cfg.output_dir}/")


if __name__ == "__main__":
    main()
