"""Watch a trained agent: run greedy episodes, print the outcome, optionally save a GIF.

Rendering uses pygame's rgb_array mode, which works headless (SDL_VIDEODRIVER=dummy is set in
envs.py), so GIFs can be written on a server or in Kaggle/Colab.

CLI:  mountaincar-play checkpoints/tile-coding_disc --gif assets/tile_coding.gif
"""

from __future__ import annotations

import argparse

from mountaincar_rl.envs import make_env
from mountaincar_rl.evaluate import EVAL_SEED, run_episode
from mountaincar_rl.viz import save_gif


def play(agent, continuous: bool, episodes: int = 1, seed: int = EVAL_SEED, gif: str | None = None):
    env = make_env(continuous=continuous, seed=seed, render_mode="rgb_array" if gif else None)
    frames: list | None = [] if gif else None
    results = []
    for i in range(episodes):
        ret, steps, success = run_episode(agent, env, seed=seed + i, frames=frames)
        results.append((ret, steps, success))
        print(f"episode {i + 1}: return {ret:.1f}, steps {steps}, {'reached the flag' if success else 'timed out'}")
    if gif and frames:
        frames.append(env.render())
        print("gif saved to", save_gif(frames, gif))
    env.close()
    return results


def main(argv=None) -> None:
    from mountaincar_rl.train import load_agent

    p = argparse.ArgumentParser(description="Run a trained MountainCar agent greedily (and save a GIF).")
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--gif", default=None, help="path of a GIF to write")
    args = p.parse_args(argv)
    agent, cfg = load_agent(args.checkpoint)
    play(agent, cfg.continuous, args.episodes, args.seed, args.gif)


if __name__ == "__main__":
    main()
