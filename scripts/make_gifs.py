"""Regenerate the README animations: before/after GIFs, a grid of solved methods, a policy-map movie.

    python scripts/make_gifs.py                 # needs the runs of `mountaincar-compare --suite all`
    python scripts/make_gifs.py --only policy   # just the policy-map animation (trains tile coding, seconds)

Writes to assets/:
  before_after_discrete.gif    random agent | trained tile-coding agent, same start state (seed 1000)
  before_after_continuous.gif  random agent | trained SAC (+shaping), same start state
  solved_grid_discrete.gif     one panel per method that solved MountainCar-v0
  solved_grid_continuous.gif   one panel per method that solved MountainCarContinuous-v0
  policy_map_training.gif      greedy policy of tile coding over (position, velocity) while it learns
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from mountaincar_rl.envs import make_env  # noqa: E402
from mountaincar_rl.evaluate import EVAL_SEED  # noqa: E402

RUNS, ASSETS = Path("runs"), Path("assets")
PANEL = (240, 160)

# (label, run folder) for the grids: the best-performing variant of every method that solved it
DISCRETE = [("q-learning", "tabular/q-learning"), ("sarsa", "tabular/sarsa"),
            ("tile coding", "tabular/tile-coding_q"), ("CEM", "tabular/cem"),
            ("DQN +shaping", "discrete-deep/dqn_shaping"), ("DQN +count", "exploration/dqn_count"),
            ("double DQN", "discrete-deep/double-dqn"), ("dueling +shaping", "discrete-deep/dueling-dqn_shaping"),
            ("REINFORCE +shaping", "discrete-deep/reinforce_shaping"), ("A2C +shaping", "discrete-deep/a2c_shaping"),
            ("PPO", "discrete-deep/ppo"), ("DQN +RND", "exploration/dqn_rnd")]
CONTINUOUS = [("SAC +shaping", "continuous/sac_shaping"), ("TD3 +shaping", "continuous/td3_shaping"),
              ("DDPG +shaping", "continuous/ddpg_shaping"), ("PPO +shaping", "continuous/ppo_shaping"),
              ("CEM", "continuous/cem")]


class RandomAgent:
    continuous = False

    def __init__(self, continuous: bool, seed: int = 0):
        self.continuous, self.rng = continuous, np.random.default_rng(seed)

    def start_episode(self):
        pass

    def act(self, obs, explore=False):
        if self.continuous:
            return self.rng.uniform(-1, 1, size=1).astype(np.float32)
        return int(self.rng.integers(3))


def record(agent, continuous: bool, seed: int = EVAL_SEED):
    """Frames of one greedy episode + (steps, reached the flag)."""
    env = make_env(continuous=continuous, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    agent.start_episode()
    frames, done, steps, success = [env.render()], False, 0, False
    while not done:
        obs, _, term, trunc, _ = env.step(agent.act(obs, explore=False))
        frames.append(env.render())
        steps, done, success = steps + 1, term or trunc, bool(term)
    env.close()
    return frames, steps, success


def panel(frame, title: str, color=(0, 0, 0)) -> Image.Image:
    img = Image.fromarray(frame).resize(PANEL)
    canvas = Image.new("RGB", (PANEL[0], PANEL[1] + 22), "white")
    canvas.paste(img, (0, 22))
    ImageDraw.Draw(canvas).text((6, 5), title, fill=color)
    return canvas


def compose(episodes, labels, cols: int, every: int) -> list[Image.Image]:
    """Synchronise episodes by step; a finished episode holds its last frame."""
    longest = max(len(f) for f, _, _ in episodes)
    rows = int(np.ceil(len(episodes) / cols))
    out = []
    for t in list(range(0, longest, every)) + [longest - 1] * 8:   # hold the end for a moment
        sheet = Image.new("RGB", (cols * PANEL[0], rows * (PANEL[1] + 22)), "white")
        for k, ((frames, steps, success), label) in enumerate(zip(episodes, labels)):
            i = min(t, len(frames) - 1)
            if i == len(frames) - 1:
                status = f"flag at step {steps}" if success else f"time out ({steps} steps)"
                color = (0, 120, 60) if success else (190, 30, 45)
            else:
                status, color = f"step {i}", (0, 0, 0)
            sheet.paste(panel(frames[i], f"{label}: {status}", color),
                        ((k % cols) * PANEL[0], (k // cols) * (PANEL[1] + 22)))
        out.append(sheet)
    return out


def save(images: list[Image.Image], path: Path, fps: int = 20) -> None:
    images = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in images]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"{path}: {len(images)} frames, {path.stat().st_size / 1e6:.2f} MB")


def load(run: str):
    from mountaincar_rl.train import load_agent

    agent, _ = load_agent(RUNS / run)
    return agent


def solved(run: str) -> bool:
    f = RUNS / run / "eval.json"
    return f.exists() and json.loads(f.read_text())["success_rate"] > 0


def before_after() -> None:
    for continuous, run, label, every in [(False, "tabular/tile-coding_q", "trained (tile coding)", 2),
                                          (True, "continuous/sac_shaping", "trained (SAC)", 6)]:
        episodes = [record(RandomAgent(continuous), continuous), record(load(run), continuous)]
        name = "continuous" if continuous else "discrete"
        save(compose(episodes, ["random", label], cols=2, every=every), ASSETS / f"before_after_{name}.gif")


def grids() -> None:
    for continuous, table, every, cols in [(False, DISCRETE, 3, 4), (True, CONTINUOUS, 6, 5)]:
        rows = [(label, run) for label, run in table if solved(run)]
        episodes = [record(load(run), continuous) for _, run in rows]
        name = "continuous" if continuous else "discrete"
        save(compose(episodes, [label for label, _ in rows], cols=cols, every=every),
             ASSETS / f"solved_grid_{name}.gif", fps=15)


def policy_animation(episodes: int = 200, every: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mountaincar_rl.algorithms import make_agent
    from mountaincar_rl.config import recommended_config
    from mountaincar_rl.envs import GOAL_POS, state_grid
    from mountaincar_rl.viz import ACTION_CMAP, EXTENT

    cfg = recommended_config("tile-coding", episodes=episodes)
    agent, env = make_agent(cfg), make_env(seed=cfg.seed)
    _, _, states = state_grid(70)
    images, successes = [], 0
    for ep in range(episodes + 1):
        if ep % every == 0 or ep == episodes:
            fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=80)
            ax.imshow(agent.policy_grid(states).reshape(70, 70).T, origin="lower", extent=EXTENT, aspect="auto",
                      cmap=ACTION_CMAP, vmin=-0.5, vmax=2.5)
            ax.axvline(GOAL_POS, color="k", ls="--", lw=1)
            ax.set(xlabel="position", ylabel="velocity",
                   title=f"tile coding, episode {ep}: {successes} flags so far\n"
                         "red = left, yellow = none, blue = right")
            ax.title.set_fontsize(9)
            fig.tight_layout()
            fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
            plt.close(fig)
        if ep == episodes:
            break
        agent.progress = ep / episodes
        agent.start_episode()
        obs, _ = env.reset(seed=cfg.seed * 100_000 + ep)
        done = False
        while not done:
            a = agent.act(obs)
            nxt, r, term, trunc, _ = env.step(a)
            agent.observe(obs, a, r, nxt, term, trunc)
            obs, done = nxt, term or trunc
        successes += int(term)
    save(images + [images[-1]] * 10, ASSETS / "policy_map_training.gif", fps=6)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", choices=["before-after", "grids", "policy"], default=None)
    args = p.parse_args(argv)
    ASSETS.mkdir(exist_ok=True)
    if args.only in (None, "before-after"):
        before_after()
    if args.only in (None, "grids"):
        grids()
    if args.only in (None, "policy"):
        policy_animation()


if __name__ == "__main__":
    sys.exit(main())
