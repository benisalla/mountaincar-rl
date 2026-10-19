"""01: the problem. Look at MountainCar before learning anything.

* the track, the start region and the flag;
* what a random policy gets (it almost never reaches the flag: that is the sparse reward);
* the hand-written "push in the direction you move" policy, which solves it: the task is easy
  to *act* in, hard to *discover*;
* the energy potential used by reward shaping, drawn over (position, velocity);
* one headless rgb frame (pygame with SDL_VIDEODRIVER=dummy).

Run from the repo root:  python examples/01_explore_the_environment.py   (a few seconds)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mountaincar_rl.envs import energy, make_env, state_grid
from mountaincar_rl.viz import EXTENT, plot_track

OUT = Path("examples/output")
OUT.mkdir(parents=True, exist_ok=True)


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, done, steps = 0.0, False, 0
    while not done:
        obs, r, term, trunc, _ = env.step(policy(obs))
        total, steps, done = total + r, steps + 1, term or trunc
    return total, steps, term


env = make_env()
print("observation space:", env.observation_space, "| actions:", env.action_space, "(0 left, 1 none, 2 right)")
rng = np.random.default_rng(0)
random_results = [rollout(env, lambda s: int(rng.integers(3)), seed=i) for i in range(20)]
print(f"random policy:     success {np.mean([r[2] for r in random_results]):.0%}, "
      f"mean return {np.mean([r[0] for r in random_results]):.1f}")
pump = [rollout(env, lambda s: 2 if s[1] >= 0 else 0, seed=i) for i in range(20)]
print(f"'push with the velocity' policy: success {np.mean([r[2] for r in pump]):.0%}, "
      f"mean return {np.mean([r[0] for r in pump]):.1f}")

plot_track(OUT / "track.png")
pos, vel, states = state_grid(80)
fig, ax = plt.subplots(figsize=(5.5, 4))
im = ax.imshow(energy(states).reshape(80, 80).T, origin="lower", extent=EXTENT, aspect="auto", cmap="viridis")
fig.colorbar(im, label="energy (1 = standing at the flag)")
ax.set(xlabel="position", ylabel="velocity", title="shaping potential Phi(s) = energy")
fig.savefig(OUT / "energy_potential.png", dpi=110, bbox_inches="tight")

env = make_env(render_mode="rgb_array")
env.reset(seed=0)
frame = env.render()
plt.imsave(OUT / "frame.png", frame)
print("rgb frame", frame.shape, "-> figures in", OUT)
