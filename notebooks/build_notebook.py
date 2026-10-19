"""Build notebooks/train_colab_kaggle.ipynb from the package source.

The notebook installs the package from GitHub, imports it, and explains every method step by step.
The code shown in the markdown cells (each algorithm's update) is copied from src/mountaincar_rl at
build time, so it is always the code that actually runs. Regenerate after changing src/:

    python notebooks/build_notebook.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mountaincar_rl"
OUT = ROOT / "notebooks" / "train_colab_kaggle.ipynb"
REPO = "https://github.com/benisalla/mountaincar-rl"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def source_of(module: str, qualname: str) -> str:
    """Source of a function, class or method (`Class.method`) in src/mountaincar_rl/<module>."""
    text = (SRC / module).read_text()
    tree = ast.parse(text)
    parts = qualname.split(".")
    nodes = tree.body
    node = None
    for part in parts:
        node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
        nodes = node.body
    lines = text.splitlines()[node.lineno - 1 - len(node.decorator_list): node.end_lineno]
    indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    return "\n".join(line[indent:] for line in lines)


def show(module: str, *names: str) -> str:
    body = "\n\n".join(source_of(module, n) for n in names)
    return f"```python\n# src/mountaincar_rl/{module}\n{body}\n```"


# ---------------------------------------------------------------------------------------------
md(f"""
# MountainCar with every RL family: a step-by-step notebook (Kaggle / Colab)

This notebook uses the [`mountaincar-rl`]({REPO}) package to train, explain and compare on the
same sparse-reward task:

| family | methods (`algo`) |
|---|---|
| tabular, discretized state | `q-learning`, `sarsa` |
| linear function approximation | `tile-coding` (semi-gradient SARSA or Q-learning) |
| value-based deep RL | `dqn`, `double-dqn`, `dueling-dqn` (+ prioritized replay, n-step) |
| on-policy policy gradient | `reinforce` (with baseline), `a2c`, `ppo` |
| off-policy actor-critic (continuous) | `ddpg`, `td3`, `sac` |
| black-box search | `cem` (cross-entropy method) |

plus the exploration aids: potential-based **reward shaping** (energy or height), **count-based**
and **RND** intrinsic bonuses, and epsilon schedules.

Everything runs on a CPU. A GPU does not help much here (the networks have a few thousand weights
and the environment is the bottleneck), so pick a CPU session on Kaggle or Colab. Set `FULL = True`
in section 8 for the longer, multi-seed comparison.

**Sections:** 1 Install · 2 The problem: sparse reward · 3 Three state representations ·
4 Tabular · 5 Tile coding · 6 DQN family · 7 Policy gradient and CEM · 8 Continuous control ·
9 Exploration aids · 10 Full comparison · 11 Watch the car
""")

md("""
## 1. Install

The repository is private: on Kaggle add a secret `GITHUB_TOKEN` (*Add-ons -> Secrets*), on Colab
put it in the *Secrets* panel, or paste a token below. If the repo is public, no token is needed.
`gymnasium[classic-control]` brings pygame, used headless (`SDL_VIDEODRIVER=dummy`) to draw the car.
""")
code(f"""
import os
import subprocess
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
try:  # Kaggle secrets
    from kaggle_secrets import UserSecretsClient
    TOKEN = TOKEN or UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
try:  # Colab secrets
    from google.colab import userdata
    TOKEN = TOKEN or userdata.get("GITHUB_TOKEN")
except Exception:
    pass
url = "{REPO}".replace("https://", f"https://{{TOKEN}}@" if TOKEN else "https://")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"git+{{url}}.git"])

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image, display

import mountaincar_rl
from mountaincar_rl.config import TrainConfig, recommended_config
from mountaincar_rl.envs import energy, make_env, state_grid
from mountaincar_rl.metrics import markdown_table, summary_row
from mountaincar_rl.train import train
from mountaincar_rl.viz import EXTENT, plot_agent_maps, plot_curves, plot_track

torch.set_num_threads(1)
print("torch", torch.__version__)
""")

md("""
## 2. The problem: a sparse reward and a weak engine

The state is `(position, velocity)`. The track is `y = sin(3x)`; the car starts at rest near the
bottom and the flag is at `x = 0.5`. The engine force (0.001) is smaller than gravity on the slope
(up to 0.0025), so driving right fails: the car has to swing back and forth to build energy.

The reward is **-1 every step** until the flag or 200 steps. Every failed episode returns exactly
-200, whatever the agent did. A method only learns something useful *after* it reached the flag
at least once by itself. That is the exploration problem, and the whole notebook is about how each
method gets (or does not get) that first success.

Two policies below: random actions (fails), and the one-line rule "push in the direction you are
moving" (succeeds). The task is easy to *act* in, hard to *discover*.
""")
code("""
def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(policy(obs))
        total, done = total + r, term or trunc
    return total, term

env = make_env()
rng = np.random.default_rng(0)
rand = [rollout(env, lambda s: int(rng.integers(3)), i) for i in range(20)]
pump = [rollout(env, lambda s: 2 if s[1] >= 0 else 0, i) for i in range(20)]
print("random policy:          success", np.mean([r[1] for r in rand]), "return", np.mean([r[0] for r in rand]))
print("push-with-velocity rule: success", np.mean([r[1] for r in pump]), "return", np.mean([r[0] for r in pump]))
plot_track(None); plt.show()
""")

md("""
## 3. Three ways to represent the state

The state is continuous; a value function or a policy needs a representation of it:

1. **Discretization**: cut the box into a `bins x bins` grid. Simple and exact per cell, but
   *no generalization*: what is learnt in one cell says nothing about its neighbour.
2. **Tile coding**: several grids, each shifted by a fraction of a cell. A state activates one
   tile per grid; `Q(s,a)` is the sum of the active tiles' weights. Neighbouring states share
   most tiles, so learning spreads locally, and the overlaps keep a fine resolution. Linear,
   fast, and with guarantees.
3. **Neural network**: an MLP on normalised `(x, v)`. Generalizes globally and smoothly, but in
   an uncontrolled way, and needs replay buffers and target networks to be stable.

The tile coder in a few lines (`active` returns one feature index per tiling):
""" + "\n\n" + show("algorithms/tile_coding.py", "TileCoder"))
code("""
from mountaincar_rl.algorithms import TileCoder
tc = TileCoder(tilings=8, tiles=8)
a, b, c = tc.active([-0.5, 0.0]), tc.active([-0.49, 0.001]), tc.active([0.4, 0.06])
print("active tiles of s:", a)
print("shared with a close state:", len(set(a) & set(b)), "of 8 | with a far state:", len(set(a) & set(c)))
""")

md("""
## 4. Tabular Q-learning and SARSA

One update per step, on the table entry of the current cell:

- **Q-learning** (off-policy): `Q(s,a) += alpha (r + gamma max_b Q(s',b) - Q(s,a))`
- **SARSA** (on-policy): `Q(s,a) += alpha (r + gamma Q(s',a') - Q(s,a))`, where `a'` is the action
  that will really be taken next (the agent picks it inside the update and stores it).

**Why it can work without any exploration trick:** the table starts at 0 and every true return
is negative. Tried actions drop below untried ones, so even a fully greedy agent (epsilon = 0)
keeps trying new actions in new cells. This is *optimistic initialization*. Its weakness: every
one of the 16 x 16 x 3 entries has to be learnt on its own, so it needs thousands of episodes.
""" + "\n\n" + show("algorithms/tabular.py", "TabularAgent.observe"))
code("""
histories = {}
for algo in ("q-learning", "sarsa"):
    cfg = recommended_config(algo, episodes=3000, log_every=500, output_dir=f"runs/nb/{algo}")
    agent, h, visits, ev = train(cfg)
    histories[algo] = h
    plot_agent_maps(agent, visits, None, title=algo); plt.show()
plot_curves(histories, None, title="tabular"); plt.show()
""")
md("""
Reading the maps: **left** where the car went during training (log counts), **middle** the
learned `max_a Q(s,a)` (higher = closer to the flag in steps), **right** the greedy action. A
solved MountainCar shows a spiral: push right when moving right, left when moving left.
""")

md("""
## 5. Tile coding: semi-gradient SARSA (Sutton & Barto, Example 10.1)

`Q(s,a) = w_a . x(s)` with `x(s)` the binary tile features. The gradient of a linear function
w.r.t. its weights is the feature vector, so the update moves only the 8 active weights:
`w[a, active(s)] += alpha / 8 * (target - Q(s,a))`. It is *semi*-gradient because the target
also depends on `w` and is treated as a constant. Optimistic zeros again do the exploring, and
generalization makes a single success spread to the neighbourhood: about a hundred episodes.
""" + "\n\n" + show("algorithms/tile_coding.py", "TileCodingAgent.observe"))
code("""
cfg = recommended_config("tile-coding", episodes=300, output_dir="runs/nb/tile")
tile_agent, h, visits, ev = train(cfg)
print("first success at episode", h.first_success, "| greedy eval", ev)
plot_agent_maps(tile_agent, visits, None, title="tile coding"); plt.show()
""")

md("""
## 6. DQN, Double DQN, Dueling DQN (+ prioritized replay, n-step)

A neural Q-network trained by regression on a replay buffer:

- **DQN**: `y = r + gamma (1 - done) max_a' Q_target(s', a')`, Huber loss, target net copied every 500 steps;
- **Double DQN**: the online net *chooses* `a'`, the target net *scores* it, which removes the
  max-operator's over-estimation;
- **Dueling**: `Q = V(s) + A(s,a) - mean A`, so `V` learns from every transition;
- **prioritized replay**: replay transitions with large TD error more often (sum tree), corrected
  by importance weights; **n-step**: `y = r_t + ... + gamma^(n-1) r_t+n-1 + gamma^n max Q(s_t+n)`.

`done` is 1 only at the flag; a time-out still bootstraps (the clock stopped, the state is fine).

**The lesson:** without help this usually fails. The network is not optimistic and epsilon-greedy
dithering (random actions each step) almost never pumps enough energy in 200 steps. Until the
flag is in the buffer, every target says "-1 forever". Reward shaping fixes it (section 9).
""" + "\n\n" + show("algorithms/dqn.py", "DQNAgent.update"))
code("""
histories = {}
for shaping in ("none", "energy"):
    cfg = recommended_config("double-dqn", shaping=shaping, episodes=300, output_dir=f"runs/nb/ddqn_{shaping}")
    agent, h, visits, ev = train(cfg)
    histories[f"double-dqn shaping={shaping}"] = h
    plot_agent_maps(agent, visits, None, title=f"double DQN, shaping={shaping}"); plt.show()
plot_curves(histories, None, title="double DQN"); plt.show()
""")

md("""
## 7. Policy gradient (REINFORCE, A2C, PPO) and the cross-entropy method

All three gradient methods push up `log pi(a|s)` in proportion to an advantage estimate:

- **REINFORCE with baseline**: after each episode, `A_t = G_t - V(s_t)` (Monte Carlo return minus a
  learned baseline, which only reduces variance);
- **A2C**: every 16 steps, `A_t` = n-step return bootstrapped with the critic, minus `V(s_t)`;
- **PPO**: every 2048 steps, GAE advantages, then 10 epochs of minibatch updates on the clipped
  ratio objective `min(rho A, clip(rho, 0.8, 1.2) A)`; the clip keeps each update small.

With a raw -1 reward every episode returns -200, the advantages are noise, and nothing is learnt:
they need shaping. **CEM** needs no gradient: it samples 20 linear policies, keeps the best 4,
and refits a Gaussian to them. It works because a linear policy can solve the task.
""" + "\n\n" + show("algorithms/policy_gradient.py", "PolicyGradientAgent._advantages",
                    "PolicyGradientAgent._update_actor_critic")
   + "\n\n" + show("algorithms/cem.py", "CEMAgent.end_episode"))
code("""
rows, histories = [], {}
for algo, extra in [("reinforce", {"shaping": "energy"}), ("a2c", {"shaping": "energy"}),
                    ("ppo", {"shaping": "energy"}), ("ppo", {}), ("cem", {})]:
    name = algo + (" +shaping" if extra else "")
    cfg = recommended_config(algo, output_dir=f"runs/nb/{name.replace(' +', '_')}", **extra)
    agent, h, visits, ev = train(cfg)
    histories[name] = h
    rows.append(summary_row(name, h, ev))
plot_curves(histories, None, title="policy gradient and CEM"); plt.show()
print(markdown_table(rows))
""")

md("""
## 8. Continuous control: DDPG, TD3, SAC, PPO on MountainCarContinuous-v0

The force is a real number in [-1, 1], the reward is `-0.1 a^2` per step and `+100` at the flag,
for up to 999 steps. Before the first success, the only thing the critic can learn is that
pushing costs reward, so the policy drifts to **doing nothing**: a famous local optimum.

- **DDPG**: deterministic actor `mu(s)`, critic `Q(s,a)`, actor follows `dQ/da`; Gaussian action noise;
- **TD3**: two critics and the min of the two in the target, noisy target actions, delayed actor;
- **SAC**: a stochastic tanh-Gaussian actor maximising reward + `alpha` x entropy, with `alpha`
  tuned automatically; the entropy keeps it exploring longer.

The first 5 000 actions are uniform random (warm-up), which is where the first success usually
comes from.
""" + "\n\n" + show("algorithms/actor_critic.py", "OffPolicyActorCritic.update"))
code("""
rows = []
for algo in ("sac", "td3", "ddpg", "ppo", "cem"):
    for shaping in ("none", "energy"):
        name = f"{algo} shaping={shaping}"
        cfg = recommended_config(algo, continuous=True, shaping=shaping, eval_episodes=10, log_every=10,
                                 output_dir=f"runs/nb/cont_{algo}_{shaping}")
        agent, h, visits, ev = train(cfg)
        rows.append(summary_row(name, h, ev))
    plot_agent_maps(agent, visits, None, title=f"{algo} continuous, shaped"); plt.show()
print(markdown_table(rows))
""")

md("""
## 9. Exploration aids

All of them are environment wrappers, so any method can use any of them:

- **Reward shaping** (`shaping="energy"` or `"height"`): add `F = gamma Phi(s') - Phi(s)`. Because
  `F` is a difference of potentials it telescopes over an episode, so the optimal policy is
  unchanged (Ng, Harada & Russell 1999), but every step that adds energy is now rewarded. It is
  *domain knowledge*: results with it are always labelled "+shaping".
- **Count-based bonus** (`intrinsic="count"`): `0.1 / sqrt(N(cell))`, large in rarely visited cells.
- **RND** (`intrinsic="rnd"`): the error of a network trained to imitate a frozen random network;
  high on states it has not been trained on, i.e. novel states.
- **Epsilon schedules** (`eps_schedule`): `const`, `linear` or `exp` decay of epsilon-greedy.
""" + "\n\n" + show("envs.py", "ShapingWrapper.step", "RND.__call__"))
code("""
pos, vel, states = state_grid(80)
plt.imshow(energy(states).reshape(80, 80).T, origin="lower", extent=EXTENT, aspect="auto")
plt.colorbar(label="Phi = energy"); plt.xlabel("position"); plt.ylabel("velocity"); plt.title("shaping potential")
plt.show()
rows = []
for name, extra in [("dqn", {}), ("dqn +count", {"intrinsic": "count"}), ("dqn +rnd", {"intrinsic": "rnd"}),
                    ("dqn +height", {"shaping": "height"}), ("dqn +energy", {"shaping": "energy"})]:
    cfg = recommended_config("dqn", output_dir=f"runs/nb/expl_{name.replace(' +', '_')}", **extra)
    agent, h, visits, ev = train(cfg)
    rows.append(summary_row(name, h, ev))
print(markdown_table(rows))
""")

md("""
## 10. Full comparison

`run_suite` trains every run of a suite with the same seed, budget rules and greedy evaluation,
and `make_figures` writes the curves, visitation small multiples, maps and a markdown table (the
README's tables come from exactly these calls, with `FULL = False`). With `FULL = True` every suite
is repeated over three seeds with twice the budget, which takes a few hours on a CPU session.
""")
code("""
from mountaincar_rl.compare import SUITES, make_figures, run_suite

FULL = False
SEEDS = [0, 1, 2] if FULL else [0]
for seed in SEEDS:
    for suite in ("tabular", "discrete-deep", "exploration", "continuous"):
        episodes = None
        if FULL:
            episodes = {"tabular": 6000, "continuous": 60}.get(suite, 600)
        rows = run_suite(suite, out=f"runs/full_seed{seed}", seed=seed, episodes=episodes, verbose=False)
        make_figures(suite, out=f"runs/full_seed{seed}", assets=f"assets_seed{seed}")
        print(f"\\n### {suite}, seed {seed}\\n")
        print(markdown_table(rows))
display(Image(f"assets_seed{SEEDS[0]}/discrete-deep_curves.png"))
""")

md("""
## 11. Watch the car

The greedy tile-coding agent from section 5, rendered headless to a GIF.
""")
code("""
from mountaincar_rl.play import play
play(tile_agent, continuous=False, gif="tile_coding.gif")
display(Image("tile_coding.gif"))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT} ({len(cells)} cells)")
