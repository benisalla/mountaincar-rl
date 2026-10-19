"""Every figure: return curves, success curves, heatmaps over (position, velocity), policy maps, GIFs.

The heatmaps all share the same axes: position on x (the flag is at 0.5, on the right), velocity
on y. A good MountainCar policy looks like a spiral-shaped split: push right when moving right,
push left when moving left, and the value function rises along the spiral towards the flag.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from mountaincar_rl.envs import GOAL_POS, POS_MAX, POS_MIN, VEL_MAX, VEL_MIN, state_grid
from mountaincar_rl.metrics import moving_average

EXTENT = [POS_MIN, POS_MAX, VEL_MIN, VEL_MAX]
ACTION_CMAP = ListedColormap(["#d1495b", "#edae49", "#00798c"])   # left, none, right


def _finish(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_curves(histories: dict, path=None, window: int = 20, title: str = "MountainCar"):
    """Smoothed true return and success rate per training episode, one line per run."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    colors = plt.get_cmap("tab20").colors
    for k, (name, h) in enumerate(histories.items()):
        style = {"color": colors[k % 20], "lw": 1.6, "ls": "-" if "+" in name else "--"}   # dashed = no aid
        axes[0].plot(moving_average(h.returns, window), label=name, **style)
        axes[1].plot(moving_average(np.asarray(h.successes, float), window), label=name, **style)
    axes[0].set(title=f"{title}: return (moving avg {window})", xlabel="training episode", ylabel="true return")
    axes[1].set(title="success rate (reached the flag)", xlabel="training episode", ylabel="success",
                ylim=(-0.03, 1.03))
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    return _finish(fig, path)


def _style(ax, title):
    ax.axvline(GOAL_POS, color="k", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set(title=title, xlabel="position", ylabel="velocity")


def plot_agent_maps(agent, visits=None, path=None, title: str = "", n: int = 80):
    """Three panels: training visitation (log counts), value V(s) or max_a Q(s, a), greedy policy."""
    pos, vel, states = state_grid(n)
    policy = np.asarray(agent.policy_grid(states), dtype=float).reshape(n, n)
    value = agent.value_grid(states)
    panels = 1 + (visits is not None) + (value is not None)
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.2))
    axes = np.atleast_1d(axes)
    k = 0
    if visits is not None:
        im = axes[k].imshow(np.log1p(visits).T, origin="lower", extent=EXTENT, aspect="auto", cmap="magma")
        fig.colorbar(im, ax=axes[k], label="log(1 + visits)")
        _style(axes[k], "states visited in training")
        k += 1
    if value is not None:
        im = axes[k].imshow(np.asarray(value).reshape(n, n).T, origin="lower", extent=EXTENT, aspect="auto",
                            cmap="viridis")
        fig.colorbar(im, ax=axes[k], label="value")
        _style(axes[k], "learned value  V(s) / max_a Q(s,a)")
        k += 1
    if agent.continuous:
        im = axes[k].imshow(policy.T, origin="lower", extent=EXTENT, aspect="auto", cmap="coolwarm_r", vmin=-1, vmax=1)
        fig.colorbar(im, ax=axes[k], label="force")
    else:
        im = axes[k].imshow(policy.T, origin="lower", extent=EXTENT, aspect="auto", cmap=ACTION_CMAP,
                            vmin=-0.5, vmax=2.5)
        cb = fig.colorbar(im, ax=axes[k], ticks=[0, 1, 2])
        cb.ax.set_yticklabels(["left", "none", "right"])
    _style(axes[k], "greedy policy")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, path)


def plot_visits_grid(visits: dict, path=None, cols: int = 4):
    """Small multiples of the training visitation of several runs: who explored where."""
    rows = int(np.ceil(len(visits) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False)
    for ax, (name, v) in zip(axes.ravel(), visits.items()):
        ax.imshow(np.log1p(v).T, origin="lower", extent=EXTENT, aspect="auto", cmap="magma")
        ax.axvline(GOAL_POS, color="w", ls="--", lw=1)
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(visits):]:
        ax.axis("off")
    fig.suptitle("training visitation over (position, velocity); dashed line = flag")
    fig.tight_layout()
    return _finish(fig, path)


def plot_comparison_bars(rows: list[dict], path=None, title: str = "greedy evaluation"):
    names = [r["run"] for r in rows]
    succ = [100 * r["eval_success"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rows) + 2), 4))
    bars = ax.bar(range(len(rows)), succ, color=["#00798c" if s > 0 else "#bbbbbb" for s in succ])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set(ylabel="success rate (%)", ylim=(0, 105), title=title)
    for b, s in zip(bars, succ):
        ax.text(b.get_x() + b.get_width() / 2, s + 1, f"{s:.0f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _finish(fig, path)


def save_gif(frames: list, path, fps: int = 30, every: int = 2) -> Path:
    from PIL import Image

    images = [Image.fromarray(f).resize((300, 200)) for f in frames[::every]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 * every / fps), loop=0)
    return path


def plot_track(path=None):
    """The MountainCar track y = sin(3x), with the start region and the flag."""
    x = np.linspace(POS_MIN, POS_MAX, 300)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, np.sin(3 * x), "k")
    ax.axvspan(-0.6, -0.4, color="#00798c", alpha=0.2, label="start region")
    ax.axvline(GOAL_POS, color="#d1495b", ls="--", label="flag (x = 0.5)")
    ax.set(xlabel="position", ylabel="height sin(3x)", title="MountainCar track")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, path)
