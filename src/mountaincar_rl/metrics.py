"""Training history and the comparison numbers: episodes to first success, success rates, smoothing.

An episode is a *success* when the car reaches the flag (`terminated=True`), as opposed to
running out of time (`truncated=True` after 200 steps, 999 in the continuous version).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class History:
    """Per-training-episode log. `returns` are TRUE environment returns (no shaping, no bonus)."""

    returns: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    learner_returns: list[float] = field(default_factory=list)   # what the agent optimised (shaped)
    wall_time: float = 0.0

    def add(self, ret: float, steps: int, success: bool, learner_ret: float) -> None:
        self.returns.append(float(ret))
        self.steps.append(int(steps))
        self.successes.append(bool(success))
        self.learner_returns.append(float(learner_ret))

    @property
    def first_success(self) -> int | None:
        """1-based index of the first training episode that reached the flag, or None."""
        return next((i + 1 for i, s in enumerate(self.successes) if s), None)

    def success_rate(self, last: int = 50) -> float:
        tail = self.successes[-last:]
        return float(np.mean(tail)) if tail else 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> History:
        return cls(**json.loads(Path(path).read_text()))


def moving_average(x, window: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    window = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    head = c[1:window] / np.arange(1, window)      # growing window at the start
    return np.concatenate([head, (c[window:] - c[:-window]) / window])


def summary_row(name: str, history: History, evaluation: dict) -> dict:
    return {
        "run": name,
        "episodes": len(history.returns),
        "first_success": history.first_success,
        "train_success_last50": history.success_rate(50),
        "eval_success": evaluation["success_rate"],
        "eval_return": evaluation["mean_return"],
        "eval_steps": evaluation["mean_steps"],
        "wall_time_s": history.wall_time,
    }


def markdown_table(rows: list[dict]) -> str:
    head = ("| run | episodes | first success (episode) | train success (last 50) | eval success | eval return "
            "| eval steps | time |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        first = "never" if r["first_success"] is None else str(r["first_success"])
        lines.append(
            f"| `{r['run']}` | {r['episodes']} | {first} | {100 * r['train_success_last50']:.0f}% | "
            f"**{100 * r['eval_success']:.0f}%** | {r['eval_return']:.1f} | {r['eval_steps']:.0f} | "
            f"{r['wall_time_s']:.0f} s |"
        )
    return "\n".join(lines)
