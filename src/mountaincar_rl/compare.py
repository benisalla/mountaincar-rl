"""Run several methods with the same seed, budget rules and evaluation, then table + figures.

A *suite* is a list of runs (name, algo, continuous, overrides). Each run is trained with its
recommended settings (config.RECOMMENDED) plus the overrides, saved in `<out>/<suite>/<name>/`,
and summarised as one table row: episodes to first success, training success rate over the last
50 episodes, greedy evaluation (success, return, steps) and wall time. The suites below are
exactly the runs behind the README tables.

CLI:  mountaincar-compare --suite tabular            # seconds to a minute
      mountaincar-compare --suite discrete-deep      # neural nets, with and without shaping
      mountaincar-compare --suite continuous
      mountaincar-compare --suite exploration        # DQN + each exploration aid
      mountaincar-compare --suite tabular --plots-only   # redraw from saved runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mountaincar_rl.config import recommended_config
from mountaincar_rl.metrics import History, markdown_table, summary_row

SHAPED = {"shaping": "energy"}

SUITES: dict[str, list[tuple[str, str, bool, dict]]] = {
    "tabular": [
        ("q-learning", "q-learning", False, {}),
        ("sarsa", "sarsa", False, {}),
        ("tile-coding (sarsa)", "tile-coding", False, {}),
        ("tile-coding (q)", "tile-coding", False, {"td_target": "q"}),
        ("cem", "cem", False, {}),
    ],
    "discrete-deep": [
        (name, algo, False, extra)
        for algo in ("dqn", "double-dqn", "dueling-dqn", "reinforce", "a2c", "ppo")
        for name, extra in ((algo, {}), (f"{algo} +shaping", SHAPED))
    ],
    # "dqn" (epsilon linear) and "dqn +shaping" (energy) are in discrete-deep; the README table joins them
    "exploration": [
        ("dqn eps exp", "dqn", False, {"eps_schedule": "exp"}),
        ("dqn +count", "dqn", False, {"intrinsic": "count"}),
        ("dqn +rnd", "dqn", False, {"intrinsic": "rnd"}),
        ("dqn +per +n3", "dqn", False, {"prioritized": True, "n_step": 3}),
        ("dqn +height shaping", "dqn", False, {"shaping": "height"}),
        ("q-learning +energy shaping", "q-learning", False, SHAPED),
    ],
    "continuous": [
        (name, algo, True, extra)
        for algo in ("ppo", "ddpg", "td3", "sac", "cem")
        for name, extra in ((algo, {}), (f"{algo} +shaping", SHAPED))
    ],
}


def slug(name: str) -> str:
    return name.replace(" +", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")


def run_suite(suite: str, out: str = "runs", seed: int = 0, episodes: int | None = None,
              skip_existing: bool = True, only: list[str] | None = None, verbose: bool = True) -> list[dict]:
    from mountaincar_rl.train import train

    for name, algo, continuous, extra in SUITES[suite]:
        if only and name not in only:
            continue
        run_dir = Path(out) / suite / slug(name)
        if skip_existing and (run_dir / "eval.json").exists():
            continue
        overrides = dict(extra, seed=seed, output_dir=str(run_dir), log_every=100 if not continuous else 10)
        if episodes is not None:
            overrides["episodes"] = episodes
        cfg = recommended_config(algo, continuous, **overrides)
        train(cfg, verbose=verbose)
    return collect(suite, out)


def collect(suite: str, out: str = "runs") -> list[dict]:
    rows = []
    for name, *_ in SUITES[suite]:
        run_dir = Path(out) / suite / slug(name)
        if not (run_dir / "eval.json").exists():
            continue
        rows.append(summary_row(name, History.load(run_dir / "history.json"),
                                json.loads((run_dir / "eval.json").read_text())))
    return rows


def make_figures(suite: str, out: str = "runs", assets: str = "assets") -> list[Path]:
    from mountaincar_rl.train import load_agent
    from mountaincar_rl.viz import plot_agent_maps, plot_comparison_bars, plot_curves, plot_visits_grid

    rows = collect(suite, out)
    histories, visits, written = {}, {}, []
    for row in rows:
        run_dir = Path(out) / suite / slug(row["run"])
        histories[row["run"]] = History.load(run_dir / "history.json")
        visits[row["run"]] = np.load(run_dir / "visits.npy")
        agent, cfg = load_agent(run_dir)
        p = Path(assets) / suite / f"maps_{slug(row['run'])}.png"
        plot_agent_maps(agent, visits[row["run"]], p, title=f"{row['run']} ({cfg.env_id})")
        written.append(p)
    a = Path(assets)
    window = 5 if suite == "continuous" else 20
    plot_curves(histories, a / f"{suite}_curves.png", window=window, title=suite)
    plot_visits_grid(visits, a / f"{suite}_visits.png")
    plot_comparison_bars(rows, a / f"{suite}_eval.png", title=f"{suite}: greedy evaluation success")
    written += [a / f"{suite}_curves.png", a / f"{suite}_visits.png", a / f"{suite}_eval.png"]
    (a / f"{suite}_table.md").write_text(markdown_table(rows) + "\n")
    (a / f"{suite}_results.json").write_text(json.dumps(rows, indent=2))
    return written


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compare RL methods on MountainCar with the same evaluation.")
    p.add_argument("--suite", choices=sorted(SUITES) + ["all"], default="tabular")
    p.add_argument("--out", default="runs")
    p.add_argument("--assets", default="assets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=None, help="override every run's episode budget")
    p.add_argument("--only", nargs="*", default=None, help="names of the runs to (re)train")
    p.add_argument("--rerun", action="store_true", help="retrain runs that already exist")
    p.add_argument("--plots-only", action="store_true")
    args = p.parse_args(argv)
    suites = sorted(SUITES) if args.suite == "all" else [args.suite]
    for suite in suites:
        if not args.plots_only:
            run_suite(suite, args.out, args.seed, args.episodes, not args.rerun, args.only)
        make_figures(suite, args.out, args.assets)
        print(f"\n## {suite}\n")
        print(markdown_table(collect(suite, args.out)))


if __name__ == "__main__":
    main()
