"""05: on-policy policy gradient (REINFORCE, A2C, PPO) and the cross-entropy method.

The three gradient methods are trained with energy shaping (without it every return is -200 and
the gradient carries no signal: try `shaping="none"`). CEM needs no shaping: it searches a
9-weight linear policy directly and only compares returns.

Each method uses its recommended budget (500 episodes for the gradient methods: A2C and
REINFORCE first reach the flag only after 200-400 episodes, 300 for CEM).

Run from the repo root:  python examples/05_policy_gradient_and_cem.py   (about 5 minutes)
"""

from mountaincar_rl.config import recommended_config
from mountaincar_rl.metrics import markdown_table, summary_row
from mountaincar_rl.train import train
from mountaincar_rl.viz import plot_agent_maps, plot_curves

runs = [("reinforce", {"shaping": "energy"}), ("a2c", {"shaping": "energy"}),
        ("ppo", {"shaping": "energy"}), ("cem", {})]
histories, rows = {}, []
for algo, extra in runs:
    name = algo + (" +shaping" if extra else "")
    cfg = recommended_config(algo, log_every=100, output_dir=f"examples/output/{algo}", **extra)
    agent, history, visits, ev = train(cfg)
    histories[name] = history
    rows.append(summary_row(name, history, ev))
    plot_agent_maps(agent, visits, f"examples/output/{algo}_maps.png", title=algo)
plot_curves(histories, "examples/output/pg_cem_curves.png", title="policy gradient and CEM")
print(markdown_table(rows))
