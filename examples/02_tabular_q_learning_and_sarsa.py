"""02: tabular Q-learning and SARSA on a 16 x 16 grid of (position, velocity).

The table starts at 0, which is optimistic (every true return is negative), so the greedy agent
keeps trying actions it has not tried yet: exploration without epsilon. It still needs a couple
of thousand episodes, because each cell must be learnt on its own. Draws the visitation, the
max_a Q(s, a) value and the greedy policy for both methods.

Run from the repo root:  python examples/02_tabular_q_learning_and_sarsa.py   (1-4 minutes depending on the machine)
"""

from mountaincar_rl.config import recommended_config
from mountaincar_rl.train import train
from mountaincar_rl.viz import plot_agent_maps, plot_curves

histories = {}
for algo in ("q-learning", "sarsa"):
    cfg = recommended_config(algo, episodes=1500, log_every=500, output_dir=f"examples/output/{algo}")
    agent, history, visits, ev = train(cfg)
    histories[algo] = history
    plot_agent_maps(agent, visits, f"examples/output/{algo}_maps.png", title=algo)
plot_curves(histories, "examples/output/tabular_curves.png", title="tabular")
print("figures in examples/output/")
