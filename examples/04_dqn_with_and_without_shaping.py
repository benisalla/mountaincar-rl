"""04: Double DQN on the raw sparse reward vs with potential-based energy shaping.

The same network, the same budget; only the reward the agent learns from changes. Without
shaping, epsilon-greedy dithering rarely reaches the flag in 200 steps and the net learns
"-1 forever" everywhere. With shaping F = gamma Phi(s') - Phi(s) (Phi = mechanical energy), each
step that pumps energy into the car is rewarded and the flag is found quickly. Evaluation always
uses the true reward.

Run from the repo root:  python examples/04_dqn_with_and_without_shaping.py   (a few minutes)
"""

from mountaincar_rl.config import recommended_config
from mountaincar_rl.train import train
from mountaincar_rl.viz import plot_agent_maps, plot_curves

histories = {}
for shaping in ("none", "energy"):
    name = f"double-dqn shaping={shaping}"
    cfg = recommended_config("double-dqn", episodes=150, shaping=shaping, log_every=25,
                             output_dir=f"examples/output/ddqn_{shaping}")
    agent, history, visits, ev = train(cfg)
    histories[name] = history
    plot_agent_maps(agent, visits, f"examples/output/ddqn_{shaping}_maps.png", title=name)
plot_curves(histories, "examples/output/dqn_shaping_curves.png", window=10, title="double DQN")
