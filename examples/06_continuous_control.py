"""06: MountainCarContinuous-v0 with SAC, TD3, DDPG, PPO and CEM.

Here the reward is +100 at the flag minus 0.1 a^2 per step, over 999 steps: an agent that has
never seen the flag learns that pushing costs reward and settles on "do nothing". This script
trains each method for a short budget with energy shaping (CEM without), then prints the
same evaluation table as every other example. Change SHAPING to "none" to see the trap.

Run from the repo root:  python examples/06_continuous_control.py   (about a minute on one CPU)
"""

from mountaincar_rl.config import recommended_config
from mountaincar_rl.metrics import markdown_table, summary_row
from mountaincar_rl.train import train
from mountaincar_rl.viz import plot_agent_maps

SHAPING = "energy"
runs = [("sac", 10), ("td3", 10), ("ddpg", 10), ("ppo", 30), ("cem", 60)]
rows = []
for algo, episodes in runs:
    shaping = "none" if algo == "cem" else SHAPING
    cfg = recommended_config(algo, continuous=True, episodes=episodes, shaping=shaping, warmup_steps=2000,
                             eval_episodes=5, log_every=5, output_dir=f"examples/output/{algo}_continuous")
    agent, history, visits, ev = train(cfg)
    rows.append(summary_row(f"{algo} (shaping={shaping})", history, ev))
    plot_agent_maps(agent, visits, f"examples/output/{algo}_continuous_maps.png", title=f"{algo} continuous")
print(markdown_table(rows))
