"""03: semi-gradient SARSA with tile coding, the Sutton & Barto solution, and a GIF of the car.

8 offset tilings of 8 x 8 tiles: each state switches on 8 binary features and Q(s, a) is the sum
of their weights. Generalisation across neighbouring states makes it reach the flag in about a
hundred episodes, against thousands for the plain grid of example 02.

Run from the repo root:  python examples/03_tile_coding.py   (under a minute)
"""

from mountaincar_rl.config import recommended_config
from mountaincar_rl.play import play
from mountaincar_rl.train import train
from mountaincar_rl.viz import plot_agent_maps

cfg = recommended_config("tile-coding", episodes=200, output_dir="examples/output/tile-coding")
agent, history, visits, ev = train(cfg)
print("first success at training episode", history.first_success)
plot_agent_maps(agent, visits, "examples/output/tile_coding_maps.png", title="tile coding (semi-gradient SARSA)")
play(agent, continuous=False, gif="examples/output/tile_coding.gif")
