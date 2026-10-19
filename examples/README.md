# Examples

Six short scripts that use the `mountaincar_rl` package from Python, in the order one would learn
it. Each is one command, runs on a CPU with small defaults, and writes its figures to
`examples/output/`. Run them from the repository root after `pip install -e ".[dev]"`.

| script | what it shows | time (1 CPU thread) |
|---|---|---|
| [01_explore_the_environment.py](01_explore_the_environment.py) | the track, why a random policy never reaches the flag, the one-line policy that does, the energy potential used for shaping, a headless rgb frame | seconds |
| [02_tabular_q_learning_and_sarsa.py](02_tabular_q_learning_and_sarsa.py) | Q-learning and SARSA on a 16 x 16 grid with an optimistic table: visitation, value and policy maps | 1-4 minutes |
| [03_tile_coding.py](03_tile_coding.py) | semi-gradient SARSA with 8 tilings (Sutton & Barto 10.1), its maps and a GIF of the car | under a minute |
| [04_dqn_with_and_without_shaping.py](04_dqn_with_and_without_shaping.py) | the same Double DQN on the raw sparse reward and with energy shaping | about 2 minutes |
| [05_policy_gradient_and_cem.py](05_policy_gradient_and_cem.py) | REINFORCE, A2C, PPO (with shaping) and the cross-entropy method, one comparison table | about 3 minutes |
| [06_continuous_control.py](06_continuous_control.py) | SAC, TD3, DDPG, PPO and CEM on MountainCarContinuous-v0 | about a minute |

```bash
python examples/01_explore_the_environment.py
python examples/02_tabular_q_learning_and_sarsa.py
python examples/03_tile_coding.py
python examples/04_dqn_with_and_without_shaping.py
python examples/05_policy_gradient_and_cem.py
python examples/06_continuous_control.py
```

The same runs from the command line: `mountaincar-train --algo <method> [--continuous] [--shaping energy]`,
then `mountaincar-eval`, `mountaincar-play --gif` and `mountaincar-compare --suite <suite>` (see the
main README).
