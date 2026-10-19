# Evaluation

## Protocol (`evaluate.py`)

Every method is judged the same way, whatever it learnt from:

1. **greedy policy**: `agent.act(obs, explore=False)`: no epsilon, no action noise, the mean
   action of a stochastic policy, CEM's mean weights;
2. **fixed start states**: episodes reset with seeds 1000, 1001, ..., so all methods face the same
   start positions;
3. **true reward**: shaping and intrinsic bonuses are never applied during evaluation;
4. three numbers: **success rate** (reached the flag before the time limit), **mean return**,
   **mean steps** (fewer is better driving).

During training `train.py` also logs, per episode, the true return, steps and success, from which
`metrics.py` derives **episodes to first success** (how fast a method *discovers* the flag, the
exploration question) and the **training success rate over the last 50 episodes**.

Why both? A method can find the flag early but still have an unstable greedy policy at the end
(`dqn +per +n3` below), or rarely succeed in training (epsilon / noise) yet have a perfect greedy
policy (`double-dqn`, `reinforce +shaping`).

```bash
mountaincar-eval checkpoints/<run> --episodes 100       # JSON with the three numbers
mountaincar-play checkpoints/<run> --gif car.gif        # watch it
```

## The runs behind the README

`mountaincar-compare --suite tabular|discrete-deep|exploration|continuous` with the recommended
settings of `config.py`, seed 0, 20 evaluation episodes, `OMP_NUM_THREADS=1`. Each run is in
`runs/<suite>/<name>/` (not committed) and every figure and table in `assets/` is redrawn from
them with `--plots-only`. These are single-seed short CPU runs on a shared, loaded machine: they
show which method learns under a small budget, not a benchmark. Wall times vary with the load.

### tabular

| run | episodes | first success (episode) | train success (last 50) | eval success | eval return | eval steps | time |
|---|---|---|---|---|---|---|---|
| `q-learning` | 3000 | 215 | 70% | **55%** | -161.8 | 162 | 14 s |
| `sarsa` | 3000 | 57 | 94% | **90%** | -158.0 | 158 | 11 s |
| `tile-coding (sarsa)` | 300 | 23 | 100% | **95%** | -136.8 | 137 | 4 s |
| `tile-coding (q)` | 300 | 27 | 100% | **100%** | -112.6 | 113 | 6 s |
| `cem` | 300 | 21 | 100% | **100%** | -113.2 | 113 | 2 s |

### discrete-deep

| run | episodes | first success (episode) | train success (last 50) | eval success | eval return | eval steps | time |
|---|---|---|---|---|---|---|---|
| `dqn` | 300 | never | 0% | **0%** | -200.0 | 200 | 197 s |
| `dqn +shaping` | 300 | 65 | 100% | **95%** | -159.8 | 160 | 204 s |
| `double-dqn` | 300 | 165 | 4% | **80%** | -141.8 | 142 | 196 s |
| `double-dqn +shaping` | 300 | 65 | 100% | **100%** | -123.3 | 123 | 143 s |
| `dueling-dqn` | 300 | never | 0% | **0%** | -200.0 | 200 | 274 s |
| `dueling-dqn +shaping` | 300 | 65 | 100% | **100%** | -138.3 | 138 | 149 s |
| `reinforce` | 500 | never | 0% | **0%** | -200.0 | 200 | 45 s |
| `reinforce +shaping` | 500 | 238 | 24% | **95%** | -129.2 | 129 | 34 s |
| `a2c` | 500 | never | 0% | **0%** | -200.0 | 200 | 83 s |
| `a2c +shaping` | 500 | 371 | 100% | **100%** | -141.1 | 141 | 80 s |
| `ppo` | 500 | 198 | 100% | **100%** | -142.2 | 142 | 76 s |
| `ppo +shaping` | 500 | 135 | 100% | **100%** | -140.0 | 140 | 203 s |

### exploration

| run | episodes | first success (episode) | train success (last 50) | eval success | eval return | eval steps | time |
|---|---|---|---|---|---|---|---|
| `dqn eps exp` | 300 | 250 | 0% | **0%** | -200.0 | 200 | 77 s |
| `dqn +count` | 300 | 59 | 100% | **100%** | -119.5 | 119 | 62 s |
| `dqn +rnd` | 300 | 250 | 100% | **100%** | -152.8 | 153 | 84 s |
| `dqn +per +n3` | 300 | 98 | 46% | **0%** | -200.0 | 200 | 104 s |
| `dqn +height shaping` | 300 | 59 | 28% | **90%** | -149.1 | 149 | 73 s |
| `q-learning +energy shaping` | 3000 | 38 | 100% | **100%** | -132.3 | 132 | 16 s |

### continuous

| run | episodes | first success (episode) | train success (last 50) | eval success | eval return | eval steps | time |
|---|---|---|---|---|---|---|---|
| `ppo` | 60 | never | 0% | **0%** | -0.0 | 999 | 59 s |
| `ppo +shaping` | 60 | 3 | 100% | **100%** | 94.7 | 150 | 20 s |
| `ddpg` | 30 | never | 0% | **0%** | -0.0 | 999 | 105 s |
| `ddpg +shaping` | 30 | 6 | 83% | **100%** | 90.2 | 101 | 6 s |
| `td3` | 30 | never | 0% | **0%** | -0.3 | 999 | 73 s |
| `td3 +shaping` | 30 | 6 | 83% | **100%** | 93.1 | 85 | 11 s |
| `sac` | 30 | never | 0% | **0%** | -0.0 | 999 | 120 s |
| `sac +shaping` | 30 | 6 | 83% | **100%** | 92.3 | 109 | 23 s |
| `cem` | 100 | 7 | 92% | **100%** | 96.9 | 290 | 1 s |
| `cem +shaping` | 100 | 7 | 92% | **100%** | 95.3 | 404 | 1 s |

A longer greedy evaluation of the tile-coding (Q) agent: `mountaincar-eval runs/tabular/tile-coding_q
--episodes 100` gives **100% success, mean return -118.9 (std 15.2), 119 steps**.

## What the numbers say

- **Optimism is exploration.** The tabular and tile-coding agents start at Q = 0 above every real
  return and learn without any aid; neural Q-networks do not have this property.
- **Generalisation decides the speed.** Tile coding: first flag at episode 23-27, 95-100% after 300.
  The grid: 3000 episodes for 55-90%.
- **Without help, deep RL on a sparse reward is a lottery.** Of six neural methods on the raw
  reward, two found the flag once by chance (Double DQN, PPO) and then learnt it; four never did.
- **A dense signal fixes everything that bootstraps.** Energy shaping: every TD / actor-critic
  method solves both versions. It does little for Monte Carlo REINFORCE (the shaping terms
  telescope into a baseline), which needs a short horizon.
- **Shaping is not the only way.** A count-based novelty bonus (no physics knowledge) solves DQN
  at least as well as energy shaping here; RND also works, later.
- **Continuous: the do-nothing trap.** The action cost makes PPO, DDPG, TD3 and SAC converge to
  zero force without shaping. CEM with three weights solves it in a few generations.

## Figures

| file | what |
|---|---|
| `assets/<suite>_curves.png` | moving average of the true return and of success, per training episode |
| `assets/<suite>_visits.png` | training visitation over (position, velocity) for every run of the suite |
| `assets/<suite>_eval.png` | greedy success rate per run |
| `assets/<suite>/maps_<run>.png` | visitation, learned value and greedy policy of one run |
| `assets/before_after_*.gif`, `assets/solved_grid_*.gif`, `assets/policy_map_training.gif` | random vs trained, every solved method, the policy map while learning (`scripts/make_gifs.py`) |
