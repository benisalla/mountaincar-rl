# Training

```bash
mountaincar-train --algo tile-coding                       # recommended settings for the method
mountaincar-train --algo dqn --shaping energy --episodes 300
mountaincar-train --algo dueling-dqn --prioritized --n-step 3
mountaincar-train --algo q-learning --eps-schedule linear --eps-start 0.1 --eps-end 0
mountaincar-train --algo sac --continuous --episodes 30
mountaincar-train --algo ppo --continuous --shaping energy
```

Every field of `config.TrainConfig` is a flag (`--n-step`, `--shaping-scale`, ...).
**Flags you do not give take the method's recommended value** (`config.RECOMMENDED` and
`RECOMMENDED_CONTINUOUS`), then the `TrainConfig` default. The output folder defaults to
`checkpoints/<run name>/` and holds:

| file | content |
|---|---|
| `agent.pt` | `{"config": ..., "agent": state dict}`, loaded by `train.load_agent` and every CLI |
| `history.json` | per training episode: true return, steps, success, the learner's (shaped) return, wall time |
| `visits.npy` | 50 x 50 visitation counts over (position, velocity) during training |
| `eval.json` | greedy evaluation (success rate, mean return, mean steps) |

## The loop (`train.py`)

The same loop runs every method; that is what makes the comparison fair:

```python
for ep in range(cfg.episodes):
    agent.progress = ep / cfg.episodes          # drives epsilon schedules, PER beta, CEM noise
    agent.start_episode()
    obs, _ = env.reset(seed=cfg.seed * 100_000 + ep)
    while not done:
        action = agent.act(obs, explore=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated, truncated)
        ...
    agent.end_episode()
```

`observe` is where each algorithm learns: one table entry (tabular), 8 weights (tile coding), one
minibatch step (DQN, DDPG, TD3, SAC), or it only stores the transition until a rollout is full
(A2C every 16 steps, PPO every 2 048 steps, REINFORCE and CEM at the end of the episode).

## Budgets and recommended settings

Episodes are the common unit (MountainCar-v0 episodes are at most 200 steps, the continuous ones
999). The defaults are chosen to finish on one CPU thread in seconds (tabular, tile coding, CEM)
to a few minutes (neural methods):

| method | episodes | key settings |
|---|---|---|
| `q-learning`, `sarsa` | 3 000 | 16 x 16 grid, alpha 0.5, epsilon 0, Q init 0 (optimistic) |
| `tile-coding` | 300 | 8 tilings of 8 x 8, alpha 0.5 (/8), epsilon 0 |
| `dqn`, `double-dqn`, `dueling-dqn` | 300 | 64-64 MLP, Adam 1e-3, buffer 50k, batch 64, target every 500, epsilon 1 -> 0.01 over half |
| `reinforce`, `a2c`, `ppo` | 500 | lr 1e-3 / 7e-4 / 3e-4, A2C 16-step, PPO 2 048-step rollouts x 10 epochs |
| `cem` | 300 | population 20, elite 20 %, one episode per candidate |
| continuous `ddpg`, `td3`, `sac` | 30 | lr 1e-3, 5 000 random warm-up steps, tau 0.005, noise 0.5 (DDPG/TD3) |
| continuous `reinforce`, `a2c`, `ppo` | 60 | as above, Gaussian policy |
| continuous `cem` | 100 | 3 weights, `a = tanh(w . phi)` |

## Threads and reproducibility

`--threads 1` (default) calls `torch.set_num_threads(1)`; the CLI also sets `OMP_NUM_THREADS=1`.
The networks are tiny, so more threads do not help. Seeds: `--seed` seeds numpy, torch, the
agent's own generator and the reset seeds (`seed * 100000 + episode`), so a run is repeatable on
the same machine.

## Comparing methods

```bash
mountaincar-compare --suite tabular          # q-learning, sarsa, tile coding (sarsa / q), cem
mountaincar-compare --suite discrete-deep    # 6 neural methods, each with and without shaping
mountaincar-compare --suite exploration      # DQN with each exploration aid
mountaincar-compare --suite continuous       # ppo, ddpg, td3, sac, cem, with and without shaping
mountaincar-compare --suite all --plots-only # redraw figures and tables from runs/
```

Runs are saved under `runs/<suite>/<name>/` and skipped when they exist (`--rerun` to retrain).
Figures and markdown tables go to `assets/`.
