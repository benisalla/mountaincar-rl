# Testing

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q       # about a minute on one CPU
.venv/bin/ruff check src tests examples notebooks scripts
```

The suite uses the real gymnasium environments (no mocks) with short episodes (`max_steps=60`)
where a full episode is not needed.

| file | checks |
|---|---|
| `test_config.py` | validation errors (unknown algo, wrong action space, ...), run names, dict round trip, recommended settings for every method |
| `test_envs.py` | state normalisation, energy = 1 at the flag, **shaping telescopes** (sum of shaping terms = Phi(s_T) - Phi(s_0) with gamma = 1, so it is potential-based), true reward kept in `info`, count bonus decays, RND learns a repeated state, headless rgb rendering |
| `test_tile_coding.py` | one tile per tiling, indices in range, neighbours share tiles and far states do not, distinct offsets |
| `test_replay.py` | sum tree totals and prefix search, buffer wrap-around, **prioritized replay samples the high-error transition**, n-step returns and the flush at the flag |
| `test_algorithms.py` | epsilon schedules; **every method x action space trains, saves, reloads and acts identically**; the Q-learning update equals the textbook formula; **Double DQN overfits a fixed batch**; **REINFORCE learns a bandit**; SAC log-probabilities are finite |
| `test_learning.py` | real short runs: **tile coding solves MountainCar in < 150 episodes**, **CEM finds a working linear policy** |
| `test_cli.py` | `mountaincar-train` -> `mountaincar-eval` -> `mountaincar-play --gif`, and `mountaincar-compare` on a two-run suite with all figures and the table |

Beyond unit tests, the longer runs behind the README tables (`mountaincar-compare`) are the real
check that each method learns, or, for several methods without shaping, that it does not.
