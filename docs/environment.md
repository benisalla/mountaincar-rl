# The environment: MountainCar, discrete and continuous

Both come from `gymnasium[classic-control]` (no download, no data folder). `envs.make_env` builds
them and adds the optional wrappers.

| | `MountainCar-v0` (`--no-continuous`, default) | `MountainCarContinuous-v0` (`--continuous`) |
|---|---|---|
| state | `(position, velocity)`, x in [-1.2, 0.6], v in [-0.07, 0.07] | same |
| actions | 0 = push left, 1 = no push, 2 = push right | force in [-1, 1] |
| reward | -1 per step | -0.1 a^2 per step, +100 at the flag |
| episode end | flag (x >= 0.5) = `terminated`; 200 steps = `truncated` | flag; 999 steps |
| start | x uniform in [-0.6, -0.4], v = 0 | same |

## Why it is hard, in plain words

The engine is weaker than gravity on the slope, so the only way up is to rock: go left up the
other hill, roll back down with speed, and use it to climb the right hill. The reward never says
"you are getting closer": a car that came within a metre of the flag and one that stood still
both get -200. **So a learner sees no difference between good and bad behaviour until it reaches
the flag once, by chance or by systematic exploration.** Each family of methods answers that
differently, and that difference is what the comparison measures.

The continuous version adds a trap: pushing costs `0.1 a^2` per step, so before any success the
best known behaviour is "push as little as possible", which never reaches the flag.

## Success

Success = the car reached the flag (`terminated=True`) before the time limit. Training logs and
evaluation both use this, plus the true return and the number of steps (fewer = better driving).

## Reward shaping and intrinsic bonuses

The wrappers change only the reward the *agent learns from*; the true environment reward is kept
in `info["env_reward"]` and that is what every log, curve and table shows.

* `ShapingWrapper` (`--shaping energy|height`, `--shaping-scale 10`): `r + gamma Phi(s') - Phi(s)`.
  The energy potential is `E = (0.0025/3 sin(3x) + v^2 / 2)` scaled so standing at the flag is 1:
  it rewards each step that pumps energy into the car. The height potential `sin(3x)` only rewards
  being high up, a weaker hint.
* `IntrinsicWrapper` (`--intrinsic count|rnd`, `--intrinsic-scale 0.1`): `r + 0.1 * bonus(s')`.
  Count: `1/sqrt(N)` on a 40 x 40 grid. RND: normalised prediction error of an online-trained MLP
  against a frozen random MLP.

## Rendering

`render_mode="rgb_array"` uses pygame. `envs.py` sets `SDL_VIDEODRIVER=dummy` before gymnasium
imports pygame, so rendering works without a display (servers, Kaggle, Colab, tests).
`mountaincar-play <checkpoint> --gif car.gif` writes a GIF.

## Helpers

* `normalize_obs(s)`: maps the state box to [-1, 1]^2 (input of every network and of CEM).
* `discretize(s, bins)`: grid cell `(i, j)`; used by the tabular methods, the count bonus and the
  visitation heatmaps.
* `state_grid(n)`: an `n x n` grid of states for the value and policy maps.
