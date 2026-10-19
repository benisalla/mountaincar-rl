import numpy as np
import pytest

from mountaincar_rl.envs import (
    RND,
    CountBonus,
    ShapingWrapper,
    discretize,
    energy,
    make_env,
    normalize_obs,
    state_grid,
)


def test_normalize_maps_the_box_to_unit_square():
    assert np.allclose(normalize_obs([-1.2, -0.07]), [-1, -1])
    assert np.allclose(normalize_obs([0.6, 0.07]), [1, 1])


def test_energy_is_one_at_the_flag_and_grows_with_speed():
    assert energy([0.5, 0.0]) == pytest.approx(1.0)
    assert energy([-0.5, 0.05]) > energy([-0.5, 0.0])


def test_discretize_corners():
    assert discretize([-1.2, -0.07], 10) == (0, 0)
    assert discretize([0.6, 0.07], 10) == (9, 9)


def test_shaping_is_potential_based_and_keeps_true_reward():
    """With gamma = 1 the shaping terms telescope: sum F = Phi(s_T) - Phi(s_0)."""
    env = make_env(shaping="energy", shaping_scale=10.0, gamma=1.0, max_steps=50)
    assert isinstance(env, ShapingWrapper)
    obs, _ = env.reset(seed=3)
    phi0, shaped, true = env.potential(obs), 0.0, 0.0
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(2)
        shaped += r
        true += info["env_reward"]
        done = term or trunc
    assert true == -50.0
    assert shaped - true == pytest.approx(env.potential(obs) - phi0, abs=1e-6)


def test_count_bonus_decays_and_rnd_learns_a_state():
    bonus = CountBonus()
    s = np.array([-0.5, 0.0])
    assert bonus(s) > bonus(s) > bonus(s)
    rnd = RND(batch=8)
    first = rnd(s)
    for _ in range(400):
        rnd(s)
    assert rnd(s) < first


def test_intrinsic_wrapper_adds_bonus():
    env = make_env(intrinsic="count", intrinsic_scale=1.0)
    env.reset(seed=0)
    _, r, _, _, info = env.step(1)
    assert info["env_reward"] == -1.0 and r == pytest.approx(0.0)   # -1 + 1/sqrt(1)


def test_headless_rgb_render():
    env = make_env(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    env.close()


def test_state_grid_shape():
    pos, vel, states = state_grid(7)
    assert states.shape == (49, 2) and pos[0] == -1.2 and vel[-1] == pytest.approx(0.07)
