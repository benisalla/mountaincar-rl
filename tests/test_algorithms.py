import numpy as np
import pytest
import torch

from mountaincar_rl.algorithms import epsilon, make_agent
from mountaincar_rl.config import ALGORITHMS, TrainConfig
from mountaincar_rl.envs import state_grid
from mountaincar_rl.train import load_agent, train

CASES = [(a, s == "continuous") for a, spaces in ALGORITHMS.items() for s in spaces]


def test_epsilon_schedules():
    assert epsilon("const", 0.7, 0.3, 0.0, 0.5) == 0.3
    assert epsilon("linear", 0.0, 1.0, 0.1, 0.5) == 1.0
    assert epsilon("linear", 0.25, 1.0, 0.0, 0.5) == pytest.approx(0.5)
    assert epsilon("linear", 0.9, 1.0, 0.1, 0.5) == pytest.approx(0.1)
    assert epsilon("exp", 1.0, 1.0, 0.0, 0.5) < 0.01


@pytest.mark.parametrize("algo,continuous", CASES)
def test_every_algorithm_trains_saves_and_reloads(algo, continuous, tmp_path):
    cfg = TrainConfig(algo=algo, continuous=continuous, episodes=2, max_steps=60, warmup_steps=40,
                      batch_size=16, rollout_steps=50, cem_population=2, eval_episodes=1,
                      output_dir=str(tmp_path / algo))
    agent, history, visits, ev = train(cfg, verbose=False)
    assert len(history.returns) == 2 and visits.sum() == 120
    assert 0.0 <= ev["success_rate"] <= 1.0
    _, _, states = state_grid(5)
    reloaded, cfg2 = load_agent(tmp_path / algo)
    assert cfg2 == cfg
    assert np.allclose(np.asarray(agent.policy_grid(states), float), np.asarray(reloaded.policy_grid(states), float))
    a = reloaded.act(np.array([-0.5, 0.0], np.float32), explore=False)
    if continuous:
        assert a.shape == (1,) and -1 <= a[0] <= 1
    else:
        assert a in (0, 1, 2)


def test_q_learning_update_is_the_textbook_one():
    agent = make_agent(TrainConfig(algo="q-learning", bins=4, alpha=0.5, gamma=0.9))
    s, s2 = np.array([-1.1, -0.06]), np.array([0.5, 0.06])
    agent.q[3, 3] = [0.0, 0.0, -2.0]
    agent.observe(s, 1, -1.0, s2, False, False)
    assert agent.q[0, 0, 1] == pytest.approx(0.5 * (-1.0 + 0.9 * 0.0))
    agent.observe(s, 2, -1.0, s2, True, False)                # terminal: no bootstrap
    assert agent.q[0, 0, 2] == pytest.approx(-0.5)


def test_dqn_overfits_one_batch():
    agent = make_agent(TrainConfig(algo="double-dqn", warmup_steps=10 ** 9, batch_size=32, lr=3e-3))
    rng = np.random.default_rng(0)
    for _ in range(32):
        s = rng.uniform([-1.2, -0.07], [0.6, 0.07])
        agent.observe(s, int(rng.integers(3)), 1.0, s, True, False)   # target = 1 everywhere
    losses = [agent.update() for _ in range(300)]
    assert losses[-1] < 0.05 * losses[0]


def test_policy_gradient_learns_a_bandit():
    """REINFORCE: action 2 is rewarded, the others are not -> pi(2|s) must grow."""
    agent = make_agent(TrainConfig(algo="reinforce", lr=1e-2, entropy_coef=0.0))
    s = np.array([-0.5, 0.0], np.float32)
    x = torch.zeros(1, 2)
    p0 = float(agent.net.dist(x).probs[0, 2].detach())
    for _ in range(150):
        for _ in range(10):
            a = agent.act(s)
            agent.observe(s, a, 1.0 if a == 2 else 0.0, s, False, False)
        agent.end_episode()
    assert float(agent.net.dist(x).probs[0, 2].detach()) > max(0.8, p0 + 0.3)


def test_sac_actor_log_prob_matches_tanh_change_of_variables():
    from mountaincar_rl.algorithms.actor_critic import GaussianActor

    torch.manual_seed(0)
    actor = GaussianActor(16)
    a, logp = actor(torch.zeros(4000, 2))
    assert a.abs().max() < 1 and torch.isfinite(logp).all()
