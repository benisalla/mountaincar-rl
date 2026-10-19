"""The methods really learn: short real runs on the real environment (a few seconds each)."""

from mountaincar_rl.config import recommended_config
from mountaincar_rl.train import train


def test_tile_coding_solves_mountaincar():
    cfg = recommended_config("tile-coding", episodes=150, eval_episodes=10)
    _, history, _, ev = train(cfg, verbose=False, save=False)
    assert history.first_success is not None and history.first_success < 150
    assert ev["success_rate"] >= 0.8 and ev["mean_steps"] < 200


def test_cem_finds_a_working_linear_policy():
    cfg = recommended_config("cem", episodes=160, eval_episodes=10)
    _, history, _, ev = train(cfg, verbose=False, save=False)
    assert ev["success_rate"] >= 0.8
