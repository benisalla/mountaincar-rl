import pytest

from mountaincar_rl.config import ALGORITHMS, TrainConfig, recommended_config


def test_defaults_validate_and_env_id():
    cfg = TrainConfig().validate()
    assert cfg.env_id == "MountainCar-v0"
    assert TrainConfig(algo="sac", continuous=True).validate().env_id == "MountainCarContinuous-v0"


@pytest.mark.parametrize("bad", [dict(algo="nope"), dict(algo="sac"), dict(algo="dqn", continuous=True),
                                 dict(shaping="x"), dict(intrinsic="x"), dict(eps_schedule="x"), dict(episodes=0)])
def test_invalid_configs_raise(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad).validate()


def test_round_trip_and_run_name():
    cfg = TrainConfig(algo="dqn", shaping="energy", intrinsic="rnd", prioritized=True, n_step=3)
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.run_name == "dqn_disc_shape-energy_rnd_per_n3"


def test_recommended_settings_and_overrides():
    cfg = recommended_config("tile-coding", episodes=7)
    assert cfg.episodes == 7 and cfg.eps_start == 0.0
    for algo, spaces in ALGORITHMS.items():
        for space in spaces:
            recommended_config(algo, space == "continuous")
