import numpy as np

from mountaincar_rl.algorithms import TileCoder


def test_one_active_tile_per_tiling_and_in_range():
    tc = TileCoder(8, 8)
    for s in [[-1.2, -0.07], [0.6, 0.07], [-0.5, 0.0]]:
        idx = tc.active(s)
        assert len(idx) == 8 and len(set(idx)) == 8
        assert idx.min() >= 0 and idx.max() < tc.n_features


def test_neighbours_share_tiles_far_states_do_not():
    tc = TileCoder(8, 8)
    a = set(tc.active([-0.5, 0.0]))
    near = set(tc.active([-0.49, 0.001]))
    far = set(tc.active([0.4, 0.06]))
    assert len(a & near) >= 4
    assert len(a & far) == 0


def test_offsets_differ_between_tilings():
    tc = TileCoder(4, 4)
    assert len(np.unique(tc.offsets, axis=0)) == 4
