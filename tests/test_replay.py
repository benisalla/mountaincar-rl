import numpy as np
import pytest

from mountaincar_rl.replay import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer, SumTree


def test_sum_tree_total_and_find():
    t = SumTree(4)
    for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
        t.update(i, p)
    assert t.total == 10.0
    assert [t.find(m) for m in (0.5, 1.5, 3.5, 9.9)] == [0, 1, 2, 3]


def test_uniform_buffer_wraps():
    b = ReplayBuffer(5)
    for i in range(8):
        b.add([i, i], 1, -1.0, [i, i], 0.0)
    assert len(b) == 5
    assert b.sample(16)["obs"].shape == (16, 2)


def test_prioritized_buffer_prefers_large_td_errors():
    b = PrioritizedReplayBuffer(64, alpha=1.0, seed=0)
    for i in range(64):
        b.add([i, 0], 0, 0.0, [i, 0], 0.0)
    b.update_priorities(np.arange(64), np.where(np.arange(64) == 7, 100.0, 0.01))
    batch = b.sample(64)
    assert (batch["idx"] == 7).mean() > 0.5
    assert batch["weights"].max() == pytest.approx(1.0)


def test_n_step_returns():
    acc = NStepAccumulator(3, 0.5)
    assert acc.push("s0", 0, 1.0, "s1", False, False) == []
    assert acc.push("s1", 0, 1.0, "s2", False, False) == []
    (s, a, ret, s2, done, disc), = acc.push("s2", 0, 1.0, "s3", False, False)
    assert s == "s0" and s2 == "s3" and ret == pytest.approx(1.75) and disc == pytest.approx(0.125)
    out = acc.push("s3", 0, 10.0, "goal", True, False)   # flag: flush the queue, done = 1
    assert [o[0] for o in out] == ["s1", "s2", "s3"]
    assert out[-1][2] == 10.0 and all(o[4] == 1.0 for o in out)
