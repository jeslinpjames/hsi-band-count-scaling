"""The separated split keeps every test window away from every training window."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bcs.splits import make_split  # noqa: E402


def synthetic_gt(H=200, W=160, classes=5, seed=0):
    rng = np.random.RandomState(seed)
    gt = np.zeros((H, W), np.float32)
    for c in range(1, classes + 1):                       # rectangular fields, as in agricultural scenes
        for _ in range(6):
            r, q = rng.randint(0, H - 30), rng.randint(0, W - 30)
            gt[r:r + rng.randint(10, 30), q:q + rng.randint(10, 30)] = c
    return gt


def test_random_split_counts():
    gt = synthetic_gt()
    tr, va, te = make_split("random", gt, 5, 10, 5, 0)
    assert len(tr) == 50 and len(va) == 25 and len(te) == int((gt > 0).sum()) - 75
    assert len(set(tr) | set(va) | set(te)) == len(tr) + len(va) + len(te)


def test_separated_split_gap():
    gt = synthetic_gt()
    for seed in range(3):
        tr, va, te = make_split("separated", gt, 5, 10, 5, seed)
        W = gt.shape[1]
        T = np.stack([np.r_[tr, va] // W, np.r_[tr, va] % W], 1)
        S = np.stack([te // W, te % W], 1)
        d = np.abs(S[:, None, :] - T[None, :, :]).max(-1).min()
        assert d >= 33                                    # no 33 x 33 test window overlaps a training window


def test_split_is_reproducible():
    gt = synthetic_gt()
    a, b = make_split("separated", gt, 5, 10, 5, 4), make_split("separated", gt, 5, 10, 5, 4)
    assert all((x == y).all() for x, y in zip(a, b))
