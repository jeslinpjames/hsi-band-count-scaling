"""Train/validation/test splits.

random     ten training and five validation pixels per class drawn at random, all other labelled pixels are test
           pixels (at most half of a class is used for training).
separated  spatially separated split: training and validation pixels come from a few 32 x 32 tiles ("training
           regions"); test pixels are all labelled pixels outside these regions that lie at least 33 pixels
           (Chebyshev) from every training and validation pixel, so no 33 x 33 test window shares a pixel with any
           training window. Called "disjoint" in the file names of the results.

Both take a numpy RandomState; the paper uses np.random.RandomState(seed) with seed = split index.
"""
import math
import numpy as np
from scipy.ndimage import maximum_filter

GAP = 33
BLOCK = 32
MIN_TEST = 50


def _per_class(idx, train_num, val_num):
    n = len(idx)
    n_tr = min(train_num, max(1, n // 2))
    n_va = min(val_num, max(1, (n - n_tr) // 2))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def split_random(gt_flat, class_num, train_num, val_num, rng):
    tr, va, te = [], [], []
    for c in range(1, class_num + 1):
        idx = np.where(gt_flat == c)[0]
        rng.shuffle(idx)
        a, b, t = _per_class(idx, train_num, val_num)
        tr.append(a); va.append(b); te.append(t)
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def split_separated(gt_flat, class_num, train_num, val_num, rng, shape, block=BLOCK, min_test=MIN_TEST, verbose=False):
    """shape = (H, W) of the scene. Draws are repeated until every class has at least min_test test pixels."""
    H, W = shape
    nby, nbx = math.ceil(H / block), math.ceil(W / block)
    bid = ((np.arange(H)[:, None] // block) * nbx + (np.arange(W)[None, :] // block)).reshape(-1)
    need = train_num + val_num
    for attempt in range(500):
        tiles = np.zeros(nby * nbx, bool)
        for c in rng.permutation(class_num) + 1:          # classes in random order
            in_c = bid[gt_flat == c]
            if np.isin(in_c, np.where(tiles)[0]).sum() >= need:
                continue
            cnt = np.bincount(in_c, minlength=nby * nbx)
            cand = np.where((cnt >= need) & ~tiles)[0]
            if len(cand) == 0:
                cand = np.where((cnt > 0) & ~tiles)[0]
            tiles[rng.choice(cand)] = True
        region = tiles[bid]
        tr, va = [], []
        for c in range(1, class_num + 1):
            idx = np.where((gt_flat == c) & region)[0]
            rng.shuffle(idx)
            a, b, _ = _per_class(idx, train_num, val_num)
            tr.append(a); va.append(b)
        tv = np.zeros(H * W, bool)
        tv[np.concatenate(tr + va)] = True
        near = maximum_filter(tv.reshape(H, W), size=2 * (GAP - 1) + 1).reshape(-1)   # within GAP-1 px of train/val
        te = [np.where((gt_flat == c) & ~region & ~near)[0] for c in range(1, class_num + 1)]
        if min(len(t) for t in te) < min_test:
            continue
        tr, va, te = np.concatenate(tr), np.concatenate(va), np.concatenate(te)
        if verbose:
            print("[split] separated: %d tiles, attempt %d, train %d val %d test %d" % (
                tiles.sum(), attempt + 1, len(tr), len(va), len(te)), flush=True)
        return tr, va, te
    raise RuntimeError("no valid separated split after 500 attempts")


def make_split(mode, gt, class_num, train_num, val_num, seed):
    """gt: (H, W) array. Returns flat indices (train, val, test)."""
    rng = np.random.RandomState(seed)
    gt_flat = gt.reshape(-1)
    if mode == "random":
        return split_random(gt_flat, class_num, train_num, val_num, rng)
    if mode in ("separated", "disjoint"):
        return split_separated(gt_flat, class_num, train_num, val_num, rng, gt.shape)
    raise ValueError(mode)
