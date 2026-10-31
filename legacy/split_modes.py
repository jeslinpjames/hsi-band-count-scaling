"""split_modes.py : random (paper protocol) and spatially disjoint train/test splits.

SPLIT_MODE=random   (default)  identical to make_split() in train_dofa.py / the classical baselines
SPLIT_MODE=disjoint            training pixels come from a few labelled regions (32x32 tiles); test pixels lie outside
                               those regions and at least 33 pixels from every training/validation pixel, so no
                               33x33 test window shares a pixel with any training window (see make_split_disjoint).

Self-check (run before any training):   python split_modes.py --check
"""
import os, math, inspect
import numpy as np
from scipy.ndimage import maximum_filter

GAP = 33
BLOCK = int(os.environ.get("SPLIT_BLOCK", "32"))
MIN_TEST = 50
SHAPES = {207400: (610, 340), 111104: (512, 217), 22201: (149, 149), 21025: (145, 145)}   # gt_flat length -> (H, W)


def make_split_random(gt_flat, class_num, train_num, val_num, rng):      # verbatim from train_dofa.py
    tr, va, te = [], [], []
    for c in range(1, class_num + 1):
        idx = np.where(gt_flat == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = min(train_num, max(1, n // 2))
        n_va = min(val_num, max(1, (n - n_tr) // 2))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    return (np.concatenate(tr), np.concatenate(va), np.concatenate(te))


def make_split_disjoint(gt_flat, class_num, train_num, val_num, rng, block=None, verbose=True, min_test=MIN_TEST):
    """Labelled training REGIONS instead of scattered pixels.
    1. The scene is cut into block x block tiles. Classes are visited in random order; for a class that does not yet
       have train_num + val_num pixels inside the chosen tiles, one random tile holding that many pixels of the class
       is added. The result is a small set of tiles (the 'training regions', about 3-10 tiles).
    2. Training and validation pixels are drawn at random inside the training regions (same per-class rule as the
       random split).
    3. Test pixels are all labelled pixels OUTSIDE the training regions that are at least GAP = 33 pixels (Chebyshev)
       from every training/validation pixel, so no test window shares a pixel with any training window.
    The draw is repeated until every class has at least min_test test pixels."""
    block = block or BLOCK
    if len(gt_flat) not in SHAPES:
        raise SystemExit("split_modes: unknown scene size %d - add it to SHAPES" % len(gt_flat))
    H, W = SHAPES[len(gt_flat)]
    nby, nbx = math.ceil(H / block), math.ceil(W / block)
    bid = ((np.arange(H)[:, None] // block) * nbx + (np.arange(W)[None, :] // block)).reshape(-1)
    need = train_num + val_num
    for attempt in range(500):
        tiles = np.zeros(nby * nbx, bool)
        for c in rng.permutation(class_num) + 1:
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
            n = len(idx)
            n_tr = min(train_num, max(1, n // 2))
            n_va = min(val_num, max(1, (n - n_tr) // 2))
            tr.append(idx[:n_tr]); va.append(idx[n_tr:n_tr + n_va])
        tv = np.zeros(H * W, bool); tv[np.concatenate(tr + va)] = True
        near = maximum_filter(tv.reshape(H, W), size=2 * (GAP - 1) + 1).reshape(-1)    # within 32 px of a train/val pixel
        te = [np.where((gt_flat == c) & ~region & ~near)[0] for c in range(1, class_num + 1)]
        if min(len(t) for t in te) < min_test:
            continue
        tr, va, te = np.concatenate(tr), np.concatenate(va), np.concatenate(te)
        if verbose:
            print("[split] disjoint (training regions: %d tiles of %d px, gap %d, attempt %d): train %d  val %d  test %d of %d labelled" % (
                tiles.sum(), block, GAP, attempt + 1, len(tr), len(va), len(te), int((gt_flat > 0).sum())), flush=True)
        return tr, va, te
    raise SystemExit("split_modes: no valid disjoint split after 500 attempts")


def wrap(orig):
    """Used by the patched training scripts: dispatch on SPLIT_MODE, otherwise call the original function."""
    sig = inspect.signature(orig)
    def make_split(*a, **k):
        if os.environ.get("SPLIT_MODE", "random") != "disjoint":
            return orig(*a, **k)
        b = sig.bind(*a, **k); b.apply_defaults(); v = list(b.arguments.values())
        return make_split_disjoint(v[0], v[1], v[2], v[3], v[4])
    print("[split] mode = %s" % os.environ.get("SPLIT_MODE", "random"), flush=True)
    return make_split


def make_split(gt_flat, class_num, train_num, val_num, rng):
    if os.environ.get("SPLIT_MODE", "random") == "disjoint":
        return make_split_disjoint(gt_flat, class_num, train_num, val_num, rng)
    return make_split_random(gt_flat, class_num, train_num, val_num, rng)


if __name__ == "__main__":
    import sys, scipy.io as sio
    sys.path.insert(0, ".")
    from data_common import DATASETS
    for ds in ("pavia", "salinas"):
        fg, kg = DATASETS[ds][2], DATASETS[ds][3]
        gt = sio.loadmat(fg)[kg].astype(np.float32)
        gf = gt.reshape(-1); C = int(gt.max())
        print("\n%s: gt %s, %d classes" % (ds, gt.shape, C))
        for seed in range(8):
            tr, va, te = make_split_disjoint(gf, C, 10, 5, np.random.RandomState(seed), verbose=False)
            H, W = gt.shape
            r = lambda ix: np.stack([ix // W, ix % W], 1)
            T = r(np.concatenate([tr, va])); S = r(te)          # exact minimum Chebyshev distance, all test pixels
            mind = min(np.abs(S[i:i + 2000, None, :] - T[None, :, :]).max(-1).min() for i in range(0, len(S), 2000))
            per = [int((gf[te] == c).sum()) for c in range(1, C + 1)]
            print("  seed %d: train %3d val %3d test %6d | min test-train distance %d px | smallest test class %d" % (
                seed, len(tr), len(va), len(te), mind, min(per)))
