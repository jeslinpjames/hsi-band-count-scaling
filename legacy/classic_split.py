"""classic_split.py : the SVM baselines of the paper (spectrum-only SVM and the 33x33 spectral-spatial SVM) on the
spatially disjoint split.  Identical to baseline_classic.py / baseline_spatial.py except for the split.

usage:  python classic_split.py --dataset pavia --seeds 0-7
writes: results_classic_disjoint_<dataset>.csv   (methods svm, svm_sp33)
"""
import argparse, os, sys, time
sys.path.insert(0, ".")
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_common import load, scores, done, write
from split_modes import make_split_disjoint

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=["ip", "salinas", "pavia"])
p.add_argument("--seeds", default="0-7")
args = p.parse_args()
cube, gt, _, _ = load(args.dataset)
H, W, B = cube.shape
gt_flat = gt.reshape(-1); C = int(gt.max())
FEATS = {"svm": cube.reshape(-1, B),
         "svm_sp33": np.concatenate([cube.reshape(-1, B), uniform_filter(cube, size=(33, 33, 1), mode="reflect").reshape(-1, B)], 1)}
csv = "results_classic_disjoint_%s.csv" % args.dataset
lo, hi = (int(x) for x in args.seeds.split("-"))
for seed in range(lo, hi + 1):
    tr, va, te = make_split_disjoint(gt_flat, C, 10, 5, np.random.RandomState(seed))
    ytr, yva, yte = gt_flat[tr], gt_flat[va], gt_flat[te]
    for name, X in FEATS.items():
        if done(csv, name, 10, seed):
            continue
        t0 = time.time(); D = X.shape[1]; best = None
        for Cc in (1, 10, 100, 1000, 10000):
            for g in (0.1, 0.3, 1, 3, 10):
                acc = accuracy_score(yva, SVC(C=Cc, gamma=g / D).fit(X[tr], ytr).predict(X[va]))
                if best is None or acc > best[0]:
                    best = (acc, Cc, g / D)
        oa, aa, kap = scores(yte, SVC(C=best[1], gamma=best[2]).fit(X[tr], ytr).predict(X[te]), C)
        write(csv, name, 10, seed, 0, oa, aa, kap, 0, 0.0, -1, -1, time.time() - t0)
        print("seed %d %-9s OA %.4f AA %.4f kappa %.4f" % (seed, name, oa, aa, kap), flush=True)
print("CLASSIC_DISJOINT_DONE", args.dataset)
