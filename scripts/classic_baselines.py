"""Classical baselines on exactly the same pixels as the foundation-model runs.

  svm       RBF-SVM on the pixel spectrum
  rf        random forest (500 trees) on the pixel spectrum
  svm_sp7   RBF-SVM on [pixel spectrum, mean spectrum of the 7 x 7 window]
  svm_sp33  RBF-SVM on [pixel spectrum, mean spectrum of the 33 x 33 window] (the window DOFA sees)

Hyper-parameters are selected on the validation pixels; the final model is fit on the training pixels only.

  python scripts/classic_baselines.py --dataset ip --seeds 0-15 --csv results/new/classic_ip.csv
  python scripts/classic_baselines.py --dataset pavia --split separated --methods svm,svm_sp33 --seeds 0-7 --csv ...
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from bcs.data import append_row, is_done, load, scores
from bcs.splits import make_split

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", required=True, choices=["ip", "salinas", "pavia"])
p.add_argument("--split", default="random", choices=["random", "separated"])
p.add_argument("--methods", default="svm,rf,svm_sp7,svm_sp33")
p.add_argument("--seeds", default="0-15")
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--n_jobs", type=int, default=8)
p.add_argument("--csv", required=True)
args = p.parse_args()
lo, hi = (int(v) for v in args.seeds.split("-"))

cube, gt, _, _ = load(args.dataset)
H, W, B = cube.shape
gt_flat = gt.reshape(-1)
C = int(gt.max())
spec = cube.reshape(-1, B)


def features(name):
    if name in ("svm", "rf"):
        return spec
    k = int(name.replace("svm_sp", ""))                       # neighbourhood mean, image borders mirrored
    return np.concatenate([spec, uniform_filter(cube, size=(k, k, 1), mode="reflect").reshape(-1, B)], 1).astype(np.float32)


FEATS = {m: features(m) for m in args.methods.split(",")}
for seed in range(lo, hi + 1):
    tr, va, te = make_split(args.split, gt, C, args.train_num, args.val_num, seed)
    ytr, yva, yte = (gt_flat[i].astype(int) for i in (tr, va, te))
    for name, X in FEATS.items():
        if is_done(args.csv, name, args.train_num, seed):
            continue
        t0, best = time.time(), None
        if name == "rf":
            for mf in ("sqrt", 0.1, 0.3):
                acc = accuracy_score(yva, RandomForestClassifier(n_estimators=500, max_features=mf, random_state=seed,
                                                                 n_jobs=args.n_jobs).fit(X[tr], ytr).predict(X[va]))
                if best is None or acc > best[0]:
                    best = (acc, mf)
            clf = RandomForestClassifier(n_estimators=500, max_features=best[1], random_state=seed, n_jobs=args.n_jobs)
        else:
            D = X.shape[1]                                    # gamma relative to 1 / number of features
            for c_ in (1, 10, 100, 1000, 10000):
                for g in (0.1, 0.3, 1, 3, 10):
                    acc = accuracy_score(yva, SVC(C=c_, gamma=g / D, kernel="rbf").fit(X[tr], ytr).predict(X[va]))
                    if best is None or acc > best[0]:
                        best = (acc, c_, g / D)
            clf = SVC(C=best[1], gamma=best[2], kernel="rbf")
        oa, aa, k = scores(yte, clf.fit(X[tr], ytr).predict(X[te]), C)
        append_row(args.csv, name, args.train_num, seed, 0, oa, aa, k, 0, 0.0, -1, -1, time.time() - t0, best_val=best[0])
        print("seed %2d %-9s OA %.4f AA %.4f kappa %.4f  (test %d)" % (seed, name, oa, aa, k, len(te)), flush=True)
