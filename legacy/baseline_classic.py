"""baseline_classic.py : classic pixel-wise baselines (SVM-RBF, Random Forest) on EXACTLY the same
train / val / test pixels as the DOFA runs (same make_split code, same RandomState(seed), same
per-band standardisation). Hyper-parameters are chosen on the same 5-per-class validation pixels
DOFA uses for early stopping; the final model is fit on the training pixels only (as DOFA is).

usage:  python baseline_classic.py --dataset ip|salinas|pavia --seeds 0-15 [--n_jobs 8]
writes: results_classic_<dataset>.csv  (same columns as the DOFA CSVs; method = svm / rf)
"""
import argparse, os, time
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

DATASETS = {   # identical paths / keys to train_dofa.py, train_dofa_salinas.py, train_dofa_pavia.py
    "ip":      ("../repos/HyperSIGMA/ImageClassification/data/Indian_pines_corrected.mat", "data",
                "../repos/HyperSIGMA/ImageClassification/data/Indian_pines_gt.mat", "groundT"),
    "salinas": ("../data/Salinas/Salinas_corrected.mat", "salinas_corrected",
                "../data/Salinas/Salinas_gt.mat", "salinas_gt"),
    "pavia":   ("../data/PaviaU/paviaU.mat", "ori_data",
                "../data/PaviaU/paviaU_7gt.mat", "map"),
}

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=list(DATASETS))
p.add_argument("--seeds", default="0-15")
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--n_jobs", type=int, default=8)
p.add_argument("--csv", default=None)
args = p.parse_args()
lo, hi = (int(x) for x in args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds),) * 2
csv = args.csv or "results_classic_%s.csv" % args.dataset

# ---------------- data: same loading + per-band standardisation as the DOFA scripts ----------------
fc, kc, fg, kg = DATASETS[args.dataset]
cube = sio.loadmat(fc)[kc].astype(np.float32)
gt = sio.loadmat(fg)[kg].astype(np.float32)
H, W, B = cube.shape
flat = cube.reshape(-1, B)
mu, sigma = flat.mean(0), flat.std(0) + 1e-8
X_all = ((flat - mu) / sigma).astype(np.float32)          # one spectrum per pixel (row-major, same order as gt_flat)
class_num = int(gt.max())
gt_flat = gt.reshape(-1)
print("dataset %s: cube %s, %d classes" % (args.dataset, cube.shape, class_num), flush=True)


# ---------------- split: copied verbatim from train_dofa.py ----------------
def make_split(gt_flat, class_num, train_num, val_num, rng):
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


def scores(yt, pt):
    oa = accuracy_score(yt, pt)
    kappa = cohen_kappa_score(yt, pt)
    cm = confusion_matrix(yt, pt, labels=list(range(1, class_num + 1)))
    aa = (cm.diagonal() / np.maximum(cm.sum(1), 1)).mean()
    return oa, aa, kappa


def done(method, seed):
    if not os.path.exists(csv):
        return False
    key = "%s,%d,%d,0," % (method, args.train_num, seed)
    return any(l.startswith(key) for l in open(csv))


def write(method, seed, oa, aa, kappa, t):
    new = not os.path.exists(csv)
    with open(csv, "a") as f:
        if new:
            f.write("method,train_num,seed,lora_r,OA,AA,kappa,trainable,trainable_pct,best_epoch,epochs_run,train_time_s\n")
        f.write("%s,%d,%d,0,%.4f,%.4f,%.4f,0,0.0000,-1,-1,%.1f\n" % (method, args.train_num, seed, oa, aa, kappa, t))


# hyper-parameter grids (chosen on the validation pixels only)
SVM_GRID = [(C, g / B) for C in (1, 10, 100, 1000, 10000) for g in (0.1, 0.3, 1, 3, 10)]   # gamma relative to 1/B ("scale")
RF_GRID = ["sqrt", 0.1, 0.3]

for seed in range(lo, hi + 1):
    rng = np.random.RandomState(seed)
    tr, va, te = make_split(gt_flat, class_num, args.train_num, args.val_num, rng)
    Xtr, ytr = X_all[tr], gt_flat[tr].astype(int)
    Xva, yva = X_all[va], gt_flat[va].astype(int)
    Xte, yte = X_all[te], gt_flat[te].astype(int)

    if not done("svm", seed):
        t0 = time.time()
        best = None
        for C, g in SVM_GRID:
            acc = accuracy_score(yva, SVC(C=C, gamma=g, kernel="rbf").fit(Xtr, ytr).predict(Xva))
            if best is None or acc > best[0]:
                best = (acc, C, g)
        m = SVC(C=best[1], gamma=best[2], kernel="rbf").fit(Xtr, ytr)
        oa, aa, k = scores(yte, m.predict(Xte))
        write("svm", seed, oa, aa, k, time.time() - t0)
        print("seed %2d  SVM  OA %.4f  AA %.4f  kappa %.4f   (C=%g, gamma=%.2g/B, val %.3f)   train %d / test %d" % (
            seed, oa, aa, k, best[1], best[2] * B, best[0], len(tr), len(te)), flush=True)

    if not done("rf", seed):
        t0 = time.time()
        best = None
        for mf in RF_GRID:
            acc = accuracy_score(yva, RandomForestClassifier(n_estimators=500, max_features=mf, random_state=seed,
                                                             n_jobs=args.n_jobs).fit(Xtr, ytr).predict(Xva))
            if best is None or acc > best[0]:
                best = (acc, mf)
        m = RandomForestClassifier(n_estimators=500, max_features=best[1], random_state=seed, n_jobs=args.n_jobs).fit(Xtr, ytr)
        oa, aa, k = scores(yte, m.predict(Xte))
        write("rf", seed, oa, aa, k, time.time() - t0)
        print("seed %2d  RF   OA %.4f  AA %.4f  kappa %.4f   (max_features=%s, val %.3f)" % (
            seed, oa, aa, k, best[1], best[0]), flush=True)
print("CLASSIC_DONE %s -> %s" % (args.dataset, csv), flush=True)
