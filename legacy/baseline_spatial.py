"""baseline_spatial.py : spectral-SPATIAL classic baseline. SVM-RBF on [pixel spectrum, mean spectrum of its
k x k neighbourhood] for k = 7 and k = 33 (33 = the same window DOFA sees). Everything else is identical to
baseline_classic.py: same split code, same RandomState(seed), same per-band standardisation, same
validation-based hyper-parameter choice, final fit on training pixels only.

usage:  python baseline_spatial.py --dataset ip|salinas|pavia --seeds 0-15
writes: results_classic_<dataset>.csv  (methods svm_sp7, svm_sp33; same columns as the DOFA CSVs)
"""
import argparse, os, time
import numpy as np
import scipy.io as sio
from scipy.ndimage import uniform_filter
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

DATASETS = {
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
p.add_argument("--windows", default="7,33")
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--csv", default=None)
args = p.parse_args()
lo, hi = (int(x) for x in args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds),) * 2
csv = args.csv or "results_classic_%s.csv" % args.dataset
WINDOWS = [int(w) for w in args.windows.split(",")]

fc, kc, fg, kg = DATASETS[args.dataset]
cube = sio.loadmat(fc)[kc].astype(np.float32)
gt = sio.loadmat(fg)[kg].astype(np.float32)
H, W, B = cube.shape
flat = cube.reshape(-1, B)
mu, sigma = flat.mean(0), flat.std(0) + 1e-8
cube = ((flat - mu) / sigma).reshape(H, W, B).astype(np.float32)   # same per-band standardisation as DOFA
class_num = int(gt.max())
gt_flat = gt.reshape(-1)
FEATS = {}
for k in WINDOWS:   # [own spectrum, k x k neighbourhood mean spectrum]; image borders mirrored
    mean_k = uniform_filter(cube, size=(k, k, 1), mode="reflect").reshape(-1, B)
    FEATS[k] = np.concatenate([cube.reshape(-1, B), mean_k], axis=1).astype(np.float32)
print("dataset %s: cube %s, %d classes, windows %s" % (args.dataset, cube.shape, class_num, WINDOWS), flush=True)


def make_split(gt_flat, class_num, train_num, val_num, rng):     # verbatim from train_dofa.py
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
    cm = confusion_matrix(yt, pt, labels=list(range(1, class_num + 1)))
    return accuracy_score(yt, pt), (cm.diagonal() / np.maximum(cm.sum(1), 1)).mean(), cohen_kappa_score(yt, pt)


def done(method, seed):
    key = "%s,%d,%d,0," % (method, args.train_num, seed)
    return os.path.exists(csv) and any(l.startswith(key) for l in open(csv))


def write(method, seed, oa, aa, kappa, t):
    new = not os.path.exists(csv)
    with open(csv, "a") as f:
        if new:
            f.write("method,train_num,seed,lora_r,OA,AA,kappa,trainable,trainable_pct,best_epoch,epochs_run,train_time_s\n")
        f.write("%s,%d,%d,0,%.4f,%.4f,%.4f,0,0.0000,-1,-1,%.1f\n" % (method, args.train_num, seed, oa, aa, kappa, t))


D = 2 * B
GRID = [(C, g / D) for C in (1, 10, 100, 1000, 10000) for g in (0.1, 0.3, 1, 3, 10)]
for seed in range(lo, hi + 1):
    tr, va, te = make_split(gt_flat, class_num, args.train_num, args.val_num, np.random.RandomState(seed))
    ytr, yva, yte = (gt_flat[i].astype(int) for i in (tr, va, te))
    for k in WINDOWS:
        name = "svm_sp%d" % k
        if done(name, seed):
            continue
        X = FEATS[k]
        t0 = time.time()
        best = None
        for C, g in GRID:
            acc = accuracy_score(yva, SVC(C=C, gamma=g).fit(X[tr], ytr).predict(X[va]))
            if best is None or acc > best[0]:
                best = (acc, C, g)
        oa, aa, kap = scores(yte, SVC(C=best[1], gamma=best[2]).fit(X[tr], ytr).predict(X[te]))
        write(name, seed, oa, aa, kap, time.time() - t0)
        print("seed %2d  %-9s OA %.4f  AA %.4f  kappa %.4f  (C=%g, gamma=%.2g/D, val %.3f)  test %d" % (
            seed, name, oa, aa, kap, best[1], best[2] * D, best[0], len(te)), flush=True)
print("SPATIAL_DONE %s -> %s" % (args.dataset, csv), flush=True)
