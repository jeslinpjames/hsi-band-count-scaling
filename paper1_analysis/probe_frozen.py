"""Frozen-feature probe: does a correction improve the PRETRAINED representation itself (no fine-tuning at all)?

For one scene and each correction (none, bcr, gtn), it extracts the features entering DOFA's head for every labelled
pixel (all bands, 33x33 window upsampled to 224, as in training), then for splits 0-7 (the paper's splits, 10 training
and 5 validation pixels per class) fits
  * logistic regression on standardised features (C chosen on the validation pixels),
  * k-nearest neighbours (k chosen on the validation pixels, cosine distance),
and evaluates on the test pixels.
Output: results/paper1_analysis/probe_<scene>.csv and feat_<scene>_<corr>.npy (float16 features)

  python paper1_analysis/probe_frozen.py --dataset ip
"""
import argparse
import os

from common import OUT, Windows, corrections, dofa, first_tensor, head_of, import_dofa, load_scene, make_split

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--seeds", default="0-7")
p.add_argument("--bs", type=int, default=64)
args = p.parse_args()
out_csv = os.path.join(OUT, "probe_%s.csv" % args.dataset)
if os.path.exists(out_csv):
    print("already done:", out_csv)
    raise SystemExit
import_dofa()
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from bcs.data import scores
torch.manual_seed(0)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cube, gt, wl = load_scene(args.dataset)
C = int(gt.max())
gt_flat = gt.reshape(-1).astype(int)
lab = np.where(gt_flat > 0)[0]
win = Windows(cube, 33)
model = dofa.build(C + 1, "base").to(dev).eval()
head = head_of(model)
feats = {}
store = []
head.register_forward_pre_hook(lambda m, i: store.append(first_tensor(i).float().cpu().numpy()))

for corr in ("none", "bcr", "gtn"):
    fpath = os.path.join(OUT, "feat_%s_%s.npy" % (args.dataset, corr))
    if os.path.exists(fpath):
        feats[corr] = np.load(fpath).astype(np.float32)
        continue
    corrections.patch_dofa(corr)
    store.clear()
    with torch.no_grad():
        for i in range(0, len(lab), 2048):
            X = torch.tensor(win(lab[i:i + 2048]), dtype=torch.float32).permute(0, 3, 1, 2)
            for j in range(0, len(X), args.bs):
                model(F.interpolate(X[j:j + args.bs], size=(224, 224), mode="bilinear", align_corners=False).to(dev),
                      wave_list=wl)
            print("  %s %s features %d / %d" % (args.dataset, corr, min(i + 2048, len(lab)), len(lab)), flush=True)
    feats[corr] = np.concatenate(store)
    np.save(fpath, feats[corr].astype(np.float16))
corrections.patch_dofa("none")

pos = {v: k for k, v in enumerate(lab)}                     # flat pixel index -> row in the feature matrix
lo, hi = (int(v) for v in args.seeds.split("-"))
rows = ["correction,classifier,seed,OA,AA,kappa,param"]
for seed in range(lo, hi + 1):
    tr, va, te = make_split("random", gt, C, 10, 5, seed)
    r_tr, r_va, r_te = ([pos[i] for i in s] for s in (tr, va, te))
    ytr, yva, yte = gt_flat[tr], gt_flat[va], gt_flat[te]
    for corr, Fm in feats.items():
        sc = StandardScaler().fit(Fm[r_tr])
        Xtr, Xva, Xte = sc.transform(Fm[r_tr]), sc.transform(Fm[r_va]), sc.transform(Fm[r_te])
        best = max(((accuracy_score(yva, LogisticRegression(C=c, max_iter=3000).fit(Xtr, ytr).predict(Xva)), c)
                    for c in (0.001, 0.01, 0.1, 1, 10)))
        pred = LogisticRegression(C=best[1], max_iter=3000).fit(Xtr, ytr).predict(Xte)
        oa, aa, k = scores(yte, pred, C)
        rows.append("%s,logreg,%d,%.4f,%.4f,%.4f,%g" % (corr, seed, oa, aa, k, best[1]))
        bestk = max(((accuracy_score(yva, KNeighborsClassifier(n_neighbors=kk, metric="cosine").fit(Fm[r_tr], ytr)
                                     .predict(Fm[r_va])), kk) for kk in (1, 3, 5)))
        pred = KNeighborsClassifier(n_neighbors=bestk[1], metric="cosine").fit(Fm[r_tr], ytr).predict(Fm[r_te])
        oa2, aa2, k2 = scores(yte, pred, C)
        rows.append("%s,knn,%d,%.4f,%.4f,%.4f,%d" % (corr, seed, oa2, aa2, k2, bestk[1]))
        print("seed %d %-4s logreg OA %.4f  knn OA %.4f" % (seed, corr, oa, oa2), flush=True)
open(out_csv, "w").write("\n".join(rows) + "\n")
print("RESULT written", out_csv)
