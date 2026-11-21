"""Inference throughput: DOFA (Generic LoRA, no correction / BCR / GTN), SSRN and the SVM on the same pixels.

Measures pixels per second for classifying labelled pixels of Indian Pines (200 bands) and Pavia (102 bands):
DOFA and SSRN on one GPU (batch 64 and 512, after warm-up, median of 5 repetitions), the SVM on 8 CPU threads.
Untrained weights are used: throughput does not depend on the weights. Output: results/paper1_analysis/throughput.csv

  python paper1_analysis/throughput.py
"""
import os
import time

from common import OUT, Windows, corrections, dofa, import_dofa, load_scene

out_csv = os.path.join(OUT, "throughput.csv")
if os.path.exists(out_csv):
    print("already done:", out_csv)
    raise SystemExit
import_dofa()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from sklearn.svm import SVC

torch.manual_seed(0)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
rows = ["scene,method,correction,batch,pixels_per_s,ms_per_pixel"]


class ResBlock(nn.Module):
    def __init__(self, ch, k, pad):
        super().__init__()
        self.a = nn.Sequential(nn.Conv3d(ch, ch, k, padding=pad), nn.BatchNorm3d(ch), nn.ReLU(),
                               nn.Conv3d(ch, ch, k, padding=pad), nn.BatchNorm3d(ch))

    def forward(self, x):
        return F.relu(x + self.a(x))


class SSRN(nn.Module):
    def __init__(self, bands, n_cls):
        super().__init__()
        d = (bands - 7) // 2 + 1
        self.spec = nn.Sequential(nn.Conv3d(1, 24, (7, 1, 1), stride=(2, 1, 1)), nn.BatchNorm3d(24), nn.ReLU(),
                                  ResBlock(24, (7, 1, 1), (3, 0, 0)), ResBlock(24, (7, 1, 1), (3, 0, 0)),
                                  nn.Conv3d(24, 128, (d, 1, 1)), nn.BatchNorm3d(128), nn.ReLU())
        self.spat = nn.Sequential(nn.Conv3d(1, 24, (128, 3, 3)), nn.BatchNorm3d(24), nn.ReLU(),
                                  ResBlock(24, (1, 3, 3), (0, 1, 1)), ResBlock(24, (1, 3, 3), (0, 1, 1)))
        self.head = nn.Linear(24, n_cls)

    def forward(self, x):
        return self.head(self.spat(self.spec(x.unsqueeze(1)).transpose(1, 2)).mean(dim=(2, 3, 4)))


def timed(fn, n_pix, reps=5):
    fn(); sync()                                           # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    t = float(np.median(ts))
    return n_pix / t, 1000 * t / n_pix


for scene in ("ip", "pavia"):
    cube, gt, wl = load_scene(scene)
    C = int(gt.max())
    lab = np.where(gt.reshape(-1) > 0)[0]
    pix = np.random.RandomState(0).choice(lab, min(512, len(lab)), replace=False)
    # DOFA
    r = dofa.PAPER["generic"]["r"]
    model = dofa.adapt(dofa.build(C + 1, "base"), "generic", r, 2 * r).to(dev).eval()
    X33 = torch.tensor(Windows(cube, 33)(pix[:64]), dtype=torch.float32).permute(0, 3, 1, 2).to(dev)
    for corr in ("none", "bcr", "gtn"):
        corrections.patch_dofa(corr)

        def run_dofa():
            with torch.no_grad():
                model(F.interpolate(X33, size=(224, 224), mode="bilinear", align_corners=False), wave_list=wl)
        pps, ms = timed(run_dofa, 64)
        rows.append("%s,dofa_generic,%s,64,%.1f,%.4f" % (scene, corr, pps, ms))
        print(rows[-1], flush=True)
    corrections.patch_dofa("none")
    del model
    # SSRN
    net = SSRN(cube.shape[2], C).to(dev).eval()
    X7 = torch.tensor(Windows(cube, 7)(pix), dtype=torch.float32).permute(0, 3, 1, 2).to(dev)

    def run_ssrn():
        with torch.no_grad():
            net(X7)
    pps, ms = timed(run_ssrn, len(pix))
    rows.append("%s,ssrn,-,%d,%.1f,%.4f" % (scene, len(pix), pps, ms))
    print(rows[-1], flush=True)
    # SVM (+33x33 mean), fitted on 160 random labelled pixels, prediction of 5000 pixels on the CPU
    B = cube.shape[2]
    Xs = np.concatenate([cube.reshape(-1, B), uniform_filter(cube, size=(33, 33, 1), mode="reflect").reshape(-1, B)], 1)
    rs = np.random.RandomState(0)
    fit_idx = rs.choice(lab, 160, replace=False)
    svm = SVC(C=100, gamma=1.0 / Xs.shape[1]).fit(Xs[fit_idx], gt.reshape(-1)[fit_idx].astype(int))
    pred_idx = rs.choice(lab, min(5000, len(lab)), replace=False)
    pps, ms = timed(lambda: svm.predict(Xs[pred_idx]), len(pred_idx), reps=3)
    rows.append("%s,svm_sp33,-,%d,%.1f,%.4f" % (scene, len(pred_idx), pps, ms))
    print(rows[-1], flush=True)
open(out_csv, "w").write("\n".join(rows) + "\n")
print("RESULT written", out_csv)
