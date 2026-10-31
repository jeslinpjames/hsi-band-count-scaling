"""data_common.py : data loading shared by the new scripts (same files and preprocessing as the DOFA runs)."""
import importlib
import numpy as np
import scipy.io as sio

DATASETS = {   # cube file, key, gt file, key, wavelength module (in exp/), bandwidth in nm (approximate)
    "ip":      ("../repos/HyperSIGMA/ImageClassification/data/Indian_pines_corrected.mat", "data",
                "../repos/HyperSIGMA/ImageClassification/data/Indian_pines_gt.mat", "groundT", "indian_pines_wavelengths", 10.0),
    "salinas": ("../data/Salinas/Salinas_corrected.mat", "salinas_corrected",
                "../data/Salinas/Salinas_gt.mat", "salinas_gt", "salinas_wavelengths", 10.0),
    "pavia":   ("../data/PaviaU/paviaU.mat", "ori_data", "../data/PaviaU/paviaU_7gt.mat", "map", "pavia_wavelengths", 4.0),
}


def load(ds):
    """Returns per-band standardised cube (H, W, B) float32, gt (H, W) int, wavelengths in um (list), bandwidth nm."""
    fc, kc, fg, kg, wmod, bw = DATASETS[ds]
    cube = sio.loadmat(fc)[kc].astype(np.float32)
    gt = sio.loadmat(fg)[kg].astype(np.int64)
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    cube = ((flat - flat.mean(0)) / (flat.std(0) + 1e-8)).reshape(H, W, B).astype(np.float32)
    wl = list(importlib.import_module(wmod).wavelengths_um)
    assert len(wl) == B, (ds, len(wl), B)
    return cube, gt, wl, bw


class Patches:
    """Zero-padded square windows around flat pixel indices, cut on the fly (low memory)."""
    def __init__(self, cube, size):
        self.r = size // 2; self.size = size; self.W = cube.shape[1]
        self.pad = np.pad(cube, ((self.r, self.r), (self.r, self.r), (0, 0)), mode="constant")

    def __call__(self, idx):
        rows, cols = np.asarray(idx) // self.W, np.asarray(idx) % self.W
        return np.stack([self.pad[i:i + self.size, j:j + self.size] for i, j in zip(rows, cols)])   # (n, s, s, B)


def scores(yt, pt, class_num):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    cm = confusion_matrix(yt, pt, labels=list(range(1, class_num + 1)))
    return accuracy_score(yt, pt), (cm.diagonal() / np.maximum(cm.sum(1), 1)).mean(), cohen_kappa_score(yt, pt)


HEADER = "method,train_num,seed,lora_r,OA,AA,kappa,trainable,trainable_pct,best_epoch,epochs_run,train_time_s\n"


def done(csv, method, train_num, seed, r=0):
    import os
    key = "%s,%d,%d,%d," % (method, train_num, seed, r)
    return os.path.exists(csv) and any(l.startswith(key) for l in open(csv))


def write(csv, method, train_num, seed, r, oa, aa, kappa, trainable, pct, best_ep, ep_run, t):
    import os
    new = not os.path.exists(csv)
    with open(csv, "a") as f:
        if new:
            f.write(HEADER)
        f.write("%s,%d,%d,%d,%.4f,%.4f,%.4f,%d,%.4f,%d,%d,%.1f\n" % (method, train_num, seed, r, oa, aa, kappa, trainable, pct,
                                                                      best_ep, ep_run, t))
