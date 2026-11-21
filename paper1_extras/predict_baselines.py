"""Paper 1 (TGRS) extras: SVM (+ 33 x 33 mean spectrum) and SSRN on one split, saving the prediction for EVERY
labelled pixel (train + validation + test) for classification maps and per-class accuracies.

Same features, hyper-parameter search, network and training as scripts/classic_baselines.py (svm_sp33) and
scripts/deep_baselines.py (ssrn). Output: results/paper1_extras/pred_<svm_sp33|ssrn>_<scene>_s<seed>.npz

  python paper1_extras/predict_baselines.py --dataset ip --seed 0
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", required=True, choices=["ip", "salinas", "pavia"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--methods", default="svm_sp33,ssrn")
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "paper1_extras"))
args = p.parse_args()
try:
    from _gpu import pick_free_gpu
    pick_free_gpu()
except ImportError:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from bcs.data import Windows, append_row, load, scores
from bcs.splits import make_split

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out_dir, exist_ok=True)

loaded = load(args.dataset)
cube, gt = loaded[0], loaded[1]
H, W, B = cube.shape
gt_flat = gt.reshape(-1).astype(np.int64)
C = int(gt.max())
tr, va, te = make_split("random", gt, C, args.train_num, args.val_num, args.seed)
all_idx = np.concatenate([tr, va, te])


def save(name, all_pred, t, n_par, best_epoch, best_val):
    yt = gt_flat[te]
    te_pred = all_pred[len(tr) + len(va):]
    oa, aa, kappa = scores(yt, te_pred, C)
    per_class = np.array([(te_pred[yt == c] == c).mean() for c in range(1, C + 1)])
    out = os.path.join(args.out_dir, "pred_%s_%s_s%d.npz" % (name, args.dataset, args.seed))
    np.savez_compressed(out, idx=all_idx, pred=np.asarray(all_pred).astype(np.int16), y=gt_flat[all_idx].astype(np.int16), n_train=len(tr), n_val=len(va),
                        shape=np.array(gt.shape), oa=oa, aa=aa, kappa=kappa, per_class=per_class,
                        best_epoch=best_epoch, train_time=t, trainable=n_par)
    append_row(os.path.join(args.out_dir, "runs_baselines.csv"), "%s_%s" % (args.dataset, name), args.train_num,
               args.seed, 0, oa, aa, kappa, n_par, 0.0, best_epoch, -1, t, best_val=best_val)
    print("RESULT %s %s seed %d: OA %.4f AA %.4f kappa %.4f -> %s" % (args.dataset, name, args.seed, oa, aa, kappa, out),
          flush=True)


def done(name):
    return os.path.exists(os.path.join(args.out_dir, "pred_%s_%s_s%d.npz" % (name, args.dataset, args.seed)))


# ------------------------------------------------------------------ SVM on [spectrum, 33 x 33 mean spectrum]
if "svm_sp33" in args.methods and not done("svm_sp33"):
    spec = cube.reshape(-1, B)
    X = np.concatenate([spec, uniform_filter(cube, size=(33, 33, 1), mode="reflect").reshape(-1, B)], 1).astype(np.float32)
    ytr, yva = gt_flat[tr], gt_flat[va]
    t0, best = time.time(), None
    D = X.shape[1]
    for c_ in (1, 10, 100, 1000, 10000):
        for g in (0.1, 0.3, 1, 3, 10):
            acc = accuracy_score(yva, SVC(C=c_, gamma=g / D, kernel="rbf").fit(X[tr], ytr).predict(X[va]))
            if best is None or acc > best[0]:
                best = (acc, c_, g / D)
    clf = SVC(C=best[1], gamma=best[2], kernel="rbf").fit(X[tr], ytr)
    save("svm_sp33", clf.predict(X[all_idx]), time.time() - t0, 0, -1, best[0])


# ------------------------------------------------------------------ SSRN (copy of scripts/deep_baselines.py)
class ResBlock(nn.Module):
    def __init__(self, ch, k, pad):
        super().__init__()
        self.a = nn.Sequential(nn.Conv3d(ch, ch, k, padding=pad), nn.BatchNorm3d(ch), nn.ReLU(),
                               nn.Conv3d(ch, ch, k, padding=pad), nn.BatchNorm3d(ch))

    def forward(self, x):
        return F.relu(x + self.a(x))


class SSRN(nn.Module):
    def __init__(self, bands, n_cls, size):
        super().__init__()
        d = (bands - 7) // 2 + 1
        self.spec = nn.Sequential(nn.Conv3d(1, 24, (7, 1, 1), stride=(2, 1, 1)), nn.BatchNorm3d(24), nn.ReLU(),
                                  ResBlock(24, (7, 1, 1), (3, 0, 0)), ResBlock(24, (7, 1, 1), (3, 0, 0)),
                                  nn.Conv3d(24, 128, (d, 1, 1)), nn.BatchNorm3d(128), nn.ReLU())
        self.spat = nn.Sequential(nn.Conv3d(1, 24, (128, 3, 3)), nn.BatchNorm3d(24), nn.ReLU(),
                                  ResBlock(24, (1, 3, 3), (0, 1, 1)), ResBlock(24, (1, 3, 3), (0, 1, 1)))
        self.head = nn.Linear(24, n_cls)

    def forward(self, x):
        x = self.spec(x.unsqueeze(1))
        x = self.spat(x.transpose(1, 2))
        return self.head(x.mean(dim=(2, 3, 4)))


if "ssrn" in args.methods and not done("ssrn"):
    size = 7
    cut = Windows(cube, size)

    def predict(model, idx, bs=512):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                x = torch.from_numpy(cut(idx[i:i + bs])).permute(0, 3, 1, 2).to(dev)
                out.append(model(x).argmax(1).cpu().numpy() + 1)
        return np.concatenate(out)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SSRN(B, C, size).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = torch.from_numpy(cut(tr)).permute(0, 3, 1, 2)
    ytr = torch.from_numpy(gt_flat[tr] - 1)
    best, best_ep, best_sd, t0 = -1, 0, None, time.time()
    for ep in range(300):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 16):
            b = perm[i:i + 16]
            x = Xtr[b]
            if np.random.rand() < 0.5:
                x = x.flip(3)
            if np.random.rand() < 0.5:
                x = x.flip(2)
            x = torch.rot90(x, int(np.random.randint(4)), (2, 3))
            if len(b) < 2:
                continue
            loss = F.cross_entropy(model(x.to(dev)), ytr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = (predict(model, va) == gt_flat[va]).mean()
        if acc > best:
            best, best_ep = acc, ep
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= 50:
            break
    model.load_state_dict(best_sd)
    n_par = sum(q.numel() for q in model.parameters())
    save("ssrn", predict(model, all_idx), time.time() - t0, n_par, best_ep, best)
