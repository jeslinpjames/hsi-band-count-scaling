"""Deep hyperspectral classifiers trained from scratch on exactly the same splits as the foundation-model runs.

  hybridsn  Roy et al., IEEE GRSL 2020: PCA to 30 components, 25x25 window, 3-D + 2-D convolutions
  ssrn      Zhong et al., IEEE TGRS 2018: all bands, 7x7 window, spectral then spatial residual 3-D blocks

Training (both): Adam lr 1e-3, batch 16, random flips + 90-degree rotations, at most 300 epochs, early stopping on
validation OA with patience 50, best-validation weights evaluated on the test set.

  python scripts/deep_baselines.py --model ssrn --dataset ip --seeds 0-15 --csv results/new/deep_random_ip.csv
  python scripts/deep_baselines.py --model hybridsn --dataset pavia --split separated --seeds 0-7 --csv ...
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _gpu import pick_free_gpu

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True, choices=["hybridsn", "ssrn"])
p.add_argument("--dataset", required=True, choices=["ip", "salinas", "pavia"])
p.add_argument("--seeds", default="0-15")
p.add_argument("--split", default="random", choices=["random", "separated"])
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--patience", type=int, default=50)
p.add_argument("--csv", required=True)
args = p.parse_args()
pick_free_gpu()
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from bcs.data import load, Windows, scores, is_done, append_row
from bcs.splits import make_split
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HybridSN(nn.Module):
    def __init__(self, bands, n_cls, size):
        super().__init__()
        self.c3 = nn.Sequential(nn.Conv3d(1, 8, (7, 3, 3)), nn.ReLU(), nn.Conv3d(8, 16, (5, 3, 3)), nn.ReLU(),
                                nn.Conv3d(16, 32, (3, 3, 3)), nn.ReLU())
        d, s = bands - 12, size - 6
        self.c2 = nn.Sequential(nn.Conv2d(32 * d, 64, 3), nn.ReLU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * (s - 2) ** 2, 256), nn.ReLU(), nn.Dropout(0.4),
                                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, n_cls))

    def forward(self, x):                       # x: (n, B, s, s)
        x = self.c3(x.unsqueeze(1))             # (n, 32, B-12, s-6, s-6)
        return self.fc(self.c2(x.flatten(1, 2)))


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

    def forward(self, x):                       # x: (n, B, s, s)
        x = self.spec(x.unsqueeze(1))           # (n, 128, 1, s, s)
        x = self.spat(x.transpose(1, 2))        # (n, 24, 1, s-2, s-2)
        return self.head(x.mean(dim=(2, 3, 4)))


cube, gt, _, _ = load(args.dataset)
H, W, B = cube.shape
if args.model == "hybridsn":
    size, n_pc = 25, 30
    feat = PCA(n_components=n_pc, random_state=0).fit_transform(cube.reshape(-1, B)).reshape(H, W, n_pc).astype(np.float32)
else:
    size, feat = 7, cube
cut = Windows(feat, size)
gt_flat = gt.reshape(-1).astype(np.int64); C = int(gt.max())
csv = args.csv
lo, hi = (int(x) for x in args.seeds.split("-"))
print("%s on %s (%s split): input %s, window %d, device %s" % (args.model, args.dataset, args.split, feat.shape, size, dev), flush=True)


def predict(model, idx, bs=512):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x = torch.from_numpy(cut(idx[i:i + bs])).permute(0, 3, 1, 2).to(dev)
            out.append(model(x).argmax(1).cpu().numpy() + 1)
    return np.concatenate(out)


for seed in range(lo, hi + 1):
    if is_done(csv, args.model, args.train_num, seed):
        continue
    tr, va, te = make_split(args.split, gt, C, args.train_num, args.val_num, seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = (HybridSN if args.model == "hybridsn" else SSRN)(feat.shape[2], C, size).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = torch.from_numpy(cut(tr)).permute(0, 3, 1, 2); ytr = torch.from_numpy(gt_flat[tr] - 1)
    best, best_ep, best_sd, t0, ep = -1, 0, None, time.time(), 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 16):
            b = perm[i:i + 16]
            x = Xtr[b]
            if np.random.rand() < 0.5: x = x.flip(3)
            if np.random.rand() < 0.5: x = x.flip(2)
            x = torch.rot90(x, int(np.random.randint(4)), (2, 3))
            if len(b) < 2:                       # BatchNorm needs > 1 sample
                continue
            loss = F.cross_entropy(model(x.to(dev)), ytr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        acc = (predict(model, va) == gt_flat[va]).mean()
        if acc > best:
            best, best_ep = acc, ep
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= args.patience:
            break
    model.load_state_dict(best_sd)
    oa, aa, kap = scores(gt_flat[te], predict(model, te), C)
    n_par = sum(p.numel() for p in model.parameters())
    append_row(csv, args.model, args.train_num, seed, 0, oa, aa, kap, n_par, 100.0, best_ep, ep + 1, time.time() - t0, best_val=best)
    print("seed %2d  %-8s OA %.4f AA %.4f kappa %.4f  (best epoch %d, val %.3f, %d params, %.0fs, test %d)" % (
        seed, args.model, oa, aa, kap, best_ep, best, n_par, time.time() - t0, len(te)), flush=True)
