"""Paper 1 (TGRS) extras: train DOFA on one split exactly as scripts/train_dofa.py does, then save the prediction for
EVERY labelled pixel (train + validation + test) so that classification maps and per-class accuracies can be made.

Same seeding, split, model, data loading, training loop and early stopping as scripts/train_dofa.py (the code below is
a copy of it with a prediction step added at the end). Output:
  results/paper1_extras/pred_dofa_<scene>_<method>_<correction>_s<seed>.npz
  results/paper1_extras/runs_dofa.csv  (one row per run, same format as the paper's result files)

  python paper1_extras/train_predict_dofa.py --dataset ip --correction gtn --seed 0
"""
import argparse
import copy
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from bcs import corrections, dofa
from bcs.data import Windows, append_row, load, scores
from bcs.splits import make_split

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", required=True, choices=["ip", "salinas", "pavia"])
p.add_argument("--correction", default="none", choices=corrections.CHOICES)
p.add_argument("--method", default="generic", choices=dofa.METHODS)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--max_epoch", type=int, default=400)
p.add_argument("--patience", type=int, default=100)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--img_size", type=int, default=33)
p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "paper1_extras"))
args = p.parse_args()
cfg = dofa.PAPER[args.method]
lora_r, lora_alpha, lr = cfg["r"], 2 * cfg["r"], cfg["lr"]
os.makedirs(args.out_dir, exist_ok=True)
out_npz = os.path.join(args.out_dir, "pred_dofa_%s_%s_%s_s%d.npz" % (args.dataset, args.method, args.correction, args.seed))
if os.path.exists(out_npz):
    print("already done: %s" % out_npz)
    sys.exit(0)
print(vars(args), "lr", lr, "r", lora_r, flush=True)

# DOFA's wave_dynamic_layer.py calls torch.manual_seed() at import: import it BEFORE seeding (as train_dofa.py does)
dofa.add_dofa_to_path()
import dofa_v1  # noqa: E402,F401
import wave_dynamic_layer  # noqa: E402,F401

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
corrections.patch_dofa(args.correction, corrections.R_DEFAULT)

loaded = load(args.dataset, "all")
cube, gt, wavelengths_um = loaded[0], loaded[1], loaded[2]
class_num = int(gt.max())
tr_idx, va_idx, te_idx = make_split("random", gt, class_num, args.train_num, args.val_num, args.seed)
gt_flat = gt.reshape(-1)
windows = Windows(cube, args.img_size)
print("%s: cube %s, %d classes, train %d val %d test %d" % (args.dataset, cube.shape, class_num, len(tr_idx),
                                                              len(va_idx), len(te_idx)), flush=True)


def gather(idx):
    x = torch.tensor(windows(idx), dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(gt_flat[idx].astype(np.int64), dtype=torch.long)
    return x, y


train_x, train_y = gather(tr_idx)
val_x, val_y = gather(va_idx)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                           batch_size=args.batch_size, shuffle=True, drop_last=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_x, val_y), batch_size=64)

model = dofa.build(class_num + 1, "base")
model = dofa.adapt(model, args.method, lora_r, lora_alpha)
n_train_p = sum(q.numel() for q in model.parameters() if q.requires_grad)
n_all_p = sum(q.numel() for q in model.parameters())
model = model.to(device)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.05)
crit = nn.CrossEntropyLoss()


def prep(x):
    return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


def augment(x):
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k, dims=[2, 3])
    return x


@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        out = model(prep(x).to(device), wave_list=wavelengths_um)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return np.array(trues), np.array(preds)


@torch.no_grad()
def predict_idx(idx, chunk=2048, bs=64):
    """Predictions for flat pixel indices, cutting windows chunk by chunk (keeps memory small)."""
    model.eval()
    out = []
    for i in range(0, len(idx), chunk):
        x = torch.tensor(windows(idx[i:i + chunk]), dtype=torch.float32).permute(0, 3, 1, 2)
        for j in range(0, len(x), bs):
            out.append(model(prep(x[j:j + bs]).to(device), wave_list=wavelengths_um).argmax(1).cpu().numpy())
        print("  predicted %d / %d" % (min(i + chunk, len(idx)), len(idx)), flush=True)
    return np.concatenate(out)


best_val, best_state, best_epoch, bad = -1.0, None, -1, 0
t0 = time.time()
for epoch in range(args.max_epoch + 1):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x = augment(x)
        x, y = prep(x).to(device), y.to(device)
        opt.zero_grad()
        out = model(x, wave_list=wavelengths_um)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    yv, pv = evaluate(val_loader)
    val_acc = accuracy_score(yv, pv)
    if val_acc > best_val:
        best_val, best_epoch, bad = val_acc, epoch, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        bad += 1
    if epoch % 10 == 0:
        print("epoch %4d loss %.5f train_acc %.4f val_acc %.4f best %.4f@%d" % (
            epoch, tot_loss / total, correct / total, val_acc, best_val, best_epoch), flush=True)
    if bad >= args.patience:
        print("early stop at epoch %d" % epoch, flush=True)
        break
train_time = time.time() - t0
model.load_state_dict(best_state)

print("predicting all labelled pixels", flush=True)
all_idx = np.concatenate([tr_idx, va_idx, te_idx])
all_pred = predict_idx(all_idx)
n_tr, n_va = len(tr_idx), len(va_idx)
te_pred = all_pred[n_tr + n_va:]
yt = gt_flat[te_idx].astype(np.int64)
oa, aa, kappa = scores(yt, te_pred, class_num)
per_class = np.array([(te_pred[yt == c] == c).mean() if (yt == c).any() else np.nan for c in range(1, class_num + 1)])
np.savez_compressed(out_npz, idx=all_idx, pred=all_pred.astype(np.int16), y=gt_flat[all_idx].astype(np.int16), n_train=n_tr, n_val=n_va,
                    shape=np.array(gt.shape), oa=oa, aa=aa, kappa=kappa, per_class=per_class,
                    best_epoch=best_epoch, train_time=train_time, trainable=n_train_p)
append_row(os.path.join(args.out_dir, "runs_dofa.csv"), "%s_%s_%s" % (args.dataset, args.method, args.correction),
           args.train_num, args.seed, lora_r, oa, aa, kappa, n_train_p, 100 * n_train_p / n_all_p, best_epoch, epoch,
           train_time, best_val=best_val)
print("RESULT %s %s %s seed %d: OA %.4f AA %.4f kappa %.4f (best epoch %d) -> %s" % (
    args.dataset, args.method, args.correction, args.seed, oa, aa, kappa, best_epoch, out_npz), flush=True)
