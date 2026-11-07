"""Fine-tune DOFA on one split of one scene and append one row to a results CSV.

This single script replaces the per-scene copies used for the paper (legacy/train_dofa*.py); for the same arguments
it performs the same computation in the same order (seeding, split, model construction, data loading, training
loop, early stopping, evaluation). See docs/REPRODUCE.md for the command behind every result.

examples
  # Indian Pines, Generic LoRA, gated token normalisation, split 0 (paper setting)
  python scripts/train_dofa.py --dataset ip --method generic --correction gtn --seed 0
  # Pavia University, full fine-tuning, band-count rescaling, spatially separated split
  python scripts/train_dofa.py --dataset pavia --method full --correction bcr --split separated --seed 3
"""
import argparse
import copy
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from bcs import corrections, dofa
from bcs.data import Windows, append_row, is_done, load, scores
from bcs.splits import make_split

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", required=True, choices=["ip", "salinas", "pavia", "houston13"])
p.add_argument("--bands", default="all", help="all | uniform:N | vnir")
p.add_argument("--correction", default="none", choices=corrections.CHOICES)
p.add_argument("--R", type=float, default=corrections.R_DEFAULT, help="threshold / reference band count (63)")
p.add_argument("--split", default="random", choices=["random", "separated"])
p.add_argument("--method", default="generic", choices=dofa.METHODS)
p.add_argument("--model", default="base", choices=["base", "large"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--lr", type=float, default=None, help="default: the paper's rate for --method")
p.add_argument("--lora_r", type=int, default=None, help="default: 32 generic, 4 spectral, 8 otherwise (unused)")
p.add_argument("--lora_alpha", type=int, default=None, help="default: 2 x lora_r")
p.add_argument("--max_epoch", type=int, default=400)
p.add_argument("--patience", type=int, default=100)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--img_size", type=int, default=33, help="window around the labelled pixel")
p.add_argument("--augment", type=int, default=1)
p.add_argument("--csv", required=True, help="results file (one row is appended)")
p.add_argument("--skip_done", type=int, default=1, help="exit if this run is already in --csv")
args = p.parse_args()
default = dofa.PAPER.get(args.method, dict(r=8, alpha=16, lr=6e-5))
args.lora_r = args.lora_r if args.lora_r is not None else default["r"]
args.lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
args.lr = args.lr if args.lr is not None else default["lr"]
if args.skip_done and is_done(args.csv, args.method, args.train_num, args.seed, args.lora_r):
    print("already in %s - skipped" % args.csv)
    sys.exit(0)
print(vars(args), flush=True)

# DOFA's wave_dynamic_layer.py calls torch.manual_seed() when it is imported, so it must be imported BEFORE the run is
# seeded (as in the original scripts); importing it later would reset the torch RNG to DOFA's fixed seed.
dofa.add_dofa_to_path()
import dofa_v1  # noqa: E402,F401
import wave_dynamic_layer  # noqa: E402,F401

# ---------------------------------------------------------------- seeding (same order as the original runs)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- correction (patches DOFA's patch embedding)
corrections.patch_dofa(args.correction, args.R)

# ---------------------------------------------------------------- data
cube, gt, wavelengths_um, _ = load(args.dataset, args.bands)
class_num = int(gt.max())
print("%s: cube %s, %d classes, %d bands %.3f-%.3f um" % (args.dataset, cube.shape, class_num, len(wavelengths_um),
                                                          min(wavelengths_um), max(wavelengths_um)), flush=True)
tr_idx, va_idx, te_idx = make_split(args.split, gt, class_num, args.train_num, args.val_num, args.seed)
gt_flat = gt.reshape(-1)
windows = Windows(cube, args.img_size)


def gather(idx):
    x = torch.tensor(windows(idx), dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(gt_flat[idx].astype(np.int64), dtype=torch.long)
    return x, y


train_x, train_y = gather(tr_idx)
val_x, val_y = gather(va_idx)
test_x, test_y = gather(te_idx)
print("train %s  val %s  test %s" % (tuple(train_x.shape), tuple(val_x.shape), tuple(test_x.shape)), flush=True)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                           batch_size=args.batch_size, shuffle=True, drop_last=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_x, val_y), batch_size=64)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=64)

# ---------------------------------------------------------------- model
model = dofa.build(class_num + 1, args.model)          # output 0 (background) is never a target
model = dofa.adapt(model, args.method, args.lora_r, args.lora_alpha)
n_train_p = sum(q.numel() for q in model.parameters() if q.requires_grad)
n_all_p = sum(q.numel() for q in model.parameters())
print("trainable %d / %d = %.4f%%" % (n_train_p, n_all_p, 100 * n_train_p / n_all_p), flush=True)
model = model.to(device)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.05)
crit = nn.CrossEntropyLoss()


def prep(x):
    return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


def augment(x):                                        # spatial only: flips and 90-degree rotations
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


# ---------------------------------------------------------------- training with early stopping on validation OA
best_val, best_state, best_epoch, bad = -1.0, None, -1, 0
t0 = time.time()
for epoch in range(args.max_epoch + 1):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        if args.augment:
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
if best_state is not None:
    model.load_state_dict(best_state)

# ---------------------------------------------------------------- test
yt, pt = evaluate(test_loader)
oa, aa, kappa = scores(yt, pt, class_num)
print("RESULT %s %s %s %s seed %d: OA %.4f AA %.4f kappa %.4f (best epoch %d)" % (
    args.dataset, args.split, args.method, args.correction, args.seed, oa, aa, kappa, best_epoch), flush=True)
append_row(args.csv, args.method, args.train_num, args.seed, args.lora_r, oa, aa, kappa, n_train_p,
           100 * n_train_p / n_all_p, best_epoch, epoch, train_time, best_val=best_val)
