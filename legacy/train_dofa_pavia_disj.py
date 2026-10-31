import sys, os, time, random, argparse, copy
sys.path.insert(0, '../repos/DOFA')
sys.path.insert(0, '../repos/HyperSIGMA/ImageClassification')
sys.path.insert(0, '.')

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from dofa_v1 import vit_base_patch16
from peft import LoraConfig, get_peft_model
from model import split_data as sd_mod
from pavia_wavelengths import wavelengths_um

# ---------------- args ----------------
p = argparse.ArgumentParser()
p.add_argument('--method', default='generic',
               choices=['generic', 'spectral', 'both', 'linear', 'full'])
p.add_argument('--train_num', type=int, default=10)
p.add_argument('--val_num', type=int, default=5, help='overridden below if not explicitly set')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--max_epoch', type=int, default=300)
p.add_argument('--patience', type=int, default=50, help='early-stop patience in epochs')
p.add_argument('--lr', type=float, default=6e-5)
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--lora_r', type=int, default=8)
p.add_argument('--lora_alpha', type=int, default=16)
p.add_argument('--img_size', type=int, default=33)
p.add_argument('--augment', type=int, default=1)
p.add_argument('--csv', default='results.csv')
args = p.parse_args()

tag = f"{args.method}_n{args.train_num}_seed{args.seed}"
print(f"=== RUN {tag} ===", flush=True)
print(vars(args), flush=True)

# ---------------- seeding ----------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- data ----------------
DATA = '../data/PaviaU'
cube = sio.loadmat(f'{DATA}/paviaU.mat')['ori_data'].astype(np.float32)
gt = sio.loadmat(f'{DATA}/paviaU_7gt.mat')['map'].astype(np.float32)

# per-band standardization (each band gets its own mean/std)
H, W, B = cube.shape
flat = cube.reshape(-1, B)
mu, sigma = flat.mean(0), flat.std(0) + 1e-8
cube = ((flat - mu) / sigma).reshape(H, W, B).astype(np.float32)
print(f"cube {cube.shape}  per-band normalized", flush=True)

class_num = int(gt.max())
gt_flat = gt.reshape(-1)

# --- our own split: per class, shuffle, take train/val, rest is test ---
def _make_split_random(gt_flat, class_num, train_num, val_num, rng):
    tr, va, te = [], [], []
    print(f"{'class':>6}{'total':>8}{'train':>7}{'val':>6}{'test':>8}", flush=True)
    for c in range(1, class_num + 1):
        idx = np.where(gt_flat == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        # never take more than half a class for training; always leave >=1 test pixel
        n_tr = min(train_num, max(1, n // 2))
        n_va = min(val_num, max(1, (n - n_tr) // 2))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
        print(f"{c:>6}{n:>8}{n_tr:>7}{n_va:>6}{len(idx)-n_tr-n_va:>8}", flush=True)
    return (np.concatenate(tr), np.concatenate(va), np.concatenate(te))
import split_modes as _sm
make_split = _sm.wrap(_make_split_random)   # added by patch_split.py

rng = np.random.RandomState(args.seed)
tr_idx, va_idx, te_idx = make_split(gt_flat, class_num, args.train_num, args.val_num, rng)

all_patches, all_labels = sd_mod.create_patches(
    cube, gt, window_size=args.img_size, remove_zero_labels=False)

def gather(idx):
    x = torch.tensor(all_patches[idx], dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(all_labels[idx], dtype=torch.long)
    return x, y

train_x, train_y = gather(tr_idx)
val_x, val_y     = gather(va_idx)
test_x, test_y   = gather(te_idx)
print(f"train {tuple(train_x.shape)}  val {tuple(val_x.shape)}  test {tuple(test_x.shape)}", flush=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_x, train_y),
    batch_size=args.batch_size, shuffle=True, drop_last=False)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_x, val_y), batch_size=64)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_x, test_y), batch_size=64)

# ---------------- model ----------------
model = vit_base_patch16()
ckpt = torch.load('../DOFA_weights/DOFA_ViT_base_e100.pth', map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.head = nn.Linear(768, class_num + 1)

TARGETS = {
    'generic':  ["qkv"],
    'spectral': ["fc_weight", "fc_bias"],
    'both':     ["qkv", "fc_weight", "fc_bias"],
}

if args.method in TARGETS:
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
                     target_modules=TARGETS[args.method], modules_to_save=["head"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
elif args.method == 'linear':
    for n_, p_ in model.named_parameters():
        p_.requires_grad = n_.startswith('head')
elif args.method == 'full':
    for p_ in model.parameters():
        p_.requires_grad = True

n_train_p = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
n_all_p   = sum(p_.numel() for p_ in model.parameters())
print(f"trainable {n_train_p:,} / {n_all_p:,} = {100*n_train_p/n_all_p:.4f}%", flush=True)

model = model.to(device)
opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                        lr=args.lr, weight_decay=0.05)
crit = nn.CrossEntropyLoss()

def prep(x):
    return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

def augment(x):
    # spatial only: random flips + 90-degree rotations. spectral axis untouched.
    if random.random() < 0.5: x = torch.flip(x, dims=[2])
    if random.random() < 0.5: x = torch.flip(x, dims=[3])
    k = random.randint(0, 3)
    if k: x = torch.rot90(x, k, dims=[2, 3])
    return x

@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = prep(x).to(device)
        out = model(x, wave_list=wavelengths_um)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return np.array(trues), np.array(preds)

# ---------------- train ----------------
best_val, best_state, best_epoch, bad = -1.0, None, -1, 0
t0 = time.time()
for epoch in range(args.max_epoch + 1):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        if args.augment: x = augment(x)
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
        print(f"epoch {epoch:4d} loss {tot_loss/total:.5f} train_acc {correct/total:.4f} "
              f"val_acc {val_acc:.4f} best {best_val:.4f}@{best_epoch}", flush=True)

    if bad >= args.patience:
        print(f"early stop at epoch {epoch} (no val gain for {args.patience})", flush=True)
        break

train_time = time.time() - t0
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"restored best checkpoint from epoch {best_epoch} (val {best_val:.4f})", flush=True)

# ---------------- test ----------------
yt, pt = evaluate(test_loader)
oa = accuracy_score(yt, pt)
kappa = cohen_kappa_score(yt, pt)
cm = confusion_matrix(yt, pt, labels=list(range(1, class_num + 1)))
per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
aa = per_class.mean()

print(f"\nRESULT {tag}", flush=True)
print(f"test OA={oa:.4f}", flush=True)
print(f"kappa={kappa:.4f}", flush=True)
print(f"AA={aa:.4f}", flush=True)
print(f"each_acc={np.round(per_class, 4).tolist()}", flush=True)
print(f"train_time_sec={train_time:.1f}  best_epoch={best_epoch}", flush=True)

# ---------------- csv ----------------
new = not os.path.exists(args.csv)
with open(args.csv, 'a') as f:
    if new:
        f.write("method,train_num,seed,lora_r,OA,AA,kappa,trainable,trainable_pct,best_epoch,epochs_run,train_time_s\n")
    f.write(f"{args.method},{args.train_num},{args.seed},{args.lora_r},{oa:.4f},{aa:.4f},{kappa:.4f},"
            f"{n_train_p},{100*n_train_p/n_all_p:.4f},{best_epoch},{epoch},{train_time:.1f}\n")
print(f"appended to {args.csv}", flush=True)
