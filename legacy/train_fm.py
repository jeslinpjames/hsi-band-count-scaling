"""train_fm.py : Copernicus-FM-Base with Generic LoRA (qkv of all blocks, r=32, alpha=64, dropout 0.1) under the
three conditions of the paper, on exactly the same splits as the DOFA runs.
Pipeline as for DOFA: per-band standardisation, 33x33 zero-padded window -> bilinear 224x224, flips + 90-degree
rotations, AdamW (weight decay 0.05, batch 16), at most 400 epochs, early stopping on validation OA (patience 100),
best-validation weights evaluated on the test set.

usage:  python train_fm.py --dataset ip --corr bcr --seed 0 [--lr 6e-4] [--split random|disjoint]
writes: results_cfm_<split>_<dataset>_<corr>.csv  (method generic, lora_r 32; same columns as the DOFA CSVs)
"""
import argparse, os, sys, time, subprocess
sys.path.insert(0, ".")
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=["ip", "salinas", "pavia"])
p.add_argument("--corr", required=True, choices=["none", "bcr", "gtn"])
p.add_argument("--seed", type=int, required=True)
p.add_argument("--split", default="random", choices=["random", "disjoint"])
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--max_epoch", type=int, default=400)
p.add_argument("--patience", type=int, default=100)
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--smoke", action="store_true", help="2 epochs, 200 test pixels - only to check that it runs")
args = p.parse_args()
os.environ["SPLIT_MODE"] = args.split
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]).decode().split("\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = max((int(l.split(",")[1]), l.split(",")[0].strip()) for l in out if l.strip())[1]
    except Exception:
        pass
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch, torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from data_common import load, Patches, scores, done, write
from split_modes import make_split
from cfm_common import build, Corrected

torch.set_num_threads(8)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv = "results_cfm_%s_%s_%s.csv" % (args.split, args.dataset, args.corr)
if done(csv, "generic", args.train_num, args.seed, 32) and not args.smoke:
    print("already done"); sys.exit(0)

cube, gt, wl_um, bw = load(args.dataset)
wl_nm = [w * 1000.0 for w in wl_um]; bws = [bw] * len(wl_nm)
gt_flat = gt.reshape(-1); C = int(gt.max())
tr, va, te = make_split(gt_flat, C, args.train_num, args.val_num, np.random.RandomState(args.seed))
if args.smoke:
    te = te[np.random.RandomState(0).choice(len(te), 200, replace=False)]
cut = Patches(cube, 33)
torch.manual_seed(args.seed); np.random.seed(args.seed)

base = build(num_classes=C, weights=os.environ.get("CFM_RANDOM") is None)
cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1, target_modules=["qkv"], modules_to_save=["head", "fc_norm"])
base = get_peft_model(base, cfg)
model = Corrected(base, args.corr).to(dev)
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad); n_all = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)


def to_input(x):          # (n, 33, 33, B) numpy -> (n, B, 224, 224) tensor
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
    return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


def predict(idx, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            out.append(model(to_input(cut(idx[i:i + bs])), wl_nm, bws).argmax(1).cpu().numpy() + 1)
    return np.concatenate(out)


Xtr = cut(tr); ytr = torch.from_numpy(gt_flat[tr] - 1).to(dev)
best, best_ep, best_sd, t0 = -1.0, 0, None, time.time()
epochs = 2 if args.smoke else args.max_epoch
print("Copernicus-FM %s | %s | corr %s | seed %d | %d bands | train %d val %d test %d | trainable %d (%.2f%%)" % (
    args.dataset, args.split, args.corr, args.seed, len(wl_nm), len(tr), len(va), len(te), n_tr, 100.0 * n_tr / n_all), flush=True)
for ep in range(epochs):
    model.train()
    perm = np.random.permutation(len(tr))
    for i in range(0, len(tr), 16):
        b = perm[i:i + 16]
        x = torch.from_numpy(Xtr[b]).permute(0, 3, 1, 2)
        if np.random.rand() < 0.5: x = x.flip(3)
        if np.random.rand() < 0.5: x = x.flip(2)
        x = torch.rot90(x, int(np.random.randint(4)), (2, 3))
        x = F.interpolate(x.to(dev), size=(224, 224), mode="bilinear", align_corners=False)
        loss = F.cross_entropy(model(x, wl_nm, bws), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step()
    acc = float((predict(va) == gt_flat[va]).mean())
    if acc > best:
        best, best_ep = acc, ep
        best_sd = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k or "modules_to_save" in k}
    if ep % 20 == 0:
        print("epoch %d  loss %.4f  val %.4f  best %.4f@%d" % (ep, loss.item(), acc, best, best_ep), flush=True)
    if ep - best_ep >= args.patience:
        break
model.load_state_dict(best_sd, strict=False)
oa, aa, kap = scores(gt_flat[te], predict(te), C)
print("RESULT %s %s %s seed %d: OA %.4f AA %.4f kappa %.4f (best epoch %d, %.0f min)" % (
    args.dataset, args.split, args.corr, args.seed, oa, aa, kap, best_ep, (time.time() - t0) / 60), flush=True)
if not args.smoke:
    write(csv, "generic", args.train_num, args.seed, 32, oa, aa, kap, n_tr, 100.0 * n_tr / n_all, best_ep, ep + 1, time.time() - t0)
