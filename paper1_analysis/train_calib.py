"""Calibration, uncertainty and optimisation study: train DOFA (Generic LoRA) on one split exactly as
scripts/train_dofa.py does, and save
  * the softmax probabilities of every TEST pixel (for expected calibration error, reliability diagrams,
    confidence on errors and predictive-entropy maps),
  * per epoch: training loss, validation accuracy and the gradient norm of the trainable parameters
    (mean over the iterations of the epoch) and of the LoRA parameters of the first and last block.
Output: results/paper1_analysis/calib_<scene>_<method>_<correction>_s<seed>.npz, runs_calib.csv

  python paper1_analysis/train_calib.py --dataset ip --correction gtn --seed 0
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
p.add_argument("--dataset", required=True, choices=["ip", "salinas", "pavia", "longkou"])
p.add_argument("--correction", default="none", choices=corrections.CHOICES)
p.add_argument("--method", default="generic", choices=dofa.METHODS)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--train_num", type=int, default=10)
p.add_argument("--val_num", type=int, default=5)
p.add_argument("--max_epoch", type=int, default=400)
p.add_argument("--patience", type=int, default=100)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--img_size", type=int, default=33)
p.add_argument("--precision", type=int, default=1, help="1: also test fp16 / bf16 / int8 inference")
p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "paper1_analysis"))
args = p.parse_args()
cfg = dofa.PAPER[args.method]
lora_r, lora_alpha, lr = cfg["r"], 2 * cfg["r"], cfg["lr"]
os.makedirs(args.out_dir, exist_ok=True)
out_npz = os.path.join(args.out_dir, "calib_%s_%s_%s_s%d.npz" % (args.dataset, args.method, args.correction, args.seed))
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


best_val, best_state, best_epoch, bad = -1.0, None, -1, 0
t0 = time.time()
lora_names = [n for n, q in model.named_parameters() if q.requires_grad and "lora_" in n]
blk_ids = sorted({int(n.split("blocks.")[1].split(".")[0]) for n in lora_names if "blocks." in n}) or [0]
first_blk, last_blk = "blocks.%d." % blk_ids[0], "blocks.%d." % blk_ids[-1]
hist = {k: [] for k in ("loss", "val_acc", "grad_all", "grad_first", "grad_last")}
for epoch in range(args.max_epoch + 1):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    ep_g = []
    for x, y in train_loader:
        x = augment(x)
        x, y = prep(x).to(device), y.to(device)
        opt.zero_grad()
        out = model(x, wave_list=wavelengths_um)
        loss = crit(out, y)
        loss.backward()
        g_all = g_first = g_last = 0.0
        for n, q in model.named_parameters():
            if q.grad is None:
                continue
            g2 = q.grad.detach().float().pow(2).sum().item()
            g_all += g2
            if first_blk in n and "lora_" in n:
                g_first += g2
            if last_blk in n and "lora_" in n:
                g_last += g2
        ep_g.append((g_all ** 0.5, g_first ** 0.5, g_last ** 0.5))
        opt.step()
        tot_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    yv, pv = evaluate(val_loader)
    val_acc = accuracy_score(yv, pv)
    g = np.mean(ep_g, axis=0)
    for k, v in zip(("loss", "val_acc", "grad_all", "grad_first", "grad_last"), (tot_loss / total, val_acc, g[0], g[1], g[2])):
        hist[k].append(float(v))
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

print("predicting test pixels (probabilities)", flush=True)


@torch.no_grad()
def probs_idx(idx, chunk=2048, bs=64):
    model.eval()
    out = []
    for i in range(0, len(idx), chunk):
        x = torch.tensor(windows(idx[i:i + chunk]), dtype=torch.float32).permute(0, 3, 1, 2)
        for j in range(0, len(x), bs):
            out.append(torch.softmax(model(prep(x[j:j + bs]).to(device), wave_list=wavelengths_um).float(), 1).cpu().numpy())
        print("  predicted %d / %d" % (min(i + chunk, len(idx)), len(idx)), flush=True)
    return np.concatenate(out)


sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
sync(); t1 = time.time()
P = probs_idx(te_idx)
sync(); t_fp32 = (time.time() - t1) / len(te_idx)
yt = gt_flat[te_idx].astype(np.int64)
te_pred = P.argmax(1)
oa, aa, kappa = scores(yt, te_pred, class_num)

# ---- reduced precision: fp16 / bf16 autocast on the same device (all test pixels), dynamic int8 on the CPU (subset)
extra = {"t_fp32": t_fp32}
for name, dt in ((("fp16", torch.float16), ("bf16", torch.bfloat16)) if args.precision else ()):
    try:
        sync(); t1 = time.time()
        with torch.autocast(device_type=device.type, dtype=dt):
            Pq = probs_idx(te_idx)
        sync()
        extra["pred_" + name] = Pq.argmax(1).astype(np.int16)
        extra["t_" + name] = (time.time() - t1) / len(te_idx)
        print("%s: agreement with fp32 %.4f" % (name, (Pq.argmax(1) == te_pred).mean()), flush=True)
    except Exception as e:                                   # e.g. bf16 unsupported on an old GPU
        print("%s skipped: %s" % (name, e), flush=True)
try:
    if not args.precision:
        raise RuntimeError("--precision 0")
    q_idx = np.sort(np.random.RandomState(0).choice(len(te_idx), min(1000, len(te_idx)), replace=False))
    m_cpu = copy.deepcopy(model).cpu().float().eval()
    try:
        m_cpu = m_cpu.merge_and_unload()                     # fold LoRA into the weights
    except Exception as e:
        print("LoRA not merged (%s); quantising the wrapped layers" % e, flush=True)
    # only the Linear layers of the transformer blocks: the wavelength hypernetwork takes 1-D inputs, which dynamic
    # int8 Linear layers do not accept, and it runs once per band set rather than per pixel
    q_names = {n for n, mod in m_cpu.named_modules() if isinstance(mod, torch.nn.Linear) and "blocks." in n and "patch_embed" not in n}
    m_q = torch.ao.quantization.quantize_dynamic(m_cpu, q_names, dtype=torch.qint8)
    print("int8: %d Linear layers of the transformer blocks quantised" % len(q_names), flush=True)
    outs = {}
    for name, m in (("cpu_fp32", m_cpu), ("int8", m_q)):
        t1 = time.time(); o = []
        with torch.no_grad():
            x = torch.tensor(windows(te_idx[q_idx]), dtype=torch.float32).permute(0, 3, 1, 2)
            for j in range(0, len(x), 64):
                o.append(m(prep(x[j:j + 64]), wave_list=wavelengths_um).argmax(1).numpy())
        outs[name] = np.concatenate(o)
        extra["t_" + name] = (time.time() - t1) / len(q_idx)
    extra.update(q_idx=q_idx, pred_cpu_fp32=outs["cpu_fp32"].astype(np.int16), pred_int8=outs["int8"].astype(np.int16))
    print("int8 (CPU, %d px): agreement with fp32 %.4f" % (len(q_idx), (outs["int8"] == outs["cpu_fp32"]).mean()), flush=True)
    del m_cpu, m_q
except Exception as e:
    print("int8 skipped: %s" % e, flush=True)
np.savez_compressed(out_npz, idx=te_idx, y=yt.astype(np.int16), probs=P.astype(np.float16), shape=np.array(gt.shape),
                    oa=oa, aa=aa, kappa=kappa, best_epoch=best_epoch, train_time=train_time,
                    **{"hist_" + k: np.array(v) for k, v in hist.items()}, **extra)
append_row(os.path.join(args.out_dir, "runs_calib.csv"), "%s_%s_%s" % (args.dataset, args.method, args.correction),
           args.train_num, args.seed, lora_r, oa, aa, kappa, n_train_p, 100 * n_train_p / n_all_p, best_epoch, epoch,
           train_time, best_val=best_val)
print("RESULT %s %s %s seed %d: OA %.4f AA %.4f kappa %.4f (best epoch %d) -> %s" % (
    args.dataset, args.method, args.correction, args.seed, oa, aa, kappa, best_epoch, out_npz), flush=True)
