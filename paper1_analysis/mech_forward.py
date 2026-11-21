"""Mechanism study (forward passes of the PRETRAINED model only, no training).

For one scene, every band count N in {9, 25, 50, 63, 100, 150, all} and every correction (none, bcr, gtn), it passes
256 labelled pixels (fixed sample) through DOFA-Base and records for each of the 12 transformer blocks
  * the mean norm of the residual stream entering the block,
  * the mean norm of the block's update (output - input), and their ratio ("relative update"),
and for the whole network
  * the mean patch-token norm and the mean norm of the features entering the head,
  * linear CKA between the mean-pooled patch tokens and the head features (how much the 12 blocks change the
    representation; CKA near 1 = the backbone is effectively bypassed).
Output: results/paper1_analysis/mech_<scene>.csv  (one row per N x correction x block; block = -1 for global values)

  python paper1_analysis/mech_forward.py --dataset ip
"""
import argparse
import os

from common import (NS, OUT, Windows, blocks_of, corrections, dofa, first_tensor, head_of, import_dofa, linear_cka,
                    load_scene, patch_embed_of)

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--n_pix", type=int, default=256)
p.add_argument("--bs", type=int, default=32)
args = p.parse_args()
out_csv = os.path.join(OUT, "mech_%s.csv" % args.dataset)
if os.path.exists(out_csv):
    print("already done:", out_csv)
    raise SystemExit
import_dofa()
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, gt, wl_all = load_scene(args.dataset)
C = int(gt.max())
lab = np.where(gt.reshape(-1) > 0)[0]
pix = np.random.RandomState(0).choice(lab, min(args.n_pix, len(lab)), replace=False)

model = dofa.build(C + 1, "base").to(dev).eval()
blocks, head, pe = blocks_of(model), head_of(model), patch_embed_of(model)
rec = {}


def blk_hook(i):
    def f(mod, inp, out):
        x, y = first_tensor(inp).float(), first_tensor(out).float()
        rec.setdefault("in", {}).setdefault(i, []).append(x.norm(dim=-1).mean().item())
        rec.setdefault("upd", {}).setdefault(i, []).append((y - x).norm(dim=-1).mean().item())
        # ratio of mean norms (a per-token ratio explodes on near-zero tokens such as an all-zero class token)
        rec.setdefault("rel", {}).setdefault(i, []).append(((y - x).norm(dim=-1).mean() / x.norm(dim=-1).mean().clamp_min(1e-8)).item())
    return f


hooks = [b.register_forward_hook(blk_hook(i)) for i, b in enumerate(blocks)]
hooks.append(pe.register_forward_hook(lambda m, i, o: rec.setdefault("pe", []).append(first_tensor(o).float().detach().cpu())))
hooks.append(head.register_forward_pre_hook(lambda m, i: rec.setdefault("feat", []).append(first_tensor(i).float().detach().cpu())))

rows = ["N,correction,block,in_norm,update_norm,rel_update,token_norm,feat_norm,cka_tokens_feat"]
for N in NS:
    if N != "all" and N >= len(wl_all):             # e.g. N = 150 on Pavia (102 bands)
        continue
    bands = "all" if N == "all" else "uniform:%d" % N
    cube, _, wl = load_scene(args.dataset, bands)
    win = Windows(cube, 33)
    X = torch.tensor(win(pix), dtype=torch.float32).permute(0, 3, 1, 2)
    nb = len(wl)
    for corr in ("none", "bcr", "gtn"):
        corrections.patch_dofa(corr)
        rec.clear()
        with torch.no_grad():
            for i in range(0, len(X), args.bs):
                x = F.interpolate(X[i:i + args.bs], size=(224, 224), mode="bilinear", align_corners=False).to(dev)
                model(x, wave_list=wl)
        tok = torch.cat(rec["pe"])                     # (n, T, D) patch tokens (after correction)
        feat = torch.cat(rec["feat"])                  # (n, D) features entering the head
        tnorm = tok.norm(dim=-1).mean().item()
        fnorm = feat.norm(dim=-1).mean().item()
        cka = linear_cka(tok.mean(1).numpy(), feat.numpy())
        rows.append("%d,%s,-1,,,,%.4f,%.4f,%.4f" % (nb, corr, tnorm, fnorm, cka))
        for b in sorted(rec["in"]):
            rows.append("%d,%s,%d,%.4f,%.4f,%.5f,,," % (nb, corr, b, np.mean(rec["in"][b]), np.mean(rec["upd"][b]),
                                                       np.mean(rec["rel"][b])))
        print("%s N=%3d %-4s token %.1f  feat %.1f  CKA %.3f  rel.update blk0 %.4f blk11 %.4f" % (
            args.dataset, nb, corr, tnorm, fnorm, cka, np.mean(rec["rel"][0]), np.mean(rec["rel"][max(rec["rel"])])),
            flush=True)
corrections.patch_dofa("none")
open(out_csv, "w").write("\n".join(rows) + "\n")
print("RESULT written", out_csv)
