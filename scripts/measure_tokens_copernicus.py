"""Token norm and coherence of Copernicus-FM-Base versus the number of bands (Fig. 2c, 2d; Supplementary S10).

Forward passes only, no training (about 10 minutes on one GPU). Requires the Copernicus-FM environment.

  python scripts/measure_tokens_copernicus.py --out results/new
writes  <out>/cfm_embed_norm.csv, <out>/cfm_coherence.csv, <out>/CFM_TOKENS_SUMMARY.txt
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _gpu import pick_free_gpu
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--datasets", default="ip,salinas,pavia")
p.add_argument("--n_pix", type=int, default=128)
p.add_argument("--out", default=".")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
pick_free_gpu()
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch, torch.nn.functional as F
from bcs.data import load
from bcs.copernicus import build
from bcs.corrections import R_DEFAULT as REF

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary = open(os.path.join(args.out, "CFM_TOKENS_SUMMARY.txt"), "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); summary.write(s + "\n"); summary.flush()

model = build(weights=os.environ.get("CFM_RANDOM") is None).eval().to(dev)
pe = model.patch_embed_spectral
cap = {}
pe.register_forward_hook(lambda m, i, o: cap.__setitem__("embed", o))
model.blocks[-1].register_forward_hook(lambda m, i, o: cap.__setitem__("last", o))
say("Copernicus-FM-Base on", dev, "| patch-embed module:", type(pe).__name__)

C1 = open(os.path.join(args.out, "cfm_embed_norm.csv"), "w"); C1.write("model,dataset,n_bands,condition,embed_norm_mean,output_norm_mean\n")
C2 = open(os.path.join(args.out, "cfm_coherence.csv"), "w"); C2.write("model,dataset,n_bands,coherence,mean_cosine\n")


def cut(cube, pix, size=33):
    r = size // 2
    p = np.pad(cube, ((r, r), (r, r), (0, 0)), mode="reflect")
    return np.stack([p[i:i + size, j:j + size] for i, j in pix])


summary_rows = []
for ds in args.datasets.split(","):
    cube, gt, wl_um, bw = load(ds)
    lab = np.argwhere(gt > 0)
    pix = lab[np.random.RandomState(0).choice(len(lab), args.n_pix, replace=False)]
    patches = cut(cube, pix)
    B = patches.shape[-1]
    Ns = sorted(set([n for n in (3, 6, 9, 13, 25, 50, 63, 100, 150, 200) if n < B] + [B]))
    res = {}
    for N in Ns:
        sel = sorted(set(np.linspace(0, B - 1, N).round().astype(int).tolist()))
        wl = [wl_um[i] * 1000.0 for i in sel]; bws = [bw] * len(sel)
        for cname in ("none", "bcr", "gtn"):
            s = min(1.0, REF / len(sel)) if cname == "bcr" else 1.0
            e_n, o_n = [], []
            with torch.no_grad():
                for b0 in range(0, len(patches), 16):
                    x = torch.tensor(patches[b0:b0 + 16][..., sel]).permute(0, 3, 1, 2)
                    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(dev) * s
                    if cname == "gtn" and len(sel) > REF:
                        h = pe.register_forward_hook(lambda m, i, o: F.layer_norm(o, (o.shape[-1],), eps=1e-6))
                    else:
                        h = None
                    meta = torch.full((x.shape[0], 4), float("nan"), device=dev)
                    model.forward_features(x, meta, wavelengths=wl, bandwidths=bws, input_mode="spectral")
                    if h is not None:
                        h.remove()
                    emb = cap["embed"]
                    if cname == "gtn" and len(sel) > REF:
                        emb = F.layer_norm(emb, (emb.shape[-1],), eps=1e-6)
                    e_n.append(emb.float().norm(dim=-1).reshape(-1).cpu()); o_n.append(cap["last"].float().norm(dim=-1).reshape(-1).cpu())
            en, on = torch.cat(e_n).mean().item(), torch.cat(o_n).mean().item()
            res[(len(sel), cname)] = (en, on)
            C1.write("cfm_base,%s,%d,%s,%.4f,%.4f\n" % (ds, len(sel), cname, en, on)); C1.flush()
    # coherence at several N (32 patches)
    for N in sorted(set([n for n in (9, 25, 63, 100) if n < B] + [B])):
        sel = sorted(set(np.linspace(0, B - 1, N).round().astype(int).tolist()))
        wl = torch.tensor([wl_um[i] * 1000.0 for i in sel], device=dev).float(); bws = torch.full_like(wl, bw)
        with torch.no_grad():
            x = torch.tensor(patches[:32][..., sel]).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(dev)
            bias = pe(torch.zeros_like(x[:1]), wavelengths=wl, bandwidths=bws)[0]
            total = pe(x, wavelengths=wl, bandwidths=bws) - bias
            sq = torch.zeros(total.shape[:2], device=dev)
            for c in range(len(sel)):
                xc = torch.zeros_like(x); xc[:, c] = x[:, c]
                u = pe(xc, wavelengths=wl, bandwidths=bws) - bias
                sq += (u.float() ** 2).sum(-1)
            coh = ((total.float() ** 2).sum(-1) / sq.clamp_min(1e-12)).mean().item()
        cos = (coh - 1) / max(1, len(sel) - 1)
        C2.write("cfm_base,%s,%d,%.4f,%.4f\n" % (ds, len(sel), coh, cos)); C2.flush()
        say("  %-8s N=%3d  coherence %6.2f  (1 = unrelated, %d = identical)  mean cosine %.3f" % (ds, len(sel), coh, len(sel), cos))
    nmax = max(n for n, _ in res)
    slope = np.log(res[(nmax, "none")][0] / res[(25, "none")][0]) / np.log(nmax / 25.0)
    slope_o = np.log(res[(nmax, "none")][1] / res[(25, "none")][1]) / np.log(nmax / 25.0)
    say("%-8s token norm: N=9 %.1f -> N=%d %.1f | log-log slope 25->all: input %.2f, output %.2f | output all bands: none %.0f, BCR %.0f, GTN %.0f" % (
        ds, res[(9, "none")][0], nmax, res[(nmax, "none")][0], slope, slope_o, res[(nmax, "none")][1], res[(nmax, "bcr")][1], res[(nmax, "gtn")][1]))
say("done: cfm_embed_norm.csv, cfm_coherence.csv")
