"""cfm_tokens.py : the token-norm and coherence measurement of the paper (Fig. 2) for Copernicus-FM-Base.
Forward passes only, no training (~10 minutes on one GPU).

usage:  python cfm_tokens.py            -> cfm_embed_norm.csv, cfm_coherence.csv, CFM_TOKENS_SUMMARY.txt
"""
import os, sys, subprocess
sys.path.insert(0, ".")
import numpy as np
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]).decode().split("\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = max((int(l.split(",")[1]), l.split(",")[0].strip()) for l in out if l.strip())[1]
    except Exception:
        pass
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch, torch.nn.functional as F
from data_common import load
from cfm_common import build, REF

torch.set_num_threads(8)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary = open("CFM_TOKENS_SUMMARY.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); summary.write(s + "\n"); summary.flush()

model = build(weights=os.environ.get("CFM_RANDOM") is None).eval().to(dev)
pe = model.patch_embed_spectral
cap = {}
pe.register_forward_hook(lambda m, i, o: cap.__setitem__("embed", o))
model.blocks[-1].register_forward_hook(lambda m, i, o: cap.__setitem__("last", o))
say("Copernicus-FM-Base on", dev, "| patch-embed module:", type(pe).__name__)

C1 = open("cfm_embed_norm.csv", "w"); C1.write("model,dataset,n_bands,condition,embed_norm_mean,output_norm_mean\n")
C2 = open("cfm_coherence.csv", "w"); C2.write("model,dataset,n_bands,coherence,mean_cosine\n")


def cut(cube, pix, size=33):
    r = size // 2
    p = np.pad(cube, ((r, r), (r, r), (0, 0)), mode="reflect")
    return np.stack([p[i:i + size, j:j + size] for i, j in pix])


summary_rows = []
for ds in os.environ.get("CFM_DATASETS", "ip,salinas,pavia").split(","):
    cube, gt, wl_um, bw = load(ds)
    lab = np.argwhere(gt > 0)
    pix = lab[np.random.RandomState(0).choice(len(lab), int(os.environ.get("CFM_NPIX", "128")), replace=False)]
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
