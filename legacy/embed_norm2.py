"""embed_norm2.py : reviewer checks M2, M3 and M9 - forward passes only, no training (about 15-30 minutes on one GPU).

 M2  Does the band-count growth also happen in other DOFA checkpoints?  -> DOFA-Base, DOFA-Large, DOFA v2 (if it loads)
 M3  Is it caused by our per-band z-score preprocessing?                -> three different input normalisations
 M9  Is the growth really the "coherent sum" we describe in the paper?   -> directly measures how aligned the
     per-band contributions u_c are:  coherence = ||sum_c u_c||^2 / sum_c ||u_c||^2
       = 1 if the band contributions were unrelated (token norm would grow like sqrt(N))
       = N if they were identical            (token norm grows like N)
     and the mean cosine similarity between band contributions = (coherence - 1) / (N - 1).

usage:  python embed_norm2.py                      (all checks)
        python embed_norm2.py --models base         (quick test)
writes: embed_norm2.csv, coherence.csv, EMBED_NORM2_SUMMARY.txt
"""
import argparse, importlib, os, subprocess, sys, glob, re, math
sys.path.insert(0, "../repos/DOFA"); sys.path.insert(0, ".")
import numpy as np
import scipy.io as sio


def pick_gpu():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                      timeout=30).decode().strip().splitlines()
        return max((int(l.split(",")[1]), l.split(",")[0].strip()) for l in out)[1]
    except Exception:
        return None


_g = pick_gpu()
if _g is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _g
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--models", default="v2,base,large")
p.add_argument("--norms", default="zscore,minmax,global")
p.add_argument("--datasets", default="ip,salinas,pavia")
p.add_argument("--n_pix", type=int, default=128)
p.add_argument("--n_coh", type=int, default=32, help="patches used for the coherence measurement")
p.add_argument("--batch", type=int, default=16)
args = p.parse_args()
torch.set_num_threads(8)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_txt = open("EMBED_NORM2_SUMMARY.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out_txt.write(s + "\n"); out_txt.flush()
say("device", device, "(physical GPU %s)" % os.environ.get("CUDA_VISIBLE_DEVICES", "?"))

# ------------------------------------------------------------------ what normalisation does DOFA's own code use?
say("\n=== DOFA repository: lines mentioning normalisation constants (to compare with our per-band z-score) ===")
hits = 0
for f in sorted(glob.glob("../repos/DOFA/**/*.py", recursive=True)):
    for i, line in enumerate(open(f, errors="ignore")):
        if re.search(r"Normalize|10000|mean *=|std *=|/ *255|reflectance|percentile", line) and hits < 60:
            say("  %s:%d: %s" % (os.path.relpath(f, "../repos/DOFA"), i + 1, line.strip()[:150])); hits += 1
say("  model builders in dofa_v1.py:", re.findall(r"def (vit_\w+)", open("../repos/DOFA/dofa_v1.py").read()))
say("  other .py files in repo root:", sorted(os.path.basename(f) for f in glob.glob("../repos/DOFA/*.py")))

DATASETS = {
    "ip":      ("../repos/HyperSIGMA/ImageClassification/data/Indian_pines_corrected.mat", "data",
                "../repos/HyperSIGMA/ImageClassification/data/Indian_pines_gt.mat", "groundT", "indian_pines_wavelengths"),
    "salinas": ("../data/Salinas/Salinas_corrected.mat", "salinas_corrected",
                "../data/Salinas/Salinas_gt.mat", "salinas_gt", "salinas_wavelengths"),
    "pavia":   ("../data/PaviaU/paviaU.mat", "ori_data", "../data/PaviaU/paviaU_7gt.mat", "map", "pavia_wavelengths"),
}
CKPT = {"base": ("vit_base_patch16", "../DOFA_weights/DOFA_ViT_base_e100.pth"),
        "large": ("vit_large_patch16", "../DOFA_weights/DOFA_ViT_large_e100.pth"),
        "v2": ("vit_base_patch16", "../DOFA_weights/dofav2_vit_base_e150.pth")}
import dofa_v1

def _candidates(name):
    builder, ck = CKPT[name]
    c = []
    if hasattr(dofa_v1, builder):
        c.append((builder, getattr(dofa_v1, builder)))
    if name == "v2" and hasattr(dofa_v1, "OFAViT"):           # DOFA v2 uses 14x14 patch kernels
        import functools, torch.nn as nn
        c.append(("OFAViT(patch_size=14, base)", functools.partial(
            dofa_v1.OFAViT, patch_size=14, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            norm_layer=functools.partial(nn.LayerNorm, eps=1e-6))))
    return c

def load_model(name):
    builder, ck = CKPT[name]
    if not os.path.exists(ck):
        say("  [%s] checkpoint %s not found - skipped" % (name, ck)); return None
    sd = torch.load(ck, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    for label, fn in _candidates(name):
        try:
            m = fn()
        except Exception as e:
            say("  [%s] could not build %s: %s" % (name, label, str(e)[:120])); continue
        msd = m.state_dict()
        bad = [k for k in sd if k in msd and tuple(msd[k].shape) != tuple(sd[k].shape)]
        miss = [k for k in msd if k not in sd and not k.startswith("head") and "fc_norm" not in k]
        say("  [%s] %s: %d tensors, %d missing, %d wrong shape, %d unexpected" % (
            name, label, len(msd), len(miss), len(bad), len([k for k in sd if k not in msd])))
        if bad or len(miss) > 0.05 * len(msd):
            say("        does not fit (e.g. %s) - trying next option" % ((bad or miss)[:2],)); continue
        m.load_state_dict({k: v for k, v in sd.items() if k in msd}, strict=False)
        return m.eval().to(device)
    say("  [%s] no compatible model definition in dofa_v1.py - skipped (needs the DOFA v2 model code)" % name)
    return None

def normalise(cube, how):
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(np.float64)
    if how == "zscore":       # ours: every band to mean 0, sd 1
        flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-8)
    elif how == "minmax":     # every band to [0, 1] (keeps all values positive)
        flat = (flat - flat.min(0)) / (flat.max(0) - flat.min(0) + 1e-8)
    elif how == "global":     # one mean/sd for the whole cube: keeps the shape of every spectrum
        flat = (flat - flat.mean()) / (flat.std() + 1e-8)
    return flat.reshape(H, W, B).astype(np.float32)

def load_patches(ds, n):
    fc, kc, fg, kg, wmod = DATASETS[ds]
    cube = sio.loadmat(fc)[kc].astype(np.float32); gt = sio.loadmat(fg)[kg]
    wl_all = list(importlib.import_module(wmod).wavelengths_um)
    assert len(wl_all) == cube.shape[2]
    rng = np.random.RandomState(0)
    lab = np.argwhere(gt > 0)
    pix = lab[rng.choice(len(lab), size=min(n, len(lab)), replace=False)]
    return cube, pix, wl_all

def cut(cube, pix, size=33):
    r = size // 2
    padded = np.pad(cube, ((r, r), (r, r), (0, 0)), mode="reflect")
    return np.stack([padded[i:i + size, j:j + size] for i, j in pix])

captured = {}
def attach(model):
    captured.clear()
    def hk(name):
        def f(_m, _i, out):
            captured[name] = out[0] if isinstance(out, (tuple, list)) else out
        return f
    hs = [model.patch_embed.register_forward_hook(hk("embed")), model.blocks[-1].register_forward_hook(hk("last"))]
    return hs

rows, coh_rows = [], []
CSV1 = open("embed_norm2.csv", "w"); CSV1.write("model,normalisation,dataset,n_bands,condition,scale,embed_norm_mean,output_norm_mean\n")
CSV2 = open("coherence.csv", "w"); CSV2.write("model,dataset,n_bands,coherence,mean_cosine\n")
raw = {ds: load_patches(ds, args.n_pix) for ds in args.datasets.split(",")}

for mname in args.models.split(","):
    say("\n=== model %s ===" % mname)
    try:
        model = load_model(mname)
    except Exception as e:
        say("  [%s] failed to load: %s - skipped" % (mname, str(e)[:200])); model = None
    if model is None:
        continue
    hs = attach(model)
    for how in args.norms.split(","):
        for ds, (cube, pix, wl_all) in raw.items():
            patches = cut(normalise(cube, how), pix)
            B = patches.shape[-1]
            Ns = sorted(set([n for n in (3, 6, 9, 13, 25, 50, 63, 100, 150, 200) if n < B] + [B]))
            for N in Ns:
                sel = sorted(set(np.linspace(0, B - 1, N).round().astype(int).tolist()))
                wl = [wl_all[i] for i in sel]
                for cname, ref in (("no fix", None), ("REF 63", 63)):
                    s = 1.0 if ref is None else min(1.0, ref / float(len(sel)))
                    e_n, o_n = [], []
                    with torch.no_grad():
                        for b0 in range(0, len(patches), args.batch):
                            x = torch.tensor(patches[b0:b0 + args.batch][..., sel]).permute(0, 3, 1, 2)
                            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(device) * s
                            model(x, wave_list=wl)
                            e_n.append(captured["embed"].float().norm(dim=-1).reshape(-1).cpu())
                            o_n.append(captured["last"].float().norm(dim=-1).reshape(-1).cpu())
                    rows.append((mname, how, ds, len(sel), cname, s, torch.cat(e_n).mean().item(), torch.cat(o_n).mean().item()))
                    CSV1.write("%s,%s,%s,%d,%s,%.4f,%.4f,%.4f\n" % rows[-1]); CSV1.flush()
            say("  %-5s %-6s %-8s done (%d band counts)" % (mname, how, ds, len(Ns)))

        # ---- M9: coherence of the per-band contributions at the full band count (z-score inputs only)
        if how != "zscore":
            continue
        for ds, (cube, pix, wl_all) in raw.items():
            patches = cut(normalise(cube, "zscore"), pix[:args.n_coh])
            B = patches.shape[-1]
            for N in sorted(set([n for n in (9, 25, 63, 100) if n < B] + [B])):
                sel = sorted(set(np.linspace(0, B - 1, N).round().astype(int).tolist()))
                wl = torch.tensor([wl_all[i] for i in sel], device=device).float()
                with torch.no_grad():
                    x = torch.tensor(patches[..., sel]).permute(0, 3, 1, 2)
                    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(device)
                    pe = model.patch_embed
                    bias_only = pe(torch.zeros_like(x[:1]), wl)[0]                 # tokens of an all-zero input = bias
                    total = pe(x, wl)[0] - bias_only                              # sum_c u_c   (P, 196, D)
                    sq_sum = torch.zeros(total.shape[:2], device=device)
                    for c in range(len(sel)):                                     # each band alone, same wavelength set
                        xc = torch.zeros_like(x); xc[:, c] = x[:, c]
                        u = pe(xc, wl)[0] - bias_only
                        sq_sum += (u.float() ** 2).sum(-1)
                    coh = ((total.float() ** 2).sum(-1) / sq_sum.clamp_min(1e-12))   # per token
                cm = coh.mean().item(); n = len(sel)
                coh_rows.append((mname, ds, n, cm, (cm - 1) / (n - 1)))
                CSV2.write("%s,%s,%d,%.4f,%.4f\n" % coh_rows[-1]); CSV2.flush()
                say("  coherence %-5s %-8s N=%3d : ||sum u||^2 / sum ||u||^2 = %7.2f   (1 = unrelated, %d = identical)   mean cosine %.3f" % (
                    mname, ds, n, cm, n, (cm - 1) / (n - 1)))
    for h in hs:
        h.remove()
    del model; torch.cuda.empty_cache()

CSV1.close(); CSV2.close()

# ------------------------------------------------------------------ summary table for the paper
say("\n=== SUMMARY: log-log slope of the token norm vs N (25 bands -> all bands); 1 = proportional to N ===")
say("%-6s %-7s %-8s %9s %9s %10s %10s %12s %12s" % ("model", "norm", "dataset", "in@9", "in@all", "slope_in",
                                                     "slope_out", "out@all", "out@all+fix"))
import collections
D = collections.defaultdict(dict)
for m, how, ds, n, c, s, e, o in rows:
    D[(m, how, ds)][(n, c)] = (e, o)
for (m, how, ds), v in D.items():
    ns = sorted(set(n for n, _ in v))
    nall = ns[-1]
    def g(n, c="no fix"): return v.get((n, c), (float("nan"), float("nan")))
    si = math.log(g(nall)[0] / g(25)[0]) / math.log(nall / 25.0)
    so = math.log(g(nall)[1] / g(25)[1]) / math.log(nall / 25.0)
    say("%-6s %-7s %-8s %9.1f %9.1f %10.2f %10.2f %12.1f %12.1f" % (m, how, ds, g(9)[0], g(nall)[0], si, so,
                                                                    g(nall)[1], g(nall, "REF 63")[1]))
say("\nwrote embed_norm2.csv, coherence.csv, EMBED_NORM2_SUMMARY.txt")
