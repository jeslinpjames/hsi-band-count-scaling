"""Token norm and coherence of DOFA versus the number of bands (Fig. 2a, 2b, 2d; Supplementary Section S3).

Forward passes of the pretrained models only, no training (about 15-30 minutes on one GPU). For N evenly spaced
bands it records the mean norm of the patch tokens and of the tokens after the last block, without correction and
with BCR, for DOFA-Base and DOFA-Large and three input normalisations; and the coherence
||sum_c u_c||^2 / sum_c ||u_c||^2 of the per-band responses u_c (1 = unrelated, N = identical), from which the mean
cosine is (coherence - 1) / (N - 1).

  python scripts/measure_tokens_dofa.py --out results/new
writes  <out>/embed_norm2.csv, <out>/coherence.csv, <out>/EMBED_NORM2_SUMMARY.txt
"""
import argparse, os, sys, math, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _gpu import pick_free_gpu
pick_free_gpu()
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
import torch.nn.functional as F
from bcs import dofa as bdofa
from bcs.data import load_raw
from bcs.wavelengths import WAVELENGTHS
from bcs.paths import DOFA_WEIGHTS

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--models", default="base,large")
p.add_argument("--norms", default="zscore,minmax,global")
p.add_argument("--datasets", default="ip,salinas,pavia")
p.add_argument("--n_pix", type=int, default=128)
p.add_argument("--n_coh", type=int, default=32, help="patches used for the coherence measurement")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--out", default=".")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_txt = open(os.path.join(args.out, "EMBED_NORM2_SUMMARY.txt"), "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out_txt.write(s + "\n"); out_txt.flush()

bdofa.add_dofa_to_path()
import dofa_v1
CKPT = {"base": ("vit_base_patch16", "DOFA_ViT_base_e100.pth"), "large": ("vit_large_patch16", "DOFA_ViT_large_e100.pth")}

def load_model(name):
    builder, ck = CKPT[name]
    sd = torch.load(os.path.join(DOFA_WEIGHTS, ck), map_location="cpu")
    m = getattr(dofa_v1, builder)()
    msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd}, strict=False)
    return m.eval().to(device)


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
    cube, gt = load_raw(ds)
    wl_all = list(WAVELENGTHS[ds])
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
CSV1 = open(os.path.join(args.out, "embed_norm2.csv"), "w"); CSV1.write("model,normalisation,dataset,n_bands,condition,scale,embed_norm_mean,output_norm_mean\n")
CSV2 = open(os.path.join(args.out, "coherence.csv"), "w"); CSV2.write("model,dataset,n_bands,coherence,mean_cosine\n")
raw = {ds: load_patches(ds, args.n_pix) for ds in args.datasets.split(",")}

for mname in args.models.split(","):
    say("\n=== model %s ===" % mname)
    model = load_model(mname)
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
