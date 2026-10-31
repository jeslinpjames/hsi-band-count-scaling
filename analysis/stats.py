"""stats.py : every number in the paper, recomputed from the per-run CSVs in ../data.
Writes analysis/output/numbers.json (flat key -> value).  Run from the repository root:  python analysis/stats.py
Reads the per-run CSVs from results/ (or from the folder in the environment variable BCS_RESULTS).
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats

import os as _os
RES = _os.path.join(_os.environ.get("BCS_RESULTS", "results"), "")
OUTDIR = "analysis/output/"
_os.makedirs(OUTDIR + "figures", exist_ok=True)
D = RES
OUT = OUTDIR
LR = {"generic": ("6e-4", 32), "spectral": ("2e-4", 4), "full": ("2e-5", 8)}
DS = ["ip", "salinas", "pavia"]
CONDS = ["nofix", "fix63", "tokenln"]
SEEDS = list(range(16))
N = {}  # numbers


def dofa(ds, cond, m, metric="OA", seeds=SEEDS):
    lr, r = LR[m]
    for f in (f"results_fixvar_{ds}_{cond}_lr{lr}.csv", f"results_lr_{ds}_{cond}_lr{lr}.csv"):
        if os.path.exists(D + f):
            x = pd.read_csv(D + f)
            x = x[(x.method == m) & (x.lora_r == r)].drop_duplicates("seed", keep="last").set_index("seed")
            if len(x):
                s = x[metric] * 100
                return s[s.index.isin(seeds)].sort_index()
    raise FileNotFoundError((ds, cond, m))


def classic(ds, m, metric="OA"):
    x = pd.read_csv(D + f"results_classic_{ds}.csv")
    x = x[x.method == m].drop_duplicates("seed", keep="last").set_index("seed")
    return (x[metric] * 100).sort_index()


def paired(a, b, conf=0.95):
    s = sorted(set(a.index) & set(b.index))
    d = a[s].values - b[s].values
    n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(1 - (1 - conf) / 2, n - 1) * se
    p = 2 * stats.t.sf(abs(mu / se), n - 1) if se > 0 else 1.0
    try:
        pw = stats.wilcoxon(d).pvalue
    except ValueError:
        pw = 1.0
    return dict(n=n, diff=mu, lo=mu - h, hi=mu + h, p=p, pw=pw, wins=int((d > 0).sum()))


def tost(a, b, M=3.0):
    s = sorted(set(a.index) & set(b.index))
    d = a[s].values - b[s].values
    n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n); df = n - 1
    t90 = stats.t.ppf(0.95, df)
    p_lo = stats.t.sf((mu + M) / se, df)       # H0: diff <= -M
    p_hi = stats.t.cdf((mu - M) / se, df)      # H0: diff >= +M
    return dict(diff=mu, lo90=mu - t90 * se, hi90=mu + t90 * se, p_ni=p_lo, p_tost=max(p_lo, p_hi))


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


# ------------------------------------------------------------------ main table
main = {}
for ds in DS:
    for c in CONDS:
        for m in LR:
            oa = dofa(ds, c, m)
            assert len(oa) == 16, (ds, c, m, len(oa))
            main[(ds, c, m)] = dict(OA=oa.mean(), SD=oa.std(), AA=dofa(ds, c, m, "AA").mean(), K=dofa(ds, c, m, "kappa").mean())
    for b in ("rf", "svm", "svm_sp7", "svm_sp33"):
        oa = classic(ds, b); assert len(oa) == 16
        main[(ds, "classic", b)] = dict(OA=oa.mean(), SD=oa.std(), AA=classic(ds, b, "AA").mean(), K=classic(ds, b, "kappa").mean())
for k, v in main.items():
    for f, x in v.items():
        N["main.%s.%s.%s.%s" % (k + (f,))] = round(x, 2)

# ------------------------------------------------------------------ correction effects (family: 27 tests)
eff = []
for ds in DS:
    for m in LR:
        base, r63, ln = dofa(ds, "nofix", m), dofa(ds, "fix63", m), dofa(ds, "tokenln", m)
        eff.append((ds, m, "fix63-nofix", paired(r63, base)))
        eff.append((ds, m, "tokenln-nofix", paired(ln, base)))
        eff.append((ds, m, "tokenln-fix63", paired(ln, r63)))
# two families: effect of each correction vs none (18 tests) and head-to-head LayerNorm vs rescaling (9 tests)
for fam in (("fix63-nofix", "tokenln-nofix"), ("tokenln-fix63",)):
    idx = [i for i, e in enumerate(eff) if e[2] in fam]
    for i, a in zip(idx, holm([eff[i][3]["p"] for i in idx])):
        eff[i][3]["holm"] = a
for (ds, m, cmp_, r) in eff:
    for f, x in r.items():
        N["eff.%s.%s.%s.%s" % (ds, m, cmp_, f)] = round(float(x), 4)
g63 = [r["diff"] for ds, m, c, r in eff if c == "fix63-nofix"]
gln = [r["diff"] for ds, m, c, r in eff if c == "tokenln-nofix"]
N["eff.mean.fix63"] = round(np.mean(g63), 2); N["eff.mean.tokenln"] = round(np.mean(gln), 2)
N["eff.nsig.ln_better"] = sum(1 for ds, m, c, r in eff if c == "tokenln-fix63" and r["holm"] < 0.05 and r["diff"] > 0)
N["eff.nsig.r63_better"] = sum(1 for ds, m, c, r in eff if c == "tokenln-fix63" and r["holm"] < 0.05 and r["diff"] < 0)
N["eff.nsig.ln_better_raw"] = sum(1 for ds, m, c, r in eff if c == "tokenln-fix63" and r["p"] < 0.05 and r["diff"] > 0)
N["eff.nsig.r63_better_raw"] = sum(1 for ds, m, c, r in eff if c == "tokenln-fix63" and r["p"] < 0.05 and r["diff"] < 0)

# ------------------------------------------------------------------ LoRA vs full (TOST, margin 3)
for ds in DS:
    for c in CONDS:
        F = dofa(ds, c, "full")
        for m in ("generic", "spectral"):
            t = tost(dofa(ds, c, m), F)
            for f, x in t.items():
                N["tost.%s.%s.%s.%s" % (ds, c, m, f)] = round(float(x), 4)
for c in CONDS:
    N["tost.count_ni.%s" % c] = sum(1 for ds in DS for m in ("generic", "spectral") if N["tost.%s.%s.%s.p_ni" % (ds, c, m)] < 0.05)
    N["tost.count_eq.%s" % c] = sum(1 for ds in DS for m in ("generic", "spectral") if N["tost.%s.%s.%s.p_tost" % (ds, c, m)] < 0.05)

# ------------------------------------------------------------------ Spectral minus Generic (family 9)
sg = [(ds, c, paired(dofa(ds, c, "spectral"), dofa(ds, c, "generic"))) for ds in DS for c in CONDS]
for (ds, c, r), a in zip(sg, holm([x[2]["p"] for x in sg])):
    r["holm"] = a
    for f, x in r.items():
        N["sg.%s.%s.%s" % (ds, c, f)] = round(float(x), 4)

# ------------------------------------------------------------------ classical (family: 4 baselines x 3 conds x 3 methods x 3 scenes = 108)
cl = []
for ds in DS:
    for b in ("rf", "svm", "svm_sp7", "svm_sp33"):
        for c in CONDS:
            for m in LR:
                cl.append((ds, b, c, m, paired(dofa(ds, c, m), classic(ds, b))))
for (ds, b, c, m, r), a in zip(cl, holm([x[4]["p"] for x in cl])):
    r["holm"] = a
    for f, x in r.items():
        N["cl.%s.%s.%s.%s.%s" % (ds, b, c, m, f)] = round(float(x), 4)
so = [r["diff"] for ds, b, c, m, r in cl if b in ("rf", "svm") and c != "nofix"]
N["cl.spectrum_only.corrected.min"] = round(min(so), 2); N["cl.spectrum_only.corrected.max"] = round(max(so), 2)
N["cl.spectrum_only.corrected.maxholm"] = max(r["holm"] for ds, b, c, m, r in cl if b in ("rf", "svm") and c != "nofix")

# ------------------------------------------------------------------ alternative corrections (IP, Pavia; generic/full; seeds 0-7)
alt = []
for ds in ("ip", "pavia"):
    for m in ("generic", "full"):
        base = dofa(ds, "nofix", m, seeds=range(8)); r63 = dofa(ds, "fix63", m, seeds=range(8))
        N["alt.%s.%s.nofix.OA" % (ds, m)] = round(base.mean(), 2)
        N["alt.%s.%s.fix63.OA" % (ds, m)] = round(r63.mean(), 2)
        for v in ("sqrt63", "ref9", "tokenln"):
            a = dofa(ds, v, m, seeds=range(8))
            N["alt.%s.%s.%s.OA" % (ds, m, v)] = round(a.mean(), 2)
            alt.append((ds, m, v, "none", paired(a, base)))
            alt.append((ds, m, v, "fix63", paired(a, r63)))
        alt.append((ds, m, "fix63", "none", paired(r63, base)))
for (ds, m, v, ref, r), a in zip(alt, holm([x[4]["p"] for x in alt])):
    r["holm"] = a
    for f, x in r.items():
        N["alt.%s.%s.%s.vs_%s.%s" % (ds, m, v, ref, f)] = round(float(x), 4)

# ------------------------------------------------------------------ safety (few-band inputs)
pool = []
for ds in ("ip25", "houston"):
    for m in ("generic", "full"):
        lr, r = LR[m]
        def ld(c):
            x = pd.read_csv(D + f"results_fixvar_{ds}_{c}_lr{lr}.csv")
            x = x[(x.method == m) & (x.lora_r == r)].drop_duplicates("seed", keep="last").set_index("seed")
            return x
        a, b = ld("tokenln"), ld("nofix")
        r_ = paired(a.OA * 100, b.OA * 100)
        N["safe.%s.%s.nofix.OA" % (ds, m)] = round(b.OA.mean() * 100, 2)
        N["safe.%s.%s.tokenln.OA" % (ds, m)] = round(a.OA.mean() * 100, 2)
        for f, x in r_.items():
            N["safe.%s.%s.%s" % (ds, m, f)] = round(float(x), 4)
        s = sorted(set(a.index) & set(b.index)); pool.append((a.OA[s].values - b.OA[s].values) * 100)
x = np.concatenate(pool)
N["safe.pool.n"] = len(x); N["safe.pool.diff"] = round(x.mean(), 2); N["safe.pool.p"] = round(stats.ttest_1samp(x, 0).pvalue, 4)
N["safe.pool.worse"] = int((x < 0).sum())
y = np.concatenate([(pool[0] + pool[1]) / 2, (pool[2] + pool[3]) / 2])
N["safe.seedavg.p"] = round(stats.ttest_1samp(y, 0).pvalue, 4); N["safe.seedavg.n"] = len(y)

# ------------------------------------------------------------------ learning-rate sweep (seeds 0-7, validation & test)
lr = pd.read_csv(D + "lr_runs.csv")
lr = lr[lr.seed < 8].drop_duplicates(["dataset", "condition", "method", "lora_r", "lr", "seed"], keep="last")
lr["lr"] = lr["lr"].astype(str)
g = lr.groupby(["method", "lr", "dataset", "condition"]).agg(val=("best_val", "mean"), test=("test_OA", "mean"), n=("seed", "size")).reset_index()
g.to_csv(OUT + "lr_sweep_table.csv", index=False)
pooled = g.groupby(["method", "lr"]).agg(val=("val", "mean"), test=("test", "mean")).reset_index()
for _, rw in pooled.iterrows():
    N["lr.%s.%s.val" % (rw.method, rw.lr)] = round(rw.val * 100, 2); N["lr.%s.%s.test" % (rw.method, rw.lr)] = round(rw.test * 100, 2)
for m in pooled.method.unique():
    N["lr.best.%s" % m] = pooled[pooled.method == m].sort_values("val").iloc[-1].lr

# ------------------------------------------------------------------ mechanism
en = pd.read_csv(D + "embed_norm2.csv"); co = pd.read_csv(D + "coherence.csv")
for (mdl, nm, ds), d in en.groupby(["model", "normalisation", "dataset"]):
    nf = d[d.condition == "no fix"].set_index("n_bands").sort_index()
    nmax = nf.index.max()
    lg = lambda col: np.log(nf.loc[nmax, col] / nf.loc[25, col]) / np.log(nmax / 25.0)   # 25 bands -> all bands
    N["mech.%s.%s.%s.slope_in" % (mdl, nm, ds)] = round(lg("embed_norm_mean"), 2)
    N["mech.%s.%s.%s.slope_out" % (mdl, nm, ds)] = round(lg("output_norm_mean"), 2)
    for n_ in (9, 63, nmax):
        if n_ in nf.index:
            N["mech.%s.%s.%s.in@%s" % (mdl, nm, ds, "all" if n_ == nmax else n_)] = round(nf.loc[n_, "embed_norm_mean"], 1)
            N["mech.%s.%s.%s.out@%s" % (mdl, nm, ds, "all" if n_ == nmax else n_)] = round(nf.loc[n_, "output_norm_mean"], 1)
    lowout = nf[nf.index <= 13].output_norm_mean
    N["mech.%s.%s.%s.out_low_min" % (mdl, nm, ds)] = round(lowout.min(), 0); N["mech.%s.%s.%s.out_low_max" % (mdl, nm, ds)] = round(lowout.max(), 0)
    fx = d[d.condition != "no fix"].set_index("n_bands").sort_index()
    if len(fx):
        N["mech.%s.%s.%s.out@all_fix" % (mdl, nm, ds)] = round(fx.loc[nmax, "output_norm_mean"], 1)
        N["mech.%s.%s.%s.in@all_fix" % (mdl, nm, ds)] = round(fx.loc[nmax, "embed_norm_mean"], 1)
for _, rw in co.iterrows():
    N["coh.%s.%s.%d.ratio" % (rw.model, rw.dataset, rw.n_bands)] = round(rw.coherence, 2)
    N["coh.%s.%s.%d.cos" % (rw.model, rw.dataset, rw.n_bands)] = round(rw.mean_cosine, 3)
N["fix_conditions_in_embed"] = sorted(en.condition.unique().tolist())

# ------------------------------------------------------------------ trainable parameters
for m in LR:
    lr_, r = LR[m]
    x = pd.read_csv(D + f"results_lr_ip_nofix_lr{lr_}.csv"); x = x[(x.method == m) & (x.lora_r == r)]
    N["params.%s" % m] = int(x.trainable.iloc[0]); N["params_pct.%s" % m] = round(float(x.trainable_pct.iloc[0]), 2)
    x = pd.read_csv(D + f"results_lr_pavia_nofix_lr{lr_}.csv"); x = x[(x.method == m) & (x.lora_r == r)]
    N["params_pavia.%s" % m] = int(x.trainable.iloc[0])


# ------------------------------------------------------------------ deep baselines on the random splits (seeds 0-15)
def deep(ds, m, split="random", seeds=range(16), metric="OA"):
    x = pd.read_csv(D + "results_deep_%s_%s.csv" % (split, ds)); x = x[x.method == m].drop_duplicates("seed", keep="last").set_index("seed")
    s = x[metric] * 100
    return s[s.index.isin(list(seeds))].sort_index()


for ds in DS:
    for b in ("hybridsn", "ssrn"):
        oa = deep(ds, b); assert len(oa) == 16, (ds, b, len(oa))
        for f, v in (("OA", oa.mean()), ("SD", oa.std()), ("AA", deep(ds, b, metric="AA").mean()), ("K", deep(ds, b, metric="kappa").mean()),
                     ("median", oa.median()), ("min", oa.min())):
            N["main.%s.deep.%s.%s" % (ds, b, f)] = round(v, 2)
dp = []
for ds in DS:
    for b in ("hybridsn", "ssrn"):
        for c in CONDS:
            for m in LR:
                dp.append((ds, b, c, m, paired(dofa(ds, c, m), deep(ds, b))))
for (ds, b, c, m, r), a in zip(dp, holm([x[4]["p"] for x in dp])):
    r["holm"] = a
    for f, x in r.items():
        N["dp.%s.%s.%s.%s.%s" % (ds, b, c, m, f)] = round(float(x), 4)

# ------------------------------------------------------------------ spatially disjoint split (Pavia, Salinas; seeds 0-7)
DJ = ["pavia", "salinas"]
def disj(ds, c, m):
    lr, r = LR[m]
    x = pd.read_csv(D + "results_disj_%s_%s_lr%s.csv" % (ds, c, lr)); x = x[(x.method == m) & (x.lora_r == r)]
    x = x.drop_duplicates("seed", keep="last").set_index("seed"); s = x.OA * 100
    return s[s.index < 8].sort_index()
def cdisj(ds, b):
    x = pd.read_csv(D + "results_classic_disjoint_%s.csv" % ds); x = x[x.method == b].drop_duplicates("seed", keep="last").set_index("seed")
    return (x.OA * 100)[lambda s: s.index < 8].sort_index()
BASE = {"svm": lambda ds: cdisj(ds, "svm"), "svm_sp33": lambda ds: cdisj(ds, "svm_sp33"),
        "hybridsn": lambda ds: deep(ds, "hybridsn", "disjoint", range(8)), "ssrn": lambda ds: deep(ds, "ssrn", "disjoint", range(8))}
for ds in DJ:
    for b, fn in BASE.items():
        s = fn(ds); assert len(s) == 8, (ds, b, len(s))
        N["dj.%s.base.%s.OA" % (ds, b)] = round(s.mean(), 2); N["dj.%s.base.%s.SD" % (ds, b)] = round(s.std(), 2)
    for m in ("generic", "full"):
        for c in CONDS:
            s = disj(ds, c, m); assert len(s) == 8, (ds, c, m, len(s))
            N["dj.%s.%s.%s.OA" % (ds, c, m)] = round(s.mean(), 2); N["dj.%s.%s.%s.SD" % (ds, c, m)] = round(s.std(), 2)
    for b in ("svm", "svm_sp33"):     # random-split reference for the drop
        N["dj.%s.drop.%s" % (ds, b)] = round(N["main.%s.classic.%s.OA" % (ds, b)] - N["dj.%s.base.%s.OA" % (ds, b)], 1)
dje = []
for ds in DJ:
    for m in ("generic", "full"):
        dje += [(ds, m, "fix63-nofix", paired(disj(ds, "fix63", m), disj(ds, "nofix", m))),
                (ds, m, "tokenln-nofix", paired(disj(ds, "tokenln", m), disj(ds, "nofix", m))),
                (ds, m, "tokenln-fix63", paired(disj(ds, "tokenln", m), disj(ds, "fix63", m)))]
for (ds, m, cm, r), a in zip(dje, holm([x[3]["p"] for x in dje])):
    r["holm"] = a
    for f, x in r.items():
        N["dje.%s.%s.%s.%s" % (ds, m, cm, f)] = round(float(x), 4)
djb = []
for ds in DJ:
    for m in ("generic", "full"):
        for c in CONDS:
            for b, fn in BASE.items():
                djb.append((ds, m, c, b, paired(disj(ds, c, m), fn(ds))))
for (ds, m, c, b, r), a in zip(djb, holm([x[4]["p"] for x in djb])):
    r["holm"] = a
    for f, x in r.items():
        N["djb.%s.%s.%s.%s.%s" % (ds, m, c, b, f)] = round(float(x), 4)

# ------------------------------------------------------------------ Copernicus-FM
ce = pd.read_csv(D + "cfm_embed_norm.csv"); cc = pd.read_csv(D + "cfm_coherence.csv")
for ds in DS:
    d = ce[ce.dataset == ds]
    nf = d[d.condition == "none"].set_index("n_bands").sort_index(); nmax = nf.index.max()
    for col, k in (("embed_norm_mean", "in"), ("output_norm_mean", "out")):
        N["cfm.mech.%s.slope_%s" % (ds, k)] = round(np.log(nf.loc[nmax, col] / nf.loc[25, col]) / np.log(nmax / 25.0), 2)
        N["cfm.mech.%s.%s@9" % (ds, k)] = round(nf.loc[9, col], 1); N["cfm.mech.%s.%s@all" % (ds, k)] = round(nf.loc[nmax, col], 1)
    for c in ("bcr", "gtn"):
        N["cfm.mech.%s.out@all_%s" % (ds, c)] = round(d[(d.condition == c) & (d.n_bands == nmax)].output_norm_mean.iloc[0], 1)
    lowout = nf[nf.index <= 13].output_norm_mean
    N["cfm.mech.%s.out_low_min" % ds] = round(lowout.min(), 0); N["cfm.mech.%s.out_low_max" % ds] = round(lowout.max(), 0)
for _, rw in cc.iterrows():
    N["cfm.coh.%s.%d.ratio" % (rw.dataset, rw.n_bands)] = round(rw.coherence, 2)
    N["cfm.coh.%s.%d.cos" % (rw.dataset, rw.n_bands)] = round(rw.mean_cosine, 3)
def cfm(ds, c, metric="OA"):
    x = pd.read_csv(D + "results_cfm_random_%s_%s.csv" % (ds, c)).drop_duplicates("seed", keep="last").set_index("seed")
    s = x[metric] * 100
    return s[s.index < 8].sort_index()
cf = []
for ds in DS:
    for c in ("none", "bcr", "gtn"):
        s = cfm(ds, c); assert len(s) == 8, (ds, c, len(s))
        N["cfm.%s.%s.OA" % (ds, c)] = round(s.mean(), 2); N["cfm.%s.%s.SD" % (ds, c)] = round(s.std(), 2)
        N["cfm.%s.%s.AA" % (ds, c)] = round(cfm(ds, c, "AA").mean(), 2); N["cfm.%s.%s.K" % (ds, c)] = round(cfm(ds, c, "kappa").mean(), 2)
    cf += [(ds, "bcr-none", paired(cfm(ds, "bcr"), cfm(ds, "none"))), (ds, "gtn-none", paired(cfm(ds, "gtn"), cfm(ds, "none"))),
           (ds, "gtn-bcr", paired(cfm(ds, "gtn"), cfm(ds, "bcr")))]
for (ds, cm, r), a in zip(cf, holm([x[2]["p"] for x in cf])):
    r["holm"] = a
    for f, x in r.items():
        N["cfe.%s.%s.%s" % (ds, cm, f)] = round(float(x), 4)
x = pd.read_csv(D + "results_cfm_random_ip_none.csv"); N["cfm.params"] = int(x.trainable.iloc[0]); N["cfm.params_pct"] = round(float(x.trainable_pct.iloc[0]), 2)

# ------------------------------------------------------------------ run count & time (all runs, incl. results/unreported)
tot = 0; hours = 0.0
_files = [D + f for f in os.listdir(D)] + ([D + "unreported/" + f for f in os.listdir(D + "unreported")] if os.path.isdir(D + "unreported") else [])
for path in _files:
    f = os.path.basename(path)
    if f.startswith("results_") and f.endswith(".csv") and "classic" not in f and f != "results_rank_patience100.csv":
        x = pd.read_csv(path)
        if "seed" in x:
            x = x[x.seed < 16]
        if "train_time_s" in x:
            tot += len(x); hours += x.train_time_s.clip(lower=0).sum() / 3600
N["runs.total_dofa"] = tot; N["runs.gpu_hours"] = round(hours)

json.dump(N, open(OUT + "numbers.json", "w"), indent=0, sort_keys=True, default=str)
print("wrote", len(N), "numbers")
