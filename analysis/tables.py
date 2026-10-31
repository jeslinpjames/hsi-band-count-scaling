"""tables.py : LaTeX table bodies of the paper and supplement from analysis/output/numbers.json.
Run from the repository root after stats.py:  python analysis/tables.py   (writes analysis/output/*.tex)"""
import os as _os
RES = _os.path.join(_os.environ.get("BCS_RESULTS", "results"), "")
OUTDIR = "analysis/output/"
_os.makedirs(OUTDIR + "figures", exist_ok=True)

import json
N = json.load(open(OUTDIR + "numbers.json"))
DS = ["ip", "salinas", "pavia"]
M = [("full", "Full fine-tuning", r"100\%"), ("generic", r"Generic LoRA", r"1.06\%"), ("spectral", r"Spectral LoRA", r"0.72\%")]
C = [("nofix", "none"), ("fix63", "BCR"), ("tokenln", "GTN")]


def f2(x):
    return "%.2f" % x


# ---------------------------------------------------------------- main table
best = {ds: max([N["main.%s.%s.%s.OA" % (ds, c, m)] for c, _ in C for m, _, _ in M] + [N["main.%s.deep.%s.OA" % (ds, b)] for b in ("hybridsn", "ssrn")]) for ds in DS}
L = []
for b, name in (("rf", "Random forest (spectrum)"), ("svm", "SVM (spectrum)"), ("svm_sp7", r"SVM + $7{\times}7$ mean"),
                ("svm_sp33", r"SVM + $33{\times}33$ mean")):
    row = [name, "--", "--"]
    for ds in DS:
        k = "main.%s.classic.%s." % (ds, b)
        row += ["%s$\\pm$%s" % (f2(N[k + "OA"]), f2(N[k + "SD"])), f2(N[k + "AA"]), f2(N[k + "K"])]
    L.append(" & ".join(row) + r"\\")
for b, name in (("hybridsn", "HybridSN"), ("ssrn", "SSRN")):
    row = [name + r"$^\ddagger$", "--", "100\\%"]
    for ds in DS:
        k = "main.%s.deep.%s." % (ds, b)
        oa = f2(N[k + "OA"]) if N[k + "OA"] != best[ds] else r"\textbf{%s}" % f2(N[k + "OA"])
        row += ["%s$\\pm$%s" % (oa, f2(N[k + "SD"])), f2(N[k + "AA"]), f2(N[k + "K"])]
    L.append(" & ".join(row) + r"\\")
for m, mname, tr in M:
    L.append(r"\midrule")
    for i, (c, cname) in enumerate(C):
        row = [mname if i == 0 else "", cname, tr if i == 0 else ""]
        for ds in DS:
            k = "main.%s.%s.%s." % (ds, c, m)
            oa = f2(N[k + "OA"])
            if N[k + "OA"] == best[ds]:
                oa = r"\textbf{%s}" % oa
            row += ["%s$\\pm$%s" % (oa, f2(N[k + "SD"])), f2(N[k + "AA"]), f2(N[k + "K"])]
        L.append(" & ".join(row) + r"\\")
open(OUTDIR + "tab_main.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- safety table
L = []
for ds, name in (("ip25", r"IP, 25 bands"), ("houston", r"Houston, 48")):
    for m, mname in (("generic", "Generic LoRA"), ("full", "Full FT")):
        k = "safe.%s.%s." % (ds, m)
        L.append(r"%s & %s & %s & %s & $%+.2f$ [$%+.2f$, $%+.2f$] & %d/8\\" % (
            name if m == "generic" else "", mname, f2(N[k + "nofix.OA"]), f2(N[k + "tokenln.OA"]), N[k + "diff"], N[k + "lo"], N[k + "hi"], N[k + "wins"]))
L.append(r"\midrule")
L.append(r"\multicolumn{4}{l}{Pooled (32 paired runs)} & $%+.2f$ ($p=%.2f$) & %d/32\\" % (
    N["safe.pool.diff"], N["safe.pool.p"], 32 - N["safe.pool.worse"]))
open(OUTDIR + "tab_safety.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- supplement: correction effects (all 27)
L = []
nm = {"ip": "IP", "salinas": "SA", "pavia": "PU"}
mn = {"generic": "Generic", "spectral": "Spectral", "full": "Full FT"}
cmpn = {"fix63-nofix": "BCR $-$ none", "tokenln-nofix": "GTN $-$ none", "tokenln-fix63": "GTN $-$ BCR"}


def pf(p):
    return "$<$0.001" if p < 0.001 else "%.3f" % p


for cm in ("fix63-nofix", "tokenln-nofix", "tokenln-fix63"):
    for ds in DS:
        for m in ("generic", "spectral", "full"):
            k = "eff.%s.%s.%s." % (ds, m, cm)
            L.append(r"%s & %s & %s & $%+.2f$ [$%+.2f$, $%+.2f$] & %s & %s & %s & %d/16\\" % (
                cmpn[cm], nm[ds], mn[m], N[k + "diff"], N[k + "lo"], N[k + "hi"], pf(N[k + "p"]), pf(N[k + "pw"]), pf(N[k + "holm"]), N[k + "wins"]))
    L.append(r"\midrule")
open(OUTDIR + "tabS_effects.tex", "w").write("\n".join(L[:-1]) + "\n")

# ---------------------------------------------------------------- supplement: TOST
L = []
cn = {"nofix": "none", "fix63": "BCR", "tokenln": "GTN"}
for ds in DS:
    for m in ("generic", "spectral"):
        for c in ("nofix", "fix63", "tokenln"):
            k = "tost.%s.%s.%s." % (ds, c, m)
            ni = N[k + "p_ni"] < 0.05; eq = N[k + "p_tost"] < 0.05
            L.append(r"%s & %s & %s & $%+.2f$ & [$%+.2f$, $%+.2f$] & %s & %s & %s\\" % (
                nm[ds], mn[m], cn[c], N[k + "diff"], N[k + "lo90"], N[k + "hi90"], pf(N[k + "p_ni"]), pf(N[k + "p_tost"]),
                "equivalent" if eq else ("non-inferior" if ni else "--")))
    L.append(r"\midrule")
open(OUTDIR + "tabS_tost.tex", "w").write("\n".join(L[:-1]) + "\n")

# ---------------------------------------------------------------- supplement: spectral vs generic
L = []
for ds in DS:
    for c in ("nofix", "fix63", "tokenln"):
        k = "sg.%s.%s." % (ds, c)
        L.append(r"%s & %s & $%+.2f$ [$%+.2f$, $%+.2f$] & %s & %s & %d/16\\" % (
            nm[ds], cn[c], N[k + "diff"], N[k + "lo"], N[k + "hi"], pf(N[k + "p"]), pf(N[k + "holm"]), N[k + "wins"]))
open(OUTDIR + "tabS_sg.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- supplement: classical
L = []
bn = {"rf": "RF", "svm": "SVM", "svm_sp7": r"SVM+$7{\times}7$", "svm_sp33": r"SVM+$33{\times}33$"}
for ds in DS:
    for b in ("svm_sp33", "svm_sp7", "svm", "rf"):
        cells = []
        for c in ("nofix", "fix63", "tokenln"):
            for m in ("generic", "spectral", "full"):
                k = "cl.%s.%s.%s.%s." % (ds, b, c, m)
                s = "$%+.1f$" % N[k + "diff"]
                if N[k + "holm"] < 0.05:
                    s = r"\textbf{%s}" % s
                cells.append(s)
        L.append(r"%s & %s & %s\\" % (nm[ds] if b == "svm_sp33" else "", bn[b], " & ".join(cells)))
    L.append(r"\midrule")
open(OUTDIR + "tabS_classic.tex", "w").write("\n".join(L[:-1]) + "\n")

# ---------------------------------------------------------------- supplement: alternatives
L = []
vn = {"sqrt63": r"$\min(1,\sqrt{63/N})$", "ref9": r"$\min(1,9/N)$", "tokenln": "token normalization", "fix63": r"BCR, $\min(1,63/N)$"}
for ds in ("ip", "pavia"):
    for m in ("generic", "full"):
        L.append(r"%s & %s & none & %s & -- & --\\" % (nm[ds], mn[m], f2(N["alt.%s.%s.nofix.OA" % (ds, m)])))
        for v in ("fix63", "sqrt63", "ref9", "tokenln"):
            k = "alt.%s.%s.%s." % (ds, m, v)
            a = r"$%+.2f$ (%s, %d/8)" % (N[k + "vs_none.diff"], pf(N[k + "vs_none.p"]), N[k + "vs_none.wins"])
            b = "--" if v == "fix63" else r"$%+.2f$ (%s, %d/8)" % (N[k + "vs_fix63.diff"], pf(N[k + "vs_fix63.p"]), N[k + "vs_fix63.wins"])
            L.append(r" & & %s & %s & %s & %s\\" % (vn[v], f2(N[k + "OA"]), a, b))
        L.append(r"\midrule")
open(OUTDIR + "tabS_alt.tex", "w").write("\n".join(L[:-1]) + "\n")
print("tables written")

# ---------------------------------------------------------------- supplement: per-seed OA
import os
import pandas as pd
LRm = {"generic": ("6e-4", 32), "spectral": ("2e-4", 4), "full": ("2e-5", 8)}


def seeds(ds, c, m):
    lr, r = LRm[m]
    for f in (RES + "results_fixvar_%s_%s_lr%s.csv" % (ds, c, lr), RES + "results_lr_%s_%s_lr%s.csv" % (ds, c, lr)):
        if os.path.exists(f):
            x = pd.read_csv(f); x = x[(x.method == m) & (x.lora_r == r)].drop_duplicates("seed", keep="last").set_index("seed")
            if len(x):
                return x.OA * 100


out = []
for si, (ds, name) in enumerate((("ip", "Indian Pines"), ("salinas", "Salinas"), ("pavia", "Pavia University"))):
    cols = [(m, c) for m in ("full", "generic", "spectral") for c in ("nofix", "fix63", "tokenln")]
    S = {k: seeds(ds, k[1], k[0]) for k in cols}
    out.append(r"\begin{table}[H]\centering\small")
    out.append(r"\caption{Per-Split OA (\%%), %s}\label{tab:seed%d}" % (name, si + 1))
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\begin{tabular}{r ccc ccc ccc}")
    out.append(r"\toprule")
    out.append(r"& \multicolumn{3}{c}{Full fine-tuning} & \multicolumn{3}{c}{Generic LoRA} & \multicolumn{3}{c}{Spectral LoRA}\\")
    out.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    out.append(r"Split & none & BCR & GTN & none & BCR & GTN & none & BCR & GTN\\")
    out.append(r"\midrule")
    for s in range(16):
        out.append("%d & " % s + " & ".join("%.2f" % S[k][s] for k in cols) + r"\\")
    out.append(r"\midrule")
    out.append("Mean & " + " & ".join("%.2f" % S[k][list(range(16))].mean() for k in cols) + r"\\")
    out.append(r"\bottomrule\end{tabular}\end{table}")
open(OUTDIR + "perseed.tex", "w").write("\n".join(out) + "\n")

# ---------------------------------------------------------------- supplement: learning-rate sweep
lr = pd.read_csv(RES + "lr_runs.csv"); lr = lr[lr.seed < 8].drop_duplicates(["dataset", "condition", "method", "lora_r", "lr", "seed"], keep="last")
lr["lrv"] = lr.lr.astype(float)
lr = lr[lr.dataset.isin(["ip", "pavia"])]
L = []
for m in ("generic", "spectral", "full"):
    d = lr[lr.method == m]
    for v in sorted(d.lrv.unique()):
        g = d[d.lrv == v]
        cells = []
        for ds in ("ip", "pavia"):
            for c in ("nofix", "fix63"):
                h = g[(g.dataset == ds) & (g.condition == c)]
                cells += ["%.1f" % (h.best_val.mean() * 100), "%.2f" % (h.test_OA.mean() * 100)]
        a = g[g.condition == "fix63"].set_index(["dataset", "seed"]).test_OA
        b = g[g.condition == "nofix"].set_index(["dataset", "seed"]).test_OA
        ipd = (a.loc["ip"] - b.loc["ip"]).mean() * 100; pud = (a.loc["pavia"] - b.loc["pavia"]).mean() * 100
        sel = r"$\star$" if ("%g" % v) == {"generic": "0.0006", "spectral": "0.0002", "full": "2e-05"}[m] else ""
        L.append(r"%s & %s%s & %s & %.2f & $%+.2f$ & $%+.2f$\\" % (mn[m] if v == sorted(d.lrv.unique())[0] else "", "%g" % v, sel,
                 " & ".join(cells), g.best_val.mean() * 100, ipd, pud))
    L.append(r"\midrule")
open(OUTDIR + "tabS_lr.tex", "w").write("\n".join(L[:-1]) + "\n")

# ---------------------------------------------------------------- supplement: slopes / coherence
L = []
for mdl in ("base", "large"):
    for nm_, nmn in (("zscore", "per-band z-score"), ("minmax", "per-band min--max"), ("global", "global z-score")):
        cells = []
        for ds in DS:
            k = "mech.%s.%s.%s." % (mdl, nm_, ds)
            cells += ["%.1f" % N[k + "in@9"], "%.1f" % N[k + "in@all"], "%.2f" % N[k + "slope_in"], "%.0f" % N[k + "out@all"], "%.0f" % N[k + "out@all_fix"]]
        L.append(r"%s & %s & %s\\" % ({"base": "Base", "large": "Large"}[mdl] if nm_ == "zscore" else "", nmn, " & ".join(cells)))
    L.append(r"\midrule")
open(OUTDIR + "tabS_norms.tex", "w").write("\n".join(L[:-1]) + "\n")
L = []
for mdl in ("base", "large"):
    for ds in DS:
        nmax = {"ip": 200, "salinas": 204, "pavia": 102}[ds]
        cells = []
        for n in (9, 25, 63, 100, nmax):
            cells.append("%.2f (%.3f)" % (N["coh.%s.%s.%d.ratio" % (mdl, ds, n)], N["coh.%s.%s.%d.cos" % (mdl, ds, n)]))
        L.append(r"%s & %s & %s & %d\\" % ({"base": "Base", "large": "Large"}[mdl] if ds == "ip" else "", nm[ds], " & ".join(cells), nmax))
open(OUTDIR + "tabS_coh.tex", "w").write("\n".join(L) + "\n")
print("supplement tables written")

# ---------------------------------------------------------------- Copernicus-FM table
L = []
for c, cn in (("none", "none"), ("bcr", "BCR"), ("gtn", "GTN")):
    row = [cn]
    for ds in DS:
        k = "cfm.%s.%s." % (ds, c)
        row.append("%s$\\pm$%s" % (f2(N[k + "OA"]), f2(N[k + "SD"])))
        if c == "none":
            row.append("--")
        else:
            e = "cfe.%s.%s-none." % (ds, c)
            g = "$%+.2f$ (%d/8)" % (N[e + "diff"], N[e + "wins"])
            if N[e + "p"] < 0.05:
                g = r"\textbf{%s}" % g
            row.append(g)
    L.append(" & ".join(row) + r"\\")
open(OUTDIR + "tab_cfm.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- separated split table (supplement and ISPRS main)
L = []
for lab, key in (("SVM (spectrum)", "base.svm"), (r"SVM + $33{\times}33$ mean", "base.svm_sp33"), ("HybridSN", "base.hybridsn"), ("SSRN", "base.ssrn")):
    L.append(lab + " & " + " & ".join("%s$\\pm$%s" % (f2(N["dj.%s.%s.OA" % (ds, key)]), f2(N["dj.%s.%s.SD" % (ds, key)])) for ds in ("pavia", "salinas")) + r"\\")
L.append(r"\midrule")
for m, mn_ in (("generic", "Generic LoRA"), ("full", "Full fine-tuning")):
    for c, cn in (("nofix", "none"), ("fix63", "BCR"), ("tokenln", "GTN")):
        cells = []
        for ds in ("pavia", "salinas"):
            cell = "%s$\\pm$%s" % (f2(N["dj.%s.%s.%s.OA" % (ds, c, m)]), f2(N["dj.%s.%s.%s.SD" % (ds, c, m)]))
            if c != "nofix":
                e = "dje.%s.%s.%s-nofix." % (ds, m, c)
                g = "$%+.2f$, %d/8" % (N[e + "diff"], N[e + "wins"])
                cell += " (%s)" % (r"\textbf{%s}" % g if N[e + "p"] < 0.05 else g)
            cells.append(cell)
        L.append("DOFA, %s, %s & %s\\\\" % (mn_, cn, " & ".join(cells)))
open(OUTDIR + "tab_disjoint.tex", "w").write("\n".join(L) + "\n")
print("cfm/disjoint tables written")

# ================================================================ supplement additions for the new experiments
def pfx(p):
    return "$<$0.001" if p < 0.001 else "%.3f" % p

# Copernicus-FM token measurement
L = []
for ds in DS:
    nmax = {"ip": 200, "salinas": 204, "pavia": 102}[ds]
    L.append(r"%s & %.1f & %.1f & %.2f & %.2f & %.2f (%.3f) & %.0f & %.0f & %.0f\\" % (
        nm[ds], N["cfm.mech.%s.in@9" % ds], N["cfm.mech.%s.in@all" % ds], N["cfm.mech.%s.slope_in" % ds], N["cfm.mech.%s.slope_out" % ds],
        N["cfm.coh.%s.%d.ratio" % (ds, nmax)], N["cfm.coh.%s.%d.cos" % (ds, nmax)],
        N["cfm.mech.%s.out@all" % ds], N["cfm.mech.%s.out@all_bcr" % ds], N["cfm.mech.%s.out@all_gtn" % ds]))
open(OUTDIR + "tabS_cfm_tokens.tex", "w").write("\n".join(L) + "\n")
# Copernicus-FM comparisons
L = []
cmn = {"bcr-none": "BCR $-$ none", "gtn-none": "GTN $-$ none", "gtn-bcr": "GTN $-$ BCR"}
for ds in DS:
    for cm in ("bcr-none", "gtn-none", "gtn-bcr"):
        k = "cfe.%s.%s." % (ds, cm)
        L.append(r"%s & %s & $%+.2f$ [$%+.2f$, $%+.2f$] & %s & %s & %s & %d/8\\" % (
            nm[ds] if cm == "bcr-none" else "", cmn[cm], N[k + "diff"], N[k + "lo"], N[k + "hi"], pfx(N[k + "p"]), pfx(N[k + "pw"]), pfx(N[k + "holm"]), N[k + "wins"]))
open(OUTDIR + "tabS_cfm_cmp.tex", "w").write("\n".join(L) + "\n")
# DOFA minus deep baselines
L = []
for ds in DS:
    for b, bn_ in (("hybridsn", "HybridSN"), ("ssrn", "SSRN")):
        cells = []
        for c in ("nofix", "fix63", "tokenln"):
            for m in ("generic", "spectral", "full"):
                k = "dp.%s.%s.%s.%s." % (ds, b, c, m)
                s_ = "$%+.1f$" % N[k + "diff"]
                cells.append(r"\textbf{%s}" % s_ if N[k + "holm"] < 0.05 else s_)
        L.append(r"%s & %s & %s\\" % (nm[ds] if b == "hybridsn" else "", bn_, " & ".join(cells)))
open(OUTDIR + "tabS_deep_cmp.tex", "w").write("\n".join(L) + "\n")
# per-split OA of deep baselines and Copernicus-FM
def series(f, method=None):
    x = pd.read_csv(RES + "" + f)
    if method:
        x = x[x.method == method]
    x = x.drop_duplicates("seed", keep="last").set_index("seed")
    return x.OA * 100
L = []
for s_ in range(16):
    row = ["%d" % s_]
    for ds in DS:
        for b in ("hybridsn", "ssrn"):
            row.append("%.2f" % series("results_deep_random_%s.csv" % ds, b)[s_])
    L.append(" & ".join(row) + r"\\")
open(OUTDIR + "tabS_deep_seeds.tex", "w").write("\n".join(L) + "\n")
L = []
for s_ in range(8):
    row = ["%d" % s_]
    for ds in DS:
        for c in ("none", "bcr", "gtn"):
            row.append("%.2f" % series("results_cfm_random_%s_%s.csv" % (ds, c))[s_])
    L.append(" & ".join(row) + r"\\")
open(OUTDIR + "tabS_cfm_seeds.tex", "w").write("\n".join(L) + "\n")
# separated split: all DOFA-vs-baseline comparisons, and per-split OA
L = []
bnm = {"svm": "SVM", "svm_sp33": r"SVM+$33{\times}33$", "hybridsn": "HybridSN", "ssrn": "SSRN"}
for ds in ("pavia", "salinas"):
    for b in ("svm", "svm_sp33", "hybridsn", "ssrn"):
        cells = []
        for m in ("generic", "full"):
            for c in ("nofix", "fix63", "tokenln"):
                k = "djb.%s.%s.%s.%s." % (ds, m, c, b)
                s_ = "$%+.1f$ (%d)" % (N[k + "diff"], N[k + "wins"])
                cells.append(r"\textbf{%s}" % s_ if N[k + "holm"] < 0.05 else s_)
        L.append(r"%s & %s & %s\\" % (nm[ds] if b == "svm" else "", bnm[b], " & ".join(cells)))
open(OUTDIR + "tabS_disj_cmp.tex", "w").write("\n".join(L) + "\n")
L = []
for s_ in range(8):
    row = ["%d" % s_]
    for ds in ("pavia", "salinas"):
        for f, mth in (("results_classic_disjoint_%s.csv", "svm"), ("results_classic_disjoint_%s.csv", "svm_sp33"),
                       ("results_deep_disjoint_%s.csv", "hybridsn"), ("results_deep_disjoint_%s.csv", "ssrn")):
            row.append("%.1f" % series(f % ds, mth)[s_])
        for c in ("nofix", "fix63", "tokenln"):
            row.append("%.1f" % series("results_disj_%s_%s_lr6e-4.csv" % (ds, c), "generic")[s_])
    L.append(" & ".join(row) + r"\\")
open(OUTDIR + "tabS_disj_seeds.tex", "w").write("\n".join(L) + "\n")
print("new supplement tables written")

# transposed Copernicus-FM table for the two-column TGRS layout
L = []
for ds, name in (("ip", "Indian Pines"), ("salinas", "Salinas"), ("pavia", "Pavia Univ.")):
    cells = ["%s$\\pm$%s" % (f2(N["cfm.%s.none.OA" % ds]), f2(N["cfm.%s.none.SD" % ds]))]
    for c in ("bcr", "gtn"):
        e = "cfe.%s.%s-none." % (ds, c)
        g = "$%+.2f$" % N[e + "diff"]
        if N[e + "p"] < 0.05:
            g = r"\textbf{%s}" % g
        cells.append("%s (%s)" % (f2(N["cfm.%s.%s.OA" % (ds, c)]), g))
    L.append(name + " & " + " & ".join(cells) + r"\\")
open(OUTDIR + "tab_cfm_t.tex", "w").write("\n".join(L) + "\n")
