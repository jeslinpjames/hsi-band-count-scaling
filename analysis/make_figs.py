"""make_figs.py : all paper and supplement figures from the per-run CSVs.
Run from the repository root:  python analysis/make_figs.py   (writes analysis/output/figures/)"""
import os as _os
RES = _os.path.join(_os.environ.get("BCS_RESULTS", "results"), "")
OUTDIR = "analysis/output/"
_os.makedirs(OUTDIR + "figures", exist_ok=True)

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

D, OUT = RES, OUTDIR + "figures/"
COL, DCOL = 3.5, 7.16
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "font.family": "DejaVu Sans"})
INK2 = "#52514e"
C_DS = {"ip": "#2a78d6", "salinas": "#eb6834", "pavia": "#1baf7a"}
NAME = {"ip": "Indian Pines", "salinas": "Salinas", "pavia": "Pavia Univ."}
C_M = {"generic": "#2a78d6", "spectral": "#eb6834", "full": "#1baf7a"}
MNAME = {"generic": "Generic LoRA", "spectral": "Spectral LoRA", "full": "Full fine-tuning"}
C_NONE, C_BCR, C_GTN = "#8d8c86", "#2a78d6", "#eb6834"
LR = {"generic": ("6e-4", 32), "spectral": ("2e-4", 4), "full": ("2e-5", 8)}


def dofa(ds, cond, m, seeds=range(16)):
    lr, r = LR[m]
    for f in (f"results_fixvar_{ds}_{cond}_lr{lr}.csv", f"results_lr_{ds}_{cond}_lr{lr}.csv"):
        if os.path.exists(D + f):
            x = pd.read_csv(D + f)
            x = x[(x.method == m) & (x.lora_r == r)].drop_duplicates("seed", keep="last").set_index("seed")
            if len(x):
                s = x.OA * 100
                return s[s.index.isin(list(seeds))].sort_index()
    raise FileNotFoundError((ds, cond, m))


def paired(a, b, conf=0.95):
    s = sorted(set(a.index) & set(b.index)); d = a[s].values - b[s].values
    n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return dict(n=n, diff=mu, lo=mu - h, hi=mu + h, wins=int((d > 0).sum()))


def save(fig, name):
    fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png", dpi=200); plt.close(fig); print("wrote", name)


# ---------------------------------------------------------------- Fig. 2 mechanism
def fig_mech():
    en = pd.read_csv(D + "embed_norm2.csv"); co = pd.read_csv(D + "coherence.csv")
    ce = pd.read_csv(D + "cfm_embed_norm.csv"); cc = pd.read_csv(D + "cfm_coherence.csv")
    en = en[(en.model == "base") & (en.normalisation == "zscore")]
    fig, ax = plt.subplots(1, 4, figsize=(DCOL, 2.15))
    for ds in ("ip", "salinas", "pavia"):
        d = en[en.dataset == ds]
        nf = d[d.condition == "no fix"].sort_values("n_bands"); fx = d[d.condition != "no fix"].sort_values("n_bands")
        for a, col in ((ax[0], "embed_norm_mean"), (ax[1], "output_norm_mean")):
            a.plot(nf.n_bands, nf[col], "-o", color=C_DS[ds], ms=2.6, mec="white", mew=0.4)
            a.plot(fx.n_bands, fx[col], "--", color=C_DS[ds], lw=0.9)
        e = ce[ce.dataset == ds]
        a = e[e.condition == "none"].sort_values("n_bands"); b = e[e.condition == "bcr"].sort_values("n_bands")
        ax[2].plot(a.n_bands, a.embed_norm_mean, "-s", color=C_DS[ds], ms=2.6, mec="white", mew=0.4)
        ax[2].plot(b.n_bands, b.embed_norm_mean, "--", color=C_DS[ds], lw=0.9)
        c = co[(co.model == "base") & (co.dataset == ds)].sort_values("n_bands")
        ax[3].plot(c.n_bands, c.coherence, "-o", color=C_DS[ds], ms=2.6, mec="white", mew=0.4)
        c = cc[cc.dataset == ds].sort_values("n_bands")
        ax[3].plot(c.n_bands, c.coherence, ":s", color=C_DS[ds], ms=2.6, mec="white", mew=0.4)
    nn = np.array([3, 210])
    ax[3].plot(nn, nn, color=INK2, lw=0.6, ls=":"); ax[3].text(13, 40, "identical\n(= N)", fontsize=5.8, color=INK2)
    ax[3].axhline(1, color=INK2, lw=0.6, ls=":"); ax[3].text(13, 0.62, "unrelated (= 1)", fontsize=5.8, color=INK2)
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log"); a.set_xticks([3, 9, 25, 63, 200]); a.set_xticklabels(["3", "9", "25", "63", "200"])
        a.minorticks_off(); a.set_xlabel("number of bands N")
    for a in ax[:3]:
        a.axvspan(2.8, 9, color="#e4e3df", alpha=0.6, lw=0)
    ax[0].set_ylabel("mean token norm")
    ax[0].set_title("(a) DOFA, patch tokens", loc="left"); ax[1].set_title("(b) DOFA, output", loc="left")
    ax[2].set_title("(c) Copernicus-FM, patch", loc="left"); ax[3].set_title("(d) coherence", loc="left")
    ax[0].set_yticks([5, 10, 30, 100, 300]); ax[0].set_yticklabels(["5", "10", "30", "100", "300"])
    ax[1].set_yticks([100, 200, 400, 800]); ax[1].set_yticklabels(["100", "200", "400", "800"])
    ax[2].set_yticks([10, 30, 100, 300, 1000]); ax[2].set_yticklabels(["10", "30", "100", "300", "1000"])
    ax[3].set_yticks([1, 3, 10, 30, 100]); ax[3].set_yticklabels(["1", "3", "10", "30", "100"]); ax[3].set_ylim(0.5, 250)
    h = [Line2D([], [], color=C_DS[k], marker="o", ms=3, label=NAME[k]) for k in C_DS]
    h += [Line2D([], [], color=INK2, ls="-", label="no correction"), Line2D([], [], color=INK2, ls="--", lw=0.9, label=r"BCR ($R=63$)"),
          Line2D([], [], color=INK2, ls="-", marker="o", ms=3, label="DOFA (d)"), Line2D([], [], color=INK2, ls=":", marker="s", ms=3, label="Copernicus-FM (d)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=7, fontsize=6.3, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_mechanism")


# ---------------------------------------------------------------- Fig. 3 correction effects (16 seeds)
def cfm(ds, c, seeds=range(8)):
    x = pd.read_csv(D + "results_cfm_random_%s_%s.csv" % (ds, c)).drop_duplicates("seed", keep="last").set_index("seed")
    s = x.OA * 100
    return s[s.index.isin(list(seeds))].sort_index()


def fig_effects():
    fig, a = plt.subplots(figsize=(COL, 4.1))
    y, yt, ytl = 0, [], []
    for ds in ("pavia", "salinas", "ip"):
        for m in ("full", "spectral", "generic"):
            base = dofa(ds, "nofix", m)
            for k, (cond, col, off, mk) in enumerate((("fix63", C_BCR, 0.17, "o"), ("tokenln", C_GTN, -0.17, "s"))):
                r = paired(dofa(ds, cond, m), base)
                a.errorbar(r["diff"], y + off, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt=mk, color=col,
                           mec="white", mew=0.5, ms=4.2, capsize=1.5, elinewidth=0.9)
            yt.append(y); ytl.append(MNAME[m]); y += 1
        a.text(1.0, y - 0.45, "DOFA, " + NAME[ds], fontsize=7.3, fontweight="bold", ha="right", va="bottom", transform=a.get_yaxis_transform())
        y += 0.75
    for ds in ("pavia", "salinas", "ip"):                          # Copernicus-FM, Generic LoRA, 8 splits
        base = cfm(ds, "none")
        for cond, col, off, mk in (("bcr", C_BCR, 0.17, "o"), ("gtn", C_GTN, -0.17, "s")):
            r = paired(cfm(ds, cond), base)
            a.errorbar(r["diff"], y + off, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt=mk, color=col,
                       mec="white", mew=0.5, ms=4.2, capsize=1.5, elinewidth=0.9)
        yt.append(y); ytl.append(NAME[ds]); y += 1
    a.text(1.0, y - 0.45, "Copernicus-FM, Generic LoRA", fontsize=7.3, fontweight="bold", ha="right", va="bottom", transform=a.get_yaxis_transform())
    y += 0.75
    a.axvline(0, color=INK2, lw=0.8)
    a.set_yticks(yt); a.set_yticklabels(ytl); a.grid(axis="y", visible=False)
    a.set_xlabel("OA gain over no correction (points)\npaired splits (DOFA 16, Copernicus-FM 8); mean and 95% CI")
    a.legend(handles=[Line2D([], [], color=C_BCR, marker="o", ls="", label=r"band-count rescaling ($R=63$)"),
                      Line2D([], [], color=C_GTN, marker="s", ls="", label="gated token normalization")],
             loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=6.8)
    save(fig, "fig3_correction_effects")


# ---------------------------------------------------------------- Fig. 4 alternative corrections (8 seeds)
def fig_alternatives():
    variants = [("sqrt63", r"$\min(1,\sqrt{63/N})$", "#86b6ef", "v"), ("fix63", r"$\min(1,63/N)$ (BCR)", C_BCR, "o"),
                ("ref9", r"$\min(1,9/N)$", "#104281", "D"), ("tokenln", "token norm. (GTN)", C_GTN, "s")]
    fig, ax = plt.subplots(1, 2, figsize=(COL, 2.35), sharey=True)
    for a, ds in zip(ax, ("ip", "pavia")):
        for i, m in enumerate(("generic", "full")):
            base = dofa(ds, "nofix", m, range(8))
            for j, (v, lab, col, mk) in enumerate(variants):
                r = paired(dofa(ds, v, m, range(8)), base)
                x = i + (j - 1.5) * 0.18
                a.errorbar(x, r["diff"], yerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt=mk, color=col,
                           mec="white", mew=0.5, ms=4.2, capsize=1.5, elinewidth=0.9)
        a.axhline(0, color=INK2, lw=0.8); a.set_xticks([0, 1]); a.set_xticklabels(["Generic LoRA", "Full FT"])
        a.set_xlim(-0.55, 1.55); a.grid(axis="x", visible=False); a.set_title(NAME[ds], loc="left")
    ax[0].set_ylabel("OA gain over no correction (points)")
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, ls="", label=l) for v, l, c, mk in variants],
               loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=6.6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_alternatives")


# ---------------------------------------------------------------- Fig. 5 LoRA minus full fine-tuning (TOST view)
def fig_tost():
    conds = [("nofix", "none", C_NONE, "o"), ("fix63", "BCR", C_BCR, "o"), ("tokenln", "GTN", C_GTN, "s")]
    fig, a = plt.subplots(figsize=(COL, 2.55))
    xt, xl, x = [], [], 0
    for ds in ("ip", "salinas", "pavia"):
        for m in ("generic", "spectral"):
            for j, (c, lab, col, mk) in enumerate(conds):
                r = paired(dofa(ds, c, m), dofa(ds, c, "full"), conf=0.90)
                a.errorbar(x + (j - 1) * 0.22, r["diff"], yerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt=mk,
                           color=col, mec="white", mew=0.5, ms=4, capsize=1.5, elinewidth=0.9)
            xt.append(x); xl.append("Gen." if m == "generic" else "Spec."); x += 1
        a.text(x - 1.5, 1.0, NAME[ds], transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, fontweight="bold")
        x += 0.6
    a.axhspan(-3, 3, color="#e4e3df", alpha=0.7, lw=0); a.axhline(0, color=INK2, lw=0.7)
    a.set_xticks(xt); a.set_xticklabels(xl); a.grid(axis="x", visible=False)
    a.set_ylabel("LoRA minus full fine-tuning\n(OA points, mean and 90% CI)")
    a.legend(handles=[Line2D([], [], color=c, marker=mk, ls="", label=l) for _, l, c, mk in conds], loc="lower left",
             ncol=3, fontsize=6.6)
    a.set_ylim(-8.2, 5.2)
    save(fig, "fig5_lora_vs_full")


# ---------------------------------------------------------------- Supplement: learning-rate sweep
def figS_lr():
    lr = pd.read_csv(D + "lr_runs.csv"); lr = lr[lr.seed < 8]
    lr = lr.drop_duplicates(["dataset", "condition", "method", "lora_r", "lr", "seed"], keep="last")
    lr["lrv"] = lr.lr.astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.2), sharey=False)
    for a, m in zip(ax, ("generic", "spectral", "full")):
        d = lr[(lr.method == m) & lr.dataset.isin(["ip", "pavia"])]
        for (ds, cond), g in d.groupby(["dataset", "condition"]):
            g = g.groupby("lrv").agg(val=("best_val", "mean"), test=("test_OA", "mean")).reset_index()
            a.plot(g.lrv, g.test * 100, "-o" if cond == "nofix" else "--s", color=C_DS[ds], ms=3, mec="white", mew=0.4,
                   label="%s, %s" % (NAME[ds], "no corr." if cond == "nofix" else "BCR"))
        p = d.groupby("lrv").best_val.mean() * 100
        best = p.idxmax()
        a.axvline(best, color=INK2, lw=0.7, ls=":")
        a.set_xscale("log"); a.set_title("%s (selected %g)" % (MNAME[m], best), loc="left"); a.set_xlabel("learning rate")
        tk = sorted(d.lrv.unique()); a.set_xticks(tk); a.set_xticklabels(["%g" % t for t in tk]); a.minorticks_off()
    ax[0].set_ylabel("test OA (%), seeds 0-7")
    ax[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "figS1_lr_sweep")


# ---------------------------------------------------------------- Supplement: DOFA-Large and normalizations
def figS_norms():
    en = pd.read_csv(D + "embed_norm2.csv"); en = en[en.condition == "no fix"]
    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.2), sharey=True)
    ls = {"zscore": "-", "minmax": "--", "global": ":"}
    for a, ds in zip(ax, ("ip", "salinas", "pavia")):
        for (mdl, nm), d in en[en.dataset == ds].groupby(["model", "normalisation"]):
            d = d.sort_values("n_bands")
            a.plot(d.n_bands, d.embed_norm_mean, ls[nm], color="#2a78d6" if mdl == "base" else "#104281",
                   lw=1.2 if mdl == "base" else 1.6, label="%s, %s" % ({"base": "DOFA-Base", "large": "DOFA-Large"}[mdl],
                                                                     {"zscore": "per-band z-score", "minmax": "per-band min-max", "global": "global z-score"}[nm]))
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(NAME[ds], loc="left"); a.set_xlabel("number of bands N")
        a.set_xticks([3, 9, 25, 63, 200]); a.set_xticklabels(["3", "9", "25", "63", "200"]); a.minorticks_off()
    ax[0].set_ylabel("mean patch-token norm")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.4)
    fig.tight_layout(w_pad=0.8)
    save(fig, "figS2_norms_large")


# ---------------------------------------------------------------- Supplement: accuracy vs bands (original learning rate)
def ser(files, method, n, r, seeds):
    parts = [pd.read_csv(D + f) for f in files if os.path.exists(D + f)]
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.method == method) & (d.train_num == n) & (d.lora_r == r)].drop_duplicates("seed", keep="last")
    s = pd.Series(d.OA.values * 100, index=d.seed.values).sort_index()
    return s[s.index.isin(list(seeds))]


IP_NOFIX = ["results_rank.csv", "results_ip_n10_rep.csv", "results_baselines.csv", "results_baselines_rep.csv"]
IP_FIX = ["results_ip_bnref63.csv", "results_ip_bnsafe.csv", "results_ip_bnsafe_full.csv"]


def mci(s):
    return s.mean(), stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))


def figS_bands():
    nof = {25: ["results_ip_uniform25.csv"], 50: ["results_ip_uniform50.csv"], 63: ["results_ip_uniform63.csv"],
           100: ["results_ip_uniform100.csv"], 150: ["results_ip_uniform150.csv"], 200: IP_NOFIX}
    fix = {100: ["results_ip_u100_bnsafe.csv"], 150: ["results_ip_u150_bnsafe.csv"], 200: IP_FIX}
    fig, a = plt.subplots(figsize=(COL, 2.5))
    for m, r in (("generic", 32), ("spectral", 4)):
        for cond, ls, filled in (("no correction", "--", False), ("BCR", "-", True)):
            xs, ys, es = [], [], []
            for n in (25, 50, 63, 100, 150, 200):
                fl = nof[n] if (cond == "no correction" or n <= 63) else fix[n]
                mu, h = mci(ser(fl, m, 10, r, range(8))); xs.append(n); ys.append(mu); es.append(h)
            a.errorbar(xs, ys, yerr=es, color=C_M[m], ls=ls, marker="o" if m == "generic" else "s", mfc=C_M[m] if filled else "white",
                       mec=C_M[m], capsize=2, elinewidth=0.7, label="%s, %s" % (MNAME[m], cond))
    v = ser(["results_ip_vnir.csv"], "generic", 10, 32, range(8)); mu, h = mci(v)
    a.errorbar([63], [mu], yerr=[h], fmt="*", color="#104281", ms=8, capsize=2, label="Generic LoRA, 63 VNIR bands")
    a.axvline(63, color=INK2, lw=0.6, ls=":")
    a.set_xscale("log"); a.set_xticks([25, 50, 63, 100, 150, 200]); a.set_xticklabels(["25", "50", "63", "100", "150", "200"]); a.minorticks_off()
    a.set_xlabel("number of bands (uniformly spaced), Indian Pines"); a.set_ylabel("OA (%)")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.0)
    save(fig, "figS3_bands")


def figS_labels():
    pts = {5: ["results_ip_labelcurve.csv"], 10: IP_NOFIX, 20: ["results_ip_labelcurve.csv"],
           50: ["results_rank_n50_100.csv", "results_rank_n50_r4r32.csv"], 100: ["results_rank_n50_100.csv"]}
    fig, a = plt.subplots(figsize=(COL, 2.2))
    for m, r in (("generic", 32), ("spectral", 4)):
        xs, ys, es = [], [], []
        for n, f in pts.items():
            s = ser(f, m, n, r, range(8)); mu, h = mci(s); xs.append(n); ys.append(mu); es.append(h)
        a.errorbar(xs, ys, yerr=es, color=C_M[m], marker="o" if m == "generic" else "s", mec="white", mew=0.5, capsize=2,
                   elinewidth=0.7, label=MNAME[m])
    a.set_xscale("log"); a.set_xticks(list(pts)); a.set_xticklabels([str(k) for k in pts]); a.minorticks_off()
    a.set_xlabel("labeled pixels per class (Indian Pines, no correction)"); a.set_ylabel("OA (%)"); a.legend(loc="lower right")
    save(fig, "figS4_labels")


def deep(ds, m, split="random", seeds=range(16)):
    x = pd.read_csv(D + "results_deep_%s_%s.csv" % (split, ds)); x = x[x.method == m].drop_duplicates("seed", keep="last").set_index("seed")
    s = x.OA * 100
    return s[s.index.isin(list(seeds))]


def classic(ds, m, split="random", seeds=range(16)):
    f = "results_classic_%s.csv" % ds if split == "random" else "results_classic_disjoint_%s.csv" % ds
    x = pd.read_csv(D + f); x = x[x.method == m].drop_duplicates("seed", keep="last").set_index("seed")
    s = x.OA * 100
    return s[s.index.isin(list(seeds))]


def disj(ds, c, m):
    lr, r = LR[m]
    x = pd.read_csv(D + "results_disj_%s_%s_lr%s.csv" % (ds, c, lr)); x = x[(x.method == m) & (x.lora_r == r)]
    x = x.drop_duplicates("seed", keep="last").set_index("seed"); s = x.OA * 100
    return s[s.index < 8]


def fig_disjoint():
    rows = [("SVM (spectrum)", lambda ds, sp: classic(ds, "svm", sp, range(16) if sp == "random" else range(8)), "#8d8c86"),
            (r"SVM + 33$\times$33 mean", lambda ds, sp: classic(ds, "svm_sp33", sp, range(16) if sp == "random" else range(8)), "#8d8c86"),
            ("HybridSN", lambda ds, sp: deep(ds, "hybridsn", sp, range(16) if sp == "random" else range(8)), "#52514e"),
            ("SSRN", lambda ds, sp: deep(ds, "ssrn", sp, range(16) if sp == "random" else range(8)), "#52514e")]
    for m in ("generic", "full"):
        for c, cn, col in (("nofix", "none", C_NONE), ("fix63", "BCR", C_BCR), ("tokenln", "GTN", C_GTN)):
            rows.append(("DOFA %s, %s" % ("Gen. LoRA" if m == "generic" else "full FT", cn),
                         (lambda c, m: lambda ds, sp: dofa(ds, c, m) if sp == "random" else disj(ds, c, m))(c, m), col))
    fig, ax = plt.subplots(1, 2, figsize=(COL, 2.9), sharey=True)
    for a, ds in zip(ax, ("pavia", "salinas")):
        for i, (lab, fn, col) in enumerate(rows[::-1]):
            r, d = fn(ds, "random"), fn(ds, "disjoint")
            a.plot([d.mean(), r.mean()], [i, i], color="#cfcdc7", lw=1.2, zorder=1)
            a.plot(r.mean(), i, "o", mfc="white", mec=col, mew=1.0, ms=4.2, zorder=2)
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            a.errorbar(d.mean(), i, xerr=h, fmt="o", color=col, mec="white", mew=0.4, ms=4.5, capsize=1.5, elinewidth=0.8, zorder=3)
        a.set_title(NAME[ds], loc="left"); a.set_xlabel("OA (%)"); a.grid(axis="y", visible=False); a.set_xlim(20, 100)
    ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.5)
    fig.legend(handles=[Line2D([], [], color=INK2, marker="o", mfc="white", ls="", label="random split (16 seeds)"),
                        Line2D([], [], color=INK2, marker="o", ls="", label="separated split (8 seeds, 95% CI)")],
               loc="lower center", bbox_to_anchor=(0.6, 0.97), ncol=2, fontsize=6.5)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig6_disjoint")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_mech, fig_effects, fig_alternatives, fig_tost, fig_disjoint, figS_lr, figS_norms, figS_bands, figS_labels):
        f()
