"""Turns the outputs of the paper-1 analysis jobs into figures, LaTeX tables and a plain-text summary.

Runs on a laptop (no GPU). Every section is skipped when its inputs are missing, so it can be run while jobs are still
going.
  python paper1_analysis/analyze.py --res results/paper1_analysis --extras results/paper1_extras --out paper1_analysis/output
Outputs (in --out): fig_mech.pdf, fig_calib.pdf, fig_grad.pdf, fig_confusion_pavia.pdf, tab_probe.tex, tab_calib.tex,
tab_quant.tex, tab_throughput.tex, tab_f1.tex, summary.txt
"""
import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/paper1_analysis")
p.add_argument("--extras", default="results/paper1_extras")
p.add_argument("--out", default="paper1_analysis/output")
a = p.parse_args()
warnings.filterwarnings("ignore")
os.makedirs(a.out, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "font.family": "DejaVu Sans"})
INK2 = "#52514e"
SCENES = ("ip", "salinas", "pavia", "longkou")
NAME = {"ip": "Indian Pines", "salinas": "Salinas", "pavia": "Pavia Univ.", "longkou": "WHU-Hi-LongKou"}
CORRS = ("none", "bcr", "gtn")
CNAME = {"none": "No correction", "bcr": "BCR", "gtn": "GTN"}
COL = {"none": "#8d8c86", "bcr": "#2a78d6", "gtn": "#eb6834"}
LS = {"ip": "-", "salinas": "--", "pavia": ":", "longkou": "-."}
PAVIA = ["Trees", "Asphalt", "Bricks", "Bitumen", "Shadows", "Meadows", "Bare soil"]
summary = []


def say(*s):
    line = " ".join(str(x) for x in s)
    summary.append(line)
    print(line)


def ms(v, scale=100, d=1):
    v = np.asarray(v, float) * scale
    return ("%.*f" % (d, v.mean())) + (r" $\pm$ %.*f" % (d, v.std(ddof=1)) if len(v) > 1 else "")


# =============================================================== 1. mechanism (forward passes)
mech = {sc: pd.read_csv(os.path.join(a.res, "mech_%s.csv" % sc)) for sc in SCENES
        if os.path.exists(os.path.join(a.res, "mech_%s.csv" % sc))}
if mech:
    say("== Mechanism (pretrained DOFA, no training)")
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.15), gridspec_kw=dict(wspace=0.38))
    for sc, d in mech.items():
        Nall = d.N.max()
        g = d[d.block == -1]
        for c in CORRS:
            b = d[(d.N == Nall) & (d.correction == c) & (d.block >= 0)].sort_values("block")
            ax[0].plot(b.block + 1, b.rel_update, LS[sc], color=COL[c], marker="o" if sc == "ip" else None)
            gc = g[g.correction == c].sort_values("N")
            ax[1].plot(gc.N, gc.token_norm, LS[sc], color=COL[c], marker="o")
            ax[2].plot(gc.N, gc.cka_tokens_feat, LS[sc], color=COL[c], marker="o")
        for c in CORRS:
            b = d[(d.N == Nall) & (d.correction == c) & (d.block >= 0)]
            gc = g[(g.correction == c) & (g.N == Nall)].iloc[0]
            say("%-8s N=%d %-4s token norm %.1f  feature norm %.1f  CKA(tokens, features) %.3f  "
                "relative update: mean %.3f, block 1 %.3f, block 12 %.3f" % (
                    sc, Nall, c, gc.token_norm, gc.feat_norm, gc.cka_tokens_feat, b.rel_update.mean(),
                    b[b.block == 0].rel_update.iloc[0], b[b.block == b.block.max()].rel_update.iloc[0]))
    ax[0].set_xlabel("Transformer block"); ax[0].set_ylabel(r"$\|\Delta h\| \,/\, \|h\|$ (all bands)")
    ax[0].set_yscale("log"); ax[0].set_xticks([1, 4, 8, 12])
    for k in (1, 2):
        ax[k].set_xscale("log"); ax[k].set_xlabel("Number of bands $N$")
        ax[k].axvline(63, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax[k].set_xticks([9, 25, 63, 200]); ax[k].set_xticklabels(["9", "25", "63", "200"])
    ax[1].set_yscale("log"); ax[1].set_ylabel("Mean patch-token norm")
    ax[2].set_ylabel("CKA(patch tokens, features)")
    for k, t in enumerate("abc"):
        ax[k].set_title("(%s)" % t, loc="left", fontsize=8.5, pad=2)
    hand = [Line2D([], [], color=COL[c], lw=1.6, label=CNAME[c]) for c in CORRS] + \
           [Line2D([], [], color=INK2, ls=LS[sc], lw=1.1, label=NAME[sc]) for sc in mech]
    fig.legend(handles=hand, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.2), handlelength=2.2)
    fig.savefig(os.path.join(a.out, "fig_mech.pdf")); fig.savefig(os.path.join(a.out, "fig_mech.png"), dpi=200)
    plt.close(fig)

# =============================================================== 2. frozen-feature probes
probe = {sc: pd.read_csv(os.path.join(a.res, "probe_%s.csv" % sc)) for sc in SCENES
         if os.path.exists(os.path.join(a.res, "probe_%s.csv" % sc))}
if probe:
    from scipy.stats import ttest_rel, wilcoxon
    say("== Frozen-feature probes (OA %, mean +- std over splits; paired tests against no correction)")
    lines = []
    for clf, cname in (("logreg", "Logistic regression"), ("knn", "$k$-NN (cosine)")):
        cells = []
        for sc, d in probe.items():
            base = d[(d.classifier == clf) & (d.correction == "none")].sort_values("seed").OA.values
            for c in CORRS:
                v = d[(d.classifier == clf) & (d.correction == c)].sort_values("seed").OA.values
                txt = ms(v)
                if c != "none" and len(v) > 1:
                    t, pt = ttest_rel(v, base)
                    try:
                        pw = wilcoxon(v, base).pvalue
                    except ValueError:
                        pw = 1.0
                    say("%-8s %-6s %-4s OA %s  diff %+.2f  t-test p=%.3g  Wilcoxon p=%.3g" % (
                        sc, clf, c, txt.replace(r" $\pm$ ", " +- "), 100 * (v - base).mean(), pt, pw))
                    if pt < 0.05 and v.mean() > base.mean():
                        txt = r"\textbf{%s}" % txt
                else:
                    say("%-8s %-6s %-4s OA %s" % (sc, clf, c, txt.replace(r" $\pm$ ", " +- ")))
                cells.append(txt)
        lines.append("%s & %s \\\\" % (cname, " & ".join(cells)))
    open(os.path.join(a.out, "tab_probe.tex"), "w").write("\n".join(lines) + "\n")

# =============================================================== 3. calibration, uncertainty, errors, precision
cal_files = sorted(glob.glob(os.path.join(a.res, "calib_*_generic_*_s*.npz")))


def calib_metrics(d, nbins=15):
    P = d["probs"].astype(np.float64)[:, 1:]                  # column 0 = background, never a test label
    P /= P.sum(1, keepdims=True)
    y = d["y"].astype(int) - 1
    pred = P.argmax(1)
    conf = P.max(1)
    ok = pred == y
    C = P.shape[1]
    bins = np.minimum((conf * nbins).astype(int), nbins - 1)
    ece = sum(abs(ok[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(nbins) if (bins == b).any())
    nll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()
    brier = ((P - np.eye(C)[y]) ** 2).sum(1).mean()
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1) / np.log(C)
    # misclassification detection with the max probability (AUROC), and selective risk (AURC)
    order = np.argsort(-conf)
    risk = np.cumsum(~ok[order]) / np.arange(1, len(y) + 1)
    aurc = risk.mean()
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
    auroc = roc_auc_score(ok, conf) if 0 < ok.mean() < 1 else np.nan
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=range(C), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(C))
    fp = cm.sum(0) - np.diag(cm)
    fpr = fp / np.maximum(len(y) - cm.sum(1), 1)                # per-class false-positive rate
    cov90 = (conf >= 0.9).mean(); acc90 = ok[conf >= 0.9].mean() if (conf >= 0.9).any() else np.nan
    return dict(OA=ok.mean(), ECE=ece, NLL=nll, Brier=brier, AUROC=auroc, AURC=aurc, conf_ok=conf[ok].mean(),
                conf_err=conf[~ok].mean() if (~ok).any() else np.nan, ent_err=ent[~ok].mean() if (~ok).any() else np.nan,
                ent_ok=ent[ok].mean(), macroP=pr.mean(), macroR=rc.mean(), macroF1=f1.mean(), meanFPR=fpr.mean(),
                cov90=cov90, acc90=acc90), (conf, ok, risk, cm)


if cal_files:
    rows, keep = [], {}
    for f in cal_files:
        m = re.match(r"calib_(\w+?)_generic_(\w+)_s(\d+)\.npz", os.path.basename(f))
        sc, c, s = m.group(1), m.group(2), int(m.group(3))
        d = np.load(f)
        met, raw = calib_metrics(d)
        met.update(scene=sc, correction=c, seed=s, best_epoch=int(d["best_epoch"]), train_time=float(d["train_time"]))
        # reduced precision
        te_pred = d["probs"].astype(np.float32).argmax(1)          # same rule as in train_calib.py
        y = d["y"].astype(int)
        for q in ("fp16", "bf16"):
            if "pred_" + q in d.files:
                met["agree_" + q] = (d["pred_" + q] == te_pred).mean()
                met["dOA_" + q] = (d["pred_" + q] == y).mean() - (te_pred == y).mean()
                met["speed_" + q] = float(d["t_fp32"]) / float(d["t_" + q])
        if "pred_int8" in d.files:
            qi = d["q_idx"]
            met["agree_int8"] = (d["pred_int8"] == d["pred_cpu_fp32"]).mean()
            met["dOA_int8"] = (d["pred_int8"] == y[qi]).mean() - (d["pred_cpu_fp32"] == y[qi]).mean()
            met["speed_int8"] = float(d["t_cpu_fp32"]) / float(d["t_int8"])
        rows.append(met)
        keep[(sc, c, s)] = (raw, [len(v) for v in [d["hist_grad_all"]]][0], d)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "calib_all_runs.csv"), index=False)
    say("== Calibration / uncertainty / errors (fine-tuned Generic LoRA, test pixels; mean +- std over seeds)")
    keys = ["OA", "ECE", "NLL", "Brier", "AUROC", "AURC", "conf_err", "macroF1", "meanFPR", "cov90", "acc90"]
    for sc in SCENES:
        for c in CORRS:
            g = R[(R.scene == sc) & (R.correction == c)]
            if len(g):
                say("%-8s %-4s n=%d  " % (sc, c, len(g)) + "  ".join(
                    "%s %.3f+-%.3f" % (k, g[k].mean(), g[k].std(ddof=1) if len(g) > 1 else 0) for k in keys))
    # table: rows = scene x correction
    lines = []
    for sc in SCENES:
        first = True
        for c in CORRS:
            g = R[(R.scene == sc) & (R.correction == c)]
            if not len(g):
                continue
            cells = [ms(g.OA), ms(g.ECE), "%.2f" % g.NLL.mean(), ms(g.AUROC, d=1), ms(g.AURC), ms(g.conf_err),
                     ms(g.macroF1), ms(g.meanFPR, d=2)]
            lines.append("%s & %s & %s \\\\" % (r"\multirow{3}{*}{%s}" % NAME[sc] if first else "", CNAME[c], " & ".join(cells)))
            first = False
        if lines and lines[-1] != "\\midrule":
            lines.append("\\midrule")
    open(os.path.join(a.out, "tab_calib.tex"), "w").write("\n".join(lines[:-1] if lines else []) + "\n")

    # reduced-precision table
    qcols = [k for k in ("agree_fp16", "dOA_fp16", "speed_fp16", "agree_bf16", "dOA_bf16", "speed_bf16",
                         "agree_int8", "dOA_int8", "speed_int8") if k in R]
    if qcols:
        say("== Reduced precision (agreement with fp32 predictions, OA change in points, speed-up)")
        lines = []
        for sc in SCENES:
            for c in ("none", "gtn"):
                g = R[(R.scene == sc) & (R.correction == c)]
                if not len(g):
                    continue
                say("%-8s %-4s " % (sc, c) + "  ".join("%s %.4f" % (k, g[k].mean()) for k in qcols))
                cells = []
                for q in ("fp16", "bf16", "int8"):
                    if "agree_" + q in g:
                        cells += ["%.2f" % (100 * g["agree_" + q].mean()), "%+.2f" % (100 * g["dOA_" + q].mean()),
                                  r"%.1f$\times$" % g["speed_" + q].mean()]
                lines.append("%s & %s & %s \\\\" % (NAME[sc], CNAME[c], " & ".join(cells)))
        open(os.path.join(a.out, "tab_quant.tex"), "w").write("\n".join(lines) + "\n")

    # reliability diagram + risk-coverage (pooled over seeds), per scene
    scs = [sc for sc in SCENES if ((R.scene == sc)).any()]
    fig, ax = plt.subplots(2, len(scs), figsize=(7.16, 3.9), squeeze=False, gridspec_kw=dict(hspace=0.5, wspace=0.3))
    for j, sc in enumerate(scs):
        for c in CORRS:
            ks = [k for k in keep if k[0] == sc and k[1] == c]
            if not ks:
                continue
            conf = np.concatenate([keep[k][0][0] for k in ks]); ok = np.concatenate([keep[k][0][1] for k in ks])
            edges = np.linspace(0, 1, 11); b = np.minimum((conf * 10).astype(int), 9)
            xs = [conf[b == i].mean() for i in range(10) if (b == i).sum() >= 20]
            ys = [ok[b == i].mean() for i in range(10) if (b == i).sum() >= 20]
            ece = R[(R.scene == sc) & (R.correction == c)].ECE.mean()
            ax[0, j].plot(xs, ys, "-o", color=COL[c], label="%s (ECE %.1f%%)" % (CNAME[c], 100 * ece))
            cov = np.arange(1, len(conf) + 1) / len(conf)
            order = np.argsort(-conf); risk = np.cumsum(~ok[order]) / np.arange(1, len(conf) + 1)
            ax[1, j].plot(cov, 100 * risk, color=COL[c], label=CNAME[c])
        ax[0, j].plot([0, 1], [0, 1], color=INK2, lw=0.7, ls=(0, (2, 2)))
        ax[0, j].set_xlim(0, 1); ax[0, j].set_ylim(0, 1)
        ax[0, j].set_title(NAME[sc], fontsize=8.5, pad=3)
        ax[0, j].set_xlabel("Confidence"); ax[0, j].legend(loc="upper left", fontsize=6.8, handlelength=1.4)
        ax[1, j].set_xlabel("Coverage (most confident first)")
        ax[1, j].set_xlim(0, 1); ax[1, j].set_ylim(bottom=0)
    ax[0, 0].set_ylabel("Accuracy"); ax[1, 0].set_ylabel("Error on kept pixels (%)")
    ax[1, 0].legend(loc="upper left", fontsize=6.8)
    fig.savefig(os.path.join(a.out, "fig_calib.pdf")); fig.savefig(os.path.join(a.out, "fig_calib.png"), dpi=200)
    plt.close(fig)

    # gradient norms during fine-tuning (first and last LoRA block), mean over seeds
    fig, ax = plt.subplots(1, len(scs), figsize=(7.16, 2.0), squeeze=False, gridspec_kw=dict(wspace=0.3))
    say("== Gradient norms (mean over the first 50 epochs; first / last LoRA block)")
    for j, sc in enumerate(scs):
        for c in CORRS:
            ks = [k for k in keep if k[0] == sc and k[1] == c]
            if not ks:
                continue
            L = min(len(keep[k][2]["hist_grad_first"]) for k in ks)
            gf = np.mean([keep[k][2]["hist_grad_first"][:L] for k in ks], 0)
            gl = np.mean([keep[k][2]["hist_grad_last"][:L] for k in ks], 0)
            ax[0, j].plot(np.arange(L), gf, "-", color=COL[c], label=CNAME[c] + ", block 1")
            ax[0, j].plot(np.arange(L), gl, ":", color=COL[c], label=CNAME[c] + ", block 12")
            say("%-8s %-4s grad first %.4g  last %.4g  ratio first/last %.3f" % (
                sc, c, gf[:50].mean(), gl[:50].mean(), gf[:50].mean() / max(gl[:50].mean(), 1e-12)))
        ax[0, j].set_yscale("log"); ax[0, j].set_xlabel("Epoch"); ax[0, j].set_title(NAME[sc], fontsize=8.5, pad=3)
    ax[0, 0].set_ylabel("LoRA gradient norm")
    ax[0, -1].legend(fontsize=6.3, loc="upper right", ncol=1)
    fig.savefig(os.path.join(a.out, "fig_grad.pdf")); plt.close(fig)

# =============================================================== 4. throughput
tp = os.path.join(a.res, "throughput.csv")
if os.path.exists(tp):
    T = pd.read_csv(tp)
    say("== Inference throughput")
    say(T.to_string(index=False))
    lines = []
    lab = {"dofa_generic": "DOFA + Generic LoRA", "ssrn": "SSRN (GPU)", "svm_sp33": r"SVM + $33{\times}33$ mean (CPU)"}
    for (m, c), g in T.groupby(["method", "correction"], sort=False):
        cells = ["%.0f" % g[g.scene == sc].pixels_per_s.iloc[0] if (g.scene == sc).any() else "--" for sc in ("ip", "pavia")]
        lines.append("%s & %s & %s \\\\" % (lab.get(m, m), CNAME.get(c, "--"), " & ".join(cells)))
    open(os.path.join(a.out, "tab_throughput.tex"), "w").write("\n".join(lines) + "\n")

# =============================================================== 5. precision / recall / F1 / false positives (split 0)
METHODS = [("svm_sp33", "SVM + 33x33 mean"), ("ssrn", "SSRN"), ("dofa_generic_none", "DOFA, no corr."),
           ("dofa_generic_gtn", "DOFA + GTN")]


def xpath(m, sc):
    return os.path.join(a.extras, ("pred_dofa_%s_%s_s0.npz" % (sc, m[5:])) if m.startswith("dofa") else "pred_%s_%s_s0.npz" % (m, sc))


if any(os.path.exists(xpath(m, sc)) for m, _ in METHODS for sc in SCENES):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    say("== Split 0, test pixels: macro precision / recall / F1, mean per-class false-positive rate (%)")
    lines, cms = [], {}
    for sc in SCENES:
        cells = []
        for m, mn in METHODS:
            if not os.path.exists(xpath(m, sc)):
                cells.append("--"); continue
            d = np.load(xpath(m, sc))
            k = int(d["n_train"]) + int(d["n_val"])
            y, pr = d["y"][k:].astype(int), d["pred"][k:].astype(int)
            C = int(max(y.max(), pr.max()))
            P_, R_, F_, _ = precision_recall_fscore_support(y, pr, labels=range(1, C + 1), zero_division=0)
            cm = confusion_matrix(y, pr, labels=range(1, C + 1))
            fpr = (cm.sum(0) - np.diag(cm)) / np.maximum(len(y) - cm.sum(1), 1)
            cms[(m, sc)] = cm
            say("%-8s %-18s P %.1f  R %.1f  F1 %.1f  FPR %.2f" % (sc, mn, 100 * P_.mean(), 100 * R_.mean(), 100 * F_.mean(),
                                                               100 * fpr.mean()))
            cells.append("%.1f" % (100 * F_.mean()))
        lines.append("%s & %s \\\\" % (NAME[sc], " & ".join(cells)))
    open(os.path.join(a.out, "tab_f1.tex"), "w").write("\n".join(lines) + "\n")
    if ("dofa_generic_none", "pavia") in cms and ("dofa_generic_gtn", "pavia") in cms:
        fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.1), gridspec_kw=dict(wspace=0.08))
        for j, (m, t) in enumerate((("dofa_generic_none", "DOFA, no correction"), ("dofa_generic_gtn", "DOFA + GTN"))):
            cm = cms[(m, "pavia")].astype(float)
            cmn = cm / cm.sum(1, keepdims=True)
            ax[j].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax[j].grid(False)
            for r in range(len(cm)):
                for c in range(len(cm)):
                    if cmn[r, c] >= 0.01:
                        ax[j].text(c, r, "%.0f" % (100 * cmn[r, c]), ha="center", va="center", fontsize=6.5,
                                   color="white" if cmn[r, c] > 0.55 else "#0b0b0b")
            ax[j].set_xticks(range(len(PAVIA))); ax[j].set_xticklabels(PAVIA, rotation=45, ha="right", fontsize=7)
            ax[j].set_yticks(range(len(PAVIA))); ax[j].set_yticklabels(PAVIA if j == 0 else [], fontsize=7)
            ax[j].set_title(t, fontsize=8.5, pad=3); ax[j].set_xlabel("Predicted")
        ax[0].set_ylabel("True class")
        fig.savefig(os.path.join(a.out, "fig_confusion_pavia.pdf")); plt.close(fig)

open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(summary) + "\n")
print("written to", a.out)
