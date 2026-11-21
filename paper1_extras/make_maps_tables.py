"""Builds the classification-map figure and the per-class accuracy tables from results/paper1_extras/*.npz.

  python paper1_extras/make_maps_tables.py --res results/paper1_extras --out paper1_extras/output
Outputs: fig_maps.pdf (+ .png), tab_perclass_<scene>.tex, summary.txt
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

p = argparse.ArgumentParser()
p.add_argument("--res", default="results/paper1_extras")
p.add_argument("--out", default="paper1_extras/output")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

SCENES = [("ip", "Indian Pines"), ("salinas", "Salinas"), ("pavia", "Pavia University")]
METHODS = [("svm_sp33", "SVM + $33{\\times}33$ mean"), ("ssrn", "SSRN"),
           ("dofa_generic_none", "DOFA, no corr."), ("dofa_generic_gtn", "DOFA + GTN")]
CLASSES = {
    "ip": ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
           "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
           "Bldg-grass-trees-drives", "Stone-steel-towers"],
    "salinas": ["Brocoli green weeds 1", "Brocoli green weeds 2", "Fallow", "Fallow rough plow", "Fallow smooth",
                "Stubble", "Celery", "Grapes untrained", "Soil vinyard develop", "Corn senesced green weeds",
                "Lettuce romaine 4wk", "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk",
                "Vinyard untrained", "Vinyard vertical trellis"],
    "pavia": ["Trees", "Asphalt", "Bricks", "Bitumen", "Shadows", "Meadows", "Bare soil"],
}
PALETTE = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
                    [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195]]) / 255.0


def path(m, sc):
    if m.startswith("dofa"):
        return os.path.join(a.res, "pred_dofa_%s_%s_s%d.npz" % (sc, m[5:], a.seed))
    return os.path.join(a.res, "pred_%s_%s_s%d.npz" % (m, sc, a.seed))


def to_img(d, which):
    H, W = d["shape"]
    img = np.zeros(H * W, dtype=int)
    img[d["idx"]] = d[which]
    img = img.reshape(H, W)
    if H == 149 and W == 149:                        # Indian Pines: drop the 2-pixel zero border
        img = img[2:-2, 2:-2]
    return img


avail = {(m, sc): np.load(path(m, sc)) for sc, _ in SCENES for m, _ in METHODS if os.path.exists(path(m, sc))}
summary = []
for sc, name in SCENES:
    for m, mname in METHODS:
        if (m, sc) in avail:
            d = avail[(m, sc)]
            summary.append("%-8s %-18s OA %.2f AA %.2f kappa %.2f" % (sc, m, 100 * d["oa"], 100 * d["aa"], 100 * d["kappa"]))
open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(summary) + "\n")
print("\n".join(summary))

# ------------------------------------------------------------------ maps: rows = scenes, cols = GT + methods
cmap = ListedColormap(PALETTE)
ncol = 1 + len(METHODS)
imgs = {}
for sc, _ in SCENES:
    ref = next((avail[(m, sc)] for m, _ in METHODS if (m, sc) in avail), None)
    row = [to_img(ref, "y")] + [to_img(avail[(m, sc)], "pred") if (m, sc) in avail else None for m, _ in METHODS]
    row = [np.rot90(im) if im is not None and im.shape[0] > 1.3 * im.shape[1] else im for im in row]
    imgs[sc] = row
ratios = [imgs[sc][0].shape[0] / imgs[sc][0].shape[1] for sc, _ in SCENES]
colw = 7.16 / ncol
fig, axes = plt.subplots(len(SCENES), ncol, figsize=(7.16, colw * sum(ratios) + 0.75),
                         gridspec_kw=dict(wspace=0.03, hspace=0.16, height_ratios=ratios))
titles = ["Ground truth"] + [mn for _, mn in METHODS]
for r, (sc, name) in enumerate(SCENES):
    for c in range(ncol):
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        im = imgs[sc][c]
        if im is None:
            continue
        ax.imshow(im, cmap=cmap, vmin=0, vmax=16, interpolation="nearest", aspect="equal")
        if r == 0:
            ax.set_title(titles[c], fontsize=7, pad=3)
        if c > 0:
            ax.set_xlabel("OA %.1f%%" % (100 * avail[(METHODS[c - 1][0], sc)]["oa"]), fontsize=6.5, labelpad=1.5)
    axes[r, 0].set_ylabel(name, fontsize=7)
fig.savefig(os.path.join(a.out, "fig_maps.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(a.out, "fig_maps.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)

# ------------------------------------------------------------------ per-class tables (test pixels of split 0)
for sc, name in SCENES:
    cols = [(m, mn) for m, mn in METHODS if (m, sc) in avail]
    if not cols:
        continue
    pcs = [avail[(m, sc)]["per_class"] for m, _ in cols]
    ref = avail[(cols[0][0], sc)]
    y = ref["y"][int(ref["n_train"]) + int(ref["n_val"]):]
    lines = []
    for k, cname in enumerate(CLASSES[sc]):
        vals = [100 * pc[k] for pc in pcs]
        best = max(vals)
        cells = ["\\textbf{%.1f}" % v if abs(v - best) < 1e-9 else "%.1f" % v for v in vals]
        lines.append("%d & %s & %d & %s \\\\" % (k + 1, cname, int((y == k + 1).sum()), " & ".join(cells)))
    lines.append("\\midrule")
    for key, lab in (("oa", "OA"), ("aa", "AA"), ("kappa", "$\\kappa$")):
        vals = [100 * float(avail[(m, sc)][key]) for m, _ in cols]
        best = max(vals)
        lines.append("\\multicolumn{3}{l}{%s} & %s \\\\" % (lab, " & ".join(
            "\\textbf{%.1f}" % v if abs(v - best) < 1e-9 else "%.1f" % v for v in vals)))
    open(os.path.join(a.out, "tab_perclass_%s.tex" % sc), "w").write("\n".join(lines) + "\n")
print("written to", a.out)
