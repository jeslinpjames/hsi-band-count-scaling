"""make_splitfig.py : Fig. 3 of the paper - random versus spatially separated split, split 0.
Needs the ground-truth files (see docs/DATA.md). Run from the repository root:  python analysis/make_splitfig.py"""
import sys, math
sys.path.insert(0, ".")
import numpy as np, scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from bcs.data import load_gt
from bcs.splits import make_split, BLOCK
import os
os.makedirs("analysis/output/figures", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False, "legend.fontsize": 7})
NAMES = {"pavia": "Pavia University", "salinas": "Salinas"}
# 0 background, 1 test, 2 labelled but excluded (in a training region or within 32 px of a training pixel)
CM = ListedColormap(["#ffffff", "#9cc0e8", "#f2cdb5"])

def panel(ax, ds, mode, seed=0):
    name = NAMES[ds]
    gt = load_gt(ds); H, W = gt.shape; gf = gt.reshape(-1); C = int(gt.max())
    tr, va, te = make_split(mode, gt, C, 10, 5, seed)
    m = np.zeros(H * W); m[gf > 0] = 2; m[te] = 1
    ax.imshow(m.reshape(H, W), cmap=CM, vmin=0, vmax=2, interpolation="nearest")
    tv = np.concatenate([tr, va])
    if mode != "random":
        nbx = math.ceil(W / BLOCK)
        for t in np.unique((tv // W // BLOCK) * nbx + (tv % W) // BLOCK):
            ax.add_patch(Rectangle(((t % nbx) * BLOCK - 0.5, (t // nbx) * BLOCK - 0.5), BLOCK, BLOCK, fill=False, ec="#1b1b1b", lw=0.8))
    ax.plot(tv % W, tv // W, "o", ms=1.4, mfc="#c2410c", mec="none")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("%s\n%s" % (name, "random split" if mode == "random" else "separated split"), fontsize=7.5)
    return len(tr), len(va), len(te)

fig, ax = plt.subplots(1, 3, figsize=(7.16, 3.6), gridspec_kw={"width_ratios": [340, 340, 217]})
for a, (ds, mode), lab in zip(ax, [("pavia", "random"), ("pavia", "separated"), ("salinas", "separated")], "abc"):
    n = panel(a, ds, mode)
    a.text(0, -0.02, "(%s) train %d, val %d, test %d" % (lab, *n), transform=a.transAxes, va="top", fontsize=6.5)
fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#c2410c", mec="none", ms=3.5, label="training and validation pixels"),
                    Patch(fc="none", ec="#1b1b1b", label=r"training region ($32\times32$ tile)"),
                    Patch(fc="#9cc0e8", label="test pixels"),
                    Patch(fc="#f2cdb5", label="labeled, not used (inside a region or < 33 px from a training pixel)")],
           loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.tight_layout(w_pad=0.6)
fig.savefig("analysis/output/figures/fig_splits.pdf"); fig.savefig("analysis/output/figures/fig_splits.png", dpi=200)
