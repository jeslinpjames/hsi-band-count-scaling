"""Adds the WHU-Hi-LongKou scene ("longkou") to the repository and checks the data. Safe to run more than once.

Data (MATLAB format, academic use; cite Zhong et al., Remote Sens. Environ. 250, 112012, 2020) goes in
  $BCS_DATA/WHU-Hi-LongKou/WHU_Hi_LongKou.mat      (550 x 400 x 270, 400-1000 nm, Headwall Nano-Hyperspec)
  $BCS_DATA/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat   (9 classes, 204,542 labelled pixels)

  python paper1_analysis/add_longkou.py            # patch + check
  python paper1_analysis/add_longkou.py --no_check # patch only
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def edit(rel, fn, marker, what):
    path = os.path.join(ROOT, rel)
    s = open(path).read()
    if marker in s:
        print("already done: %s (%s)" % (rel, what))
        return
    new = fn(s)
    if new == s:
        sys.exit("cannot patch %s (%s): pattern not found" % (rel, what))
    open(path, "w").write(new)
    print("patched: %s (%s)" % (rel, what))


def dict_add(name, entry):
    """Inserts entry as the first item of the dict literal assigned to name (one-line or multi-line)."""
    return lambda s: re.sub(r"(^%s\s*=\s*\{)" % name, r"\g<1>%s, " % entry.replace("\\", "\\\\"), s, count=1, flags=re.M)


SCENE = ('"longkou": ("WHU-Hi-LongKou/WHU_Hi_LongKou.mat", ("WHU_Hi_LongKou", "data"), '
         '"WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat", ("WHU_Hi_LongKou_gt", "gt", "map"))')
edit("bcs/data.py", dict_add("SCENES", SCENE), '"longkou": ("WHU-Hi-LongKou', "scene files")
edit("bcs/data.py", dict_add("LABELED", '"longkou": 204542'), '"longkou": 204542', "labelled-pixel count")
edit("bcs/wavelengths.py",
     lambda s: re.sub(r"^(WAVELENGTHS\s*=)", "# WHU-Hi-LongKou, 270 bands (Headwall Nano-Hyperspec, 400-1000 nm)\n"
                      "LONGKOU = (np.linspace(400, 1000, 270) / 1000.0).tolist()\n\\1", s, count=1, flags=re.M),
     "LONGKOU = ", "wavelength grid")
edit("bcs/wavelengths.py", dict_add("WAVELENGTHS", '"longkou": LONGKOU'), '"longkou": LONGKOU', "wavelength table")
edit("bcs/wavelengths.py", dict_add("BANDWIDTH_NM", '"longkou": 6.0'), '"longkou": 6.0', "bandwidth table")
for rel in ("scripts/classic_baselines.py", "scripts/deep_baselines.py", "scripts/train_dofa.py"):
    edit(rel, lambda s: re.sub(r'("--dataset",\s*required=True,\s*choices=\[)', r'\g<1>"longkou", ', s, count=1),
         '"longkou", ', "--dataset choice")

if "--no_check" not in sys.argv:
    sys.path.insert(0, ROOT)
    import numpy as np
    from bcs.data import load
    cube, gt, wl, _ = load("longkou")
    n_lab = int((gt > 0).sum())
    print("longkou: cube %s, %d classes, %d labelled pixels, %d bands %.3f-%.3f um, cube mean %.3f std %.3f" % (
        cube.shape, int(gt.max()), n_lab, len(wl), wl[0], wl[-1], cube.mean(), cube.std()))
    print("pixels per class:", np.bincount(gt.astype(int).reshape(-1))[1:].tolist())
    ok = cube.shape[2] == 270 and int(gt.max()) == 9 and n_lab == 204542 and np.isfinite(cube).all()
    print("CHECK OK" if ok else "CHECK FAILED: expected 270 bands, 9 classes, 204542 labelled pixels")
    sys.exit(0 if ok else 1)
