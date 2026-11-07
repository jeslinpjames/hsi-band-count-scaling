"""Checks that the hyperspectral scenes are in place and match the files used in the paper.

  python scripts/check_data.py            (all scenes)
  python scripts/check_data.py pavia      (one scene)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from bcs.data import LABELED, SCENES, load_raw
from bcs.paths import DATA_DIR
from bcs.wavelengths import WAVELENGTHS

EXPECTED = {"ip": ((145, 145, 200), 16), "salinas": ((512, 217, 204), 16), "pavia": ((610, 340, 102), 7),
            "houston13": ((954, 210, 48), 7)}
ok = True
for scene in (sys.argv[1:] or list(SCENES)):
    try:
        cube, gt = load_raw(scene)
    except (FileNotFoundError, KeyError, OSError) as e:
        print("%-10s MISSING  (%s)  expected %s under %s" % (scene, e, SCENES[scene][0], DATA_DIR))
        ok = False
        continue
    shape, n_cls = EXPECTED[scene]
    lab = int((gt > 0).sum())
    good = cube.shape == shape and int(gt.max()) == n_cls and lab == LABELED[scene] and len(WAVELENGTHS[scene]) == shape[2]
    ok &= good
    print("%-10s %s  cube %s, %d classes, %d labelled pixels, per class %s" % (
        scene, "OK      " if good else "MISMATCH", cube.shape, int(gt.max()), lab,
        np.bincount(gt.astype(int).reshape(-1))[1:].tolist()))
    if not good:
        print("           expected cube %s, %d classes, %d labelled pixels" % (shape, n_cls, LABELED[scene]))
sys.exit(0 if ok else 1)
