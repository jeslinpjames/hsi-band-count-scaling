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

# Indian Pines: 145 x 145 scene inside a 2-pixel zero border (see bcs/data.py); both copies are padded to 149 x 149.
EXPECTED = {"ip": ((149, 149, 200), 16), "salinas": ((512, 217, 204), 16), "pavia": ((610, 340, 102), 7),
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
    if scene == "ip":
        import scipy.io as sio
        stored = [shp for name, shp, _ in sio.whosmat(os.path.join(DATA_DIR, SCENES["ip"][0]))
                  if name in SCENES["ip"][1]][0]
        print("ip         stored as %s: %s" % (stored, "HyperSIGMA copy (used in the paper)" if stored[0] == 149
                                                 else "EHU original, padded to 149 x 149 on loading"))
        border = np.concatenate([cube[:2].ravel(), cube[-2:].ravel(), cube[:, :2].ravel(), cube[:, -2:].ravel()])
        if np.abs(border).max() != 0 or gt[:2].max() or gt[-2:].max() or gt[:, :2].max() or gt[:, -2:].max():
            print("           border is not zero: this is not the expected Indian Pines file")
            ok = False
    lab = int((gt > 0).sum())
    good = cube.shape == shape and int(gt.max()) == n_cls and lab == LABELED[scene] and len(WAVELENGTHS[scene]) == shape[2]
    ok &= good
    print("%-10s %s  cube %s, %d classes, %d labelled pixels, per class %s" % (
        scene, "OK      " if good else "MISMATCH", cube.shape, int(gt.max()), lab,
        np.bincount(gt.astype(int).reshape(-1))[1:].tolist()))
    if not good:
        print("           expected cube %s, %d classes, %d labelled pixels" % (shape, n_cls, LABELED[scene]))
sys.exit(0 if ok else 1)
