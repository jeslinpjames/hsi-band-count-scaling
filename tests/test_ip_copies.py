"""The EHU Indian Pines file (145 x 145) and the HyperSIGMA copy used in the paper (149 x 149, 2-pixel zero border)
must give identical standardised inputs and ground truth."""
import importlib

import numpy as np
import scipy.io as sio


def _write(root, cube, gt, keys):
    (root / "IndianPines").mkdir(parents=True)
    sio.savemat(root / "IndianPines/Indian_pines_corrected.mat", {keys[0]: cube})
    sio.savemat(root / "IndianPines/Indian_pines_gt.mat", {keys[1]: gt})


def test_both_copies_identical(tmp_path, monkeypatch):
    r = np.random.default_rng(0)
    cube = r.integers(955, 9000, (145, 145, 200)).astype(np.uint16)
    gt = r.integers(0, 17, (145, 145)).astype(np.uint8)
    _write(tmp_path / "hypersigma", np.pad(cube, ((2, 2), (2, 2), (0, 0))), np.pad(gt, 2), ("data", "groundT"))
    _write(tmp_path / "ehu", cube, gt, ("indian_pines_corrected", "indian_pines_gt"))
    out = []
    for copy in ("hypersigma", "ehu"):
        monkeypatch.setenv("BCS_DATA", str(tmp_path / copy))
        import bcs.paths, bcs.data
        importlib.reload(bcs.paths)
        importlib.reload(bcs.data)
        c, g, _, _ = bcs.data.load("ip")
        out.append((c, g, bcs.data.load_gt("ip")))
    for a, b in zip(*out):
        assert a.shape[:2] == (149, 149)
        assert np.array_equal(a, b)
