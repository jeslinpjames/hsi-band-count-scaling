"""Scene loading, per-band standardisation, window extraction, metrics and the per-run CSV format."""
import os
import numpy as np
import scipy.io as sio

from .paths import DATA_DIR
from .wavelengths import WAVELENGTHS, BANDWIDTH_NM, band_subset

# scene -> (cube file, accepted keys, ground-truth file, accepted keys). Paths are relative to BCS_DATA.
# Indian Pines: the copy distributed with HyperSIGMA uses keys data/groundT, the EHU original
# indian_pines_corrected/indian_pines_gt; both are accepted.
SCENES = {
    "ip": ("IndianPines/Indian_pines_corrected.mat", ("data", "indian_pines_corrected"),
           "IndianPines/Indian_pines_gt.mat", ("groundT", "indian_pines_gt")),
    "salinas": ("Salinas/Salinas_corrected.mat", ("salinas_corrected",),
                "Salinas/Salinas_gt.mat", ("salinas_gt",)),
    "pavia": ("PaviaU/paviaU.mat", ("ori_data",), "PaviaU/paviaU_7gt.mat", ("map",)),
    "houston13": ("Houston/Houston13.mat", ("ori_data",), "Houston/Houston13_7gt.mat", ("map",)),
}
LABELED = {"ip": 10249, "salinas": 54129, "pavia": 39332, "houston13": 2530}   # labelled pixels, for checks


def _read(path, keys):
    try:
        m = sio.loadmat(path)
        for k in keys:
            if k in m:
                return np.asarray(m[k]), False
        raise KeyError("%s has none of the keys %s (found %s)" % (path, keys, [k for k in m if not k.startswith("__")]))
    except NotImplementedError:                     # MATLAB v7.3 (HDF5) file, e.g. the Houston release
        import h5py
        with h5py.File(path, "r") as f:
            for k in keys:
                if k in f:
                    return np.array(f[k]), True
        raise KeyError("%s has none of the keys %s" % (path, keys))


def load_raw(scene):
    """Cube (H, W, B) float32 as stored, ground truth (H, W) float32 (0 = unlabelled)."""
    fc, kc, fg, kg = SCENES[scene]
    cube, h5 = _read(os.path.join(DATA_DIR, fc), kc)
    gt, _ = _read(os.path.join(DATA_DIR, fg), kg)
    if h5:                                          # HDF5 stores the cube band-first and transposed
        cube = cube.transpose(1, 2, 0)
    if scene == "ip" and cube.shape[:2] == (145, 145):
        cube, gt = _pad_ip(cube, gt)
    return cube.astype(np.float32), gt.astype(np.float32)


# The Indian Pines copy used in the paper (distributed with HyperSIGMA) is the 145 x 145 scene surrounded by a
# 2-pixel border of zeros (149 x 149). The border has no labels, but it enters the per-band standardisation and the
# window coordinates. The EHU original (145 x 145) is padded in the same way, so both copies give identical inputs.
IP_BORDER = 2


def _pad_ip(cube, gt):
    b = IP_BORDER
    return (np.pad(cube, ((b, b), (b, b), (0, 0))), np.pad(gt, ((b, b), (b, b))))


def load_gt(scene):
    """Ground truth only (H, W) float32 (Indian Pines padded to 149 x 149 as in load_raw)."""
    _, _, fg, kg = SCENES[scene]
    gt = _read(os.path.join(DATA_DIR, fg), kg)[0]
    if scene == "ip" and gt.shape == (145, 145):
        gt = np.pad(gt, IP_BORDER)
    return gt.astype(np.float32)


def standardise(cube):
    """Per-band standardisation over the whole scene (mean 0, standard deviation 1 for every band)."""
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    mu, sigma = flat.mean(0), flat.std(0) + 1e-8
    return ((flat - mu) / sigma).reshape(H, W, B).astype(np.float32)


def load(scene, bands="all"):
    """Standardised cube (H, W, B'), ground truth (H, W) float32, wavelengths (um) of the B' bands, bandwidth (nm).

    Standardisation is done on all bands BEFORE a band subset is taken, as in the experiments of the paper."""
    cube, gt = load_raw(scene)
    cube = standardise(cube)
    wl = WAVELENGTHS[scene]
    assert len(wl) == cube.shape[2], (scene, len(wl), cube.shape)
    sel = band_subset(wl, bands)
    if len(sel) != len(wl):
        cube = cube[:, :, np.asarray(sel)]
        wl = [wl[i] for i in sel]
    return cube, gt, wl, BANDWIDTH_NM[scene]


class Windows:
    """size x size windows around flat pixel indices, zero-padded at the border, cut on demand.

    Identical values to HyperSIGMA's create_patches(..., window_size=size) used in the original runs, without
    materialising the windows of every pixel of the scene."""

    def __init__(self, cube, size):
        self.r, self.size, self.W = size // 2, size, cube.shape[1]
        self.pad = np.pad(cube, ((self.r, self.r), (self.r, self.r), (0, 0)), mode="constant")

    def __call__(self, idx):
        idx = np.asarray(idx)
        rows, cols = idx // self.W, idx % self.W
        return np.stack([self.pad[i:i + self.size, j:j + self.size] for i, j in zip(rows, cols)]).astype(np.float32)


def scores(y_true, y_pred, class_num):
    """Overall accuracy, average (per-class) accuracy and Cohen's kappa."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, class_num + 1)))
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return accuracy_score(y_true, y_pred), per_class.mean(), cohen_kappa_score(y_true, y_pred)


# ---------------------------------------------------------------------------- per-run CSV (one row per run)
# The result files of the paper have the first twelve columns; runs made with this repository add best_val (the
# validation OA of the selected epoch), which the learning-rate selection uses.
HEADER = "method,train_num,seed,lora_r,OA,AA,kappa,trainable,trainable_pct,best_epoch,epochs_run,train_time_s"


def is_done(csv, method, train_num, seed, r=0):
    key = "%s,%d,%d,%d," % (method, train_num, seed, r)
    return os.path.exists(csv) and any(line.startswith(key) for line in open(csv))


def append_row(csv, method, train_num, seed, r, oa, aa, kappa, trainable, pct, best_ep, ep_run, t, best_val=None):
    """Appends one run; safe when several runs write to the same file at once (file lock, Linux/macOS)."""
    import fcntl
    os.makedirs(os.path.dirname(os.path.abspath(csv)), exist_ok=True)
    with open(csv, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        new = f.tell() == 0
        with_val = new or open(csv).readline().strip().endswith(",best_val")
        if new:
            f.write(HEADER + ",best_val\n")
        row = "%s,%d,%d,%d,%.4f,%.4f,%.4f,%d,%.4f,%d,%d,%.1f" % (
            method, train_num, seed, r, oa, aa, kappa, trainable, pct, best_ep, ep_run, t)
        if with_val:
            row += ",%s" % ("" if best_val is None else "%.4f" % best_val)
        f.write(row + "\n")
