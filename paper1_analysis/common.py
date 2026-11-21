"""Shared helpers for the paper-1 analysis jobs (imports the repository's bcs package)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "paper1_analysis")
os.makedirs(OUT, exist_ok=True)

import numpy as np  # noqa: E402

from bcs import corrections, dofa  # noqa: E402
from bcs.data import Windows, load  # noqa: E402
from bcs.splits import make_split  # noqa: E402

SCENES = ("ip", "salinas", "pavia", "longkou")
NS = (9, 25, 50, 63, 100, 150, "all")


def import_dofa():
    """DOFA's wave_dynamic_layer.py reseeds torch at import: call this BEFORE seeding."""
    dofa.add_dofa_to_path()
    import dofa_v1  # noqa: F401
    import wave_dynamic_layer  # noqa: F401


def load_scene(scene, bands="all"):
    r = load(scene, bands)
    return r[0], r[1], r[2]          # cube, gt, wavelengths (um)


def blocks_of(model):
    """The list of transformer blocks of a (possibly PEFT-wrapped) DOFA model."""
    for name, mod in model.named_modules():
        if name.endswith("blocks") and hasattr(mod, "__len__") and len(mod) > 0:
            return mod
    raise RuntimeError("no transformer blocks found")


def head_of(model):
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "head" and hasattr(mod, "weight"):
            return mod
    raise RuntimeError("no head found")


def patch_embed_of(model):
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "patch_embed":
            return mod
    raise RuntimeError("no patch_embed found")


def first_tensor(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def linear_cka(X, Y):
    """Linear CKA between two (n, d) feature matrices (Kornblith et al., 2019)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro") + 1e-12))


def done(path):
    return os.path.exists(path)
