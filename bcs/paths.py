"""Locations of data, third-party code and weights. Override any of them with environment variables."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("BCS_DATA", os.path.join(ROOT, "data"))                       # hyperspectral scenes
DOFA_REPO = os.environ.get("DOFA_REPO", os.path.join(ROOT, "third_party", "DOFA"))       # github.com/zhu-xlab/DOFA
DOFA_WEIGHTS = os.environ.get("DOFA_WEIGHTS", os.path.join(ROOT, "weights"))           # folder with DOFA_ViT_*_e100.pth
RESULTS_DIR = os.environ.get("BCS_RESULTS", os.path.join(ROOT, "results"))             # per-run CSVs

DOFA_COMMIT = "0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9"   # DOFA revision used for every run in the paper
