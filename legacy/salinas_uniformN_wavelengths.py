"""salinas: N bands spread evenly over the full range. N from env var NBANDS."""
import os, numpy as np
from salinas_wavelengths import wavelengths_um as _full
N = int(os.environ['NBANDS'])
band_mask = sorted(set(np.linspace(0, len(_full) - 1, N).round().astype(int).tolist()))
wavelengths_um = [_full[i] for i in band_mask]
