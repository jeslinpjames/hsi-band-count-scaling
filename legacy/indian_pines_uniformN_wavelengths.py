"""Band-count sweep: N bands spread evenly over the full range. N comes from env var NBANDS.
Exports the same names as the VNIR module so training scripts are drop-in copies."""
import os, numpy as np
from indian_pines_wavelengths import wavelengths_um
N = int(os.environ['NBANDS'])
vnir_mask = sorted(set(np.linspace(0, len(wavelengths_um) - 1, N).round().astype(int).tolist()))
wavelengths_um_vnir = [wavelengths_um[i] for i in vnir_mask]
if __name__ == "__main__":
    print(f"uniform subset: {len(vnir_mask)} bands, {min(wavelengths_um_vnir):.3f}-{max(wavelengths_um_vnir):.3f} um")
