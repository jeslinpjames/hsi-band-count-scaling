"""Control for the VNIR ablation: same band COUNT as VNIR-only,
spread evenly across the full ~400-2500 nm range (includes SWIR).
Exports the same names as the VNIR module so the training script is a drop-in copy."""
import numpy as np
from indian_pines_wavelengths import wavelengths_um
from indian_pines_vnir_wavelengths import vnir_mask as _vnir
N = len(_vnir)
vnir_mask = sorted(set(np.linspace(0, len(wavelengths_um) - 1, N).round().astype(int).tolist()))
wavelengths_um_vnir = [wavelengths_um[i] for i in vnir_mask]
if __name__ == "__main__":
    print(f"uniform subset: {len(vnir_mask)} bands, {min(wavelengths_um_vnir):.3f}-{max(wavelengths_um_vnir):.3f} um")
