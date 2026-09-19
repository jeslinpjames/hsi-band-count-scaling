"""
VNIR-only subset of Indian Pines' wavelength range, for the controlled
ablation testing the spectral-coverage hypothesis. Restricts to bands
whose approximate wavelength falls under ~1000nm (matching the load_data
truncation below).
"""
from indian_pines_wavelengths import wavelengths_um
import numpy as np

# indices where wavelength < 1.0 um (VNIR cutoff)
vnir_mask = [i for i, w in enumerate(wavelengths_um) if w < 1.0]
wavelengths_um_vnir = [wavelengths_um[i] for i in vnir_mask]

if __name__ == "__main__":
    print(f"VNIR-only bands: {len(wavelengths_um_vnir)} of {len(wavelengths_um)}")
