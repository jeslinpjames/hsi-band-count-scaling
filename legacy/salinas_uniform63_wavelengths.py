"""Salinas reduced to 63 bands spread evenly over the full range (replicates the IP band-count effect)."""
import numpy as np
from salinas_wavelengths import wavelengths_um as _full
sal_mask = sorted(set(np.linspace(0, len(_full) - 1, 63).round().astype(int).tolist()))
wavelengths_um = [_full[i] for i in sal_mask]
if __name__ == "__main__":
    print(f"Salinas uniform subset: {len(sal_mask)} of {len(_full)} bands, {min(wavelengths_um):.3f}-{max(wavelengths_um):.3f} um")
