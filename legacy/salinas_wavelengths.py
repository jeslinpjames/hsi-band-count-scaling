"""
Wavelength list for Salinas' 204 retained bands (AVIRIS sensor).
Same sensor family as Indian Pines (400-2500nm nominal range), but Salinas
retains 204 bands (vs Indian Pines' 200) after removing a slightly
different water-absorption band set. As with Indian Pines, this assumes
uniform spacing across the nominal 224-band range as an approximation --
not real per-band calibration.
"""
import numpy as np

full_224_nm = np.linspace(400, 2500, 224)

# Salinas' commonly-cited removed bands (1-indexed): 108-112, 154-167, 224
removed_1idx = list(range(108, 113)) + list(range(154, 168)) + [224]
removed_0idx = [b - 1 for b in removed_1idx]

keep_mask = np.ones(224, dtype=bool)
keep_mask[removed_0idx] = False

wavelengths_nm = full_224_nm[keep_mask]
print(f"Bands retained: {len(wavelengths_nm)}")  # sanity check before asserting

wavelengths_um = (wavelengths_nm / 1000.0).tolist()

if __name__ == "__main__":
    print(f"Generated {len(wavelengths_um)} wavelengths")
    print(f"Range: {wavelengths_um[0]:.3f} - {wavelengths_um[-1]:.3f} um")
