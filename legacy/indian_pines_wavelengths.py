"""
Wavelength list for Indian Pines' 200 retained bands.

Band-removal spec is well-established and cross-confirmed across multiple
papers citing the dataset's original documentation (Baumgardner, Biehl &
Landgrebe, 2015, Purdue Research Repository, doi:10.4231/R7RX991C):
  - AVIRIS sensor covers 220 nominal bands, ~400-2500 nm
  - Bands removed (1-indexed): 104-108, 150-163, 220 (20 bands total)
  - 200 bands remain

CAVEAT: this still assumes uniform spacing across the 220 nominal bands to
derive each *retained* band's center wavelength. Real AVIRIS bands are not
perfectly uniformly spaced (band width varies slightly across the range).
This is a substantial improvement over the earlier approximation (correct
band-removal indices, correct band count basis) but is still not pulled
from AVIRIS's actual per-band calibration file. Flag this limitation in
any written report of these results.
"""
import numpy as np

full_220_nm = np.linspace(400, 2500, 220)

# 1-indexed removed bands -> convert to 0-indexed
removed_1idx = list(range(104, 109)) + list(range(150, 164)) + [220]
removed_0idx = [b - 1 for b in removed_1idx]

keep_mask = np.ones(220, dtype=bool)
keep_mask[removed_0idx] = False

wavelengths_nm = full_220_nm[keep_mask]
assert len(wavelengths_nm) == 200, f"Expected 200 bands, got {len(wavelengths_nm)}"

wavelengths_um = (wavelengths_nm / 1000.0).tolist()

if __name__ == "__main__":
    print(f"Generated {len(wavelengths_um)} wavelengths")
    print(f"Range: {wavelengths_um[0]:.3f} - {wavelengths_um[-1]:.3f} um")
    print(wavelengths_um[:5], "...")
