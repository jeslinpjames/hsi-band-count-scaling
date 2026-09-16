"""
Approximate wavelength list for this Pavia University variant's 102 bands
(Data-CSHSI mirror, cross-scene-screened -- 102 bands, 7 classes, not the
'standard' 103-band/9-class PaviaU distribution). ROSIS sensor, nominal
430-860nm range. Same uniform-spacing caveat as Indian Pines/the original
103-band Pavia estimate -- not real per-band calibration.
"""
import numpy as np

wavelengths_nm = np.linspace(430, 860, 102)
wavelengths_um = (wavelengths_nm / 1000.0).tolist()

if __name__ == "__main__":
    print(f"Generated {len(wavelengths_um)} wavelengths")
    print(f"Range: {wavelengths_um[0]:.3f} - {wavelengths_um[-1]:.3f} um")
