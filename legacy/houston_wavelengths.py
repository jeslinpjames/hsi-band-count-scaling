"""
Wavelength list for Houston 2013/2018's 48 bands (CASI sensor, VNIR only).
Nominal range ~364-1046nm. Same uniform-spacing approximation as other
datasets in this project -- not real per-band calibration.
Used for both Houston13 and Houston18 scenes (Data-CSHSI cross-scene
benchmark keeps identical band count/wavelength coverage across both).
"""
import numpy as np

wavelengths_nm = np.linspace(364, 1046, 48)
wavelengths_um = (wavelengths_nm / 1000.0).tolist()

if __name__ == "__main__":
    print(f"Generated {len(wavelengths_um)} wavelengths")
    print(f"Range: {wavelengths_um[0]:.3f} - {wavelengths_um[-1]:.3f} um")
