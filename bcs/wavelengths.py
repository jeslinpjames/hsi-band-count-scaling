"""Band-centre wavelengths (micrometres) of every scene.

The centres are APPROXIMATED by uniform spacing over the nominal range of each sensor (no calibration files were
used); this is stated as a limitation in the paper. The water-absorption bands removed from the AVIRIS scenes are
omitted from the uniform grid, so each retained band keeps its nominal position.
"""
import numpy as np


def _aviris(n_nominal, removed_1idx):
    grid = np.linspace(400, 2500, n_nominal)
    keep = np.ones(n_nominal, dtype=bool)
    keep[[b - 1 for b in removed_1idx]] = False
    return (grid[keep] / 1000.0).tolist()


# Indian Pines: 220 nominal AVIRIS bands, removed (1-indexed) 104-108, 150-163, 220  -> 200 bands
INDIAN_PINES = _aviris(220, list(range(104, 109)) + list(range(150, 164)) + [220])
# Salinas: 224 nominal AVIRIS bands, removed 108-112, 154-167, 224  -> 204 bands
SALINAS = _aviris(224, list(range(108, 113)) + list(range(154, 168)) + [224])
# Pavia University, 102-band cross-scene release (ROSIS, 430-860 nm)
PAVIA_U = (np.linspace(430, 860, 102) / 1000.0).tolist()
# Houston 2013, 48-band cross-scene release (CASI, 364-1046 nm)
HOUSTON = (np.linspace(364, 1046, 48) / 1000.0).tolist()

assert len(INDIAN_PINES) == 200 and len(SALINAS) == 204

WAVELENGTHS = {"ip": INDIAN_PINES, "salinas": SALINAS, "pavia": PAVIA_U, "houston13": HOUSTON}
BANDWIDTH_NM = {"ip": 10.0, "salinas": 10.0, "pavia": 4.0, "houston13": 10.0}   # nominal, used by Copernicus-FM only


def band_subset(wavelengths_um, spec):
    """Indices of the bands used.

    spec = "all"         every band
           "uniform:N"   N bands spread evenly over the full list (indices linspace(0, B-1, N), rounded, unique)
           "vnir"        bands below 1.0 um
    """
    B = len(wavelengths_um)
    if spec in (None, "", "all"):
        return list(range(B))
    if spec.startswith("uniform:"):
        n = int(spec.split(":")[1])
        return sorted(set(np.linspace(0, B - 1, n).round().astype(int).tolist()))
    if spec == "vnir":
        return [i for i, w in enumerate(wavelengths_um) if w < 1.0]
    raise ValueError("unknown band subset %r" % spec)
