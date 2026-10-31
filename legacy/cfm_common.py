"""cfm_common.py : Copernicus-FM (torchgeo) wrapper with the two corrections of the paper.

Copernicus-FM inherits DOFA's dynamic patch embedding: one kernel per band generated from wavelength (+ bandwidth),
F.conv2d sums over bands, and nothing divides by the number of bands N.

  CORR=none     unchanged
  CORR=bcr      input x min(1, 63/N)                         (band-count rescaling)
  CORR=gtn      layer norm (no parameters) of every patch token when N > 63   (gated token normalisation)
"""
import os
import math
import torch
import torch.nn.functional as F

REF = 63


def build(num_classes=0, weights=True):
    from torchgeo.models import copernicusfm_base, CopernicusFM_Base_Weights
    w = CopernicusFM_Base_Weights.CopernicusFM_ViT if weights else None
    local = os.environ.get("CFM_CKPT")                       # optional: path to a manually downloaded .pth
    if local:
        m = copernicusfm_base(weights=None, num_classes=num_classes)
        sd = torch.load(local, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert set(missing) <= {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}, missing
        assert not unexpected, unexpected[:5]
        return m
    m = copernicusfm_base(weights=w, num_classes=num_classes)
    return m


class Corrected(torch.nn.Module):
    """Calls Copernicus-FM in spectral mode with unknown metadata and applies BCR / GTN."""
    def __init__(self, model, corr="none"):
        super().__init__()
        assert corr in ("none", "bcr", "gtn"), corr
        self.m, self.corr = model, corr
        self._n = None
        if corr == "gtn":
            self.m.patch_embed_spectral.register_forward_hook(self._gtn_hook)

    def _gtn_hook(self, _mod, _inp, out):
        if self._n is not None and self._n > REF:
            return F.layer_norm(out, (out.shape[-1],), eps=1e-6)
        return out

    def forward(self, x, wavelengths_nm, bandwidths_nm):
        n = x.shape[1]; self._n = n
        if self.corr == "bcr" and n > REF:
            x = x * (REF / float(n))
        meta = torch.full((x.shape[0], 4), float("nan"), device=x.device, dtype=x.dtype)
        return self.m(x, meta, wavelengths=list(wavelengths_nm), bandwidths=list(bandwidths_nm), input_mode="spectral")
