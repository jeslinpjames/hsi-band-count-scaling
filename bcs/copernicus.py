"""Copernicus-FM-Base (TorchGeo 0.7.0) with the corrections of the paper.

Copernicus-FM inherits DOFA's dynamic patch embedding: one kernel per band is generated from wavelength and
bandwidth, the convolution sums over the bands, and nothing divides by the number of bands N.
"""
import os

import torch

from .corrections import R_DEFAULT, token_norm

CHOICES = ("none", "bcr", "gtn")


def build(num_classes=0, weights=True):
    """Pretrained Copernicus-FM-Base. Set CFM_CKPT to a local .pth file if the weights cannot be downloaded."""
    from torchgeo.models import CopernicusFM_Base_Weights, copernicusfm_base
    local = os.environ.get("CFM_CKPT")
    if local:
        m = copernicusfm_base(weights=None, num_classes=num_classes)
        missing, unexpected = m.load_state_dict(torch.load(local, map_location="cpu"), strict=False)
        assert set(missing) <= {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}, missing
        assert not unexpected, unexpected[:5]
        return m
    return copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT if weights else None,
                             num_classes=num_classes)


class Corrected(torch.nn.Module):
    """Calls Copernicus-FM in spectral mode with unknown location/time metadata and applies BCR or GTN."""

    def __init__(self, model, correction="none", R=R_DEFAULT):
        super().__init__()
        assert correction in CHOICES, correction
        self.m, self.corr, self.R, self._n = model, correction, R, None
        if correction == "gtn":
            self.m.patch_embed_spectral.register_forward_hook(self._gtn_hook)

    def _gtn_hook(self, _module, _inputs, out):
        if self._n is not None and self._n > self.R:
            return token_norm(out)
        return out

    def forward(self, x, wavelengths_nm, bandwidths_nm):
        n = x.shape[1]
        self._n = n
        if self.corr == "bcr" and n > self.R:
            x = x * (self.R / float(n))
        meta = torch.full((x.shape[0], 4), float("nan"), device=x.device, dtype=x.dtype)
        return self.m(x, meta, wavelengths=list(wavelengths_nm), bandwidths=list(bandwidths_nm), input_mode="spectral")
