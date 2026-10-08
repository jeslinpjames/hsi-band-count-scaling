"""Safe band-count normalisation for DOFA: multiplies the input by s = min(1, REF/N_bands)
before the dynamic patch embedding (equivalent to averaging band contributions and rescaling
to REF bands). Never amplifies sensors with fewer than REF bands. REF from env BANDNORM_REF (default 63)."""
import os
import wave_dynamic_layer as _wdl
REF = float(os.environ.get('BANDNORM_REF', '63'))
_orig_forward = _wdl.Dynamic_MLP_OFA.forward
def _forward(self, img_feat, wvs, *args, **kwargs):
    s = min(1.0, REF / img_feat.shape[1])
    if s < 1.0:
        img_feat = img_feat * s
    return _orig_forward(self, img_feat, wvs, *args, **kwargs)
_wdl.Dynamic_MLP_OFA.forward = _forward
print(f"[bandnorm-safe] input scaled by min(1, {REF:g}/N_bands)", flush=True)
