"""Band-count normalisation for DOFA: scales the input by REF/N before the dynamic patch
embedding, so the embedding size no longer grows with band count. REF from env var BANDNORM_REF (default 9)."""
import os
import wave_dynamic_layer as _wdl
REF = float(os.environ.get('BANDNORM_REF', '9'))
_orig_forward = _wdl.Dynamic_MLP_OFA.forward
def _forward(self, img_feat, wvs, *args, **kwargs):
    return _orig_forward(self, img_feat * (REF / img_feat.shape[1]), wvs, *args, **kwargs)
_wdl.Dynamic_MLP_OFA.forward = _forward
print(f"[bandnorm] DOFA patch-embedding input scaled by {REF:g}/N_bands", flush=True)
