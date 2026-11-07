"""The band-count corrections studied in the paper.

  none     unchanged model
  bcr      band-count rescaling: input x min(1, R/N)                       (paper default, R = 63)
  gtn      gated token normalisation: layer norm without learned parameters of every patch token when N > R
  sqrt     input x min(1, sqrt(R/N))     - alternative: correct if band responses were unrelated (norm ~ sqrt N)
  ref9     input x min(1, 9/N)            - alternative: rescale to the 9-band level
  ln       UNGATED token normalisation (applied for every N) - used only to show why GTN needs its gate

For N > R, gtn and ln are identical. In the result file names, none / bcr / gtn / sqrt / ref9 / ln appear as
nofix / fix63 / tokenln / sqrt63 / ref9 / tokenln (the ungated runs are the 25-band and 48-band inputs).
"""
import math
import torch.nn.functional as F

R_DEFAULT = 63
CHOICES = ("none", "bcr", "gtn", "sqrt", "ref9", "ln")
FILE_TAG = {"none": "nofix", "bcr": "fix63", "gtn": "tokenln", "sqrt": "sqrt63", "ref9": "ref9", "ln": "tokenln"}


def input_scale(correction, n_bands, R=R_DEFAULT):
    """Factor applied to the input before the band sum (1 for the token-level corrections)."""
    if correction == "bcr":
        return min(1.0, R / float(n_bands))
    if correction == "sqrt":
        return min(1.0, math.sqrt(R / float(n_bands)))
    if correction == "ref9":
        return min(1.0, 9.0 / n_bands)
    return 1.0


def normalises_tokens(correction, n_bands, R=R_DEFAULT):
    return correction == "ln" or (correction == "gtn" and n_bands > R)


def token_norm(x):
    """Layer normalisation over the channel dimension, no learned scale or shift (eps as in the paper's runs)."""
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


_PATCHED = {}


def patch_dofa(correction, R=R_DEFAULT, dofa_module=None):
    """Applies a correction to DOFA by wrapping Dynamic_MLP_OFA.forward (the dynamic patch embedding).

    Call once, before the model is used (import DOFA before seeding: its wave_dynamic_layer.py reseeds torch at
    import time). Equivalent to the patch modules of the original runs
    (bandnorm_safe_patch.py, fix_variants_patch.py; see legacy/)."""
    assert correction in CHOICES, correction
    if dofa_module is None:
        import wave_dynamic_layer as dofa_module          # from the DOFA repository (on sys.path)
    cls = dofa_module.Dynamic_MLP_OFA
    orig = _PATCHED.setdefault(cls, cls.forward)          # always wrap the ORIGINAL forward
    if correction == "none":
        cls.forward = orig
        return

    def forward(self, img_feat, wvs, *args, **kwargs):
        n = img_feat.shape[1]
        s = input_scale(correction, n, R)
        out = orig(self, img_feat * s if s != 1.0 else img_feat, wvs, *args, **kwargs)
        if normalises_tokens(correction, n, R):
            if isinstance(out, tuple):
                out = (token_norm(out[0]),) + tuple(out[1:])
            else:
                out = token_norm(out)
        return out

    cls.forward = forward
