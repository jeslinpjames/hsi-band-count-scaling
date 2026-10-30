"""fix_variants_patch.py : alternative corrections for DOFA's band-count scaling, for the comparison
"why min(1, REF/N) and not something else?". Selected with the environment variable FIX_VARIANT:

  sqrt63   input x min(1, sqrt(63/N))   - the correction you would choose if band contributions were
                                           uncorrelated (norm ~ sqrt(N)); identity for N <= 63
  ref9     input x min(1, 9/N)          - linear, but down to the multispectral scale DOFA saw most (9 bands)
  tokenln  LayerNorm (no learned parameters) on every patch token after the embedding - the generic
           architectural fix; makes the token norm independent of N, but changes ALL inputs, also N <= 63
  safe63   input x min(1, 63/N)         - the paper's correction (same as bandnorm_safe_patch), for checks

Import it once, before the model is built (the training scripts do this at the top).
"""
import os, sys, math
sys.path.insert(0, "../repos/DOFA")
import torch.nn.functional as F
import wave_dynamic_layer as _wdl

VARIANT = os.environ.get("FIX_VARIANT", "")
if VARIANT not in ("sqrt63", "ref9", "tokenln", "safe63"):
    raise SystemExit("fix_variants_patch: set FIX_VARIANT to sqrt63, ref9, tokenln or safe63 (got %r)" % VARIANT)

_cls = _wdl.Dynamic_MLP_OFA
_orig_forward = _cls.forward

def _scale(n):
    if VARIANT == "sqrt63":
        return min(1.0, math.sqrt(63.0 / n))
    if VARIANT == "ref9":
        return min(1.0, 9.0 / n)
    if VARIANT == "safe63":
        return min(1.0, 63.0 / n)
    return 1.0

def _forward(self, img_feat, wvs, *args, **kwargs):
    s = _scale(img_feat.shape[1])
    out = _orig_forward(self, img_feat * s if s != 1.0 else img_feat, wvs, *args, **kwargs)
    if VARIANT == "tokenln":
        x, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, ())
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
        out = (x,) + tuple(rest) if isinstance(out, tuple) else x
    return out

_cls.forward = _forward
print("[fix-variant] %s active on %s.forward" % (VARIANT, _cls.__name__), flush=True)
