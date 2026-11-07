"""Unit tests of the corrections on a stand-in for DOFA's dynamic patch embedding (no weights or GPU needed)."""
import math
import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bcs import corrections  # noqa: E402

D, P = 32, 4


def fake_dofa_module():
    """Module with a Dynamic_MLP_OFA whose forward, like DOFA's, sums one generated kernel per band."""
    class Dynamic_MLP_OFA(torch.nn.Module):
        def forward(self, img_feat, wvs):
            g = torch.Generator().manual_seed(0)
            k = 0.01 * torch.randn(D, img_feat.shape[1], P, P, generator=g) + 0.02   # weakly aligned kernels
            out = F.conv2d(img_feat, k, torch.full((D,), 0.01), stride=P)
            return out.flatten(2).transpose(1, 2), wvs                               # (B, tokens, D)
    return types.SimpleNamespace(Dynamic_MLP_OFA=Dynamic_MLP_OFA)


def tokens(mod, n, correction):
    corrections.patch_dofa(correction, dofa_module=mod)
    x = torch.randn(2, n, 16, 16, generator=torch.Generator().manual_seed(n))
    return mod.Dynamic_MLP_OFA()(x, torch.ones(n))[0]


def test_identity_at_or_below_threshold():
    mod = fake_dofa_module()
    for n in (3, 25, 48, 63):
        ref = tokens(mod, n, "none")
        for c in ("bcr", "gtn", "sqrt"):
            assert torch.allclose(tokens(mod, n, c), ref), (c, n)


def test_bcr_scales_input_linearly():
    mod = fake_dofa_module()
    n = 200
    none, bcr = tokens(mod, n, "none"), tokens(mod, n, "bcr")
    bias = 0.01
    assert torch.allclose(bcr - bias, (none - bias) * 63 / n, atol=1e-5)    # kernels scaled, bias not


def test_gtn_fixes_token_norm():
    mod = fake_dofa_module()
    for n in (64, 102, 200):
        t = tokens(mod, n, "gtn")
        assert torch.allclose(t.norm(dim=-1), torch.full(t.shape[:2], math.sqrt(D)), rtol=1e-3)


def test_ungated_ln_changes_few_band_inputs():
    mod = fake_dofa_module()
    assert not torch.allclose(tokens(mod, 25, "ln"), tokens(mod, 25, "none"))


def test_alternative_factors():
    assert corrections.input_scale("sqrt", 252) == math.sqrt(63 / 252)
    assert corrections.input_scale("ref9", 200) == 9 / 200
    assert corrections.input_scale("bcr", 40) == 1.0
