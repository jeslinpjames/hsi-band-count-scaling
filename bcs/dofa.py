"""DOFA model construction and the adaptation methods compared in the paper."""
import os
import sys

import torch
import torch.nn as nn

from .paths import DOFA_REPO, DOFA_WEIGHTS

# method -> LoRA target modules (None = no LoRA)
LORA_TARGETS = {"generic": ["qkv"], "spectral": ["fc_weight", "fc_bias"], "both": ["qkv", "fc_weight", "fc_bias"]}
METHODS = ("generic", "spectral", "both", "full", "linear", "bitfit")
# settings of the paper (rank, alpha, learning rate)
PAPER = {"generic": dict(r=32, alpha=64, lr=6e-4), "spectral": dict(r=4, alpha=8, lr=2e-4), "full": dict(r=8, alpha=16, lr=2e-5)}


def add_dofa_to_path():
    if not os.path.isfile(os.path.join(DOFA_REPO, "dofa_v1.py")):
        raise FileNotFoundError("DOFA code not found in %s - see README (git clone + checkout, or set DOFA_REPO)" % DOFA_REPO)
    if DOFA_REPO not in sys.path:
        sys.path.insert(0, DOFA_REPO)


def build(num_outputs, size="base"):
    """DOFA ViT with pretrained weights and a new linear head with num_outputs outputs.

    Note: DOFA's wave_dynamic_layer.py calls torch.manual_seed() at import time. Import dofa_v1 (or call
    add_dofa_to_path() and import it) before seeding your run, as scripts/train_dofa.py does."""
    add_dofa_to_path()
    from dofa_v1 import vit_base_patch16, vit_large_patch16
    model = vit_base_patch16() if size == "base" else vit_large_patch16()
    ckpt = os.path.join(DOFA_WEIGHTS, "DOFA_ViT_%s_e100.pth" % size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.head = nn.Linear(model.head.in_features, num_outputs)
    return model


def adapt(model, method, r=8, alpha=16):
    """Freezes / wraps the model for one adaptation method. Returns the (possibly wrapped) model."""
    if method in LORA_TARGETS:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.1, target_modules=LORA_TARGETS[method],
                         modules_to_save=["head"])
        return get_peft_model(model, cfg)
    for name, p in model.named_parameters():
        if method == "full":
            p.requires_grad = True
        elif method == "linear":
            p.requires_grad = name.startswith("head")
        elif method == "bitfit":
            p.requires_grad = ("bias" in name) or name.startswith("head")
        else:
            raise ValueError(method)
    return model
