# Original scripts

These are the scripts with which the runs in `results/` were made, copied unchanged from the experiment folder
(paths such as `../repos/DOFA` and `../data/` refer to that folder's layout). They are kept for provenance; use the
scripts in `scripts/` to reproduce the results. `docs/REPRODUCE.md` lists which new script replaces which of these.

* `train_dofa*.py`: one copy of the DOFA training script per scene, band subset, correction and split; they differ only
  in the data file, the wavelength module, and the imported patch module.
* `bandnorm_safe_patch.py` (BCR), `bandnorm_patch.py` (rescaling without the min(1, ·), used in early runs),
  `fix_variants_patch.py` (GTN/ungated normalisation, sqrt and 9/N rescaling): wrap DOFA's patch-embedding forward.
* `split_modes.py`, `patch_split.py`: the spatially separated split and the tool that added it to copies of the scripts.
* `lr_sweep.py`, `launchers/`: job launchers (learning-rate sweep, later batches); `queue_lib.sh` shows the default
  settings of the early runs (learning rate 6e-5, 400 epochs, patience 100).
* `train_fm.py`, `cfm_common.py`, `cfm_tokens.py`: Copernicus-FM. `embed_norm2.py`: DOFA token measurement.
* `baseline_classic.py`, `baseline_spatial.py`, `classic_split.py`, `deep_baselines.py`: baselines.
