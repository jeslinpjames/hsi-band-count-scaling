# Per-run results

One row per training run, in the format written by the training scripts:

| column | meaning |
|---|---|
| method | `generic`, `spectral`, `full`, `linear`, `bitfit`, `both` (DOFA / Copernicus-FM); `svm`, `rf`, `svm_sp7`, `svm_sp33`; `hybridsn`, `ssrn` |
| train_num | labelled training pixels per class |
| seed | split index (the split is drawn with `numpy.random.RandomState(seed)`) |
| lora_r | LoRA rank (8 is written, but unused, for full fine-tuning; 0 for the baselines) |
| OA, AA, kappa | overall accuracy, average accuracy, Cohen's kappa on the test pixels (fractions) |
| trainable, trainable_pct | trainable parameters |
| best_epoch, epochs_run | epoch of the selected (best-validation) weights, last epoch run |
| train_time_s | training time in seconds |

File names: `results_<group>_<scene>_<correction>_lr<learning rate>.csv`, with scenes `ip` (Indian Pines), `salinas`,
`pavia` (Pavia University, 7-class release), `houston` (Houston 2013), `ip25` (25 evenly spaced Indian Pines bands);
corrections `nofix` (none), `fix63` (BCR), `tokenln` (GTN; ungated token normalisation for `ip25` and `houston`),
`sqrt63`, `ref9`; groups `lr` (learning-rate sweep and main runs), `fixvar` (GTN and alternative corrections), `disj`
(spatially separated split), `cfm_random` (Copernicus-FM). `docs/REPRODUCE.md` maps every file to the paper.

Other files:

* `lr_runs.csv`: validation and test accuracy of the learning-rate sweep (Supplementary Section S2);
  `run_log_full.csv` is the complete log of the run launcher, including validation accuracy of later runs.
* `embed_norm2.csv`, `coherence.csv`, `cfm_embed_norm.csv`, `cfm_coherence.csv`, `CFM_TOKENS_SUMMARY.txt`:
  token norms and coherence (Fig. 2, Supplementary Sections S3 and S10).
* `results_deep_random_ip.csv` contains one extra row with seed 99 from a test of the script; the analysis uses
  seeds 0–15 only.
* `unreported/`: exploratory runs made during the project that are not part of the paper (earlier normalisation
  variants, other label budgets and scenes, BitFit, combined LoRA). They are counted in the total number of runs
  stated in the paper.
