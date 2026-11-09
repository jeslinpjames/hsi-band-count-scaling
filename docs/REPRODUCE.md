# Reproducing the paper

There are two levels of reproduction.

1. **From the per-run results (no GPU, about one minute).** `results/` holds one CSV row for every training run
   behind the paper. The analysis scripts recompute every number, table and figure from these files.
2. **From scratch (GPUs).** `scripts/run_experiments.py` re-runs any experiment and writes CSVs with the same file
   names to a new folder, so that the same analysis can be applied to the new runs.

## 1. Numbers, tables and figures from the per-run results

```bash
python analysis/stats.py        # every number in the text  -> analysis/output/numbers.json (3 174 values)
python analysis/tables.py       # LaTeX bodies of all tables -> analysis/output/*.tex
python analysis/make_figs.py    # all data figures           -> analysis/output/figures/
python analysis/make_splitfig.py   # Fig. 3 (needs the ground-truth files, see docs/DATA.md)
```

`numbers.json` uses flat keys, e.g. `eff.ip.spectral.fix63-nofix.d` is the mean paired OA gain of BCR for Spectral
LoRA on Indian Pines (8.52). `nofix`, `fix63` and `tokenln` denote no correction, BCR and GTN.

## 2. Re-running the experiments

```bash
python scripts/run_experiments.py --list                       # experiments and number of runs
python scripts/run_experiments.py --exp main --gpus 0,1,2,3 --out results/new
python scripts/run_experiments.py --exp copernicus --gpus 0 --out results/new \
       --python_copernicus envs/copernicus/bin/python
BCS_RESULTS=results/new python analysis/stats.py                # analysis of the new runs
```

Runs already present in their CSV are skipped, so an interrupted command can simply be started again. A single run
can also be started directly, e.g.

```bash
python scripts/train_dofa.py --dataset ip --method generic --correction gtn --seed 0 --csv results/new/test.csv
```

Training is not bit-for-bit deterministic on GPUs (cuDNN), so a single run re-run with the same seed can differ from
`results/` by one to two points of OA, and occasionally more; compare means over seeds, not single runs. The splits,
however, are identical (they depend only on the seed). Example, Indian Pines, GTN, seed 0, re-run on the same GPU type:

| Run | Paper (`results/`) | Original script, re-run | This repository |
|---|---|---|---|
| Copernicus-FM, Generic LoRA | 80.20 | 79.26 | 77.30 |
| DOFA, Generic LoRA | 75.48 | – | 73.96 |

All differences are well within the seed-to-seed spread (Copernicus-FM: 71.9–81.6 over seeds 0–7). The original runs used
`OMP_NUM_THREADS=8`; the number of CPU threads also changes floating-point summation order, so keep it at 8 for the
closest match.

## Where each result comes from

| Paper | Experiment (`--exp`) | Result files in `results/` | Analysis output |
|---|---|---|---|
| Fig. 2 (token norm, coherence) | `tokens` | `embed_norm2.csv`, `coherence.csv`, `cfm_embed_norm.csv`, `cfm_coherence.csv` | `figures/fig2_mechanism` |
| Fig. 3 (the two protocols) | – | ground-truth files | `figures/fig_splits` |
| Table 2, DOFA rows | `main` | `results_lr_<scene>_{nofix,fix63}_lr<lr>.csv`, `results_fixvar_<scene>_tokenln_lr<lr>.csv` | `tab_main.tex` |
| Table 2, classical rows | `classic` | `results_classic_<scene>.csv` | `tab_main.tex` |
| Table 2, HybridSN and SSRN | `deep` | `results_deep_random_<scene>.csv` | `tab_main.tex` |
| Table 3 (Copernicus-FM) | `copernicus` | `results_cfm_random_<scene>_<correction>.csv` | `tab_cfm.tex` |
| Fig. 4 (gains of the corrections) | `main`, `copernicus` | as above | `figures/fig3_correction_effects` |
| Fig. 5 (alternative corrections) | `alternatives` | `results_fixvar_{ip,pavia}_{sqrt63,ref9}_lr<lr>.csv` | `figures/fig4_alternatives` |
| Table 4 (ungated normalisation) | `ungated` | `results_fixvar_{ip25,houston}_{nofix,tokenln}_lr<lr>.csv` | `tab_safety.tex` |
| Fig. 6 (LoRA vs full fine-tuning) | `main` | as Table 2 | `figures/fig5_lora_vs_full` |
| Table 5, Fig. 7 (separated split) | `separated`, `classic`, `deep` | `results_disj_*.csv`, `results_classic_disjoint_*.csv`, `results_deep_disjoint_*.csv` | `tab_disjoint.tex`, `figures/fig6_disjoint` |
| Supplementary S2 (learning rates) | `lr_selection` | `results_lr_*.csv`, `lr_runs.csv` | `tabS_lr.tex`, `figures/figS1_lr_sweep` |
| Supplementary S3 (DOFA-Large, normalisations) | `tokens` | `embed_norm2.csv`, `coherence.csv` | `tabS_norms.tex`, `tabS_coh.tex`, `figures/figS2_norms_large` |
| Supplementary S4 (complete statistics) | `main`, `alternatives` | as above | `tabS_effects.tex`, `tabS_tost.tex`, `tabS_alt.tex` |
| Supplementary S5, S6 (rank, labels, band subsets) | `supplementary` | the files listed in `scripts/experiments.py` (`SUPP`) | `tabS_sg.tex`, `figures/figS3_bands`, `figures/figS4_labels` |
| Supplementary S7–S12 | as Tables 2–5 | as above | `tabS_classic.tex`, `perseed.tex`, `tabS_cfm_*.tex`, `tabS_deep_*.tex`, `tabS_disj_*.tex` |

File names in the paper's figures are numbered as in an earlier draft (`fig3_correction_effects` is Fig. 4, and so on).

## Settings

| | DOFA | Copernicus-FM | HybridSN / SSRN |
|---|---|---|---|
| Input | 33×33 window → 224×224 (bilinear) | same | 25×25 window of 30 PCs / 7×7 window of all bands |
| Generic LoRA | qkv of all 12 blocks, r 32, α 64, dropout 0.1, lr 6e-4 | same | – |
| Spectral LoRA | `fc_weight`, `fc_bias`, r 4, α 8, dropout 0.1, lr 2e-4 | – | – |
| Full fine-tuning | all parameters, lr 2e-5 | – | all parameters, Adam lr 1e-3 |
| Optimiser | AdamW, weight decay 0.05, batch 16 | same | Adam, batch 16 |
| Epochs | ≤ 400, early stopping on validation OA, patience 100 | same | ≤ 300, patience 50 |
| Augmentation | random flips and 90° rotations | same | same |
| Splits | random: 16 seeds (0–15); separated: 8 seeds | random, 8 seeds | as DOFA |

The supplementary experiments in Sections S5 and S6 were run earlier with learning rate 6e-5 for every method
(`scripts/experiments.py`, `SUPP`).

## Correspondence with the original scripts

The runs of the paper were made with per-scene copies of one training script and a few small patch modules,
kept unchanged in `legacy/`. This repository replaces them with one script per model:

| Original | Here |
|---|---|
| `train_dofa.py`, `train_dofa_{salinas,pavia,houston13}.py`, `train_dofa_ip_{u25,u50,…,vnir,uniform63}.py` | `scripts/train_dofa.py --dataset … --bands …` |
| `*_bnsafe.py` + `bandnorm_safe_patch.py` (and `*_bnref63.py`) | `--correction bcr` |
| `*_fixvar.py` + `fix_variants_patch.py` (`FIX_VARIANT=tokenln/sqrt63/ref9`) | `--correction gtn` (N > 63) or `ln` (N ≤ 63), `sqrt`, `ref9` |
| `*_disj.py` + `split_modes.py` (`SPLIT_MODE=disjoint`) | `--split separated` |
| `train_fm.py`, `cfm_common.py` | `scripts/train_copernicus.py`, `bcs/copernicus.py` |
| `baseline_classic.py`, `baseline_spatial.py`, `classic_split.py` | `scripts/classic_baselines.py` |
| `deep_baselines.py` | `scripts/deep_baselines.py` |
| `embed_norm2.py`, `cfm_tokens.py` | `scripts/measure_tokens_{dofa,copernicus}.py` |
| `lr_sweep.py`, `launchers/*.sh` | `scripts/run_experiments.py`, `scripts/experiments.py` |

Checks made when the repository was assembled:

* the random and separated splits of `bcs/splits.py` are identical to those of the original code (Pavia University
  and Salinas, seeds 0–7);
* `scripts/train_dofa.py` and the original scripts reach bit-identical training data, initial weights, trainable
  parameters after the first optimiser steps and random-number state, for the same arguments (synthetic scene, fixed
  weights, CPU), for GTN with Generic LoRA, BCR with Spectral LoRA, full fine-tuning, the VNIR subset and ungated
  normalisation of 25 bands;
* on the GPU, `scripts/train_dofa.py` and `scripts/train_copernicus.py` give results within the run-to-run variation
  of the original scripts (table above);
* `scripts/classic_baselines.py` and `scripts/deep_baselines.py` give results identical to the original scripts;
* `analysis/` reproduces all 3 174 numbers and all tables of the paper exactly from `results/`;
* `scripts/experiments.py` lists exactly the runs contained in `results/` (no run missing, none extra).
