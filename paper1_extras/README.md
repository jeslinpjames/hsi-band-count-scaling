# Paper 1 (TGRS) extras: classification maps and per-class accuracies

Small, self-contained add-on. It does not touch the Phase II scheduler, its queue, results or logs.

| File | Purpose |
|---|---|
| `train_predict_dofa.py` | DOFA, Generic LoRA, split 0 — identical to `scripts/train_dofa.py`, plus a prediction for every labelled pixel |
| `predict_baselines.py` | SVM (+33×33 mean) and SSRN on split 0, same code as the paper's baselines, predictions for every labelled pixel |
| `run_all.sh` | launcher: 6 DOFA runs (IP, Salinas, Pavia × none/GTN), at most 3 at a time, each on the GPU with the most free memory (≥ 20 GB), waits while < 300 GB RAM is free; baselines in one background process |
| `progress.sh` | status of every run, GPU memory, server load |
| `make_maps_tables.py` | (run on the laptop / by Claude) map figure + per-class LaTeX tables from the `.npz` files |

Outputs: `results/paper1_extras/*.npz`, `results/paper1_extras/runs_*.csv`, logs in `logs/paper1_extras/`.
Restart-safe: a run whose `.npz` exists is skipped.
