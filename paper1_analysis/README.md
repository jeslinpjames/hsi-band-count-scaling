# paper1_analysis: extra analyses for the TGRS paper

Extra experiments that strengthen the paper's analysis section. They run separately from the Phase II scheduler,
with their own queue, logs (`logs/paper1_analysis`) and results (`results/paper1_analysis`).

| Job | Script | What it measures | GPU time |
|---|---|---|---|
| mech_{ip,salinas,pavia} | `mech_forward.py` | Pretrained DOFA, no training. For N in {9, 25, 50, 63, 100, 150, all} and each correction: patch-token norm, the update each block adds relative to its input, and CKA between patch tokens and head features (does the backbone still do anything when N is large?) | minutes |
| throughput | `throughput.py` | Pixels per second at inference for DOFA (none / BCR / GTN), SSRN and the SVM | minutes |
| probe_{ip,salinas,pavia} | `probe_frozen.py` | Frozen pretrained features plus logistic regression or kNN, 8 splits: does the correction improve the representation without any fine-tuning? | 10 to 30 min |
| calib_{scene}_{corr}_s{0,1,2} (27) | `train_calib.py` | Same training as the paper (Generic LoRA, 10 labels/class). Saves test-pixel probabilities (calibration, ECE, uncertainty, selective prediction, precision/recall/F1, false positives), per-epoch gradient norms of the first and last LoRA block, and fp16 / bf16 / int8 inference agreement and speed | ~as a normal run each |

`analyze.py` (no GPU) turns the outputs into figures, LaTeX tables and `summary.txt`.

## Run
```bash
cd ~/hsi-band-count-scaling
tmux new -s p1a
bash paper1_analysis/run_all.sh          # Ctrl+b d to detach
bash paper1_analysis/progress.sh         # any time
```
Settings (environment variables): `MAX_PAR` (3), `NEED_GB` free GPU memory to start a job (20), `KEEP_RAM_GB` (300),
`SEEDS` ("0 1 2"), `CALIB_ARGS` (extra arguments for `train_calib.py`). The queue is restart-safe: a job whose output
file exists is skipped, so after a crash just start it again. A failed job leaves `logs/paper1_analysis/<job>.log`.

## Collect
```bash
tar czf ~/paper1_analysis_results.tgz results/paper1_analysis/*.csv results/paper1_analysis/*.npz logs/paper1_analysis
```
(`feat_*.npy` are large and not needed.)

## Second batch (JOBSET=more): seeds 3-7 + WHU-Hi-LongKou
1. Put the LongKou MATLAB files in `$BCS_DATA/WHU-Hi-LongKou/` (`WHU_Hi_LongKou.mat`, `WHU_Hi_LongKou_gt.mat`), then
   `python paper1_analysis/add_longkou.py` (adds the scene to `bcs/` and the baseline scripts, prints CHECK OK).
2. `JOBSET=more MAX_PAR=12 NEED_GB=24 bash paper1_analysis/run_all.sh` (97 jobs):
   * LongKou: mechanism, SVM / SVM + 33x33 mean (16 splits), SSRN and HybridSN (16 splits),
     DOFA Generic LoRA none / BCR / GTN on 16 splits (fp16 / bf16 / int8 on splits 0-2 only);
   * Indian Pines, Salinas, Pavia: calibration runs for seeds 3-7 (now with the int8 test).
