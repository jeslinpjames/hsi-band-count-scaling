#!/bin/bash
set -u
source ~/hsi_foundation_adapt/hsigma_env/bin/activate
cd ~/hsi_foundation_adapt/exp

# Parameter-matched ablation: shrink spectral, grow generic, see who wins at equal budget
for S in 0 1 2; do
  for R in 1 2 4; do
    echo "### spectral r=$R seed=$S ###"
    python -u train_dofa.py --method spectral --train_num 10 --val_num 5 --seed $S \
      --lora_r $R --lora_alpha $((R*2)) --max_epoch 300 --patience 50 \
      --csv results_rank.csv 2>&1 | tee logs/spec_r${R}_n10_s${S}.log
  done
  for R in 16 32; do
    echo "### generic r=$R seed=$S ###"
    python -u train_dofa.py --method generic --train_num 10 --val_num 5 --seed $S \
      --lora_r $R --lora_alpha $((R*2)) --max_epoch 300 --patience 50 \
      --csv results_rank.csv 2>&1 | tee logs/gen_r${R}_n10_s${S}.log
  done
done
echo "RANK_SWEEP_DONE"
