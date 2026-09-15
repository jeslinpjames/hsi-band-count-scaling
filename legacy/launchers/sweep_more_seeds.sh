#!/bin/bash
set -u
source ~/hsi_foundation_adapt/hsigma_env/bin/activate
cd ~/hsi_foundation_adapt/exp

for S in 3 4 5 6 7; do
  for M in generic spectral; do
    echo "### $M n=10 seed=$S ###"
    python -u train_dofa.py --method $M --train_num 10 --val_num 5 --seed $S \
      --max_epoch 300 --patience 50 --csv results.csv \
      2>&1 | tee logs/${M}_n10_s${S}.log
  done
done
echo "MORE_SEEDS_DONE"
