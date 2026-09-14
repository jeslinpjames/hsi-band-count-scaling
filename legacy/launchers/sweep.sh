#!/bin/bash
set -u
source ~/hsi_foundation_adapt/hsigma_env/bin/activate
cd ~/hsi_foundation_adapt/exp

for N in 10 50 100; do
  for M in generic spectral; do
    for S in 0 1 2; do
      echo "###### $M n=$N seed=$S ######"
      python -u train_dofa.py --method $M --train_num $N --seed $S \
             --max_epoch 300 --patience 50 --csv results.csv \
             2>&1 | tee logs/${M}_n${N}_s${S}.log
    done
  done
done
echo "SWEEP DONE"
