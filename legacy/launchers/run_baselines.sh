#!/bin/bash
# run_baselines.sh : deep baselines (random + disjoint splits) and the SVMs on the disjoint split.
# Three streams in parallel, each on the GPU with the most free memory when it starts. Safe to re-run: finished
# seeds are skipped.   usage (inside tmux):  bash run_baselines.sh
cd ~/hsi_foundation_adapt/exp
source ~/hsi_foundation_adapt/hsigma_env/bin/activate
export OMP_NUM_THREADS=8
mkdir -p logs_isprs
stream() {   # $1 = model
  for ds in ip salinas pavia; do python -u deep_baselines.py --model $1 --dataset $ds --seeds 0-15 >> logs_isprs/deep_$1.log 2>&1; done
  for ds in pavia salinas; do python -u deep_baselines.py --model $1 --dataset $ds --seeds 0-7 --split disjoint >> logs_isprs/deep_$1.log 2>&1; done
  echo "STREAM $1 FINISHED" >> logs_isprs/deep_$1.log
}
stream hybridsn &
sleep 120
stream ssrn &
( for ds in pavia salinas; do python -u classic_split.py --dataset $ds --seeds 0-7 >> logs_isprs/classic_disjoint.log 2>&1; done
  echo "STREAM classic FINISHED" >> logs_isprs/classic_disjoint.log ) &
wait
echo "ALL BASELINES FINISHED"
