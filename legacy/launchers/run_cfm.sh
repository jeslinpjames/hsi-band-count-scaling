#!/bin/bash
# run_cfm.sh : Copernicus-FM token measurement, then Generic LoRA fine-tuning with none / BCR / GTN on Indian Pines
# and Pavia (seeds 0-7, 48 runs). One stream per correction (3 GPUs at a time). Finished runs are skipped.
# usage (inside tmux):  bash run_cfm.sh
cd ~/hsi_foundation_adapt/exp
source ~/hsi_foundation_adapt/cfm_env/bin/activate
export OMP_NUM_THREADS=8
mkdir -p logs_isprs
[ -f cfm_coherence.csv ] || python -u cfm_tokens.py > logs_isprs/cfm_tokens.log 2>&1
stream() {   # $1 = correction
  for ds in ip pavia; do
    for s in 0 1 2 3 4 5 6 7; do
      unset CUDA_VISIBLE_DEVICES
      python -u train_fm.py --dataset $ds --corr $1 --seed $s >> logs_isprs/cfm_$1.log 2>&1
    done
  done
  echo "STREAM $1 FINISHED" >> logs_isprs/cfm_$1.log
}
stream none &
sleep 180
stream bcr &
sleep 180
stream gtn &
wait
echo "ALL CFM FINISHED"
