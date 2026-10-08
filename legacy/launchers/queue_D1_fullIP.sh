source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=2
export BANDNORM_REF=63
# full fine-tuning + safe fix, seeds 0-7 first (most important), then 8-15
for s in 0 1 2 3 4 5 6 7; do run train_dofa_bnsafe.py results_ip_bnsafe_full.csv full 8 10 $s; done
# LoRA + safe fix, new seeds 8-15 (seeds 0-7 already exist in results_ip_bnref63.csv, identical setting)
for s in 8 9 10 11 12 13 14 15; do
  run train_dofa_bnsafe.py results_ip_bnsafe.csv spectral 4 10 $s
  run train_dofa_bnsafe.py results_ip_bnsafe.csv generic 32 10 $s
done
for s in 8 9 10 11 12 13 14 15; do run train_dofa_bnsafe.py results_ip_bnsafe_full.csv full 8 10 $s; done
# full fine-tuning WITHOUT fix, seeds 8-15 (seeds 0-7 exist in results_baselines.csv)
for s in 8 9 10 11 12 13 14 15; do run train_dofa.py results_baselines_rep.csv full 8 10 $s; done
echo "QUEUE_D1_DONE $(date)"
