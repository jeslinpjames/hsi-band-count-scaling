source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=0
export BANDNORM_REF=63
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_pavia_bnsafe.py   results_pavia_bnsafe_full.csv   full 8 10 $s
  run train_dofa_salinas.py        results_salinas_baselines.csv   full 8 10 $s
  run train_dofa_salinas_bnsafe.py results_salinas_bnsafe_full.csv full 8 10 $s
done
echo "QUEUE_D2_DONE $(date)"
