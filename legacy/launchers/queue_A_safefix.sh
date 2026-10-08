source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=6
export BANDNORM_REF=63
for ds in salinas pavia; do for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_${ds}_bnsafe.py results_${ds}_bnsafe.csv spectral 4 10 $s
  run train_dofa_${ds}_bnsafe.py results_${ds}_bnsafe.csv generic 32 10 $s
done; done
echo "QUEUE_A_DONE $(date)"
