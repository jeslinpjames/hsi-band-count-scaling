source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=5
export BANDNORM_REF=63
for nb in 100 150; do
  export NBANDS=$nb
  for s in 0 1 2 3 4 5 6 7; do
    run train_dofa_ip_u${nb}_bnsafe.py results_ip_u${nb}_bnsafe.csv spectral 4 10 $s
    run train_dofa_ip_u${nb}_bnsafe.py results_ip_u${nb}_bnsafe.csv generic 32 10 $s
  done
done
echo "QUEUE_B_DONE $(date)"
