source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=7
for nb in 25 100 150 50; do
  export NBANDS=$nb
  for s in 0 1 2 3 4 5 6 7; do
    run train_dofa_ip_u${nb}.py results_ip_uniform${nb}.csv spectral 4 10 $s
    run train_dofa_ip_u${nb}.py results_ip_uniform${nb}.csv generic 32 10 $s
  done
done
echo "QUEUE_BANDSWEEP_DONE $(date)"
