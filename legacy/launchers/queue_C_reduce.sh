source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=7
for nb in 25 50; do
  export NBANDS=$nb
  for ds in pavia salinas; do for s in 0 1 2 3 4 5 6 7; do
    run train_dofa_${ds}_u${nb}.py results_${ds}_u${nb}.csv spectral 4 10 $s
    run train_dofa_${ds}_u${nb}.py results_${ds}_u${nb}.csv generic 32 10 $s
  done; done
done
echo "QUEUE_C_DONE $(date)"
