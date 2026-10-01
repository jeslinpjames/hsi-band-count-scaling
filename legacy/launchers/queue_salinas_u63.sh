source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=5
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_salinas_u63.py results_salinas_uniform63.csv spectral 4 10 $s
  run train_dofa_salinas_u63.py results_salinas_uniform63.csv generic 32 10 $s
done
echo "QUEUE_SALINAS_U63_DONE $(date)"
