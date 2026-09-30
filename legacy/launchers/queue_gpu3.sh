source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=3
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_ip_uniform63.py results_ip_uniform63.csv spectral 4 10 $s
  run train_dofa_ip_uniform63.py results_ip_uniform63.csv generic 32 10 $s
done
echo "QUEUE_GPU3_DONE $(date)"
