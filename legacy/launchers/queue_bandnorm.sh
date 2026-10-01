source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=6
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_bandnorm.py results_ip_bandnorm.csv spectral 4 10 $s
  run train_dofa_bandnorm.py results_ip_bandnorm.csv generic 32 10 $s
done
echo "QUEUE_BANDNORM_DONE $(date)"
