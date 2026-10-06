source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=6
# Exp 4: fix applied on top of band reduction
export NBANDS=25
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_ip_u25_bn.py results_ip_u25_bandnorm.csv spectral 4 10 $s
  run train_dofa_ip_u25_bn.py results_ip_u25_bandnorm.csv generic 32 10 $s
done
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_ip_u63_bn.py results_ip_u63_bandnorm.csv spectral 4 10 $s
  run train_dofa_ip_u63_bn.py results_ip_u63_bandnorm.csv generic 32 10 $s
done
echo "QUEUE_FIXD_DONE $(date)"
