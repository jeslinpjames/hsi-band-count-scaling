source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=1
# Exp 2: linear probe + full fine-tuning WITH the fix (rank 8 = same setting as original baselines)
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_bandnorm.py results_ip_bandnorm_baselines.csv linear 8 10 $s
  run train_dofa_bandnorm.py results_ip_bandnorm_baselines.csv full 8 10 $s
done
# Exp 1 (continued): Houston13 (48 bands) and Houston13 -> Houston18 cross-sensor
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_houston13_bn.py results_houston13_bandnorm.csv spectral 4 10 $s
  run train_dofa_houston13_bn.py results_houston13_bandnorm.csv generic 32 10 $s
done
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_houston_cross_bn.py results_houston_cross_bandnorm.csv spectral 4 10 $s
  run train_dofa_houston_cross_bn.py results_houston_cross_bandnorm.csv generic 32 10 $s
done
echo "QUEUE_FIXB_DONE $(date)"
