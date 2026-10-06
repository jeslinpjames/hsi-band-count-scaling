source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=7
# Exp 1: volume fix on Salinas (204 bands) and Pavia (102 bands)
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_salinas_bn.py results_salinas_bandnorm.csv spectral 4 10 $s
  run train_dofa_salinas_bn.py results_salinas_bandnorm.csv generic 32 10 $s
done
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_pavia_bn.py results_pavia_bandnorm.csv spectral 4 10 $s
  run train_dofa_pavia_bn.py results_pavia_bandnorm.csv generic 32 10 $s
done
echo "QUEUE_FIXA_DONE $(date)"
