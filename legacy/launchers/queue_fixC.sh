source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=5
# Exp 3: reference value of the fix (default was 9)
for ref in 25 63; do
  export BANDNORM_REF=$ref
  for s in 0 1 2 3 4 5 6 7; do
    run train_dofa_bnref${ref}.py results_ip_bnref${ref}.csv spectral 4 10 $s
    run train_dofa_bnref${ref}.py results_ip_bnref${ref}.csv generic 32 10 $s
  done
done
echo "QUEUE_FIXC_DONE $(date)"
