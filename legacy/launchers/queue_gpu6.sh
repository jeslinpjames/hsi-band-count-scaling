source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=6
# A) VNIR-truncation ablation on Indian Pines
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_ip_vnir.py results_ip_vnir.csv spectral 4 10 $s
  run train_dofa_ip_vnir.py results_ip_vnir.csv generic 32 10 $s
done
# F) Replication of the IP n=10 headline with 8 NEW seeds
for s in 8 9 10 11 12 13 14 15; do
  run train_dofa.py results_ip_n10_rep.csv spectral 4 10 $s
  run train_dofa.py results_ip_n10_rep.csv generic 32 10 $s
done
# B) Label-budget curve on Indian Pines: n=5 and n=20
for n in 5 20; do for s in 0 1 2 3 4 5 6 7; do
  run train_dofa.py results_ip_labelcurve.csv spectral 4 $n $s
  run train_dofa.py results_ip_labelcurve.csv generic 32 $n $s
done; done
echo "QUEUE_GPU6_DONE $(date)"
