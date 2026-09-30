source ./queue_lib.sh
export CUDA_VISIBLE_DEVICES=4
# C) Salinas n=5: prediction test (mirror of Pavia n=5; theory predicts Spectral advantage grows)
for s in 0 1 2 3 4 5 6 7; do
  run train_dofa_salinas.py results_salinas_n5.csv spectral 4 5 $s
  run train_dofa_salinas.py results_salinas_n5.csv generic 32 5 $s
done
# D) Complete n=50 at r4/r32 (seeds 0-2 already exist) -> same pairing at every label budget
for s in 3 4 5 6 7; do
  run train_dofa.py results_rank_n50_r4r32.csv spectral 4 50 $s
  run train_dofa.py results_rank_n50_r4r32.csv generic 32 50 $s
done
# E) Complete n=100 at r2/r16 -> both pairings at every budget (lowest priority, slowest)
for s in 3 4 5 6 7; do
  run train_dofa.py results_rank_n100_r2r16.csv spectral 2 100 $s
  run train_dofa.py results_rank_n100_r2r16.csv generic 16 100 $s
done
echo "QUEUE_GPU4_DONE $(date)"
