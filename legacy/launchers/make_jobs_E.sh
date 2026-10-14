#!/bin/bash
# Builds the pending job list for batch E (skips anything already in its target CSV).
cd ~/hsi_foundation_adapt/exp
J=jobs_E_pending.txt; : > $J.tmp
add () {  # REF SCRIPT CSV METHOD R N SEED
  local key="^$4,$6,$7,$5,"
  if [ -f "$3" ] && grep -q "$key" "$3"; then return; fi
  echo "$1|$2|$3|$4|$5|$6|$7|0" >> $J.tmp
}
# one short Pavia job first, so the result-merging step is verified within ~15 min
add 63 train_dofa_pavia_bnsafe.py results_pavia_bnsafe.csv generic 32 10 8
# Salinas seeds 8-15 (longest runs first = best load balancing)
for s in 8 9 10 11 12 13 14 15; do
  add 63   train_dofa_salinas_bnsafe.py results_salinas_bnsafe.csv      spectral 4 10 $s
  add none train_dofa_salinas.py        results_salinas.csv             spectral 4 10 $s
  add 63   train_dofa_salinas_bnsafe.py results_salinas_bnsafe_full.csv full     8 10 $s
  add none train_dofa_salinas.py        results_salinas_baselines.csv   full     8 10 $s
  add 63   train_dofa_salinas_bnsafe.py results_salinas_bnsafe.csv      generic 32 10 $s
  add none train_dofa_salinas.py        results_salinas.csv             generic 32 10 $s
done
# Pavia seeds 8-15
for s in 8 9 10 11 12 13 14 15; do
  add 63   train_dofa_pavia_bnsafe.py results_pavia_bnsafe.csv      spectral 4 10 $s
  add none train_dofa_pavia.py        results_pavia_rank.csv        spectral 4 10 $s
  add 63   train_dofa_pavia_bnsafe.py results_pavia_bnsafe_full.csv full     8 10 $s
  add none train_dofa_pavia.py        results_pavia_baselines.csv   full     8 10 $s
  add 63   train_dofa_pavia_bnsafe.py results_pavia_bnsafe.csv      generic 32 10 $s
  add none train_dofa_pavia.py        results_pavia_rank.csv        generic 32 10 $s
done
# Pavia REF 25, seeds 0-7 (completes the REF 9 / 25 / 63 comparison on a second dataset)
for s in 0 1 2 3 4 5 6 7; do
  add 25 train_dofa_pavia_bnref25.py results_pavia_bnref25.csv spectral 4 10 $s
  add 25 train_dofa_pavia_bnref25.py results_pavia_bnref25.csv generic 32 10 $s
done
# Salinas REF 25, seeds 0-7
for s in 0 1 2 3 4 5 6 7; do
  add 25 train_dofa_salinas_bnref25.py results_salinas_bnref25.csv spectral 4 10 $s
  add 25 train_dofa_salinas_bnref25.py results_salinas_bnref25.csv generic 32 10 $s
done
mv $J.tmp $J
echo "jobs pending: $(wc -l < $J)"
