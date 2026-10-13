#!/bin/bash
# worker_E.sh GPU WORKER_ID : takes jobs from jobs_E_pending.txt until it is empty.
cd ~/hsi_foundation_adapt/exp
source ~/hsi_foundation_adapt/hsigma_env/bin/activate
source ./queue_lib.sh
set +e
export CUDA_VISIBLE_DEVICES=$1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=$2
mkdir -p tmpcsv
while true; do
  job=$(flock jobs_E.lock bash -c 'l=$(head -n 1 jobs_E_pending.txt 2>/dev/null); if [ -n "$l" ]; then sed -i 1d jobs_E_pending.txt; fi; echo "$l"')
  [ -z "$job" ] && break
  IFS='|' read -r REF SCRIPT CSV M R N S TRY <<< "$job"
  key="^${M},${N},${S},${R},"
  if [ -f "$CSV" ] && grep -q "$key" "$CSV"; then echo "SKIP (already done): $job"; continue; fi
  if [ "$REF" = "none" ]; then unset BANDNORM_REF; else export BANDNORM_REF=$REF; fi
  T=tmpcsv/w${W}_$(basename "$CSV" .csv)_${M}_r${R}_n${N}_s${S}.csv
  rm -f "$T"
  echo ">>> START [GPU $1 W$W] ${SCRIPT%.py} $M r$R n$N s$S REF=$REF -> $CSV  $(date '+%d %b %H:%M')"
  run "$SCRIPT" "$T" "$M" "$R" "$N" "$S"
  if [ -f "$T" ] && grep -q "$key" "$T"; then
    flock "$CSV.lock" bash -c '
      if [ -f "$1" ]; then grep "$2" "$3" | tail -n 1 >> "$1"; else cp "$3" "$1"; fi' _ "$CSV" "$key" "$T"
    echo "+++ DONE $M r$R s$S -> $CSV  $(date '+%d %b %H:%M')"
  else
    TRY=$((TRY + 1))
    if [ "$TRY" -lt 3 ]; then
      echo "!!! FAILED $M r$R s$S (try $TRY), requeued, waiting 60 s"
      flock jobs_E.lock bash -c 'echo "$1" >> jobs_E_pending.txt' _ "$REF|$SCRIPT|$CSV|$M|$R|$N|$S|$TRY"
      sleep 60
    else
      echo "!!! GAVE UP $M r$R s$S after 3 tries" | tee -a jobs_E_failed.txt
    fi
  fi
done
echo "WORKER_E${W}_DONE $(date)"
