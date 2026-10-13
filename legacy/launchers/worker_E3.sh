#!/bin/bash
# worker_E3.sh WORKER_ID : takes the first queued run that fits on any GPU now (per-run-type memory need).
cd ~/hsi_foundation_adapt/exp
source ~/hsi_foundation_adapt/hsigma_env/bin/activate
source ./queue_lib.sh
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cap CPU threads per training run (PyTorch otherwise starts one per core = 256)
T_N=$(cat threads_per_run.txt 2>/dev/null || echo 8)
export OMP_NUM_THREADS=$T_N MKL_NUM_THREADS=$T_N OPENBLAS_NUM_THREADS=$T_N NUMEXPR_NUM_THREADS=$T_N VECLIB_MAXIMUM_THREADS=$T_N
W=$1
mkdir -p tmpcsv status logs/failed
st () { echo "$1|$2|$3|$4|$(date +%s)" > status/w$W.txt; }   # state|gpu|tag|job|time
while true; do
  out=$(flock jobs_E.lock python claim_next.py $W)
  if [ "$out" = "empty" ]; then break; fi
  if [ "$out" = "none" ] || [ -z "$out" ]; then
    st waiting - "nothing-fits-yet" "-"; sleep 60; continue
  fi
  G=${out%%|*}; rest=${out#*|}; NEED=${rest%%|*}; job=${rest#*|}
  IFS='|' read -r REF SCRIPT CSV M R N S TRY <<< "$job"
  OOM=$(echo "$TRY" | cut -s -d: -f2); TRY=${TRY%%:*}; OOM=${OOM:-0}
  key="^${M},${N},${S},${R},"
  if [ -f "$CSV" ] && grep -q "$key" "$CSV"; then rm -f claims/w$W; continue; fi
  tag="${SCRIPT%.py}_${M}_r${R}_n${N}_s${S}"
  export CUDA_VISIBLE_DEVICES=$G
  if [ "$REF" = "none" ]; then unset BANDNORM_REF; else export BANDNORM_REF=$REF; fi
  T=tmpcsv/w${W}_$(basename "$CSV" .csv)_${M}_r${R}_n${N}_s${S}.csv; rm -f "$T"
  st running "$G" "$tag" "$job"; t0=$(date +%s)
  echo ">>> START [GPU $G W$W need ${NEED}MB] $tag REF=$REF -> $CSV  $(date '+%d %b %H:%M')"
  run "$SCRIPT" "$T" "$M" "$R" "$N" "$S"
  dt=$(( $(date +%s) - t0 ))
  if [ -f "$T" ] && grep -q "$key" "$T"; then
    flock "$CSV.lock" bash -c 'if [ -f "$1" ]; then grep "$2" "$3" | tail -n 1 >> "$1"; else cp "$3" "$1"; fi' _ "$CSV" "$key" "$T"
    echo "$tag $dt ok GPU$G $(date '+%d_%H:%M')" >> jobs_E_times.txt
    echo "+++ DONE $tag in $((dt/60)) min -> $CSV"
  else
    cp "logs/$tag.log" "logs/failed/$tag.$(date +%d%H%M).log" 2>/dev/null
    if grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_.*ALLOC" "logs/$tag.log"; then
      OOM=$((OOM + 1)); why="out of memory on GPU $G"
      echo "$tag $dt oom GPU$G $(date '+%d_%H:%M')" >> jobs_E_times.txt
    else
      TRY=$((TRY + 1)); why="error (not memory), try $TRY of 3"
      echo "$tag $dt error GPU$G $(date '+%d_%H:%M')" >> jobs_E_times.txt
    fi
    if [ "$TRY" -lt 3 ]; then
      echo "!!! FAILED $tag: $why -> requeued at front"
      flock jobs_E.lock bash -c 'f=jobs_E2_pending.txt; { echo "$1"; cat $f 2>/dev/null; true; } > $f.tmp; mv $f.tmp $f' _ "$REF|$SCRIPT|$CSV|$M|$R|$N|$S|$TRY:$OOM"
      st cooldown "$G" "$tag" "$job"; sleep 90
    else
      echo "!!! GAVE UP $tag after 3 non-memory errors (see logs/failed/)" | tee -a jobs_E_failed.txt
    fi
  fi
  rm -f claims/w$W
done
rm -f status/w$W.txt claims/w$W
echo "WORKER_G${W}_DONE $(date)"
