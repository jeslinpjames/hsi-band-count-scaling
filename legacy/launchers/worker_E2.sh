#!/bin/bash
# worker_E2.sh WORKER_ID : picks the freest GPU for every run; out-of-memory crashes are retried without limit.
cd ~/hsi_foundation_adapt/exp
source ~/hsi_foundation_adapt/hsigma_env/bin/activate
source ./queue_lib.sh
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=$1
mkdir -p tmpcsv status logs/failed
st () { echo "$1|$2|$3|$4|$(date +%s)" > status/w$W.txt; }   # state|gpu|tag|job
pop () { flock jobs_E.lock bash -c '
  for f in jobs_E2_pending.txt jobs_E_pending.txt; do
    l=$(head -n 1 $f 2>/dev/null); if [ -n "$l" ]; then sed -i 1d $f; echo "$l"; exit; fi
  done'; }
while true; do
  job=$(pop)
  [ -z "$job" ] && break
  IFS='|' read -r REF SCRIPT CSV M R N S TRY <<< "$job"
  OOM=$(echo "$TRY" | cut -s -d: -f2); TRY=${TRY%%:*}; OOM=${OOM:-0}
  key="^${M},${N},${S},${R},"
  if [ -f "$CSV" ] && grep -q "$key" "$CSV"; then continue; fi
  tag="${SCRIPT%.py}_${M}_r${R}_n${N}_s${S}"
  while true; do
    G=$(flock gpu_pick.lock python pick_gpu.py $W)
    [ "$G" != "none" ] && break
    st waiting - "$tag" "$job"; echo "... no GPU with ${MIN_FREE_MB:-12000} MB free, waiting 60 s ($(date '+%H:%M'))"; sleep 60
  done
  export CUDA_VISIBLE_DEVICES=$G
  if [ "$REF" = "none" ]; then unset BANDNORM_REF; else export BANDNORM_REF=$REF; fi
  T=tmpcsv/w${W}_$(basename "$CSV" .csv)_${M}_r${R}_n${N}_s${S}.csv; rm -f "$T"
  st running "$G" "$tag" "$job"; t0=$(date +%s)
  echo ">>> START [GPU $G W$W] $tag REF=$REF -> $CSV  $(date '+%d %b %H:%M')"
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
done
rm -f status/w$W.txt claims/w$W
echo "WORKER_F${W}_DONE $(date)"
