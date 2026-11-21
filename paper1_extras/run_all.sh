#!/usr/bin/env bash
# Paper 1 (TGRS) extras: 6 DOFA runs (Generic LoRA; Indian Pines, Salinas, Pavia; no correction and GTN; split 0)
# + SVM(33x33 mean) and SSRN on the same split, all saving predictions for every labelled pixel.
# Independent of the Phase II scheduler: own queue, own logs (logs/paper1_extras), own results (results/paper1_extras).
#
#   tmux new -s p1x          then:   bash paper1_extras/run_all.sh        (Ctrl+b d to detach)
#   progress:                        bash paper1_extras/progress.sh
#
# Safe to restart: finished runs (their .npz exists) are skipped.
set -o pipefail
cd "$(dirname "$0")/.."
H=${H:-$HOME/hsi_foundation_adapt}
PY=${PY:-$H/hsigma_env/bin/python}
export BCS_DATA=${BCS_DATA:-$H/data} DOFA_REPO=${DOFA_REPO:-$H/repos/DOFA} DOFA_WEIGHTS=${DOFA_WEIGHTS:-$H/DOFA_weights}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
OUT=results/paper1_extras; LOG=logs/paper1_extras
mkdir -p "$OUT" "$LOG"
MAX_PAR=${MAX_PAR:-3}          # DOFA runs at the same time
NEED_GB=${NEED_GB:-20}         # free GPU memory needed to start one DOFA run (measured peak ~15 GB)
KEEP_RAM_GB=${KEEP_RAM_GB:-300}
START_GAP=${START_GAP:-120}  # seconds between two starts
POLL=${POLL:-60}
SEED=0
JOBS=("ip none" "ip gtn" "salinas none" "salinas gtn" "pavia none" "pavia gtn")

ts() { date '+%d %b %H:%M:%S'; }
npz() { echo "$OUT/pred_dofa_$1_generic_$2_s$SEED.npz"; }
ram_free() { free -g | awk '/^Mem:/{print $7}'; }
best_gpu() {   # prints "index freeMiB" of the GPU with the most free memory
  timeout 30 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | sort -t, -k2 -nr | head -1 | tr ',' ' '
}

echo "$(ts) start (pid $$)" | tee -a "$LOG/run_all.log"
echo $$ > "$LOG/run_all.pid"

# ---- baselines (SVM on CPU, SSRN small on GPU): one background process, scenes one after another
if [ ! -f "$OUT/pred_ssrn_pavia_s$SEED.npz" ] || [ ! -f "$OUT/pred_svm_sp33_pavia_s$SEED.npz" ]; then
  ( for sc in ip salinas pavia; do
      echo "running" > "$LOG/baselines_$sc.status"
      if nice -n 5 "$PY" -u paper1_extras/predict_baselines.py --dataset "$sc" --seed $SEED > "$LOG/baselines_$sc.log" 2>&1 \
         && [ -f "$OUT/pred_ssrn_${sc}_s$SEED.npz" ]; then echo "done" > "$LOG/baselines_$sc.status"
      else echo "failed" > "$LOG/baselines_$sc.status"; fi
    done ) &
  echo "$(ts) baselines started (pid $!)" | tee -a "$LOG/run_all.log"
fi

# ---- DOFA queue
declare -A PID=()
pending=()
for j in "${JOBS[@]}"; do
  set -- $j
  if [ -f "$(npz $1 $2)" ]; then echo "done" > "$LOG/dofa_$1_$2.status"; else pending+=("$j"); echo "waiting" > "$LOG/dofa_$1_$2.status"; fi
done

while [ ${#pending[@]} -gt 0 ] || [ ${#PID[@]} -gt 0 ]; do
  # collect finished runs
  for key in "${!PID[@]}"; do
    if ! kill -0 "${PID[$key]}" 2>/dev/null; then
      set -- $key
      if [ -f "$(npz $1 $2)" ]; then echo "done" > "$LOG/dofa_$1_$2.status"; echo "$(ts) done $key" | tee -a "$LOG/run_all.log"
      else echo "failed" > "$LOG/dofa_$1_$2.status"; echo "$(ts) FAILED $key (see $LOG/dofa_$1_$2.log)" | tee -a "$LOG/run_all.log"; fi
      unset "PID[$key]"
    fi
  done
  # start one more run if allowed
  if [ ${#pending[@]} -gt 0 ] && [ ${#PID[@]} -lt "$MAX_PAR" ]; then
    read -r gpu freemib _ <<< "$(best_gpu)"
    ram=$(ram_free)
    if [[ "${gpu:-}" =~ ^[0-9]+$ && "${freemib:-}" =~ ^[0-9]+$ && "${ram:-}" =~ ^[0-9]+$ ]] \
       && [ "$freemib" -ge $((NEED_GB * 1024)) ] && [ "$ram" -ge "$KEEP_RAM_GB" ]; then
      j="${pending[0]}"; pending=("${pending[@]:1}")
      set -- $j
      CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -u paper1_extras/train_predict_dofa.py --dataset "$1" --correction "$2" \
        --seed $SEED ${DOFA_ARGS:-} > "$LOG/dofa_$1_$2.log" 2>&1 &
      PID["$j"]=$!
      echo "running gpu $gpu since $(date +%s)" > "$LOG/dofa_$1_$2.status"
      echo "$(ts) started $j on GPU $gpu (free ${freemib} MiB, RAM free ${ram} GB, pid ${PID[$j]})" | tee -a "$LOG/run_all.log"
      sleep "$START_GAP"   # let the run allocate its GPU memory before choosing the next GPU
      continue
    else
      echo "waiting: best GPU ${gpu:-?} has ${freemib:-?} MiB free (need $((NEED_GB * 1024))), RAM free ${ram:-?} GB" \
        > "$LOG/queue_note.txt"
    fi
  fi
  sleep "$POLL"
done
wait
echo "$(ts) all DOFA runs finished" | tee -a "$LOG/run_all.log"
rm -f "$LOG/run_all.pid" "$LOG/queue_note.txt"
