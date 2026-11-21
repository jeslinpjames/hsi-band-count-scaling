#!/usr/bin/env bash
# Paper 1 (TGRS) analysis jobs. Own queue, logs (logs/paper1_analysis) and results (results/paper1_analysis);
# independent of the Phase II scheduler. Safe to restart: a job whose output exists is skipped.
#
#   tmux new -s p1a        then:   bash paper1_analysis/run_all.sh        (Ctrl+b d to detach)
#   progress:                      bash paper1_analysis/progress.sh
#
#   JOBSET=more MAX_PAR=12 bash paper1_analysis/run_all.sh      second batch (see the job list below)
set -o pipefail
cd "$(dirname "$0")/.."
H=${H:-$HOME/hsi_foundation_adapt}
PY=${PY:-$H/hsigma_env/bin/python}
export BCS_DATA=${BCS_DATA:-$H/data} DOFA_REPO=${DOFA_REPO:-$H/repos/DOFA} DOFA_WEIGHTS=${DOFA_WEIGHTS:-$H/DOFA_weights}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
OUT=results/paper1_analysis; LOG=logs/paper1_analysis
mkdir -p "$OUT" "$LOG"
MAX_PAR=${MAX_PAR:-3}
NEED_GB=${NEED_GB:-20}
KEEP_RAM_GB=${KEEP_RAM_GB:-300}
START_GAP=${START_GAP:-45}
POLL=${POLL:-60}
CALIB_ARGS=${CALIB_ARGS:-}

# job list: "name|output file|shell command". JOBSET=main: the first 34 jobs (done). JOBSET=more: seeds 3-7 of the
# calibration runs (with the int8 fix) and the WHU-Hi-LongKou scene (16 splits, DOFA none/BCR/GTN + baselines).
JOBSET=${JOBSET:-main}
P="nice -n 5 $PY -u"
JOBS=()
if [ "$JOBSET" = main ]; then
  for sc in ip salinas pavia; do JOBS+=("mech_$sc|$OUT/mech_$sc.csv|$P paper1_analysis/mech_forward.py --dataset $sc"); done
  JOBS+=("throughput|$OUT/throughput.csv|$P paper1_analysis/throughput.py")
  for sc in ip salinas pavia; do JOBS+=("probe_$sc|$OUT/probe_$sc.csv|$P paper1_analysis/probe_frozen.py --dataset $sc"); done
  for s in ${SEEDS:-0 1 2}; do for sc in ip salinas pavia; do for c in none bcr gtn; do
    JOBS+=("calib_${sc}_${c}_s$s|$OUT/calib_${sc}_generic_${c}_s$s.npz|$P paper1_analysis/train_calib.py --dataset $sc --correction $c --seed $s $CALIB_ARGS")
  done; done; done
else
  # LongKou first: its runs are the longest (204k test pixels) and the baselines run alongside them
  JOBS+=("mech_longkou|$OUT/mech_longkou.csv|$P paper1_analysis/mech_forward.py --dataset longkou")
  JOBS+=("classic_longkou|$OUT/done_classic_longkou|$P scripts/classic_baselines.py --dataset longkou --seeds 0-15 --methods svm,svm_sp33 --csv $OUT/results_classic_longkou.csv && touch $OUT/done_classic_longkou")
  for m in ssrn hybridsn; do
    JOBS+=("${m}_longkou|$OUT/done_${m}_longkou|$P scripts/deep_baselines.py --model $m --dataset longkou --seeds 0-15 --csv $OUT/results_deep_longkou.csv && touch $OUT/done_${m}_longkou")
  done
  for s in $(seq 0 15); do for c in none bcr gtn; do
    pr=0; [ "$s" -le 2 ] && pr=1          # fp16 / bf16 / int8 tests on three splits only (they triple prediction time)
    JOBS+=("calib_longkou_${c}_s$s|$OUT/calib_longkou_generic_${c}_s$s.npz|$P paper1_analysis/train_calib.py --dataset longkou --correction $c --seed $s --precision $pr $CALIB_ARGS")
  done; done
  for s in 3 4 5 6 7; do for sc in ip salinas pavia; do for c in none bcr gtn; do
    JOBS+=("calib_${sc}_${c}_s$s|$OUT/calib_${sc}_generic_${c}_s$s.npz|$P paper1_analysis/train_calib.py --dataset $sc --correction $c --seed $s $CALIB_ARGS")
  done; done; done
fi

ts() { date '+%d %b %H:%M:%S'; }
ram_free() { free -g | awk '/^Mem:/{print $7}'; }
best_gpu() {
  timeout 30 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | sort -t, -k2 -nr | head -1 | tr ',' ' '
}

echo "$(ts) start (pid $$, ${#JOBS[@]} jobs)" | tee -a "$LOG/run_all.log"
echo $$ > "$LOG/run_all.pid"
declare -A PID=() OUTF=()
pending=()
for j in "${JOBS[@]}"; do
  IFS='|' read -r name outf cmd <<< "$j"
  if [ -f "$outf" ]; then echo "done" > "$LOG/$name.status"; else pending+=("$j"); echo "waiting" > "$LOG/$name.status"; fi
done

while [ ${#pending[@]} -gt 0 ] || [ ${#PID[@]} -gt 0 ]; do
  for name in "${!PID[@]}"; do
    if ! kill -0 "${PID[$name]}" 2>/dev/null; then
      if [ -f "${OUTF[$name]}" ]; then echo "done" > "$LOG/$name.status"; echo "$(ts) done $name" | tee -a "$LOG/run_all.log"
      else echo "failed" > "$LOG/$name.status"; echo "$(ts) FAILED $name (see $LOG/$name.log)" | tee -a "$LOG/run_all.log"; fi
      unset "PID[$name]"
    fi
  done
  if [ ${#pending[@]} -gt 0 ] && [ ${#PID[@]} -lt "$MAX_PAR" ]; then
    read -r gpu freemib _ <<< "$(best_gpu)"
    ram=$(ram_free)
    if [[ "${gpu:-}" =~ ^[0-9]+$ && "${freemib:-}" =~ ^[0-9]+$ && "${ram:-}" =~ ^[0-9]+$ ]] \
       && [ "$freemib" -ge $((NEED_GB * 1024)) ] && [ "$ram" -ge "$KEEP_RAM_GB" ]; then
      IFS='|' read -r name outf cmd <<< "${pending[0]}"; pending=("${pending[@]:1}")
      CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$cmd" > "$LOG/$name.log" 2>&1 &
      PID[$name]=$!; OUTF[$name]=$outf
      echo "running gpu $gpu since $(date +%s)" > "$LOG/$name.status"
      echo "$(ts) started $name on GPU $gpu (free ${freemib} MiB, RAM free ${ram} GB, pid ${PID[$name]})" | tee -a "$LOG/run_all.log"
      sleep "$START_GAP"
      continue
    else
      echo "waiting: best GPU ${gpu:-?} has ${freemib:-?} MiB free (need $((NEED_GB * 1024))), RAM free ${ram:-?} GB" > "$LOG/queue_note.txt"
    fi
  fi
  sleep "$POLL"
done
wait
echo "$(ts) all jobs finished" | tee -a "$LOG/run_all.log"
rm -f "$LOG/run_all.pid" "$LOG/queue_note.txt"
