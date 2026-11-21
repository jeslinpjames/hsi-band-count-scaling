#!/usr/bin/env bash
# Progress of the paper-1 extra runs (separate from Phase II's scripts/progress.sh).
#   bash paper1_extras/progress.sh            (watch -n 60 bash paper1_extras/progress.sh  to refresh)
cd "$(dirname "$0")/.."
LOG=logs/paper1_extras; OUT=results/paper1_extras
echo "=================== PAPER 1 EXTRAS (maps + per-class)  $(date '+%a %d %b %H:%M') ==================="
if [ -f "$LOG/run_all.pid" ] && kill -0 "$(cat $LOG/run_all.pid)" 2>/dev/null; then
  echo "Launcher: RUNNING (pid $(cat $LOG/run_all.pid))"
elif [ -f "$LOG/run_all.log" ] && grep -q "all DOFA runs finished" "$LOG/run_all.log"; then
  echo "Launcher: FINISHED"
else
  echo "Launcher: NOT RUNNING  (start: tmux new -s p1x, then bash paper1_extras/run_all.sh)"
fi
[ -f "$LOG/queue_note.txt" ] && echo "Queue: $(cat $LOG/queue_note.txt)"
echo
printf "  %-22s %-10s %-6s %-9s %s\n" "run" "status" "GPU" "elapsed" "latest"
now=$(date +%s); ndone=0; ntot=0
for sc in ip salinas pavia; do
  for c in none gtn; do
    ntot=$((ntot+1)); st="-"; gpu=""; el=""; last=""
    f="$LOG/dofa_${sc}_${c}.status"; lg="$LOG/dofa_${sc}_${c}.log"
    [ -f "$f" ] && st=$(awk '{print $1}' "$f")
    if [ "$st" = "running" ]; then gpu=$(awk '{print $3}' "$f"); t0=$(awk '{print $5}' "$f"); el="$(( (now-t0)/60 ))m"; fi
    if [ -f "$lg" ]; then
      last=$(grep -E "^RESULT|^epoch|predicted|early stop|Error|error" "$lg" | tail -1 | cut -c1-70)
      grep -q "^RESULT" "$lg" && last=$(grep "^RESULT" "$lg" | sed 's/.*: \(OA [0-9.]*\).*/\1/')
    fi
    [ -f "$OUT/pred_dofa_${sc}_generic_${c}_s0.npz" ] && { st="done"; ndone=$((ndone+1)); }
    printf "  %-22s %-10s %-6s %-9s %s\n" "DOFA $sc $c" "$st" "$gpu" "$el" "$last"
  done
done
for sc in ip salinas pavia; do
  for m in svm_sp33 ssrn; do
    ntot=$((ntot+1)); st="-"; last=""
    [ -f "$LOG/baselines_$sc.status" ] && st=$(cat "$LOG/baselines_$sc.status")
    if [ -f "$OUT/pred_${m}_${sc}_s0.npz" ]; then st="done"; ndone=$((ndone+1))
      last=$(grep "^RESULT $sc $m" "$LOG/baselines_$sc.log" 2>/dev/null | sed 's/.*: \(OA [0-9.]*\).*/\1/')
    elif [ "$st" = "done" ] || [ "$st" = "running" ]; then st="running"; fi
    printf "  %-22s %-10s %-6s %-9s %s\n" "$m $sc" "$st" "" "" "$last"
  done
done
echo
echo "  TOTAL $ndone / $ntot done"
echo "  Server: load $(awk '{print $1" / "$2" / "$3}' /proc/loadavg) | RAM available $(free -g | awk '/^Mem:/{print $7}') GB"
echo "  GPU free memory: $(timeout 20 nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' '{printf "%s:%dG/%s%%  ", $1, $2/1024, $3}')"
[ $ndone -eq $ntot ] && echo "  ALL DONE -> pack with:  tar czf ~/paper1_extras_results.tgz results/paper1_extras logs/paper1_extras"
echo "======================================================================================================"
