#!/usr/bin/env bash
# Progress of the paper-1 analysis queue:   bash paper1_analysis/progress.sh
cd "$(dirname "$0")/.."
LOG=logs/paper1_analysis
[ -d "$LOG" ] || { echo "not started yet"; exit 0; }
now=$(date +%s); d=0; r=0; w=0; f=0
printf "%-26s %-22s %s\n" JOB STATUS "LAST LOG LINE"
for s in "$LOG"/*.status; do
  [ -f "$s" ] || continue
  name=$(basename "$s" .status); st=$(cat "$s")
  case "$st" in
    done*) d=$((d+1)); continue ;;
    running*) r=$((r+1)); t0=${st##* }; st="running ($(( (now - t0) / 60 )) min, gpu $(echo "$st" | awk '{print $3}'))" ;;
    waiting*) w=$((w+1)); continue ;;
    failed*) f=$((f+1)) ;;
  esac
  last=$(tail -c 2000 "$LOG/$name.log" 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-90)
  printf "%-26s %-22s %s\n" "$name" "$st" "$last"
done
echo "----"
echo "done $d   running $r   waiting $w   failed $f   (total $((d+r+w+f)))"
if [ -f "$LOG/run_all.pid" ] && kill -0 "$(cat "$LOG/run_all.pid")" 2>/dev/null; then echo "queue: alive (pid $(cat "$LOG/run_all.pid"))"
else echo "queue: not running"; fi
[ -f "$LOG/queue_note.txt" ] && cat "$LOG/queue_note.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
free -g | awk '/^Mem:/{print "RAM available: "$7" GB"}'
