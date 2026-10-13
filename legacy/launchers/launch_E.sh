#!/bin/bash
# launch_E.sh [NUM_WORKERS] : starts workers on the least-loaded GPUs (safe to rerun after a reboot).
cd ~/hsi_foundation_adapt/exp
source ~/hsi_foundation_adapt/hsigma_env/bin/activate
NW=${1:-8}
if tmux ls 2>/dev/null | grep -q "^wE"; then echo "batch E workers already running:"; tmux ls | grep "^wE"; exit 0; fi
bash make_jobs_E.sh
# GPUs with < 60 GB used, least memory first; workers are dealt round-robin over the best 6
mapfile -t G < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 60000' | sort -t, -k2 -n | head -6 | cut -d, -f1)
[ ${#G[@]} -eq 0 ] && { echo "No GPU under 60 GB, not starting"; exit 1; }
for w in $(seq 1 $NW); do
  g=${G[$(( (w-1) % ${#G[@]} ))]}
  tmux new -d -s wE$w
  tmux send-keys -t wE$w "bash ~/hsi_foundation_adapt/exp/worker_E.sh $g $w" Enter
  echo "worker wE$w -> GPU $g"
  sleep 20      # stagger starts so data loading doesn't all hit at once
done
