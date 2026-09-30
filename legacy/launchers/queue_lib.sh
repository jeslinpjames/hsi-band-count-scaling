source ~/hsi_foundation_adapt/hsigma_env/bin/activate
cd ~/hsi_foundation_adapt/exp
mkdir -p logs
COMMON="--max_epoch 400 --patience 100"
# usage: run SCRIPT CSV METHOD RANK N SEED
run () {
  local script=$1 csv=$2 m=$3 r=$4 n=$5 s=$6
  local alpha=$((2*r))                      # matches all previous runs: alpha = 2 x rank
  local v
  case $n in 5) v=3;; 10) v=5;; 20) v=5;; 50) v=10;; 100) v=20;;
    *) echo "!! no val_num rule for n=$n, skipping"; return;; esac
  local tag="${script%.py}_${m}_r${r}_n${n}_s${s}"
  if grep -q "^${m},${n},${s},${r}," "$csv" 2>/dev/null; then echo "SKIP ${tag}"; return; fi
  echo ">>> START ${tag} (alpha=${alpha}, val=${v})  $(date)"
  python -u "$script" --method "$m" --lora_r "$r" --lora_alpha "$alpha" --train_num "$n" \
    --val_num "$v" --seed "$s" $COMMON --csv "$csv" 2>&1 | tee "logs/${tag}.log"
}
