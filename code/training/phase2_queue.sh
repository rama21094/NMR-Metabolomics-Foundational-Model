#!/usr/bin/env bash
# Phase 2 -- pretrain the three SSL objectives, several seeds, across free GPUs.
#
# Ordered SEED-MAJOR: all three objectives at seed 0, then all three at seed 1,
# and so on. An interrupted queue therefore still yields a complete, analysable
# set of seeds rather than one objective finished and two half-done.
#
# GPU 0 is deliberately left alone -- another user's job is on it.
set -u
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics
DATA=$ROOT/rebuild/pretrain/corpus_train.npy
GPUS=(1 2)
SEEDS=(0 1 2 3 4)
mkdir -p $ROOT/results/phase2 $ROOT/models/phase2 $ROOT/results/phase2/logs

run_one () {          # $1 objective  $2 seed  $3 gpu
  local obj=$1 seed=$2 gpu=$3
  local tag="${obj}_seed${seed}"
  local log=$ROOT/results/phase2/logs/${tag}.log
  [ -f "$ROOT/results/phase2/${tag}.done" ] && { echo "skip $tag"; return; }
  echo "$(date +%H:%M:%S) start $tag on cuda:$gpu"
  case $obj in
    masking)
      $PY $ROOT/code/training/trainer_revised.py \
          --data-path "$DATA" --seed $seed --device cuda:$gpu \
          --patch-size 1024 --batch-size 32 --num-epochs 400 --patience 50 \
          --d-model 192 --nhead 6 --num-layers 4 --dim-feedforward 768 \
          > "$log" 2>&1 ;;
    jigsaw)
      $PY $ROOT/code/training/train_jigsaw_spectra.py \
          --data-path "$DATA" --seed $seed --device cuda:$gpu \
          --bin-size 1024 --epochs 400 --patience 50 \
          --out-dir $ROOT/results/phase2/$tag --model-dir $ROOT/models/phase2/$tag \
          > "$log" 2>&1 ;;
    joint)
      $PY $ROOT/code/training/train_joint_ssl.py \
          --data-path "$DATA" --seed $seed --device cuda:$gpu \
          --mask-bin-size 1024 --epochs 400 --patience 50 \
          --out-dir $ROOT/results/phase2/$tag --model-dir $ROOT/models/phase2/$tag \
          > "$log" 2>&1 ;;
  esac
  if [ $? -eq 0 ]; then touch "$ROOT/results/phase2/${tag}.done"; echo "$(date +%H:%M:%S) OK   $tag"
  else echo "$(date +%H:%M:%S) FAIL $tag (see $log)"; fi
}

# Build the seed-major job list, then deal jobs to GPUs as each falls free.
JOBS=()
for s in "${SEEDS[@]}"; do for o in masking jigsaw joint; do JOBS+=("$o:$s"); done; done

declare -A PIDS
i=0
while [ $i -lt ${#JOBS[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do
    if [ -z "${PIDS[$g]:-}" ] && [ $i -lt ${#JOBS[@]} ]; then
      IFS=: read -r obj seed <<< "${JOBS[$i]}"
      run_one "$obj" "$seed" "$g" &
      PIDS[$g]=$!
      i=$((i+1))
    fi
  done
  sleep 20
  for g in "${GPUS[@]}"; do
    if [ -n "${PIDS[$g]:-}" ] && ! kill -0 "${PIDS[$g]}" 2>/dev/null; then unset PIDS[$g]; fi
  done
done
echo "$(date +%H:%M:%S) queue complete"
