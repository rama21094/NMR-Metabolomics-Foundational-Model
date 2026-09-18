#!/usr/bin/env bash
# Phase 4.1 -- pretrain masking on every corpus subset, distributed across hosts.
#
# Masking only: Phase 3 established it is the strongest objective by a clear
# margin at every label budget (0.549-0.672 vs jigsaw 0.498-0.587, joint
# 0.516-0.611), so the scaling question is asked of the best representation we
# have rather than of all three at triple the cost.
#
# Usage:  phase4_scaling_queue.sh <gpu-list> <slots-per-gpu> <shard-index> <n-shards>
# e.g.    phase4_scaling_queue.sh "1,2" 2 0 3
#
# Jobs are sharded deterministically by index, so each host takes a disjoint
# slice of the same job list with no coordination between hosts.
set -u
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics
SUB=$ROOT/rebuild/pretrain/scaling
GPUS_CSV=${1:-1,2}; SLOTS=${2:-2}; SHARD=${3:-0}; NSHARD=${4:-1}
IFS=, read -ra GPUS <<< "$GPUS_CSV"
mkdir -p $ROOT/results/phase4/logs $ROOT/models/phase4

# Job list: every subset. Study-axis subsets carry no seed of their own, so they
# get three training seeds; row and fixed-budget subsets already vary by seed.
JOBS=()
for f in $(ls $SUB/*.npy | sort); do
  b=$(basename "$f" .npy)
  case $b in
    studies*) for s in 0 1 2; do JOBS+=("$b:$s"); done ;;
    *)        JOBS+=("$b:0") ;;
  esac
done

run_one () {
  local sub=$1 seed=$2 gpu=$3
  local tag="${sub}_train${seed}"
  [ -f "$ROOT/results/phase4/${tag}.done" ] && return
  echo "$(date +%H:%M:%S) $(hostname -s) start $tag on cuda:$gpu"
  $PY $ROOT/code/training/trainer_revised.py \
      --data-path "$SUB/${sub}.npy" --seed $seed --device cuda:$gpu \
      --patch-size 1024 --batch-size 32 --num-epochs 400 --patience 50 \
      --d-model 192 --nhead 6 --num-layers 4 --dim-feedforward 768 \
      --num-workers 3 \
      > "$ROOT/results/phase4/logs/${tag}.log" 2>&1 \
    && touch "$ROOT/results/phase4/${tag}.done" \
    && echo "$(date +%H:%M:%S) $(hostname -s) OK $tag" \
    || echo "$(date +%H:%M:%S) $(hostname -s) FAIL $tag"
}

# Take this host's shard, then fill GPU slots as they free up.
MINE=()
for i in "${!JOBS[@]}"; do
  [ $(( i % NSHARD )) -eq $SHARD ] && MINE+=("${JOBS[$i]}")
done
echo "$(hostname -s): ${#MINE[@]} of ${#JOBS[@]} jobs, gpus ${GPUS_CSV}, ${SLOTS} slots each"

declare -A PIDS
i=0
while [ $i -lt ${#MINE[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do
    for slot in $(seq 1 $SLOTS); do
      key="${g}_${slot}"
      if [ -z "${PIDS[$key]:-}" ] && [ $i -lt ${#MINE[@]} ]; then
        IFS=: read -r sub seed <<< "${MINE[$i]}"
        run_one "$sub" "$seed" "$g" &
        PIDS[$key]=$!
        i=$((i+1))
      fi
    done
  done
  sleep 20
  for key in "${!PIDS[@]}"; do
    kill -0 "${PIDS[$key]}" 2>/dev/null || unset PIDS[$key]
  done
done
echo "$(date +%H:%M:%S) $(hostname -s) shard complete"
