#!/usr/bin/env bash
# Phase 6 -- capacity sweep ("if no - more parameters").
#
# The whiteboard's branch after a G3 "no". It is run as a TWO-SIDED test, not as
# "add parameters": the G1 fine-tuning result showed the 4.0M baseline already
# overfits 40-142 sample cohorts (full fine-tune worse than last-block, worse
# than frozen). If that reading is right, capacity BELOW the baseline should do
# as well or better, and capacity above should do worse. A sweep that only went
# upward could not distinguish "capacity is not the lever" from "we did not add
# enough".
#
# nhead divides d_model in every row. patch_size is held at 1024 throughout: the
# prior project already swept it (1024 vs 256 vs 128) and 1024 won 5 of 5, so it
# is not re-opened here.
#
# The baseline row is NOT retrained -- five seeds already exist under
# models/masked_ssl from Phase 2 and are reused by the evaluator.
#
# Usage: phase6_capacity_queue.sh <gpu-list> <slots-per-gpu>
set -u
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics
DATA=$ROOT/rebuild/pretrain/corpus_train.npy
GPUS_CSV=${1:-2}; SLOTS=${2:-2}
IFS=, read -ra GPUS <<< "$GPUS_CSV"
mkdir -p $ROOT/results/phase6/logs $ROOT/models/phase6

#      name     d_model nhead layers ff     (approx params)
ROWS=(
  "tiny        96  4 2  384"
  "small      128  4 3  512"
  "large      320  8 6 1280"
  "xlarge     448  8 8 1792"
)
SEEDS=(0 1 2)

JOBS=()
for r in "${ROWS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$r|$s"); done; done

run_one () {
  local name=$1 dm=$2 nh=$3 nl=$4 ff=$5 seed=$6 gpu=$7
  local tag="cap_${name}_seed${seed}"
  [ -f "$ROOT/results/phase6/${tag}.done" ] && return
  echo "$(date +%H:%M:%S) start $tag (d_model=$dm layers=$nl) on cuda:$gpu" \
    >> $ROOT/results/phase6/queue.log
  $PY $ROOT/code/training/trainer_revised.py \
      --data-path "$DATA" --seed $seed --device cuda:$gpu \
      --patch-size 1024 --batch-size 32 --num-epochs 400 --patience 50 \
      --d-model $dm --nhead $nh --num-layers $nl --dim-feedforward $ff \
      --num-workers 3 \
      > "$ROOT/results/phase6/logs/${tag}.log" 2>&1 \
    && touch "$ROOT/results/phase6/${tag}.done" \
    && echo "$(date +%H:%M:%S) OK $tag" >> $ROOT/results/phase6/queue.log \
    || echo "$(date +%H:%M:%S) FAIL $tag" >> $ROOT/results/phase6/queue.log
}

echo "$(date +%H:%M:%S) sweep start: ${#JOBS[@]} jobs" >> $ROOT/results/phase6/queue.log
declare -A PIDS
i=0
while [ $i -lt ${#JOBS[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do
    for slot in $(seq 1 $SLOTS); do
      key="${g}_${slot}"
      if [ -z "${PIDS[$key]:-}" ] && [ $i -lt ${#JOBS[@]} ]; then
        IFS='|' read -r spec seed <<< "${JOBS[$i]}"
        read -r nm dm nh nl ff <<< "$spec"
        run_one "$nm" "$dm" "$nh" "$nl" "$ff" "$seed" "$g" &
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
echo "$(date +%H:%M:%S) sweep complete" >> $ROOT/results/phase6/queue.log
