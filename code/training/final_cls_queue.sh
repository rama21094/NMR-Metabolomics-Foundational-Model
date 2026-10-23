#!/usr/bin/env bash
# Final experiment 4 (Shyam meeting 2026-10-23): CLS token, pretrained on the
# FULL rebuilt corpus (8,545 training + 935 formerly held-out spectra = 9,480;
# effective studies 4.09 -> 5.91). Two arms, baseline architecture, 3 seeds:
#   full      : full corpus, no CLS  (control for the corpus change)
#   fullcls   : full corpus, CLS token
# Data files are symlinks named full9480_* so checkpoints are named
# full9480_train_merged2_* and cannot match the Phase 3 "corpus_train_*" glob.
# Usage: final_cls_queue.sh <gpu-list> <slots-per-gpu>
set -u
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics
D=$ROOT/rebuild/pretrain/full
GPUS_CSV=${1:-0,1,2}; SLOTS=${2:-2}
IFS=, read -ra GPUS <<< "$GPUS_CSV"
mkdir -p $ROOT/results/final/logs
JOBS=(); for arm in full fullcls; do for s in 0 1 2; do JOBS+=("$arm|$s"); done; done
run_one () {
  local arm=$1 seed=$2 gpu=$3 tag="final_${1}_seed${2}"
  [ -f "$ROOT/results/final/${tag}.done" ] && return
  local extra=(); [ "$arm" = fullcls ] && extra=(--cls-token)
  echo "$(date +%F' '%T) start $tag on cuda:$gpu" >> $ROOT/results/final/queue.log
  $PY -u $ROOT/code/training/trainer_revised.py \
      --data-path "$D/full9480_train.npy" "$D/full9480_heldout.npy" --seed $seed --device cuda:$gpu \
      --patch-size 1024 --batch-size 32 --num-epochs 400 --patience 50 \
      --d-model 192 --nhead 6 --num-layers 4 --dim-feedforward 768 \
      --num-workers 3 "${extra[@]}" \
      > "$ROOT/results/final/logs/${tag}.log" 2>&1 \
    && touch "$ROOT/results/final/${tag}.done" \
    && echo "$(date +%F' '%T) OK $tag" >> $ROOT/results/final/queue.log \
    || echo "$(date +%F' '%T) FAIL $tag" >> $ROOT/results/final/queue.log
}
declare -A PIDS; i=0
while [ $i -lt ${#JOBS[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do for slot in $(seq 1 $SLOTS); do
    key="${g}_${slot}"
    if [ -z "${PIDS[$key]:-}" ] && [ $i -lt ${#JOBS[@]} ]; then
      IFS='|' read -r arm seed <<< "${JOBS[$i]}"
      run_one "$arm" "$seed" "$g" & PIDS[$key]=$!; i=$((i+1))
    fi
  done; done
  sleep 20
  for key in "${!PIDS[@]}"; do kill -0 "${PIDS[$key]}" 2>/dev/null || unset PIDS[$key]; done
done
echo "$(date +%F' '%T) queue complete" >> $ROOT/results/final/queue.log
