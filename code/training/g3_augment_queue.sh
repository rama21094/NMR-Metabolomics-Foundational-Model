#!/usr/bin/env bash
# G3 -- does adding GAN v2 spectra to pretraining help transfer?
#
# Two arms, baseline masking architecture (192/6/4/768), 3 seeds each:
#   realblank  : the real corpus, edges blanked             (control)
#   realgan    : the real corpus + 4,000 GAN v2, edges blanked
# Both arms are blanked so the only difference between them is the GAN spectra.
# The existing unblanked Phase 2 baseline is NOT the control: it differs in two
# ways (edges and GAN), so a difference against it would be uninterpretable.
#
# Caveat: trainer_revised takes a random 80/20 split for early stopping, so in
# the realgan arm about 800 of the validation spectra are synthetic.
#
# Usage: g3_augment_queue.sh <gpu-list> <slots-per-gpu>
set -u
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics
D=$ROOT/rebuild/pretrain/g3
GPUS_CSV=${1:-0,1,2}; SLOTS=${2:-2}
IFS=, read -ra GPUS <<< "$GPUS_CSV"
mkdir -p $ROOT/results/g3_augment/logs

JOBS=()
for arm in realblank realgan; do for s in 0 1 2; do JOBS+=("$arm|$s"); done; done

run_one () {
  local arm=$1 seed=$2 gpu=$3 tag="g3_${1}_seed${2}"
  [ -f "$ROOT/results/g3_augment/${tag}.done" ] && return
  local data=("$D/g3real_blank.npy")
  [ "$arm" = realgan ] && data+=("$D/g3gan_v2_blank.npy")
  echo "$(date +%F' '%H:%M:%S) start $tag on cuda:$gpu" >> $ROOT/results/g3_augment/queue.log
  $PY $ROOT/code/training/trainer_revised.py \
      --data-path "${data[@]}" --seed $seed --device cuda:$gpu \
      --patch-size 1024 --batch-size 32 --num-epochs 400 --patience 50 \
      --d-model 192 --nhead 6 --num-layers 4 --dim-feedforward 768 \
      --num-workers 3 \
      > "$ROOT/results/g3_augment/logs/${tag}.log" 2>&1 \
    && touch "$ROOT/results/g3_augment/${tag}.done" \
    && echo "$(date +%F' '%H:%M:%S) OK $tag" >> $ROOT/results/g3_augment/queue.log \
    || echo "$(date +%F' '%H:%M:%S) FAIL $tag" >> $ROOT/results/g3_augment/queue.log
}

declare -A PIDS
i=0
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
echo "$(date +%F' '%H:%M:%S) queue complete" >> $ROOT/results/g3_augment/queue.log
