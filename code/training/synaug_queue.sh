#!/usr/bin/env bash
# Item 5 pretraining curve: real corpus (8,545) + 1k / 4k / 16k diffusion spectra
# generated with an EVEN study mix (ldm_v2, uniform; 9.8 effective studies,
# stratified realism AUC 0.88). Baseline architecture, 3 seeds per volume.
# Control = Phase 2 masking baseline (same real corpus, same architecture).
# Each arm has its own first data file (synaug<N>_real.npy, a symlink to the
# real corpus) so checkpoint names cannot collide across arms.
set -u
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics; D=$ROOT/rebuild/pretrain/synaug
GPUS_CSV=${1:-0,1,2}; SLOTS=${2:-2}; IFS=, read -ra GPUS <<< "$GPUS_CSV"
mkdir -p $ROOT/results/final/synaug_logs
JOBS=(); for v in 1k 4k 16k; do for s in 0 1 2; do JOBS+=("$v|$s"); done; done
run_one () {
  local v=$1 seed=$2 gpu=$3 tag="synaug${1}_seed${2}"
  [ -f "$ROOT/results/final/${tag}.done" ] && return
  echo "$(date +%F' '%T) start $tag on cuda:$gpu" >> $ROOT/results/final/synaug_queue.log
  $PY -u $ROOT/code/training/trainer_revised.py \
      --data-path "$D/synaug${v}_real.npy" "$D/synaug${v}_synth.npy" --seed $seed --device cuda:$gpu \
      --patch-size 1024 --batch-size 32 --num-epochs 400 --patience 50 \
      --d-model 192 --nhead 6 --num-layers 4 --dim-feedforward 768 --num-workers 3 \
      > "$ROOT/results/final/synaug_logs/${tag}.log" 2>&1 \
    && touch "$ROOT/results/final/${tag}.done" \
    && echo "$(date +%F' '%T) OK $tag" >> $ROOT/results/final/synaug_queue.log \
    || echo "$(date +%F' '%T) FAIL $tag" >> $ROOT/results/final/synaug_queue.log
}
declare -A PIDS; i=0
while [ $i -lt ${#JOBS[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do for slot in $(seq 1 $SLOTS); do
    key="${g}_${slot}"
    if [ -z "${PIDS[$key]:-}" ] && [ $i -lt ${#JOBS[@]} ]; then
      IFS='|' read -r v seed <<< "${JOBS[$i]}"; run_one "$v" "$seed" "$g" & PIDS[$key]=$!; i=$((i+1))
    fi
  done; done
  sleep 20
  for key in "${!PIDS[@]}"; do kill -0 "${PIDS[$key]}" 2>/dev/null || unset PIDS[$key]; done
done
echo "$(date +%F' '%T) queue complete" >> $ROOT/results/final/synaug_queue.log
