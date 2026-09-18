#!/usr/bin/env bash
# Finish the two seed-1 study-axis jobs left in yttrium's serial shard, on an idle host.
# run_one is a copy of the queue's: it skips any tag that already has a .done file,
# so running this alongside the queue cannot duplicate work.
set -u
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
SUB=$ROOT/rebuild/pretrain/scaling
LOG=$ROOT/results/phase4/queue_rebal.log
mkdir -p $ROOT/results/phase4/logs
run_one () {
  local sub=$1 gpu=$2 tag="${1}_train1"
  [ -f "$ROOT/results/phase4/${tag}.done" ] && return
  echo "$(date +%H:%M:%S) $(hostname -s) start $tag on cuda:$gpu" >> $LOG
  $PY $ROOT/code/training/trainer_revised.py \
      --data-path "$SUB/${sub}.npy" --seed 1 --device cuda:$gpu \
      --patch-size 1024 --batch-size 32 --num-epochs 400 --patience 50 \
      --d-model 192 --nhead 6 --num-layers 4 --dim-feedforward 768 \
      --num-workers 3 > "$ROOT/results/phase4/logs/${tag}.log" 2>&1 \
    && touch "$ROOT/results/phase4/${tag}.done" \
    && echo "$(date +%H:%M:%S) $(hostname -s) OK $tag" >> $LOG \
    || echo "$(date +%H:%M:%S) $(hostname -s) FAIL $tag" >> $LOG
}
run_one studies6 1 &
run_one studies9 2 &
wait
echo "$(date +%H:%M:%S) $(hostname -s) rebalance complete" >> $LOG
