#!/usr/bin/env bash
# Rerun every current-pipeline evaluation that touches BrC-T2D after the
# 2026-10-15 label correction (115 spectra -> 78 patients). Previous outputs are
# archived under results/archive_brc115/. Three GPU lanes plus one CPU job.
set -u
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics
cd $ROOT; L=results/rerun_brc_logs; mkdir -p $L
BRC="brc_t2d_cancer brc_t2d_diabetes brc_t2d_4class"
run () { local tag=$1; shift; echo "$(date +%F' '%T) start $tag" >> $L/queue.log
  "$@" > $L/$tag.log 2>&1 && echo "$(date +%F' '%T) OK $tag" >> $L/queue.log \
    || echo "$(date +%F' '%T) FAIL $tag" >> $L/queue.log; }
( run phase1 $PY -u code/evaluation/phase1_classical_bar.py ) &
( run transfer_brc $PY -u code/evaluation/phase3_transfer.py --device cuda:0 --targets $BRC --random-init --out results/phase3/transfer_brc.json
  run scaling45 $PY -u code/evaluation/phase4_scaling_eval.py --device cuda:0 --out results/phase4/scaling_eval_45.json ) &
( run g1_fixed $PY -u code/evaluation/phase3_fixed_eval.py --device cuda:1
  run g3_augment $PY -u code/evaluation/g3_augment_eval.py --device cuda:1 ) &
( run finetune_B $PY -u code/evaluation/phase3_finetune.py --device cuda:2 --targets brc_t2d_cancer brc_t2d_diabetes --random-init --out results/phase5/finetune_B.json
  run finetune_C $PY -u code/evaluation/phase3_finetune.py --device cuda:2 --targets brc_t2d_4class mtbls563_3class --random-init --out results/phase5/finetune_C.json
  run capacity $PY -u code/evaluation/phase6_capacity_eval.py --device cuda:2 ) &
wait; echo "$(date +%F' '%T) ALL DONE" >> $L/queue.log
