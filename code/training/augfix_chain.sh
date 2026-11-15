#!/usr/bin/env bash
# Corrected augmentation reruns (fixed real split, synthetic in training only).
# Waits for the ensnet_nmr training to leave the GPUs, then runs the diffusion
# volume curve (9 runs) and the GAN arm (3 runs), then scores both.
set -u
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics; cd $ROOT
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
LOG=results/final/augfix_chain.log
echo "$(date +%F' '%T) waiting for ensnet_nmr training to finish" >> $LOG
quiet=0
while [ $quiet -lt 2 ]; do      # absent on two checks 5 min apart
  if pgrep -f "ensnet_nmr.cli.train" >/dev/null; then quiet=0; else quiet=$((quiet+1)); fi
  sleep 300
done
echo "$(date +%F' '%T) GPUs free, starting synaug queue" >> $LOG
code/training/synaug_queue.sh 0,1,2 2
echo "$(date +%F' '%T) starting GAN arm" >> $LOG
code/training/g3_augment_queue.sh 0,1,2 2
# keep the pre-fix (pooled-split) scores under *_merged2 names
[ -f results/final/synaug_eval_regional16_merged2.json ] || for s in json meta.json; do
  cp results/final/synaug_eval_regional16.$s results/final/synaug_eval_regional16_merged2.$s
  cp results/g3_augment/g3_augment_eval.$s results/g3_augment/g3_augment_eval_merged2.$s; done
$PY -u code/evaluation/g3_augment_eval.py --device cuda:0 --experiment synaug --pooling regional:16 \
    --out results/final/synaug_eval_regional16.json > results/final/synaug_eval.log 2>&1
echo "$(date +%F' '%T) synaug eval exit $?" >> $LOG
$PY -u code/evaluation/g3_augment_eval.py --device cuda:0 > results/g3_augment/eval.log 2>&1
echo "$(date +%F' '%T) gan eval exit $?" >> $LOG
echo "$(date +%F' '%T) chain complete" >> $LOG
