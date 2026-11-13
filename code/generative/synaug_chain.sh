#!/usr/bin/env bash
set -u
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics; cd $ROOT
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
while [ ! -f rebuild/pretrain/synthetic/ldm_v2_uniform_b.npy ] || pgrep -f "[l]dm_v2.py sample --ldm ldm1 --tag ldm_v2_uniform_b" >/dev/null; do sleep 60; done
$PY code/pipeline_v2/build_synaug_data.py > results/final/synaug_build.log 2>&1 && code/training/synaug_queue.sh 0,1,2 2
