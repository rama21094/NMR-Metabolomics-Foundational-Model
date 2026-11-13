#!/usr/bin/env bash
# Item 5 pipeline: wait for the autoencoders, then train the latent diffusion
# model and sample from everything. Logs to results/final/ldm_v2/.
set -u
PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
ROOT=/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics; L=$ROOT/results/final/ldm_v2
cd $ROOT
wait_for () { while pgrep -f "[l]dm_v2.py ae --tag $1 " >/dev/null; do sleep 60; done; }
wait_for vae_b1e-2
$PY -u code/generative/ldm_v2.py vae-sample --ae vae_b1e-2 --tag vae_v2 --device cuda:2 > $L/vae_sample.log 2>&1 &
wait_for ae_b1e-6
$PY -u code/generative/ldm_v2.py ldm --ae ae_b1e-6 --tag ldm1 --device cuda:1 > $L/ldm1.log 2>&1
for mix in proportional uniform; do
  $PY -u code/generative/ldm_v2.py sample --ldm ldm1 --tag ldm_v2_${mix} --study-mix $mix --guidance 1.5 \
      --device cuda:1 > $L/sample_${mix}.log 2>&1
done
echo "$(date +%F' '%T) chain complete" >> $L/chain.log
