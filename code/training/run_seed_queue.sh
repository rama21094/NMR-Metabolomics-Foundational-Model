#!/usr/bin/env bash
# Serialised, memory-gated launcher for the seed-replicate study (docs §15).
#
# Why this exists: on 2026-08-09 four concurrent trainings OOM-killed this host.
# GPU capacity was never the limit (770 MB of 46 GB per card); system RAM was.
# Even after the trainer's memory fix (commit 119f025) the LOAD phase still
# peaks near 25 GB per process -- np.unique on the ~10 GB float64 corpus needs
# the input and a sorted copy simultaneously -- while steady-state training is
# far cheaper. So the danger is two jobs hitting np.unique at the same moment.
#
# This launcher therefore starts one run at a time and refuses to start the next
# until (a) enough RAM is free for a load peak, and (b) the previous run is past
# its own peak (its first epoch has been logged). Steady-state concurrency is
# fine; simultaneous load peaks are not.
#
# Usage:  bash code/training/run_seed_queue.sh
# Safe to re-run: any queue entry whose log already shows "Training completed
# after" is skipped, so an interrupted queue resumes where it left off.

set -u
cd "$(dirname "$0")/../.." || exit 1

PY=/home/nmrbox/0012/shasharma/anaconda3/envs/NMR/bin/python
V3=data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3.npy
V4=data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4.npy

# Free GB required before starting a run. The load peak is ~25 GB; the margin
# covers the 26 other users on this shared host.
MIN_FREE_GB=45

# tag : gpu : seed : corpus
QUEUE=(
  "v3_seed404:1:404:$V3"
  "v4_seed303:2:303:$V4"
  "v4_seed404:1:404:$V4"
)

free_gb() { free -g | awk 'NR==2{print $7}'; }   # "available", not "free":
                                                 # excludes reclaimable cache

for entry in "${QUEUE[@]}"; do
  IFS=: read -r tag gpu seed corpus <<< "$entry"
  log="logs/exp15_${tag}.log"

  if [ -f "$log" ] && grep -q "Training completed after" "$log" 2>/dev/null; then
    echo "[queue] $tag already completed -- skipping"
    continue
  fi

  # Wait for RAM headroom.
  while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do
    echo "[queue] $tag waiting for RAM: $(free_gb)Gi available, need ${MIN_FREE_GB}Gi"
    sleep 120
  done

  echo "[queue] launching $tag on GPU $gpu (seed $seed), $(free_gb)Gi available"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u code/training/trainer_revised.py \
      --device cuda:0 --patch-size 1024 --nhead 4 --seed "$seed" \
      --data-path "$corpus" > "$log" 2>&1 &
  pid=$!
  echo "[queue] $tag pid $pid"

  # Hold until this run is past its own load peak (first epoch logged), so the
  # next launch cannot collide with it. Bail out if it dies during load.
  while ! grep -q "Epoch 1:" "$log" 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[queue] WARNING: $tag exited during load -- see $log"
      break
    fi
    sleep 20
  done
  echo "[queue] $tag past load peak ($(free_gb)Gi available); continuing"
done

echo "[queue] all entries dispatched"
