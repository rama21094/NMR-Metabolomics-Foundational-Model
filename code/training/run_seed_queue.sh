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

# Hard cap on concurrent trainings. The RAM gate alone is NOT sufficient: it
# checks availability at launch, but each admitted job then settles at ~28 GB
# (memory-fixed) to ~37 GB (pre-fix) resident. On 2026-08-09 the gate correctly
# saw 45 GB free and admitted a 4th job, after which the host sat at 172/188 GB
# with swap fully exhausted -- the same state that had already OOM-killed two
# runs earlier that day. Steady-state totals, not launch-time headroom, are what
# bound concurrency here, so cap it explicitly.
MAX_CONCURRENT=3

# tag : gpu : seed : corpus
QUEUE=(
  "v3_seed404:1:404:$V3"
  "v4_seed303:2:303:$V4"
  "v4_seed404:1:404:$V4"   # killed once at 43 epochs to relieve memory pressure
)

free_gb() { free -g | awk 'NR==2{print $7}'; }   # "available", not "free":
                                                 # excludes reclaimable cache

# Count only parent training processes. Two things must be excluded:
#   - forked DataLoader workers (16 per job), which would inflate the count ~17x
#     -> skip any process whose parent is itself a trainer
#   - pgrep -f SELF-MATCHES: any shell whose command line happens to contain the
#     pattern (including the one running this check) is returned by pgrep
#     -> require the executable itself to be python, which excludes bash
running_trainings() {
  local n=0 p ppid comm
  for p in $(pgrep -f "trainer_revised.py --device" 2>/dev/null); do
    comm=$(ps -o comm= -p "$p" 2>/dev/null)
    case "$comm" in python*) ;; *) continue ;; esac
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    ps -p "$ppid" -o cmd= 2>/dev/null | grep -q trainer_revised || n=$((n+1))
  done
  echo "$n"
}

for entry in "${QUEUE[@]}"; do
  IFS=: read -r tag gpu seed corpus <<< "$entry"
  log="logs/exp15_${tag}.log"

  if [ -f "$log" ] && grep -q "Training completed after" "$log" 2>/dev/null; then
    echo "[queue] $tag already completed -- skipping"
    continue
  fi

  # Skip anything already in flight. Without this, restarting the queue after an
  # interruption relaunches a DUPLICATE of every run still training (their logs
  # have no "Training completed after" yet), doubling memory use -- exactly what
  # this launcher exists to prevent. Match the exact seed AND corpus, requiring
  # the executable to be python so a shell whose command line contains the
  # pattern cannot self-match.
  already_running=0
  for rp in $(pgrep -f "trainer_revised.py --device" 2>/dev/null); do
    rcomm=$(ps -o comm= -p "$rp" 2>/dev/null)
    case "$rcomm" in python*) ;; *) continue ;; esac
    rcmd=$(ps -o cmd= -p "$rp" 2>/dev/null)
    case "$rcmd" in *"--seed $seed "*"$corpus"*) already_running=1 ;; esac
  done
  if [ "$already_running" -eq 1 ]; then
    echo "[queue] $tag already running -- skipping"
    continue
  fi

  # Wait for BOTH a free concurrency slot and RAM headroom.
  while :; do
    running=$(running_trainings)
    avail=$(free_gb)
    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
      echo "[queue] $tag waiting for a slot: $running/$MAX_CONCURRENT trainings running"
    elif [ "$avail" -lt "$MIN_FREE_GB" ]; then
      echo "[queue] $tag waiting for RAM: ${avail}Gi available, need ${MIN_FREE_GB}Gi"
    else
      break
    fi
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
