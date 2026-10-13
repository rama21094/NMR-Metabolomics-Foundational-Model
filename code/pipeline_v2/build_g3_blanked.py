#!/usr/bin/env python3
"""Edge-blanked pretraining arrays for the G3 real-vs-real+GAN test.

Writes rebuild/pretrain/g3/{g3real_blank,g3gan_v2_blank}.npy. The names
deliberately do NOT start with "corpus_train": the Phase 3 checkpoint glob
matches that prefix, and these checkpoints share the baseline architecture.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/synthesis"))
from edges import edge_mask  # noqa: E402

ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
cols = edge_mask(ppm)
out = ROOT / "rebuild/pretrain/g3"; out.mkdir(parents=True, exist_ok=True)
for src, dst in (("rebuild/pretrain/corpus_train.npy", "g3real_blank.npy"),
                 ("rebuild/pretrain/synthetic/gan_v2.npy", "g3gan_v2_blank.npy")):
    X = np.load(ROOT / src, mmap_mode="r")
    Y = np.lib.format.open_memmap(out / dst, mode="w+", dtype=np.float32, shape=X.shape)
    for i in range(0, X.shape[0], 500):
        b = np.array(X[i:i+500], dtype=np.float32); b[:, cols] = 0.0; Y[i:i+500] = b
    Y.flush()
    print(f"{dst}: {X.shape}, {cols.sum()} of {len(ppm)} columns zeroed")
