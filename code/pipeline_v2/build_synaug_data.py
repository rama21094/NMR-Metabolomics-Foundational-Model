#!/usr/bin/env python3
"""Synthetic-volume arms for item 5: 1k / 4k / 16k even-mix diffusion spectra.
Real corpus is linked under a per-arm name so checkpoint names cannot collide."""
import os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "rebuild/pretrain/synaug"; D.mkdir(parents=True, exist_ok=True)
A = np.load(ROOT / "rebuild/pretrain/synthetic/ldm_v2_uniform.npy", mmap_mode="r")
B = np.load(ROOT / "rebuild/pretrain/synthetic/ldm_v2_uniform_b.npy", mmap_mode="r")
assert A.shape == (4000, 131072) and B.shape == (12000, 131072), (A.shape, B.shape)
for v, n in (("1k", 1000), ("4k", 4000), ("16k", 16000)):
    link = D / f"synaug{v}_real.npy"
    if not link.exists():
        os.symlink("../corpus_train.npy", link)
    out = np.lib.format.open_memmap(D / f"synaug{v}_synth.npy", mode="w+", dtype=np.float32, shape=(n, 131072))
    out[:min(n, 4000)] = A[:min(n, 4000)]
    if n > 4000:
        out[4000:] = B[:n - 4000]
    out.flush(); print(f"synaug{v}: {n} synthetic spectra")
