#!/usr/bin/env python3
"""Barth median against the corpus median, to settle its axis by eye.

Three fitting methods disagreed about Barth's offset -- rigid cross-correlation
-6,759 pts, landmark peak positions ~-600, a small-shift scan -1,337 -- and every
one of them topped out at a correlation below 0.48. When fits disagree that badly
the fits are the wrong instrument. This plots the two medians so the question
"do these even contain the same peaks?" can be answered directly.

Usage:
    python code/plotting/plot_barth_vs_corpus.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPT, PPM_HI, SW = 131_072, 14.8237, 20.024
PPT = SW / NPT
INK, ACC, WARN = "#1f2933", "#2f6f8f", "#b04a3a"

def ppm(i): return PPM_HI - np.asarray(i) * PPT
def med(p): return np.median(np.array(np.load(ROOT / p, mmap_mode="r"), dtype=np.float64), axis=0)

def norm(y):
    y = y - np.median(y)
    return y / (np.percentile(np.abs(y), 99.9) + 1e-30)

C = norm(med("data/combined/corpus_v4_ref0ppm_clean.npy"))
B = norm(med("data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_v4.npy"))
R = norm(med("data/Barth/Barth_EDTASuppressed_v4_corpusaxis.npy"))

WIN = [("aliphatic  0.5-2.2 ppm", 2.2, 0.5),
       ("mid  2.2-3.3 ppm", 3.3, 2.2),
       ("sugar  3.3-4.3 ppm", 4.3, 3.3),
       ("formate reference  8.2-8.7 ppm", 8.7, 8.2)]

fig, axes = plt.subplots(len(WIN), 1, figsize=(11, 12))
for ax, (lab, hi_ppm, lo_ppm) in zip(axes, WIN):
    a, b = int((PPM_HI - hi_ppm) / PPT), int((PPM_HI - lo_ppm) / PPT)
    x = ppm(np.arange(a, b))
    ax.plot(x, C[a:b], lw=0.8, color=ACC, label="corpus median")
    ax.plot(x, B[a:b], lw=0.7, color="#c9a227", alpha=0.75,
            label="Barth, as stored (wrong axis)")
    ax.plot(x, R[a:b], lw=0.9, color=WARN, label="Barth, resampled to corpus axis")
    ax.set_xlim(hi_ppm, lo_ppm); ax.set_title(lab, fontsize=10, color=INK)
    ax.set_ylabel("norm. intensity", fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, fontsize=9)
axes[-1].set_xlabel("ppm")
fig.suptitle("Barth on the corpus axis: SW 12.0308 ppm resampled to 20.024 ppm",
             fontsize=12, color=INK)
fig.tight_layout()
out = ROOT / "results/figures/fig_barth_vs_corpus.png"
fig.savefig(out, dpi=150); print("wrote", out)
