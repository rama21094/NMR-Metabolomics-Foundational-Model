#!/usr/bin/env python3
"""Two plain figures for the PI deck on where GAN v2 spectra differ from real ones.

1. Window-by-window difference over the FULL axis, -0.5 to 10 ppm. The earlier
   figure (fig_gan_vs_real.py, panel a) reused validate_synthetic.binned(),
   which stops at 9.5 ppm, so the downfield edge looked empty there: it had
   not been tested, not found equal.
2. The two edges, summarised over 2,000 spectra per set instead of a few
   example traces: median and the band holding the middle 80% of spectra, in
   units of 1/1000 of each spectrum's tallest peak.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/analysis"))
from gan_vs_real_features import cliffs_delta  # noqa: E402

WATER = (4.45, 5.10)
GROUND, BLUE, ORANGE, GREY, INK = "#FFFFFF", "#2E5E9E", "#C2571A", "#8A8A8A", "#1E2761"
rng = np.random.default_rng(0)
ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
idx = np.sort(rng.choice(X.shape[0], 4000, replace=False))
R = np.asarray(X[idx], dtype=np.float32); R1, R2 = R[0::2], R[1::2]
S = np.load(ROOT / "rebuild/pretrain/synthetic/gan_v2.npy", mmap_mode="r")
S = np.asarray(S[np.sort(rng.choice(S.shape[0], 2000, replace=False))], dtype=np.float32)


def binned_full(M, width=0.02):
    keep = ~((ppm <= WATER[1]) & (ppm >= WATER[0]))
    edges = np.arange(-0.5, 10.0 + 1e-9, width)
    which = np.digitize(ppm, edges)
    out, cen = [], []
    for b in np.unique(which[keep]):
        s = keep & (which == b)
        out.append(M[:, s].mean(1)); cen.append(ppm[s].mean())
    out = np.stack(out, 1); a = np.abs(out).sum(1, keepdims=True); a[a == 0] = 1
    return out / a, np.array(cen)


B1, c = binned_full(R1); B2, _ = binned_full(R2); BS, _ = binned_full(S)
d_syn = np.array([cliffs_delta(B1[:, j], BS[:, j]) for j in range(len(c))])
d_nul = np.array([cliffs_delta(B1[:, j], B2[:, j]) for j in range(len(c))])
thr = float(np.percentile(np.abs(d_nul), 99))
zones = {"upfield edge (< -0.2 ppm)": c < -0.2, "downfield edge (> 9.4 ppm)": c > 9.4,
         "middle (-0.2 to 9.4 ppm)": (c >= -0.2) & (c <= 9.4)}
summary = {"null_99": thr}
for k, m in zones.items():
    summary[k] = {"windows": int(m.sum()), "median_abs_delta": float(np.median(np.abs(d_syn[m]))),
                  "share_beyond_null99": float(np.mean(np.abs(d_syn[m]) > thr))}
    print(f"{k:<28} {m.sum():>4} windows  median |delta| {summary[k]['median_abs_delta']:.2f}"
          f"  beyond real-vs-real {summary[k]['share_beyond_null99']:.0%}")
json.dump(summary, open(ROOT / "results/phase5/gan_window_fullrange.json", "w"), indent=1)

plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})

# ---- figure 1: window differences, full axis --------------------------------
fig, ax = plt.subplots(figsize=(16, 5.2))
ax.bar(c, d_syn, width=0.018, color=np.where(d_syn > 0, ORANGE, BLUE), alpha=0.9)
ax.axhspan(-thr, thr, color=GREY, alpha=0.3, lw=0)
for lo, hi in ((-0.55, -0.2), (9.4, 10.05)):
    ax.axvspan(lo, hi, color="#E8C9A0", alpha=0.35, lw=0)
ax.axhline(0, color="#555", lw=0.6)
ax.set_xlim(10.05, -0.55); ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("chemical shift (ppm)")
ax.set_ylabel("difference, GAN vs real\n(Cliff's delta, -1 to +1)")
ax.text(9.97, 0.97, "orange = GAN higher   blue = GAN lower   grey band = real vs real",
        fontsize=13, color="#333", va="top")
ax.text(9.72, -0.98, "edge", fontsize=13, color="#8A5A2B", ha="center")
ax.text(-0.36, -0.98, "edge", fontsize=13, color="#8A5A2B", ha="center")
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/pi16_gan_windows.png", dpi=170, facecolor="white")

# ---- figure 2: the edges, summarised -----------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True)
for ax, (lo, hi, title) in zip(axs, ((9.4, 10.0, "Downfield edge, 9.4 to 10 ppm"),
                                     (-0.5, -0.2, "Upfield edge, -0.5 to -0.2 ppm"))):
    m = np.where((ppm >= lo) & (ppm <= hi))[0][::8]
    for M, col, lab in ((R1, INK, "real"), (S, ORANGE, "GAN v2")):
        y = M[:, m] / np.abs(M).max(1, keepdims=True) * 1e3
        q10, q50, q90 = np.percentile(y, [10, 50, 90], axis=0)
        ax.fill_between(ppm[m], q10, q90, color=col, alpha=0.25, lw=0,
                        label=f"{lab}: middle 80% of spectra")
        ax.plot(ppm[m], q50, color=col, lw=2, label=f"{lab}: median")
    ax.axhline(0, color="#555", lw=0.6)
    ax.set_xlim(hi, lo); ax.set_title(title, loc="left", color=INK, weight="bold")
    ax.set_xlabel("chemical shift (ppm)")
axs[0].set_ylabel("intensity, in thousandths\nof the spectrum's tallest peak")
axs[0].set_ylim(-2.5, 2.5)
axs[1].legend(fontsize=12, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/pi16_gan_edges.png", dpi=170, facecolor="white")
print("wrote docs/figures/pi16_gan_windows.png, pi16_gan_edges.png")
