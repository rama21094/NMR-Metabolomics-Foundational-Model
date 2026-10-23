#!/usr/bin/env python3
"""Diagram of the GAN v2 architecture (code/generative/train_spectrum_gan_v2.py).

Shapes follow the code: Gen(zdim=128, ch=512) with 9 x [Upsample x2, Conv1d k9,
BatchNorm, LeakyReLU], channels max(32, c // 2); Disc(ch=32) with 9 x [Conv1d k9
stride 2, LeakyReLU], channels min(512, 2c); GEN_LEN = 32,768.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
INK, BLUE, ORANGE, GREY, BG = "#1E2761", "#2E5E9E", "#B4541A", "#6B7280", "#FFFFFF"
LBLUE, LORANGE, LGREY = "#E8EEF8", "#FCEFE3", "#F2EEE3"

fig, ax = plt.subplots(figsize=(18, 9.6))
ax.set_xlim(0, 181); ax.set_ylim(-7, 96); ax.axis("off")
fig.patch.set_facecolor(BG)


def box(x, y, w, h, title, sub="", fc=LBLUE, ec=BLUE, tc=INK, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), title, ha="center", va="center",
            fontsize=fs, weight="bold", color=tc)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=11, color=GREY)


def arrow(x1, y1, x2, y2, c=INK, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=c, lw=1.6, linestyle=ls))


def stack(x, y, w, h0, h1, fc, ec, label, sub):
    """Trapezoid: height h0 on the left, h1 on the right (signal length growing/shrinking)."""
    ax.add_patch(Polygon([(x, y - h0 / 2), (x + w, y - h1 / 2), (x + w, y + h1 / 2), (x, y + h0 / 2)],
                         closed=True, fc=fc, ec=ec, lw=1.6))
    for i in range(1, 9):
        xx = x + w * i / 9; hh = h0 + (h1 - h0) * i / 9
        ax.plot([xx, xx], [y - hh / 2, y + hh / 2], color=ec, lw=0.6, alpha=0.5)
    ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=13, weight="bold", color=INK)
    ax.text(x + w / 2, y - max(h0, h1) / 2 - 3.2, sub, ha="center", va="top", fontsize=11, color=GREY)


def spectrum(x, y, w, h, c, seed):
    rng = np.random.default_rng(seed); t = np.linspace(0, 1, 400)
    s = sum(a * np.exp(-((t - m) / 0.006) ** 2) for a, m in
            zip(rng.uniform(0.2, 1, 14), rng.uniform(0.15, 0.85, 14)))
    ax.plot(x + t * w, y + s / s.max() * h, color=c, lw=1.1)


# ---------------- Generator (top row) ----------------
ax.text(2, 91, "Generator", fontsize=19, weight="bold", color=BLUE)
gy = 74
box(2, gy - 6, 15, 12, "z ~ N(0, I)", "128 numbers")
arrow(17.8, gy, 21, gy)
box(21, gy - 6, 21, 12, "Linear", "→ 512 ch × 64 pts")
arrow(42.8, gy, 46, gy)
stack(46, gy, 48, 6, 20, LBLUE, BLUE, "9 × upsample block",
      "upsample ×2 → conv1d (k=9) → batch norm → LeakyReLU\nlength 64 → 32,768;  channels 512 → 32")
arrow(94.8, gy, 98, gy)
box(98, gy - 6, 22, 12, "conv1d → tanh", "1 × 32,768")
arrow(120.8, gy, 124, gy)
box(124, gy - 6, 19, 12, "interpolate", "→ 131,072 pts")
arrow(143.8, gy, 147, gy)
ax.add_patch(FancyBboxPatch((147, gy - 7), 31, 14, boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc="#FFFFFF", ec=BLUE, lw=1.6))
spectrum(149, gy - 5, 27, 9, BLUE, 1)
ax.text(162.5, gy + 9.5, "generated spectrum", ha="center", fontsize=12, color=BLUE, weight="bold")

# ---------------- Critic (bottom row) ----------------
ax.text(2, 50, "Critic", fontsize=19, weight="bold", color=ORANGE)
cy = 26
ax.add_patch(FancyBboxPatch((2, cy + 3), 24, 11, boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc="#FFFFFF", ec=INK, lw=1.4))
spectrum(4, cy + 4.5, 20, 7, INK, 7)
ax.text(14, cy + 16.4, "real (mean-pooled ×4)", ha="center", fontsize=11.5, color=INK, weight="bold")
ax.add_patch(FancyBboxPatch((2, cy - 14), 24, 11, boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc="#FFFFFF", ec=BLUE, lw=1.4))
spectrum(4, cy - 12.5, 20, 7, BLUE, 1)
ax.text(14, cy - 17.6, "generated (32,768 pts,\nbefore interpolation)", ha="center", va="top",
        fontsize=11.5, color=BLUE, weight="bold")
arrow(26.8, cy + 8, 31, cy + 1.5); arrow(26.8, cy - 8.5, 31, cy - 1.5)
box(31, cy - 6, 15, 12, "input", "1 × 32,768", fc=LORANGE, ec=ORANGE)
arrow(46.8, cy, 50, cy, c=ORANGE)
stack(50, cy, 48, 20, 6, LORANGE, ORANGE, "9 × strided block",
      "conv1d (k=9, stride 2) → LeakyReLU\nlength 32,768 → 64;  channels 32 → 512")
arrow(98.8, cy, 102, cy, c=ORANGE)
box(102, cy - 6, 24, 12, "flatten → Linear", "512 × 64 → 1", fc=LORANGE, ec=ORANGE)
arrow(126.8, cy, 130, cy, c=ORANGE)
box(130, cy - 6, 14, 12, "score", "", fc=ORANGE, ec=ORANGE, tc="#FFFFFF")
ax.text(137, cy - 2.6, "realness", ha="center", va="center", fontsize=11, color="#FFFFFF")
ax.text(114, cy - 9.5, "+ ⟨study embedding, features⟩\n(study-conditional version only)",
        ha="center", va="top", fontsize=11, color=GREY, style="italic")

# ---------------- Loss (right) ----------------
ax.add_patch(FancyBboxPatch((148, 9), 31, 34, boxstyle="round,pad=0.5,rounding_size=1.5",
                            fc=LGREY, ec=INK, lw=1.4))
ax.text(163.5, 39.5, "Loss (WGAN-GP)", ha="center", fontsize=14, weight="bold", color=INK)
ax.text(149.5, 35,
        "Critic minimises\n"
        r"$D(\mathrm{fake}) - D(\mathrm{real})$" "\n"
        r"$+\;10\,(\|\nabla D(\hat{x})\|_2 - 1)^2$" "\n"
        r"$\hat{x}$: random blend" "\n"
        "of real and fake\n\n"
        "Generator minimises\n"
        r"$-D(\mathrm{fake})$",
        ha="left", va="top", fontsize=11, color=INK, linespacing=1.4)
arrow(144.8, cy, 147.5, cy, c=ORANGE)

ax.text(2, -0.5, "Training: 5 critic steps per generator step · Adam, lr 1e-4, betas (0.5, 0.9) · batch 32 · 300 epochs\n"
        "Stability additions in v2: tanh output bound · generator weights averaged (EMA 0.999) · "
        "checkpoint kept where a separate realism test (AUC) is lowest · stop if generator loss > 10⁴",
        fontsize=12, color=GREY, va="top", linespacing=1.5)

out = ROOT / "docs/figures/fig_wgan_gp_architecture.png"
fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
print(f"wrote {out.relative_to(ROOT)}")
