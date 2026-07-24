#!/usr/bin/env python3
"""Presentation/paper-ready architecture diagrams for the three SSL model families.

Draws clean schematic pipelines (not checkpoint-shape introspection -- see
plot_model_architecture.py for that) for:
  1. Masked-reconstruction autoencoder (trainer_revised.NMRMaskedAutoencoder)
  2. Jigsaw / multibin permutation model (train_jigsaw_spectra.JigsawNMRModel)
  3. Joint masked + jigsaw SSL model (train_joint_ssl.JointNMRSSLModel)
  4. A combined overview panel (all three stacked, for a single methods figure)

Hyperparameters annotated on each diagram are read from the actual deployed
MetaboLights-corpus checkpoints, not placeholders. Saves both .png (300dpi,
for slides) and .pdf (vector, for LaTeX/journal submission) per figure.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# Same validated categorical palette used in plot_all_datasets_summary.py.
FAMILY_COLORS = {
    "classical": "#2a78d6",
    "masked": "#1baf7a",
    "jigsaw": "#eda100",
    "joint_ssl": "#e34948",
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def box(ax, xy, w, h, text, facecolor="white", edgecolor=INK, fontsize=10, fontweight="normal", text_color=INK, zorder=3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.3, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, color=text_color, zorder=zorder + 1, wrap=True)
    return patch


def arrow(ax, start, end, color=MUTED, style="-|>", lw=1.6, zorder=2):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14,
                             color=color, linewidth=lw, zorder=zorder, shrinkA=2, shrinkB=2)
    ax.add_patch(patch)


def side_note(ax, xy, text, fontsize=8, color=MUTED, ha="left"):
    ax.text(xy[0], xy[1], text, ha=ha, va="center", fontsize=fontsize, color=color, linespacing=1.5)


def new_figure(width, height, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(0.15, height - 0.35, title, fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        ax.text(0.15, height - 0.75, subtitle, fontsize=9.5, color=MUTED, ha="left", va="top")
    return fig, ax


def save(fig, out_path: Path):
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.with_suffix('.png')} and {out_path.with_suffix('.pdf')}")


def masked_architecture_figure(out_path: Path):
    color = FAMILY_COLORS["masked"]
    fig, ax = new_figure(12, 5.2, "Masked-reconstruction SSL",
                          "trainer_revised.NMRMaskedAutoencoder  --  patch=1024, d_model=128, 3 layers, 8 heads, FFN=256")

    y = 2.4
    h = 1.0
    box(ax, (0.3, y), 1.7, h, "Input\nspectrum\n(131,072 pts)", facecolor="white")
    arrow(ax, (2.0, y + h / 2), (2.5, y + h / 2))

    box(ax, (2.5, y), 1.7, h, "Patchify\n128 patches\n(1024 pts each)", facecolor="white")
    arrow(ax, (4.2, y + h / 2), (4.7, y + h / 2))

    box(ax, (4.7, y), 1.9, h, "Patch embedding\n(Linear+LayerNorm)\n→ d=128", facecolor="white", fontsize=8.3)
    arrow(ax, (6.6, y + h / 2), (7.1, y + h / 2))

    # masking annotation above
    box(ax, (7.1, y + 0.05), 1.9, h - 0.1, "Random mask\n20-60% of patches\n→ [MASK] token", facecolor="#eef8f2", edgecolor=color)
    arrow(ax, (9.0, y + h / 2), (9.5, y + h / 2))

    box(ax, (9.5, y - 0.15), 2.1, h + 0.3, "Shared Transformer\nencoder × 3 layers\n(pre-norm, 8 heads)", facecolor=color, text_color="white", fontweight="bold")
    arrow(ax, (11.6, y + h / 2), (11.6, y + h / 2))

    y2 = 0.5
    arrow(ax, (10.55, y - 0.15), (10.55, y2 + h))
    box(ax, (9.5, y2), 2.1, h, "Reconstruction head\nLinear→GELU→Linear", facecolor="white")
    arrow(ax, (9.5, y2 + h / 2), (7.7, y2 + h / 2))
    box(ax, (5.9, y2), 1.8, h, "+0.3× skip\n(unmasked patches\nonly)", facecolor="#f5f5f4", edgecolor=MUTED, fontsize=8.5)
    arrow(ax, (5.9, y2 + h / 2), (4.3, y2 + h / 2))
    box(ax, (2.2, y2), 2.1, h, "Reconstructed\nspectrum", facecolor="white")

    side_note(ax, (0.3, y2 + h + 0.35), "Loss: MSE on masked patches only, evaluated against the original (unmasked) input.", fontsize=8.5)
    save(fig, out_path)


def jigsaw_architecture_figure(out_path: Path):
    color = FAMILY_COLORS["jigsaw"]
    fig, ax = new_figure(12, 5.2, "Jigsaw (multibin) SSL",
                          "train_jigsaw_spectra.JigsawNMRModel  --  bin sizes {256,512,1024,2048}, d_model=192, 4 layers, 6 heads, FFN=768")

    y = 2.4
    h = 1.0
    box(ax, (0.3, y), 1.7, h, "Input\nspectrum\n(131,072 pts)", facecolor="white")
    arrow(ax, (2.0, y + h / 2), (2.5, y + h / 2))

    box(ax, (2.5, y), 2.0, h, "Bin at one of\n{256,512,1024,2048}\npts/bin", facecolor="white")
    arrow(ax, (4.5, y + h / 2), (5.0, y + h / 2))

    box(ax, (5.0, y + 0.05), 1.9, h - 0.1, "Shuffle bin order\n(random permutation)", facecolor="#fdf3e0", edgecolor=color)
    arrow(ax, (6.9, y + h / 2), (7.4, y + h / 2))

    box(ax, (7.4, y), 2.0, h, "Per-bin-size linear\nprojection + LayerNorm\n→ d=192", facecolor="white", fontsize=9)
    arrow(ax, (9.4, y + h / 2), (9.9, y + h / 2))

    box(ax, (9.9, y + 0.05), 1.8, h - 0.1, "+ learnable\nslot embedding\n(shuffled position)", facecolor="#fdf3e0", edgecolor=color, fontsize=8.5)

    y2 = 0.5
    arrow(ax, (10.8, y), (10.8, y2 + h + 0.3))
    box(ax, (9.6, y2 + h * 0.15), 2.4, h + 0.3, "Shared Transformer\nencoder × 4 layers\n(pre-norm, 6 heads)", facecolor=color, text_color="white", fontweight="bold")
    arrow(ax, (9.6, y2 + h / 2 + 0.15), (7.6, y2 + h / 2 + 0.15))

    box(ax, (5.6, y2), 2.0, h, "Per-bin-size classifier\n(Linear → n_bins-way)", facecolor="white", fontsize=9)
    arrow(ax, (5.6, y2 + h / 2), (3.9, y2 + h / 2))
    box(ax, (2.0, y2), 1.9, h, "Predicted original\nbin position\n(per token)", facecolor="white")

    side_note(ax, (0.3, y2 - 0.35), "Loss: cross-entropy between predicted and true original bin index (label smoothing 0.05); trained across all 4 bin sizes.", fontsize=8.5)
    save(fig, out_path)


def joint_architecture_figure(out_path: Path):
    color = FAMILY_COLORS["joint_ssl"]
    width, height = 13.4, 8.3
    fig, ax = new_figure(width, height, "Joint masked + jigsaw SSL",
                          "train_joint_ssl.JointNMRSSLModel  --  shared d_model=192, 4 layers, 6 heads, FFN=768, task_embed_dim=8 (bottlenecked), Fourier position bands=8")

    # two input branches
    y_top, y_bot = 5.7, 3.9
    h = 1.0
    box(ax, (0.3, y_top), 2.0, h, "Input spectrum\n→ bin @ 1024 pts,\nmask 20-60%", facecolor="#eef8f2", edgecolor=FAMILY_COLORS["masked"], fontsize=8.7)
    box(ax, (0.3, y_bot), 2.0, h, "Input spectrum\n→ bin @ {256,512,\n1024,2048}, shuffle", facecolor="#fdf3e0", edgecolor=FAMILY_COLORS["jigsaw"], fontsize=8.7)
    ax.text(0.3, y_bot - 0.35, "(one task sampled per training step)", fontsize=8, color=MUTED, ha="left", style="italic")

    arrow(ax, (2.3, y_top + h / 2), (3.0, y_top + h / 2 + 0.25))
    arrow(ax, (2.3, y_bot + h / 2), (3.0, y_bot + h / 2 - 0.25))

    mid_y = (y_top + y_bot) / 2 + h / 2
    box(ax, (3.0, mid_y - 0.8), 2.3, 1.6, "Shared per-bin-size\nlinear projection\n+ LayerNorm → d=192", facecolor="white", fontsize=8.7)
    arrow(ax, (5.3, mid_y), (5.9, mid_y))

    box(ax, (5.9, mid_y + 0.45), 2.4, 0.7, "+ Fourier position\nfeatures (8 bands)", facecolor="#f5f5f4", edgecolor=MUTED, fontsize=8.3)
    box(ax, (5.9, mid_y - 0.75), 2.4, 0.7, "+ bottlenecked task\nembedding (dim=8)", facecolor="#f5f5f4", edgecolor=MUTED, fontsize=8.3)
    ax.text(7.1, mid_y - 1.15, "(prevents the encoder forking into\ndisjoint per-task sub-networks)", fontsize=7.3, color=MUTED, ha="center", style="italic")

    arrow(ax, (8.3, mid_y), (8.9, mid_y))
    box(ax, (8.9, mid_y - 0.8), 2.5, 1.6, "Shared Transformer\nencoder × 4 layers\n(pre-norm, 6 heads)\n-- ONE backbone", facecolor=color, text_color="white", fontweight="bold", fontsize=9.5)

    # two output heads
    arrow(ax, (11.4, mid_y + 0.4), (11.9, y_top + h / 2))
    arrow(ax, (11.4, mid_y - 0.4), (11.9, y_bot + h / 2))

    box(ax, (11.9, y_top), 1.1, h, "Recon.\nhead\n+skip", facecolor="#eef8f2", edgecolor=FAMILY_COLORS["masked"], fontsize=8.3)
    box(ax, (11.9, y_bot), 1.1, h, "Jigsaw\nhead", facecolor="#fdf3e0", edgecolor=FAMILY_COLORS["jigsaw"], fontsize=8.3)

    side_note(ax, (0.3, 2.85),
               "Masked loss: MSE over only the top 17.5% highest-magnitude bins per spectrum (peak_top_fraction) -- avoids the loss being\n"
               "trivially satisfied by predicting a flat baseline. Jigsaw loss: cross-entropy, same as the standalone jigsaw model.",
               fontsize=8.3)

    # downstream pooling note
    y_pool = 0.5
    box(ax, (0.3, y_pool), 5.3, 1.1,
        "Downstream (LOOCV/CV classifiers): encode_spectrum() pools\nembeddings across all 4 bin sizes (natural order) + optionally\nthe masked-task embedding → concatenated feature vector",
        facecolor="white", fontsize=8.3)
    arrow(ax, (5.6, y_pool + 0.55), (6.3, y_pool + 0.55))
    box(ax, (6.3, y_pool), 2.7, 1.1, "Softmax classifier\nhead (per downstream\ndataset/task)", facecolor="white", fontsize=8.5)

    save(fig, out_path)


def overview_figure(individual_paths: list[Path], out_path: Path):
    """Stack the three saved PNGs into one side-by-side (top-to-bottom) overview panel."""
    import matplotlib.image as mpimg

    imgs = [mpimg.imread(str(p.with_suffix(".png"))) for p in individual_paths]
    heights = [im.shape[0] / im.shape[1] for im in imgs]
    fig, axes = plt.subplots(len(imgs), 1, figsize=(12.5, sum(h * 12.5 for h in heights) + 0.3), dpi=300)
    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, out_path)


def main():
    out_dir = Path("results/plots/model_architectures")
    out_dir.mkdir(parents=True, exist_ok=True)

    masked_path = out_dir / "fig_arch_masked"
    jigsaw_path = out_dir / "fig_arch_jigsaw"
    joint_path = out_dir / "fig_arch_joint"

    masked_architecture_figure(masked_path)
    jigsaw_architecture_figure(jigsaw_path)
    joint_architecture_figure(joint_path)
    overview_figure([masked_path, jigsaw_path, joint_path], out_dir / "fig_arch_overview")


if __name__ == "__main__":
    main()
