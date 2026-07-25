"""High-resolution visual verification of water/EDTA suppression in the
60000-80000 point-index region (covers both the water window 62500-68000 and
the EDTA search window 72000-74000), for a random sample of spectra.

Intended as a final visual sign-off after running
code/preprocessing/build_clean_datasets.py -- one plot per spectrum so fine
detail is actually visible (a shared multi-panel figure at this resolution
becomes illegible), plus one overview grid for a quick scan.

Example:
    python code/analysis/plot_suppression_verification.py \\
        --data data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v2.npy \\
        --label train_corpus_v2 --n-spectra 20
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REGION_LO, REGION_HI = 60_000, 80_000
WATER_RANGE = (62_500, 68_000)
EDTA_RANGE = (72_000, 74_000)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--label", required=True, help="used in output filenames/titles")
    p.add_argument("--n-spectra", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/analysis/suppression_verification")
    p.add_argument("--compare-data", default=None, help="optional: same dataset's OLD (pre-fix) npy, for a before/after overlay")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, mmap_mode="r")
    n = data.shape[0]
    n_plot = min(args.n_spectra, n)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n, size=n_plot, replace=False))

    compare = np.load(args.compare_data, mmap_mode="r") if args.compare_data else None

    individual_dir = out_dir / f"{args.label}_individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(REGION_LO, REGION_HI)

    for ridx in idx:
        row = np.asarray(data[ridx, REGION_LO:REGION_HI], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(16, 4.5))
        if compare is not None:
            old_row = np.asarray(compare[ridx, REGION_LO:REGION_HI], dtype=np.float64)
            ax.plot(x, old_row, color="#999999", linewidth=0.7, alpha=0.8, label="before fix")
        ax.plot(x, row, color="#2a78d6", linewidth=0.8, label="after fix" if compare is not None else "spectrum")
        ax.axvspan(*WATER_RANGE, color="#e34948", alpha=0.08, label="water window")
        ax.axvspan(*EDTA_RANGE, color="#eda100", alpha=0.12, label="EDTA search window")
        ax.set_title(f"{args.label} -- row {ridx} -- region [{REGION_LO}:{REGION_HI}]", fontsize=10)
        ax.set_xlabel("Point index")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.margins(x=0)
        fig.tight_layout()
        fig.savefig(individual_dir / f"row_{ridx}.png", dpi=170)
        plt.close(fig)

    # Overview grid (lower-res but useful for a quick scan of all n_plot at once)
    cols = 2
    rows = int(np.ceil(n_plot / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 2.6 * rows), squeeze=False)
    for i, ridx in enumerate(idx):
        ax = axes[i // cols, i % cols]
        row = np.asarray(data[ridx, REGION_LO:REGION_HI], dtype=np.float64)
        ax.plot(x, row, color="#2a78d6", linewidth=0.6)
        ax.axvspan(*WATER_RANGE, color="#e34948", alpha=0.08)
        ax.axvspan(*EDTA_RANGE, color="#eda100", alpha=0.12)
        ax.set_title(f"row {ridx}", fontsize=8)
        ax.margins(x=0)
    for j in range(n_plot, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(f"{args.label}: {n_plot} random spectra, region [{REGION_LO}:{REGION_HI}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / f"{args.label}_overview_grid.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {n_plot} individual high-res plots to {individual_dir}/")
    print(f"Wrote overview grid to {out_dir / f'{args.label}_overview_grid.png'}")


if __name__ == "__main__":
    main()
