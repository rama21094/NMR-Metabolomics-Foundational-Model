"""Validate that the water and EDTA solvent-suppression windows are actually
flat in every row of one or more .npy spectral datasets.

Two checks are run per window, per row:

1. FLATNESS (primary, foolproof): a suppression mask mechanically works by
   overwriting a contiguous range with a single constant value. Real
   biological signal is never perfectly constant over 1500-2000 contiguous
   points, so "is this window's peak-to-peak range ~0" is a direct,
   threshold-free test of "was a hard suppression mask applied here" --
   regardless of what that constant happens to equal (note: after row-min-max
   normalization a raw-zero mask does NOT land exactly on 0.0, since a row's
   raw baseline noise commonly dips slightly negative, shifting the whole
   row's zero-point -- so checking `== 0` under-detects; checking flatness
   does not). A relative epsilon (vs. that row's own global dynamic range)
   keeps this scale-independent across raw and normalized datasets.

2. PEAK-ABOVE-NOISE (secondary, diagnostic): for rows NOT flat-masked, how
   large is the residual relative to that row's own MAD-based robust noise
   floor over the whole row. This distinguishes "no mask applied, and there's
   a large genuine/residual peak sitting here" from "no mask applied, mild
   baseline-level residual". Uses the same prominence-over-noise logic as
   code/analysis/peak_extraction.py's peak-picking.

Example:
    python code/analysis/validate_suppression.py \\
        --dataset "train_corpus:data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax.npy" \\
        --dataset "barth_eval:data/Barth/aligned_128K_Workbench_Barth_Syndrome.npy" \\
        --dataset "mtbls326_eval:data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax.npy" \\
        --dataset "mtbls563_eval:data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_rowMinMax.npy"
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WATER_RANGE = (62_500, 68_000)
EDTA_RANGE = (72_000, 74_000)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", action="append", required=True,
                   help="name:path.npy -- repeatable, one per dataset to check")
    p.add_argument("--water-range", nargs=2, type=int, default=list(WATER_RANGE))
    p.add_argument("--edta-range", nargs=2, type=int, default=list(EDTA_RANGE))
    p.add_argument("--prominence-k", type=float, default=3.0,
                   help="flag a window as containing a real peak if it exceeds median + K*robust_sigma")
    p.add_argument("--flat-rel-eps", type=float, default=1e-6,
                   help="a window is 'hard-masked' if its peak-to-peak range is below this fraction of the row's own dynamic range")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--out-dir", default="results/analysis/suppression_validation")
    return p.parse_args()


def robust_stats(rows):
    """rows: (n, L). Returns per-row median and MAD-based sigma."""
    med = np.median(rows, axis=1, keepdims=True)
    mad = np.median(np.abs(rows - med), axis=1, keepdims=True)
    sigma = 1.4826 * mad
    return med[:, 0], sigma[:, 0]


def check_dataset(name, path, water_range, edta_range, k, flat_rel_eps, chunk_size, out_dir):
    data = np.load(path, mmap_mode="r")
    n, length = data.shape
    wlo, whi = water_range
    elo, ehi = edta_range

    rows_out = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = np.asarray(data[start:end], dtype=np.float64)
        chunk = np.nan_to_num(chunk, nan=0.0)
        med, sigma = robust_stats(chunk)
        sigma_safe = np.where(sigma > 0, sigma, 1e-12)

        row_max = chunk.max(axis=1)
        row_min = chunk.min(axis=1)
        row_ptp = np.maximum(row_max - row_min, 1e-12)
        dynamic_range = np.maximum(row_max - med, 1e-12)

        water_seg = chunk[:, wlo:whi]
        water_max = water_seg.max(axis=1)
        water_ptp = water_seg.max(axis=1) - water_seg.min(axis=1)
        water_argmax = wlo + water_seg.argmax(axis=1)
        water_masked = water_ptp < flat_rel_eps * row_ptp
        water_flag = (~water_masked) & (water_max > med + k * sigma_safe)
        water_ratio = (water_max - med) / dynamic_range

        edta_seg = chunk[:, elo:ehi]
        edta_max = edta_seg.max(axis=1)
        edta_ptp = edta_seg.max(axis=1) - edta_seg.min(axis=1)
        edta_argmax = elo + edta_seg.argmax(axis=1)
        edta_masked = edta_ptp < flat_rel_eps * row_ptp
        edta_flag = (~edta_masked) & (edta_max > med + k * sigma_safe)
        edta_ratio = (edta_max - med) / dynamic_range

        for i in range(end - start):
            rows_out.append({
                "row_index": start + i,
                "row_median": med[i], "row_sigma": sigma[i],
                "row_min": row_min[i], "row_max": row_max[i],
                "water_max": water_max[i], "water_argmax": int(water_argmax[i]),
                "water_masked": bool(water_masked[i]),
                "water_flag": bool(water_flag[i]), "water_ratio_of_dynamic_range": water_ratio[i],
                "edta_max": edta_max[i], "edta_argmax": int(edta_argmax[i]),
                "edta_masked": bool(edta_masked[i]),
                "edta_flag": bool(edta_flag[i]), "edta_ratio_of_dynamic_range": edta_ratio[i],
            })

    df = pd.DataFrame(rows_out)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_suppression_check.csv"
    df.to_csv(csv_path, index=False)

    n_water_masked = int(df["water_masked"].sum())
    n_edta_masked = int(df["edta_masked"].sum())
    n_water_flag = int(df["water_flag"].sum())
    n_edta_flag = int(df["edta_flag"].sum())
    summary = {
        "dataset": name, "path": str(path), "n_spectra": int(n), "length": int(length),
        "water_range": water_range, "edta_range": edta_range, "prominence_k": k, "flat_rel_eps": flat_rel_eps,
        "n_water_hard_masked": n_water_masked, "pct_water_hard_masked": 100.0 * n_water_masked / n,
        "n_water_unmasked_with_real_peak": n_water_flag, "pct_water_unmasked_with_real_peak": 100.0 * n_water_flag / n,
        "n_edta_hard_masked": n_edta_masked, "pct_edta_hard_masked": 100.0 * n_edta_masked / n,
        "n_edta_unmasked_with_real_peak": n_edta_flag, "pct_edta_unmasked_with_real_peak": 100.0 * n_edta_flag / n,
        "water_max_p50": float(df["water_max"].median()), "water_max_p99": float(df["water_max"].quantile(0.99)),
        "water_max_max": float(df["water_max"].max()),
        "edta_max_p50": float(df["edta_max"].median()), "edta_max_p99": float(df["edta_max"].quantile(0.99)),
        "edta_max_max": float(df["edta_max"].max()),
    }

    # Plot: distribution of window max values, flagged rows marked
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, flagcol, title in [
        (axes[0], "water_max", "water_flag", "water window max"),
        (axes[1], "edta_max", "edta_flag", "EDTA window max"),
    ]:
        vals = df[col].values
        ax.hist(vals, bins=60, color="#2a78d6", alpha=0.8)
        flagged = df.loc[df[flagcol], col].values
        if len(flagged):
            for v in flagged:
                ax.axvline(v, color="#e34948", linewidth=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.set_title(f"{name}: {title}\n{df[flagcol].sum()}/{n} rows flagged", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_suppression_histograms.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    if n_water_flag > 0:
        worst = df[df["water_flag"]].sort_values("water_max", ascending=False).head(10)
        print(f"\nTop water-flagged rows for {name}:")
        print(worst[["row_index", "water_max", "water_argmax", "water_ratio_of_dynamic_range"]].to_string(index=False))
    if n_edta_flag > 0:
        worst = df[df["edta_flag"]].sort_values("edta_max", ascending=False).head(10)
        print(f"\nTop EDTA-flagged rows for {name}:")
        print(worst[["row_index", "edta_max", "edta_argmax", "edta_ratio_of_dynamic_range"]].to_string(index=False))
    print()
    return summary


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in args.dataset:
        name, path = spec.split(":", 1)
        print(f"=== {name} ({path}) ===")
        summaries.append(check_dataset(
            name, path, tuple(args.water_range), tuple(args.edta_range),
            args.prominence_k, args.flat_rel_eps, args.chunk_size, out_dir,
        ))

    with open(out_dir / "suppression_validation_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote summary to {out_dir / 'suppression_validation_summary.json'}")


if __name__ == "__main__":
    main()
