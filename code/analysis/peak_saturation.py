#!/usr/bin/env python3
"""Distribution plots + data-saturation scoring for peaks extracted by
peak_extraction.py.

"Saturation" here means: does the current sample size (N spectra) already
capture the shape of each peak's value distribution well enough that more
data wouldn't meaningfully change it? Two complementary, standard
convergence metrics are used per peak (not invented ad hoc):

1. Held-out KS-distance convergence curve (primary).
   Split the peak's detected values 50/50 into a fixed reference half and a
   probe pool. Draw growing random subsamples from the probe pool (sizes
   log-spaced up to the full probe-pool size) and measure the
   Kolmogorov-Smirnov distance between each subsample's empirical CDF and
   the fixed reference half's empirical CDF. This measures agreement in the
   *whole distribution shape* (not just mean/variance), and the fixed
   reference avoids the leakage you'd get comparing a growing subsample
   against a "full sample" that contains it. Where the curve first drops to
   and stays below --ks-threshold defines the saturation point N*; report
   saturation_ratio = N* / (probe pool size). Small ratio => you had far
   more than enough data to pin down this peak's distribution shape;
   ratio ~= 1 (curve never drops below threshold) => still under-sampled.

2. Bootstrap standard-error-of-the-mean curve (secondary, classic).
   Bootstrap SE of the mean at growing subsample sizes follows the familiar
   SE ~ 1/sqrt(N) law; used to report current relative precision and to
   extrapolate the N needed for a target relative precision (default 5%).

Outputs (all written to --output-dir, default: same dir as --peaks-dir):
  saturation_summary.csv
  distributions_grid.png
  ks_convergence_grid.png
  saturation_ratio_summary.png
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp


def load_peaks(peaks_dir: Path):
    values = np.load(peaks_dir / "peak_values.npy")
    with (peaks_dir / "canonical_peaks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return values, rows


def log_spaced_grid(min_n: int, max_n: int, n_points: int) -> np.ndarray:
    if max_n <= min_n:
        return np.array([max_n])
    grid = np.unique(np.geomspace(min_n, max_n, num=n_points).astype(int))
    grid = grid[grid >= min_n]
    if grid.size == 0 or grid[-1] != max_n:
        grid = np.append(grid, max_n)
    return grid


def ks_convergence_curve(detected_values: np.ndarray, seed: int, min_n: int, n_grid_points: int,
                          n_bootstrap: int, holdout_frac: float):
    rng = np.random.default_rng(seed)
    x = detected_values[np.isfinite(detected_values)]
    n = x.size
    n_holdout = max(int(round(n * holdout_frac)), 1)
    perm = rng.permutation(n)
    reference = x[perm[:n_holdout]]
    probe_pool = x[perm[n_holdout:]]
    probe_n = probe_pool.size

    if probe_n < min_n:
        return None

    grid = log_spaced_grid(min_n, probe_n, n_grid_points)
    mean_ks = np.empty(grid.size)
    for gi, size in enumerate(grid):
        stats = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sub = rng.choice(probe_pool, size=int(size), replace=False)
            stats[b] = ks_2samp(sub, reference).statistic
        mean_ks[gi] = stats.mean()
    return {"grid": grid, "mean_ks": mean_ks, "probe_n": probe_n, "reference_n": n_holdout}


def saturation_point(grid: np.ndarray, mean_ks: np.ndarray, threshold: float):
    below = mean_ks <= threshold
    for i in range(len(grid)):
        if below[i] and np.all(below[i:]):
            return int(grid[i])
    return None


def bootstrap_sem_curve(detected_values: np.ndarray, seed: int, min_n: int, n_grid_points: int, n_bootstrap: int):
    rng = np.random.default_rng(seed + 1)
    x = detected_values[np.isfinite(detected_values)]
    n = x.size
    grid = log_spaced_grid(min_n, n, n_grid_points)
    sem = np.empty(grid.size)
    for gi, size in enumerate(grid):
        means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sub = rng.choice(x, size=int(size), replace=True)
            means[b] = sub.mean()
        sem[gi] = means.std(ddof=1)
    return grid, sem


def analyze_peak(detected_values: np.ndarray, args):
    x = detected_values[np.isfinite(detected_values)]
    n = x.size
    result = {
        "n_detected": n,
        "detection_rate": float(np.mean(np.isfinite(detected_values))),
        "mean": float(np.mean(x)) if n else float("nan"),
        "std": float(np.std(x, ddof=1)) if n > 1 else float("nan"),
        "ks_probe_n": None,
        "ks_reference_n": None,
        "ks_saturation_n": None,
        "ks_saturation_ratio": None,
        "ks_final_mean": None,
        "relative_sem_at_full_n": None,
        "n_needed_for_5pct_sem": None,
        "ks_curve": None,
        "sem_curve": None,
    }
    if n < args.min_subsample_n * 2:
        return result

    ks = ks_convergence_curve(x, args.seed, args.min_subsample_n, args.n_grid_points,
                               args.n_bootstrap, args.holdout_frac)
    if ks is not None:
        n_star = saturation_point(ks["grid"], ks["mean_ks"], args.ks_threshold)
        result.update(
            {
                "ks_probe_n": int(ks["probe_n"]),
                "ks_reference_n": int(ks["reference_n"]),
                "ks_saturation_n": n_star,
                "ks_saturation_ratio": (n_star / ks["probe_n"]) if n_star is not None else None,
                "ks_final_mean": float(ks["mean_ks"][-1]),
                "ks_curve": ks,
            }
        )

    grid, sem = bootstrap_sem_curve(x, args.seed, args.min_subsample_n, args.n_grid_points, args.n_bootstrap)
    rel_sem_full = float(sem[-1] / abs(result["mean"])) if result["mean"] else float("nan")
    n_full = int(grid[-1])
    n_needed = None
    if np.isfinite(rel_sem_full) and rel_sem_full > 0:
        n_needed = int(np.ceil(n_full * (rel_sem_full / args.target_relative_sem) ** 2))
    result.update(
        {
            "relative_sem_at_full_n": rel_sem_full,
            "n_needed_for_5pct_sem": n_needed,
            "sem_curve": (grid, sem),
        }
    )
    return result


def plot_distributions(values: np.ndarray, peak_rows, out_path: Path, log_values: bool, n_cols: int = 8):
    n_peaks = values.shape[1]
    n_rows = int(np.ceil(n_peaks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), squeeze=False)
    for i in range(n_peaks):
        ax = axes[i // n_cols, i % n_cols]
        col = values[:, i]
        col = col[np.isfinite(col)]
        det_rate = values[:, i]
        det_rate = np.mean(np.isfinite(det_rate))
        if col.size:
            plotted = np.log1p(col) if log_values else col
            ax.hist(plotted, bins=30, color="#2a78d6")
        idx = peak_rows[i]["point_index"]
        ax.set_title(f"#{i} idx={idx}\ndet={det_rate:.2f}", fontsize=6)
        ax.tick_params(labelsize=5)
    for i in range(n_peaks, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")
    xlabel = "log1p(peak value)" if log_values else "peak value"
    fig.suptitle(f"Per-peak value distributions across all spectra ({xlabel})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ks_curves(results, out_path: Path, threshold: float, n_cols: int = 8):
    n_peaks = len(results)
    n_rows = int(np.ceil(n_peaks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 1.9 * n_rows), squeeze=False)
    for i, res in enumerate(results):
        ax = axes[i // n_cols, i % n_cols]
        if res["ks_curve"] is not None:
            grid = res["ks_curve"]["grid"]
            mean_ks = res["ks_curve"]["mean_ks"]
            ax.plot(grid, mean_ks, marker="o", markersize=2, linewidth=1, color="#2a78d6")
            ax.axhline(threshold, color="red", linewidth=0.6, linestyle="--")
            ax.set_xscale("log")
            if res["ks_saturation_n"] is not None:
                ax.axvline(res["ks_saturation_n"], color="green", linewidth=0.6, linestyle=":")
        ax.set_title(f"#{i} ratio={res['ks_saturation_ratio']}", fontsize=6)
        ax.tick_params(labelsize=5)
    for i in range(n_peaks, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")
    fig.suptitle("KS distance to held-out reference vs. probe subsample size (log x)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_saturation_summary(results, out_path: Path):
    ratios = [r["ks_saturation_ratio"] for r in results]
    order = np.argsort([1.0 if r is None else r for r in ratios])[::-1]
    fig, ax = plt.subplots(figsize=(max(10, len(results) * 0.28), 5))
    colors = ["#d62728" if (ratios[i] is None) else "#2a9d8f" for i in order]
    heights = [1.0 if ratios[i] is None else ratios[i] for i in order]
    ax.bar(range(len(order)), heights, color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"#{i}" for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("KS saturation ratio (N* / probe-pool size)")
    ax.set_title("Lower = current data comfortably saturates that peak's distribution.\n"
                 "Red bars = never dropped below KS threshold using all available probe data.")
    ax.axhline(1.0, color="black", linewidth=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_summary_csv(results, peak_rows, out_path: Path):
    fields = [
        "peak_id", "point_index", "n_detected", "detection_rate", "mean", "std",
        "ks_probe_n", "ks_reference_n", "ks_saturation_n", "ks_saturation_ratio", "ks_final_mean",
        "relative_sem_at_full_n", "n_needed_for_5pct_sem",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, res in enumerate(results):
            row = {"peak_id": i, "point_index": peak_rows[i]["point_index"]}
            row.update({k: res[k] for k in fields if k not in ("peak_id", "point_index")})
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peaks-dir", required=True, help="Output dir from peak_extraction.py.")
    parser.add_argument("--output-dir", default=None, help="Default: same as --peaks-dir.")
    parser.add_argument("--min-subsample-n", type=int, default=30)
    parser.add_argument("--n-grid-points", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=40)
    parser.add_argument("--holdout-frac", type=float, default=0.5)
    parser.add_argument("--ks-threshold", type=float, default=0.05)
    parser.add_argument("--target-relative-sem", type=float, default=0.05)
    parser.add_argument("--log-values", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    peaks_dir = Path(args.peaks_dir)
    out_dir = Path(args.output_dir) if args.output_dir else peaks_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    values, peak_rows = load_peaks(peaks_dir)
    n_spectra, n_peaks = values.shape
    print(f"Loaded {n_spectra} spectra x {n_peaks} peaks from {peaks_dir}")

    results = []
    for i in range(n_peaks):
        results.append(analyze_peak(values[:, i], args))
        print(f"\rAnalyzed peak {i + 1}/{n_peaks}", end="", flush=True)
    print()

    save_summary_csv(results, peak_rows, out_dir / "saturation_summary.csv")
    plot_distributions(values, peak_rows, out_dir / "distributions_grid.png", args.log_values == "true")
    plot_ks_curves(results, out_dir / "ks_convergence_grid.png", args.ks_threshold)
    plot_saturation_summary(results, out_dir / "saturation_ratio_summary.png")

    ratios = [r["ks_saturation_ratio"] for r in results if r["ks_saturation_ratio"] is not None]
    n_never = sum(1 for r in results if r["ks_saturation_ratio"] is None)
    print(f"\nDone. {len(results)} peaks analyzed.")
    if ratios:
        print(
            f"KS saturation ratio: median={np.median(ratios):.3f}, "
            f"min={np.min(ratios):.3f}, max={np.max(ratios):.3f} "
            f"(lower = more comfortably saturated)."
        )
    print(f"{n_never} peak(s) never dropped below KS threshold {args.ks_threshold} using all available "
          f"probe data -- these are the ones most likely to still benefit from more spectra.")
    print(f"Wrote: {out_dir / 'saturation_summary.csv'}")
    print(f"Wrote: {out_dir / 'distributions_grid.png'}")
    print(f"Wrote: {out_dir / 'ks_convergence_grid.png'}")
    print(f"Wrote: {out_dir / 'saturation_ratio_summary.png'}")


if __name__ == "__main__":
    main()
