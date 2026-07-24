#!/usr/bin/env python3
"""Extract the same set of NMR peaks from every spectrum in a large corpus.

Why not just peak-pick every spectrum independently? Because noise changes
which points look like local maxima, so spectrum A's "peak #7" and spectrum
B's "peak #7" are not guaranteed to be the same metabolite resonance. Instead:

  1. Build a robust reference spectrum: the per-point median across a random
     subsample of spectra. Real metabolite resonances reinforce across
     spectra; noise mostly averages out.
  2. Peak-pick ONCE on that reference (scipy.signal.find_peaks, prominence +
     minimum-distance gated) -> a canonical list of point positions, each
     meant to represent one real metabolite resonance.
  3. For every individual spectrum, search a small window around each
     canonical position for that spectrum's own local maximum -- this
     absorbs the slight per-sample chemical-shift drift that's normal in NMR
     even after a shared 0 ppm (solvent) reference alignment. A local SNR
     gate marks genuinely absent peaks as not-detected (NaN) instead of
     recording a noise value as if it were real signal.

Known suppressed regions (water: points [62500, 68000); EDTA: a
sample-dependent window inside roughly [72000, 74000), see
code/preprocessing/WSZero_62500To68000.py and
code/preprocessing/suppress_edta_peak.py) are excluded from peak picking by
default so they can never be selected as "metabolite" peaks.

Outputs (all written to --output-dir):
  canonical_peaks.csv     One row per selected peak: point index, reference
                          prominence/value, per-spectrum search tolerance.
  peak_values.npy         (n_spectra, n_peaks) float64, NaN where undetected.
  peak_shifts.npy         (n_spectra, n_peaks) int32, matched_index - p.
  peak_detected.npy       (n_spectra, n_peaks) bool.
  alignment_shift_diagnostics.png  Per-peak shift histograms (QC: shifts
                          piling at +-tolerance means the window is too small).
  run_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


DEFAULT_DATA = "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax.npy"
WATER_SUPPRESSION_RANGE = (62_500, 68_000)
EDTA_SEARCH_RANGE = (72_000, 74_000)


def robust_noise_sd(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(1.4826 * mad, np.finfo(float).eps), med


def excluded_mask(length: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for lo, hi in ranges:
        mask[max(0, lo):min(length, hi)] = True
    return mask


def build_reference(data_path: str, reference_n: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    arr = np.load(data_path, mmap_mode="r")
    n, length = arr.shape
    rng = np.random.default_rng(seed)
    take = min(reference_n, n)
    idx = np.sort(rng.choice(n, size=take, replace=False))
    print(f"Building reference spectrum from {take}/{n} randomly sampled spectra...")
    sample = np.asarray(arr[idx], dtype=np.float64)
    reference = np.median(sample, axis=0)
    return reference, idx, n


def pick_canonical_peaks(reference: np.ndarray, exclude: np.ndarray, n_peaks: int, min_distance: int):
    search = reference.copy()
    noise_sd, baseline = robust_noise_sd(search[~exclude])
    search[exclude] = baseline  # flatten excluded regions so find_peaks can never select them

    peaks, props = find_peaks(search, distance=min_distance, prominence=noise_sd * 3.0)
    if len(peaks) < n_peaks:
        print(
            f"Only {len(peaks)} peaks passed prominence>=3*noise_sd; relaxing to "
            f"prominence>=noise_sd to reach --n-peaks={n_peaks}."
        )
        peaks, props = find_peaks(search, distance=min_distance, prominence=noise_sd)
    if len(peaks) < n_peaks:
        raise RuntimeError(
            f"Only found {len(peaks)} candidate peaks; requested {n_peaks}. "
            "Lower --n-peaks or --min-peak-distance."
        )

    prominences = props["prominences"]
    rank = np.argsort(prominences)[::-1][:n_peaks]
    top_idx = peaks[rank]
    top_prom = prominences[rank]

    order = np.argsort(top_idx)
    return top_idx[order], top_prom[order], noise_sd, baseline


def effective_tolerances(peak_indices: np.ndarray, user_tol: int) -> np.ndarray:
    tols = np.full(len(peak_indices), user_tol, dtype=int)
    for i, p in enumerate(peak_indices):
        gaps = []
        if i > 0:
            gaps.append(int(p - peak_indices[i - 1]))
        if i < len(peak_indices) - 1:
            gaps.append(int(peak_indices[i + 1] - p))
        if gaps:
            max_allowed = max(1, min(gaps) // 2 - 1)
            tols[i] = min(user_tol, max_allowed)
    return tols


def estimate_shift_for_spectrum(spectrum_segment: np.ndarray, reference_segment: np.ndarray, max_shift: int):
    """Best-fit integer shift (and a sharpness/confidence ratio) that aligns
    `spectrum_segment` to `reference_segment` via cross-correlation.

    Both arguments must already cover the same absolute point-index range
    (e.g. both arr[seg_lo:seg_hi]) so the returned shift is directly in
    points. Cross-correlating a multi-thousand-point segment (shape match)
    rather than chasing a single tall point is what makes this robust to an
    unrelated peak drifting into a naive small search window -- shared by
    estimate_local_shifts (whole-corpus batch) and the interactive viewer
    (spectra_viewer_app.py), so both use exactly the same alignment logic.
    """
    from scipy.signal import correlate

    a = reference_segment.astype(np.float64)
    a = a - a.mean()
    b = spectrum_segment.astype(np.float64)
    b = b - b.mean()
    full = correlate(b, a, mode="full", method="fft")
    center = len(a) - 1
    lag = np.arange(-max_shift, max_shift + 1)
    window = full[center - max_shift: center + max_shift + 1]
    order = np.argsort(window)[::-1]
    best_lag = int(lag[order[0]])
    far = np.abs(lag - best_lag) > 100
    second = window[far].max() if np.any(far) else np.nan
    sharpness = float(window[order[0]] / second) if second and second > 0 else float("nan")
    return best_lag, sharpness


def estimate_local_shifts(
    data_path: str,
    reference: np.ndarray,
    seg_lo: int,
    seg_hi: int,
    max_shift: int,
    chunk_size: int,
):
    """Per-spectrum coarse local shift via cross-correlation against the
    reference, restricted to one segment of the spectrum.

    Unlike a single-point argmax match, cross-correlating a several-thousand-
    point *segment* finds the shift that best aligns the local shape, so it
    isn't fooled by an unrelated tall peak drifting into a naive search
    window. Used to correct discrete, per-spectrum registration offsets
    (verified beforehand via this same technique) before the small-window
    per-peak match in extract_matched_values.
    """
    arr = np.load(data_path, mmap_mode="r")
    n, length = arr.shape
    ref_seg = reference[seg_lo:seg_hi]

    shifts = np.zeros(n, dtype=np.int64)
    sharpness = np.full(n, np.nan)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = np.asarray(arr[start:stop, seg_lo:seg_hi], dtype=np.float64)
        for i in range(chunk.shape[0]):
            shifts[start + i], sharpness[start + i] = estimate_shift_for_spectrum(chunk[i], ref_seg, max_shift)
        print(f"\rEstimated local realignment shift for {stop}/{n} spectra", end="", flush=True)
    print()
    return shifts, sharpness


def extract_matched_values(
    data_path: str,
    peak_indices: np.ndarray,
    tolerances: np.ndarray,
    exclude: np.ndarray,
    area_halfwidth: int,
    value_mode: str,
    min_snr: float,
    chunk_size: int,
    row_shifts: np.ndarray | None = None,
):
    """For every spectrum and every canonical peak, search a small window
    (offset by that spectrum's coarse row_shifts[i], if given -- see
    estimate_local_shifts) for the local maximum, and gate detection on
    local SNR. `shifts` in the return value is the *residual* shift relative
    to (p + row_shifts[i]) -- i.e. how much of the small window's radius was
    actually used -- which is what the tolerance-adequacy diagnostics care
    about; row_shifts itself carries the coarse correction, if any.
    """
    arr = np.load(data_path, mmap_mode="r")
    n, length = arr.shape
    n_peaks = len(peak_indices)
    values = np.full((n, n_peaks), np.nan, dtype=np.float64)
    shifts = np.zeros((n, n_peaks), dtype=np.int32)
    detected = np.zeros((n, n_peaks), dtype=bool)
    if row_shifts is None:
        row_shifts = np.zeros(n, dtype=np.int64)
    row_shifts = row_shifts.astype(np.int64)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = np.asarray(arr[start:stop], dtype=np.float64)
        row_off = row_shifts[start:stop]

        for j in range(n_peaks):
            p, tol = int(peak_indices[j]), int(tolerances[j])

            win_offsets = np.arange(-tol, tol + 1)
            win_idx = np.clip(p + row_off[:, None] + win_offsets[None, :], 0, length - 1)
            window = np.take_along_axis(chunk, win_idx, axis=1)
            rows = np.arange(window.shape[0])
            local_argmax = np.argmax(window, axis=1)
            matched_idx = win_idx[rows, local_argmax]
            peak_val = window[rows, local_argmax]

            flank_half = 6 * tol
            flank_offsets = np.arange(-flank_half, flank_half + 1)
            flank_idx = np.clip(p + row_off[:, None] + flank_offsets[None, :], 0, length - 1)
            flank = np.take_along_axis(chunk, flank_idx, axis=1)
            flank_excluded = exclude[flank_idx]
            flank_masked = np.where(flank_excluded, np.nan, flank)
            all_excluded = np.all(flank_excluded, axis=1)
            if np.any(all_excluded):
                flank_masked[all_excluded] = flank[all_excluded]
            med = np.nanmedian(flank_masked, axis=1)
            mad = np.nanmedian(np.abs(flank_masked - med[:, None]), axis=1)
            noise_sd = np.maximum(1.4826 * mad, np.finfo(float).eps)

            snr = (peak_val - med) / noise_sd
            is_detected = snr >= min_snr

            if value_mode == "area":
                out_val = np.empty(window.shape[0], dtype=np.float64)
                for k in range(window.shape[0]):
                    center = int(local_argmax[k])
                    a_lo = max(0, center - area_halfwidth)
                    a_hi = min(window.shape[1], center + area_halfwidth + 1)
                    out_val[k] = np.trapz(window[k, a_lo:a_hi] - med[k], dx=1.0)
                out_val = np.clip(out_val, 0, None)
            else:
                out_val = peak_val - med

            row_slice = slice(start, stop)
            values[row_slice, j] = np.where(is_detected, out_val, np.nan)
            shifts[row_slice, j] = matched_idx - (p + row_off)
            detected[row_slice, j] = is_detected

        print(f"\rExtracted peaks for {stop}/{n} spectra", end="", flush=True)
    print()
    return values, shifts, detected


def save_canonical_peaks(path, peak_indices, prominences, tolerances, detected, shifts,
                          points_per_ppm, index_at_zero_ppm):
    # Shifts are only meaningful for rows that actually passed the detection
    # gate -- an "undetected" row's matched_idx is just wherever a noise
    # fluctuation (or the flank of a peak sitting outside the window)
    # happened to be largest, so mixing those in would make this diagnostic
    # meaningless.
    detection_rate = detected.mean(axis=0)
    n_peaks = shifts.shape[1]
    mean_abs_shift = np.full(n_peaks, np.nan)
    frac_at_edge = np.full(n_peaks, np.nan)
    for j in range(n_peaks):
        mask = detected[:, j]
        if np.any(mask):
            abs_shift = np.abs(shifts[mask, j])
            mean_abs_shift[j] = abs_shift.mean()
            frac_at_edge[j] = float(np.mean(abs_shift >= tolerances[j]))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "peak_id", "point_index", "ppm", "reference_prominence", "tolerance_points",
            "detection_rate", "mean_abs_shift", "frac_shift_at_edge",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, p in enumerate(peak_indices):
            ppm = ""
            if points_per_ppm and index_at_zero_ppm is not None:
                ppm = f"{(int(p) - index_at_zero_ppm) / points_per_ppm:.4f}"
            writer.writerow(
                {
                    "peak_id": i,
                    "point_index": int(p),
                    "ppm": ppm,
                    "reference_prominence": float(prominences[i]),
                    "tolerance_points": int(tolerances[i]),
                    "detection_rate": float(detection_rate[i]),
                    "mean_abs_shift": float(mean_abs_shift[i]),
                    "frac_shift_at_edge": float(frac_at_edge[i]),
                }
            )
    return detection_rate, mean_abs_shift, frac_at_edge


def plot_shift_diagnostics(shifts, tolerances, peak_indices, out_path, n_cols=8):
    n_peaks = shifts.shape[1]
    n_rows = int(np.ceil(n_peaks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), squeeze=False)
    for i in range(n_peaks):
        ax = axes[i // n_cols, i % n_cols]
        ax.hist(shifts[:, i], bins=21, color="#2a78d6")
        tol = int(tolerances[i])
        ax.axvline(-tol, color="red", linewidth=0.6, linestyle="--")
        ax.axvline(tol, color="red", linewidth=0.6, linestyle="--")
        ax.set_title(f"#{i} idx={int(peak_indices[i])}", fontsize=6)
        ax.tick_params(labelsize=5)
    for i in range(n_peaks, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")
    fig.suptitle("Per-spectrum match shift vs. reference position (red = search tolerance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-peaks", type=int, default=60, help="Target peak count (recommended 50-70).")
    parser.add_argument("--reference-n", type=int, default=2000,
                         help="Random spectra used to build the median reference spectrum.")
    parser.add_argument("--min-peak-distance", type=int, default=40,
                         help="Minimum point spacing between distinct canonical peaks.")
    parser.add_argument("--tolerance-points", type=int, default=30,
                         help="Max per-spectrum search radius (points) around each canonical peak. "
                              "Auto-shrunk near closely spaced peaks so windows never overlap.")
    parser.add_argument("--area-halfwidth", type=int, default=3,
                         help="Half-width (points) of the small window integrated for --value-mode area.")
    parser.add_argument("--value-mode", choices=["area", "height"], default="area")
    parser.add_argument("--min-snr", type=float, default=3.0,
                         help="Matched value must exceed local baseline by this many robust noise SDs "
                              "to count as detected; otherwise recorded as NaN.")
    parser.add_argument("--exclude-water", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--exclude-edta", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--extra-exclude-ranges", type=int, nargs="*", default=[],
                         help="Extra excluded point ranges as flat lo hi lo hi ... pairs.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--points-per-ppm", type=float, default=None,
                         help="Optional: only used to add a 'ppm' column to canonical_peaks.csv. "
                              "No calibration is stored in this repo, so this is off by default.")
    parser.add_argument("--index-at-zero-ppm", type=int, default=None)
    parser.add_argument("--search-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Restrict peak picking (and matching) to this point-index window, e.g. "
                              "for a region known to be well-behaved when the whole-corpus assumption "
                              "that index == consistent chemical shift doesn't hold everywhere.")
    parser.add_argument("--realign", action="store_true",
                         help="Before the small-window peak match, estimate and correct a per-spectrum "
                              "coarse local shift via cross-correlation against the reference (segment "
                              "given by --realign-segment). Use when a diagnostic has shown a discrete, "
                              "repeatable per-spectrum offset rather than small continuous jitter.")
    parser.add_argument("--realign-segment", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Segment cross-correlated for --realign. Default: --search-range widened "
                              "by --realign-margin on each side.")
    parser.add_argument("--realign-margin", type=int, default=1000)
    parser.add_argument("--realign-max-shift", type=int, default=800,
                         help="Max +/- shift (points) searched during realignment.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.extra_exclude_ranges) % 2 != 0:
        parser.error("--extra-exclude-ranges expects an even number of ints (lo hi pairs).")
    if args.realign and args.search_range is None and args.realign_segment is None:
        parser.error("--realign requires --search-range or an explicit --realign-segment.")
    return args


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference, reference_idx, n_total = build_reference(args.data, args.reference_n, args.seed)
    length = reference.shape[0]

    ranges = []
    if args.exclude_water == "true":
        ranges.append(WATER_SUPPRESSION_RANGE)
    if args.exclude_edta == "true":
        ranges.append(EDTA_SEARCH_RANGE)
    pairs = args.extra_exclude_ranges
    ranges.extend((pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2))
    if args.search_range is not None:
        lo, hi = args.search_range
        if lo < 0 or hi > length or lo >= hi:
            raise ValueError(f"--search-range {args.search_range} is invalid for spectrum length {length}")
        ranges.append((0, lo))
        ranges.append((hi, length))
        print(f"Restricting peak picking to point-index range [{lo}, {hi}).")
    exclude = excluded_mask(length, ranges)
    print(f"Excluded ranges from peak picking: {ranges}")

    peak_indices, prominences, noise_sd, baseline = pick_canonical_peaks(
        reference, exclude, args.n_peaks, args.min_peak_distance
    )
    print(f"Selected {len(peak_indices)} canonical peaks (reference noise_sd={noise_sd:.4g}).")

    tolerances = effective_tolerances(peak_indices, args.tolerance_points)
    if np.any(tolerances < args.tolerance_points):
        n_shrunk = int(np.sum(tolerances < args.tolerance_points))
        print(f"Note: shrank search tolerance for {n_shrunk} peak(s) to avoid overlapping a neighbor.")

    row_shifts = None
    if args.realign:
        if args.realign_segment is not None:
            seg_lo, seg_hi = args.realign_segment
        else:
            search_lo, search_hi = args.search_range
            seg_lo = max(0, search_lo - args.realign_margin)
            seg_hi = min(length, search_hi + args.realign_margin)
        print(f"Estimating per-spectrum coarse realignment shift from segment [{seg_lo}, {seg_hi}) "
              f"(max shift +/-{args.realign_max_shift})...")
        row_shifts, sharpness = estimate_local_shifts(
            args.data, reference, seg_lo, seg_hi, args.realign_max_shift, args.chunk_size
        )
        np.save(out_dir / "spectrum_realignment_shift.npy", row_shifts)
        np.save(out_dir / "spectrum_realignment_sharpness.npy", sharpness)
        print(
            f"Realignment shift: mean={row_shifts.mean():.1f}, std={row_shifts.std():.1f}, "
            f"median sharpness={np.nanmedian(sharpness):.2f}"
        )

    values, shifts, detected = extract_matched_values(
        args.data, peak_indices, tolerances, exclude,
        args.area_halfwidth, args.value_mode, args.min_snr, args.chunk_size,
        row_shifts=row_shifts,
    )

    np.save(out_dir / "peak_values.npy", values)
    np.save(out_dir / "peak_shifts.npy", shifts)
    np.save(out_dir / "peak_detected.npy", detected)
    detection_rate, mean_abs_shift, edge_frac = save_canonical_peaks(
        out_dir / "canonical_peaks.csv", peak_indices, prominences, tolerances, detected, shifts,
        args.points_per_ppm, args.index_at_zero_ppm,
    )
    plot_shift_diagnostics(shifts, tolerances, peak_indices, out_dir / "alignment_shift_diagnostics.png")

    valid_edge = edge_frac[np.isfinite(edge_frac)]
    run_config = vars(args).copy()
    run_config.update(
        {
            "n_spectra": int(n_total),
            "spectrum_length": int(length),
            "reference_sample_size": int(len(reference_idx)),
            "peak_point_indices": [int(p) for p in peak_indices],
            "mean_detection_rate": float(detection_rate.mean()),
            "min_detection_rate": float(detection_rate.min()),
            "peaks_with_edge_clipping_gt_5pct": int(np.sum(edge_frac > 0.05)),
            "realignment_shift_mean": float(row_shifts.mean()) if row_shifts is not None else None,
            "realignment_shift_std": float(row_shifts.std()) if row_shifts is not None else None,
        }
    )
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    print(f"\nDone. {len(peak_indices)} peaks x {n_total} spectra.")
    print(f"Mean detection rate across peaks: {detection_rate.mean():.3f} "
          f"(min {detection_rate.min():.3f}, max {detection_rate.max():.3f})")
    if valid_edge.size and np.any(edge_frac > 0.05):
        flagged = np.where(edge_frac > 0.05)[0]
        print(
            f"Warning: {len(flagged)} peak(s) have >5% of their DETECTED matches sitting at the "
            f"tolerance edge (peak_id {flagged.tolist()}) -- their true chemical-shift drift likely "
            f"exceeds --tolerance-points; consider widening it for these regions."
        )
    print(f"Wrote: {out_dir / 'canonical_peaks.csv'}")
    print(f"Wrote: {out_dir / 'peak_values.npy'} shape={values.shape}")
    print(f"Wrote: {out_dir / 'alignment_shift_diagnostics.png'}")


if __name__ == "__main__":
    main()
