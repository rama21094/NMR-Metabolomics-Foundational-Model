"""Conservatively suppress the dominant EDTA peak in aligned plasma spectra.

The EDTA peak shifts between spectra, so this script detects its position and
boundaries independently for every row. It skips ambiguous rows rather than
risk suppressing a neighboring metabolite peak.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.signal import find_peaks, peak_prominences


# =========================
# IDE-friendly configuration
# =========================
INPUT_FILE = "./data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero.npy"
OUTPUT_FILE = "./data/BrC_T2D/BC_T2D_EDTASuppressed_WS625to680Zero.npy"
DIAGNOSTICS_FILE = "./data/BrC_T2D/edta_suppression_diagnostics.csv"

# Broad region known to contain EDTA. This whole region is never zeroed.
SEARCH_START = 72_000
SEARCH_STOP = 74_000

# Detection safeguards. Ambiguous rows are skipped and reported.
EDGE_POINTS = 200
EDGE_MARGIN = 75
MIN_PROMINENCE_SNR = 50.0
MIN_DOMINANCE_RATIO = 3.0
MIN_SUPPRESSION_WIDTH = 10
MAX_SUPPRESSION_WIDTH = 600

# Stop at whichever is higher: 0.5% of EDTA prominence or 8 local noise SDs.
TAIL_FRACTION = 0.005
NOISE_MULTIPLIER = 8.0
BOUNDARY_PADDING = 2

# Replacement options:
# - "local_noise": robust local baseline plus realistic synthetic noise
# - "local_baseline": smooth robust baseline without noise
# - "boundary_interpolate": line between immediate peak-boundary values
# - "zero": exact zeros
FILL_METHOD = "local_noise"
REPLACEMENT_FLANK_WIDTH = 500
FLANK_OUTLIER_SIGMA = 3.0
NOISE_SCALE = 0.5
BASELINE_LOWERING_NOISE_SD = 0.5
FILL_RANDOM_SEED = 42

# Keep True for the first run. It detects, reports, and plots without writing
# the large output array. Set False after reviewing the diagnostics.
DRY_RUN = False
NUM_COMPARISON_PLOTS = 6
RANDOM_SEED = 42
PLOT_FILE = "edta_suppression_preview.png"


def detect_edta_peak(spectrum: np.ndarray) -> dict:
    """Return conservative EDTA boundaries and detection diagnostics."""
    window = np.asarray(spectrum[SEARCH_START:SEARCH_STOP])
    edge = np.concatenate((window[:EDGE_POINTS], window[-EDGE_POINTS:]))
    baseline = float(np.median(edge))
    mad = float(np.median(np.abs(edge - baseline)))
    noise_sd = max(1.4826 * mad, np.finfo(float).eps)

    peaks, _ = find_peaks(window)
    if peaks.size == 0:
        return {"status": "skipped_no_peak"}

    prominences = peak_prominences(window, peaks)[0]
    order = np.argsort(prominences)[::-1]
    peak = int(peaks[order[0]])
    prominence = float(prominences[order[0]])
    second_prominence = float(prominences[order[1]]) if order.size > 1 else 0.0
    prominence_snr = prominence / noise_sd
    dominance_ratio = prominence / max(second_prominence, noise_sd)

    result = {
        "peak_index": SEARCH_START + peak,
        "peak_value": float(window[peak]),
        "baseline": baseline,
        "noise_sd": noise_sd,
        "prominence": prominence,
        "prominence_snr": prominence_snr,
        "dominance_ratio": dominance_ratio,
    }

    if peak < EDGE_MARGIN or peak >= window.size - EDGE_MARGIN:
        return {**result, "status": "skipped_near_search_edge"}
    if prominence_snr < MIN_PROMINENCE_SNR:
        return {**result, "status": "skipped_low_snr"}
    if dominance_ratio < MIN_DOMINANCE_RATIO:
        return {**result, "status": "skipped_not_dominant"}

    cutoff = baseline + max(TAIL_FRACTION * prominence, NOISE_MULTIPLIER * noise_sd)
    left = peak
    while left > 0 and window[left] > cutoff:
        left -= 1
    right = peak
    while right < window.size - 1 and window[right] > cutoff:
        right += 1

    left = max(0, left - BOUNDARY_PADDING)
    right = min(window.size, right + BOUNDARY_PADDING + 1)
    width = right - left
    result.update(
        {
            "left_index": SEARCH_START + left,
            "right_index": SEARCH_START + right,
            "suppression_width": width,
            "cutoff": cutoff,
        }
    )

    if width < MIN_SUPPRESSION_WIDTH:
        return {**result, "status": "skipped_too_narrow"}
    if width > MAX_SUPPRESSION_WIDTH:
        return {**result, "status": "skipped_too_wide"}

    return {**result, "status": "suppressed"}


def _robust_flank_center_and_residuals(flank: np.ndarray) -> tuple[float, np.ndarray]:
    """Estimate a flank baseline while excluding nearby spectral peaks."""
    center = float(np.median(flank))
    mad = float(np.median(np.abs(flank - center)))
    noise_sd = max(1.4826 * mad, np.finfo(float).eps)
    clean = flank[np.abs(flank - center) <= FLANK_OUTLIER_SIGMA * noise_sd]
    if clean.size == 0:
        clean = flank
    center = float(np.median(clean))
    return center, clean - center


def _local_baseline_and_noise(
    spectrum: np.ndarray,
    left: int,
    right: int,
) -> tuple[np.ndarray, float]:
    """Build a robust baseline line and estimate noise from both peak flanks."""
    left_flank = spectrum[max(0, left - REPLACEMENT_FLANK_WIDTH):left]
    right_flank = spectrum[right:min(spectrum.size, right + REPLACEMENT_FLANK_WIDTH)]
    if left_flank.size == 0 or right_flank.size == 0:
        raise ValueError("Replacement interval requires non-empty flanks")

    left_center, left_residuals = _robust_flank_center_and_residuals(left_flank)
    right_center, right_residuals = _robust_flank_center_and_residuals(right_flank)
    residuals = np.concatenate((left_residuals, right_residuals))
    noise_sd = max(
        1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))),
        np.finfo(float).eps,
    )
    baseline = np.linspace(left_center, right_center, right - left)
    baseline -= BASELINE_LOWERING_NOISE_SD * noise_sd
    return baseline, noise_sd


def suppress_detected_peak(spectrum: np.ndarray, detection: dict) -> np.ndarray:
    """Return a copy with the detected interval replaced using FILL_METHOD."""
    output = np.array(spectrum, copy=True)
    left = int(detection["left_index"])
    right = int(detection["right_index"])

    if FILL_METHOD == "zero":
        output[left:right] = 0.0
    elif FILL_METHOD == "boundary_interpolate":
        x = np.arange(left, right)
        output[left:right] = np.interp(
            x,
            [left - 1, right],
            [output[left - 1], output[right]],
        )
    elif FILL_METHOD in {"local_baseline", "local_noise"}:
        baseline, noise_sd = _local_baseline_and_noise(output, left, right)
        output[left:right] = baseline
        if FILL_METHOD == "local_noise":
            row_index = int(detection.get("row_index", 0))
            rng = np.random.default_rng(FILL_RANDOM_SEED + row_index)
            output[left:right] += rng.normal(
                loc=0.0,
                scale=NOISE_SCALE * noise_sd,
                size=right - left,
            )
    else:
        raise ValueError(
            "FILL_METHOD must be 'local_noise', 'local_baseline', "
            "'boundary_interpolate', or 'zero'"
        )

    return output


def write_diagnostics(rows: list[dict]) -> None:
    fields = [
        "row_index",
        "status",
        "peak_index",
        "peak_value",
        "baseline",
        "noise_sd",
        "prominence",
        "prominence_snr",
        "dominance_ratio",
        "left_index",
        "right_index",
        "suppression_width",
        "cutoff",
    ]
    with open(DIAGNOSTICS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_comparisons(spectra: np.ndarray, detections: list[dict]) -> None:
    successful = [d for d in detections if d["status"] == "suppressed"]
    if not successful or NUM_COMPARISON_PLOTS <= 0:
        return

    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(
        len(successful),
        size=min(NUM_COMPARISON_PLOTS, len(successful)),
        replace=False,
    )
    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 3 * len(selected)), squeeze=False)
    x = np.arange(SEARCH_START, SEARCH_STOP)

    for ax, selected_index in zip(axes.flat, selected):
        detection = successful[int(selected_index)]
        row_index = int(detection["row_index"])
        original = np.asarray(spectra[row_index])
        suppressed = suppress_detected_peak(original, detection)
        ax.plot(x, original[SEARCH_START:SEARCH_STOP], label="Original", lw=0.8)
        ax.plot(x, suppressed[SEARCH_START:SEARCH_STOP], label="Suppressed", lw=0.8)
        ax.axvspan(
            detection["left_index"],
            detection["right_index"],
            color="tab:red",
            alpha=0.15,
        )
        ax.set_title(
            f"Row {row_index}: peak={detection['peak_index']}, "
            f"width={detection['suppression_width']}"
        )
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.2)
        ax.legend()

    axes[-1, 0].set_xlabel("Spectrum point index")
    fig.tight_layout()
    if PLOT_FILE:
        fig.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
        print(f"Saved preview plot: {PLOT_FILE}")
    plt.show()


def main() -> None:
    spectra = np.load(INPUT_FILE, mmap_mode="r")
    if spectra.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {spectra.shape}")
    if not (0 <= SEARCH_START < SEARCH_STOP <= spectra.shape[1]):
        raise ValueError(f"Invalid search window [{SEARCH_START}:{SEARCH_STOP}]")
    if FILL_METHOD == "interpolate" and (SEARCH_START == 0 or SEARCH_STOP == spectra.shape[1]):
        raise ValueError("Interpolation requires points outside the search window")

    detections = []
    output = None
    if not DRY_RUN:
        output = open_memmap(
            OUTPUT_FILE,
            mode="w+",
            dtype=spectra.dtype,
            shape=spectra.shape,
        )

    for row_index in range(spectra.shape[0]):
        spectrum = np.asarray(spectra[row_index])
        detection = detect_edta_peak(spectrum)
        detection["row_index"] = row_index
        detections.append(detection)

        if output is not None:
            if detection["status"] == "suppressed":
                output[row_index] = suppress_detected_peak(spectrum, detection)
            else:
                output[row_index] = spectrum

        if (row_index + 1) % 500 == 0:
            print(f"Processed {row_index + 1}/{spectra.shape[0]} rows")

    if output is not None:
        output.flush()
        print(f"Saved suppressed spectra: {OUTPUT_FILE}")

    write_diagnostics(detections)
    statuses, counts = np.unique([d["status"] for d in detections], return_counts=True)
    print(f"Input shape: {spectra.shape}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'WRITE OUTPUT'}")
    for status, count in zip(statuses, counts):
        print(f"{status}: {count}")
    print(f"Saved diagnostics: {DIAGNOSTICS_FILE}")
    plot_comparisons(spectra, detections)


if __name__ == "__main__":
    main()
