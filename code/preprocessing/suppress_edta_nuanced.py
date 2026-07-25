"""Nuanced, per-spectrum EDTA peak detection and suppression.

EDTA is only used as an anticoagulant in some plasma samples, so it must be
detected and suppressed per-spectrum rather than blanket-masked -- unlike
water, which is present (and suppressed) in every spectrum.

This revises suppress_edta_peak.py's conservative detector after validating
against Barth Syndrome samples with known ground truth (the
Anticoagulant column in data/Barth/Workbench_Barth_Syndrome_metadata.csv):
the 4 confirmed-EDTA rows and the 33 confirmed-Heparin (no EDTA) rows have
STATISTICALLY INDISTINGUISHABLE dominance_ratio (peak prominence vs. the
window's 2nd-most-prominent peak): both cluster around 1.1-1.7, far below
suppress_edta_peak.py's MIN_DOMINANCE_RATIO=3.0 gate. That gate compares the
EDTA candidate against "whatever else is in this busy window", which is not
a meaningful comparison -- there's essentially always some other real
metabolite peak nearby. It is why suppress_edta_peak.py's dominance-ratio
gate skipped suppression on all 40 Barth rows regardless of EDTA status.

This version drops the whole-window dominance-ratio gate and instead
requires each candidate peak to be prominent relative to its OWN LOCAL noise
floor only (prominence_snr), which does discriminate reasonably well, and
allows suppressing more than one qualifying peak per row -- EDTA has two
chemically distinct CH2 environments, so a genuine EDTA-positive spectrum can
show two narrow, comparably-tall peaks in the search window, and requiring a
single "dominant" one was structurally the wrong ask.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from suppress_edta_peak import _local_baseline_and_noise  # reuse the validated fill logic

SEARCH_START = 72_000
SEARCH_STOP = 74_000
EDGE_MARGIN = 75
EDGE_POINTS = 200

MIN_PROMINENCE_SNR = 30.0  # validated against Barth's 4 known-EDTA rows (snr range 32.5-186.4)
MIN_SUPPRESSION_WIDTH = 10
MAX_SUPPRESSION_WIDTH = 600
MAX_PEAKS_PER_ROW = 2  # EDTA's two characteristic CH2 resonance environments

TAIL_FRACTION = 0.005
NOISE_MULTIPLIER = 8.0
BOUNDARY_PADDING = 2

FILL_METHOD = "local_noise"
NOISE_SCALE = 0.5
FILL_RANDOM_SEED = 42


def detect_edta_peaks(spectrum: np.ndarray) -> list[dict]:
    """Return 0, 1, or 2 qualifying EDTA-candidate peak detections for one row."""
    window = np.asarray(spectrum[SEARCH_START:SEARCH_STOP])
    edge = np.concatenate((window[:EDGE_POINTS], window[-EDGE_POINTS:]))
    baseline = float(np.median(edge))
    mad = float(np.median(np.abs(edge - baseline)))
    noise_sd = max(1.4826 * mad, np.finfo(float).eps)

    peaks, _ = find_peaks(window)
    if peaks.size == 0:
        return []

    prominences = peak_prominences(window, peaks)[0]
    order = np.argsort(prominences)[::-1]

    detections = []
    for rank in order[: MAX_PEAKS_PER_ROW * 3]:  # scan a few extra candidates, most get rejected
        if len(detections) >= MAX_PEAKS_PER_ROW:
            break
        peak = int(peaks[rank])
        prominence = float(prominences[rank])
        prominence_snr = prominence / noise_sd

        if peak < EDGE_MARGIN or peak >= window.size - EDGE_MARGIN:
            continue
        if prominence_snr < MIN_PROMINENCE_SNR:
            continue

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
        if not (MIN_SUPPRESSION_WIDTH <= width <= MAX_SUPPRESSION_WIDTH):
            continue

        # skip if this peak's interval overlaps one already accepted (avoid double-suppressing the same feature)
        if any(not (right <= d["left_index"] - SEARCH_START or left >= d["right_index"] - SEARCH_START) for d in detections):
            continue

        detections.append({
            "peak_index": SEARCH_START + peak,
            "peak_value": float(window[peak]),
            "baseline": baseline,
            "noise_sd": noise_sd,
            "prominence": prominence,
            "prominence_snr": prominence_snr,
            "left_index": SEARCH_START + left,
            "right_index": SEARCH_START + right,
            "suppression_width": width,
            "cutoff": cutoff,
        })

    return sorted(detections, key=lambda d: d["peak_index"])


def suppress_detections(spectrum: np.ndarray, detections: list[dict], row_index: int = 0) -> np.ndarray:
    output = np.array(spectrum, copy=True)
    for i, detection in enumerate(detections):
        left, right = detection["left_index"], detection["right_index"]
        if FILL_METHOD == "zero":
            output[left:right] = 0.0
            continue
        baseline, noise_sd = _local_baseline_and_noise(output, left, right)
        output[left:right] = baseline
        if FILL_METHOD == "local_noise":
            rng = np.random.default_rng(FILL_RANDOM_SEED + row_index * 10 + i)
            output[left:right] += rng.normal(loc=0.0, scale=NOISE_SCALE * noise_sd, size=right - left)
    return output


def process_dataset(spectra: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Run detection+suppression over every row. Returns (output_array, per-row diagnostics)."""
    n = spectra.shape[0]
    output = np.array(spectra, dtype=np.float64, copy=True)
    rows_out = []
    for i in range(n):
        spectrum = np.asarray(spectra[i], dtype=np.float64)
        detections = detect_edta_peaks(spectrum)
        if detections:
            output[i] = suppress_detections(spectrum, detections, row_index=i)
        rows_out.append({
            "row_index": i,
            "n_peaks_suppressed": len(detections),
            "peak_indices": ";".join(str(d["peak_index"]) for d in detections),
            "prominence_snrs": ";".join(f"{d['prominence_snr']:.2f}" for d in detections),
            "widths": ";".join(str(d["suppression_width"]) for d in detections),
        })
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"EDTA detection rows {i + 1}/{n}")
    return output, rows_out


def write_diagnostics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_index", "n_peaks_suppressed", "peak_indices", "prominence_snrs", "widths"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
