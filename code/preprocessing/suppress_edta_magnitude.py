"""Magnitude-based EDTA/dominant-peak suppression -- metadata-free, uniform
across all datasets.

Earlier detectors (suppress_edta_peak.py's dominance-ratio gate, and this
project's first revision suppress_edta_nuanced.py's local-prominence-SNR
gate) both asked the wrong question: "is there a peak here that's tall
relative to noise/other peaks *within this narrow window*?" That fails in
both directions -- confirmed on real data:
  - it flags real, modest J-coupling multiplets (e.g. a quartet at ~10% of
    the row's own peak scale) that are clearly not the problem, because they
    still clear a LOCAL significance bar;
  - it misses genuinely dominant peaks whenever something else in the same
    narrow window happens to be comparably tall.

The actual problem this needs to solve (per direct instruction) is
normalization corruption: a peak in the EDTA region whose magnitude is wildly
different from the rest of the spectrum's real peaks becomes the new "1.0"
after row-min-max normalization, crushing every other real peak. A peak
that's merely comparable to other real peaks in the row is not a problem,
regardless of whether it's chemically EDTA or not -- so this compares the
EDTA-window candidate directly against the REST OF THE SAME ROW, not against
itself.

Calibration (see conversation record / results/analysis/preprocessing_v2):
on a 2000-row sample of the training corpus, the ratio (EDTA-window peak
height / row's own max height elsewhere) has median 0.12, p95 0.43, and a
real heavy tail up to 7.7. A threshold of 0.5 sits just above the normal
population and cleanly catches the outlier tail. Verified on known cases:
Barth row 5's real quartet scores 0.10 (correctly left alone) vs. training
corpus row 8290's dominant peak scoring 1.37 (correctly suppressed).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from suppress_edta_peak import _local_baseline_and_noise  # reuse validated fill logic

WATER_RANGE = (62_500, 68_000)
SEARCH_START, SEARCH_STOP = 72_000, 74_000
EDGE_MARGIN = 75

RATIO_THRESHOLD = 0.5  # candidate peak height / row's own max height elsewhere
MAX_PEAKS_PER_ROW = 3  # loop guard; stop once no remaining candidate clears the bar

TAIL_FRACTION = 0.005
NOISE_MULTIPLIER = 8.0
BOUNDARY_PADDING = 2
MIN_SUPPRESSION_WIDTH = 10
MAX_SUPPRESSION_WIDTH = 600

FILL_METHOD = "local_noise"
NOISE_SCALE = 0.5
FILL_RANDOM_SEED = 42


def _row_reference_max(spectrum: np.ndarray) -> float:
    """Row's own max height, excluding the water and EDTA windows themselves."""
    ref = spectrum.copy()
    ref[WATER_RANGE[0]:WATER_RANGE[1]] = 0.0
    ref[SEARCH_START:SEARCH_STOP] = 0.0
    return float(ref.max())


def detect_dominant_peaks(spectrum: np.ndarray) -> list[dict]:
    """Return detections for peaks in the EDTA window whose height is a large
    fraction of the row's own peak scale elsewhere -- i.e. would otherwise
    dominate row-min-max normalization."""
    spectrum = np.asarray(spectrum, dtype=np.float64)
    ref_max = _row_reference_max(spectrum)
    if ref_max <= 0:
        return []

    working = spectrum.copy()
    detections = []
    for _ in range(MAX_PEAKS_PER_ROW):
        window = working[SEARCH_START:SEARCH_STOP]
        edge = np.concatenate((window[:200], window[-200:]))
        baseline = float(np.median(edge))
        mad = float(np.median(np.abs(edge - baseline)))
        noise_sd = max(1.4826 * mad, np.finfo(float).eps)

        peaks, _ = find_peaks(window)
        if peaks.size == 0:
            break
        prominences = peak_prominences(window, peaks)[0]
        order = np.argsort(prominences)[::-1]
        peak = int(peaks[order[0]])
        prominence = float(prominences[order[0]])
        peak_value = float(window[peak])
        ratio = peak_value / ref_max

        if peak < EDGE_MARGIN or peak >= window.size - EDGE_MARGIN:
            break
        if ratio < RATIO_THRESHOLD:
            break  # tallest remaining candidate isn't dominant -> nothing more to do

        cutoff = baseline + max(TAIL_FRACTION * prominence, NOISE_MULTIPLIER * noise_sd)
        # Guard: if edge noise is itself elevated, the noise-floor cutoff can exceed
        # the peak's own height, so the boundary search below never advances and a
        # confirmed-dominant peak (ratio >= RATIO_THRESHOLD) silently goes unsuppressed.
        # Cap the cutoff at the midpoint between baseline and peak so the peak always
        # clears its own boundary cutoff.
        cutoff = min(cutoff, baseline + 0.5 * max(peak_value - baseline, 0.0))
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
            break

        detections.append({
            "peak_index": SEARCH_START + peak,
            "peak_value": peak_value,
            "ratio_to_row_max": ratio,
            "row_reference_max": ref_max,
            "baseline": baseline,
            "noise_sd": noise_sd,
            "prominence": prominence,
            "left_index": SEARCH_START + left,
            "right_index": SEARCH_START + right,
            "suppression_width": width,
        })
        # flatten this peak in the working copy so the next iteration can find the next-tallest
        working[SEARCH_START + left:SEARCH_START + right] = baseline

    return detections


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
    n = spectra.shape[0]
    output = np.array(spectra, dtype=np.float64, copy=True)
    rows_out = []
    for i in range(n):
        spectrum = np.asarray(spectra[i], dtype=np.float64)
        detections = detect_dominant_peaks(spectrum)
        if detections:
            output[i] = suppress_detections(spectrum, detections, row_index=i)
        rows_out.append({
            "row_index": i,
            "n_peaks_suppressed": len(detections),
            "peak_indices": ";".join(str(d["peak_index"]) for d in detections),
            "ratios": ";".join(f"{d['ratio_to_row_max']:.3f}" for d in detections),
        })
        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"  EDTA (magnitude-based) rows {i + 1}/{n}")
    return output, rows_out


def write_diagnostics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_index", "n_peaks_suppressed", "peak_indices", "ratios"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
