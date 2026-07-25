"""Peak-list alignment via Needleman-Wunsch, for independently-shifting peaks.

The realignment in peak_extraction.py (--realign) finds ONE shift per
spectrum via cross-correlation, which assumes every peak in that spectrum
moves together. That's only a coarse approximation -- in practice each
metabolite's chemical shift responds to pH/ionic strength/temperature
independently, so peak A can drift left while peak B (a few thousand points
away) stays put or drifts right.

This module instead treats each spectrum as a *list* of picked peaks (noise
baseline already excluded by a prominence threshold) and aligns that list
against the canonical reference peak list the same way two biological
sequences are aligned: a banded, order-preserving dynamic program that
matches, or skips (gaps), each element independently.

  - "Order-preserving" (peaks essentially never cross each other along the
    spectrum) is what makes this a 1D sequence-alignment problem rather than
    an unconstrained bipartite matching problem -- it's the same assumption
    Needleman-Wunsch/Smith-Waterman make about residue order in DNA/protein
    alignment.
  - A reference peak with no acceptable nearby query peak becomes a gap
    (metabolite not detected in that spectrum) instead of being forced into
    a bad match; likewise an extra query peak with no reference counterpart
    (noise or an uncatalogued metabolite) is skipped.
  - The match score falls off linearly with position distance and is -inf
    beyond `tolerance`, so the alignment can never be tricked into pairing
    two peaks that are obviously not the same metabolite -- it will use a
    gap instead.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def pick_spectrum_peaks(
    segment: np.ndarray,
    seg_lo: int,
    min_prominence_snr: float,
    min_distance: int,
    exclude: np.ndarray | None = None,
    max_peaks: int = 200,
):
    """Peak-pick one spectrum segment, ignoring the noise baseline via a
    robust (MAD-based) prominence threshold -- the same idea a human expert
    uses when scanning a spectrum for "real" peaks vs. noise wiggle.

    `segment` covers absolute point indices [seg_lo, seg_lo + len(segment)).
    Returns (positions, prominences, noise_sd, baseline) with positions/
    prominences sorted by position and capped at `max_peaks` (keeping the
    most prominent) so a very noisy spectrum can't blow up the downstream
    O(R*Q) alignment. noise_sd/baseline are the same robust, whole-segment
    estimate used for the prominence threshold here -- callers should reuse
    them (rather than a narrow local flank) for any later detection-SNR
    check on the matched peaks, since a flank only a few tens of points wide
    is easily contaminated by the peak's own shoulder and will understate
    the true SNR.
    """
    x = np.asarray(segment, dtype=np.float64)
    if exclude is not None:
        seg_exclude = exclude[seg_lo: seg_lo + len(x)]
        valid = x[~seg_exclude] if np.any(~seg_exclude) else x
    else:
        valid = x
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    noise_sd = max(1.4826 * mad, np.finfo(float).eps)

    peaks, props = find_peaks(x, distance=min_distance, prominence=noise_sd * min_prominence_snr)
    prominences = props["prominences"]
    if len(peaks) > max_peaks:
        keep = np.argsort(prominences)[::-1][:max_peaks]
        peaks = peaks[keep]
        prominences = prominences[keep]
        order = np.argsort(peaks)
        peaks, prominences = peaks[order], prominences[order]

    positions = peaks.astype(np.int64) + seg_lo
    return positions, prominences, noise_sd, med


def align_peak_lists(
    ref_positions: np.ndarray,
    query_positions: np.ndarray,
    tolerance: float,
    gap_penalty: float,
    match_bonus: float = 10.0,
):
    """Global (Needleman-Wunsch) alignment of two position-sorted peak lists.

    Returns (matches, score) where `matches` is a dict {ref_index:
    query_index} for every aligned pair (unmatched ref/query peaks are
    simply absent from the dict -- they were "gapped"), and `score` is the
    total alignment score.

    Recurrence per cell (i, j):
        diag = score[i-1, j-1] + match_bonus * (1 - |dpos|/tolerance)   if |dpos| <= tolerance else -inf
        up   = score[i-1, j] + gap_penalty      # ref peak i unmatched (deletion)
        left = score[i, j-1] + gap_penalty      # query peak j unmatched (insertion)
        score[i, j] = max(diag, up, left)

    This is exactly Needleman-Wunsch with a position-distance substitution
    score and a linear (non-affine) gap penalty; the -inf cutoff at
    `tolerance` is what keeps the alignment order-preserving and local in
    effect (like a banded DP) without needing explicit index banding, since
    R and Q are small (tens to a couple hundred) here.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    ref_positions = np.asarray(ref_positions, dtype=np.float64)
    query_positions = np.asarray(query_positions, dtype=np.float64)
    r, q = len(ref_positions), len(query_positions)

    score = np.zeros((r + 1, q + 1), dtype=np.float64)
    ptr = np.zeros((r + 1, q + 1), dtype=np.int8)  # 0=diag(match), 1=up(ref gap), 2=left(query gap)

    for i in range(1, r + 1):
        score[i, 0] = score[i - 1, 0] + gap_penalty
        ptr[i, 0] = 1
    for j in range(1, q + 1):
        score[0, j] = score[0, j - 1] + gap_penalty
        ptr[0, j] = 2

    neg_inf = float("-inf")
    for i in range(1, r + 1):
        rp = ref_positions[i - 1]
        dpos_row = np.abs(rp - query_positions)
        for j in range(1, q + 1):
            dpos = dpos_row[j - 1]
            diag = score[i - 1, j - 1] + match_bonus * (1.0 - dpos / tolerance) if dpos <= tolerance else neg_inf
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            best, move = diag, 0
            if up > best:
                best, move = up, 1
            if left > best:
                best, move = left, 2
            score[i, j] = best
            ptr[i, j] = move

    matches: dict[int, int] = {}
    i, j = r, q
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ptr[i, j] == 0:
            matches[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or ptr[i, j] == 1):
            i -= 1
        else:
            j -= 1
    return matches, float(score[r, q])
