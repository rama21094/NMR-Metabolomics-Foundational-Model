#!/usr/bin/env python3
"""Interactive viewer for visually verifying NMR peak extraction/alignment.

Pick a handful of spectra out of a large .npy corpus, zoom into a region,
and compare up to three alignment views side by side (zoom-synced):
  - Raw, as stored.
  - Realigned: one global cross-correlation shift per spectrum
    (peak_extraction.py --realign) -- assumes every peak in a spectrum
    drifts together.
  - NW peak-list aligned: each spectrum is independently peak-picked and
    Needleman-Wunsch-aligned against the canonical peak list
    (peak_list_alignment.py), so each peak's own shift is applied only in
    its own local neighborhood -- no single-shift assumption.
This is a visual sanity check alongside the quantitative saturation
analysis (peak_saturation.py).

Run with:
    streamlit run code/analysis/spectra_viewer_app.py
Then open the printed local URL in a browser (add --server.port/--server.address
if you need to reach it from off-machine, e.g. via an SSH tunnel).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from peak_extraction import DEFAULT_DATA, estimate_shift_for_spectrum  # noqa: E402
from peak_list_alignment import align_peak_lists, pick_spectrum_peaks  # noqa: E402

st.set_page_config(page_title="NMR Spectra Alignment Viewer", layout="wide")

TRACE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@st.cache_resource(show_spinner=False)
def open_memmap(path: str):
    return np.load(path, mmap_mode="r")


@st.cache_data(show_spinner="Building reference spectrum (median across a random subsample)...")
def build_reference_cached(path: str, reference_n: int, seed: int) -> np.ndarray:
    arr = open_memmap(path)
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    take = min(reference_n, n)
    idx = np.sort(rng.choice(n, size=take, replace=False))
    sample = np.asarray(arr[idx], dtype=np.float64)
    return np.median(sample, axis=0)


def load_canonical_peaks(peaks_dir: str):
    path = Path(peaks_dir) / "canonical_peaks.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def slice_for_plot(seg: np.ndarray, seg_lo: int, lo: int, hi: int, shift: int = 0):
    """Return (x, y) for the [lo, hi) reference-frame window, reading `seg`
    (which spans absolute indices [seg_lo, seg_lo+len(seg))) at `shift` points
    off from its nominal position -- i.e. what that spectrum's own trace
    looks like once shifted back into the reference's coordinate frame.
    """
    a = max(0, lo + shift - seg_lo)
    b = min(len(seg), hi + shift - seg_lo)
    if b <= a:
        return np.array([]), np.array([])
    x = np.arange(a - shift + seg_lo, b - shift + seg_lo)
    return x, seg[a:b]


def nw_piecewise_shift_trace(
    seg: np.ndarray, seg_lo: int, lo: int, hi: int,
    canonical_positions: np.ndarray, peak_shift: dict, radii: np.ndarray,
):
    """Build a display trace where each canonical peak's own NW-matched
    shift is applied only in a local neighborhood around that peak (radius
    from `radii`, one per canonical peak); everywhere else shows the raw
    (unshifted) trace. This is what lets the plot show several peaks
    correcting by different, independent amounts within the same spectrum.
    """
    x = np.arange(lo, hi)
    shift_arr = np.zeros(len(x), dtype=np.int64)
    if len(canonical_positions):
        nearest_idx = np.clip(np.searchsorted(canonical_positions, x), 0, len(canonical_positions) - 1)
        left_idx = np.clip(nearest_idx - 1, 0, len(canonical_positions) - 1)
        dist_right = np.abs(canonical_positions[nearest_idx] - x)
        dist_left = np.abs(canonical_positions[left_idx] - x)
        use_left = dist_left < dist_right
        chosen_idx = np.where(use_left, left_idx, nearest_idx)
        chosen_dist = np.where(use_left, dist_left, dist_right)
        for i, pos in enumerate(canonical_positions):
            if pos in peak_shift:
                mask = (chosen_idx == i) & (chosen_dist <= radii[i])
                shift_arr[mask] = peak_shift[pos]
    idx = np.clip(x + shift_arr - seg_lo, 0, len(seg) - 1)
    return x, seg[idx]


def peak_radii(positions: np.ndarray, cap: int) -> np.ndarray:
    """Half the gap to each peak's nearest neighbor (capped), so adjacent
    peaks' locally-shifted regions in nw_piecewise_shift_trace never overlap."""
    radii = np.full(len(positions), cap, dtype=np.int64)
    for i, p in enumerate(positions):
        gaps = []
        if i > 0:
            gaps.append(int(p - positions[i - 1]))
        if i < len(positions) - 1:
            gaps.append(int(positions[i + 1] - p))
        if gaps:
            radii[i] = min(cap, max(1, min(gaps) // 2))
    return radii


def main():
    st.title("NMR Spectra Alignment Viewer")
    st.caption(
        "Pick a few spectra, zoom into a region, and check by eye whether the "
        "peak-matching / realignment pipeline (peak_extraction.py) is doing the right thing."
    )

    with st.sidebar:
        st.header("Data")
        data_path = st.text_input("Spectra .npy path", value=DEFAULT_DATA)
        try:
            arr = open_memmap(data_path)
        except Exception as exc:
            st.error(f"Could not open {data_path}: {exc}")
            st.stop()
        n_spectra, length = arr.shape
        st.caption(f"{n_spectra} spectra x {length} points")

        st.header("Region")
        lo, hi = st.slider(
            "Point-index range to display", min_value=0, max_value=int(length),
            value=(75000, 80000), step=100,
        )

        st.header("Spectra to show")
        n_random = st.number_input("Random spectra", min_value=0, max_value=50, value=8)
        rand_seed = st.number_input("Random selection seed", min_value=0, value=1, step=1)
        manual = st.text_input("Extra row indices (comma-separated)", value="")

        st.header("Reference (median) spectrum")
        reference_n = st.number_input("Reference sample size", min_value=100, max_value=5000, value=1000, step=100)
        show_reference = st.checkbox("Overlay reference spectrum", value=True)

        st.header("Realignment")
        do_realign = st.checkbox("Show realigned panel", value=True)
        max_shift = st.number_input("Max realignment shift (points)", min_value=50, max_value=5000, value=800, step=50)
        realign_margin = st.number_input(
            "Realignment segment margin (points beyond the displayed region)",
            min_value=0, max_value=5000, value=1000, step=100,
        )

        st.header("Canonical peaks (optional)")
        peaks_dir = st.text_input("peak_extraction.py output dir", value="")
        show_peak_markers = st.checkbox("Show canonical peak markers", value=True)

        st.header("NW peak-list alignment")
        st.caption("Requires the canonical peaks dir above (needs the reference peak positions).")
        do_nw = st.checkbox("Show NW-aligned panel", value=False)
        nw_tolerance = st.number_input("NW tolerance (points)", min_value=10, max_value=5000, value=300, step=50)
        nw_gap_penalty = st.number_input("NW gap penalty", min_value=-50.0, max_value=0.0, value=-3.0, step=0.5)
        nw_min_peak_distance = st.number_input("NW min peak distance (points)", min_value=1, value=10, step=1)
        nw_min_prominence_snr = st.number_input("NW min prominence (x noise SD)", min_value=0.5, value=2.0, step=0.5)
        nw_margin = st.number_input("NW peak-picking margin (points)", min_value=0, value=1000, step=100)

        st.header("Display")
        mode = st.radio("Layout", ["Overlay (shared baseline)", "Waterfall (offset stack)"], index=0)
        waterfall = mode.startswith("Waterfall")
        offset_step = (
            st.number_input("Waterfall offset step", min_value=0.0, value=0.15, step=0.05) if waterfall else 0.0
        )

    rng = np.random.default_rng(int(rand_seed))
    random_rows = (
        np.sort(rng.choice(n_spectra, size=int(n_random), replace=False)) if n_random > 0 else np.array([], dtype=int)
    )
    manual_rows = []
    if manual.strip():
        try:
            manual_rows = [int(x.strip()) for x in manual.split(",") if x.strip()]
        except ValueError:
            st.warning("Could not parse manual row indices; ignoring.")
            manual_rows = []
    rows = sorted(set(random_rows.tolist()) | set(manual_rows))
    if not rows:
        st.info("Select at least one spectrum: increase 'Random spectra' or enter manual row indices.")
        st.stop()
    if any(r < 0 or r >= n_spectra for r in rows):
        st.error(f"Row indices must be within [0, {n_spectra}).")
        st.stop()
    if len(rows) > 20:
        st.warning(f"{len(rows)} spectra selected -- the plot may get busy above ~15-20 traces.")

    reference = build_reference_cached(data_path, int(reference_n), 42)

    peaks_df = load_canonical_peaks(peaks_dir) if peaks_dir else None
    if peaks_df is None and peaks_dir:
        st.sidebar.warning(f"No canonical_peaks.csv found in {peaks_dir!r}.")
    canonical_positions = (
        np.sort(peaks_df["point_index"].to_numpy(dtype=np.int64)) if peaks_df is not None else np.array([], dtype=np.int64)
    )
    peak_positions = [int(p) for p in canonical_positions if lo <= int(p) <= hi] if show_peak_markers else []

    if do_nw and peaks_df is None:
        st.sidebar.warning("NW panel needs a valid peaks-dir; disabling it for this render.")
        do_nw = False

    seg_lo = max(0, lo - int(realign_margin))
    seg_hi = min(length, hi + int(realign_margin))
    ref_seg = reference[seg_lo:seg_hi]

    nw_seg_lo, nw_seg_hi, radii = seg_lo, seg_hi, np.array([], dtype=np.int64)
    if do_nw:
        nw_seg_lo = max(0, int(canonical_positions.min()) - int(nw_margin))
        nw_seg_hi = min(length, int(canonical_positions.max()) + int(nw_margin) + 1)
        radii = peak_radii(canonical_positions, cap=int(nw_tolerance))

    segments, shifts, sharpness = {}, {}, {}
    nw_segments, nw_matches, nw_n_query, nw_n_matched = {}, {}, {}, {}
    with st.spinner(f"Loading {len(rows)} spectra and estimating alignment..."):
        for r in rows:
            seg = np.asarray(arr[r, seg_lo:seg_hi], dtype=np.float64)
            segments[r] = seg
            if do_realign:
                shifts[r], sharpness[r] = estimate_shift_for_spectrum(seg, ref_seg, int(max_shift))
            if do_nw:
                nw_seg = np.asarray(arr[r, nw_seg_lo:nw_seg_hi], dtype=np.float64)
                nw_segments[r] = nw_seg
                query_pos, _query_prom, _noise_sd, _med = pick_spectrum_peaks(
                    nw_seg, nw_seg_lo, float(nw_min_prominence_snr), int(nw_min_peak_distance), max_peaks=300,
                )
                nw_n_query[r] = len(query_pos)
                matches, _score = align_peak_lists(
                    canonical_positions, query_pos, float(nw_tolerance), float(nw_gap_penalty)
                )
                nw_matches[r] = {int(canonical_positions[i]): int(query_pos[j]) for i, j in matches.items()}
                nw_n_matched[r] = len(matches)

    n_panels = 1 + int(do_realign) + int(do_nw)
    titles = (
        ["Raw (as stored)"]
        + (["Realigned (global cross-correlation shift)"] if do_realign else [])
        + (["NW peak-list aligned (independent per-peak shift)"] if do_nw else [])
    )
    fig = make_subplots(rows=n_panels, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=titles)
    nw_row = 1 + int(do_realign) + 1

    for i, r in enumerate(rows):
        color = TRACE_COLORS[i % len(TRACE_COLORS)]
        x, y = slice_for_plot(segments[r], seg_lo, lo, hi, shift=0)
        offset = i * offset_step if waterfall else 0.0
        fig.add_trace(
            go.Scatter(x=x, y=y + offset, mode="lines", name=f"row {r}",
                       line=dict(color=color, width=1), legendgroup=f"row{r}"),
            row=1, col=1,
        )
        if do_realign:
            xr, yr = slice_for_plot(segments[r], seg_lo, lo, hi, shift=shifts[r])
            fig.add_trace(
                go.Scatter(x=xr, y=yr + offset, mode="lines", name=f"row {r} (shift={shifts[r]:+d})",
                           line=dict(color=color, width=1), legendgroup=f"row{r}", showlegend=False),
                row=2, col=1,
            )
        if do_nw:
            xn, yn = nw_piecewise_shift_trace(
                nw_segments[r], nw_seg_lo, lo, hi, canonical_positions, nw_matches[r], radii
            )
            fig.add_trace(
                go.Scatter(x=xn, y=yn + offset, mode="lines",
                           name=f"row {r} ({nw_n_matched[r]}/{len(canonical_positions)} matched)",
                           line=dict(color=color, width=1), legendgroup=f"row{r}", showlegend=False),
                row=nw_row, col=1,
            )

    if show_reference:
        xref, yref = slice_for_plot(ref_seg, seg_lo, lo, hi, shift=0)
        fig.add_trace(
            go.Scatter(x=xref, y=yref, mode="lines", name="reference (median)",
                       line=dict(color="black", width=2, dash="dot")),
            row=1, col=1,
        )
        if do_realign:
            fig.add_trace(
                go.Scatter(x=xref, y=yref, mode="lines", name="reference (median)",
                           line=dict(color="black", width=2, dash="dot"), showlegend=False),
                row=2, col=1,
            )
        if do_nw:
            fig.add_trace(
                go.Scatter(x=xref, y=yref, mode="lines", name="reference (median)",
                           line=dict(color="black", width=2, dash="dot"), showlegend=False),
                row=nw_row, col=1,
            )

    for p in peak_positions:
        fig.add_vline(x=p, line=dict(color="red", width=1, dash="dash"), row=1, col=1)
        if do_realign:
            fig.add_vline(x=p, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
        if do_nw:
            fig.add_vline(x=p, line=dict(color="red", width=1, dash="dash"), row=nw_row, col=1)

    fig.update_layout(height=380 * n_panels + 100, hovermode="x unified", legend=dict(itemsizing="constant"))
    fig.update_xaxes(title_text="Point index", row=n_panels, col=1)
    fig.update_yaxes(title_text="Intensity" + (" (offset)" if waterfall else ""))
    st.plotly_chart(fig, use_container_width=True)

    if peak_positions:
        st.caption(
            f"Red dashed lines: {len(peak_positions)} canonical peak position(s) from {peaks_dir}. "
            "On the raw panel, the gap between a trace's true peak and the red line is that "
            "spectrum's drift for that peak. The realigned panel corrects one shift for the whole "
            "spectrum; the NW panel corrects each peak independently -- compare them where a "
            "spectrum's peaks don't all land on the markers by the same amount."
        )

    if do_realign:
        st.subheader("Estimated per-spectrum realignment shift")
        st.caption(
            "sharpness = best cross-correlation score / next-best score more than 100 points away. "
            "Close to 1 means the fit is ambiguous (low confidence); several-fold higher means a "
            "clear, confident match."
        )
        shift_df = pd.DataFrame(
            {
                "row": rows,
                "shift": [shifts[r] for r in rows],
                "sharpness": [round(sharpness[r], 2) if np.isfinite(sharpness[r]) else None for r in rows],
            }
        )
        st.dataframe(shift_df, hide_index=True, use_container_width=False)

    if do_nw:
        st.subheader("NW peak-list alignment diagnostics")
        st.caption(
            "n_query_peaks: how many peaks this spectrum's own peak-picking found in the NW region. "
            "n_matched: how many of the canonical peaks got a valid (within-tolerance) match -- the "
            "rest were gapped (no acceptable candidate nearby, not forced into a bad match)."
        )
        nw_df = pd.DataFrame(
            {
                "row": rows,
                "n_query_peaks": [nw_n_query[r] for r in rows],
                "n_matched": [nw_n_matched[r] for r in rows],
                "n_canonical": [len(canonical_positions)] * len(rows),
            }
        )
        st.dataframe(nw_df, hide_index=True, use_container_width=False)


if __name__ == "__main__":
    main()
