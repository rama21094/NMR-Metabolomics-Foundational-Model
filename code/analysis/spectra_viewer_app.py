#!/usr/bin/env python3
"""Interactive viewer for visually verifying NMR peak extraction/alignment.

Pick a handful of spectra out of a large .npy corpus, zoom into a region,
toggle per-spectrum realignment (the same cross-correlation logic used by
peak_extraction.py --realign), and overlay/stack them to check by eye
whether metabolite peaks line up -- a visual sanity check alongside the
quantitative saturation analysis.

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

    seg_lo = max(0, lo - int(realign_margin))
    seg_hi = min(length, hi + int(realign_margin))
    ref_seg = reference[seg_lo:seg_hi]

    segments, shifts, sharpness = {}, {}, {}
    with st.spinner(f"Loading {len(rows)} spectra and estimating alignment..."):
        for r in rows:
            seg = np.asarray(arr[r, seg_lo:seg_hi], dtype=np.float64)
            segments[r] = seg
            if do_realign:
                shifts[r], sharpness[r] = estimate_shift_for_spectrum(seg, ref_seg, int(max_shift))

    peaks_df = load_canonical_peaks(peaks_dir) if peaks_dir else None
    peak_positions = []
    if peaks_df is not None:
        if show_peak_markers:
            peak_positions = [int(p) for p in peaks_df["point_index"] if lo <= int(p) <= hi]
    elif peaks_dir:
        st.sidebar.warning(f"No canonical_peaks.csv found in {peaks_dir!r}.")

    n_panels = 2 if do_realign else 1
    titles = ["Raw (as stored)"] + (["Realigned (cross-correlation corrected)"] if do_realign else [])
    fig = make_subplots(rows=n_panels, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=titles)

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

    for p in peak_positions:
        fig.add_vline(x=p, line=dict(color="red", width=1, dash="dash"), row=1, col=1)
        if do_realign:
            fig.add_vline(x=p, line=dict(color="red", width=1, dash="dash"), row=2, col=1)

    fig.update_layout(height=380 * n_panels + 100, hovermode="x unified", legend=dict(itemsizing="constant"))
    fig.update_xaxes(title_text="Point index", row=n_panels, col=1)
    fig.update_yaxes(title_text="Intensity" + (" (offset)" if waterfall else ""))
    st.plotly_chart(fig, use_container_width=True)

    if peak_positions:
        st.caption(
            f"Red dashed lines: {len(peak_positions)} canonical peak position(s) from {peaks_dir}. "
            "On the realigned panel, traces should land close to these; on the raw panel, "
            "the gap between a trace's true peak and the red line is that spectrum's drift."
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


if __name__ == "__main__":
    main()
