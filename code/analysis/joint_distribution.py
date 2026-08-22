#!/usr/bin/env python3
"""The JOINT distribution of spectral intensities — the gap #5b identified.

WHY. `peak_saturation.py` established that the MARGINAL distribution of each
individual peak's intensity is saturated: ~360 spectra pin its shape, so 9,670
is ~10x more than needed. That is not enough to generate spectra. Generation
needs the joint distribution, and SS's route (b) makes the reason explicit:
adjacent points cannot be sampled independently or the result is not an NMR
spectrum.

This script measures the three properties of the joint distribution that a
windowed sampler actually needs.

1. ACROSS-SPECTRA CORRELATION vs LAG. For every pair of positions separated by
   lag d, how correlated are their intensities ACROSS the population? This is
   the structure a generator must reproduce, and where it decays sets the
   minimum window overlap: sampling two positions from independent windows
   destroys whatever correlation they had.

   Reported separately from:

2. WITHIN-SPECTRUM AUTOCORRELATION vs LAG. The average autocorrelation of a
   single spectrum. This is lineshape and baseline width — a smoothness scale,
   not a population property. The two are routinely conflated; only (1)
   constrains a generator.

3. PER-WINDOW EFFECTIVE RANK. For candidate window sizes, how many principal
   components does a window's joint distribution need? If a 1024-point window
   needs only ~10 components, each window is cheap to model and there are
   enough spectra (9,670) to estimate it. If it needs hundreds, the window
   model is under-determined and the window must shrink.

4. LONG-RANGE STRUCTURE. Correlation surviving beyond the patch/window size is
   exactly what window-stitching destroys. Quantified here so the cost of
   route (b) is a number rather than an assumption. Metabolite coupling lives
   here: all multiplets of one molecule scale with its concentration, however
   far apart in ppm they sit.

Normalisation is `unit_area`, per docs/PI_outline.md section 7.1: under min-max
the unit is "this spectrum's tallest peak", which is not a usable unit for a
distribution of intensities.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "cn", ROOT / "code/analysis/corpus_normalisation.py")
cn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cn)

CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "WaterFixed_EDTAMagnitude_v4.npy")
PPM_AXIS = "data/mtbls326/MTBLS326_common_ppm_axis.npy"
SIGNAL_LO_PPM, SIGNAL_HI_PPM = 0.5, 9.5   # metabolite region


def bin_spectra(X, n_bins):
    fold = X.shape[1] // n_bins
    return X[:, :n_bins * fold].reshape(X.shape[0], n_bins, fold).mean(axis=2), fold


def lag_profile(corr):
    """Mean |correlation| and mean correlation as a function of lag."""
    n = corr.shape[0]
    absmean = np.zeros(n)
    signed = np.zeros(n)
    for d in range(n):
        diag = np.diagonal(corr, offset=d)
        absmean[d] = np.abs(diag).mean()
        signed[d] = diag.mean()
    return signed, absmean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-spectra", type=int, default=2000)
    ap.add_argument("--n-bins", type=int, default=4096)
    ap.add_argument("--window-sizes", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--normalise", default="unit_area",
                    choices=["none", "rowminmax", "unit_area"])
    ap.add_argument("--n-window-probes", type=int, default=8,
                    help="window positions tiled across the signal region")
    ap.add_argument("--remove-pcs", type=int, nargs="+", default=[0, 1, 2, 5],
                    help="recompute the lag profile after removing this many "
                         "leading principal components")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/analysis/joint_distribution")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm = np.load(ROOT / PPM_AXIS)
    corpus = cn.open_corpus(ROOT / CORPUS, args.normalise, verbose=True)
    n_total, length = corpus.shape
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(n_total, min(args.n_spectra, n_total), replace=False))
    print(f"loading {len(rows)}/{n_total} spectra ...")
    X = np.asarray(corpus[rows], dtype=np.float64)

    pts_per_ppm = length / (ppm[0] - ppm[-1])
    B, fold = bin_spectra(X, args.n_bins)
    ppm_per_bin = fold / pts_per_ppm
    print(f"binned to {args.n_bins} bins ({fold} pts/bin = {ppm_per_bin:.4f} ppm/bin)")

    # ---------- 1 + 2: correlation vs lag ----------
    Z = B - B.mean(0)
    sd = Z.std(0)
    keep = sd > 0
    C = np.corrcoef(Z[:, keep].T)
    signed, absmean = lag_profile(C)
    lags = np.arange(len(signed))
    lag_ppm = lags * ppm_per_bin
    lag_pts = lags * fold

    # within-spectrum autocorrelation, for contrast
    W = B - B.mean(1, keepdims=True)
    W /= (W.std(1, keepdims=True) + 1e-12)
    nfft = 1 << int(np.ceil(np.log2(2 * args.n_bins)))
    F = np.fft.rfft(W, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :args.n_bins]
    ac /= ac[:, :1]
    ac_mean = ac.mean(0)

    prof = pd.DataFrame(dict(lag_bins=lags, lag_points=lag_pts, lag_ppm=lag_ppm,
                             across_spectra_signed=signed,
                             across_spectra_absmean=absmean,
                             within_spectrum_autocorr=ac_mean[:len(lags)]))
    prof.to_csv(out_dir / "correlation_vs_lag.csv", index=False)

    print("\n" + "=" * 78)
    print("  1/2. CORRELATION vs LAG")
    print("=" * 78)
    print(f"  {'lag':>10} {'ppm':>8} {'across-spectra |r|':>19} {'within-spectrum r':>19}")
    for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        if d < len(lags):
            print(f"  {lag_pts[d]:10d} {lag_ppm[d]:8.3f} {absmean[d]:19.3f} "
                  f"{ac_mean[d]:19.3f}")
    thr_rows = []
    for t in (0.5, 0.3, 0.2, 0.1):
        idx = np.argmax(absmean < t) if np.any(absmean < t) else len(absmean) - 1
        thr_rows.append(dict(threshold=t, lag_bins=int(idx), lag_points=int(lag_pts[idx]),
                             lag_ppm=float(lag_ppm[idx])))
        print(f"  across-spectra |r| first drops below {t:.1f} at "
              f"{lag_pts[idx]:6d} points ({lag_ppm[idx]:.3f} ppm)")
    pd.DataFrame(thr_rows).to_csv(out_dir / "correlation_thresholds.csv", index=False)

    # ---------- 2b: is long-range correlation just ONE global factor? ----------
    # Unit-area normalisation removes overall scale but not overall composition:
    # a spectrum whose signal is broadly higher lifts every bin together, which
    # correlates ALL pairs regardless of separation. §19 found PC1 alone holds
    # 51-72% of corpus variance, so before attributing long-range correlation to
    # metabolite coupling we must strip the leading component(s) and re-measure.
    print("\n" + "=" * 78)
    print("  2b. LONG-RANGE CORRELATION AFTER REMOVING LEADING COMPONENTS")
    print("=" * 78)
    Zk = Z[:, keep].copy()
    U, S, Vt = np.linalg.svd(Zk - Zk.mean(0), full_matrices=False)
    var_frac = (S ** 2) / (S ** 2).sum()
    print(f"  variance in PC1: {var_frac[0]:.3f}   PC1-2: {var_frac[:2].sum():.3f}   "
          f"PC1-5: {var_frac[:5].sum():.3f}")
    pc_rows = []
    print(f"\n  {'PCs removed':>12} {'|r| @0.09ppm':>13} {'|r| @1.5ppm':>12} {'|r| @6ppm':>10}")
    lag_of = lambda target: int(np.argmin(np.abs(lag_ppm - target)))
    d_near, d_mid, d_far = lag_of(0.09), lag_of(1.5), lag_of(6.0)
    for k in args.remove_pcs:
        if k == 0:
            am = absmean
        else:
            R = Zk - (U[:, :k] * S[:k]) @ Vt[:k]
            Ck = np.corrcoef(R.T)
            _, am = lag_profile(Ck)
        print(f"  {k:12d} {am[d_near]:13.3f} {am[d_mid]:12.3f} {am[d_far]:10.3f}")
        pc_rows.append(dict(pcs_removed=k, abs_r_at_0p09ppm=float(am[d_near]),
                            abs_r_at_1p5ppm=float(am[d_mid]),
                            abs_r_at_6ppm=float(am[d_far]),
                            mean_abs_r_beyond_1024pts=float(am[lag_pts > 1024].mean())))
    pd.DataFrame(pc_rows).to_csv(out_dir / "lag_profile_after_pc_removal.csv", index=False)

    # ---------- 4: long-range structure ----------
    print("\n" + "=" * 78)
    print("  4. LONG-RANGE STRUCTURE (what window-stitching destroys)")
    print("=" * 78)
    lr = []
    for wsize in args.window_sizes:
        beyond = lag_pts > wsize
        if not np.any(beyond):
            continue
        frac_strong = float((absmean[beyond] > 0.3).mean())
        lr.append(dict(window_points=wsize, mean_abs_r_beyond=float(absmean[beyond].mean()),
                       frac_pairs_abs_r_gt_0p3=frac_strong,
                       max_abs_r_beyond=float(absmean[beyond].max())))
        print(f"  window {wsize:5d} pts: beyond it, mean |r| = {absmean[beyond].mean():.3f}, "
              f"max {absmean[beyond].max():.3f}, "
              f"{100*frac_strong:.1f}% of lags still above 0.3")
    pd.DataFrame(lr).to_csv(out_dir / "long_range_structure.csv", index=False)

    # ---------- 3: per-window effective rank ----------
    print("\n" + "=" * 78)
    print("  3. PER-WINDOW EFFECTIVE RANK (can each window be modelled?)")
    print("=" * 78)
    # Windows are tiled across the SIGNAL region only. The water-suppression
    # window (points ~62500-68000) is set to exactly zero by preprocessing, and
    # the spectrum midpoint falls inside it -- a single centre window there is
    # all zeros, giving a degenerate SVD and a spurious "1 PC" answer.
    rank_rows = []
    sig_lo = int(np.searchsorted(-ppm, -SIGNAL_HI_PPM))
    sig_hi = int(np.searchsorted(-ppm, -SIGNAL_LO_PPM))
    for wsize in args.window_sizes:
        starts = np.linspace(sig_lo, sig_hi - wsize, args.n_window_probes, dtype=int)
        k95s, k99s, skipped = [], [], 0
        for lo in starts:
            Xi = X[:, lo:lo + wsize]
            Zi = Xi - Xi.mean(0)
            tot = float((Zi ** 2).sum())
            if not np.isfinite(tot) or tot <= 0:
                skipped += 1
                continue
            sv = np.linalg.svd(Zi, compute_uv=False)
            frac = np.cumsum(sv ** 2) / (sv ** 2).sum()
            k95s.append(int(np.searchsorted(frac, 0.95) + 1))
            k99s.append(int(np.searchsorted(frac, 0.99) + 1))
        if not k95s:
            print(f"  window {wsize:5d} pts: all probe windows degenerate; skipped")
            continue
        m95, m99 = int(np.median(k95s)), int(np.median(k99s))
        rank_rows.append(dict(window_points=wsize, n_samples=len(rows),
                              n_probe_windows=len(k95s), degenerate_skipped=skipped,
                              pcs_for_95pct_median=m95, pcs_for_95pct_min=int(min(k95s)),
                              pcs_for_95pct_max=int(max(k95s)),
                              pcs_for_99pct_median=m99,
                              samples_per_pc_95=len(rows) / max(m95, 1)))
        print(f"  window {wsize:5d} pts: median {m95:4d} PCs for 95% "
              f"(range {min(k95s)}-{max(k95s)}), {m99:4d} for 99%   "
              f"-> {len(rows)/max(m95,1):.0f} spectra per component"
              + (f"   [{skipped} degenerate window(s) skipped]" if skipped else ""))
    pd.DataFrame(rank_rows).to_csv(out_dir / "window_effective_rank.csv", index=False)

    # ---------- recommendation ----------
    i50 = int(np.argmax(absmean < 0.5)) if np.any(absmean < 0.5) else 0
    rec_overlap = int(lag_pts[i50])
    print("\n" + "=" * 78)
    print("  IMPLICATION FOR THE WINDOWED SAMPLER (route b)")
    print("=" * 78)
    print(f"  Across-spectra correlation falls below 0.5 at ~{rec_overlap} points")
    print(f"  ({lag_ppm[i50]:.3f} ppm), so windows should OVERLAP by at least that much")
    print(f"  or the join will sit inside a still-correlated stretch and show a seam.")
    print(f"\n  Long-range correlation is what stitching cannot preserve. See the table")
    print(f"  above for how much survives beyond each candidate window size.")
    print(f"\nWrote {out_dir}/correlation_vs_lag.csv")
    print(f"Wrote {out_dir}/correlation_thresholds.csv")
    print(f"Wrote {out_dir}/long_range_structure.csv")
    print(f"Wrote {out_dir}/window_effective_rank.csv")


if __name__ == "__main__":
    main()
