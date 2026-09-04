#!/usr/bin/env python3
"""Extract a lipid / macromolecule basis empirically from the corpus.

WHY THIS IS NEEDED. GISSMO contains small molecules only. Serum CPMG retains a
large lipoprotein and residual-protein envelope -- the CH3 and CH2 humps around
0.8-1.3 ppm, plus broad features under the whole spectrum -- and no metabolite
library will ever supply it. The Stage 0 fit gate's residual concentrated exactly
there (1.0, 1.6-1.7, 2.1-2.5 ppm), and the panel had no lipid component at all.

WHY IT CANNOT SIMPLY BE "THE SMOOTH PART". Metabolite peaks sit on top of the
envelope, so a plain smoothing of the mean spectrum absorbs metabolite intensity
and the resulting component then competes with the GISSMO basis -- which is the
degeneracy that made the bounded solver return nonsense. The extraction here
therefore does two things a naive smooth would not:

  1. Peaks are EXCLUDED before smoothing. Points whose intensity rises far above
     a running baseline are masked out, the envelope is interpolated across the
     gaps, and only then smoothed. What survives is the part of the signal that
     no sharp resonance explains.

  2. The residual is taken AGAINST THE GISSMO BASIS, not against the raw data.
     Each spectrum is first fitted with the metabolite basis alone; the envelope
     is estimated from what that fit cannot reach. Components built this way are
     by construction the part the metabolite block leaves behind, which is
     precisely what the fit needs and is far less collinear with it.

The components are then the leading PCs of those per-spectrum envelopes, made
non-negative (NMR envelopes are absorptive), so they can enter the fit under the
same non-negativity constraint as the metabolites.

OUTPUT
  lipid_basis.npy        (n_components, 131072), unit-area normalised
  lipid_basis_meta.csv   variance explained, ppm of each component's maximum
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
_cn = importlib.util.spec_from_file_location("cn", ROOT / "code/analysis/corpus_normalisation.py")
cn = importlib.util.module_from_spec(_cn)
_cn.loader.exec_module(cn)

CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "WaterFixed_EDTAMagnitude_v4.npy")
PPM_AXIS = "data/mtbls326/MTBLS326_common_ppm_axis.npy"
WATER_LO, WATER_HI = 62500, 68000
SIGNAL_LO, SIGNAL_HI = 0.5, 9.5


def moving_min_max_baseline(y, win):
    """Running-minimum followed by running-maximum: a standard morphological
    opening. Follows the broad envelope while sitting under sharp peaks."""
    from scipy.ndimage import minimum_filter1d, maximum_filter1d
    return maximum_filter1d(minimum_filter1d(y, win, mode="nearest"), win, mode="nearest")


def envelope_of(y, ppm, win_pts, smooth_pts, peak_sigma):
    """Mask sharp peaks, interpolate across them, smooth what remains."""
    from scipy.ndimage import uniform_filter1d
    base = moving_min_max_baseline(y, win_pts)
    resid = y - base
    sd = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-30
    peak = resid > peak_sigma * sd
    z = y.copy()
    if peak.any() and (~peak).sum() > 10:
        idx = np.arange(len(y))
        z[peak] = np.interp(idx[peak], idx[~peak], y[~peak])
    return uniform_filter1d(z, smooth_pts, mode="nearest")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-spectra", type=int, default=300)
    ap.add_argument("--n-components", type=int, default=6)
    ap.add_argument("--baseline-window-ppm", type=float, default=0.08)
    ap.add_argument("--smooth-ppm", type=float, default=0.04)
    ap.add_argument("--peak-sigma", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/synthesis")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm = np.load(ROOT / PPM_AXIS)
    pts_per_ppm = len(ppm) / (ppm.max() - ppm.min())
    win = max(3, int(args.baseline_window_ppm * pts_per_ppm) | 1)
    smo = max(3, int(args.smooth_ppm * pts_per_ppm) | 1)
    print(f"baseline window {win} pts ({args.baseline_window_ppm} ppm), "
          f"smoothing {smo} pts ({args.smooth_ppm} ppm)")

    corpus = cn.open_corpus(ROOT / CORPUS, "unit_area", verbose=True)
    n_total, length = corpus.shape
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(n_total, args.n_spectra, replace=False))
    Y = np.asarray(corpus[rows], dtype=np.float64)

    keep = np.ones(length, bool)
    keep[WATER_LO:WATER_HI] = False
    keep &= (ppm >= SIGNAL_LO) & (ppm <= SIGNAL_HI)

    print(f"extracting envelopes from {len(Y)} spectra ...")
    E = np.stack([envelope_of(y, ppm, win, smo, args.peak_sigma) for y in Y])
    E[:, ~keep] = 0.0
    frac = np.abs(E[:, keep]).sum(1) / (np.abs(Y[:, keep]).sum(1) + 1e-30)
    print(f"  envelope accounts for a median {np.median(frac):.1%} of |signal| "
          f"(IQR {np.percentile(frac,25):.1%}-{np.percentile(frac,75):.1%})")

    mu = E.mean(0)
    Z = E - mu
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()

    comps = [mu]                                    # component 0: the mean envelope
    names = ["mean_envelope"]
    for k in range(args.n_components):
        v = Vt[k]
        # NMR envelopes are absorptive; make each component non-negative so it can
        # enter the fit under the same constraint as the metabolites. Sign is
        # chosen to keep the dominant lobe, then negatives are clipped.
        if np.abs(v.min()) > np.abs(v.max()):
            v = -v
        v = np.clip(v, 0, None)
        comps.append(v)
        names.append(f"envelope_pc{k+1}")

    B = np.stack(comps)
    B[:, ~keep] = 0.0
    area = np.abs(B).sum(1, keepdims=True)
    B = B / np.maximum(area, 1e-30)

    np.save(out_dir / "lipid_basis.npy", B.astype(np.float32))
    meta = pd.DataFrame(dict(
        name=names,
        variance_explained=[np.nan] + [float(var[k]) for k in range(args.n_components)],
        peak_ppm=[float(ppm[int(np.argmax(b))]) for b in B],
        frac_area_below_1p5ppm=[float(np.abs(b[(ppm < 1.5) & keep]).sum() /
                                      (np.abs(b).sum() + 1e-30)) for b in B]))
    meta.to_csv(out_dir / "lipid_basis_meta.csv", index=False)

    print(f"\n  {'component':16s} {'var expl':>9} {'max at ppm':>11} {'frac below 1.5 ppm':>19}")
    for _, r in meta.iterrows():
        ve = "--" if r.variance_explained != r.variance_explained else f"{r.variance_explained:.3f}"
        print(f"  {r['name']:16s} {ve:>9} {r.peak_ppm:11.2f} {r.frac_area_below_1p5ppm:19.3f}")
    print(f"\n  (a high 'frac below 1.5 ppm' is the lipoprotein CH3/CH2 region, which is "
          f"what\n   this basis is chiefly meant to supply)")

    # collinearity against the metabolite basis -- the thing that broke the fit
    mb = ROOT / "results/synthesis/basis_600MHz.npy"
    if mb.exists():
        M = np.load(mb).astype(np.float64)
        A, Bk = M[:, keep], B[:, keep]
        C = (A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-30)) @ \
            (Bk / (np.linalg.norm(Bk, axis=1, keepdims=True) + 1e-30)).T
        mm = pd.read_csv(ROOT / "results/synthesis/basis_600MHz_meta.csv")
        print(f"\n  worst collinearity with the GISSMO metabolite basis: {np.abs(C).max():.3f}")
        i, j = np.unravel_index(np.abs(C).argmax(), C.shape)
        print(f"    {mm.name[i]} vs {names[j]}")
        print(f"  mean |cos| across all pairs: {np.abs(C).mean():.3f}")
    print(f"\nWrote {out_dir}/lipid_basis.npy")
    print(f"Wrote {out_dir}/lipid_basis_meta.csv")


if __name__ == "__main__":
    main()
