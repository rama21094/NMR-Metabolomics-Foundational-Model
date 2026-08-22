#!/usr/bin/env python3
"""Which per-spectrum normaliser should this pipeline use?

BACKGROUND. §11's batch audit found that metabolite-free spectral regions
predict the label better AFTER per-row min-max than before (MTBLS326
0.641 -> 0.726; BrC-T2D diabetes 0.551 -> 0.640). Per-row min-max is

    (x - x.min()) / (x.max() - x.min())          code/preprocessing/row_minmax_normalize.py

so the zero point is that spectrum's noise-floor minimum and the unit is
whichever peak happens to be tallest. Noise then reads as a fraction of peak
height -- an SNR feature, and SNR is an acquisition property (scan count,
receiver gain, probe tuning). That is a plausible route for batch to leak.

THE COUNTER-ARGUMENT, WHICH IS SOUND. NMR intensities are relative, so
per-spectrum normalisation is necessary, and bounding to [0,1] is convenient for
training a network. This script does not dispute either point. It asks a narrower
question: among per-spectrum normalisers, does the choice change how much
non-biological signal is available to a classifier?

WHAT THIS SCRIPT COMPARES
    none        pre-normalisation EDTA-suppressed array, as a floor
    rowminmax   the current pipeline
    unit_area   x / sum|x|; the simplest comparability normaliser
    pqn         probabilistic quotient normalisation (Dieterle 2006), the
                metabolomics standard for dilution-type effects

Reference-peak normalisation (divide by the standard at 0 ppm) is NOT included
because it is not implementable on this data -- see the diagnostics printed by
--diagnose: the 0 ppm window holds a median 0.7% of each spectrum's global
maximum, i.e. there is no usable reference resonance in this corpus.

TWO TESTS PER (NORMALISER, COHORT)
    1. signal-free classification -- 32 abs-area bins drawn ONLY from regions
       with no metabolite resonances (>9.5 or <-0.5 ppm), with a label
       permutation null. Lower is better: it means less non-biological signal.
    2. full-spectrum classical reference -- 1024 abs-area bins, the §3 pipeline.
       This is the number that must NOT degrade, or we have traded a leak for a
       worse baseline.

NOTE ON SCALE. Test 1 and 2 both standardise features before fitting, so a
global multiplicative constant is invisible to them. That is deliberate: it
separates the SHAPE of a normaliser (what this script measures) from its numeric
RANGE (a training-stability question, reported separately by --diagnose).
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

# reuse the cohort label/row/cv definitions so they cannot drift apart
_spec = importlib.util.spec_from_file_location(
    "bca", ROOT / "code/analysis/batch_confound_audit.py")
bca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bca)

# pre-normalisation arrays: the EDTA-suppressed stage that row_minmax_normalize
# consumes (see code/preprocessing/build_clean_datasets.py)
PRENORM = {
    "Barth": "data/Barth/aligned_128K_Workbench_Barth_Syndrome_"
             "WS625to680Zero_EDTASuppressed_v4.npy",
    "MTBLS326": "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_EDTASuppressed_v4.npy",
    "MTBLS563": "data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_EDTASuppressed_v4.npy",
    "BrC-T2D cancer": "data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_EDTASuppressed_v4.npy",
    "BrC-T2D diabetes": "data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_EDTASuppressed_v4.npy",
}


def norm_none(X):
    return X


def norm_rowminmax(X):
    lo = X.min(1, keepdims=True)
    hi = X.max(1, keepdims=True)
    return (X - lo) / np.maximum(hi - lo, 1e-12)


def norm_unit_area(X):
    return X / np.maximum(np.abs(X).sum(1, keepdims=True), 1e-12)


def norm_pqn(X, ref_quantile=0.75):
    """Probabilistic quotient normalisation (Dieterle et al. 2006).

    Integral-normalise, build a median reference spectrum, then divide each
    spectrum by the median of its quotients against that reference. Quotients
    are taken only over bins where the reference is comfortably above zero;
    including noise bins makes the median quotient meaningless.
    """
    A = norm_unit_area(X)
    ref = np.median(A, axis=0)
    thr = np.quantile(ref[ref > 0], ref_quantile) if np.any(ref > 0) else 0.0
    use = ref > thr
    if use.sum() < 50:
        use = ref > 0
    q = A[:, use] / np.maximum(ref[use][None, :], 1e-12)
    med = np.median(q, axis=1, keepdims=True)
    return A / np.maximum(med, 1e-12)


NORMALISERS = {"none": norm_none, "rowminmax": norm_rowminmax,
               "unit_area": norm_unit_area, "pqn": norm_pqn}


def diagnose(ppm):
    """Facts about the data that constrain which normalisers are even possible."""
    print("=" * 78)
    print("  DIAGNOSTICS -- what the data permits")
    print("=" * 78)
    X = np.load(ROOT / "data/combined/combine_unique_MetaboLights_Workbench_"
                "WaterFixed_EDTAMagnitude_v4.npy", mmap_mode="r")
    S = np.asarray(X[:400], dtype=np.float64)
    i0 = int(np.argmin(np.abs(ppm - 0.0)))
    w = slice(i0 - 300, i0 + 300)
    loc, glob = S[:, w].max(1), S.max(1)
    print(f"\n  1. Is there a 0 ppm reference resonance to normalise to?")
    print(f"     median (0 ppm window max / global max) = {np.median(loc/glob):.4f}")
    print(f"     spectra where it exceeds 5% of global max: {(loc/glob>0.05).mean()*100:.1f}%")
    print(f"     -> NO usable reference peak. Reference normalisation is not")
    print(f"        implementable on this corpus.")

    am = S.argmax(1)
    u, c = np.unique(np.round(ppm[am], 2), return_counts=True)
    o = np.argsort(-c)
    print(f"\n  2. What does rowMinMax actually divide by? (argmax position)")
    for k in o[:6]:
        print(f"     ppm {u[k]:7.2f} : {c[k]:4d}/400 spectra")
    print(f"     {len(u)} distinct argmax positions across 400 spectra.")
    off = c[[k for k in o if u[k] < 0.5]].sum()
    print(f"     {off}/400 ({100*off/400:.0f}%) have their tallest point below 0.5 ppm,")
    print(f"     where no metabolite resonates -- for those, the unit is an artifact.")

    print(f"\n  3. Numeric range after each normaliser + ONE global scale factor")
    print(f"     (the global factor is identical for every spectrum, so it cannot")
    print(f"      carry per-sample information, but it fixes the training range)")
    for name, fn in NORMALISERS.items():
        Z = fn(S.copy())
        g = np.percentile(Z, 99.9)
        Zs = Z / g if g > 0 else Z
        print(f"     {name:<10} raw [{Z.min():9.2e}, {Z.max():9.2e}]   "
              f"after global scale: [{Zs.min():6.2f}, {Zs.max():6.2f}]  "
              f"frac>1: {(Zs>1).mean()*100:.3f}%")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bins-noise", type=int, default=32)
    ap.add_argument("--n-bins-full", type=int, default=1024)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diagnose", action="store_true", default=True)
    ap.add_argument("--out-dir", default="results/analysis/normaliser_comparison")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm = np.load(ROOT / bca.PPM)
    noise_mask = (ppm > bca.NOISE_HI_PPM) | (ppm < bca.NOISE_LO_PPM)
    if args.diagnose:
        diagnose(ppm)

    rows = []
    for make in bca.COHORTS:
        c = make()
        name, y, cv = c["name"], c["y"], c["cv"]
        raw = np.load(ROOT / PRENORM[name], mmap_mode="r")[c["rows"]]
        raw = np.asarray(raw, dtype=np.float64)
        print("=" * 78)
        print(f"  {name}   n={len(y)}   cv={'LOOCV' if cv=='loo' else str(cv)+'-fold'}")
        print("=" * 78)
        for nname, fn in NORMALISERS.items():
            Z = fn(raw.copy())
            Fn = bca.binned_abs_area(Z, noise_mask, args.n_bins_noise)
            acc_n = bca.cv_bal_acc(Fn, y, cv, args.seed)
            null = bca.perm_null(Fn, y, cv, args.n_perm, args.seed)
            p_n = float((null >= acc_n).mean())
            Ff = bca.binned_abs_area(Z, np.ones_like(noise_mask), args.n_bins_full)
            acc_f = bca.cv_bal_acc(Ff, y, cv, args.seed)
            flag = "  <-- LEAK" if p_n < 0.05 else ""
            print(f"    {nname:<10} signal-free {acc_n:.4f} (p={p_n:.3f})   "
                  f"full-spectrum {acc_f:.4f}{flag}")
            rows.append(dict(cohort=name, normaliser=nname,
                             signal_free_acc=acc_n, signal_free_p=p_n,
                             null_p95=float(np.percentile(null, 95)),
                             full_spectrum_acc=acc_f, n=len(y)))
        print()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "normaliser_comparison.csv", index=False)

    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    piv_n = df.pivot(index="cohort", columns="normaliser", values="signal_free_acc")
    piv_f = df.pivot(index="cohort", columns="normaliser", values="full_spectrum_acc")
    cols = [c for c in ["none", "rowminmax", "unit_area", "pqn"] if c in piv_n.columns]
    print("\n  Signal-free accuracy (LOWER IS BETTER -- less non-biological signal)")
    print(piv_n[cols].round(3).to_string())
    print("\n  Full-spectrum classical accuracy (MUST NOT DEGRADE)")
    print(piv_f[cols].round(3).to_string())
    print("\n  Leaks (signal-free p < 0.05), by normaliser:")
    for nname in cols:
        g = df[df.normaliser == nname]
        bad = g[g.signal_free_p < 0.05]
        print(f"    {nname:<10} {len(bad)}/{len(g)} cohorts"
              + (f"   ({', '.join(bad.cohort)})" if len(bad) else ""))
    print(f"\n  Mean full-spectrum accuracy: "
          + "  ".join(f"{n}={piv_f[n].mean():.3f}" for n in cols))
    print(f"\nWrote {out_dir}/normaliser_comparison.csv")


if __name__ == "__main__":
    main()
