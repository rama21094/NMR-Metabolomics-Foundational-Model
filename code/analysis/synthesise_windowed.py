#!/usr/bin/env python3
"""Route (b): synthetic spectra by overlapping-window sampling.

SS's specification: sampling each x-position independently cannot produce an
NMR-like spectrum because adjacent points would be disjointed, so sample small
WINDOWS and continue with OVERLAPPING windows.

WHAT joint_distribution.py FOUND, AND WHY IT CHANGES THE DESIGN
---------------------------------------------------------------
The measured across-spectra correlation does NOT decay within any feasible
window. Mean |r| is 0.95 at 32 points, 0.73 at 1024, and still 0.45 at 6 ppm
(65,536 points) separation. It first falls below 0.5 only at ~2.2 ppm, and never
reaches 0.3 anywhere inside the spectrum. Removing the 5 leading principal
components barely dents it (0.45 -> 0.37 at 6 ppm), so this is not one global
scale factor -- it is genuinely distributed long-range structure, i.e. the
composition covariance SS warned about, where all multiplets of a metabolite
move together.

Consequence: **no overlap setting makes window-stitching lossless.** Independent
windows will always break real correlation. Per-window structure, by contrast, is
very cheap -- a 1024-point window needs a median of 2 principal components for
95% of its variance -- so the local model is easy and the long-range model is the
hard part.

The three methods below therefore differ in how much long-range structure they
try to keep, and are provided so the trade-off is visible rather than assumed.

  independent   A fresh random donor per window, crossfaded. This is the literal
                naive reading, included to SHOW the seams and the loss of
                long-range structure. Not recommended.

  quilted       Donors chosen sequentially: among all donors, take those whose
                values in the overlap region best match what has been generated
                so far, then pick randomly among the best `--top-k`. This is
                Efros-Leung style texture quilting. Local continuity is good;
                long-range structure still drifts, because nothing constrains
                the spectrum as a whole.

  quilted_base  As `quilted`, but a base donor is drawn first and candidates are
                restricted to donors globally similar to it (top `--base-pool`
                by correlation). The base supplies a coherent long-range
                envelope while windows still recombine locally. This is the
                variant that respects the measurement above, and the default.

Values are `unit_area` normalised (docs/PI_outline.md section 7.1). Output is
written un-normalised in that unit; re-normalise downstream as needed.

VALIDATION. This script reports, for the generated set: nearest-neighbour
correlation to the real corpus (is it novel or a near-copy?), the across-spectra
lag-correlation profile against the real profile (did long-range structure
survive?), and effective rank (did diversity increase?). Those are the first two
gates of the acceptance test in docs/PI_outline.md section 8a.
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


def hann_ramp(n):
    """Rising half-Hann, for crossfading two windows over their overlap."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / max(n - 1, 1)))


def synthesise_one(donors, length, window, hop, method, rng, top_k, base_pool):
    overlap = window - hop
    out = np.zeros(length)
    ramp = hann_ramp(overlap) if overlap > 0 else None

    pool = np.arange(len(donors))
    if method == "quilted_base":
        base = int(rng.integers(len(donors)))
        b = donors[base] - donors[base].mean()
        D = donors - donors.mean(1, keepdims=True)
        num = D @ b
        den = np.linalg.norm(D, axis=1) * np.linalg.norm(b) + 1e-12
        sim = num / den
        pool = np.argsort(-sim)[:max(base_pool, top_k + 1)]

    first = int(rng.choice(pool))
    out[:window] = donors[first, :window]
    pos = hop
    while pos < length:
        end = min(pos + window, length)
        w = end - pos
        if overlap > 0 and pos + overlap <= length:
            if method == "independent":
                pick = int(rng.choice(pool))
            else:
                seg = out[pos:pos + overlap]
                cand = donors[np.ix_(pool, np.arange(pos, pos + overlap))]
                sse = ((cand - seg) ** 2).sum(axis=1)
                order = np.argsort(sse)[:max(top_k, 1)]
                pick = int(pool[order[int(rng.integers(len(order)))]])
            new = donors[pick, pos:end].copy()
            k = min(overlap, w)
            out[pos:pos + k] = out[pos:pos + k] * (1 - ramp[:k]) + new[:k] * ramp[:k]
            if w > k:
                out[pos + k:end] = new[k:]
        else:
            pick = int(rng.choice(pool))
            out[pos:end] = donors[pick, pos:end]
        pos += hop
    return out


def lag_profile_binned(X, n_bins, max_lag_bins):
    fold = X.shape[1] // n_bins
    B = X[:, :n_bins * fold].reshape(X.shape[0], n_bins, fold).mean(axis=2)
    Z = B - B.mean(0)
    sd = Z.std(0)
    Z = Z[:, sd > 0]
    C = np.corrcoef(Z.T)
    n = min(C.shape[0], max_lag_bins)
    return np.array([np.abs(np.diagonal(C, offset=d)).mean() for d in range(n)]), fold


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-synth", type=int, default=100,
                    help="correlation and rank statistics are unstable below ~50")
    ap.add_argument("--n-donors", type=int, default=600)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=1024, help="overlap = window - hop")
    ap.add_argument("--method", default="quilted_base",
                    choices=["independent", "quilted", "quilted_base"])
    ap.add_argument("--also-methods", nargs="*",
                    default=["independent", "quilted"],
                    help="additional methods to generate for comparison plots")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--base-pool", type=int, default=60)
    ap.add_argument("--normalise", default="unit_area",
                    choices=["none", "rowminmax", "unit_area"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/analysis/synthetic_windowed")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = cn.open_corpus(ROOT / CORPUS, args.normalise, verbose=True)
    n_total, length = corpus.shape
    rng = np.random.default_rng(args.seed)
    drows = np.sort(rng.choice(n_total, min(args.n_donors, n_total), replace=False))
    print(f"loading {len(drows)} donor spectra ...")
    donors = np.asarray(corpus[drows], dtype=np.float64)

    methods = [args.method] + [m for m in (args.also_methods or []) if m != args.method]
    all_out = {}
    for m in methods:
        print(f"\nsynthesising {args.n_synth} spectra with method={m} "
              f"(window={args.window}, hop={args.hop}, overlap={args.window-args.hop})")
        r = np.random.default_rng(args.seed + 1)
        S = np.stack([synthesise_one(donors, length, args.window, args.hop, m,
                                     r, args.top_k, args.base_pool)
                      for _ in range(args.n_synth)])
        all_out[m] = S
        np.save(out_dir / f"synthetic_{m}.npy", S.astype(np.float32))
        print(f"  wrote synthetic_{m}.npy  shape={S.shape}")

    # ---------------- validation ----------------
    print("\n" + "=" * 78)
    print("  VALIDATION")
    print("=" * 78)
    real_prof, fold = lag_profile_binned(donors, 4096, 4096)
    ppm = np.load(ROOT / PPM_AXIS)
    pts_per_ppm = length / (ppm[0] - ppm[-1])
    ppm_per_bin = fold / pts_per_ppm
    lag_of = lambda t: int(round(t / ppm_per_bin))
    d_near, d_mid, d_far = lag_of(0.09), lag_of(1.5), lag_of(6.0)

    Dn = donors - donors.mean(1, keepdims=True)
    Dn /= np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-12

    rows = []
    print(f"  {'method':<14} {'nn r to real':>13} {'|r|@0.09ppm':>12} {'|r|@1.5ppm':>11} "
          f"{'|r|@6ppm':>9} {'PCs 95%':>8}")
    print(f"  {'REAL corpus':<14} {'--':>13} {real_prof[d_near]:12.3f} "
          f"{real_prof[d_mid]:11.3f} {real_prof[d_far]:9.3f} ", end="")
    # Effective rank is capped at n-1, so the real corpus must be subsampled to
    # the SAME n as the synthetic set or the comparison is meaningless.
    def pcs95(M):
        Z = M - M.mean(0)
        sv = np.linalg.svd(Z, compute_uv=False)
        f = np.cumsum(sv ** 2) / (sv ** 2).sum()
        return int(np.searchsorted(f, 0.95) + 1)

    rsub = np.random.default_rng(args.seed + 99).choice(
        len(donors), min(args.n_synth, len(donors)), replace=False)
    real_k95 = pcs95(donors[rsub])
    print(f"{real_k95:8d}")
    for m, S in all_out.items():
        Sn = S - S.mean(1, keepdims=True)
        Sn /= np.linalg.norm(Sn, axis=1, keepdims=True) + 1e-12
        nn = (Sn @ Dn.T).max(1)
        prof, _ = lag_profile_binned(S, 4096, 4096) if len(S) > 2 else (np.full(4096, np.nan), fold)
        k95 = pcs95(S)
        print(f"  {m:<14} {np.median(nn):13.3f} {prof[d_near]:12.3f} {prof[d_mid]:11.3f} "
              f"{prof[d_far]:9.3f} {k95:8d}")
        rows.append(dict(method=m, n=len(S), median_nn_r_to_real=float(np.median(nn)),
                         max_nn_r_to_real=float(nn.max()),
                         abs_r_at_0p09ppm=float(prof[d_near]),
                         abs_r_at_1p5ppm=float(prof[d_mid]),
                         abs_r_at_6ppm=float(prof[d_far]), pcs_for_95pct=k95))
    rows.append(dict(method="REAL", n=len(donors), median_nn_r_to_real=np.nan,
                     max_nn_r_to_real=np.nan,
                     abs_r_at_0p09ppm=float(real_prof[d_near]),
                     abs_r_at_1p5ppm=float(real_prof[d_mid]),
                     abs_r_at_6ppm=float(real_prof[d_far]),
                     pcs_for_95pct=real_k95))
    pd.DataFrame(rows).to_csv(out_dir / "validation.csv", index=False)
    np.save(out_dir / "donor_rows.npy", drows)
    print(f"\n  nn r to real: near 1.0 means the generator is copying, not generating.")
    print(f"  |r| columns: how much across-spectra correlation structure survived.")
    print(f"\nWrote {out_dir}/validation.csv")


if __name__ == "__main__":
    main()
