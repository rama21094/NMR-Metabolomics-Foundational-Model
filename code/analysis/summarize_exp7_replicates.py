#!/usr/bin/env python3
"""Experiment #7 follow-up: how big is run-to-run variance, and does peak
weighting survive a matched-corpus comparison?

Two questions, two groups of arms.

**Q1 — is the -0.069 v3-vs-v4 baseline gap (§5f) the corpus or noise?**
Three independent runs of the identical configuration on the identical v4
corpus: `exp7_D_baseline_v4` (unseeded), `exp7_D_v4_seed101`, `exp7_D_v4_seed202`.
Their spread is the run-to-run variance of a single arm. Compare that spread to
the distance from the v3 reference (`ps1024_nhead4_true`). If the three v4 runs
cluster well below v3, the corpus is the cause; if they straddle it, the gap was
noise and single-run comparisons in this document need error bars.

**Q2 — does peak weighting help, judged without a corpus confound?**
`exp7_v3_pk025_r1/_r2` are peak-weighted runs on the *v3* corpus, so the
reference is the v3 baseline every earlier number in the doc used. Arm B's
+0.011 was measured against a v4 baseline and is not a clean read.

Both r1 and r2 were launched with `--seed 101` and did NOT come out identical
(max|dW| = 5.3e-2). `cudnn.benchmark = True` autotunes conv/GEMM algorithms per
shape and AMP rescales dynamically, so kernel selection and reduction order vary
between processes no matter what the RNGs do. Seeding removes RNG variance only.
The pair therefore measures pure implementation nondeterminism -- a floor on how
reproducible any single number in this document can be.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HELD_OUT = ["barth", "mtbls326", "brc_t2d_cancer"]
SELECTION = ["mtbls563", "brc_t2d_diabetes"]
ALL5 = HELD_OUT + SELECTION

V4_REPLICATES = ["exp7_D_baseline_v4", "exp7_D_v4_seed101", "exp7_D_v4_seed202"]
V3_PEAK = ["exp7_v3_pk025_r1", "exp7_v3_pk025_r2"]
V3_REF = "ps1024_nhead4_true"
CLASSICAL = {"barth": 0.705, "mtbls326": 1.000, "mtbls563": 0.721,
             "brc_t2d_cancer": 0.937, "brc_t2d_diabetes": 0.829}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/analysis/exp7_replicates/patch_size_results.csv")
    ap.add_argument("--pooling", default="flatten")
    ap.add_argument("--out-dir", default="results/analysis/exp7_replicates")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    sub = df[df.pooling == args.pooling]

    def series(arm):
        return sub[sub.arm == arm].set_index("dataset")["balanced_accuracy"]

    def hm(arm):  # held-out mean
        return float(series(arm).reindex(HELD_OUT).mean())

    def sm(arm):  # selection mean
        return float(series(arm).reindex(SELECTION).mean())

    print(f"\n{'=' * 84}\n  Q1. Run-to-run variance of ONE arm (v4 ps1024 baseline), pooling={args.pooling}\n{'=' * 84}")
    tab = pd.DataFrame({a: series(a).reindex(ALL5) for a in V4_REPLICATES + [V3_REF]})
    tab.loc["HELD-OUT MEAN"] = [hm(a) for a in V4_REPLICATES + [V3_REF]]
    tab.loc["selection mean"] = [sm(a) for a in V4_REPLICATES + [V3_REF]]
    print(tab.round(4).to_string())

    v4h = np.array([hm(a) for a in V4_REPLICATES])
    print(f"\n  three v4 replicates, held-out mean: {np.round(v4h, 4).tolist()}")
    print(f"    mean {v4h.mean():.4f}   sd {v4h.std(ddof=1):.4f}   range {v4h.ptp():.4f}")
    print(f"  v3 reference held-out mean:          {hm(V3_REF):.4f}")
    print(f"    distance from the v4 cluster mean:  {hm(V3_REF) - v4h.mean():+.4f}"
          f"   ({(hm(V3_REF) - v4h.mean()) / v4h.std(ddof=1):.1f} sd)")
    print(f"    does v3 fall inside the v4 range?   "
          f"{'YES -- gap is noise' if v4h.min() <= hm(V3_REF) <= v4h.max() else 'NO -- gap is real'}")

    # Per-target spread, which is what a per-target claim has to clear.
    print(f"\n  per-target spread across the three v4 replicates:")
    for d in ALL5:
        vals = np.array([float(series(a).get(d)) for a in V4_REPLICATES])
        print(f"    {d:20s} {np.round(vals, 4).tolist()}  sd={vals.std(ddof=1):.4f} range={vals.ptp():.4f}"
              f"   v3={float(series(V3_REF).get(d)):.4f}")

    print(f"\n{'=' * 84}\n  Q2. Peak weighting on the v3 corpus, vs the v3 reference\n{'=' * 84}")
    tab2 = pd.DataFrame({a: series(a).reindex(ALL5) for a in [V3_REF] + V3_PEAK})
    tab2.insert(0, "classical", [CLASSICAL[d] for d in ALL5])
    tab2.loc["HELD-OUT MEAN"] = [np.mean([CLASSICAL[d] for d in HELD_OUT])] + [hm(a) for a in [V3_REF] + V3_PEAK]
    tab2.loc["selection mean"] = [np.mean([CLASSICAL[d] for d in SELECTION])] + [sm(a) for a in [V3_REF] + V3_PEAK]
    print(tab2.round(4).to_string())

    pk = np.array([hm(a) for a in V3_PEAK])
    print(f"\n  peak-weighted (v3), held-out mean: {np.round(pk, 4).tolist()}  mean {pk.mean():.4f}")
    print(f"  v3 baseline, held-out mean:        {hm(V3_REF):.4f}")
    print(f"  matched delta (peak - baseline):   {pk.mean() - hm(V3_REF):+.4f}")
    print(f"  same-seed replicate spread (r1 vs r2, pure GPU nondeterminism): {abs(pk[0] - pk[1]):.4f}")
    print(f"  --> the delta is {'INSIDE' if abs(pk.mean() - hm(V3_REF)) < abs(pk[0] - pk[1]) else 'outside'}"
          f" the same-seed replicate spread")
    for d in ALL5:
        a, b = (float(series(x).get(d)) for x in V3_PEAK)
        print(f"    {d:20s} peak r1={a:.4f} r2={b:.4f} |Δr|={abs(a - b):.4f}"
              f"   baseline={float(series(V3_REF).get(d)):.4f}")

    print(f"\n{'=' * 84}\n  Noise floor -- and why the naive estimate is far too optimistic\n{'=' * 84}")
    per_target_sd = np.array([
        np.std([float(series(a).get(d)) for a in V4_REPLICATES], ddof=1) for d in HELD_OUT])
    sig = float(np.sqrt(np.mean(per_target_sd ** 2)))
    naive = float(v4h.std(ddof=1))
    indep = sig / np.sqrt(len(HELD_OUT))
    corr = float(np.corrcoef([float(series(a).get("barth")) for a in V4_REPLICATES],
                             [float(series(a).get("brc_t2d_cancer")) for a in V4_REPLICATES])[0, 1])
    print(f"  average per-target sd across replicates          {sig:.4f}")
    print(f"  sd of the 3-target held-out mean, as observed     {naive:.4f}")
    print(f"  sd it WOULD have if targets were independent      {indep:.4f}   ({indep / naive:.1f}x larger)")
    print(f"  Barth vs cancer correlation across replicates     {corr:+.3f}")
    print(f"\n  The observed {naive:.4f} is not precision -- Barth falls while cancer rises across the")
    print(f"  three draws (r={corr:+.2f}), so the errors cancel inside the mean. With only 3 draws that")
    print(f"  cancellation is luck, not a property to rely on. Use the independence figure {indep:.3f}")
    print(f"  as the floor for a held-out-MEAN claim, and ~{sig:.3f} for any single-target claim.")

    print(f"\n{'=' * 84}\n  Recalibration: every effect this document has claimed, vs a {indep:.3f} floor\n{'=' * 84}")
    claims = [
        ("§5b  patch 256 vs 1024 (v4-vs-v4)", -0.0335),
        ("§5b  patch 128 vs 1024 (v4-vs-v4)", -0.0416),
        ("§5d  ps2048 vs ps1024 (v4-vs-v4)", +0.0202),
        ("§5d  d256L6 vs ps1024 (v4-vs-v4)", +0.0057),
        ("§7   block masking main effect", -0.0298),
        ("§7   peak weighting main effect (v4, unmatched)", +0.0108),
        ("§7b  peak weighting (v3, MATCHED)", pk.mean() - hm(V3_REF)),
        ("§5f  v3 vs v4 corpus", hm(V3_REF) - v4h.mean()),
    ]
    for name, val in claims:
        ratio = abs(val) / indep
        verdict = ("SURVIVES" if ratio >= 2 else
                   "marginal" if ratio >= 1 else "WITHIN NOISE")
        print(f"    {name:48s} {val:+.4f}   {ratio:4.1f}x floor   {verdict}")
    print(f"\n  Anything below {indep:.3f} needs replicates before it can be reported as an effect.")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tab.round(4).to_csv(out / f"q1_v4_replicates_{args.pooling}.csv")
    tab2.round(4).to_csv(out / f"q2_v3_peak_vs_baseline_{args.pooling}.csv")
    print()


if __name__ == "__main__":
    main()
