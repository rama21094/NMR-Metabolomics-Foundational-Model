#!/usr/bin/env python3
"""Experiment #15: what does a single pretraining run actually mean?

THE RESULT THAT PROMPTED THIS. §5f reported that v3-pretrained backbones
transfer +0.069 better than v4-pretrained ones -- the largest effect in the
project, which §8 and §13 then spent two more experiments failing to explain.
But that comparison put a v3 reference of n=1 against a v4 mean of n=3, and the
v3 value was the highest of six comparable v3-family arms. This study gives each
corpus n=5 seeds so the gap can be read with error bars on both sides.

WHAT IT FOUND. The gap does not shrink -- it vanishes:

    v3, n=5:  0.8165 +/- 0.0449   (0.8884, 0.8190, 0.8067, 0.8033, 0.7653)
    v4, n>=3: ~0.820
    gap as reported (v3 n=1):  +0.0687
    gap with v3 at n=5:        ~-0.003, well inside one standard error

The 0.8884 reference was a lucky draw sitting +1.60 sd above its own
distribution. So:

  * §5f is REFUTED. There is no corpus effect.
  * §8 (the 164 differing rows) and §13 (corpus size) were looking for the
    mechanism of an effect that does not exist. Their inconclusive readings
    were right, for a reason neither experiment could see.
  * The single-run sd on the held-out mean is ~0.045, not the 0.020 previously
    assumed. That 0.020 came from three v4 runs whose per-target errors happened
    to cancel inside the mean -- a cancellation flagged as a fluke at the time
    and then used as a floor anyway.

Against a 0.045 single-run sd, only §6b (masked pretraining vs a random-init
control, +0.117) survives among the single-run claims. Every patch-size,
capacity and pretext-objective result falls within noise. The PAIRED
within-checkpoint results are unaffected, because they compare transforms on a
fixed checkpoint and carry no training variance: §4b (head fix, +0.120),
§5c (pooling, +0.03..+0.13) and §14 (jigsaw/joint pooling).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HELD_OUT = ["barth", "mtbls326", "brc_t2d_cancer"]
SELECTION = ["mtbls563", "brc_t2d_diabetes"]
ALL5 = HELD_OUT + SELECTION

V3_ARMS = [("ps1024_nhead4_true", "original (unseeded)"),
           ("exp15_v3_seed202", "seed 202"),
           ("exp15_v3_seed303", "seed 303"),
           ("exp15_v3_seed404", "seed 404"),
           ("exp15_v3_seed505", "seed 505")]
V4_ARMS = [("exp7_D_baseline_v4", "original (unseeded)"),
           ("exp7_D_v4_seed101", "seed 101"),
           ("exp7_D_v4_seed202", "seed 202"),
           ("exp15_v4_seed303", "seed 303"),
           ("exp15_v4_seed404", "seed 404")]

# Single-run claims made elsewhere in the document, to be re-scored against the
# sd measured here rather than the 0.020 that was assumed.
CLAIMS = [
    ("§6b masked pretraining vs random", +0.1170),
    ("§5f v3 vs v4 corpus (as reported)", +0.0687),
    ("§7b peak weighting (v3, matched)", -0.0423),
    ("§5b patch 128 vs 1024", -0.0416),
    ("§5b patch 256 vs 1024", -0.0335),
    ("§7 block masking", -0.0298),
    ("§5d ps2048 vs ps1024", +0.0202),
    ("§7 peak weighting (v4, unmatched)", +0.0108),
    ("§5d d256L6 vs ps1024", +0.0057),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/analysis/exp15_seed_study/patch_size_results.csv")
    ap.add_argument("--pooling", default="flatten")
    ap.add_argument("--out-dir", default="results/analysis/exp15_seed_study")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    sub = df[df.pooling == args.pooling]
    present = set(sub.arm)

    def series(arm):
        return sub[sub.arm == arm].set_index("dataset")["balanced_accuracy"]

    def held(arm):
        return float(series(arm).reindex(HELD_OUT).mean())

    def collect(arms):
        return [(lab, arm, held(arm)) for arm, lab in arms if arm in present]

    v3, v4 = collect(V3_ARMS), collect(V4_ARMS)
    if len(v3) < 2 or len(v4) < 2:
        raise SystemExit(f"need >=2 arms per corpus; found v3={len(v3)}, v4={len(v4)}")

    print(f"\n{'=' * 86}\n  Experiment #15: seed replicates, pooling={args.pooling}\n{'=' * 86}")
    for name, group in (("v3", v3), ("v4", v4)):
        vals = np.array([v for _, _, v in group])
        print(f"\n--- {name} corpus, n={len(vals)} (held-out mean) ---")
        for lab, _, v in sorted(group, key=lambda t: -t[2]):
            print(f"    {lab:22s} {v:.4f}")
        print(f"    mean {vals.mean():.4f}   sd {vals.std(ddof=1):.4f}   "
              f"range {vals.ptp():.4f}   se {vals.std(ddof=1) / np.sqrt(len(vals)):.4f}")

    v3v = np.array([v for _, _, v in v3])
    v4v = np.array([v for _, _, v in v4])
    orig3 = held("ps1024_nhead4_true")
    rank = list(sorted(v3v)[::-1]).index(orig3) + 1
    z = (orig3 - v3v.mean()) / v3v.std(ddof=1)

    print(f"\n{'=' * 86}\n  The §5f comparison, then and now\n{'=' * 86}")
    print(f"  AS REPORTED   v3 (n=1) {orig3:.4f}  -  v4 (n=3) 0.8196  =  +0.0687")
    se = np.sqrt(v3v.var(ddof=1) / len(v3v) + v4v.var(ddof=1) / len(v4v))
    diff = v3v.mean() - v4v.mean()
    print(f"  WITH ERRORS   v3 (n={len(v3v)}) {v3v.mean():.4f}  -  v4 (n={len(v4v)}) {v4v.mean():.4f}  "
          f"=  {diff:+.4f}")
    print(f"                se of the difference {se:.4f}  ->  {abs(diff) / se:.1f} se")
    print(f"\n  The original v3 reference ranks {rank} of {len(v3v)} and sits {z:+.2f} sd above")
    print(f"  its own distribution. It was a lucky draw.")
    verdict = "REFUTED -- no corpus effect" if abs(diff) < 2 * se else "survives"
    print(f"\n  VERDICT: §5f {verdict}")

    sd_single = v3v.std(ddof=1)
    print(f"\n{'=' * 86}\n  Recalibration against the MEASURED single-run sd ({sd_single:.4f})\n{'=' * 86}")
    print(f"  (previously assumed 0.020, derived from three v4 runs whose per-target")
    print(f"   errors cancelled inside the mean -- a fluke flagged at the time)\n")
    for name, val in CLAIMS:
        r = abs(val) / sd_single
        v = "SURVIVES" if r >= 2 else ("marginal" if r >= 1 else "WITHIN NOISE")
        print(f"    {name:40s} {val:+.4f}  {r:4.1f}x sd   {v}")
    print(f"\n  Paired within-checkpoint results are NOT single-run comparisons and are")
    print(f"  unaffected: §4b head fix (+0.120), §5c pooling (+0.03..+0.13), §14 jigsaw/joint.")

    print(f"\n{'=' * 86}\n  Per-target sd across the {len(v3v)} v3 runs\n{'=' * 86}")
    rows = []
    for d in ALL5:
        col = np.array([float(series(a).get(d)) for _, a, _ in v3])
        rows.append(dict(target=d, mean=col.mean(), sd=col.std(ddof=1), rng=col.ptp()))
        print(f"    {d:20s} mean {col.mean():.4f}  sd {col.std(ddof=1):.4f}  range {col.ptp():.4f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"corpus": c, "arm": a, "label": l, "held_out_mean": v}
                  for c, grp in (("v3", v3), ("v4", v4)) for l, a, v in grp]
                 ).to_csv(out / f"exp15_runs_{args.pooling}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"exp15_per_target_sd_{args.pooling}.csv", index=False)
    print(f"\nWrote {out / f'exp15_runs_{args.pooling}.csv'}\n")


if __name__ == "__main__":
    main()
