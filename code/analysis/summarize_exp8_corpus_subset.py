#!/usr/bin/env python3
"""Experiment #8: does the 1.7% of rows that differ between v3 and v4 explain
the 0.069 held-out gap between them (docs §5f/§7b)?

Direct diff of the two corpora found only 164/9,670 rows (1.7%) differ at all,
but each differs almost entirely (99.998% of points changed, median max|dv| =
0.34), because rows are min-max normalized -- changing a row's max rescales the
whole row. The obvious mechanism (v4 leaves a residual EDTA artefact that
compresses those spectra) was tested and REFUTED: only 7/164 rows have their
max inside the EDTA window in either version, and v4's rows are slightly
BRIGHTER outside it, not more compressed.

Two arms settle it by ablation (build_corpus_subset.py):

  exp8_common9506           the 9,506 rows identical in v3 and v4 -- the
                            decisive arm. Verified max|dv|=0 across v3/v4 for
                            every kept row.
  exp8_v3rand9506_control    v3 with 164 DIFFERENT (always-unchanged) rows
                            dropped at random, matching common9506's size
                            exactly while keeping all 164 special rows --
                            isolates corpus SIZE from corpus CONTENT.

Reading rule -- READ COMMON-VS-CONTROL FIRST, not common-vs-{v3,v4}. The
temptation is to place `common` on the v4->v3 axis and declare a winner, but
that skips the actual control: if `control` (dropping a DIFFERENT random 164
rows) lands in the same place as `common` (dropping the SPECIFIC 164 that
differ), the content of those rows is not shown to matter -- only look at
common's position on the v3/v4 axis if common and control themselves differ by
more than the noise floor.

RESULT (2026-08-09, n=1 per arm): common9506 = 0.837, control = 0.836 held-out
mean, indistinguishable (|diff|=0.0009, floor 0.020; max per-target |diff| =
0.028, floor 0.035). Both sit close to the v4 replicate mean (0.820, |diff|=
0.018, inside the floor) and below v3 (0.888, |diff|=0.051, outside the floor).
So: the specific 164 rows are NOT established as the cause -- dropping ANY 164
rows gets a similar result. Whether that is corpus SIZE (a 1.7% cut) or just
this being a second unreplicated draw is NOT settled either; a 1.7% size cut
producing a ~0.05 swing would be disproportionate given every capacity
experiment (up to 2.9x more params) moved accuracy by <=0.02. Needs >=2 more
replicates of both arms before any reading is reported as established.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HELD_OUT = ["barth", "mtbls326", "brc_t2d_cancer"]
SELECTION = ["mtbls563", "brc_t2d_diabetes"]
ALL5 = HELD_OUT + SELECTION

V3_REF = "ps1024_nhead4_true"
V4_REPLICATES = ["exp7_D_baseline_v4", "exp7_D_v4_seed101", "exp7_D_v4_seed202"]
COMMON = "exp8_common9506"
CONTROL = "exp8_v3rand9506_control"
NOISE_FLOOR_MEAN = 0.020
NOISE_FLOOR_TARGET = 0.035

CLASSICAL = {"barth": 0.705, "mtbls326": 1.000, "mtbls563": 0.721,
             "brc_t2d_cancer": 0.937, "brc_t2d_diabetes": 0.829}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/analysis/exp8_corpus_subset/patch_size_results.csv")
    ap.add_argument("--pooling", default="flatten")
    ap.add_argument("--out-dir", default="results/analysis/exp8_corpus_subset")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    sub = df[df.pooling == args.pooling]

    def series(arm):
        return sub[sub.arm == arm].set_index("dataset")["balanced_accuracy"]

    def hm(arm):
        return float(series(arm).reindex(HELD_OUT).mean())

    def sm(arm):
        return float(series(arm).reindex(SELECTION).mean())

    arms = [V3_REF, COMMON, CONTROL] + V4_REPLICATES
    missing = [a for a in arms if a not in set(sub.arm)]
    if missing:
        raise SystemExit(f"missing arms in {args.results}: {missing}")

    print(f"\n{'=' * 88}\n  Experiment #8: corpus-subset ablation, pooling={args.pooling}\n{'=' * 88}")
    tab = pd.DataFrame({a: series(a).reindex(ALL5) for a in [V3_REF, COMMON, CONTROL]})
    v4_mean_series = pd.concat([series(a).reindex(ALL5) for a in V4_REPLICATES], axis=1).mean(axis=1)
    tab.insert(1, "v4_replicate_mean", v4_mean_series)
    tab.insert(0, "classical", [CLASSICAL[d] for d in ALL5])
    tab.loc["HELD-OUT MEAN"] = [np.mean([CLASSICAL[d] for d in HELD_OUT])] + [
        pd.concat([series(a).reindex(HELD_OUT) for a in V4_REPLICATES], axis=1).mean(axis=1).mean(),
        hm(V3_REF), hm(COMMON), hm(CONTROL)]
    tab.loc["selection mean"] = [np.mean([CLASSICAL[d] for d in SELECTION])] + [
        pd.concat([series(a).reindex(SELECTION) for a in V4_REPLICATES], axis=1).mean(axis=1).mean(),
        sm(V3_REF), sm(COMMON), sm(CONTROL)]
    print(tab.round(4).to_string())

    v4_mean = float(np.mean([hm(a) for a in V4_REPLICATES]))
    v3_mean = hm(V3_REF)
    common_mean = hm(COMMON)
    control_mean = hm(CONTROL)
    gap = v3_mean - v4_mean

    print(f"\n  reference points (held-out mean):")
    print(f"    v3                       {v3_mean:.4f}")
    print(f"    v4 (3-replicate mean)    {v4_mean:.4f}")
    print(f"    v3 - v4 gap              {gap:+.4f}  (established §5f/§7b effect)")
    print(f"\n  ablation arms (held-out mean):")
    print(f"    common9506 (164 dropped) {common_mean:.4f}   "
          f"distance to v3={common_mean - v3_mean:+.4f}  distance to v4={common_mean - v4_mean:+.4f}")
    print(f"    v3rand9506 control       {control_mean:.4f}   "
          f"distance to v3={control_mean - v3_mean:+.4f}  distance to v4={control_mean - v4_mean:+.4f}")

    # The decisive comparison is common vs control, NOT common vs {v3,v4} alone:
    # if a random-164-drop lands in the same place as dropping the SPECIFIC 164
    # differing rows, the content of those rows is not what is doing the work.
    common_series = series(COMMON).reindex(ALL5)
    control_series = series(CONTROL).reindex(ALL5)
    cc_mean_gap = abs(common_mean - control_mean)
    cc_per_target = (common_series - control_series).abs()
    print(f"\n  common vs control (the real test -- does row IDENTITY matter?):")
    print(f"    held-out mean:  common={common_mean:.4f}  control={control_mean:.4f}  "
          f"|diff|={cc_mean_gap:.4f}  (floor {NOISE_FLOOR_MEAN})")
    print(f"    per-target |diff|: " + ", ".join(f"{d}={v:.4f}" for d, v in cc_per_target.items()))
    print(f"    max per-target |diff| = {cc_per_target.max():.4f}  (floor {NOISE_FLOOR_TARGET})")

    content_matters = cc_mean_gap > NOISE_FLOOR_MEAN or cc_per_target.max() > NOISE_FLOOR_TARGET
    common_vs_v3 = abs(common_mean - v3_mean)
    common_vs_v4 = abs(common_mean - v4_mean)

    print(f"\n  {'=' * 60}\n  VERDICT\n  {'=' * 60}")
    if not content_matters:
        print("  common and control are INDISTINGUISHABLE from each other (every per-target\n"
              "  and mean gap is inside the noise floor). Dropping the 164 rows that differ\n"
              "  between v3/v4 gives the SAME result as dropping 164 arbitrary unchanged rows.\n"
              "  -> The specific CONTENT of those 164 rows is NOT established as the cause.\n"
              "     'v3's rows are helpful' / 'v4's rows are harmful' are NOT supported.")
        if common_vs_v3 > NOISE_FLOOR_MEAN and common_vs_v4 <= NOISE_FLOOR_MEAN:
            print(f"  Both ablation arms sit close to the v4 mean ({v4_mean:.4f}, |diff|="
                  f"{common_vs_v4:.4f}, within floor) and clearly below v3 ({v3_mean:.4f}, "
                  f"|diff|={common_vs_v3:.4f}, exceeds floor).\n"
                  f"  -> Best-supported reading: corpus SIZE (9506 vs 9670, a 1.7% cut) may be\n"
                  f"     doing more than expected, OR this is a second unreplicated single-run\n"
                  f"     draw and the true value is closer to v4 than to v3 anyway. A 1.7% size\n"
                  f"     cut producing ~0.05 held-out swing would be disproportionate given every\n"
                  f"     capacity experiment (up to 2.9x params) moved accuracy by <=0.02 -- so\n"
                  f"     treat 'size explains it' as unconfirmed too, not a second established fact.")
        elif common_vs_v3 <= NOISE_FLOOR_MEAN:
            print(f"  Both ablation arms sit close to v3 ({v3_mean:.4f}) -- consistent with the\n"
                  f"  164 rows barely mattering and the v3/v4 gap living elsewhere.")
        else:
            print(f"  Both ablation arms sit between v3 and v4, outside the floor from both --\n"
                  f"  genuinely ambiguous with n=1 per arm.")
    else:
        print("  common and control DIFFER by more than the noise floor -- the content of the\n"
              "  164 differing rows (not just corpus size) has a detectable effect. Proceed to\n"
              "  read common's position on the v4->v3 axis below.")
        if abs(gap) > 1e-9:
            frac = (common_mean - v4_mean) / gap
            print(f"  common9506 sits at {frac:.2f} of the way from v4 (0.0) to v3 (1.0).")

    print(f"\n  This is n=1 per arm against a per-target floor of ~{NOISE_FLOOR_TARGET} and a\n"
          f"  held-out-mean floor of ~{NOISE_FLOOR_MEAN} (§7b). Do not treat this as final --\n"
          f"  it needs >=2 more replicates of common9506 and control before any of the above\n"
          f"  is reported as established.")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tab.round(4).to_csv(out / f"exp8_summary_{args.pooling}.csv")
    print(f"\nWrote {out / f'exp8_summary_{args.pooling}.csv'}\n")


if __name__ == "__main__":
    main()
