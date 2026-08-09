#!/usr/bin/env python3
"""Experiment #13: does held-out accuracy degrade smoothly with corpus size?

Direct follow-up to experiment #8 (docs §8), which could not distinguish two
hypotheses for why v3-pretrained backbones transfer +0.069 better than
v4-pretrained ones (§5f/§7b):

    REFUTED    row CONTENT (the 164 rows that differ between v3/v4 carry the
               effect) -- common9506 and a same-size random-drop control
               landed within 0.001 of each other, all per-target gaps inside
               the 0.035 floor.
    UNCONFIRMED corpus SIZE (9,506 vs 9,670, a 1.7% cut) -- both arms sat near
               v4 (0.820) and below v3 (0.888), consistent with size mattering,
               but a 1.7% cut costing ~0.05 held-out would be disproportionate
               next to every capacity experiment in this project (up to 2.9x
               more params moved accuracy <=0.02, itself within the floor).

This sweep drops 1%, 5%, 10% of ALL v3 rows uniformly at random -- decoupled
entirely from the 164 differing rows -- and pretrains on each. If accuracy
degrades smoothly as size shrinks, size is confirmed as (part of) the story.
If cuts several times larger than 1.7% don't reproduce anything near the 0.05
swing already seen at 1.7% (experiment #8), that swing was most likely an
unlucky n=1 draw rather than a real size effect, and the honest position
reverts to: the v3-vs-v4 gap is real and large, but its cause is still open.

Every arm here is n=1 (one pretraining run per corpus size), same status as
experiment #8's arms before this follow-up. Read accordingly.
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
SWEEP = [("exp13_v3drop1pct", 1), ("exp13_v3drop5pct", 5), ("exp13_v3drop10pct", 10)]
NOISE_FLOOR_MEAN = 0.020
NOISE_FLOOR_TARGET = 0.035

CLASSICAL = {"barth": 0.705, "mtbls326": 1.000, "mtbls563": 0.721,
             "brc_t2d_cancer": 0.937, "brc_t2d_diabetes": 0.829}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/analysis/exp13_size_sweep/patch_size_results.csv")
    ap.add_argument("--pooling", default="flatten")
    ap.add_argument("--out-dir", default="results/analysis/exp13_size_sweep")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    sub = df[df.pooling == args.pooling]

    def series(arm):
        return sub[sub.arm == arm].set_index("dataset")["balanced_accuracy"]

    def hm(arm):
        return float(series(arm).reindex(HELD_OUT).mean())

    def sm(arm):
        return float(series(arm).reindex(SELECTION).mean())

    needed = [V3_REF, COMMON, CONTROL] + V4_REPLICATES + [a for a, _ in SWEEP]
    missing = [a for a in needed if a not in set(sub.arm)]
    if missing:
        raise SystemExit(f"missing arms in {args.results}: {missing}")

    v4_mean = float(np.mean([hm(a) for a in V4_REPLICATES]))
    v3_mean = hm(V3_REF)
    common_mean, control_mean = hm(COMMON), hm(CONTROL)

    # The full size axis: 0% (v3), 1%, 1.7% (the #8 pair, averaged), 5%, 10%.
    points = [(0.0, v3_mean, "v3 reference")]
    for arm, pct in SWEEP:
        points.append((float(pct), hm(arm), arm))
    points.append((1.7, float(np.mean([common_mean, control_mean])), "exp8 common+control mean"))
    points.sort(key=lambda p: p[0])

    print(f"\n{'=' * 92}\n  Experiment #13: corpus-size sweep, pooling={args.pooling}\n{'=' * 92}")
    tab = pd.DataFrame({f"{a} ({p}% dropped)": series(a).reindex(ALL5)
                        for a, p in SWEEP})
    tab.insert(0, "v3_reference (0%)", series(V3_REF).reindex(ALL5))
    tab.insert(1, "exp8_common (1.7%)", series(COMMON).reindex(ALL5))
    tab.insert(2, "exp8_control (1.7%)", series(CONTROL).reindex(ALL5))
    tab.loc["HELD-OUT MEAN"] = [v3_mean, common_mean, control_mean] + [hm(a) for a, _ in SWEEP]
    tab.loc["selection mean"] = [sm(V3_REF), sm(COMMON), sm(CONTROL)] + [sm(a) for a, _ in SWEEP]
    print(tab.round(4).to_string())

    print(f"\n  Held-out mean vs. fraction of v3 dropped (sorted by fraction):")
    for pct, val, label in points:
        bar = "#" * int(round((val - 0.78) * 200)) if val > 0.78 else ""
        print(f"    {pct:5.1f}%  {val:.4f}  {label:28s} {bar}")
    print(f"    (for reference: v4 3-replicate mean = {v4_mean:.4f}, at a DIFFERENT corpus, "
          f"not on this dropped-from-v3 axis)")

    # Monotonicity / smoothness check on the sweep-only points (excluding the
    # exp8 pair, which used a different row-selection scheme).
    sweep_vals = [v3_mean] + [hm(a) for a, _ in SWEEP]
    sweep_pcts = [0.0] + [float(p) for _, p in SWEEP]
    diffs = np.diff(sweep_vals)
    monotone_nonincreasing = bool((diffs <= NOISE_FLOOR_TARGET / 2).all())  # allow small noise-driven upticks
    print(f"\n  0% -> 1% -> 5% -> 10% sequence: " + " -> ".join(f"{v:.4f}" for v in sweep_vals))
    print(f"  step-to-step changes: " + ", ".join(f"{d:+.4f}" for d in diffs))

    max_drop_effect = v3_mean - min(sweep_vals[1:])
    largest_step = float(np.max(np.abs(diffs)))
    # A genuine size effect should be roughly monotone: MORE dropped -> AT LEAST
    # AS BAD, allowing floor-sized noise. 5% -> 10% recovering by more than the
    # floor is the opposite of what a size story predicts.
    recovers_at_larger_cut = any(
        sweep_vals[i + 1] - sweep_vals[i] > NOISE_FLOOR_MEAN for i in range(1, len(sweep_vals) - 1))
    print(f"\n  {'=' * 60}\n  VERDICT\n  {'=' * 60}")
    print(f"  Largest held-out drop anywhere in the 1-10% sweep: {max_drop_effect:.4f} "
          f"(floor {NOISE_FLOOR_MEAN})")
    print(f"  Largest single STEP change within the sweep: {largest_step:.4f}")
    print(f"  Experiment #8's 1.7% cut produced a drop of: {v3_mean - np.mean([common_mean, control_mean]):.4f}")

    if recovers_at_larger_cut:
        print("\n  NON-MONOTONIC, and in the direction that argues AGAINST a size effect: going from")
        print("  a SMALLER cut to a LARGER one (5% -> 10%) RECOVERS accuracy by more than the noise")
        print("  floor, rather than degrading it further. A genuine size effect predicts monotone-or-")
        print("  flat degradation; recovering at a larger cut is what n=1-per-point sampling noise")
        print("  looks like, not what a real effect looks like.")
        print("  -> Corpus SIZE is NOT supported by this sweep. Combined with experiment #8 (row")
        print("     content refuted), the honest position is: the v3-vs-v4 gap (+0.069, §5f) is real")
        print("     and well-supported (3 replicates, 3.4x the floor), but NEITHER row content NOR")
        print("     corpus size explains it. The cause remains open, and single-run size/subset")
        print("     ablations are not the way to find it -- each point here is itself only n=1.")
    elif max_drop_effect < NOISE_FLOOR_MEAN and abs(v3_mean - np.mean([common_mean, control_mean])) > NOISE_FLOOR_MEAN:
        print("\n  Cuts of 1/5/10% do NOT reproduce anything near experiment #8's ~0.05 drop at 1.7%.")
        print("  -> Corpus SIZE is NOT confirmed as the explanation. The #8 result (common9506 and")
        print("     its control both landing ~0.05 below v3) was most likely an unlucky n=1 draw for")
        print("     THAT SPECIFIC SET of dropped rows, not a general property of corpus size.")
        print("  -> The v3-vs-v4 gap (+0.069, §5f) remains real and well-supported (3 replicates,")
        print("     3.4x the floor) but its CAUSE is still open. Neither row content (refuted, §8)")
        print("     nor corpus size (refuted here) explains it on current evidence.")
    elif all(d <= NOISE_FLOOR_TARGET / 2 for d in diffs) and diffs.sum() < -NOISE_FLOOR_MEAN:
        print("\n  Accuracy degrades roughly monotonically as corpus size shrinks.")
        print("  -> Corpus SIZE is supported as (at least part of) the explanation.")
    else:
        print("\n  Mixed / non-monotonic pattern -- inconclusive. See the step-to-step changes above;")
        print("  this needs replicates (this is n=1 per size point) before a verdict either way.")

    print(f"\n  Standing reminder: every arm here is n=1 against a {NOISE_FLOOR_TARGET} per-target / "
          f"{NOISE_FLOOR_MEAN} held-out-mean floor (§7b).")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tab.round(4).to_csv(out / f"exp13_summary_{args.pooling}.csv")
    print(f"\nWrote {out / f'exp13_summary_{args.pooling}.csv'}\n")


if __name__ == "__main__":
    main()
