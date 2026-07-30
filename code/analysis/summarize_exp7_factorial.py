#!/usr/bin/env python3
"""Experiment #7: read the 2x2 factorial over the masking pretext task.

Arms (all ps1024, d128, L3, nhead4, 1.89M params, identical v4 corpus):

    D  sparse_random masking, uniform loss      <- reference cell
    A  block masking (8 patches), uniform loss  <- factor 1 only
    B  sparse_random masking, top-25% loss      <- factor 2 only
    C  block masking, top-25% loss              <- both

Reported through the frozen linear probe, since experiment #2 established the
fine-tuned head underfits by ~0.12 on the masking family and would swamp a
representation change.

Two reporting rules carried over from earlier experiments:

  * Split honesty (§ standing selection-bias note). MTBLS563 and BrC-T2D
    diabetes were pre-committed as the SELECTION subset; Barth, MTBLS326 and
    BrC-T2D cancer are HELD OUT. Any claim about which arm won must be read off
    the held-out mean; the selection mean is where arm choice is allowed to look.
  * Pooling (§5c). flatten is the masking family's default because it beat
    mean-pool on 5/5 targets. Both are printed because the factorial could in
    principle interact with pooling -- a block-masked model has to propagate
    information across tokens, which is exactly what mean-pooling then destroys.

Main effects are computed as the mean over the two cells at each factor level,
which is what a factorial buys over one-at-a-time ablation: (A+C)/2 - (D+B)/2
isolates block masking averaged over both loss settings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ARMS = {
    "exp7_D_baseline_v4": ("sparse", "uniform"),
    "exp7_A_blk8": ("block", "uniform"),
    "exp7_B_pk025": ("sparse", "top25%"),
    "exp7_C_blk8_pk025": ("block", "top25%"),
}
SELECTION = ["mtbls563", "brc_t2d_diabetes"]
HELD_OUT = ["barth", "mtbls326", "brc_t2d_cancer"]

# Classical logistic regression on binned_abs_area, v4 numbers, for context.
CLASSICAL = {"barth": 0.705, "mtbls326": 1.000, "mtbls563": 0.721,
             "brc_t2d_cancer": 0.937, "brc_t2d_diabetes": 0.829}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/analysis/exp7_objective_comparison/patch_size_results.csv")
    ap.add_argument("--out-dir", default="results/analysis/exp7_objective_comparison")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    exp7 = df[df.arm.isin(ARMS)].copy()
    exp7["masking"] = exp7.arm.map(lambda a: ARMS[a][0])
    exp7["loss"] = exp7.arm.map(lambda a: ARMS[a][1])

    order = list(ARMS)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for pooling in ("flatten", "mean_pool"):
        sub = exp7[exp7.pooling == pooling]
        piv = sub.pivot_table(index="dataset", columns="arm", values="balanced_accuracy")
        piv = piv.reindex(columns=[a for a in order if a in piv.columns])
        piv = piv.reindex([d for d in HELD_OUT + SELECTION if d in piv.index])
        piv.insert(0, "classical_LR", [CLASSICAL[d] for d in piv.index])

        print(f"\n{'=' * 78}\n  balanced accuracy, frozen linear probe, pooling={pooling}\n{'=' * 78}")
        print(piv.round(4).to_string())

        held = piv.loc[[d for d in HELD_OUT if d in piv.index]]
        sel = piv.loc[[d for d in SELECTION if d in piv.index]]
        print(f"\n  held-out mean (Barth, MTBLS326, cancer):")
        print("   " + held.mean().round(4).to_string().replace("\n", "\n   "))
        print(f"  selection mean (MTBLS563, diabetes):")
        print("   " + sel.mean().round(4).to_string().replace("\n", "\n   "))

        # Per-arm deltas vs the reference cell D, on held-out only.
        if "exp7_D_baseline_v4" in piv.columns:
            base = held["exp7_D_baseline_v4"].mean()
            print(f"\n  held-out delta vs arm D (={base:.4f}):")
            for a in order:
                if a in piv.columns and a != "exp7_D_baseline_v4":
                    print(f"    {a:22s} {held[a].mean() - base:+.4f}"
                          f"   (wins on {int((piv[a] > piv['exp7_D_baseline_v4']).sum())}/"
                          f"{len(piv)} targets)")

        # Factorial main effects, averaged over both levels of the other factor.
        cells = {a: sub[sub.arm == a].set_index("dataset")["balanced_accuracy"] for a in order
                 if a in set(sub.arm)}
        if len(cells) == 4:
            def mean_over(datasets, arms):
                return sum(cells[a].reindex(datasets).mean() for a in arms) / len(arms)
            for name, datasets in (("held-out", HELD_OUT), ("selection", SELECTION), ("all 5", HELD_OUT + SELECTION)):
                blk = (mean_over(datasets, ["exp7_A_blk8", "exp7_C_blk8_pk025"])
                       - mean_over(datasets, ["exp7_D_baseline_v4", "exp7_B_pk025"]))
                pk = (mean_over(datasets, ["exp7_B_pk025", "exp7_C_blk8_pk025"])
                      - mean_over(datasets, ["exp7_D_baseline_v4", "exp7_A_blk8"]))
                inter = (cells["exp7_C_blk8_pk025"].reindex(datasets).mean()
                         - cells["exp7_A_blk8"].reindex(datasets).mean()
                         - cells["exp7_B_pk025"].reindex(datasets).mean()
                         + cells["exp7_D_baseline_v4"].reindex(datasets).mean())
                print(f"\n  main effects ({name}):  block-masking {blk:+.4f}   "
                      f"peak-weighting {pk:+.4f}   interaction {inter:+.4f}")

        piv.to_csv(out / f"exp7_summary_{pooling}.csv")

    # Context: how the new arms sit against every previously evaluated backbone.
    print(f"\n{'=' * 78}\n  all arms, flatten pooling, held-out vs selection mean\n{'=' * 78}")
    fl = df[df.pooling == "flatten"]
    rows = []
    for arm, g in fl.groupby("arm"):
        s = g.set_index("dataset")["balanced_accuracy"]
        rows.append(dict(arm=arm,
                         held_out=s.reindex(HELD_OUT).mean(),
                         selection=s.reindex(SELECTION).mean(),
                         all5=s.reindex(HELD_OUT + SELECTION).mean()))
    ctx = pd.DataFrame(rows).sort_values("held_out", ascending=False)
    ctx.loc[len(ctx)] = dict(arm="classical_LR",
                             held_out=sum(CLASSICAL[d] for d in HELD_OUT) / 3,
                             selection=sum(CLASSICAL[d] for d in SELECTION) / 2,
                             all5=sum(CLASSICAL.values()) / 5)
    print(ctx.round(4).to_string(index=False))
    ctx.to_csv(out / "exp7_all_arms_context.csv", index=False)
    print()


if __name__ == "__main__":
    main()
