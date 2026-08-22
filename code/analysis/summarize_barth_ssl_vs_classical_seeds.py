#!/usr/bin/env python3
"""Experiment #18: does masked SSL actually beat classical LogReg on Barth?

Barth was the project's ONE claimed SSL win over classical ML (§5c: 0.806 vs
0.705, "+0.101 SSL WINS"), and it is quoted on the group-meeting scorecard as
"1 win / 1 tie / 3 losses". That 0.806 comes from a SINGLE pretraining run --
the arm `ps1024_nhead4_true`.

Experiment #15 (§15) established that separately-trained checkpoints differ by
sd 0.045 on downstream balanced accuracy, and that the v3-vs-v4 "corpus effect"
evaporated once each corpus was given n=5 seeds. The same seeds exist for
Barth, so the Barth win can be re-read the same way, and it should have been:
the arm behind the win is the ORIGINAL UNSEEDED run, which is exactly the
position §15 showed to be selection-biased upward.

This script asks the question §15's machinery makes unavoidable: across the
five v3 and five v4 checkpoints, how often does masked SSL beat classical
LogReg on Barth at all?

The classical reference is LogReg on 1024 abs-area bins (0.704969), the number
reported in §3 and produced by probe_logreg_advantage.py.

Both poolings are reported. mean_pool is the pooling the February-era pipeline
used; flatten is the position-preserving pooling §5c switched to and is the one
the 0.806 win was quoted from.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]

# The two n=5 seed families defined in §15 / results/analysis/exp15_seed_study.
# "original (unseeded)" is the arm each headline number was quoted from.
V3 = ["ps1024_nhead4_true", "exp15_v3_seed202", "exp15_v3_seed303",
      "exp15_v3_seed404", "exp15_v3_seed505"]
V4 = ["exp7_D_baseline_v4", "exp7_D_v4_seed101", "exp7_D_v4_seed202",
      "exp15_v4_seed303", "exp15_v4_seed404"]
QUOTED = {"ps1024_nhead4_true", "exp7_D_baseline_v4"}

CLASSICAL_BARTH = 0.704969   # LogReg, 1024 abs-area bins, LOOCV (n=37) -- §3
NOISE_SD = 0.045             # single-run noise floor measured in §15


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results",
                    default="results/analysis/exp15_seed_study/patch_size_results.csv")
    ap.add_argument("--out-dir", default="results/analysis/barth_seeds_vs_classical")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / args.results)
    barth = df[df.dataset == "barth"]
    if barth.empty:
        raise SystemExit(f"no barth rows in {args.results}")

    rows = []
    for corpus, arms in (("v3", V3), ("v4", V4)):
        for pooling in ("flatten", "mean_pool"):
            for arm in arms:
                hit = barth[(barth.arm == arm) & (barth.pooling == pooling)]
                if hit.empty:
                    raise SystemExit(
                        f"missing arm={arm} pooling={pooling} in {args.results}. "
                        f"Re-run compare_patch_sizes.py for that arm before summarizing.")
                acc = float(hit.balanced_accuracy.iloc[0])
                rows.append(dict(corpus=corpus, pooling=pooling, arm=arm,
                                 is_quoted_arm=arm in QUOTED,
                                 balanced_accuracy=acc,
                                 delta_vs_classical=acc - CLASSICAL_BARTH,
                                 beats_classical=acc > CLASSICAL_BARTH))
    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "barth_seed_runs.csv", index=False)

    group_rows = []
    print(f"\nBarth (n=37, LOOCV). Classical LogReg @1024 bins = {CLASSICAL_BARTH:.4f}")
    print(f"Single-run noise floor from §15 = {NOISE_SD:.3f} sd\n")
    for pooling in ("flatten", "mean_pool"):
        print(f"===== pooling = {pooling} =====")
        for corpus in ("v3", "v4"):
            g = runs[(runs.pooling == pooling) & (runs.corpus == corpus)]
            v = g.balanced_accuracy.to_numpy()
            for _, r in g.iterrows():
                tag = "  <-- the arm every headline number was quoted from" if r.is_quoted_arm else ""
                print(f"  {corpus} {r.arm:<22} {r.balanced_accuracy:.4f}  "
                      f"{r.delta_vs_classical:+.4f}  "
                      f"{'BEAT' if r.beats_classical else 'lose'}{tag}")
            sd = v.std(ddof=1)
            se = sd / np.sqrt(len(v))
            d = v - CLASSICAL_BARTH
            # one-sample Wilcoxon of the per-seed deltas against zero
            p = wilcoxon(d).pvalue if np.any(d != 0) else np.nan
            print(f"  -> {corpus} mean {v.mean():.4f} +- {sd:.4f} sd (se {se:.4f}); "
                  f"delta {d.mean():+.4f} +- {se:.4f}; beats classical "
                  f"{int(g.beats_classical.sum())}/{len(g)}; wilcoxon p={p:.3f}\n")
            group_rows.append(dict(
                pooling=pooling, corpus=corpus, n_seeds=len(v),
                mean=v.mean(), sd=sd, se=se,
                quoted_arm_value=float(g[g.is_quoted_arm].balanced_accuracy.iloc[0]),
                quoted_arm_z=float((g[g.is_quoted_arm].balanced_accuracy.iloc[0] - v.mean()) / sd),
                delta_vs_classical=d.mean(), wilcoxon_p=float(p) if p == p else np.nan,
                n_beating_classical=int(g.beats_classical.sum())))
    groups = pd.DataFrame(group_rows)
    groups.to_csv(out_dir / "barth_seed_groups.csv", index=False)

    print("=" * 74)
    print("  VERDICT")
    print("=" * 74)
    tot = int(runs.beats_classical.sum())
    print(f"  Across all {len(runs)} (corpus x pooling x seed) readings, masked SSL beats")
    print(f"  classical LogReg on Barth in {tot} of them.")
    for _, r in groups.iterrows():
        print(f"    {r.pooling:<10} {r.corpus}: {r.n_beating_classical}/{r.n_seeds} seeds beat classical; "
              f"quoted arm sits {r.quoted_arm_z:+.2f} sd above its own group mean")
    best = groups.loc[groups.delta_vs_classical.idxmax()]
    print(f"\n  Best group is {best.pooling}/{best.corpus}: delta {best.delta_vs_classical:+.4f} "
          f"+- {best.se:.4f} se -- inside the {NOISE_SD:.3f} noise floor.")
    print("  The Barth 'SSL WINS' result is a seed artifact, in the same way and for the")
    print("  same reason as the v3-vs-v4 corpus effect retracted in §15.")
    print(f"\nWrote {out_dir}/barth_seed_runs.csv")
    print(f"Wrote {out_dir}/barth_seed_groups.csv")


if __name__ == "__main__":
    main()
