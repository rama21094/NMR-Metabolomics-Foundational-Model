#!/usr/bin/env python3
"""Does diversity or volume drive transfer? Pooled across both row budgets.

The 2,000- and 4,000-row fixed-budget axes cannot be compared cell by cell: a
bigger budget dilutes diversity (see docs/PHASE4_budget4000_caveat.md), so
fixed4000_k12 vs fixed2000_k12 confounds +2,000 rows against -1.34 effective
studies. Pooling both budgets turns that confound into leverage -- across the 24
fixed-budget checkpoints, effective study count and row count are no longer
collinear, so a regression can separate them.

DIVERSITY IS MEASURED AS EFFECTIVE, NOT NOMINAL, STUDY COUNT:
exp(entropy of the study composition). Nominal k badly overstates diversity here
because seven of twelve studies hold 1-76 rows between them -- nominal 12 is
effectively 6.89 at a 2,000-row budget and 5.55 at 4,000.

Reported per label budget, with a permutation test on each coefficient because
n=24 and the seed spread is small but not negligible.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_eff():
    """subset -> (eff_studies, n_rows), from both fixed-budget manifests."""
    eff = {}
    for mf in ("manifest.json", "manifest_fixed4000.json"):
        p = ROOT / "rebuild/pretrain/scaling" / mf
        if not p.exists():
            continue
        for r in json.load(open(p)):
            if r.get("axis") != "fixed_budget":
                continue
            eff[r["subset"]] = (r.get("eff_studies"), r["n_rows"])
    return eff


def ols(X, y):
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ beta
    ss = 1 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return beta, ss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+",
                    default=["results/phase4/scaling_eval_full.json",
                             "results/phase4/scaling_eval_4000.json"])
    ap.add_argument("--perm", type=int, default=20000)
    args = ap.parse_args()

    rows = []
    for f in args.evals:
        p = ROOT / f
        if p.exists():
            rows += json.load(open(p))
    eff = load_eff()

    # subset x seed x budget -> mean over the 6 targets, per label budget
    agg = defaultdict(list)
    for r in rows:
        if not r["subset"].startswith("fixed"):
            continue
        agg[(r["k_per_class"], r["subset"], r["seed"])].append(r["mean"])

    rng = np.random.default_rng(0)
    for kb in sorted({k for k, _, _ in agg}):
        pts = []
        for (k, sub, seed), v in agg.items():
            if k != kb or sub not in eff or eff[sub][0] is None:
                continue
            e, n = eff[sub]
            pts.append((e, n, float(np.mean(v)), sub, seed))
        if len(pts) < 8:
            print(f"k={kb}: only {len(pts)} points, skipping")
            continue
        E = np.array([p[0] for p in pts]); N = np.array([p[1] for p in pts])
        Y = np.array([p[2] for p in pts])
        # standardise so the two coefficients are directly comparable
        Ez = (E - E.mean()) / E.std(); Nz = (N - N.mean()) / N.std()
        beta, r2 = ols(np.column_stack([Ez, Nz]), Y)
        kl = "all" if kb == 0 else str(kb)
        print(f"\nlabel budget k={kl}   n={len(pts)} checkpoints   "
              f"corr(eff, rows) = {np.corrcoef(Ez, Nz)[0,1]:+.2f}")
        print(f"  intercept            {beta[0]:+.4f}")
        names = ["effective studies", "row count"]
        for j, nm in enumerate(names):
            obs = beta[1 + j]
            null = []
            for _ in range(args.perm):
                pm = rng.permutation(len(Y))
                Z = np.column_stack([Ez, Nz])
                Z[:, j] = Z[pm, j]
                b, _ = ols(Z, Y)
                null.append(b[1 + j])
            pv = float(np.mean(np.abs(null) >= abs(obs)))
            print(f"  {nm:<20} {obs:+.4f} per sd    p = {pv:.4f}"
                  f"{'  *' if pv < 0.05 else ''}")
        print(f"  R2 = {r2:.3f}")
        for e, n, y, sub, seed in sorted(pts):
            print(f"      {sub:<18} seed{seed}  eff {e:.2f}  rows {n:>5,}  {y:.3f}")


if __name__ == "__main__":
    main()
