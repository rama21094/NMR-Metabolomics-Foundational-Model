#!/usr/bin/env python3
"""Experiment #6: does masked SSL beat classical ML in the few-shot regime?

This is the experiment the whole project's premise rests on. Every full-data
number in docs/SSL_vs_classical_analysis.md (§3) is measured with the entire
dataset available, where n=37..113 is already near the ceiling of what is
learnable -- the *worst* regime to look for a pretraining advantage. Transfer
from a pretrained backbone is supposed to pay off when labels are scarce, so
this sweeps support_per_class from 2 up to (almost) the full dataset and asks
where, if anywhere, the SSL curve is above the classical one.

WHY THE COMPARISON HERE IS PAIRED (and why that matters a lot)
--------------------------------------------------------------
`build_shared_splits` generates the (support, query) episodes ONCE per
(seed, support_per_class), independent of model, and every family consumes the
identical draws. So for a fixed (dataset, support, repeat) the classical and
masking rows were scored on exactly the same support set and exactly the same
query set. Their difference is therefore a PAIRED observation: the
episode-draw variance -- which is large here, std 0.07..0.15 on a single
episode -- cancels out of the difference.

This is the same distinction that decided experiment #15 (docs §15): unpaired
single-run comparisons at this sample size have a 0.045 noise floor and need
>=5 replicates to say anything, whereas paired within-episode comparisons
(§4b head fix, §5c pooling, §14) stayed valid throughout. We report the paired
mean difference with its own standard error, and a Wilcoxon signed-rank test,
rather than eyeballing two overlapping error bars -- comparing the marginal
std's of the two curves would badly understate the available power.

SELECTION BIAS
--------------
Quoting "best fine-tune mode per point" inflates the winner (docs §5e note).
So the headline comparison fixes ONE pre-committed mode and one classical
reference, and the per-mode table is reported separately as diagnostics, not
as the result.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]

# (label, results dir). Barth's run used the dense support grid (it was
# launched before the coarse-grid switch and its classical+masking arms had
# already completed); the rest use the coarse grid. Grid density does not
# affect any per-support-point comparison below -- it only changes which
# support values exist.
RUNS = [
    ("Barth", "results/fewshot/barth_v2_repooled", 37),
    ("MTBLS326", "results/fewshot/mtbls326_v2_coarse_pass1", 42),
    ("MTBLS563", "results/fewshot/mtbls563_v2_coarse", 113),
    ("BrC-T2D cancer", "results/fewshot/brc_t2d_cancer_v2_coarse", 78),
    ("BrC-T2D diabetes", "results/fewshot/brc_t2d_diabetes_v2_coarse", 78),
]

# Pre-committed reference arms for the headline paired comparison.
CLASSICAL_REF = "logistic_regression"   # the classical track's reported model in §2/§3
MASKING_MODE = "frozen"                 # the cheapest arm; no backbone training per episode
METRIC = "balanced_accuracy"


def load_episodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path / "fewshot_episode_metrics.csv")
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"  WARNING: {len(bad)} non-ok episodes in {path.name}; excluded")
    df = df[df.status == "ok"].copy()
    # fewshot_benchmark.py rebuilds all_rows from scratch and checkpoints after
    # every family, so pointing a second run (e.g. jigsaw+joint) at a directory
    # that already holds classical+masking OVERWRITES it. That silently reduced
    # a dataset to NaN here once; fail loudly instead.
    missing = {"classical", "masking"} - set(df.family.unique())
    if missing:
        raise SystemExit(
            f"{path}/fewshot_episode_metrics.csv is missing family {sorted(missing)} "
            f"(has {sorted(df.family.unique())}).\n"
            f"  A later run with a different --families almost certainly overwrote it. "
            f"Re-run pass 1 for this dataset into its OWN --output-dir.")
    return df


def paired_diff(df: pd.DataFrame, classical_model: str, masking_mode: str):
    """Per-(support,repeat) paired difference: masking - classical, same episode."""
    c = df[(df.family == "classical") & (df.model == classical_model)]
    m = df[(df.family == "masking") & (df.fine_tune_mode == masking_mode)]
    key = ["support_per_class", "repeat"]
    merged = c[key + [METRIC]].merge(
        m[key + [METRIC]], on=key, suffixes=("_classical", "_masking"), validate="one_to_one")
    merged["diff"] = merged[f"{METRIC}_masking"] - merged[f"{METRIC}_classical"]
    return merged


def summarize_run(name: str, path: Path, n_samples: int, rows_curve: list, rows_paired: list,
                  rows_modes: list):
    df = load_episodes(path)
    supports = sorted(df.support_per_class.unique())
    print(f"\n{'='*78}\n  {name}  (n={n_samples}, supports={supports})\n{'='*78}")

    # ---- learning curves, every arm ----
    for (family, model, mode), g in df.groupby(["family", "model", "fine_tune_mode"]):
        for s, gs in g.groupby("support_per_class"):
            v = gs[METRIC].to_numpy()
            rows_curve.append(dict(
                dataset=name, n_samples=n_samples, family=family, model=model,
                fine_tune_mode=mode, support_per_class=int(s), n_episodes=len(v),
                mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan))

    # ---- headline paired comparison, pre-committed arms ----
    merged = paired_diff(df, CLASSICAL_REF, MASKING_MODE)
    print(f"\n  PAIRED: masking/{MASKING_MODE} - classical/{CLASSICAL_REF}  (same episodes)")
    print(f"  {'supp':>5}  {'classical':>11}  {'masking':>11}  {'paired diff':>13}  {'se':>7}  {'p':>7}")
    for s, g in merged.groupby("support_per_class"):
        d = g["diff"].to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        try:
            p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else np.nan
        except ValueError:
            p = np.nan
        star = "*" if (p == p and p < 0.05) else " "
        print(f"  {int(s):5d}  {g[f'{METRIC}_classical'].mean():11.3f}  "
              f"{g[f'{METRIC}_masking'].mean():11.3f}  {d.mean():+13.3f}  {se:7.3f}  {p:7.3f}{star}")
        rows_paired.append(dict(
            dataset=name, n_samples=n_samples, support_per_class=int(s), n_episodes=len(d),
            classical_mean=float(g[f"{METRIC}_classical"].mean()),
            masking_mean=float(g[f"{METRIC}_masking"].mean()),
            paired_diff=float(d.mean()), se=float(se), wilcoxon_p=float(p) if p == p else np.nan,
            wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum())))

    # pooled across all support sizes for this dataset
    d_all = merged["diff"].to_numpy()
    se_all = d_all.std(ddof=1) / np.sqrt(len(d_all))
    p_all = wilcoxon(d_all).pvalue if np.any(d_all != 0) else np.nan
    print(f"  {'ALL':>5}  {'':>11}  {'':>11}  {d_all.mean():+13.3f}  {se_all:7.3f}  {p_all:7.3f}"
          f"   ({(d_all>0).sum()}W/{(d_all<0).sum()}L of {len(d_all)})")

    # ---- per-mode diagnostics (NOT the headline; selection-biased if cherry-picked) ----
    for mode in ["frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"]:
        mm = paired_diff(df, CLASSICAL_REF, mode)
        if mm.empty:
            continue
        d = mm["diff"].to_numpy()
        rows_modes.append(dict(
            dataset=name, fine_tune_mode=mode, n_episodes=len(d),
            paired_diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
            wins=int((d > 0).sum()), losses=int((d < 0).sum())))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/analysis/fewshot_masking_vs_classical")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_curve, rows_paired, rows_modes = [], [], []
    for name, rel, n in RUNS:
        summarize_run(name, ROOT / rel, n, rows_curve, rows_paired, rows_modes)

    curve = pd.DataFrame(rows_curve)
    paired = pd.DataFrame(rows_paired)
    modes = pd.DataFrame(rows_modes)
    curve.to_csv(out_dir / "fewshot_learning_curves.csv", index=False)
    paired.to_csv(out_dir / "fewshot_paired_masking_vs_classical.csv", index=False)
    modes.to_csv(out_dir / "fewshot_per_mode_paired.csv", index=False)

    # ---------------- cross-dataset synthesis ----------------
    print(f"\n\n{'='*78}\n  SYNTHESIS: masking/{MASKING_MODE} vs classical/{CLASSICAL_REF}, paired\n{'='*78}")
    print(f"\n  {'dataset':<18} {'lowest support':>15} {'highest support':>16}   verdict")
    for name, _, _ in RUNS:
        p = paired[paired.dataset == name].sort_values("support_per_class")
        lo, hi = p.iloc[0], p.iloc[-1]
        allp = p.paired_diff
        verdict = ("SSL ahead throughout" if (allp > 0).all() else
                   "classical ahead throughout" if (allp < 0).all() else
                   "crossover")
        print(f"  {name:<18} {lo.paired_diff:+8.3f} @{int(lo.support_per_class):<5d} "
              f"{hi.paired_diff:+8.3f} @{int(hi.support_per_class):<5d}   {verdict}")

    print(f"\n  Per-mode pooled paired diff (diagnostic; not a selection basis):")
    piv = modes.pivot_table(index="dataset", columns="fine_tune_mode", values="paired_diff")
    order = [c for c in ["frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3"] if c in piv.columns]
    print(piv[order].round(3).to_string())

    # Does the advantage shrink with more labels? Spearman of paired diff vs support.
    from scipy.stats import spearmanr
    print(f"\n  Does the SSL-vs-classical gap depend on how many labels you have?")
    for name, _, _ in RUNS:
        p = paired[paired.dataset == name]
        if len(p) < 3:
            continue
        rho, pv = spearmanr(p.support_per_class, p.paired_diff)
        print(f"    {name:<18} spearman(support, diff) = {rho:+.2f}  p={pv:.3f}")

    print(f"\nWrote {out_dir}/fewshot_learning_curves.csv")
    print(f"Wrote {out_dir}/fewshot_paired_masking_vs_classical.csv")
    print(f"Wrote {out_dir}/fewshot_per_mode_paired.csv")


if __name__ == "__main__":
    main()
