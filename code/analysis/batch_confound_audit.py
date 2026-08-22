#!/usr/bin/env python3
"""Experiment #11: run-order / batch-confound audit of every evaluation cohort.

WHY. MTBLS326 reached a perfect 1.000 balanced accuracy on n=42 and it turned
out to be confounded by acquisition batch (see §11). That immediately
invalidates the assumption behind every other target: none of them had been
checked either. This script audits all four cohorts on the same footing so
"which targets are admissible evidence" stops being a guess.

WHY A PERMUTATION NULL IS NOT ENOUGH. §6 reports label-permutation nulls at
p <= 0.02 for every target. Those cannot detect a batch effect: permuting the
labels destroys the batch structure and the biological signal simultaneously, so
a fully confounded dataset passes. A confound has to be attacked directly.

THREE TESTS PER COHORT
----------------------
1. DESIGN AUDIT. Recover an acquisition-order proxy from the sample identifiers
   and ask how well it alone predicts the label (one-feature AUC, plus
   Mann-Whitney U). AUC ~= 1.0 means cases and controls were run in separate
   blocks and biology is not separable from batch by any method. AUC ~= 0.5
   means they were interleaved, which is what a sound design looks like.

2. SIGNAL-FREE CLASSIFICATION. Classify the label using ONLY spectral regions
   with no metabolite resonances -- above 9.5 ppm and below -0.5 ppm in serum
   CPMG. A biological difference cannot appear there, so accuracy above the
   permutation null means a technical difference tracks the label.

3. CONTROL FOR TEST 2. Try to predict early-vs-late acquisition *within* the
   majority class from the same noise features. If test 2 is positive and this
   is negative, the noise separates discrete BLOCKS (separate sessions) rather
   than encoding smooth drift.

Also reports any categorical metadata variable that is associated with the label
(Fisher/chi-square), which is how Barth's anticoagulant split was found.

Both the rowMinMax array (what the reported evaluation consumed) and the
un-normalised array are audited: per-row min-max rescaling converts an absolute
intensity offset into a noise-to-peak ratio, i.e. an SNR feature, and on
MTBLS326 that made the artifact MORE learnable rather than less.

LIMITATION. The definitive run-order signal is the `##$DATE` stamp in each
sample's Bruker `acqus` file. No raw archive is present on this machine, so
every cohort uses an identifier-derived proxy, documented per cohort below. A
proxy can only ever under-detect: if it already shows separation, that is real.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

PPM = "data/mtbls326/MTBLS326_common_ppm_axis.npy"   # identical for all cohorts
NOISE_HI_PPM, NOISE_LO_PPM = 9.5, -0.5
SIGNAL_LO, SIGNAL_HI = 0.5, 9.5


def _barth():
    meta = pd.read_csv(ROOT / "data/Barth/Workbench_Barth_Syndrome_metadata.csv")
    meta["_row"] = np.arange(len(meta))
    meta = meta[meta.label != "Pool"].copy()          # pooled QC, excluded by the real run
    # SVCM#### accession number. Not a timestamp, but samples were accessioned
    # in collection order, so it is the best available ordering proxy.
    meta["order"] = meta.sample_folder.str.extract(r"(\d+)").astype(int)
    return dict(
        name="Barth", rows=meta._row.to_numpy(),
        y=(meta.label == "Case").to_numpy().astype(int), order=meta.order.to_numpy(),
        order_desc="SVCM accession number", cv="loo",
        categorical={"Anticoagulant": meta.Anticoagulant.to_numpy()},
        arrays={
            "rowMinMax_v4 (as evaluated)":
                "data/Barth/aligned_128K_Workbench_Barth_Syndrome_"
                "WS625to680Zero_EDTASuppressed_rowMinMax_v4.npy",
            "un-normalised v4":
                "data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_v4.npy",
        })


def _mtbls326():
    meta = pd.read_csv(ROOT / "data/mtbls326/MTBLS326_metadata_mapping.csv")
    meta = meta.sort_values("npy_row").reset_index(drop=True)
    meta["_row"] = meta.npy_row
    # trailing integer of folder_name is the original experiment number
    meta["order"] = meta.folder_name.str.extract(r"_(\d+)$").astype(int)
    return dict(
        name="MTBLS326", rows=meta._row.to_numpy(),
        y=(meta.label == "Yes").to_numpy().astype(int), order=meta.order.to_numpy(),
        order_desc="original experiment number", cv="loo", categorical={},
        arrays={
            "rowMinMax_v4 (as evaluated)":
                "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy",
            "un-normalised v4":
                "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_v4.npy",
        })


def _mtbls563():
    meta = pd.read_csv(ROOT / "data/mtbls563/MTBLS563_metadata_mapping.csv")
    meta = meta.sort_values("npy_row").reset_index(drop=True)
    meta["_row"] = meta.npy_row
    dx = meta["Factor Value[Diagnosis]"]
    keep = dx != "unknown"                              # the reported 3-class task, n=113
    meta = meta[keep].copy()
    meta["order"] = meta.folder_name.str.extract(r"_(\d+)$").astype(int)
    codes, _ = pd.factorize(meta["Factor Value[Diagnosis]"])
    return dict(
        name="MTBLS563", rows=meta._row.to_numpy(), y=codes.astype(int),
        order=meta.order.to_numpy(), order_desc="original experiment number",
        cv=10, categorical={},
        arrays={
            "rowMinMax_v4 (as evaluated)":
                "data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy",
            "un-normalised v4":
                "data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_v4.npy",
        })


def _brc(label_col):
    meta = pd.read_csv(ROOT / "data/BrC_T2D/BC_T2D_newlabels_metadata_mapping.csv")
    meta = meta.sort_values("npy_row").reset_index(drop=True)
    meta["_row"] = meta.npy_row
    # SM## sample code. NOTE: this metadata file also contains patient names; we
    # deliberately read only the ID and label columns so no identifying
    # information can reach any output of this script.
    meta["order"] = meta.ID.str.extract(r"(\d+)").astype(int)
    pos = "Cancer" if label_col == "cancer_status" else "Diabetes"
    return dict(
        name=f"BrC-T2D {label_col.replace('_status','')}", rows=meta._row.to_numpy(),
        y=(meta[label_col] == pos).to_numpy().astype(int), order=meta.order.to_numpy(),
        order_desc="SM sample code", cv=10, categorical={},
        arrays={
            "rowMinMax_v4 (as evaluated)":
                "data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_rowMinMax_v4.npy",
            "un-normalised v4":
                "data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_v4.npy",
        })


COHORTS = [_barth, _mtbls326, _mtbls563,
           lambda: _brc("cancer_status"), lambda: _brc("diabetes_status")]


def pipe(seed):
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=5000, C=1.0,
                                                  class_weight="balanced",
                                                  random_state=seed))])


def cv_bal_acc(X, y, cv, seed=42):
    if cv == "loo":
        splits = LeaveOneOut().split(X)
    else:
        splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed).split(X, y)
    oof = np.empty_like(y)
    for tr, te in splits:
        oof[te] = pipe(seed).fit(X[tr], y[tr]).predict(X[te])
    return float(balanced_accuracy_score(y, oof))


def perm_null(X, y, cv, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([cv_bal_acc(X, rng.permutation(y), cv, seed) for _ in range(n_perm)])


def binned_abs_area(spectra, mask, n_bins):
    sub = np.abs(np.asarray(spectra[:, mask], dtype=np.float64))
    cut = (sub.shape[1] // n_bins) * n_bins
    return sub[:, :cut].reshape(sub.shape[0], n_bins, -1).mean(axis=2)


def order_auc(order, y):
    """How well does acquisition order alone predict the label?

    Multiclass is scored one-vs-rest and reported as the maximum over classes,
    i.e. the worst case: 'is ANY class separated in acquisition order?'
    """
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, order)
        u = mannwhitneyu(order[y == 1], order[y == 0]).pvalue
        return max(auc, 1 - auc), float(u)
    aucs, ps = [], []
    for c in np.unique(y):
        b = (y == c).astype(int)
        a = roc_auc_score(b, order)
        aucs.append(max(a, 1 - a))
        ps.append(mannwhitneyu(order[b == 1], order[b == 0]).pvalue)
    return float(max(aucs)), float(min(ps))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/analysis/batch_confound_audit")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm = np.load(ROOT / PPM)
    noise_mask = (ppm > NOISE_HI_PPM) | (ppm < NOISE_LO_PPM)
    signal_mask = (ppm >= SIGNAL_LO) & (ppm <= SIGNAL_HI)
    print(f"signal-free regions: ppm > {NOISE_HI_PPM} or < {NOISE_LO_PPM}  "
          f"({noise_mask.sum()} pts, {100*noise_mask.mean():.1f}% of spectrum)\n")

    design_rows, cls_rows, cat_rows = [], [], []
    for make in COHORTS:
        c = make()
        y, order, rows = c["y"], c["order"], c["rows"]
        cvname = "LOOCV" if c["cv"] == "loo" else f"{c['cv']}-fold"
        print("=" * 78)
        print(f"  {c['name']}   n={len(y)}   classes={len(np.unique(y))}   cv={cvname}")
        print("=" * 78)

        # ---- test 1: design
        auc, mwp = order_auc(order.astype(float), y)
        disjoint = False
        if len(np.unique(y)) == 2:
            a, b = set(order[y == 1]), set(order[y == 0])
            disjoint = (not (a & b)) and (max(a) < min(b) or max(b) < min(a))
        verdict = ("CONFOUNDED BY DESIGN" if auc > 0.95 or disjoint else
                   "suspicious" if auc > 0.75 else "interleaved (ok)")
        print(f"  TEST 1  order proxy = {c['order_desc']}")
        print(f"          order-alone AUC = {auc:.3f}   Mann-Whitney p = {mwp:.4f}"
              f"   disjoint blocks = {disjoint}")
        print(f"          -> {verdict}")
        design_rows.append(dict(cohort=c["name"], n=len(y), order_proxy=c["order_desc"],
                                order_alone_auc=auc, mannwhitney_p=mwp,
                                disjoint_blocks=disjoint, design_verdict=verdict))

        # ---- categorical metadata associations
        for k, v in c["categorical"].items():
            tab = pd.crosstab(pd.Series(v, name=k), pd.Series(y, name="label"))
            if tab.shape == (2, 2):
                orr, p = fisher_exact(tab.to_numpy())
                print(f"  META    {k}: Fisher p = {p:.4f}  (odds ratio {orr:.2f})")
                cat_rows.append(dict(cohort=c["name"], variable=k, test="fisher_exact",
                                     p_value=float(p), odds_ratio=float(orr),
                                     table=tab.to_dict()))

        # ---- test 2 + 3
        for arr_label, rel in c["arrays"].items():
            path = ROOT / rel
            if not path.exists():
                print(f"  TEST 2  {arr_label}: MISSING {rel}")
                continue
            X = np.load(path, mmap_mode="r")[rows]
            for fname, mask in (("noise regions only", noise_mask),
                                ("signal region only", signal_mask)):
                F = binned_abs_area(X, mask, args.n_bins)
                acc = cv_bal_acc(F, y, c["cv"], args.seed)
                null = perm_null(F, y, c["cv"], args.n_perm, args.seed)
                p = float((null >= acc).mean())
                flag = "  <-- TECHNICAL SIGNAL" if (fname.startswith("noise") and p < 0.05) else ""
                print(f"  TEST 2  [{arr_label}] {fname:<20} acc={acc:.4f}  "
                      f"nullp95={np.percentile(null,95):.3f}  p={p:.4f}{flag}")
                cls_rows.append(dict(cohort=c["name"], array=arr_label, features=fname,
                                     balanced_accuracy=acc, p_value=p,
                                     null_p95=float(np.percentile(null, 95))))
            # control: order within the largest class
            big = np.bincount(y).argmax()
            sel = np.flatnonzero(y == big)
            if len(sel) >= 12:
                o = order[sel]
                yo = (o > np.median(o)).astype(int)
                if len(np.unique(yo)) == 2 and min(np.bincount(yo)) >= 4:
                    F = binned_abs_area(X, noise_mask, args.n_bins)[sel]
                    acc = cv_bal_acc(F, yo, "loo", args.seed)
                    null = perm_null(F, yo, "loo", args.n_perm, args.seed)
                    p = float((null >= acc).mean())
                    print(f"  TEST 3  [{arr_label}] early-vs-late within class {big}: "
                          f"acc={acc:.4f}  p={p:.4f}  (n={len(sel)})")
                    cls_rows.append(dict(cohort=c["name"], array=arr_label,
                                         features=f"[control] order within class {big}",
                                         balanced_accuracy=acc, p_value=p,
                                         null_p95=float(np.percentile(null, 95))))
        print()

    design = pd.DataFrame(design_rows)
    cls = pd.DataFrame(cls_rows)
    design.to_csv(out_dir / "design_audit.csv", index=False)
    cls.to_csv(out_dir / "signal_free_classification.csv", index=False)
    if cat_rows:
        pd.DataFrame(cat_rows).to_csv(out_dir / "categorical_associations.csv", index=False)

    # ---------------- summary ----------------
    # Ten noise-only tests were run (5 cohorts x 2 arrays), so raw p < 0.05 would
    # be expected to fire once by chance. Holm-Bonferroni across exactly that
    # family, so a single marginal hit is not reported as a confound.
    noise = cls[cls.features == "noise regions only"].copy().reset_index(drop=True)
    m = len(noise)
    order_idx = noise.p_value.sort_values().index
    holm, running = {}, 0.0
    for rank, idx in enumerate(order_idx):
        adj = min(1.0, max(running, noise.loc[idx, "p_value"] * (m - rank)))
        running = adj
        holm[idx] = adj
    noise["p_holm"] = [holm[i] for i in noise.index]
    noise.to_csv(out_dir / "noise_tests_holm.csv", index=False)

    print("=" * 78)
    print("  SUMMARY -- which cohorts are admissible evidence?")
    print("=" * 78)
    print(f"  Holm-Bonferroni over the m={m} signal-free tests.\n")
    print(f"  {'cohort':<20} {'ordAUC':>7} {'design':<22} {'worst noise test':<26} {'verdict'}")
    summary = []
    for _, d in design.iterrows():
        g = noise[noise.cohort == d.cohort]
        design_bad = d.design_verdict.startswith("CONFOUNDED")
        if len(g):
            best = g.loc[g.p_value.idxmin()]
            tech_sig = bool(best.p_holm < 0.05)
            # A rowMinMax-only hit is ambiguous: per-row min-max makes the
            # "noise" bins encode noise-relative-to-peak-height, i.e. an SNR
            # feature, which also moves with overall metabolite concentration.
            # An un-normalised hit is much closer to purely technical.
            rowminmax_only = tech_sig and "rowMinMax" in str(best["array"])
            desc = (f"{best.balanced_accuracy:.3f} p={best.p_value:.3f} "
                    f"holm={best.p_holm:.3f}")
        else:
            tech_sig, rowminmax_only, desc = False, False, "n/a"
        if design_bad and tech_sig:
            v = "CONFOUNDED"
        elif design_bad:
            v = "CONFOUNDED (design)"
        elif tech_sig and rowminmax_only:
            v = "AMBIGUOUS (SNR)"
        elif tech_sig:
            v = "TECHNICAL SIGNAL"
        elif d.design_verdict == "suspicious":
            v = "caveat (order)"
        else:
            v = "clean"
        print(f"  {d.cohort:<20} {d.order_alone_auc:7.3f} {d.design_verdict:<22} "
              f"{desc:<26} {v}")
        summary.append(dict(cohort=d.cohort, order_alone_auc=d.order_alone_auc,
                            design_verdict=d.design_verdict,
                            worst_noise_test=desc, verdict=v))
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)

    print("\n  KEY")
    print("   CONFOUNDED         cases/controls in separate acquisition blocks AND")
    print("                      metabolite-free regions predict the label. Not usable.")
    print("   TECHNICAL SIGNAL   metabolite-free regions predict the label on the")
    print("                      un-normalised array -- a real technical difference.")
    print("   AMBIGUOUS (SNR)    metabolite-free regions predict the label only after")
    print("                      rowMinMax. Those bins then encode noise-to-peak ratio,")
    print("                      which also tracks overall metabolite concentration, so")
    print("                      this may be biological. Needs a normalisation-invariant")
    print("                      re-check before it is called an artifact.")
    print("   caveat (order)     no technical signal detected, but acquisition order")
    print("                      partly tracks the label -- design is not fully balanced.")
    print("   clean              order interleaved and no signal outside the metabolites.")
    print(f"\nWrote {out_dir}/design_audit.csv")
    print(f"Wrote {out_dir}/signal_free_classification.csv")
    print(f"Wrote {out_dir}/noise_tests_holm.csv")
    print(f"Wrote {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
