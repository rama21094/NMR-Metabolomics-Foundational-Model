#!/usr/bin/env python3
"""Experiment #11: is MTBLS326's perfect 1.000 biology or a run-order artifact?

WHY THIS MATTERS. MTBLS326 is the only target where classical LogReg reaches a
perfect 1.000 balanced accuracy (n=42, LOOCV), and it is one of only two targets
where SSL is even competitive. A perfect score on 42 samples is exactly the
shape of result that turns out to be a batch effect, and §6's permutation null
CANNOT detect one: permuting labels destroys the batch structure and the
biological signal together, so a confounded dataset still passes it.

WHAT A RUN-ORDER CONFOUND LOOKS LIKE. If all cases were acquired in one session
and all controls in another, then instrument drift, shim quality, tube batch,
storage time and reagent lot all correlate perfectly with the label. The
classifier can then be reading technical drift rather than metabolism, and no
amount of cross-validation will reveal it, because the confound is in the design
rather than in the fitting.

THE TWO TESTS HERE
------------------
1. DESIGN AUDIT. Read the acquisition ordering out of the sample identifiers and
   check whether cases and controls occupy separate blocks. This is a property
   of the experiment, not a statistic, and it either is or is not confounded.

2. SIGNAL-FREE CLASSIFICATION. Classify the labels using ONLY spectral regions
   that contain no metabolite resonances -- above 9.5 ppm and below -0.5 ppm,
   which in serum CPMG hold nothing but noise and baseline. A biological
   difference cannot be visible there. If those regions alone separate the
   classes, the separation is technical. This is the test that a permutation
   null cannot substitute for.

   Control for test 2: try to predict EARLY vs LATE acquisition *within* the
   cancer group from the same noise features. That measures how much
   acquisition-order information the noise carries at all, independent of the
   label, and stops us over-reading a positive result in test 2.

Both the rowMinMax array (what the reported evaluation used) and the
un-normalised array are audited, because per-row min-max rescaling removes
absolute intensity differences and could mask a technical offset.

NOTE ON WHAT IS NOT AVAILABLE. The definitive run-order signal is the `##$DATE`
stamp in each sample's Bruker `acqus` file. The raw MTBLS326 folders are not
present on this machine (`folder_path` points at MetabolightsCPMGSingleFolder,
not retained), so the sample-number ordering is used as the proxy. If the raw
archive is restored, prefer the timestamps and re-run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

META = "data/mtbls326/MTBLS326_metadata_mapping.csv"
PPM = "data/mtbls326/MTBLS326_common_ppm_axis.npy"
ARRAYS = {
    "rowMinMax_v4 (as evaluated)":
        "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy",
    "un-normalised v4":
        "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_v4.npy",
}

# Serum CPMG has no metabolite resonances outside roughly -0.5..9.5 ppm. TSP/DSS
# sits at 0.0, the amide/aromatic envelope ends by ~9.0.
NOISE_HI_PPM = 9.5      # above this: noise only
NOISE_LO_PPM = -0.5     # below this: noise only
SIGNAL_LO, SIGNAL_HI = 0.5, 9.5


def pipeline(seed: int) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=5000, C=1.0,
                                                  class_weight="balanced",
                                                  random_state=seed))])


def loocv_bal_acc(X: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    oof = np.empty_like(y)
    for tr, te in LeaveOneOut().split(X):
        m = pipeline(seed).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return float(balanced_accuracy_score(y, oof))


def perm_null(X, y, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([loocv_bal_acc(X, rng.permutation(y)) for _ in range(n_perm)])


def binned_abs_area(spectra: np.ndarray, mask: np.ndarray, n_bins: int) -> np.ndarray:
    sub = np.abs(np.asarray(spectra[:, mask], dtype=np.float64))
    cut = (sub.shape[1] // n_bins) * n_bins
    return sub[:, :cut].reshape(sub.shape[0], n_bins, -1).mean(axis=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/analysis/mtbls326_batch_audit")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(ROOT / META).sort_values("npy_row").reset_index(drop=True)
    # trailing integer of folder_name is the original sample/experiment number
    meta["sample_no"] = meta.folder_name.str.extract(r"_(\d+)$").astype(int)
    y = (meta.label == "Yes").to_numpy().astype(int)

    # ---------------- TEST 1: design audit ----------------
    print("=" * 78)
    print("  TEST 1 -- DESIGN AUDIT (acquisition ordering vs label)")
    print("=" * 78)
    rows = []
    for lab, g in meta.groupby("label"):
        rows.append(dict(label=lab, n=len(g), sample_no_min=g.sample_no.min(),
                         sample_no_max=g.sample_no.max(),
                         npy_row_min=g.npy_row.min(), npy_row_max=g.npy_row.max()))
        print(f"  label={lab:<4} n={len(g):3d}  sample_no {g.sample_no.min():4d}..{g.sample_no.max():4d}"
              f"   npy_row {g.npy_row.min():3d}..{g.npy_row.max():3d}")
    design = pd.DataFrame(rows)
    design.to_csv(out_dir / "design_audit.csv", index=False)

    a = set(meta[meta.label == "Yes"].sample_no)
    b = set(meta[meta.label == "No"].sample_no)
    overlap = a & b
    interleaved = not (max(a) < min(b) or max(b) < min(a))
    print(f"\n  cases  : {sorted(a)}")
    print(f"  controls: {sorted(b)}")
    print(f"\n  shared sample numbers: {len(overlap)}")
    print(f"  ranges interleaved   : {interleaved}")
    confounded = (len(overlap) == 0) and not interleaved
    if confounded:
        print("\n  *** CONFOUNDED BY DESIGN ***")
        print("  Cases and controls occupy DISJOINT, CONTIGUOUS blocks of the acquisition")
        print("  order. Sample number alone separates the classes perfectly, so any")
        print("  technical variable that drifts with run order is a perfect predictor of")
        print("  the label. Biology and batch are not statistically separable in this")
        print("  dataset, by any method, at any sample size.")
    else:
        print("\n  Cases and controls are interleaved in acquisition order -- no design confound.")

    # ---------------- TEST 2: signal-free classification ----------------
    ppm = np.load(ROOT / PPM)
    noise_mask = (ppm > NOISE_HI_PPM) | (ppm < NOISE_LO_PPM)
    signal_mask = (ppm >= SIGNAL_LO) & (ppm <= SIGNAL_HI)
    print("\n" + "=" * 78)
    print("  TEST 2 -- CLASSIFY FROM SIGNAL-FREE REGIONS ONLY")
    print("=" * 78)
    print(f"  noise regions : ppm > {NOISE_HI_PPM} or ppm < {NOISE_LO_PPM}"
          f"  ({noise_mask.sum()} points, {100*noise_mask.mean():.1f}% of spectrum)")
    print(f"  signal region : {SIGNAL_LO}..{SIGNAL_HI} ppm  ({signal_mask.sum()} points)")

    results = []
    for arr_label, rel in ARRAYS.items():
        path = ROOT / rel
        if not path.exists():
            print(f"\n  {arr_label}: MISSING {rel}")
            continue
        X = np.load(path, mmap_mode="r")
        assert X.shape[0] == len(meta), f"{X.shape[0]} spectra vs {len(meta)} metadata rows"
        print(f"\n  --- {arr_label} ---")

        feats = {
            "noise regions only": binned_abs_area(X, noise_mask, args.n_bins),
            "signal region only": binned_abs_area(X, signal_mask, args.n_bins),
        }
        for fname, F in feats.items():
            acc = loocv_bal_acc(F, y, args.seed)
            null = perm_null(F, y, args.n_perm, args.seed)
            p = float((null >= acc).mean())
            print(f"    {fname:<20} bal.acc = {acc:.4f}   perm-null p95 = "
                  f"{np.percentile(null, 95):.3f}  p = {p:.4f}")
            results.append(dict(array=arr_label, features=fname, n_bins=args.n_bins,
                                balanced_accuracy=acc, null_p95=float(np.percentile(null, 95)),
                                null_max=float(null.max()), p_value=p))

        # control: predict EARLY vs LATE acquisition within the cancer block
        can = np.flatnonzero(y == 1)
        med = np.median(meta.sample_no.to_numpy()[can])
        y_order = (meta.sample_no.to_numpy()[can] > med).astype(int)
        Fn = binned_abs_area(X, noise_mask, args.n_bins)[can]
        acc_o = loocv_bal_acc(Fn, y_order, args.seed)
        null_o = perm_null(Fn, y_order, args.n_perm, args.seed)
        p_o = float((null_o >= acc_o).mean())
        print(f"    {'[control] early vs late WITHIN cases, from noise':<20}")
        print(f"      bal.acc = {acc_o:.4f}   p = {p_o:.4f}   (n={len(can)})")
        results.append(dict(array=arr_label, features="[control] acq-order within cases (noise)",
                            n_bins=args.n_bins, balanced_accuracy=acc_o,
                            null_p95=float(np.percentile(null_o, 95)),
                            null_max=float(null_o.max()), p_value=p_o))

    res = pd.DataFrame(results)
    res.to_csv(out_dir / "signal_free_classification.csv", index=False)

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    noise_rows = res[res.features == "noise regions only"]
    ctrl_rows = res[res.features.str.startswith("[control]")]
    # Significance against the label-permutation null is the right criterion here,
    # not an accuracy threshold: with n=42 an honest technical effect need not be
    # large to be real, and 0.73 at p<1e-3 is a stronger finding than 0.80 at p=0.1.
    sig = noise_rows[noise_rows.p_value < 0.05]
    print(f"  Design confounded by construction : {confounded}")
    for _, r in noise_rows.iterrows():
        print(f"  signal-free label accuracy [{r['array']}] : "
              f"{r.balanced_accuracy:.4f}  (p = {r.p_value:.4f})")
    if confounded:
        print("\n  1. MTBLS326's 1.000 cannot be attributed to biology. Cases and controls")
        print("     were acquired in separate, non-overlapping blocks, so instrument drift")
        print("     and every other run-order variable are perfectly collinear with the")
        print("     label. The permutation null in §6 does not rule this out -- permuting")
        print("     labels breaks the batch structure along with the biology.")
    if len(sig):
        best = sig.loc[sig.balanced_accuracy.idxmax()]
        print(f"\n  2. And the confound is REAL, not merely possible. Spectral regions that")
        print(f"     contain NO metabolite resonances classify the label at")
        print(f"     {best.balanced_accuracy:.3f} (p = {best.p_value:.4f}, null p95 = "
              f"{best.null_p95:.3f}) on the")
        print(f"     '{best['array']}' array. Biology cannot be visible there, so a technical")
        print("     difference tracks the label.")
        if len(ctrl_rows) and (ctrl_rows.p_value >= 0.05).all():
            print("\n  3. The control is informative: the same noise features CANNOT predict")
            print("     early-vs-late acquisition *within* the case block (p >= 0.05). So the")
            print("     noise is not encoding a smooth run-order drift -- it separates the two")
            print("     BLOCKS specifically, which is the signature of two distinct acquisition")
            print("     sessions rather than gradual instrument drift.")
        print("\n  ACTION: MTBLS326 should be reported as confounded and dropped from any")
        print("  headline claim, including 'the one target where SSL is competitive'. Its")
        print("  1.000 is not evidence about metabolomics or about representations.")
    else:
        print("\n  2. No technical difference is detectable in the signal-free regions")
        print("     (p >= 0.05 everywhere). That weakens but does not remove the concern:")
        print("     the design confound in test 1 stands regardless, and drift inside the")
        print("     metabolite region would not appear in these bins.")
    print(f"\nWrote {out_dir}/design_audit.csv")
    print(f"Wrote {out_dir}/signal_free_classification.csv")


if __name__ == "__main__":
    main()
