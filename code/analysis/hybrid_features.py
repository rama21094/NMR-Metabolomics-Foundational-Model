#!/usr/bin/env python3
"""Do binned areas and the SSL embedding carry complementary information?

Motivation (docs queue #12; §4 gap decomposition). On BrC-T2D diabetes and
Barth, the frozen SSL embedding alone BEATS same-resolution binned areas
(§4b). On the other three targets classical wins outright. Neither track
dominates the other everywhere, which is the standard signal that a
concatenation might exceed both -- the failure modes may not overlap.

This concatenates each family's best-known frozen embedding with the
standard 1024-bin classical feature (binned_abs_area, StandardScaler +
L2 logistic regression -- the exact classical pipeline everywhere else in
this project) and reads it through the same CV protocol as everything else.
Masking uses flatten pooling (experiment #4's default); jigsaw/joint use
native, pending the separate pooling sweep in sweep_pooling_jigsaw_joint.py.

No retraining: this is a frozen-feature concatenation, evaluated with the
same fixed-C=1 probe as every other number in this project, so it is directly
comparable to compare_patch_sizes.py / ssl_linear_probe_eval.py output.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for sub in ("code/evaluation", "code/training", "code/analysis"):
    sys.path.insert(0, str(ROOT / sub))

from probe_logreg_advantage import DATASETS, load_generic, logreg_pipeline  # noqa: E402
from linear_probe_frozen_embeddings import CHECKPOINTS, EMBEDDERS  # noqa: E402
from brc_t2d_common import binned_abs_area  # noqa: E402

DEFAULT_POOLING = {"masking": "flatten", "jigsaw": "native", "joint": "native"}
N_BINS = 1024


def cv_scores(features, labels, n_splits, seed=42):
    n_classes = len(np.unique(labels))
    split_iter = (LeaveOneOut().split(features) if n_splits == "loo" else
                 StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(features, labels))
    oof = np.empty_like(labels)
    proba = np.zeros((len(labels), n_classes))
    for tr, te in split_iter:
        m = logreg_pipeline(seed)
        m.fit(features[tr], labels[tr])
        oof[te] = m.predict(features[te])
        p = m.predict_proba(features[te])
        for j, cls in enumerate(m.classes_):
            proba[te, cls] = p[:, j]
    bal = float(balanced_accuracy_score(labels, oof))
    try:
        auc = (float(roc_auc_score(labels, proba[:, 1])) if n_classes == 2
               else float(roc_auc_score(labels, proba, multi_class="ovr", average="macro")))
    except ValueError:
        auc = float("nan")
    return bal, auc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["brc_t2d_cancer", "brc_t2d_diabetes", "mtbls563", "mtbls326", "barth"])
    ap.add_argument("--families", nargs="+", default=["masking", "jigsaw", "joint"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/analysis/hybrid_features")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in args.datasets:
        cfg = DATASETS[name]
        if cfg["loader"] == "brc_t2d":
            from brc_t2d_common import load_brc_t2d
            spectra, labels, _, label_names = load_brc_t2d(cfg["data"], cfg["metadata"], cfg["label_column"])
        else:
            spectra, labels, label_names = load_generic(
                cfg["data"], cfg["metadata"], cfg["label_column"], cfg.get("exclude_labels", ()))
        print(f"\n=== {name}  n={len(labels)}  cv={cfg['n_splits']} ===", flush=True)

        binned = binned_abs_area(spectra, N_BINS)
        bal_b, auc_b = cv_scores(binned, labels, cfg["n_splits"], args.seed)
        print(f"  {'classical (binned only)':28s} d={binned.shape[1]:<7d} bal={bal_b:.4f} auc={auc_b:.4f}", flush=True)
        rows.append(dict(dataset=name, arm="classical_only", embed_dim=binned.shape[1],
                         balanced_accuracy=bal_b, roc_auc=auc_b, n_samples=len(labels)))

        for family in args.families:
            pooling = DEFAULT_POOLING[family]
            emb = EMBEDDERS[family](CHECKPOINTS[family], spectra, device, pooling=pooling)
            if device.type == "cuda":
                torch.cuda.empty_cache()

            bal_e, auc_e = cv_scores(emb, labels, cfg["n_splits"], args.seed)
            rows.append(dict(dataset=name, arm=f"{family}_only", embed_dim=emb.shape[1],
                             balanced_accuracy=bal_e, roc_auc=auc_e, n_samples=len(labels)))
            print(f"  {family + '_only':28s} d={emb.shape[1]:<7d} bal={bal_e:.4f} auc={auc_e:.4f}", flush=True)

            hybrid = np.concatenate([emb, binned], axis=1)
            bal_h, auc_h = cv_scores(hybrid, labels, cfg["n_splits"], args.seed)
            rows.append(dict(dataset=name, arm=f"{family}_plus_binned", embed_dim=hybrid.shape[1],
                             balanced_accuracy=bal_h, roc_auc=auc_h, n_samples=len(labels)))
            best_solo = max(bal_e, bal_b)
            print(f"  {family + '_plus_binned':28s} d={hybrid.shape[1]:<7d} bal={bal_h:.4f} auc={auc_h:.4f}"
                  f"   vs best solo ({best_solo:.4f}): {bal_h - best_solo:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hybrid_features_results.csv", index=False)
    print(f"\nWrote {out_dir / 'hybrid_features_results.csv'}\n")

    piv = df.pivot_table(index="dataset", columns="arm", values="balanced_accuracy")
    print(piv.round(4).to_string())
    print()
    for family in ["masking", "jigsaw", "joint"]:
        solo_col, hybrid_col = f"{family}_only", f"{family}_plus_binned"
        if solo_col not in piv.columns:
            continue
        best_solo = piv[[solo_col, "classical_only"]].max(axis=1)
        delta = piv[hybrid_col] - best_solo
        wins = int((delta > 0).sum())
        print(f"{family}: hybrid beats max(solo SSL, classical) on {wins}/{len(delta)} targets, "
              f"mean delta {delta.mean():+.4f}")


if __name__ == "__main__":
    main()
