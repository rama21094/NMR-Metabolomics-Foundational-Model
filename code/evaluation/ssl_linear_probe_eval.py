#!/usr/bin/env python3
"""Experiment #5: production evaluator for a regularized linear probe on frozen
SSL embeddings.

Motivation (see docs/SSL_vs_classical_analysis.md §4b). Every SSL family ends in
`pooled embedding -> LayerNorm -> Dropout -> Linear`, fitted by Adam for ~50
epochs with early stopping on a few dozen samples. Fitting that same linear map
as a converged L2 logistic regression instead gains, on identical frozen
features, +0.120 balanced accuracy on average for the *masking* family across
all five targets (+0.077..+0.156). jigsaw and joint are unaffected.

This script makes that a first-class, reproducible evaluation rather than an
ad-hoc probe, with two improvements over the diagnostic version:

  1. The L2 strength C is selected by NESTED cross-validation inside each outer
     training fold, never on test data. The earlier probe simply inherited
     C=1.0 from the classical pipeline, which is not "properly regularized" so
     much as "arbitrary".
  2. Outputs match the other evaluators (summary.csv, oof_predictions.csv,
     fold_metrics.csv, run_config.json) via the shared aggregate_metrics, so the
     existing comparison/plotting tooling reads them without special-casing.

The probe is reported as an ADDITIONAL model per family, not a replacement for
the fine-tuned heads. That matters: on MTBLS326 the fine-tuned masking head
(0.981) genuinely beats the frozen probe (0.944), so silently swapping heads
would lose accuracy there. Keeping both lets the per-dataset choice be explicit.

A note on why computing embeddings for all samples before splitting is NOT
leakage here: the backbone is frozen and was pretrained on a separate
9,670-spectrum corpus, so the transform is fixed and label-independent. Only
StandardScaler and LogisticRegression see fold data, and both are fitted inside
the pipeline on training folds only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for sub in ("code/evaluation", "code/training", "code/analysis"):
    sys.path.insert(0, str(ROOT / sub))

from brc_t2d_common import aggregate_metrics  # noqa: E402
from linear_probe_frozen_embeddings import CHECKPOINTS, EMBEDDERS  # noqa: E402
from probe_logreg_advantage import DATASETS, load_generic  # noqa: E402

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def build_probe(seed: int, tune_c: bool, inner_splits: int):
    """Pipeline with optional nested-CV selection of the L2 strength."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, C=1.0, random_state=seed,
                                     class_weight="balanced")),
    ])
    if not tune_c:
        return pipe
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, {"model__C": C_GRID}, scoring="balanced_accuracy",
                        cv=inner, n_jobs=-1, refit=True)


def safe_inner_splits(y_train: np.ndarray, requested: int) -> int:
    """Inner CV cannot have more folds than the rarest class has members."""
    counts = np.bincount(y_train)
    counts = counts[counts > 0]
    return int(max(2, min(requested, counts.min())))


def run_cv(features, labels, n_splits, seed, tune_c, inner_splits, n_classes):
    if n_splits == "loo":
        split_iter = list(LeaveOneOut().split(features))
        fold_ids = None
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = list(splitter.split(features, labels))
        fold_ids = np.empty(len(labels), dtype=int)

    oof_pred = np.empty(len(labels), dtype=np.int64)
    oof_prob = np.zeros((len(labels), n_classes), dtype=float)
    chosen_c = []

    for fold, (tr, te) in enumerate(split_iter, start=1):
        est = build_probe(seed, tune_c, safe_inner_splits(labels[tr], inner_splits))
        est.fit(features[tr], labels[tr])
        if tune_c:
            chosen_c.append(float(est.best_params_["model__C"]))
        final = est.best_estimator_ if tune_c else est
        oof_pred[te] = final.predict(features[te])
        p = final.predict_proba(features[te])
        for j, cls in enumerate(final.classes_):
            oof_prob[te, cls] = p[:, j]
        if fold_ids is not None:
            fold_ids[te] = fold

    return oof_pred, oof_prob, chosen_c, fold_ids, split_iter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+",
                        default=["brc_t2d_cancer", "brc_t2d_diabetes", "mtbls563", "mtbls326", "barth"])
    parser.add_argument("--families", nargs="+", default=["masking", "jigsaw", "joint"])
    parser.add_argument("--tune-C", dest="tune_c", action="store_true", default=True,
                        help="select L2 strength by nested CV on training folds (default)")
    parser.add_argument("--no-tune-C", dest="tune_c", action="store_false",
                        help="use fixed C=1.0, matching the original diagnostic probe")
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default="results/linear_probe/ssl_linear_probe_v4")
    parser.add_argument(
        "--random-backbone", action="store_true",
        help="EXPERIMENT #3 CONTROL: keep each architecture but load no pretrained "
             "weights at all, so the probe reads a genuinely untrained backbone. "
             "Unlike --reinit-unfrozen-xavier in the fine-tuning scripts (which "
             "resets only the layers being fine-tuned and always keeps the patch "
             "embedding and positional encoding pretrained), this resets "
             "everything. Comparing a run with and without this flag, with the "
             "head held fixed at a converged linear probe, measures what "
             "pretraining actually contributes to the representation.")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for name in args.datasets:
        cfg = DATASETS[name]
        if cfg["loader"] == "brc_t2d":
            from brc_t2d_common import load_brc_t2d
            spectra, labels, meta, label_names = load_brc_t2d(cfg["data"], cfg["metadata"], cfg["label_column"])
        else:
            spectra, labels, label_names = load_generic(
                cfg["data"], cfg["metadata"], cfg["label_column"], cfg.get("exclude_labels", ()))
            meta = None
        n_classes = len(label_names)
        n_splits = cfg["n_splits"]
        ds_dir = out_root / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name}  n={len(labels)}  classes={label_names}  cv={n_splits} ===", flush=True)

        summary_rows, fold_rows, oof_cols = [], [], {}

        for family in args.families:
            emb = EMBEDDERS[family](CHECKPOINTS[family], spectra, device,
                                    random_init=args.random_backbone, seed=args.seed)
            if device.type == "cuda":
                torch.cuda.empty_cache()

            oof_pred, oof_prob, chosen_c, fold_ids, split_iter = run_cv(
                emb, labels, n_splits, args.seed, args.tune_c, args.inner_splits, n_classes)

            metrics = aggregate_metrics(labels, oof_pred, oof_prob, label_names)
            model_name = (f"{family}_linear_probe"
                          + ("_tunedC" if args.tune_c else "_C1")
                          + ("_randominit" if args.random_backbone else ""))
            row = dict(family=family, model=model_name, n_evaluated=len(labels), **metrics)
            if chosen_c:
                row["C_median"] = float(np.median(chosen_c))
                row["C_values"] = ";".join(f"{c:g}" for c in chosen_c)
            row["embed_dim"] = int(emb.shape[1])
            summary_rows.append(row)
            all_rows.append(dict(dataset=name, **row))

            # Per-fold metrics, where the protocol has folds to report.
            if fold_ids is not None:
                for fold in np.unique(fold_ids):
                    m = fold_ids == fold
                    try:
                        fm = aggregate_metrics(labels[m], oof_pred[m], oof_prob[m], label_names)
                    except ValueError:
                        continue
                    fold_rows.append(dict(family=family, model=model_name, fold=int(fold),
                                          n_test=int(m.sum()), **fm))

            oof_cols[f"{model_name}_prediction"] = [label_names[i] for i in oof_pred]
            for ci, cname in enumerate(label_names):
                key = f"{model_name}_prob_{cname.replace(' ', '_').replace('(', '').replace(')', '')}"
                oof_cols[key] = oof_prob[:, ci]

            cstr = f" C_median={np.median(chosen_c):g}" if chosen_c else ""
            print(f"  {family:8s} d={emb.shape[1]:<5d} bal_acc={metrics['balanced_accuracy']:.4f}{cstr}", flush=True)

        pd.DataFrame(summary_rows).to_csv(ds_dir / "summary.csv", index=False)
        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(ds_dir / "fold_metrics.csv", index=False)
        oof = pd.DataFrame({
            "row_index": np.arange(len(labels)),
            "label": [label_names[i] for i in labels],
            "target": labels,
            **oof_cols,
        })
        if meta is not None:
            oof.insert(1, "sample_id", [m.get("ID", "") for m in meta])
        oof.to_csv(ds_dir / "oof_predictions.csv", index=False)
        with (ds_dir / "run_config.json").open("w") as fh:
            json.dump(dict(dataset=name, data=cfg["data"], metadata=cfg["metadata"],
                           random_backbone=args.random_backbone,
                           label_column=cfg["label_column"], cv=str(n_splits),
                           families=args.families, checkpoints=CHECKPOINTS,
                           tune_C=args.tune_c, C_grid=C_GRID if args.tune_c else [1.0],
                           inner_splits=args.inner_splits, seed=args.seed), fh, indent=2)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_root / "all_datasets_summary.csv", index=False)
    print(f"\nWrote {out_root / 'all_datasets_summary.csv'}\n")
    cols = ["dataset", "family", "model", "embed_dim", "balanced_accuracy"]
    if "C_median" in df.columns:
        cols.append("C_median")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
