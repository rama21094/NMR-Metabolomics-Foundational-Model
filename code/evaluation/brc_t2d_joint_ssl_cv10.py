#!/usr/bin/env python3
"""Stratified K-fold classification for BrC/T2D using joint SSL checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "code" / "evaluation", ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from brc_t2d_common import (  # noqa: E402
    LABEL_MAPPINGS,
    aggregate_metrics,
    binned_abs_area,
    classical_models,
    default_output_dir,
    load_brc_t2d,
    probability_matrix,
    save_results,
)
from joint_ssl_eval_common import (  # noqa: E402
    FINE_TUNE_CHOICES,
    build_joint_classifier,
    choose_device,
    fine_tune_count,
    maybe_normalize_eval_spectra,
    set_seed,
    train_one_fold,
)


DEFAULT_DATA = "data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero_rowMinMax.npy"
DEFAULT_METADATA = "data/BrC_T2D/BC_T2D_metadata_mapping.csv"
DEFAULT_CHECKPOINT = "models/joint_ssl/latest_best.pth"
DEFAULT_OUTPUT_BASE = "results/cv10/brc_t2d_joint_ssl"


def effective_n_splits(labels: np.ndarray, requested: int) -> int:
    counts = np.bincount(labels)
    min_count = int(counts[counts > 0].min())
    n_splits = min(int(requested), min_count)
    if n_splits < 2:
        raise ValueError(f"Need at least 2 samples per class for stratified CV; class counts={counts.tolist()}")
    if n_splits != requested:
        print(f"Reducing n_splits from {requested} to {n_splits}; smallest class has {min_count} samples.")
    return n_splits


def run_classical_stratified_cv(features, labels, label_names, args):
    results = {}
    n_classes = len(label_names)
    n_splits = effective_n_splits(labels, args.n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    for name, estimator in classical_models(args.seed, args.xgb_jobs, n_classes).items():
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels), 1):
            if args.max_folds is not None and fold > int(args.max_folds):
                break
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            predictions[test_idx] = model.predict(features[test_idx])
            probabilities[test_idx] = probability_matrix(model, features[test_idx], n_classes)
            print(f"\rclassical/{name}: fold {fold}/{n_splits}", end="", flush=True)
        print()

        evaluated_mask = np.isfinite(probabilities).all(axis=1)
        if args.max_folds is None and not np.all(evaluated_mask):
            raise RuntimeError(f"classical/{name} did not produce predictions for all samples")
        eval_labels = labels[evaluated_mask]
        eval_predictions = predictions[evaluated_mask]
        eval_probabilities = probabilities[evaluated_mask]
        results[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": aggregate_metrics(eval_labels, eval_predictions, eval_probabilities, label_names),
        }
    return results


def run_joint_ssl_stratified_cv(spectra, labels, label_names, checkpoint_path, args, device):
    n_classes = len(label_names)
    n_splits = effective_n_splits(labels, args.n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    results = {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for mode in args.fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
        config = None
        normalized_spectra = None

        for fold, (train_idx, test_idx) in enumerate(splitter.split(spectra, labels), 1):
            if args.max_folds is not None and fold > int(args.max_folds):
                break
            set_seed(args.seed + fold)
            model, config = build_joint_classifier(
                checkpoint=checkpoint,
                spectra=spectra,
                n_classes=n_classes,
                head_dropout=args.head_dropout,
                normalize_input_mode=args.normalize_input,
                unfreeze_layers=unfreeze_count,
                device=device,
            )
            if normalized_spectra is None:
                normalized_spectra = maybe_normalize_eval_spectra(spectra, config["normalize_input"])
            train_one_fold(
                model,
                normalized_spectra[train_idx],
                labels[train_idx],
                device,
                args.epochs,
                args.batch_size,
                args.head_lr,
                args.backbone_lr,
                args.weight_decay,
                args.seed + fold,
            )
            model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(normalized_spectra[test_idx]).to(device)
                fold_prob = model(xb).cpu().numpy()
            probabilities[test_idx] = fold_prob
            predictions[test_idx] = np.argmax(fold_prob, axis=1)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\rjoint_ssl/{mode}: fold {fold}/{n_splits}", end="", flush=True)
        print()

        evaluated_mask = np.isfinite(probabilities).all(axis=1)
        if args.max_folds is None and not np.all(evaluated_mask):
            raise RuntimeError(f"joint_ssl/{mode} did not produce predictions for all samples")
        eval_labels = labels[evaluated_mask]
        eval_predictions = predictions[evaluated_mask]
        eval_probabilities = probabilities[evaluated_mask]
        results[mode] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": aggregate_metrics(eval_labels, eval_predictions, eval_probabilities, label_names),
            "checkpoint": str(checkpoint_path),
            "unfrozen_transformer_layers": unfreeze_count,
            "backbone_config": config,
        }
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--label-column", choices=sorted(LABEL_MAPPINGS), default="cancer_status")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--joint-only", action="store_true")
    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default="binned_auc")
    parser.add_argument("--feature-bins", type=int, default=1024)
    parser.add_argument("--fine-tune-modes", nargs="+", choices=FINE_TUNE_CHOICES, default=list(FINE_TUNE_CHOICES))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--normalize-input", choices=["checkpoint", "auto", "true", "false"], default="checkpoint")
    parser.add_argument("--xgb-jobs", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=None, help="Debug/smoke-test limit for CV folds.")
    args = parser.parse_args()

    if args.n_splits < 2:
        parser.error("--n-splits must be at least 2")
    if args.classical_only and args.joint_only:
        parser.error("--classical-only and --joint-only are mutually exclusive")
    if not args.classical_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --classical-only is used")
    if args.output_dir is None:
        args.output_dir = default_output_dir(DEFAULT_OUTPUT_BASE, args.label_column)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    spectra, labels, metadata, label_names = load_brc_t2d(args.data, args.metadata, args.label_column)
    counts = {name: int((labels == i).sum()) for i, name in enumerate(label_names)}
    print(
        f"Loaded {spectra.shape}; label_column={args.label_column}; "
        f"labels={dict(enumerate(label_names))}; counts={counts}; n_splits={args.n_splits}"
    )

    families = {}
    if not args.joint_only:
        features = binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
        families["classical"] = run_classical_stratified_cv(features, labels, label_names, args)

    if not args.classical_only:
        device = choose_device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        print(f"Joint SSL device: {device}")
        families["joint_ssl"] = run_joint_ssl_stratified_cv(
            spectra=spectra,
            labels=labels,
            label_names=label_names,
            checkpoint_path=args.checkpoint,
            args=args,
            device=device,
        )

    run_config = vars(args).copy()
    run_config.update(
        {
            "cv": "StratifiedKFold",
            "n_samples": int(len(labels)),
            "spectrum_length": int(spectra.shape[1]),
            "label_mapping": {str(i): name for i, name in enumerate(label_names)},
            "family": "joint_ssl",
        }
    )
    save_results(
        Path(args.output_dir),
        metadata,
        labels,
        label_names,
        args.label_column,
        families,
        run_config,
    )

    print(f"\nResults written to {args.output_dir}/summary.csv")
    for family, models in families.items():
        for name, result in models.items():
            m = result["metrics"]
            print(
                f"{family}/{name}: accuracy={m['accuracy']:.3f}, "
                f"balanced_accuracy={m['balanced_accuracy']:.3f}, macro_f1={m['macro_f1']:.3f}"
            )


if __name__ == "__main__":
    main()
