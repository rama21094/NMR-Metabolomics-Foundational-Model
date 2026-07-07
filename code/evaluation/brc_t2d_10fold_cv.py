#!/usr/bin/env python3
"""Stratified 10-fold CV for BrC/T2D binary labels across classical, masking, and jigsaw models."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import random
import sys
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "code" / "training"
for path in (ROOT, TRAINING_DIR, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from brc_t2d_common import (  # noqa: E402
    LABEL_MAPPINGS,
    aggregate_metrics,
    binned_abs_area,
    classical_models,
    load_brc_t2d,
    probability_matrix,
)
from brc_t2d_loocv import (  # noqa: E402
    build_foundation_model,
    checkpoint_state,
    fine_tune_count as masking_fine_tune_count,
    train_one_fold as train_masking_one_fold,
)
from brc_t2d_jigsaw_loocv import (  # noqa: E402
    build_jigsaw_classifier,
    checkpoint_label_from_path,
    discover_checkpoints as discover_jigsaw_checkpoints,
    fine_tune_count as jigsaw_fine_tune_count,
    train_one_fold as train_jigsaw_one_fold,
)


FINE_TUNE_CHOICES = ("frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3")
BINARY_LABEL_COLUMNS = ("cancer_status", "diabetes_status")
DEFAULT_DATA = "data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero_rowMinMax.npy"
DEFAULT_METADATA = "data/BrC_T2D/BC_T2D_metadata_mapping.csv"
DEFAULT_OUTPUT_ROOT = "results/cv10/brc_t2d_foundation_comparison"
DEFAULT_MASKING_GLOB = "models/SSL_models/combine_unique_Water_EDTA_Suppressed_*_bs32_mr*_ps1024_best.pth"
DEFAULT_JIGSAW_GLOB = "models/jigsaw/*/*/*_best.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_ratio_label(path: str | Path) -> str:
    match = re.search(r"_mr(\d+\.\d+)_", Path(path).name)
    if match:
        return f"mr{match.group(1)}"
    return Path(path).stem


def discover_masking_checkpoints(pattern: str, max_checkpoints: int | None = None) -> list[Path]:
    paths = sorted((Path(p) for p in glob.glob(pattern)), key=lambda p: mask_ratio_label(p))
    if not paths:
        raise FileNotFoundError(f"No masking checkpoints matched: {pattern}")
    if max_checkpoints is not None:
        paths = paths[: int(max_checkpoints)]
    return paths


def make_folds(labels: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels)]


def save_fold_indices(output_dir: Path, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    payload = [
        {
            "fold": fold_id,
            "train_idx": train_idx.astype(int).tolist(),
            "test_idx": test_idx.astype(int).tolist(),
        }
        for fold_id, (train_idx, test_idx) in enumerate(folds, 1)
    ]
    with (output_dir / "fold_indices.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def fold_metric_row(
    family: str,
    model: str,
    fold_id: int,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    test_idx: np.ndarray,
) -> dict:
    metrics = aggregate_metrics(labels[test_idx], predictions[test_idx], probabilities[test_idx], label_names)
    return {"family": family, "model": model, "fold": int(fold_id), "n_test": int(len(test_idx)), **metrics}


def load_completed_results(output_dir: Path, label_names: list[str]) -> tuple[dict, list[dict], set[tuple[str, str]]]:
    summary_path = output_dir / "summary.csv"
    fold_path = output_dir / "fold_metrics.csv"
    results = {}
    completed = set()
    if not summary_path.exists():
        return results, [], completed

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        family = row.pop("family")
        model = row.pop("model")
        pred_path = output_dir / f"{family}_{model}_oof_pred.npy"
        prob_path = output_dir / f"{family}_{model}_oof_prob.npy"
        if not pred_path.exists() or not prob_path.exists():
            print(f"Skipping cached {family}/{model}: missing OOF arrays")
            continue

        metrics = {}
        for key, value in row.items():
            if value == "":
                continue
            try:
                numeric = float(value)
                metrics[key] = int(numeric) if numeric.is_integer() and key in {"tn", "fp", "fn", "tp"} else numeric
            except ValueError:
                metrics[key] = value
        results.setdefault(family, {})[model] = {
            "predictions": np.load(pred_path),
            "probabilities": np.load(prob_path),
            "metrics": metrics,
        }
        completed.add((family, model))

    fold_rows: list[dict] = []
    if fold_path.exists():
        with fold_path.open(newline="", encoding="utf-8") as handle:
            fold_rows = list(csv.DictReader(handle))
    return results, fold_rows, completed


def run_classical_cv(features, labels, label_names, folds, args, completed):
    results = {}
    fold_rows = []
    n_classes = len(label_names)
    for name, estimator in classical_models(args.seed, args.xgb_jobs, n_classes).items():
        if args.skip_completed and ("classical", name) in completed:
            print(f"classical/{name}: already present in summary.csv; skipping")
            continue
        predictions = np.empty(len(labels), dtype=np.int64)
        probabilities = np.empty((len(labels), n_classes), dtype=np.float64)

        for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            predictions[test_idx] = model.predict(features[test_idx])
            probabilities[test_idx] = probability_matrix(model, features[test_idx], n_classes)
            fold_rows.append(
                fold_metric_row("classical", name, fold_id, labels, predictions, probabilities, label_names, test_idx)
            )
            print(f"\rclassical/{name}: fold {fold_id}/{len(folds)}", end="", flush=True)
        print()

        results[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": aggregate_metrics(labels, predictions, probabilities, label_names),
        }
    return results, fold_rows


def run_masking_cv(spectra, labels, label_names, checkpoint_paths, folds, args, device, completed):
    results = {}
    fold_rows = []
    n_classes = len(label_names)

    for checkpoint_path in checkpoint_paths:
        state = checkpoint_state(checkpoint_path)
        checkpoint_label = mask_ratio_label(checkpoint_path)
        for mode in args.fine_tune_modes:
            model_name = f"{checkpoint_label}_{mode}"
            if args.skip_completed and ("masking", model_name) in completed:
                print(f"masking/{model_name}: already present in summary.csv; skipping")
                continue

            unfreeze_count = masking_fine_tune_count(mode)
            predictions = np.empty(len(labels), dtype=np.int64)
            probabilities = np.empty((len(labels), n_classes), dtype=np.float64)
            config = None

            for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
                set_seed(args.seed + fold_id)
                model, config = build_foundation_model(
                    state=state,
                    spectrum_length=spectra.shape[1],
                    n_classes=n_classes,
                    nhead=args.masking_nhead,
                    backbone_dropout=args.masking_backbone_dropout,
                    head_dropout=args.head_dropout,
                    unfreeze_layers=unfreeze_count,
                    device=device,
                )
                train_masking_one_fold(
                    model,
                    spectra[train_idx],
                    labels[train_idx],
                    device,
                    args.epochs,
                    args.batch_size,
                    args.head_lr,
                    args.backbone_lr,
                    args.weight_decay,
                    args.seed + fold_id,
                )
                model.eval()
                with torch.no_grad():
                    prob = model(torch.from_numpy(spectra[test_idx]).to(device)).cpu().numpy()
                probabilities[test_idx] = prob
                predictions[test_idx] = np.argmax(prob, axis=1)
                fold_rows.append(
                    fold_metric_row("masking", model_name, fold_id, labels, predictions, probabilities, label_names, test_idx)
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"\rmasking/{model_name}: fold {fold_id}/{len(folds)}", end="", flush=True)
            print()

            results[model_name] = {
                "predictions": predictions,
                "probabilities": probabilities,
                "metrics": aggregate_metrics(labels, predictions, probabilities, label_names),
                "checkpoint": str(checkpoint_path),
                "unfrozen_transformer_layers": unfreeze_count,
                "backbone_config": config,
            }
    return results, fold_rows


def run_jigsaw_cv(spectra, labels, label_names, checkpoint_paths, folds, args, device, completed):
    results = {}
    fold_rows = []
    n_classes = len(label_names)

    for checkpoint_path in checkpoint_paths:
        base_label = checkpoint_label_from_path(checkpoint_path)
        for mode in args.fine_tune_modes:
            model_name = f"{base_label}_{mode}"
            if args.skip_completed and ("jigsaw", model_name) in completed:
                print(f"jigsaw/{model_name}: already present in summary.csv; skipping")
                continue

            unfreeze_count = jigsaw_fine_tune_count(mode)
            predictions = np.empty(len(labels), dtype=np.int64)
            probabilities = np.empty((len(labels), n_classes), dtype=np.float64)
            config = None

            for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
                set_seed(args.seed + fold_id)
                model, config = build_jigsaw_classifier(
                    checkpoint_path, spectra, n_classes, args, device, unfreeze_count
                )
                train_jigsaw_one_fold(
                    model,
                    spectra[train_idx],
                    labels[train_idx],
                    device,
                    args.epochs,
                    args.batch_size,
                    args.head_lr,
                    args.backbone_lr,
                    args.weight_decay,
                    args.seed + fold_id,
                )
                model.eval()
                with torch.no_grad():
                    prob = model(torch.from_numpy(spectra[test_idx]).to(device)).cpu().numpy()
                probabilities[test_idx] = prob
                predictions[test_idx] = np.argmax(prob, axis=1)
                fold_rows.append(
                    fold_metric_row("jigsaw", model_name, fold_id, labels, predictions, probabilities, label_names, test_idx)
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"\rjigsaw/{model_name}: fold {fold_id}/{len(folds)}", end="", flush=True)
            print()

            results[model_name] = {
                "predictions": predictions,
                "probabilities": probabilities,
                "metrics": aggregate_metrics(labels, predictions, probabilities, label_names),
                "checkpoint": str(checkpoint_path),
                "unfrozen_transformer_layers": unfreeze_count,
                "backbone_config": config,
            }
    return results, fold_rows


def write_csv(path: Path, rows: list[dict], preferred: list[str]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_cv_results(
    output_dir: Path,
    metadata,
    labels,
    label_names,
    label_column: str,
    families: dict,
    fold_rows: list[dict],
    run_config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    prediction_rows = []

    for i, row in enumerate(metadata):
        prediction_rows.append(
            {
                "npy_row": row["npy_row"],
                "sample_id": row.get("ID", ""),
                "sample_name": row.get("Sample name/ID", row.get("Sample Name", "")),
                "label_column": label_column,
                "label": label_names[int(labels[i])],
                "target": int(labels[i]),
            }
        )

    for family, models in families.items():
        for model_name, result in models.items():
            summary_rows.append({"family": family, "model": model_name, **result["metrics"]})
            np.save(output_dir / f"{family}_{model_name}_oof_pred.npy", result["predictions"])
            np.save(output_dir / f"{family}_{model_name}_oof_prob.npy", result["probabilities"])
            prefix = f"{family}_{model_name}"
            for i, pred in enumerate(result["predictions"]):
                prediction_rows[i][f"{prefix}_prediction"] = label_names[int(pred)]
                for class_idx, class_name in enumerate(label_names):
                    safe_name = class_name.lower().replace(" ", "_").replace("/", "_")
                    prediction_rows[i][f"{prefix}_prob_{safe_name}"] = float(result["probabilities"][i, class_idx])

    if not summary_rows:
        raise ValueError("No models were evaluated or loaded; nothing to save.")

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        ["family", "model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    )
    write_csv(
        output_dir / "fold_metrics.csv",
        fold_rows,
        ["family", "model", "fold", "n_test", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    )
    write_csv(
        output_dir / "oof_predictions.csv",
        prediction_rows,
        ["npy_row", "sample_id", "sample_name", "label_column", "label", "target"],
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--label-columns", nargs="+", choices=BINARY_LABEL_COLUMNS, default=list(BINARY_LABEL_COLUMNS))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--masking-checkpoint-glob", default=DEFAULT_MASKING_GLOB)
    parser.add_argument("--masking-checkpoint", action="append", default=[], help="Explicit masking checkpoint; can be repeated.")
    parser.add_argument("--jigsaw-checkpoint-glob", default=DEFAULT_JIGSAW_GLOB)
    parser.add_argument("--jigsaw-checkpoint", action="append", default=[], help="Explicit jigsaw checkpoint; can be repeated.")
    parser.add_argument("--max-masking-checkpoints", type=int, default=None)
    parser.add_argument("--max-jigsaw-checkpoints", type=int, default=None)
    parser.add_argument("--fine-tune-modes", nargs="+", choices=FINE_TUNE_CHOICES, default=list(FINE_TUNE_CHOICES))
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--foundation-only", action="store_true", help="Skip classical ML models.")
    parser.add_argument("--masking-only", action="store_true")
    parser.add_argument("--jigsaw-only", action="store_true")
    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default="binned_auc")
    parser.add_argument("--feature-bins", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--masking-backbone-dropout", type=float, default=0.15)
    parser.add_argument("--masking-nhead", type=int, default=8)
    parser.add_argument("--normalize-input", choices=["auto", "true", "false"], default="true")
    parser.add_argument("--xgb-jobs", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.classical_only and args.foundation_only:
        parser.error("--classical-only and --foundation-only are mutually exclusive")
    if args.masking_only and args.jigsaw_only:
        parser.error("--masking-only and --jigsaw-only are mutually exclusive")
    if args.classical_only and (args.masking_only or args.jigsaw_only):
        parser.error("--classical-only cannot be combined with --masking-only or --jigsaw-only")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.masking_checkpoint:
        masking_paths = [Path(p) for p in args.masking_checkpoint]
        if args.max_masking_checkpoints is not None:
            masking_paths = masking_paths[: args.max_masking_checkpoints]
    else:
        masking_paths = discover_masking_checkpoints(args.masking_checkpoint_glob, args.max_masking_checkpoints)

    if args.jigsaw_checkpoint:
        jigsaw_paths = [Path(p) for p in args.jigsaw_checkpoint]
        if args.max_jigsaw_checkpoints is not None:
            jigsaw_paths = jigsaw_paths[: args.max_jigsaw_checkpoints]
    else:
        jigsaw_paths = discover_jigsaw_checkpoints(args.jigsaw_checkpoint_glob, args.max_jigsaw_checkpoints)

    print("Masking checkpoints:")
    for path in masking_paths:
        print(f"  - {mask_ratio_label(path)}: {path}")
    print("Jigsaw checkpoints:")
    for path in jigsaw_paths:
        print(f"  - {checkpoint_label_from_path(path)}: {path}")

    device = resolve_device(args.device)
    print(f"Foundation-model device: {device}")

    for label_column in args.label_columns:
        output_dir = Path(args.output_root) / label_column
        spectra, labels, metadata, label_names = load_brc_t2d(args.data, args.metadata, label_column)
        counts = {name: int((labels == i).sum()) for i, name in enumerate(label_names)}
        print(f"\nLoaded {spectra.shape}; label_column={label_column}; labels={dict(enumerate(label_names))}; counts={counts}")

        folds = make_folds(labels, args.folds, args.seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_fold_indices(output_dir, folds)

        families = {}
        fold_rows = []
        completed = set()
        if args.skip_completed:
            families, fold_rows, completed = load_completed_results(output_dir, label_names)
            print(f"Loaded {len(completed)} completed result(s) from {output_dir / 'summary.csv'}")

        if not args.foundation_only and not args.masking_only and not args.jigsaw_only:
            features = binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
            classical_results, classical_fold_rows = run_classical_cv(features, labels, label_names, folds, args, completed)
            if classical_results:
                families.setdefault("classical", {}).update(classical_results)
                fold_rows.extend(classical_fold_rows)

        if not args.classical_only and not args.jigsaw_only:
            masking_results, masking_fold_rows = run_masking_cv(
                spectra, labels, label_names, masking_paths, folds, args, device, completed
            )
            if masking_results:
                families.setdefault("masking", {}).update(masking_results)
                fold_rows.extend(masking_fold_rows)

        if not args.classical_only and not args.masking_only:
            jigsaw_results, jigsaw_fold_rows = run_jigsaw_cv(
                spectra, labels, label_names, jigsaw_paths, folds, args, device, completed
            )
            if jigsaw_results:
                families.setdefault("jigsaw", {}).update(jigsaw_results)
                fold_rows.extend(jigsaw_fold_rows)

        run_config = vars(args).copy()
        run_config.update(
            {
                "label_column": label_column,
                "n_samples": int(len(labels)),
                "spectrum_length": int(spectra.shape[1]),
                "label_mapping": {str(i): name for i, name in enumerate(label_names)},
                "masking_checkpoint_paths": [str(p) for p in masking_paths],
                "jigsaw_checkpoint_paths": [str(p) for p in jigsaw_paths],
            }
        )
        save_cv_results(output_dir, metadata, labels, label_names, label_column, families, fold_rows, run_config)

        print(f"\nResults written to {output_dir / 'summary.csv'}")
        for family, models in families.items():
            for name, result in models.items():
                m = result["metrics"]
                print(
                    f"{family}/{name}: accuracy={m['accuracy']:.3f}, "
                    f"balanced_accuracy={m['balanced_accuracy']:.3f}, "
                    f"weighted_f1={m['weighted_f1']:.3f}"
                )


if __name__ == "__main__":
    main()
