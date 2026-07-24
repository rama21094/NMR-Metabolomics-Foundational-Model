#!/usr/bin/env python3
"""Stratified 10-fold CV for the MTBLS563 dataset across all model families.

3-class problem (control / bacterial infection / viral infection) built from
`Factor Value[Diagnosis]`; the `unknown` diagnosis rows are excluded by
default since that label isn't a diagnosis outcome.

At 142 samples, LOOCV would run 142 folds x 4 fine-tune modes x 3 SSL
families; stratified 10-fold CV keeps class balance per fold while cutting
the fold count by ~14x, matching the `brc_t2d_10fold_cv.py` convention used
elsewhere in this repo for datasets too large for LOOCV to be practical.

Reuses the multiclass-safe metric/classical-model helpers from
`brc_t2d_common.py` and the masking/jigsaw/joint-SSL classifier-building code
from `barth_all_models_loocv.py` / `joint_ssl_eval_common.py` rather than
redefining them.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from brc_t2d_common import binned_abs_area, classical_models, probability_matrix  # noqa: E402
from barth_all_models_loocv import (  # noqa: E402
    build_jigsaw_classifier,
    build_masked_classifier,
    checkpoint_state,
    finalize_result,
    load_barth as load_metadata_dataset,
    load_jigsaw_checkpoint_file,
    train_classifier_one_fold,
)
from joint_ssl_eval_common import (  # noqa: E402
    build_joint_classifier,
    choose_device,
    fine_tune_count,
    maybe_normalize_eval_spectra,
    set_seed,
    train_one_fold as train_joint_one_fold,
)

FINE_TUNE_CHOICES = ("frozen", "unfreeze_last_1", "unfreeze_last_2", "unfreeze_last_3")


def load_mtbls563(data_path, metadata_path, label_column, exclude_labels):
    # `load_barth` is dataset-agnostic (label_column + exclude_labels -> spectra/
    # labels/metadata/label_names from npy_row-indexed metadata); reused as-is.
    return load_metadata_dataset(data_path, metadata_path, label_column, exclude_labels)


def run_classical_cv(features, labels, label_names, splitter, args):
    results = {}
    n_classes = len(label_names)
    for name, estimator in classical_models(args.seed, args.xgb_jobs, n_classes).items():
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels), 1):
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            predictions[test_idx] = model.predict(features[test_idx])
            probabilities[test_idx] = probability_matrix(model, features[test_idx], n_classes)
            print(f"\rclassical/{name}: CV fold {fold}/{args.n_splits}", end="", flush=True)
        print()
        results[name] = finalize_result(labels, label_names, predictions, probabilities)
    return results


def run_masking_cv(spectra, labels, label_names, checkpoint_path, args, device, splitter):
    results = {}
    n_classes = len(label_names)
    state = checkpoint_state(checkpoint_path)
    for mode in args.fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
        config = None
        for fold, (train_idx, test_idx) in enumerate(splitter.split(spectra, labels), 1):
            set_seed(args.seed + fold)
            model, config = build_masked_classifier(
                state, spectra.shape[1], n_classes, args, device, unfreeze_count
            )
            train_classifier_one_fold(
                model,
                spectra[train_idx],
                labels[train_idx],
                device,
                args.epochs,
                args.batch_size,
                args.head_lr,
                args.backbone_lr,
                args.weight_decay,
                args.seed + fold,
                lambda m: m.backbone.encoder.transformer.layers,
            )
            model.eval()
            with torch.no_grad():
                prob = model(torch.from_numpy(spectra[test_idx]).to(device)).cpu().numpy()
            probabilities[test_idx] = prob
            predictions[test_idx] = np.argmax(prob, axis=1)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\rmasking/{mode}: CV fold {fold}/{args.n_splits}", end="", flush=True)
        print()
        results[mode] = finalize_result(
            labels, label_names, predictions, probabilities,
            checkpoint=str(checkpoint_path), unfrozen_transformer_layers=unfreeze_count,
            backbone_config=config,
        )
    return results


def run_jigsaw_cv(spectra, labels, label_names, checkpoint_path, args, device, splitter):
    results = {}
    n_classes = len(label_names)
    checkpoint = load_jigsaw_checkpoint_file(checkpoint_path)
    for mode in args.fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
        config = None
        for fold, (train_idx, test_idx) in enumerate(splitter.split(spectra, labels), 1):
            set_seed(args.seed + fold)
            model, config = build_jigsaw_classifier(
                checkpoint, spectra, n_classes, args, device, unfreeze_count
            )
            train_classifier_one_fold(
                model,
                spectra[train_idx],
                labels[train_idx],
                device,
                args.epochs,
                args.batch_size,
                args.head_lr,
                args.backbone_lr,
                args.weight_decay,
                args.seed + fold,
                lambda m: m.backbone.transformer.layers,
            )
            model.eval()
            with torch.no_grad():
                prob = model(torch.from_numpy(spectra[test_idx]).to(device)).cpu().numpy()
            probabilities[test_idx] = prob
            predictions[test_idx] = np.argmax(prob, axis=1)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\rjigsaw/{mode}: CV fold {fold}/{args.n_splits}", end="", flush=True)
        print()
        results[mode] = finalize_result(
            labels, label_names, predictions, probabilities,
            checkpoint=str(checkpoint_path), unfrozen_transformer_layers=unfreeze_count,
            backbone_config=config,
        )
    return results


def run_joint_ssl_cv(spectra, labels, label_names, checkpoint_path, args, device, splitter):
    results = {}
    n_classes = len(label_names)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for mode in args.fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), n_classes), np.nan, dtype=np.float64)
        config = None
        normalized_spectra = None
        for fold, (train_idx, test_idx) in enumerate(splitter.split(spectra, labels), 1):
            set_seed(args.seed + fold)
            model, config = build_joint_classifier(
                checkpoint=checkpoint,
                spectra=spectra,
                n_classes=n_classes,
                head_dropout=args.head_dropout,
                normalize_input_mode=args.joint_normalize_input,
                unfreeze_layers=unfreeze_count,
                device=device,
                include_masked_task=args.joint_include_masked_task,
                reinit_unfrozen=getattr(args, "reinit_unfrozen_xavier", False),
            )
            if normalized_spectra is None:
                normalized_spectra = maybe_normalize_eval_spectra(spectra, config["normalize_input"])
            train_joint_one_fold(
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
                prob = model(torch.from_numpy(normalized_spectra[test_idx]).to(device)).cpu().numpy()
            probabilities[test_idx] = prob
            predictions[test_idx] = np.argmax(prob, axis=1)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\rjoint_ssl/{mode}: CV fold {fold}/{args.n_splits}", end="", flush=True)
        print()
        results[mode] = finalize_result(
            labels, label_names, predictions, probabilities,
            checkpoint=str(checkpoint_path), unfrozen_transformer_layers=unfreeze_count,
            backbone_config=config,
        )
    return results


def save_results(output_dir, metadata, labels, label_names, label_column, families, run_config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    prediction_rows = []

    for i, row in enumerate(metadata):
        prediction_rows.append(
            {
                "npy_row": row.get("npy_row", ""),
                "sample_id": row.get("Sample Name", row.get("folder_name", "")),
                "label_column": label_column,
                "label": label_names[int(labels[i])],
                "target": int(labels[i]),
            }
        )

    for family, models in families.items():
        for model_name, result in models.items():
            summary_rows.append(
                {
                    "family": family,
                    "model": model_name,
                    "n_evaluated": result.get("n_evaluated", len(labels)),
                    **result["metrics"],
                }
            )
            np.save(output_dir / f"{family}_{model_name}_oof_pred.npy", result["predictions"])
            np.save(output_dir / f"{family}_{model_name}_oof_prob.npy", result["probabilities"])
            for i, pred in enumerate(result["predictions"]):
                prefix = f"{family}_{model_name}"
                prediction_rows[i][f"{prefix}_prediction"] = (
                    label_names[int(pred)] if int(pred) >= 0 else ""
                )
                for class_idx, class_name in enumerate(label_names):
                    safe_name = class_name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                    prob = result["probabilities"][i, class_idx]
                    prediction_rows[i][f"{prefix}_prob_{safe_name}"] = (
                        float(prob) if np.isfinite(prob) else ""
                    )

    if not summary_rows:
        raise ValueError("No models were evaluated; nothing to save.")

    fieldnames = sorted({key for row in summary_rows for key in row})
    preferred = ["family", "model", "n_evaluated", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    prediction_fields = sorted({key for row in prediction_rows for key in row})
    preferred_prediction = ["npy_row", "sample_id", "label_column", "label", "target"]
    prediction_fields = [key for key in preferred_prediction if key in prediction_fields] + [
        key for key in prediction_fields if key not in preferred_prediction
    ]
    with (output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_rowMinMax.npy")
    parser.add_argument("--metadata", default="data/mtbls563/MTBLS563_metadata_mapping.csv")
    parser.add_argument("--label-column", default="Factor Value[Diagnosis]")
    parser.add_argument("--exclude-labels", nargs="*", default=["unknown"])
    parser.add_argument("--output-dir", default="results/loocv/mtbls563_all_models")
    parser.add_argument(
        "--families", nargs="+", choices=["classical", "masking", "jigsaw", "joint_ssl", "all"], default=["all"]
    )
    parser.add_argument("--masking-checkpoint", help="Single masked-autoencoder .pth checkpoint")
    parser.add_argument("--jigsaw-checkpoint", help="Single jigsaw .pth checkpoint")
    parser.add_argument("--joint-checkpoint", help="Single joint-SSL .pth checkpoint")
    parser.add_argument("--fine-tune-modes", nargs="+", choices=FINE_TUNE_CHOICES, default=list(FINE_TUNE_CHOICES))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default="binned_auc")
    parser.add_argument("--feature-bins", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--backbone-dropout", type=float, default=0.15)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--normalize-input", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--joint-normalize-input", choices=["checkpoint", "auto", "true", "false"], default="checkpoint")
    parser.add_argument("--joint-include-masked-task", action="store_true", default=True)
    parser.add_argument("--xgb-jobs", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reinit-unfrozen-xavier", action="store_true",
        help="Ablation: reinitialize the just-unfrozen layers with Xavier init instead of "
             "keeping their pretrained weights, before fine-tuning.",
    )
    args = parser.parse_args()

    families = set(args.families)
    if "all" in families:
        families = {"classical", "masking", "jigsaw", "joint_ssl"}
    args.families = families

    if "masking" in families and not args.masking_checkpoint:
        parser.error("--masking-checkpoint is required when 'masking' is in --families")
    if "jigsaw" in families and not args.jigsaw_checkpoint:
        parser.error("--jigsaw-checkpoint is required when 'jigsaw' is in --families")
    if "joint_ssl" in families and not args.joint_checkpoint:
        parser.error("--joint-checkpoint is required when 'joint_ssl' is in --families")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    spectra, labels, metadata, label_names = load_mtbls563(
        args.data, args.metadata, args.label_column, args.exclude_labels
    )
    counts = {name: int((labels == i).sum()) for i, name in enumerate(label_names)}
    print(
        f"Loaded {spectra.shape}; label_column={args.label_column}; "
        f"labels={dict(enumerate(label_names))}; counts={counts}"
    )

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    families = {}

    run_config = vars(args).copy()
    run_config["families"] = sorted(args.families)
    run_config.update(
        {
            "n_samples": int(len(labels)),
            "spectrum_length": int(spectra.shape[1]),
            "label_mapping": {str(i): name for i, name in enumerate(label_names)},
        }
    )

    def checkpoint_now():
        # Persist whatever families have finished so far after each family, so
        # a crash/interruption partway through doesn't discard already-
        # completed CV work from earlier families (see barth_all_models_loocv.py
        # for the incident that motivated this).
        save_results(args.output_dir, metadata, labels, label_names, args.label_column, families, run_config)

    if "classical" in args.families:
        features = binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
        families["classical"] = run_classical_cv(features, labels, label_names, splitter, args)
        checkpoint_now()

    foundation_families = {"masking", "jigsaw", "joint_ssl"} & args.families
    if foundation_families:
        device = choose_device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        print(f"Foundation-model device: {device}")

    if "masking" in args.families:
        print(f"Masking checkpoint: {args.masking_checkpoint}")
        families["masking"] = run_masking_cv(
            spectra, labels, label_names, args.masking_checkpoint, args, device, splitter
        )
        checkpoint_now()

    if "jigsaw" in args.families:
        print(f"Jigsaw checkpoint: {args.jigsaw_checkpoint}")
        families["jigsaw"] = run_jigsaw_cv(
            spectra, labels, label_names, args.jigsaw_checkpoint, args, device, splitter
        )
        checkpoint_now()

    if "joint_ssl" in args.families:
        print(f"Joint SSL checkpoint: {args.joint_checkpoint}")
        families["joint_ssl"] = run_joint_ssl_cv(
            spectra, labels, label_names, args.joint_checkpoint, args, device, splitter
        )
        checkpoint_now()

    save_results(args.output_dir, metadata, labels, label_names, args.label_column, families, run_config)

    print(f"\nResults written to {args.output_dir}/summary.csv")
    for family, models in families.items():
        for name, result in models.items():
            m = result["metrics"]
            print(f"{family}/{name}: balanced_accuracy={m['balanced_accuracy']:.3f}, weighted_f1={m['weighted_f1']:.3f}")


if __name__ == "__main__":
    main()
