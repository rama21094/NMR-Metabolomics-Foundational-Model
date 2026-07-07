#!/usr/bin/env python3
"""
Run a classifier hyperparameter sweep for the few-shot comparison.

This script reuses fewshot_ml_comparison.py and evaluates all configured
SVM/LR/XGBoost variants on the same episode splits.
"""

import argparse
import copy

import torch

import fewshot_ml_comparison as base


USE_IDE_CONFIG = True

IDE_CONFIG = {
    "dataset_preset": "MTBLS326",
    "data_path": "data/mtbls326/MTBLS326_aligned_spectra.npy",
    "metadata_csv": "data/mtbls326/MTBLS326_metadata_mapping.csv",
    "model_path": "models/SSL_models/combined_unique_WS625to680Zero_20260601_084533_bs32_mr0.50_ps1024_best.pth",
    "label_column": "Factor Value[IP3R expression]",
    "index_column": "npy_row",
    "support_per_class": 6,
    "episodes": 100,
    "classifiers": ["svm", "logreg", "xgboost"],
    "run_foundation_backbone": True,
    "run_direct_binned": True,
    "run_prototype_baseline": True,
    "bin_counts": [1024, 2048],
    "bin_reductions": ["auc"],
    "svm_c": [0.01, 0.03, 0.1, 0.3, 1.0],
    "lr_c": [0.01, 0.03, 0.1, 0.3, 1.0],
    "xgb_max_depth": [1, 2, 3],
    "device": "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    "out_dir": "results/fewshot/hparam_sweep_0.5",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_csv_list(text):
    if isinstance(text, list):
        return text
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_float_csv_list(text):
    return [float(x) for x in parse_csv_list(text)]


def parse_int_csv_list(text):
    return [int(x) for x in parse_csv_list(text)]


def args_from_ide_config():
    args = copy.deepcopy(base.args_from_ide_config())
    for key, value in IDE_CONFIG.items():
        setattr(args, key, value)
    return args


def build_parser():
    parser = argparse.ArgumentParser(description="Run few-shot classifier hyperparameter sweep.")
    parser.add_argument("--model-path", default=IDE_CONFIG["model_path"])
    parser.add_argument("--support-per-class", type=int, default=IDE_CONFIG["support_per_class"])
    parser.add_argument("--episodes", type=int, default=IDE_CONFIG["episodes"])
    parser.add_argument("--svm-c", type=parse_float_csv_list, default=IDE_CONFIG["svm_c"])
    parser.add_argument("--lr-c", type=parse_float_csv_list, default=IDE_CONFIG["lr_c"])
    parser.add_argument("--xgb-max-depth", type=parse_int_csv_list, default=IDE_CONFIG["xgb_max_depth"])
    parser.add_argument("--device", default=IDE_CONFIG["device"])
    parser.add_argument("--out-dir", default=IDE_CONFIG["out_dir"])
    parser.add_argument("--run-foundation-backbone", type=str2bool, default=IDE_CONFIG["run_foundation_backbone"])
    parser.add_argument("--run-direct-binned", type=str2bool, default=IDE_CONFIG["run_direct_binned"])
    parser.add_argument("--run-prototype-baseline", type=str2bool, default=IDE_CONFIG["run_prototype_baseline"])
    return parser


def args_from_cli():
    args = copy.deepcopy(base.args_from_ide_config())
    cli = build_parser().parse_args()
    for key, value in vars(cli).items():
        setattr(args, key, value)
    for key, value in IDE_CONFIG.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.dataset_preset = IDE_CONFIG["dataset_preset"]
    args.data_path = IDE_CONFIG["data_path"]
    args.metadata_csv = IDE_CONFIG["metadata_csv"]
    args.label_column = IDE_CONFIG["label_column"]
    args.index_column = IDE_CONFIG["index_column"]
    args.classifiers = IDE_CONFIG["classifiers"]
    args.bin_counts = IDE_CONFIG["bin_counts"]
    args.bin_reductions = IDE_CONFIG["bin_reductions"]
    return args


def main():
    args = args_from_ide_config() if USE_IDE_CONFIG else args_from_cli()
    base.run_comparison(args)


if __name__ == "__main__":
    main()
