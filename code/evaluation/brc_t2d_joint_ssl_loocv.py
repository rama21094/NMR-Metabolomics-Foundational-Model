#!/usr/bin/env python3
"""LOOCV classification for BrC/T2D using joint SSL checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "code" / "evaluation", ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from brc_t2d_common import (  # noqa: E402
    LABEL_MAPPINGS,
    aggregate_metrics,
    binned_abs_area,
    default_output_dir,
    load_brc_t2d,
    run_classical_loocv,
    save_results,
)
from joint_ssl_eval_common import FINE_TUNE_CHOICES, choose_device, run_joint_ssl_loocv, set_seed  # noqa: E402


DEFAULT_DATA = "data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero_rowMinMax.npy"
DEFAULT_METADATA = "data/BrC_T2D/BC_T2D_metadata_mapping.csv"
DEFAULT_CHECKPOINT = "models/joint_ssl/latest_best.pth"
DEFAULT_OUTPUT_BASE = "results/loocv/brc_t2d_joint_ssl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--label-column", choices=sorted(LABEL_MAPPINGS), default="cancer_status")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
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
    parser.add_argument("--max-folds", type=int, default=None, help="Debug/smoke-test limit for LOOCV folds.")
    args = parser.parse_args()

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
    print(f"Loaded {spectra.shape}; label_column={args.label_column}; labels={dict(enumerate(label_names))}; counts={counts}")

    families = {}
    if not args.joint_only:
        features = binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
        families["classical"] = run_classical_loocv(features, labels, label_names, args.seed, args.xgb_jobs)

    if not args.classical_only:
        device = choose_device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        print(f"Joint SSL device: {device}")
        families["joint_ssl"] = run_joint_ssl_loocv(
            spectra=spectra,
            labels=labels,
            label_names=label_names,
            checkpoint_path=args.checkpoint,
            args=args,
            device=device,
            metric_fn=lambda y, pred, prob: aggregate_metrics(y, pred, prob, label_names),
        )

    run_config = vars(args).copy()
    run_config.update(
        {
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
