#!/usr/bin/env python3
"""
Run few-shot comparison sweeps across MAE masking ratios and support counts.

This script reuses fewshot_ml_comparison.py as the single-run evaluator and
adds only the experiment grid orchestration.
"""

import argparse
import copy
from pathlib import Path

import pandas as pd

import fewshot_ml_comparison as base


USE_IDE_CONFIG = True

IDE_CONFIG = {
    "masking_checkpoints": {
        "0.20": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_084450_bs32_mr0.20_ps1024_best.pth",
        "0.30": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_114013_bs32_mr0.30_ps1024_best.pth",
        "0.40": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_084724_bs32_mr0.40_ps1024_best.pth",
        "0.50": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_090604_bs32_mr0.50_ps1024_best.pth",
    },
    "support_values": [1, 2, 4, 6],
    "episodes": 100,
    "root_out_dir": "results/fewshot/support_sweep_EDTA_Suppressed_MTBLS326",
    "force_rerun": False,
    "plot_after": True,
    "plot_out_dir": "results/fewshot/support_sweep_EDTA_Suppressed_plots_MTBLS326",
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


def parse_int_csv_list(text):
    return [int(x) for x in parse_csv_list(text)]


def mask_dir_name(masking_ratio):
    return f"mask_{float(masking_ratio):.2f}"


def build_run_args(base_args, masking_ratio, checkpoint_path, support_value, root_out_dir, episodes):
    args = copy.deepcopy(base_args)
    args.model_path = str(checkpoint_path)
    args.support_per_class = int(support_value)
    args.episodes = int(episodes)
    args.out_dir = str(Path(root_out_dir) / mask_dir_name(masking_ratio) / f"support_{int(support_value)}")
    return args


def expected_summary_path(out_dir):
    return Path(out_dir) / "fewshot_ml_comparison_summary.csv"


def validate_checkpoints(masking_checkpoints):
    missing = []
    for ratio, path in masking_checkpoints.items():
        if not Path(path).exists():
            missing.append(f"{ratio}: {path}")
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))


def run_sweep(args):
    validate_checkpoints(args.masking_checkpoints)
    base_args = base.args_from_ide_config()

    records = []
    root_out_dir = Path(args.root_out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    for masking_ratio, checkpoint_path in sorted(args.masking_checkpoints.items(), key=lambda item: float(item[0])):
        for support_value in args.support_values:
            run_args = build_run_args(
                base_args=base_args,
                masking_ratio=masking_ratio,
                checkpoint_path=checkpoint_path,
                support_value=support_value,
                root_out_dir=root_out_dir,
                episodes=args.episodes,
            )
            summary_path = expected_summary_path(run_args.out_dir)
            status = "completed"

            if summary_path.exists() and not args.force_rerun:
                print(f"Skipping existing run: mask={masking_ratio}, support={support_value} -> {summary_path}")
                status = "skipped_existing"
            else:
                print(f"\nRunning: mask={masking_ratio}, support={support_value}")
                base.run_comparison(run_args)

            records.append(
                {
                    "masking_ratio": float(masking_ratio),
                    "support_per_class": int(support_value),
                    "checkpoint_path": str(checkpoint_path),
                    "out_dir": str(run_args.out_dir),
                    "summary_path": str(summary_path),
                    "status": status,
                }
            )

    manifest = pd.DataFrame(records)
    manifest_path = root_out_dir / "support_sweep_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved sweep manifest: {manifest_path}")

    if args.plot_after:
        import plot_support_sweep_comparison as plotter

        plot_args = plotter.args_from_ide_config()
        plot_args.root_dir = str(root_out_dir)
        plot_args.out_dir = str(args.plot_out_dir)
        plotter.run_plotting(plot_args)

    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Run support-count few-shot sweeps across masking-ratio checkpoints.")
    parser.add_argument("--support-values", type=parse_int_csv_list, default=IDE_CONFIG["support_values"])
    parser.add_argument("--episodes", type=int, default=IDE_CONFIG["episodes"])
    parser.add_argument("--root-out-dir", default=IDE_CONFIG["root_out_dir"])
    parser.add_argument("--force-rerun", type=str2bool, default=IDE_CONFIG["force_rerun"])
    parser.add_argument("--plot-after", type=str2bool, default=IDE_CONFIG["plot_after"])
    parser.add_argument("--plot-out-dir", default=IDE_CONFIG["plot_out_dir"])
    return parser


def args_from_ide_config():
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in IDE_CONFIG.items():
        setattr(args, key, value)
    return args


def main():
    args = args_from_ide_config() if USE_IDE_CONFIG else build_parser().parse_args()
    if not hasattr(args, "masking_checkpoints"):
        args.masking_checkpoints = IDE_CONFIG["masking_checkpoints"]
    run_sweep(args)


if __name__ == "__main__":
    main()
