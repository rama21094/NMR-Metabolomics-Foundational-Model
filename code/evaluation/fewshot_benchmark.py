#!/usr/bin/env python3
"""Rigorous few-shot benchmark: 3 classical baselines + 3 SSL families, all
evaluated on the *same* class-balanced support-set draws, across a sweep of
support sizes from 2 up to (almost) the full dataset.

For a given --dataset (barth | mtbls326 | mtbls563) this:
  1. Loads spectra/labels with the same loader used by that dataset's LOOCV/CV
     script, so results are directly comparable to the full-data numbers.
  2. Computes the largest usable support_per_class (smallest class size minus
     --min-query-per-class) and builds one shared set of class-balanced
     (support, query) episodes per support size -- generated once, independent
     of model, and reused identically by every family below.
  3. Evaluates classical (logreg / SVM-RBF / XGBoost) on binned-AUC features,
     and masked / jigsaw / joint SSL checkpoints fine-tuned fresh from the
     pretrained weights on each episode's support set, swept over
     --fine-tune-modes (frozen == linear probe on frozen embeddings, plus
     unfreeze_last_1/2/3).
  4. Writes per-episode metrics + a support_per_class-level mean/std summary,
     checkpointing after every family so a partial run is never lost.

MTBLS563 is the only 3-class dataset here; every metric/model helper reused
below (`aggregate_metrics`, `classical_models`, `build_*_classifier`) is
already n_classes-generic, so the same code path covers it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "code" / "evaluation", ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from brc_t2d_common import binned_abs_area, load_brc_t2d  # noqa: E402
from joint_ssl_eval_common import FINE_TUNE_CHOICES, choose_device, set_seed  # noqa: E402
from barth_all_models_loocv import load_barth  # noqa: E402
from mtbls326_loocv import load_mtbls326  # noqa: E402
from fewshot_common import (  # noqa: E402
    build_shared_splits,
    evaluate_classical_fewshot,
    evaluate_jigsaw_fewshot,
    evaluate_joint_fewshot,
    evaluate_masking_fewshot,
    max_support_per_class,
    support_size_grid,
)


DATASET_DEFAULTS = {
    # v4-suppressed data, matching every full-data-CV number in
    # docs/SSL_vs_classical_analysis.md (§3), so the few-shot curve here is
    # directly comparable to the reported full-data points at its top end.
    "barth": dict(
        data="data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_rowMinMax_v4.npy",
        metadata="data/Barth/Workbench_Barth_Syndrome_metadata.csv",
        label_column="label",
        exclude_labels=["Pool"],
    ),
    "mtbls326": dict(
        data="data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy",
        metadata="data/mtbls326/MTBLS326_metadata_mapping.csv",
        label_column=None,
        exclude_labels=[],
    ),
    "mtbls563": dict(
        data="data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy",
        metadata="data/mtbls563/MTBLS563_metadata_mapping.csv",
        label_column="Factor Value[Diagnosis]",
        exclude_labels=["unknown"],
    ),
    # BrC-T2D was previously unsupported here (§6 note) -- load_brc_t2d already
    # returns the (spectra, labels, metadata, label_names) shape this script
    # expects, so wiring it in is just two DATASET_DEFAULTS entries + a
    # load_dataset branch, not a new loader.
    "brc_t2d_cancer": dict(
        data="data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_rowMinMax_v4.npy",
        metadata="data/BrC_T2D/BC_T2D_newlabels_metadata_mapping.csv",
        label_column="cancer_status",
        exclude_labels=[],
    ),
    "brc_t2d_diabetes": dict(
        data="data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_rowMinMax_v4.npy",
        metadata="data/BrC_T2D/BC_T2D_newlabels_metadata_mapping.csv",
        label_column="diabetes_status",
        exclude_labels=[],
    ),
}

# The same 2026-07-25 checkpoints every full-data number in the analysis doc
# is read against (§9 provenance) -- NOT one of the exp15 seed replicates.
# Cherry-picking a lucky seed for this run would just reintroduce the §15
# sampling-artifact mistake one level up; use the one reference checkpoint
# everything else in the document is comparable to.
DEFAULT_CHECKPOINTS = dict(
    masking="models/masked_ssl/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth",
    jigsaw="models/jigsaw/multibin/20260725_085608/multibin_20260725_085608_best.pth",
    joint_ssl="models/joint_ssl/joint_ssl_20260725_085627/joint_ssl_20260725_085627_best.pth",
)


def load_dataset(name: str, args):
    if name == "barth":
        return load_barth(args.data, args.metadata, args.label_column, args.exclude_labels)
    if name == "mtbls326":
        return load_mtbls326(args.data, args.metadata)
    if name == "mtbls563":
        return load_barth(args.data, args.metadata, args.label_column, args.exclude_labels)
    if name in ("brc_t2d_cancer", "brc_t2d_diabetes"):
        return load_brc_t2d(args.data, args.metadata, args.label_column)
    raise ValueError(f"Unknown dataset {name!r}")


def selected_families(args) -> set:
    families = set(args.families)
    if "all" in families:
        return {"classical", "masking", "jigsaw", "joint_ssl"}
    return families


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASET_DEFAULTS), required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--exclude-labels", nargs="*", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--families", nargs="+", choices=["all", "classical", "masking", "jigsaw", "joint_ssl"], default=["all"]
    )

    parser.add_argument("--support-min", type=int, default=2)
    parser.add_argument("--support-max", type=int, default=None, help="Default: largest usable size (auto).")
    parser.add_argument("--support-step", type=int, default=1)
    parser.add_argument("--support-sizes", type=int, nargs="+", default=None,
                         help="Explicit support_per_class list; overrides --support-min/--support-max/--support-step.")
    parser.add_argument("--min-query-per-class", type=int, default=2,
                         help="How many samples per class must remain for querying when auto-computing --support-max.")
    parser.add_argument("--repeats", type=int, default=10, help="Independent random support draws per support size.")

    parser.add_argument("--classical-features", choices=["binned_auc", "raw"], default="binned_auc")
    parser.add_argument("--feature-bins", type=int, default=1024)
    parser.add_argument("--xgb-jobs", type=int, default=4)

    parser.add_argument(
        "--pooling", default="regional:16",
        help="Position-preserving pooling (docs §5c/§14) applied in place of the SSL heads' "
             "hardcoded mean-pool: 'mean_pool' (old behaviour), 'flatten', or 'regional:G'. "
             "regional:G is clamped per-component to that component's own token count, so one "
             "flag applies cleanly across masking/jigsaw/joint despite their different token "
             "counts per bin size. Default matches the safer moderate-G default §14 settled on "
             "over flatten for n=37..113; at the smallest few-shot support sizes even that may "
             "be too high-dimensional -- watch for it in the learning curve, don't assume it away.",
    )
    parser.add_argument("--masking-checkpoint", default=DEFAULT_CHECKPOINTS["masking"])
    parser.add_argument("--jigsaw-checkpoint", default=DEFAULT_CHECKPOINTS["jigsaw"])
    parser.add_argument("--joint-checkpoint", default=DEFAULT_CHECKPOINTS["joint_ssl"])
    parser.add_argument("--fine-tune-modes", nargs="+", choices=FINE_TUNE_CHOICES, default=list(FINE_TUNE_CHOICES))

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
    parser.add_argument("--joint-include-masked-task", choices=["true", "false"], default="true")
    parser.add_argument(
        "--reinit-unfrozen-xavier", action="store_true",
        help="Ablation: reinitialize just-unfrozen layers with Xavier init instead of pretrained weights.",
    )

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make-plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="make_plots", action="store_false")

    args = parser.parse_args()
    preset = DATASET_DEFAULTS[args.dataset]
    args.data = args.data or preset["data"]
    args.metadata = args.metadata or preset["metadata"]
    if args.label_column is None:
        args.label_column = preset["label_column"]
    if args.exclude_labels is None:
        args.exclude_labels = preset["exclude_labels"]
    args.joint_include_masked_task = args.joint_include_masked_task == "true"
    args.families = selected_families(args)
    return args


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    metric_cols = [
        c for c in ok.columns
        if c not in {"family", "model", "fine_tune_mode", "support_per_class", "repeat",
                     "n_train", "n_query", "status", "error", "confusion_matrix"}
        and pd.api.types.is_numeric_dtype(ok[c])
    ]
    agg = {c: ["mean", "std"] for c in metric_cols}
    grouped = ok.groupby(["family", "model", "fine_tune_mode", "support_per_class"], as_index=False).agg(agg)
    grouped.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    n_ep = ok.groupby(["family", "model", "fine_tune_mode", "support_per_class"], as_index=False).size()
    n_ep = n_ep.rename(columns={"size": "n_episodes"})
    grouped = grouped.merge(n_ep, on=["family", "model", "fine_tune_mode", "support_per_class"])
    return grouped.sort_values(["family", "model", "fine_tune_mode", "support_per_class"]).reset_index(drop=True)


def make_plots(summary_df: pd.DataFrame, out_dir: Path, dataset: str, metric: str = "balanced_accuracy_mean"):
    if summary_df.empty or metric not in summary_df.columns:
        return []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    families = sorted(summary_df["family"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), sharey=True)
    if len(families) == 1:
        axes = [axes]
    std_col = metric.replace("_mean", "_std")
    cmap = plt.get_cmap("tab10")
    for ax, family in zip(axes, families):
        sub = summary_df[summary_df["family"] == family]
        lines = sorted(sub[["model", "fine_tune_mode"]].drop_duplicates().itertuples(index=False), key=str)
        for i, (model, mode) in enumerate(lines):
            line_df = sub[(sub["model"] == model) & (sub["fine_tune_mode"] == mode)].sort_values("support_per_class")
            label = model if mode == "-" else f"{model}/{mode}"
            y = line_df[metric].to_numpy()
            yerr = line_df[std_col].fillna(0.0).to_numpy() if std_col in line_df else None
            ax.errorbar(line_df["support_per_class"], y, yerr=yerr, marker="o", markersize=3,
                        capsize=2, linewidth=1.3, color=cmap(i % 10), label=label)
        ax.set_title(family)
        ax.set_xlabel("Support samples per class")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(metric.replace("_mean", "").replace("_", " ").title())
    fig.suptitle(f"{dataset}: few-shot learning curves ({metric.replace('_mean', '')})")
    fig.tight_layout()
    path = plots_dir / f"fewshot_{metric.replace('_mean', '')}_by_family.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))
    return saved


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spectra, labels, metadata, label_names = load_dataset(args.dataset, args)
    n_classes = len(label_names)
    counts = {name: int((labels == i).sum()) for i, name in enumerate(label_names)}
    print(f"Loaded {spectra.shape}; label mapping={dict(enumerate(label_names))}; counts={counts}")

    if args.support_sizes:
        support_sizes = sorted(set(args.support_sizes))
    else:
        support_max = args.support_max or max_support_per_class(labels, n_classes, args.min_query_per_class)
        support_sizes = support_size_grid(args.support_min, support_max, args.support_step)
    n_episodes_per_size = args.repeats
    print(f"Support sizes: {support_sizes} (x{n_episodes_per_size} repeats each)")

    splits_by_size = build_shared_splits(labels, n_classes, support_sizes, args.repeats, args.seed)
    print("Built shared episode splits (identical across all families).")

    all_rows = []

    def checkpoint_now():
        df = pd.DataFrame(all_rows)
        df.to_csv(out_dir / "fewshot_episode_metrics.csv", index=False)
        summary_df = summarize(df)
        summary_df.to_csv(out_dir / "fewshot_summary.csv", index=False)
        return df, summary_df

    if "classical" in args.families:
        features = (
            binned_abs_area(spectra, args.feature_bins) if args.classical_features == "binned_auc" else spectra
        )
        all_rows.extend(
            evaluate_classical_fewshot(features, labels, n_classes, label_names, splits_by_size, args.seed, args.xgb_jobs)
        )
        checkpoint_now()
        print("Classical baselines done.")

    ssl_families = {"masking", "jigsaw", "joint_ssl"} & args.families
    if ssl_families:
        device = choose_device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        print(f"SSL device: {device}")

        if "masking" in args.families:
            all_rows.extend(
                evaluate_masking_fewshot(
                    spectra, labels, n_classes, label_names, args.masking_checkpoint,
                    args.fine_tune_modes, splits_by_size, args, device,
                )
            )
            checkpoint_now()
            print("Masked-SSL done.")

        if "jigsaw" in args.families:
            all_rows.extend(
                evaluate_jigsaw_fewshot(
                    spectra, labels, n_classes, label_names, args.jigsaw_checkpoint,
                    args.fine_tune_modes, splits_by_size, args, device,
                )
            )
            checkpoint_now()
            print("Jigsaw-SSL done.")

        if "joint_ssl" in args.families:
            all_rows.extend(
                evaluate_joint_fewshot(
                    spectra, labels, n_classes, label_names, args.joint_checkpoint,
                    args.fine_tune_modes, splits_by_size, args, device,
                )
            )
            checkpoint_now()
            print("Joint-SSL done.")

    df, summary_df = checkpoint_now()

    run_config = vars(args).copy()
    run_config["families"] = sorted(args.families)
    run_config.update(
        {
            "n_samples": int(len(labels)),
            "spectrum_length": int(spectra.shape[1]),
            "label_mapping": {str(i): name for i, name in enumerate(label_names)},
            "class_counts": counts,
            "support_sizes": support_sizes,
        }
    )
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    plot_paths = []
    if args.make_plots and not summary_df.empty:
        metric = "balanced_accuracy_mean" if "balanced_accuracy_mean" in summary_df.columns else None
        if metric:
            plot_paths = make_plots(summary_df, out_dir, args.dataset, metric)

    print(f"\nResults written to {out_dir}/fewshot_summary.csv")
    print(f"Per-episode metrics: {out_dir}/fewshot_episode_metrics.csv")
    if plot_paths:
        print("Plots:")
        for p in plot_paths:
            print(f"  - {p}")
    n_errors = int((df["status"] == "error").sum())
    if n_errors:
        print(f"Warning: {n_errors} episodes errored (see status/error columns in fewshot_episode_metrics.csv).")


if __name__ == "__main__":
    main()
