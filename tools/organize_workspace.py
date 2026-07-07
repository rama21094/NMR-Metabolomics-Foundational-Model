#!/usr/bin/env python3
"""Organize the NMR metabolomics workspace without deleting files.

This script is intentionally conservative:
- generate docs and inventories before moving anything;
- move known files/folders into a clearer layout;
- leave root-level compatibility symlinks for moved top-level entries;
- never remove original content.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover - inventory still works without numpy
    np = None


ROOT = Path(__file__).resolve().parents[1]

HIDDEN_OR_INTERNAL = {
    ".git",
    ".venv",
    ".agents",
    ".codex",
}

DOCS_DIR = ROOT / "docs"

SCRIPT_CATEGORIES = {
    "preprocessing": {
        "alignSpectra.py",
        "WSZero_62500To68000.py",
        "suppress_edta_peak.py",
        "removeZeroTail.py",
        "combine_unique_npy.py",
        "visualiseAfterNormalise.py",
        "findDuplicates.py",
    },
    "training": {
        "trainer_revised.py",
        "transformer.py",
        "transformer_augment.py",
        "CNN_ContrastiveLearning.py",
        "autoencoder.py",
    },
    "evaluation": {
        "Test_suite.py",
        "Test_suite_v2.py",
        "classify_binned_auc_cv.py",
        "classify_embeddings_cv.py",
        "classify_head_cv.py",
        "fewshot_ml_comparison.py",
        "prototype_fewshot.py",
        "mtbls326_loocv.py",
        "run_hparam_sweep_comparison.py",
        "run_support_sweep_comparison.py",
        "gradient_analysis.py",
    },
    "plotting": {
        "compare_mtbls326_summaries.py",
        "plot_compare_summaries.py",
        "plot_masking_model_comparison.py",
        "plot_model.py",
        "plot_model_architecture.py",
        "plot_recon_peaks.py",
        "plot_support_sweep_comparison.py",
        "visualise_random_spectra_slice.py",
        "compare_loocv_summaries.py",
    },
    "utils": {
        "check_npy_compare.py",
        "check_npy_shape.py",
        "check_npy_sizes.py",
    },
}

PURPOSES = {
    "alignSpectra.py": "Align and preprocess NMR spectra to a common length.",
    "WSZero_62500To68000.py": "Zero/suppress the water-region window 62500:68000.",
    "suppress_edta_peak.py": "Detect and suppress the dominant EDTA peak in aligned plasma spectra.",
    "removeZeroTail.py": "Inspect and remove/plot zero-tail regions in spectra.",
    "combine_unique_npy.py": "Combine .npy spectra arrays and remove duplicate rows.",
    "visualiseAfterNormalise.py": "Normalize spectra and visualize before/after examples.",
    "findDuplicates.py": "Find duplicate rows in a .npy spectra array.",
    "trainer_revised.py": "Train the NMR masked autoencoder/foundation model.",
    "transformer.py": "Transformer MAE model/training experiment script.",
    "transformer_augment.py": "Transformer MAE experiment with augmentation.",
    "CNN_ContrastiveLearning.py": "CNN contrastive learning experiment for NMR spectra.",
    "autoencoder.py": "Convolutional autoencoder prototype experiment.",
    "Test_suite.py": "Evaluate trained MAE checkpoints and save reconstruction metrics.",
    "Test_suite_v2.py": "Updated MAE evaluation/test suite with worker support.",
    "classify_binned_auc_cv.py": "Classical CV classification on binned/AUC spectra features.",
    "classify_embeddings_cv.py": "CV classification using frozen MAE embeddings.",
    "classify_head_cv.py": "CV classification with frozen and finetuned MAE heads.",
    "fewshot_ml_comparison.py": "Few-shot comparison of MAE embeddings, binned spectra, and prototype baselines.",
    "prototype_fewshot.py": "Prototype few-shot classifier using frozen MAE embeddings.",
    "mtbls326_loocv.py": "LOOCV classification workflow for MTBLS326.",
    "run_hparam_sweep_comparison.py": "Run few-shot classifier hyperparameter sweeps.",
    "run_support_sweep_comparison.py": "Run support-count sweeps across masking-ratio checkpoints.",
    "gradient_analysis.py": "Analyze gradients and training behavior for NMR MAE.",
    "compare_mtbls326_summaries.py": "Compare MTBLS326 LOOCV summaries and create plots.",
    "plot_compare_summaries.py": "Compare test summary JSONs across masking ratios.",
    "plot_masking_model_comparison.py": "Plot few-shot masking-ratio comparison summaries.",
    "plot_model.py": "Prototype model architecture plotting script.",
    "plot_model_architecture.py": "Infer and plot MAE checkpoint architecture.",
    "plot_recon_peaks.py": "Plot reconstruction peak examples.",
    "plot_support_sweep_comparison.py": "Plot support-count sweep summaries.",
    "visualise_random_spectra_slice.py": "Plot random slices from a spectra .npy file.",
    "compare_loocv_summaries.py": "Compare LOOCV summaries from result folders.",
    "check_npy_compare.py": "Compare .npy files as sets of rows.",
    "check_npy_shape.py": "Print .npy shapes and dtypes.",
    "check_npy_sizes.py": "Print .npy file sizes.",
}

RESULT_DIR_RULES = (
    ("classification_binned_auc_results", "results/classification/binned_auc"),
    ("classification_head_results", "results/classification/head"),
    ("classification_results", "results/classification/embeddings"),
    ("results/fewshot/hparam_sweep_0.5", "results/fewshot/hparam_sweep_0.5"),
    ("results/fewshot/masking_comparison_plots", "results/fewshot/masking_comparison_plots"),
    ("results/fewshot/prototype_default", "results/fewshot/prototype_default"),
    ("results/fewshot/mask_0.2", "results/fewshot/mask_0.2"),
    ("results/fewshot/mask_0.3", "results/fewshot/mask_0.3"),
    ("results/fewshot/mask_0.4", "results/fewshot/mask_0.4"),
    ("results/fewshot/mask_0.5", "results/fewshot/mask_0.5"),
    ("results/fewshot/cli_563_quick", "results/fewshot/cli_563_quick"),
    ("results/fewshot/ide_quick", "results/fewshot/ide_quick"),
    ("results/fewshot/mtbls326_sanity", "results/fewshot/mtbls326_sanity"),
    ("results/fewshot/mtbls326_sanity_mahal", "results/fewshot/mtbls326_sanity_mahal"),
    ("results/fewshot/new_sanity", "results/fewshot/new_sanity"),
    ("fewshot_sandbox", "results/fewshot/sandbox"),
    ("results/fewshot/support_sweep", "results/fewshot/support_sweep"),
    ("results/fewshot/support_sweep_EDTA_Suppressed_MTBLS326", "results/fewshot/support_sweep_EDTA_Suppressed_MTBLS326"),
    ("results/fewshot/support_sweep_EDTA_Suppressed_plots_MTBLS326", "results/fewshot/support_sweep_EDTA_Suppressed_plots_MTBLS326"),
    ("results/fewshot/support_sweep_MTBLS326", "results/fewshot/support_sweep_MTBLS326"),
    ("results/fewshot/support_sweep_MTBLS563", "results/fewshot/support_sweep_MTBLS563"),
    ("results/fewshot/support_sweep_plots_MTBLS326", "results/fewshot/support_sweep_plots_MTBLS326"),
    ("results/fewshot/support_sweep_plots_MTBLS563", "results/fewshot/support_sweep_plots_MTBLS563"),
    ("gradient_analysis_results", "results/gradient_analysis"),
    ("mtbls326_loocv_comparison_plots", "results/loocv/mtbls326_comparison_plots"),
    ("mtbls326_loocv_results_0.20", "results/loocv/mtbls326_mask_0.20"),
    ("mtbls326_loocv_results_0.30", "results/loocv/mtbls326_mask_0.30"),
    ("mtbls326_loocv_results_0.40", "results/loocv/mtbls326_mask_0.40"),
    ("mtbls326_loocv_results_0.50", "results/loocv/mtbls326_mask_0.50"),
    ("mtbls326_loocv_results_0.50_oldCominedModel", "results/loocv/mtbls326_mask_0.50_oldCominedModel"),
    ("test_results", "results/testing/combined"),
    ("test_results_debug", "results/testing/debug"),
    ("test_results_plasma", "results/testing/plasma"),
    ("Plots and Visualisations", "results/plots/Plots and Visualisations"),
    ("VisualisationWhileDeveloping", "results/plots/VisualisationWhileDeveloping"),
    ("recon_peaks", "results/reconstruction/recon_peaks"),
)


@dataclass(frozen=True)
class MoveItem:
    source: Path
    target: Path
    kind: str
    category: str
    symlink: bool = False

    @property
    def source_display(self) -> str:
        return str(self.source.relative_to(ROOT))

    @property
    def target_display(self) -> str:
        return str(self.target.relative_to(ROOT))


def relative_symlink_target(link_path: Path, target_path: Path) -> str:
    return os.path.relpath(target_path, start=link_path.parent)


def category_for_script(name: str) -> str:
    for category, names in SCRIPT_CATEGORIES.items():
        if name in names:
            return category
    if name.startswith("classify_") or name.startswith("run_"):
        return "evaluation"
    if name.startswith("plot_") or name.startswith("visualise_") or name.startswith("compare_"):
        return "plotting"
    if name.startswith("check_"):
        return "utils"
    return "utils"


def target_for_script(path: Path) -> Path:
    category = category_for_script(path.name)
    return ROOT / "code" / category / path.name


def target_for_data_file(path: Path) -> Path:
    name = path.name
    suffix = path.suffix.lower()

    if name.startswith("MTBLS326"):
        return ROOT / "data" / "mtbls326" / name
    if name.startswith("MTBLS563"):
        return ROOT / "data" / "mtbls563" / name
    if "TBI_Tirupati" in name or name == "title_labels.csv":
        return ROOT / "data" / "tbi_tirupati" / name
    if name.startswith("plasma") or "Plasma" in name:
        if suffix == ".png":
            return ROOT / "results" / "plots" / "preprocessing" / name
        return ROOT / "data" / "plasma" / name
    if name.startswith("serum") or "Serum" in name:
        if suffix == ".png":
            return ROOT / "results" / "plots" / "preprocessing" / name
        return ROOT / "data" / "serum" / name
    if name.startswith("combined") or name.startswith("combine_unique"):
        if suffix == ".png":
            return ROOT / "results" / "plots" / "preprocessing" / name
        return ROOT / "data" / "combined" / name
    if name.startswith("aligned_128K") or name.startswith("aligned_nmr_spectra"):
        return ROOT / "data" / "aligned" / name
    if name.startswith("water_suppressed") or name == "nmr_spectra.npy":
        return ROOT / "data" / "source" / name
    if name.startswith("edta_suppression"):
        return ROOT / "results" / "preprocessing_diagnostics" / name
    if name in {"zero_tail_processing_results.png"}:
        return ROOT / "results" / "preprocessing_diagnostics" / name
    if name in {"model_plot_summary.txt", "nmr_mae_summary.txt", "nmr_model_architecture"}:
        return ROOT / "results" / "model_architecture" / name
    if suffix == ".png":
        return ROOT / "results" / "plots" / "preprocessing" / name
    if suffix == ".csv":
        return ROOT / "data" / "metadata" / name
    if suffix == ".npy":
        return ROOT / "data" / "misc" / name
    return ROOT / "archive" / "misc_root_files" / name


def discover_move_items() -> list[MoveItem]:
    items: list[MoveItem] = []

    for path in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        name = path.name
        if name in HIDDEN_OR_INTERNAL or name in {"docs", "code", "data", "models", "results", "archive", "tools"}:
            continue
        if path.is_symlink():
            continue

        if path.is_file() and path.suffix == ".py":
            target = target_for_script(path)
            items.append(MoveItem(path, target, "file", f"code/{target.parent.name}"))
        elif path.is_file() and (
            path.suffix.lower() in {".npy", ".csv", ".png", ".txt", ".pth", ".json"} or path.suffix == ""
        ):
            target = target_for_data_file(path)
            items.append(MoveItem(path, target, "file", target.parent.relative_to(ROOT).as_posix()))
        elif path.is_dir() and name == "SSL_models":
            items.append(MoveItem(path, ROOT / "models" / "SSL_models", "directory", "models"))
        elif path.is_dir() and name == "Old Models":
            items.append(MoveItem(path, ROOT / "models" / "Old Models", "directory", "models"))
        elif path.is_dir() and name == "output":
            items.append(MoveItem(path, ROOT / "archive" / "inventory_output", "directory", "archive"))
        elif path.is_dir() and name == "__pycache__":
            items.append(MoveItem(path, ROOT / "archive" / "python_cache" / "root___pycache__", "directory", "archive", symlink=False))
        elif path.is_dir() and name == "scripts":
            items.append(MoveItem(path, ROOT / "archive" / "legacy_scripts_dir", "directory", "archive"))
        elif path.is_dir():
            for source_name, target_rel in RESULT_DIR_RULES:
                if name == source_name:
                    items.append(MoveItem(path, ROOT / target_rel, "directory", target_rel))
                    break

    return items


def iter_inventory_files() -> Iterable[Path]:
    allowed = {".npy", ".npz", ".csv", ".png", ".txt", ".pth", ".json"}
    for path in ROOT.rglob("*"):
        rel_parts = path.relative_to(ROOT).parts
        if not rel_parts:
            continue
        if rel_parts[0] in HIDDEN_OR_INTERNAL:
            continue
        if path.is_file() and (path.suffix.lower() in allowed or path.name == "nmr_model_architecture"):
            yield path


def npy_shape_dtype(path: Path) -> tuple[str, str, str]:
    if np is None or path.suffix.lower() not in {".npy", ".npz"}:
        return "", "", ""
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if hasattr(arr, "files"):
            shapes = []
            dtypes = []
            for key in arr.files:
                shapes.append(f"{key}:{arr[key].shape}")
                dtypes.append(f"{key}:{arr[key].dtype}")
            return ";".join(shapes), ";".join(dtypes), ""
        return str(arr.shape), str(arr.dtype), ""
    except Exception as exc:
        return "", "", str(exc)


def extract_docstring_or_comment(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    match = re.search(r'^\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')', text, re.S)
    if match:
        value = (match.group(1) or match.group(2) or "").strip().splitlines()
        return " ".join(line.strip() for line in value[:2] if line.strip())
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            return stripped.lstrip("#").strip()
    return ""


def extract_path_hints(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", ""
    input_patterns = [
        r"(?:INPUT_FILE|DATA_PATH|MODEL_PATH|SAVE_DIR|OUTPUT_FILE)\s*=\s*['\"]([^'\"]+)['\"]",
        r"default=['\"]([^'\"]+\.(?:npy|csv|pth|json))['\"]",
        r"['\"]([^'\"]+\.(?:npy|csv|pth|json))['\"]",
    ]
    output_patterns = [
        r"(?:OUTPUT_FILE|SAVE_DIR|output_dir|out_dir|save_dir)\s*[:=]\s*['\"]([^'\"]+)['\"]",
        r"['\"]([^'\"]*(?:results|fewshot|classification|plots|SSL_models)[^'\"]*)['\"]",
    ]
    inputs: list[str] = []
    outputs: list[str] = []
    for pattern in input_patterns:
        inputs.extend(re.findall(pattern, text))
    for pattern in output_patterns:
        outputs.extend(re.findall(pattern, text))
    return "; ".join(sorted(set(inputs))[:12]), "; ".join(sorted(set(outputs))[:12])


def write_move_plan(items: list[MoveItem]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with (DOCS_DIR / "move_plan.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "target", "kind", "category", "compatibility_symlink"])
        for item in items:
            writer.writerow([item.source_display, item.target_display, item.kind, item.category, item.symlink])


def write_script_catalog(items: list[MoveItem]) -> None:
    script_paths = [item.source for item in items if item.source.suffix == ".py"]
    script_paths.extend(
        p for p in (ROOT / "scripts").rglob("*.py") if (ROOT / "scripts").exists()
    )
    seen: set[Path] = set()
    rows = []
    for path in sorted(script_paths, key=lambda p: str(p.relative_to(ROOT))):
        if path in seen:
            continue
        seen.add(path)
        category = category_for_script(path.name)
        inputs, outputs = extract_path_hints(path)
        rows.append(
            {
                "script": str(path.relative_to(ROOT)),
                "planned_location": str(target_for_script(path).relative_to(ROOT)),
                "category": category,
                "purpose": PURPOSES.get(path.name) or extract_docstring_or_comment(path),
                "input_hints": inputs,
                "output_hints": outputs,
            }
        )
    with (DOCS_DIR / "script_catalog.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["script", "planned_location", "category", "purpose", "input_hints", "output_hints"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_data_inventory() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(iter_inventory_files(), key=lambda p: str(p.relative_to(ROOT))):
        shape, dtype, error = npy_shape_dtype(path)
        stat = path.lstat()
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "extension": path.suffix.lower() or "<none>",
                "size_bytes": str(stat.st_size),
                "size_mb": f"{stat.st_size / (1024 * 1024):.2f}",
                "shape": shape,
                "dtype": dtype,
                "read_error": error,
            }
        )
    with (DOCS_DIR / "data_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "extension", "size_bytes", "size_mb", "shape", "dtype", "read_error"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_duplicate_candidates(inventory_rows: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in inventory_rows:
        if row["extension"] not in {".npy", ".npz"}:
            continue
        key = (row["extension"], row["size_bytes"], row["shape"], row["dtype"])
        groups.setdefault(key, []).append(row)

    output_rows = []
    group_num = 1
    for key, rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2])):
        if len(rows) < 2:
            continue
        group_id = f"candidate_{group_num:03d}"
        group_num += 1
        for row in rows:
            output_rows.append(
                {
                    "group_id": group_id,
                    "reason": "same extension, byte size, shape, and dtype; content not hash-verified",
                    "path": row["path"],
                    "size_bytes": row["size_bytes"],
                    "shape": row["shape"],
                    "dtype": row["dtype"],
                }
            )

    with (DOCS_DIR / "duplicate_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group_id", "reason", "path", "size_bytes", "shape", "dtype"],
        )
        writer.writeheader()
        writer.writerows(output_rows)


def write_organization_md(items: list[MoveItem], inventory_rows: list[dict[str, str]]) -> None:
    top_large = sorted(
        inventory_rows,
        key=lambda row: int(row["size_bytes"]),
        reverse=True,
    )[:20]
    counts: dict[str, int] = {}
    for item in items:
        counts[item.category] = counts.get(item.category, 0) + 1

    lines = [
        "# NMR Metabolomics Workspace Organization",
        "",
        "Generated by `tools/organize_workspace.py`.",
        "",
        "## Layout",
        "",
        "- `code/preprocessing`: alignment, water/EDTA suppression, normalization, deduplication scripts.",
        "- `code/training`: MAE/SSL training scripts and model definitions.",
        "- `code/evaluation`: test suites, classification, LOOCV, few-shot, and gradient analysis scripts.",
        "- `code/plotting`: plotting and comparison scripts.",
        "- `code/utils`: small inspection utilities.",
        "- `data`: large input and processed arrays, grouped by source/dataset.",
        "- `models`: current and old checkpoint folders.",
        "- `results`: experiment outputs, plots, diagnostics, and model architecture artifacts.",
        "- `archive`: legacy inventory output, Python caches, and uncategorized legacy folders.",
        "",
        "## Compatibility",
        "",
        "Moved top-level files and directories use canonical `code/`, `data/`, `models/`, and `results/` paths. Root-level compatibility symlinks are intentionally not preserved.",
        "",
        "## Move Summary",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`: {count} planned move(s)")

    lines.extend(
        [
            "",
            "## Largest Inventory Entries",
            "",
            "| Path | Size MB | Shape | Dtype |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in top_large:
        lines.append(f"| `{row['path']}` | {row['size_mb']} | `{row['shape']}` | `{row['dtype']}` |")

    lines.extend(
        [
            "",
            "## Supporting Files",
            "",
            "- `docs/move_plan.tsv`: source-to-target move list.",
            "- `docs/script_catalog.csv`: script purpose, category, and path hints.",
            "- `docs/data_inventory.csv`: artifact sizes plus `.npy/.npz` shapes and dtypes.",
            "- `docs/duplicate_candidates.csv`: duplicate-looking array groups by size/shape/dtype only.",
            "",
            "No files were intentionally deleted by the organizer.",
            "",
        ]
    )
    (DOCS_DIR / "ORGANIZATION.md").write_text("\n".join(lines), encoding="utf-8")


def merge_or_move(source: Path, target: Path) -> None:
    if not source.exists() and not source.is_symlink():
        return
    if target.exists():
        raise FileExistsError(f"Target already exists, refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def create_compat_symlink(source: Path, target: Path) -> None:
    if source.exists() or source.is_symlink():
        return
    link_target = relative_symlink_target(source, target)
    source.symlink_to(link_target, target_is_directory=target.is_dir())


def perform_moves(items: list[MoveItem]) -> None:
    for item in items:
        merge_or_move(item.source, item.target)
        if item.symlink:
            create_compat_symlink(item.source, item.target)


def main() -> None:
    os.chdir(ROOT)
    items = discover_move_items()

    write_move_plan(items)
    write_script_catalog(items)
    inventory_rows = write_data_inventory()
    write_duplicate_candidates(inventory_rows)
    write_organization_md(items, inventory_rows)

    perform_moves(items)

    print(f"Wrote docs in {DOCS_DIR.relative_to(ROOT)}")
    print(f"Moved {len(items)} top-level item(s)")
    print("Compatibility symlinks created for moved top-level entries where requested")


if __name__ == "__main__":
    main()
