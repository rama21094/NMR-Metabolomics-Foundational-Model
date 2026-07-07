#!/usr/bin/env python3
"""Preprocess Barth syndrome spectra with water, EDTA, and row min-max steps.

Pipeline:
1. Set the fixed water window 62500:68000 to zero.
2. Conservatively detect and suppress prominent EDTA peaks.
3. Apply row-wise min-max normalization.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from numpy.lib.format import open_memmap


ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = ROOT / "code" / "preprocessing"
for path in (ROOT, PREPROCESSING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import suppress_edta_peak as edta  # noqa: E402
from row_minmax_normalize import row_minmax_normalize  # noqa: E402


DEFAULT_INPUT = Path("data/Barth/aligned_128K_Workbench_Barth_Syndrome.npy")
DEFAULT_WATER_OUTPUT = Path("data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero.npy")
DEFAULT_EDTA_OUTPUT = Path("data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed.npy")
DEFAULT_FINAL_OUTPUT = Path(
    "data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_rowMinMax.npy"
)
DEFAULT_EDTA_DIAGNOSTICS = Path("data/Barth/barth_edta_suppression_diagnostics.csv")
DEFAULT_ROWMINMAX_DIAGNOSTICS = Path("data/Barth/barth_row_minmax_diagnostics.csv")


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    spectra = np.load(path, mmap_mode="r")
    if spectra.ndim != 2:
        raise ValueError(f"Expected a 2D spectra array, got shape {spectra.shape}")
    return spectra


def write_edta_diagnostics(path: Path, rows: list[dict]) -> None:
    fields = [
        "row_index",
        "status",
        "peak_index",
        "peak_value",
        "baseline",
        "noise_sd",
        "prominence",
        "prominence_snr",
        "dominance_ratio",
        "left_index",
        "right_index",
        "suppression_width",
        "cutoff",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def suppress_water_window(
    input_path: Path,
    output_path: Path,
    lower_bound: int,
    upper_bound: int,
    threshold: float,
    overwrite: bool,
    chunk_size: int,
) -> dict:
    spectra = validate_matrix(input_path)
    if not (0 <= lower_bound < upper_bound <= spectra.shape[1]):
        raise ValueError(f"Invalid water window [{lower_bound}:{upper_bound}] for spectra width {spectra.shape[1]}")
    ensure_can_write(output_path, overwrite)

    output = open_memmap(output_path, mode="w+", dtype=spectra.dtype, shape=spectra.shape)
    changed_rows = 0
    n_rows = spectra.shape[0]
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        chunk = np.asarray(spectra[start:stop]).copy()
        region = chunk[:, lower_bound:upper_bound]
        changed = np.any(np.abs(region) > threshold, axis=1)
        if np.any(changed):
            chunk[changed, lower_bound:upper_bound] = 0.0
            changed_rows += int(np.sum(changed))
        output[start:stop] = chunk
        print(f"Water suppression rows {start}:{stop} / {n_rows}")
    output.flush()
    return {"rows": int(n_rows), "changed_rows": int(changed_rows), "output": str(output_path)}


def configure_edta_detector(args) -> None:
    edta.SEARCH_START = int(args.edta_search_start)
    edta.SEARCH_STOP = int(args.edta_search_stop)
    edta.MIN_PROMINENCE_SNR = float(args.edta_min_prominence_snr)
    edta.MIN_DOMINANCE_RATIO = float(args.edta_min_dominance_ratio)
    edta.MIN_SUPPRESSION_WIDTH = int(args.edta_min_suppression_width)
    edta.MAX_SUPPRESSION_WIDTH = int(args.edta_max_suppression_width)
    edta.FILL_METHOD = str(args.edta_fill_method)
    edta.FILL_RANDOM_SEED = int(args.seed)


def suppress_edta(
    input_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    spectra = validate_matrix(input_path)
    if not (0 <= edta.SEARCH_START < edta.SEARCH_STOP <= spectra.shape[1]):
        raise ValueError(f"Invalid EDTA search window [{edta.SEARCH_START}:{edta.SEARCH_STOP}]")
    if not dry_run:
        ensure_can_write(output_path, overwrite)
        output = open_memmap(output_path, mode="w+", dtype=spectra.dtype, shape=spectra.shape)
    else:
        output = None

    detections = []
    for row_index in range(spectra.shape[0]):
        spectrum = np.asarray(spectra[row_index])
        detection = edta.detect_edta_peak(spectrum)
        detection["row_index"] = row_index
        detections.append(detection)

        if output is not None:
            if detection["status"] == "suppressed":
                output[row_index] = edta.suppress_detected_peak(spectrum, detection)
            else:
                output[row_index] = spectrum

        if (row_index + 1) % 100 == 0 or row_index + 1 == spectra.shape[0]:
            print(f"EDTA detection rows {row_index + 1}/{spectra.shape[0]}")

    if output is not None:
        output.flush()

    write_edta_diagnostics(diagnostics_path, detections)
    statuses, counts = np.unique([d["status"] for d in detections], return_counts=True)
    status_counts = {str(status): int(count) for status, count in zip(statuses, counts)}
    return {
        "rows": int(spectra.shape[0]),
        "status_counts": status_counts,
        "output": None if dry_run else str(output_path),
        "diagnostics": str(diagnostics_path),
        "dry_run": bool(dry_run),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--water-output", type=Path, default=DEFAULT_WATER_OUTPUT)
    parser.add_argument("--edta-output", type=Path, default=DEFAULT_EDTA_OUTPUT)
    parser.add_argument("--final-output", type=Path, default=DEFAULT_FINAL_OUTPUT)
    parser.add_argument("--edta-diagnostics", type=Path, default=DEFAULT_EDTA_DIAGNOSTICS)
    parser.add_argument("--rowminmax-diagnostics", type=Path, default=DEFAULT_ROWMINMAX_DIAGNOSTICS)
    parser.add_argument("--water-start", type=int, default=62500)
    parser.add_argument("--water-stop", type=int, default=68000)
    parser.add_argument("--water-threshold", type=float, default=1e-3)
    parser.add_argument("--edta-search-start", type=int, default=72000)
    parser.add_argument("--edta-search-stop", type=int, default=74000)
    parser.add_argument("--edta-min-prominence-snr", type=float, default=50.0)
    parser.add_argument("--edta-min-dominance-ratio", type=float, default=3.0)
    parser.add_argument("--edta-min-suppression-width", type=int, default=10)
    parser.add_argument("--edta-max-suppression-width", type=int, default=600)
    parser.add_argument(
        "--edta-fill-method",
        choices=["local_noise", "local_baseline", "boundary_interpolate", "zero"],
        default="local_noise",
    )
    parser.add_argument("--dry-run-edta", type=str2bool, default=False)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_edta_detector(args)

    print("Step 1/3: water suppression")
    water_summary = suppress_water_window(
        input_path=args.input,
        output_path=args.water_output,
        lower_bound=args.water_start,
        upper_bound=args.water_stop,
        threshold=args.water_threshold,
        overwrite=args.overwrite,
        chunk_size=args.chunk_size,
    )
    print(f"Water output: {water_summary['output']}")
    print(f"Rows changed in water window: {water_summary['changed_rows']}/{water_summary['rows']}")

    print("\nStep 2/3: EDTA suppression")
    edta_summary = suppress_edta(
        input_path=args.water_output,
        output_path=args.edta_output,
        diagnostics_path=args.edta_diagnostics,
        overwrite=args.overwrite,
        dry_run=args.dry_run_edta,
    )
    print(f"EDTA diagnostics: {edta_summary['diagnostics']}")
    for status, count in sorted(edta_summary["status_counts"].items()):
        print(f"{status}: {count}")
    if args.dry_run_edta:
        print("Dry-run requested; skipping row min-max normalization because EDTA output was not written.")
        return

    print("\nStep 3/3: row-wise min-max normalization")
    row_minmax_normalize(
        input_path=args.edta_output,
        output_path=args.final_output,
        diagnostics_path=args.rowminmax_diagnostics,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    print("\nPipeline complete.")
    print(f"Final output: {args.final_output}")


if __name__ == "__main__":
    main()
