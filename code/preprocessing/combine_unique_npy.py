#!/usr/bin/env python3
"""Combine one or more .npy files, remove duplicate rows, and save a unique output.

IDE-friendly usage:
- Edit INPUT_FILES / OUTPUT_FILE / SKIP_COPY below.
- Run this script directly in your IDE.

CLI usage (optional):
    python combine_unique_npy.py file1.npy file2.npy --output data/combined/combined_unique.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

# ---------------- IDE-friendly config ----------------
INPUT_FILES: List[str] = [
    "data/plasma/plasma_unique_EDTASuppressed.npy",
    "data/serum/serum_unique_WS625to680Zero.npy",
]
OUTPUT_FILE: str = "data/combined/combine_unique_Water_EDTA_Suppressed.npy"
SKIP_COPY: bool = False
# -----------------------------------------------------


def copy_npy_file(src_path: Path) -> Path:
    arr = np.load(src_path, allow_pickle=False)
    dst_path = src_path.with_name(src_path.stem + "_copy" + src_path.suffix)
    np.save(dst_path, arr)
    return dst_path


def _load_and_validate(paths: List[Path], skip_copy: bool) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []

    for p in paths:
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p}")

        if skip_copy:
            arr = np.load(p, allow_pickle=False)
            print(f"Loaded: {p} -> {arr.shape}")
        else:
            copied = copy_npy_file(p)
            print(f"Saved copy: {copied}")
            arr = np.load(copied, allow_pickle=False)
            print(f"Loaded copy: {copied} -> {arr.shape}")

        if arr.ndim != 2:
            raise ValueError(f"{p} must be 2D, got shape {arr.shape}")
        arrays.append(arr)

    n_points = arrays[0].shape[1]
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape[1] != n_points:
            raise ValueError(
                f"All files must have same number of columns. "
                f"Expected {n_points}, got {arr.shape[1]} in {paths[i]}"
            )

    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine one or more .npy files, deduplicate rows, and save one output."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Input .npy files. If omitted, INPUT_FILES from script is used.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Output .npy file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        default=SKIP_COPY,
        help="Skip creating _copy files and only create combined output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chosen_files = args.files if args.files else INPUT_FILES
    if len(chosen_files) == 0:
        raise SystemExit("No input files provided. Set INPUT_FILES or pass files via CLI.")

    paths = [Path(f) for f in chosen_files]
    output_path = Path(args.output)

    print(f"Input file count: {len(paths)}")
    arrays = _load_and_validate(paths, skip_copy=args.skip_copy)

    combined = np.vstack(arrays)
    print(f"Combined shape before deduplication: {combined.shape}")

    unique_combined = np.unique(combined, axis=0)
    print(f"Shape after deduplication: {unique_combined.shape}")

    np.save(output_path, unique_combined)
    print(f"Saved combined unique file to: {output_path}")


if __name__ == "__main__":
    main()
