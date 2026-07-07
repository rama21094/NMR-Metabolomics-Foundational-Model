"""Apply row-wise min-max normalization to a 2D spectra .npy file.

Each spectrum is normalized independently:

    normalized = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

Constant rows are written as zeros by default. The script streams through rows
with memory mapping, so it can be reused for large spectra matrices.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


# =========================
# IDE-friendly configuration
# =========================
INPUT_FILE = "./data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero.npy"
OUTPUT_FILE = "./data/BrC_T2D/BC_T2D_aligned_spectra_WS625to680Zero_rowMinMax.npy"
DIAGNOSTICS_FILE = "./data/BrC_T2D/row_minmax_diagnostics.csv"
CHUNK_SIZE = 256
OVERWRITE = False


def default_output_path(input_path: Path) -> Path:
    """Return a sibling path with `_rowMinMax` added before `.npy`."""
    return input_path.with_name(f"{input_path.stem}_rowMinMax.npy")


def normalize_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a 2D chunk row-wise and return diagnostics arrays."""
    chunk = np.asarray(chunk, dtype=np.float64)
    chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

    row_min = np.min(chunk, axis=1, keepdims=True)
    row_max = np.max(chunk, axis=1, keepdims=True)
    row_range = row_max - row_min
    non_constant = row_range[:, 0] > np.finfo(np.float64).eps

    normalized = np.zeros_like(chunk, dtype=np.float64)
    normalized[non_constant] = (
        chunk[non_constant] - row_min[non_constant]
    ) / row_range[non_constant]

    return normalized, row_min[:, 0], row_max[:, 0], row_range[:, 0]


def write_diagnostics(
    diagnostics_path: Path,
    row_min: np.ndarray,
    row_max: np.ndarray,
    row_range: np.ndarray,
) -> None:
    """Write per-row min/max/range diagnostics."""
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "input_min", "input_max", "input_range", "constant_row"])
        for i, (mn, mx, span) in enumerate(zip(row_min, row_max, row_range)):
            writer.writerow([i, float(mn), float(mx), float(span), bool(span <= np.finfo(np.float64).eps)])


def row_minmax_normalize(
    input_path: Path,
    output_path: Path,
    diagnostics_path: Path | None = None,
    chunk_size: int = 256,
    overwrite: bool = False,
) -> None:
    """Stream a 2D .npy matrix and save row-wise min-max normalized output."""
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if diagnostics_path is not None:
        diagnostics_path = diagnostics_path.expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.suffix != ".npy":
        raise ValueError(f"Expected a .npy input file, got: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            "Use --overwrite if you want to replace it."
        )

    spectra = np.load(input_path, mmap_mode="r")
    if spectra.ndim != 2:
        raise ValueError(f"Expected a 2D spectra array, got shape {spectra.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = open_memmap(output_path, mode="w+", dtype=np.float64, shape=spectra.shape)

    n_rows = spectra.shape[0]
    all_min = np.empty(n_rows, dtype=np.float64)
    all_max = np.empty(n_rows, dtype=np.float64)
    all_range = np.empty(n_rows, dtype=np.float64)

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        normalized, row_min, row_max, row_range = normalize_chunk(spectra[start:stop])
        output[start:stop] = normalized
        all_min[start:stop] = row_min
        all_max[start:stop] = row_max
        all_range[start:stop] = row_range
        print(f"Normalized rows {start}:{stop} / {n_rows}")

    output.flush()

    if diagnostics_path is not None:
        write_diagnostics(diagnostics_path, all_min, all_max, all_range)

    print("Row-wise min-max normalization complete.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if diagnostics_path is not None:
        print(f"Diagnostics: {diagnostics_path}")
    print(f"Shape: {spectra.shape}")
    print(f"Constant rows: {int(np.sum(all_range <= np.finfo(np.float64).eps))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_FILE, help="Input 2D spectra .npy file.")
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="Output .npy file. Defaults to the configured BrC/T2D rowMinMax path.",
    )
    parser.add_argument(
        "--diagnostics",
        default=DIAGNOSTICS_FILE,
        help="Optional diagnostics CSV path. Use '' to skip writing diagnostics.",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    diagnostics_path = Path(args.diagnostics) if args.diagnostics else None

    row_minmax_normalize(
        input_path=input_path,
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
