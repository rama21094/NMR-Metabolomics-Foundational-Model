#!/usr/bin/env python3
"""Compare two or more .npy files as sets of rows.

Edit FILES in this script, then run:
    python check_npy_compare.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Edit this list in your IDE.
FILES: List[str] = [
    # "data/combined/combined_unique.npy",
    # "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero.npy",
    # "data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero.npy",
    "data/plasma/aligned_nmr_spectra_128K_Plasma_WS625to680Zero.npy",
    "data/plasma/aligned_nmr_spectra_128K_Plasma_NoSuppress.npy"
]


def row_tokens(arr: np.ndarray) -> np.ndarray:
    """Return hashable row-wise tokens for exact row comparison."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D/2D array, got {arr.ndim}D")
    c = np.ascontiguousarray(arr)
    token_dtype = np.dtype((np.void, c.dtype.itemsize * c.shape[1]))
    return c.view(token_dtype).ravel()


def relation(set_a: set, set_b: set) -> str:
    if set_a == set_b:
        return "same"
    if set_a.issubset(set_b):
        return "A is subset of B"
    if set_a.issuperset(set_b):
        return "A is superset of B"
    if set_a.isdisjoint(set_b):
        return "disjoint (no overlap)"
    return "partial overlap"


def load_file(path: str) -> Tuple[np.ndarray, np.ndarray, set]:
    arr = np.load(path, allow_pickle=False)
    tokens = row_tokens(arr)
    token_set = set(tokens.tolist())
    return arr, tokens, token_set


def main() -> None:
    if len(FILES) < 2:
        raise SystemExit("Please add at least 2 .npy filenames to FILES.")

    missing = [f for f in FILES if not Path(f).exists()]
    if missing:
        raise SystemExit(f"Missing file(s): {missing}")

    data: Dict[str, Dict[str, object]] = {}
    for f in FILES:
        arr, tokens, token_set = load_file(f)
        n_rows = arr.shape[0] if arr.ndim > 1 else 1
        n_unique = len(token_set)
        data[f] = {
            "arr": arr,
            "tokens": tokens,
            "set": token_set,
            "n_rows": n_rows,
            "n_unique": n_unique,
        }

    print("=== File Summary ===")
    for f in FILES:
        arr = data[f]["arr"]
        n_rows = data[f]["n_rows"]
        n_unique = data[f]["n_unique"]
        print(f"{f}")
        print(f"  shape={arr.shape}, dtype={arr.dtype}")
        print(f"  rows={n_rows}, unique_rows={n_unique}, duplicate_rows={n_rows - n_unique}")

    print("\n=== Pairwise Comparison ===")
    for fa, fb in combinations(FILES, 2):
        sa = data[fa]["set"]
        sb = data[fb]["set"]
        inter = sa & sb

        only_a = len(sa - sb)
        only_b = len(sb - sa)
        overlap = len(inter)

        rel = relation(sa, sb)

        print(f"{fa}  vs  {fb}")
        print(f"  relation: {rel}")
        print(f"  overlap_unique_rows: {overlap}")
        print(f"  only_in_A_unique_rows: {only_a}")
        print(f"  only_in_B_unique_rows: {only_b}")
        print(f"  A_coverage_by_B: {overlap}/{len(sa)}")
        print(f"  B_coverage_by_A: {overlap}/{len(sb)}")


if __name__ == "__main__":
    main()
