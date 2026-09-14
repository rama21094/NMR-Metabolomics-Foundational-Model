#!/usr/bin/env python3
"""Drop the acquisition failures identified by code/analysis/flag_atypical_spectra.py.

Removes only rows flagged by >=2 of 3 independent atypicality scores AND visually
confirmed to be processing failures rather than unusual biology: inverted phase,
severe baseline roll, and one essentially empty spectrum (see
results/figures/fig_atypical_spectra.png).

DELIBERATELY NARROW. Atypical is not the same as wrong. A real disease phenotype
should look unusual against a mostly healthy corpus, and filtering the corpus
toward typicality would worsen the near-duplication problem that
docs/SSL_vs_classical_analysis.md section 19 identifies as the core defect. Only
spectra that are broken as MEASUREMENTS are removed here. Row 6200, which
prompted the whole enquiry, is NOT removed -- it sits at the 19th percentile of
shape correlation, inside normal corpus diversity.

The row mapping is written out, because every earlier per-row artifact (lags,
shifts, atypicality scores) is indexed against the 9,670-row corpus and will
otherwise silently misalign against the 9,635-row one.

Usage:
    python code/preprocessing/drop_broken_spectra.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ("data/combined/combine_unique_MetaboLights_Workbench_"
       "Water_EDTA_Suppressed_ref0ppm.npy")
DST = ("data/combined/combine_unique_MetaboLights_Workbench_"
       "Water_EDTA_Suppressed_ref0ppm_clean.npy")
SCORES = "results/analysis/atypical/atypical_scores.csv"
OUT = ROOT / "results/analysis/atypical"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--scores", default=SCORES,
                    help="atypicality CSV matching --src; a mismatched file "
                         "would drop rows by index against a different corpus")
    args = ap.parse_args()

    d = pd.read_csv(ROOT / args.scores)
    drop = np.sort(d.loc[d["votes"] >= args.min_votes, "row"].to_numpy())
    X = np.load(ROOT / args.src, mmap_mode="r")
    n = X.shape[0]
    keep = np.setdiff1d(np.arange(n), drop)

    print(f"source {Path(args.src).name}: {n:,} spectra")
    print(f"dropping {len(drop)} flagged by >={args.min_votes}/3 methods")
    print(f"  rows: {drop.tolist()}")
    print(f"keeping {len(keep):,}\n")

    Y = np.lib.format.open_memmap(ROOT / args.dst, mode="w+",
                                  dtype=X.dtype, shape=(len(keep), X.shape[1]))
    for i in range(0, len(keep), 200):
        j = min(i + 200, len(keep))
        Y[i:j] = X[keep[i:j]]
        print(f"\r  written {j}/{len(keep)}", end="", flush=True)
    Y.flush(); del Y
    print()

    pd.DataFrame({"new_row": np.arange(len(keep)), "old_row": keep}).to_csv(
        OUT / "row_map_clean.csv", index=False)
    (OUT / "drop_summary.json").write_text(json.dumps({
        "source": args.src, "dest": args.dst,
        "n_before": int(n), "n_after": int(len(keep)),
        "dropped_rows": drop.tolist(),
        "min_votes": args.min_votes}, indent=2))
    print(f"wrote {args.dst}")
    print(f"wrote {(OUT / 'row_map_clean.csv').relative_to(ROOT)}  "
          "(every per-row artifact is indexed on the 9,670-row corpus; "
          "use this to translate)")


if __name__ == "__main__":
    main()
