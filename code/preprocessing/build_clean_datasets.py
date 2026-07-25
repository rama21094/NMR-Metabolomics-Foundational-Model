"""Rebuild corrected, verified water/EDTA-suppressed + row-min-max normalized
spectra for all four datasets used in this project (the pretraining corpus,
Barth, MTBLS326, MTBLS563).

Background (see results/analysis/suppression_validation/ for the full
diagnosis): validate_suppression.py found the water-suppression window was
NOT masked in 69.7% of the pretraining corpus and 100% of the Barth
evaluation data (a bug in which npy file the Barth eval scripts point to --
they use the raw pre-pipeline file, not preprocess_barth_syndrome.py's actual
output).

EDTA suppression went through two earlier revisions before landing on the
current approach (suppress_edta_magnitude.py), both of which asked the wrong
question ("is this peak tall relative to noise/other peaks *within the
72000-74000 window*?"):
  - suppress_edta_peak.py's original dominance-ratio gate: confirmed-EDTA and
    confirmed-Heparin Barth rows have statistically indistinguishable
    dominance ratios, so it almost never fires (1/9670 on the full corpus).
  - suppress_edta_nuanced.py's local-prominence-SNR gate: fires on 32/33
    non-EDTA Barth rows too, AND (caught by direct visual inspection of
    row 5) suppressed a real 4-line J-coupling multiplet at ~10% of that
    row's own peak scale -- clearly not a normalization problem, and clearly
    not what should be removed.

The actual goal (per direct instruction) is narrower and metadata-free:
suppress a peak in the EDTA region only when its magnitude is wildly
different from the rest of that row's real peaks, since THAT is what
corrupts row-min-max normalization -- a peak merely comparable to other real
peaks is not a problem, regardless of whether it's chemically EDTA.
suppress_edta_magnitude.py implements exactly this (peak height compared to
the row's own max elsewhere) and is applied identically to all four
datasets, with no per-dataset metadata gating.

Every dataset gets: (1) hard water-window suppression (threshold-gated: only
overwrite if the window isn't already ~flat), (2) the same magnitude-based
EDTA suppression, (3) row-wise min-max normalization. Outputs get a "_v2"
suffix so they're unambiguously distinct from the old, bugged files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suppress_edta_magnitude as edta_magnitude  # noqa: E402
from row_minmax_normalize import row_minmax_normalize  # noqa: E402

WATER_START, WATER_STOP = 62_500, 68_000
WATER_THRESHOLD = 1e-3
CHUNK_SIZE = 256


def suppress_water(input_path: Path, output_path: Path, chunk_size: int = CHUNK_SIZE) -> dict:
    spectra = np.load(input_path, mmap_mode="r")
    output = open_memmap(output_path, mode="w+", dtype=np.float64, shape=spectra.shape)
    n_rows = spectra.shape[0]
    changed = 0
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        chunk = np.asarray(spectra[start:stop], dtype=np.float64).copy()
        region = chunk[:, WATER_START:WATER_STOP]
        row_changed = np.any(np.abs(region) > WATER_THRESHOLD, axis=1)
        if np.any(row_changed):
            chunk[row_changed, WATER_START:WATER_STOP] = 0.0
            changed += int(row_changed.sum())
        output[start:stop] = chunk
        print(f"  water suppression rows {start}:{stop}/{n_rows}")
    output.flush()
    return {"rows": int(n_rows), "changed_rows": int(changed)}


def edta_magnitude_detector(input_path: Path, output_path: Path, **_) -> dict:
    """Uniform, metadata-free EDTA suppression: suppress only peaks in the
    search window whose height is a large fraction of the row's own peak
    scale elsewhere (see suppress_edta_magnitude.py for the full rationale).
    Applied identically to every dataset."""
    spectra = np.load(input_path, mmap_mode="r")
    output = open_memmap(output_path, mode="w+", dtype=np.float64, shape=spectra.shape)
    n = spectra.shape[0]
    rows_out = []
    n_suppressed_rows = 0
    for i in range(n):
        spectrum = np.asarray(spectra[i], dtype=np.float64)
        detections = edta_magnitude.detect_dominant_peaks(spectrum)
        if detections:
            output[i] = edta_magnitude.suppress_detections(spectrum, detections, row_index=i)
            n_suppressed_rows += 1
        else:
            output[i] = spectrum
        rows_out.append({
            "row_index": i, "n_peaks_suppressed": len(detections),
            "peak_indices": ";".join(str(d["peak_index"]) for d in detections),
            "ratios": ";".join(f"{d['ratio_to_row_max']:.3f}" for d in detections),
        })
        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"  EDTA (magnitude-based) rows {i + 1}/{n}")
    output.flush()
    return {"rows": int(n), "n_rows_suppressed": n_suppressed_rows, "diagnostics": rows_out}


DATASETS = {
    "train_corpus": dict(
        input="data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed.npy",
        water_output="data/combined/combine_unique_MetaboLights_Workbench_WaterFixed_v3.npy",
        edta_output="data/combined/combine_unique_MetaboLights_Workbench_WaterFixed_EDTAMagnitude_v3.npy",
        final_output="data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3.npy",
        edta_strategy=edta_magnitude_detector,
        edta_kwargs={},
    ),
    "barth": dict(
        input="data/Barth/aligned_128K_Workbench_Barth_Syndrome.npy",
        water_output="data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_v3.npy",
        edta_output="data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_v3.npy",
        final_output="data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_rowMinMax_v3.npy",
        edta_strategy=edta_magnitude_detector,
        edta_kwargs={},
    ),
    "mtbls326": dict(
        input="data/mtbls326/MTBLS326_aligned_spectra.npy",
        water_output="data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_v3.npy",
        edta_output="data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_EDTASuppressed_v3.npy",
        final_output="data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax_v3.npy",
        edta_strategy=edta_magnitude_detector,
        edta_kwargs={},
    ),
    "mtbls563": dict(
        input="data/mtbls563/MTBLS563_aligned_spectra.npy",
        water_output="data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_v3.npy",
        edta_output="data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_EDTASuppressed_v3.npy",
        final_output="data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_rowMinMax_v3.npy",
        edta_strategy=edta_magnitude_detector,
        edta_kwargs={},
    ),
}


def run_dataset(name: str, cfg: dict, out_dir: Path) -> dict:
    print(f"\n=== {name} ===")
    input_path = Path(cfg["input"])
    water_output = Path(cfg["water_output"])
    edta_output = Path(cfg["edta_output"])
    final_output = Path(cfg["final_output"])
    for p in (water_output, edta_output, final_output):
        p.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1/3: water suppression")
    water_summary = suppress_water(input_path, water_output)
    print(f"  changed {water_summary['changed_rows']}/{water_summary['rows']} rows")

    print("Step 2/3: EDTA suppression")
    edta_summary = cfg["edta_strategy"](water_output, edta_output, **cfg["edta_kwargs"])
    print(f"  {edta_summary}")

    print("Step 3/3: row-wise min-max normalization")
    diag_path = out_dir / f"{name}_rowminmax_diagnostics.csv"
    row_minmax_normalize(edta_output, final_output, diagnostics_path=diag_path, overwrite=True)

    summary = {"dataset": name, "input": str(input_path), "final_output": str(final_output),
               "water_summary": water_summary, "edta_summary": {k: v for k, v in edta_summary.items() if k != "diagnostics"}}
    if "diagnostics" in edta_summary:
        diag_df = pd.DataFrame(edta_summary["diagnostics"])
        diag_df.to_csv(out_dir / f"{name}_edta_diagnostics.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()), default=list(DATASETS.keys()))
    parser.add_argument("--out-dir", default="results/analysis/preprocessing_v2")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in args.datasets:
        summaries.append(run_dataset(name, DATASETS[name], out_dir))

    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nWrote summary to {out_dir / 'build_summary.json'}")
    for s in summaries:
        print(f"  {s['dataset']}: {s['final_output']}")


if __name__ == "__main__":
    main()
