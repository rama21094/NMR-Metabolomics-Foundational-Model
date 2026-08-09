#!/usr/bin/env python3
"""Build a pretraining-corpus subset that isolates the v3-vs-v4 effect.

Why this exists (docs/SSL_vs_classical_analysis.md §5f, §7b). Pretraining on the
v4 corpus transfers 0.069 WORSE on the held-out mean than pretraining on v3, at
byte-identical configuration, established over three replicates against a 0.020
noise floor. That is the largest effect measured anywhere in the project, and it
points the opposite way to the data-cleaning work.

A direct diff of the two corpora narrows it sharply:

    9,670 rows total, of which only 164 (1.7%) differ at all.
    Each differing row differs almost ENTIRELY -- 99.998% of its 131,072 points
    change, median max|dv| = 0.34 -- because the rows are min-max normalised, so
    changing a row's maximum rescales the whole row.

So 1.7% of the corpus appears to carry a 3.4x-noise-floor effect. The obvious
mechanism (v4 leaves a residual EDTA artefact that compresses the spectrum) was
tested and REFUTED: only 7 of the 164 rows have their maximum inside the EDTA
window in either version, and the v4 rows are slightly BRIGHTER outside it
(99.9th percentile 0.969 vs 0.917), i.e. less compressed rather than more.

This script builds the corpora needed to settle it by ablation:

  --mode common          The 9,506 rows that are identical in v3 and v4.
                         This is the decisive arm. Pretrain on it and compare
                         against the existing v3 (0.888) and v4 (0.820) runs:
                           ~0.888  -> v4's 164 rows are actively HARMFUL
                           ~0.820  -> v3's 164 rows are actively HELPFUL
                           between -> the effect is diffuse, and the 0.069 point
                                      estimate was partly run-to-run variance,
                                      which needs more than three replicates
  --mode random-control  v3 with the SAME NUMBER of rows dropped at random.
                         Size-matched control: it rules out the possibility that
                         the common subset differs simply because it is 1.7%
                         smaller. Cheap insurance -- run it on a second GPU
                         concurrently, so it costs no extra wall-clock.

RESULT of that ablation (docs §8): common9506 and the control landed within
0.001 of each other on the held-out mean (0.837 vs 0.836) -- INDISTINGUISHABLE.
Row content is refuted as the explanation. Both sit near v4 (0.820) and below
v3 (0.888), suggestively pointing at corpus SIZE (9506 vs 9670, a 1.7% cut) --
but that is unconfirmed on n=1 per arm, and a 1.7% cut producing a ~0.05 swing
would be disproportionate next to every capacity experiment in this project
(up to 2.9x more params moved accuracy <=0.02). Hence:

  --mode size-sweep      v3 with `--frac` of ALL rows dropped uniformly at
                         random (not restricted to the 164 differing rows --
                         this tests size in general, decoupled from content).
                         Sweep --frac over {0.01, 0.05, 0.10, ...} to see
                         whether held-out accuracy degrades smoothly with
                         corpus size. If it does, size is confirmed as (part
                         of) the story; if 5-10% cuts don't reproduce anything
                         near the 0.05 swing already seen at 1.7%, the original
                         common9506/control runs were likely an unlucky n=1
                         draw rather than a real size effect.

Output dtype and normalisation are preserved exactly (float64, already row
min-max normalised), because the comparison arms were trained that way and a
dtype change would be a second uncontrolled difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_V3 = ("data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_"
              "Suppressed_rowMinMax_v3.npy")
DEFAULT_V4 = ("data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_"
              "Suppressed_rowMinMax_v4.npy")


def find_differing_rows(v3, v4, tol: float, chunk: int):
    """Row indices where the two corpora differ by more than `tol` at any point."""
    idx, stats = [], []
    for s in range(0, v3.shape[0], chunk):
        a = np.asarray(v3[s:s + chunk])
        b = np.asarray(v4[s:s + chunk])
        d = np.abs(a - b)
        rowmax = d.max(axis=1)
        hit = np.nonzero(rowmax > tol)[0]
        idx.extend((s + hit).tolist())
        stats.extend(rowmax[hit].tolist())
    return np.array(idx, dtype=np.int64), np.array(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3", default=DEFAULT_V3)
    ap.add_argument("--v4", default=DEFAULT_V4)
    ap.add_argument("--mode", choices=["common", "random-control", "size-sweep"], default="common")
    ap.add_argument("--frac", type=float, default=None,
                    help="--mode size-sweep only: fraction of ALL v3 rows to drop uniformly "
                         "at random (e.g. 0.05 for 5%%), decoupled from the 164 differing rows.")
    ap.add_argument("--out", default=None, help="Output .npy. Default is derived from --mode.")
    ap.add_argument("--seed", type=int, default=7,
                    help="Seed for --mode random-control / size-sweep.")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Two rows count as identical if max|dv| <= tol.")
    ap.add_argument("--chunk", type=int, default=250,
                    help="Rows per I/O chunk. 250 keeps peak RAM near 0.5 GB.")
    ap.add_argument("--index-dir", default="results/analysis/corpus_v3_v4_diff")
    ap.add_argument("--recompute-diff", action="store_true",
                    help="Rescan v3 vs v4 for differing rows even if a cached "
                         "differing_rows.csv already exists in --index-dir.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written, then stop before writing the .npy.")
    args = ap.parse_args()

    if args.mode == "size-sweep" and args.frac is None:
        raise SystemExit("--mode size-sweep requires --frac")

    v3 = np.load(args.v3, mmap_mode="r")
    n, length = v3.shape
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    cache = index_dir / "differing_rows.csv"

    if args.mode == "size-sweep":
        # Pure size question, decoupled from the 164 differing rows -- no need
        # to load v4 or scan for the diff at all.
        v4 = None
        print(f"v3: {args.v3}\nshape {v3.shape}  dtype {v3.dtype}\n")
        diff_idx = np.array([], dtype=np.int64)
        n_diff = 0
    else:
        v4 = np.load(args.v4, mmap_mode="r")
        if v3.shape != v4.shape:
            raise SystemExit(f"shape mismatch: v3 {v3.shape} vs v4 {v4.shape} — not comparable")
        print(f"v3: {args.v3}\nv4: {args.v4}\nshape {v3.shape}  dtype {v3.dtype}\n")

        if cache.exists() and not args.recompute_diff:
            diff_idx = np.loadtxt(cache, skiprows=1, dtype=np.int64).reshape(-1)
            print(f"Using cached differing-row list: {cache} ({len(diff_idx)} rows). "
                  f"Pass --recompute-diff to rescan.")
        else:
            print("Scanning for differing rows ...", flush=True)
            diff_idx, diff_mag = find_differing_rows(v3, v4, args.tol, args.chunk)
            if len(diff_idx):
                print(f"  max|dv| among them: median {np.median(diff_mag):.4f}  max {diff_mag.max():.4f}")
            np.savetxt(cache, diff_idx, fmt="%d", header="row_index", comments="")
        n_diff = len(diff_idx)
        print(f"  rows differing: {n_diff}/{n} ({100 * n_diff / n:.2f}%)")

    # Choose which rows to drop, and from which source corpus.
    if args.mode == "size-sweep":
        source, source_name = v3, "v3"
        rng = np.random.default_rng(args.seed)
        n_drop = int(round(args.frac * n))
        drop = np.sort(rng.choice(n, size=n_drop, replace=False))
        default_out = (f"data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_"
                       f"Suppressed_rowMinMax_v3drop{round(args.frac * 100)}pct_seed{args.seed}.npy")
    elif args.mode == "common":
        source, source_name = v3, "v3"
        drop = diff_idx
        # The kept rows are identical in both by construction, so either corpus
        # is a valid source; v3 is used so the file is bit-identical to a v3
        # subset and the arm differs from the v3 run ONLY by the missing rows.
        default_out = (f"data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_"
                       f"Suppressed_rowMinMax_common{n - n_diff}.npy")
    else:
        source, source_name = v3, "v3"
        rng = np.random.default_rng(args.seed)
        eligible = np.setdiff1d(np.arange(n), diff_idx)
        if n_diff > len(eligible):
            raise SystemExit("cannot drop more rows than are eligible")
        # Drop from the UNCHANGED rows only, so this control keeps exactly the
        # same 164 special rows the full v3 run had -- it varies size alone.
        drop = np.sort(rng.choice(eligible, size=n_diff, replace=False))
        default_out = (f"data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_"
                       f"Suppressed_rowMinMax_v3rand{n - n_diff}_seed{args.seed}.npy")

    out_path = Path(args.out or default_out)
    keep = np.setdiff1d(np.arange(n), drop)
    print(f"\nmode={args.mode}  source={source_name}  keeping {len(keep)} rows, dropping {len(drop)}")
    print(f"out: {out_path}")
    est_gb = len(keep) * length * source.dtype.itemsize / 1e9
    print(f"estimated size: {est_gb:.2f} GB")

    if args.dry_run:
        print("\n--dry-run: stopping before write.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(out_path, mode="w+",
                                    dtype=source.dtype, shape=(len(keep), length))
    for w in range(0, len(keep), args.chunk):
        rows = keep[w:w + args.chunk]
        out[w:w + len(rows)] = source[rows]
        if w % (args.chunk * 20) == 0:
            print(f"  wrote {w:6d}/{len(keep)}", flush=True)
    out.flush()

    # Verify: for the common subset every kept row must match BOTH corpora.
    if args.mode == "common":
        print("\nVerifying kept rows are identical in v3 and v4 ...", flush=True)
        worst = 0.0
        for w in range(0, len(keep), args.chunk):
            rows = keep[w:w + args.chunk]
            worst = max(worst, float(np.abs(np.asarray(v3[rows]) - np.asarray(v4[rows])).max()))
        print(f"  worst max|dv| over kept rows: {worst:.3e}  "
              f"({'PASS' if worst <= args.tol else 'FAIL'})")
        if worst > args.tol:
            raise SystemExit("kept rows are not identical across corpora — aborting")

    meta = {
        "mode": args.mode,
        "source_corpus": source_name,
        "v3": args.v3, "v4": args.v4 if args.mode != "size-sweep" else None,
        "frac": args.frac if args.mode == "size-sweep" else None,
        "n_total": int(n), "n_kept": int(len(keep)), "n_dropped": int(len(drop)),
        "n_differing_rows": int(n_diff),
        "dropped_row_indices": drop.tolist(),
        "tol": args.tol,
        "seed": args.seed if args.mode in ("random-control", "size-sweep") else None,
        "dtype": str(source.dtype),
        "output": str(out_path),
        "sha256_first_mb": hashlib.sha256(
            np.asarray(out[0]).tobytes()[:1_000_000]).hexdigest(),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_path}\nWrote {meta_path}")
    print(f"Dropped-row indices are recorded in {meta_path} and "
          f"{index_dir / 'differing_rows.csv'}")


if __name__ == "__main__":
    main()
