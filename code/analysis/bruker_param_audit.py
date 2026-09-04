#!/usr/bin/env python3
"""Audit the Bruker acquisition/processing parameters behind the training corpus.

Input: the two CSVs produced by code/preprocessing/extract_bruker_params.py on
the raw HDD, one per source pull:

    data/bruker_params_PlasmaNMRData.csv  -> aligned_128K_Plasma_NoSuppress.npy
    data/bruker_params_SerumNMRData.csv   -> aligned_nmr_spectra_128K_WSNoise.npy

These are the two branches that make up the 9,670-spectrum corpus (see
docs/PI_outline.md 5g and the provenance tree in docs/ORGANIZATION.md).

The questions this answers, in priority order:

1. FIELD STRENGTH.  The GISSMO basis in code/synthesis/build_basis.py is
   simulated at a single field (600 MHz).  If the corpus is mixed-field, every
   J-multiplet in the basis is wrong for part of the corpus, because J splittings
   are constant in Hz while chemical shifts scale with field -- so the ppm gap
   between multiplet lines changes with field.  That would be an independent
   source of misfit on top of alignment.

2. PPM AXIS CONSISTENCY.  This is the fit_gate question, and the first version of
   this script asked it wrongly, via SR = (SF - BF1) * 1e6.  SR is the referencing
   the operator applied, but proc_OFFSET is reported AFTER referencing, so the ppm
   axis of each spectrum already absorbs its own SR.  SR spread is therefore not
   evidence of misalignment.  MTBLS798 makes this concrete: SR = -81 Hz against
   +6 Hz for MTBLS147, yet their OFFSETs are 14.82 vs 14.71 ppm -- nothing like the
   0.14 ppm the SR difference would imply.

   What actually matters is that code/preprocessing/alignSpectra.py aligns spectra
   BY INDEX -- align_spectra_to_longest() interpolates every row to a common point
   count and never reads a ppm axis at all.  So a row's peak lands at an index set
   by its own (proc_OFFSET, proc_SW_p, proc_SF), and rows whose acquisition window
   differs land the same metabolite at a different index.  This script measures that
   displacement in points, which is the unit the model and the basis both see.

3. PULSE PROGRAM.  CPMG (cpmgpr*) suppresses broad macromolecule/lipid signal;
   a plain 1D (zg*, noesypr*) retains it.  Mixing the two means the corpus has
   two different underlying signal models, which bears directly on whether the
   empirical lipid basis in code/synthesis/lipid_basis.py is even well defined.

4. SENSITIVITY / SCALE.  NS and RG set SNR and absolute intensity.  Relevant
   only as context for the rowMinMax decision (docs/PI_outline.md 7.1).

Nothing here reads a FID or a spectrum; it is a pure metadata summary.

Usage:
    python code/analysis/bruker_param_audit.py
    python code/analysis/bruker_param_audit.py --out results/analysis/bruker_params
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "plasma": ("data/bruker_params_PlasmaNMRData.csv",
               "aligned_128K_Plasma_NoSuppress.npy", 15266, 6746),
    "serum": ("data/bruker_params_SerumNMRData.csv",
              "aligned_nmr_spectra_128K_WSNoise.npy", 2148, 2146),
}

# Bruker DATE is a unix timestamp; anything outside this window is a bad parse.
DATE_MIN, DATE_MAX = 946_684_800, 1_893_456_000   # 2000-01-01 .. 2030-01-01

NPT = 131_072          # corpus point count after index-alignment
LINEWIDTH_PPM = 0.002  # ~1.2 Hz at 600 MHz, a typical serum peak


def study_of(relpath: str) -> str:
    """First path segment is the study folder; pull the accession out of it."""
    head = str(relpath).split("/")[0]
    m = re.match(r"(MTBLS\d+|ST\d+)", head)
    return m.group(1) if m else head


def load(csv_rel: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / csv_rel, low_memory=False)
    df["study"] = df["relpath"].map(study_of)
    # A "sample" is the experiment directory: strip the trailing /<procno>.
    df["sample"] = df["relpath"].map(lambda p: "/".join(str(p).split("/")[:-1]))
    return df


def fmt_counts(s: pd.Series, top: int = 12) -> str:
    vc = s.value_counts(dropna=False)
    lines = []
    for k, v in vc.head(top).items():
        lines.append(f"      {str(k)[:44]:<44s} {v:6d}  ({100 * v / len(s):5.1f}%)")
    if len(vc) > top:
        lines.append(f"      ... {len(vc) - top} more distinct value(s)")
    return "\n".join(lines)


def describe_numeric(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "      (no parseable values)"
    q = x.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return ("      n={:d}  mean={:.4g}  sd={:.4g}\n"
            "      min={:.4g}  p5={:.4g}  p25={:.4g}  med={:.4g}  "
            "p75={:.4g}  p95={:.4g}  max={:.4g}").format(
        len(x), x.mean(), x.std(),
        q[0.0], q[0.05], q[0.25], q[0.5], q[0.75], q[0.95], q[1.0])


def ppm_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct each spectrum's ppm axis from the processing parameters.

    Bruker reports proc_OFFSET as the ppm of the FIRST processed point and
    proc_SW_p as the processed spectral width in Hz, so the axis runs from
    OFFSET down to OFFSET - SW_p/SF.  Both are post-referencing, which is why
    SR does not enter here.
    """
    out = df.copy()
    for src, dst in (("proc_SF", "SF"), ("proc_SW_p", "SW_p"),
                     ("proc_OFFSET", "OFFSET"), ("proc_SI", "SI")):
        out[dst] = pd.to_numeric(out[src], errors="coerce")
    out = out.dropna(subset=["SF", "SW_p", "OFFSET"])
    out["sw_ppm"] = out["SW_p"] / out["SF"]        # window width in ppm
    out["ppm_hi"] = out["OFFSET"]                  # first point
    out["ppm_lo"] = out["OFFSET"] - out["sw_ppm"]  # last point
    return out


def index_of(ppm_hi, sw_ppm, delta: float, npt: int = NPT):
    """Index at which chemical shift `delta` lands after index-alignment to npt.

    Index-alignment stretches whatever the row originally had across all npt
    points, so the mapping is linear in the row's own window regardless of its
    original SI.
    """
    return (ppm_hi - delta) * npt / sw_ppm


def axis_audit(df: pd.DataFrame, probes=(8.0, 3.0, 1.0)) -> tuple:
    """Displacement, in corpus points, of each study relative to the largest.

    Returns (per-study table, pooled per-row displacement at 3 ppm, reference
    study name).  Points are the meaningful unit: a 1.2 Hz linewidth at 600 MHz
    is 0.002 ppm, which over a 20 ppm window at 131,072 points is ~13 points.
    So a displacement of a few hundred points means peaks do not overlap at all.
    """
    a = ppm_axis(df)
    ref_study = a["study"].value_counts().index[0]
    r = a[a["study"] == ref_study]
    hi0, sw0 = r["ppm_hi"].median(), r["sw_ppm"].median()

    rows = []
    for st, g in a.groupby("study"):
        hi, sw = g["ppm_hi"].median(), g["sw_ppm"].median()
        rec = {"study": st, "n": len(g),
               "sw_ppm": round(float(sw), 3),
               "ppm_hi": round(float(hi), 3),
               "SI": int(g["SI"].median()) if g["SI"].notna().any() else -1}
        for d in probes:
            rec[f"dx@{d}ppm_pts"] = int(round(
                index_of(hi, sw, d) - index_of(hi0, sw0, d)))
        rec["stretch_vs_ref"] = round(float(sw0 / sw), 4)
        rows.append(rec)
    per_study = pd.DataFrame(rows).sort_values("n", ascending=False)

    dx = (index_of(a["ppm_hi"], a["sw_ppm"], 3.0)
          - index_of(hi0, sw0, 3.0))
    return per_study, dx, ref_study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/analysis/bruker_params")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    frames = {}

    for name, (csv_rel, npy, n_raw, n_unique) in SOURCES.items():
        path = ROOT / csv_rel
        if not path.exists():
            print(f"!! missing {csv_rel} -- skipping {name}")
            continue
        df = load(csv_rel)
        frames[name] = df

        print("=" * 78)
        print(f"{name.upper()}  <- {csv_rel}")
        print(f"  corresponds to {npy}: {n_raw} rows raw, {n_unique} after dedup")
        print(f"  parameter rows: {len(df)}   experiment dirs: {df['sample'].nunique()}"
              f"   studies: {df['study'].nunique()}")
        print()

        print("  [1] FIELD STRENGTH (rounded MHz)")
        print(fmt_counts(df["field_mhz_rounded"]))
        print("      BF1 (basic transmitter freq, MHz)")
        print(describe_numeric(df["acqu_BF1"]))
        print()

        print("  [2] PPM AXIS  (OFFSET = ppm of first point, SW_p/SF = width)")
        ax = ppm_axis(df)
        print("      window width, ppm")
        print(describe_numeric(ax["sw_ppm"]))
        print("      OFFSET, ppm of first point")
        print(describe_numeric(ax["ppm_hi"]))
        print("      SR = (SF - BF1)*1e6, Hz  -- already absorbed by OFFSET, "
              "reported for the record only")
        print(describe_numeric(df["SR_hz"]))
        print()

        print("  [3] PULSE PROGRAM")
        print(fmt_counts(df["acqu_PULPROG"]))
        print()

        print("  [4] ACQUISITION SCALE")
        print("      NS (scans)")
        print(describe_numeric(df["acqu_NS"]))
        print("      RG (receiver gain)")
        print(describe_numeric(df["acqu_RG"]))
        print("      TD / SI (acquired / processed points)")
        print(fmt_counts(df["acqu_TD"], top=6))
        print(fmt_counts(df["proc_SI"], top=6))
        print()

        print("  [5] HARDWARE / CONDITIONS")
        print("      INSTRUM")
        print(fmt_counts(df["acqu_INSTRUM"], top=8))
        print("      PROBHD")
        print(fmt_counts(df["acqu_PROBHD"], top=8))
        print("      SOLVENT")
        print(fmt_counts(df["acqu_SOLVENT"], top=6))
        print("      TE (K)")
        print(describe_numeric(df["acqu_TE"]))
        print()

        print("  [6] ACQUISITION DATE")
        ts = pd.to_numeric(df["acqu_DATE"], errors="coerce")
        good = ts[(ts > DATE_MIN) & (ts < DATE_MAX)]
        if good.empty:
            print("      no valid timestamps")
        else:
            dt = pd.to_datetime(good, unit="s")
            print(f"      n={len(good)} valid of {len(ts)}"
                  f"   {dt.min():%Y-%m-%d} .. {dt.max():%Y-%m-%d}")
            print("      by year:")
            print(fmt_counts(dt.dt.year, top=20))
        print()

        # Per-study table -- the useful artifact for deciding what to align.
        g = df.groupby("study")
        per_study = pd.DataFrame({
            "n": g.size(),
            "field_mhz": g["field_mhz_rounded"].agg(
                lambda s: "/".join(str(int(v)) for v in sorted(s.dropna().unique()))),
            "pulprog": g["acqu_PULPROG"].agg(
                lambda s: "/".join(sorted(set(str(v) for v in s.dropna()))[:3])),
            "sr_mean_hz": g["SR_hz"].mean().round(3),
            "sr_sd_hz": g["SR_hz"].std().round(3),
            "ns_med": g["acqu_NS"].median(),
        }).sort_values("n", ascending=False)
        per_study.to_csv(out / f"per_study_{name}.csv")
        print(f"  [7] PER-STUDY BREAKDOWN  ({len(per_study)} studies)"
              f"  -> {args.out}/per_study_{name}.csv")
        print(per_study.head(25).to_string())
        print()

    if not frames:
        raise SystemExit("no CSVs found")

    # ---- combined, corpus-weighted view -------------------------------------
    print("=" * 78)
    print("CORPUS-WIDE (plasma + serum, weighted as they enter the corpus)")
    allp = pd.concat(frames.values(), keys=frames.keys(), names=["pull"]).reset_index(0)
    print(f"  rows {len(allp)}   studies {allp['study'].nunique()}")
    print("  field strength")
    print(fmt_counts(allp["field_mhz_rounded"]))
    print("  pulse program family")
    fam = allp["acqu_PULPROG"].astype(str).str.replace(r"[\d.].*$", "", regex=True)
    print(fmt_counts(fam, top=10))
    print()

    # ---- ppm-axis audit: the fit_gate question -----------------------------
    per_study, dx, ref_study = axis_audit(allp)
    per_study.to_csv(out / "axis_displacement.csv", index=False)
    print("  PPM-AXIS DISPLACEMENT, in corpus points, vs the largest study "
          f"({ref_study})")
    print(per_study.to_string(index=False))
    print()

    lw_pts = LINEWIDTH_PPM * NPT / per_study.iloc[0]["sw_ppm"]
    print(f"  scale: a {LINEWIDTH_PPM} ppm (~1.2 Hz) linewidth = "
          f"{lw_pts:.0f} corpus points")
    for k in (10, 50, 100, 300, 1000):
        frac = 100.0 * (dx.abs() > k).mean()
        print(f"      |displacement @3 ppm| > {k:5d} pts : {frac:5.1f}% of rows"
              f"  ({k / lw_pts:5.1f} linewidths)")
    print(f"      sd {dx.std():.0f} pts   range {dx.max() - dx.min():.0f} pts")
    print()

    # ---- verdicts -----------------------------------------------------------
    print("=" * 78)
    print("VERDICTS")

    fv = allp["field_mhz_rounded"].value_counts(normalize=True, dropna=True)
    dom_f, dom_frac = fv.index[0], fv.iloc[0]
    print(f"\n  FIELD: {dom_f:.0f} MHz covers {100 * dom_frac:.1f}% of rows.")
    if dom_frac >= 0.95:
        print(f"    -> The single-field GISSMO basis at {dom_f:.0f} MHz is appropriate.")
        print("       Field is NOT a contributor to the fit_gate ceiling.")
    else:
        print("    -> Corpus is mixed-field. J-multiplet spacings in ppm differ")
        print("       between fields, so one simulated basis cannot fit all rows.")

    famv = fam.value_counts(normalize=True)
    print(f"\n  PULSE PROGRAM: '{famv.index[0]}' covers {100 * famv.iloc[0]:.1f}% of rows.")
    if famv.iloc[0] < 0.95:
        print("    -> Mixed sequences; the empirical lipid basis averages two")
        print("       different signal models. Worth splitting.")
    else:
        print("    -> Homogeneous CPMG. The empirical lipid basis is well defined,")
        print("       and broad macromolecule signal is suppressed corpus-wide.")

    print("\n  PPM AXIS -- the fit_gate bottleneck:")
    n_axes = len(per_study.groupby(["sw_ppm", "ppm_hi"]).size())
    big = per_study[per_study["n"] >= 0.01 * len(allp)]
    worst = int(big[[c for c in big.columns if c.startswith("dx@")]].abs().max().max())
    print(f"    {n_axes} distinct acquisition windows across "
          f"{per_study['study'].nunique()} studies.")
    print(f"    Worst displacement among studies >=1% of the corpus: "
          f"{worst} points = {worst / lw_pts:.0f} linewidths.")
    if (dx.abs() > 10 * lw_pts).mean() > 0.05:
        print("    -> The corpus is NOT on a common chemical-shift axis. Because")
        print("       alignSpectra.py interpolates by index, each study's peaks sit")
        print("       at its own indices. A simulated basis referenced to true ppm")
        print("       cannot match them, which caps fit_gate regardless of solver.")
        print("       FIX: re-interpolate every spectrum onto one ppm grid using")
        print("       (proc_OFFSET, proc_SW_p, proc_SF). Deterministic, no fitting.")
        print("       This requires re-deriving the corpus from the raw HDD data,")
        print("       because the row -> study mapping was never recorded.")
    else:
        print("    -> Axes are consistent; misfit is real shift variation (pH,")
        print("       ionic strength, protein binding), so fit shifts per spectrum.")

    print(f"\n  Artifacts written to {args.out}/")


if __name__ == "__main__":
    main()
