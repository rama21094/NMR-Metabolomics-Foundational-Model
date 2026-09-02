#!/usr/bin/env python3
"""Stage 0b: build the LC basis matrix from GISSMO at 600 MHz.

Reads each panel compound's pre-simulated 600 MHz spectrum, resamples it onto
the project's own ppm axis, applies the same masking the real pipeline applies,
and writes a matrix B of shape (n_metabolites, 131072).

FOUR THINGS THAT HAVE TO BE RIGHT
---------------------------------
1. FIELD. J-couplings are fixed in Hz, so a multiplet's apparent width in ppm
   scales as 1/field. Simulating at the wrong field misplaces every splitting.
   The instrument here is 600 MHz and GISSMO ships sim_600MHz.csv, so no spin
   simulation is needed.

2. GRID. GISSMO simulates 131,068 points over -1..12 ppm (~10,082 pts/ppm); this
   project's axis is 131,072 points over -1.5..10.5 (~10,923 pts/ppm). The two
   are close but not identical, so the basis is interpolated onto OUR axis.
   Interpolation slightly alters effective linewidth; `--extra-broadening-hz`
   exists to match the real corpus's linewidth afterwards if the fit gate shows
   systematically narrow basis peaks.

3. THE SAME MASKING. The real pipeline zeroes the water-suppression window
   (points 62,500-68,000). Basis vectors must be zeroed there too, or the fit
   will try to explain a region that has been deleted from the data.

4. WHAT THE COEFFICIENT MEANS. Each basis vector is scaled to unit total
   absolute area over the retained region. A fitted coefficient is therefore a
   RELATIVE SIGNAL CONTRIBUTION, not a molar concentration. Molar quantification
   would need (a) division by the compound's proton count and (b) an internal
   reference of known concentration -- and this corpus has NO usable 0 ppm
   reference resonance (median 0.7% of global max), so absolute concentrations
   are not recoverable here. Relative units are sufficient for generation.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_sp = importlib.util.spec_from_file_location("sp", ROOT / "code/synthesis/serum_panel.py")
sp = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(sp)

PPM_AXIS = "data/mtbls326/MTBLS326_common_ppm_axis.npy"
WATER_LO, WATER_HI = 62500, 68000          # zeroed by build_clean_datasets.py


def lorentzian_broaden(y, ppm, extra_hz, field_mhz):
    """Convolve with a Lorentzian of the given FWHM in Hz."""
    if extra_hz <= 0:
        return y
    pts_per_ppm = len(ppm) / (ppm.max() - ppm.min())
    fwhm_ppm = extra_hz / field_mhz
    hw = max(fwhm_ppm * pts_per_ppm / 2.0, 1e-6)
    n = int(min(len(y) // 4, max(64, 12 * hw)))
    x = np.arange(-n, n + 1)
    k = 1.0 / (1.0 + (x / hw) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gissmo", default="/reboxitory/2023/12/GISSMO")
    ap.add_argument("--field-mhz", type=int, default=600)
    ap.add_argument("--catalogue", default="results/synthesis/gissmo_catalogue.csv")
    ap.add_argument("--extra-broadening-hz", type=float, default=0.0)
    ap.add_argument("--ph-tolerance", type=float, default=1.0,
                    help="warn if an entry's pH is further than this from 7.4")
    ap.add_argument("--out-dir", default="results/synthesis")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ppm = np.load(ROOT / PPM_AXIS)
    length = len(ppm)
    cat = pd.read_csv(ROOT / args.catalogue).drop_duplicates("bmse_id").set_index("bmse_id")

    keep = np.ones(length, dtype=bool)
    keep[WATER_LO:WATER_HI] = False

    rows, vectors = [], []
    print(f"building basis at {args.field_mhz} MHz on a {length}-point axis "
          f"({ppm.min()}..{ppm.max()} ppm)\n")
    for name, bmse, formula in sp.PANEL:
        if bmse not in cat.index:
            raise SystemExit(f"{name}: {bmse} absent from {args.catalogue}")
        row = cat.loc[bmse]
        if str(row["formula"]) != formula:
            raise SystemExit(f"{name}: {bmse} formula is {row['formula']}, expected "
                             f"{formula}. The library may have changed -- re-verify "
                             f"the panel before proceeding.")
        csv = Path(str(row["sim_csv"]))
        if not csv.exists():
            csv = Path(args.gissmo) / bmse / row["simulation"] / "B0s" / f"sim_{args.field_mhz}MHz.csv"
        if not csv.exists():
            raise SystemExit(f"{name}: no sim_{args.field_mhz}MHz.csv for {bmse}")

        d = pd.read_csv(csv)
        gp = d["ppm"].to_numpy(dtype=np.float64)
        gv = d["val"].to_numpy(dtype=np.float64)
        order = np.argsort(gp)
        v = np.interp(ppm, gp[order], gv[order], left=0.0, right=0.0)
        if args.extra_broadening_hz > 0:
            v = lorentzian_broaden(v, ppm, args.extra_broadening_hz, args.field_mhz)
        v[~keep] = 0.0
        area = np.abs(v).sum()
        if area <= 0:
            raise SystemExit(f"{name}: basis vector is all zero after masking")
        v = v / area

        ph = pd.to_numeric(pd.Series([row["ph"]]), errors="coerce").iloc[0]
        flag = ""
        if ph == ph and abs(ph - 7.4) > args.ph_tolerance:
            flag = f"  <-- pH {ph} is {abs(ph-7.4):.1f} from serum 7.4"
        vectors.append(v)
        rows.append(dict(name=name, bmse_id=bmse, formula=formula,
                         gissmo_name=row["name"], ph=row["ph"],
                         temperature_k=row["temperature_k"], solvent=row["solvent"],
                         roi_rmsd=row["roi_rmsd"],
                         peak_ppm=float(ppm[int(np.argmax(v))]),
                         frac_area_in_water_window=0.0))
        print(f"  {name:20s} {bmse}  {str(row['name'])[:28]:30s} "
              f"tallest peak {ppm[int(np.argmax(v))]:6.2f} ppm{flag}")

    B = np.stack(vectors)
    np.save(out_dir / "basis_600MHz.npy", B.astype(np.float32))
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "basis_600MHz_meta.csv", index=False)

    # conditioning: collinear basis vectors cannot be separately identified
    Bk = B[:, keep]
    G = Bk @ Bk.T
    dn = np.sqrt(np.diag(G))
    C = G / np.outer(dn, dn)
    np.fill_diagonal(C, 0.0)
    iu = np.triu_indices(len(B), 1)
    worst = np.argsort(-C[iu])[:8]
    sv = np.linalg.svd(Bk, compute_uv=False)
    print(f"\n  basis shape {B.shape}, condition number {sv[0]/sv[-1]:.1f}")
    print(f"  most collinear pairs (these will trade off in the fit):")
    for k in worst:
        i, j = iu[0][k], iu[1][k]
        print(f"    {meta.name[i]:20s} vs {meta.name[j]:20s} cos = {C[i, j]:.3f}")
    print(f"\nWrote {out_dir}/basis_600MHz.npy")
    print(f"Wrote {out_dir}/basis_600MHz_meta.csv")


if __name__ == "__main__":
    main()
