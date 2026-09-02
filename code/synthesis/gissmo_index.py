#!/usr/bin/env python3
"""Stage 0a: catalogue the local GISSMO library.

GISSMO (Guided Ideographic Spin System Model Optimization, NMRFAM/BMRB) stores
experimentally-refined spin-system parameters per compound, plus spectra
PRE-SIMULATED at 19 field strengths. It ships with NMRbox under /reboxitory, so
nothing needs downloading and no spin simulator has to be run -- for a 600 MHz
instrument we read `B0s/sim_600MHz.csv` directly.

This script writes one row per (compound, simulation) with the fields needed to
choose a physiologically correct basis:

  name, InChI, inchi_skeleton   identity. Name matching alone is unsafe: a
                                substring search for "leucine" also hits
                                L-isoleucine, "glutamate" hits
                                N-carbamyl-L-glutamate, and "ethanol" hits
                                ethanolamine. The skeleton (InChI formula +
                                connectivity layers, stripped of the /h, /q, /p,
                                /t, /m, /s layers) identifies constitution while
                                ignoring stereochemistry and protonation, so it
                                groups (+/-)-Glucose with D-glucose and
                                R-Lactate with L-lactate. Choose within a group
                                by name and by pH.

  ph, temperature, solvent      GISSMO entries are measured under stated
                                conditions. 1H shifts are pH-sensitive
                                (histidine, citrate move by ~0.1 ppm across
                                physiological pH), so an entry recorded far from
                                serum pH 7.4 needs a shift correction or should
                                be rejected.

  has_600MHz                    whether the field we need is present.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GISSMO = "/reboxitory/2023/12/GISSMO"

_TAG = lambda t, s: (re.search(rf"<{t}>(.*?)</{t}>", s, re.S).group(1).strip()
                     if re.search(rf"<{t}>(.*?)</{t}>", s, re.S) else "")


def inchi_skeleton(inchi: str) -> str:
    """Formula + connectivity only: drops /h /q /p /t /m /s /b /i layers.

    Keeps enough to say "this is glucose" while ignoring which anomer or
    protonation state the GISSMO entry happened to be measured in.
    """
    if not inchi.startswith("InChI="):
        return ""
    body = inchi.split("=", 1)[1]
    parts = body.split("/")
    if len(parts) < 2:
        return ""
    out = parts[:2]                        # version, then formula (e.g. 1S, C6H12O6)
    for p in parts[2:]:
        if p[:1] == "c":                   # connectivity layer
            out.append(p)
        elif p[:1] in "hqptmsbi":          # H, charge, proton, stereo layers -> stop
            break
    return "/".join(out)


def inchi_formula(inchi: str) -> str:
    """The molecular-formula layer alone, e.g. C6H12O6. Used to cross-check that
    a name match is really the intended compound."""
    if not inchi.startswith("InChI="):
        return ""
    parts = inchi.split("=", 1)[1].split("/")
    return parts[1] if len(parts) > 1 else ""


def read_aux(gissmo: Path, field: str, bmse: str) -> str:
    p = gissmo / "aux_info" / field / bmse
    if not p.exists():
        return ""
    raw = p.read_text(errors="ignore").strip()
    # values look like "7.4(pH)" or "298(K)"; keep the numeric/text payload
    m = re.match(r"^([^(]*)", raw)
    return (m.group(1).strip() if m else raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gissmo", default=DEFAULT_GISSMO)
    ap.add_argument("--field-mhz", type=int, default=600)
    ap.add_argument("--out-dir", default="results/synthesis")
    args = ap.parse_args()

    gissmo = Path(args.gissmo)
    if not gissmo.exists():
        raise SystemExit(f"GISSMO not found at {gissmo}. Other snapshots: "
                         f"{sorted(glob.glob('/reboxitory/*/GISSMO'))}")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    xmls = sorted(gissmo.glob("bmse*/simulation_*/spin_simulation.xml"))
    print(f"scanning {len(xmls)} spin_simulation.xml files under {gissmo} ...")
    for x in xmls:
        try:
            txt = x.read_text(errors="ignore")
        except OSError:
            continue
        bmse = x.parts[-3]
        sim = x.parts[-2]
        inchi = _TAG("InChI", txt)
        simcsv = x.parent / "B0s" / f"sim_{args.field_mhz}MHz.csv"
        rows.append(dict(
            bmse_id=bmse, simulation=sim, name=_TAG("name", txt), inchi=inchi,
            inchi_skeleton=inchi_skeleton(inchi),
            formula=inchi_formula(inchi),
            ref_field_mhz=_TAG("field_strength", txt),
            n_sim_points=_TAG("num_simulation_points", txt),
            roi_rmsd=_TAG("roi_rmsd", txt),
            ph=read_aux(gissmo, "ph", bmse),
            temperature_k=read_aux(gissmo, "temperature", bmse),
            solvent=read_aux(gissmo, "solvent", bmse),
            reference=read_aux(gissmo, "reference", bmse),
            sim_csv=str(simcsv) if simcsv.exists() else "",
            has_field=bool(simcsv.exists()),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gissmo_catalogue.csv", index=False)

    print(f"\n  entries scanned                 : {len(df)}")
    print(f"  with a name                     : {int((df.name != '').sum())}")
    print(f"  with an InChI                   : {int((df.inchi != '').sum())}")
    print(f"  with sim_{args.field_mhz}MHz.csv          : {int(df.has_field.sum())}")
    print(f"  distinct InChI skeletons        : {df[df.inchi_skeleton!=''].inchi_skeleton.nunique()}")
    ph = pd.to_numeric(df.ph, errors="coerce")
    print(f"  pH recorded                     : {int(ph.notna().sum())}  "
          f"(median {ph.median():.2f}, {int(((ph-7.4).abs()<=0.5).sum())} within 0.5 of serum 7.4)")
    tk = pd.to_numeric(df.temperature_k, errors="coerce")
    print(f"  temperature recorded            : {int(tk.notna().sum())}  "
          f"(median {tk.median():.0f} K)")
    dup = df[df.has_field & (df.inchi_skeleton != "")].groupby("inchi_skeleton").size()
    print(f"  skeletons with >1 usable entry  : {int((dup > 1).sum())} "
          f"(these need adjudication by name/pH)")
    print(f"\nWrote {out_dir}/gissmo_catalogue.csv")


if __name__ == "__main__":
    main()
