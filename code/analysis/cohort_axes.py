#!/usr/bin/env python3
"""The acquisition axis of every dataset, from submitter metadata. The registry.

This replaces check_spectral_width.py, which tried to RECOVER the spectral width
from the spectra. It could not be made to work, and neither could three other
approaches. All four are recorded here because each looked convincing until it was
checked against a case whose answer we knew:

  1. Sub-band shift spread. A wrong width makes different regions want different
     shifts -- true, but any window wide enough to see the drift is wide enough
     for the dense sugar region (3.2-4.1 ppm) to lock onto the wrong glucose peak,
     and any window narrow enough to stop that clips the drift and reports a
     ceiling. At +/-800 points it scored Barth-as-stored 790 ("suspect") when
     Barth-as-stored is wrong by a factor of 0.6.
  2. Joint scale+shift correlation search. Pinned at its scan boundary.
  3. Direct scale fit over a constrained range. Returned 0.977 for Barth-as-stored
     (truth 0.601) and 0.545 for BrC-T2D-as-stored (truth 0.993). The metabolite
     region is peak-dense, so a wrong scale plus a shift beats the right one.
  4. Landmark peak separations. The corpus's own lactate-to-creatine separation
     comes out 3.6% from theory, and MTBLS563's lactate-to-alanine pick is plainly
     a different peak (472 points against 1058).

The lesson is the same one Barth taught: **the spectra cannot settle their own
axis, and acquisition metadata is not optional.** Four of the five cohorts now
have it. Do not reintroduce a detector; obtain the metadata.

Usage:
    python code/analysis/cohort_axes.py
"""
from __future__ import annotations

CORPUS_SW = 20.024          # ppm, from Bruker acqus across the source studies

# name, SW (ppm), field (MHz), reference compound, source of the numbers
AXES = [
    ("corpus",   20.024,  600, "none -- see 5n",                 "Bruker acqus/procs"),
    ("Barth",    12.0308, 950, "formate, 8.44 ppm",              "submitter methods sheet"),
    ("MTBLS326", 20.02,   800, "TSP, co-axial insert (external)", "ISA-Tab i_Investigation"),
    ("MTBLS563", 20.0,    700, "indirect, anomeric glucose 5.204", "ISA-Tab i_Investigation"),
    ("BrC-T2D",  20.1587, 800, "not stated in acqus/procs",      "Bruker acqus/procs, n=364"),
    ("TBI",      None,    None, "unknown",                        "NONE -- excluded from evaluation"),
]


def main() -> None:
    print(f"corpus spectral width {CORPUS_SW} ppm\n")
    print(f"{'dataset':10s} {'SW (ppm)':>9s} {'MHz':>5s} {'resample by':>12s}  "
          f"{'reference compound':34s} source")
    for name, sw, mhz, refc, src in AXES:
        if sw is None:
            print(f"{name:10s} {'--':>9s} {'--':>5s} {'unknown':>12s}  {refc:34s} {src}")
            continue
        ratio = CORPUS_SW / sw
        act = "no (matches)" if abs(ratio - 1.0) < 0.002 else f"{ratio:.5f}"
        print(f"{name:10s} {sw:9.4f} {mhz:5d} {act:>12s}  {refc:34s} {src}")
    print("\nA ratio within 0.2% of 1 is left alone: that is below the residual "
          "shift\nthat has to be fitted afterwards in any case.")


if __name__ == "__main__":
    main()
