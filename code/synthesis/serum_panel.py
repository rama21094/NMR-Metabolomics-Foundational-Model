#!/usr/bin/env python3
"""The candidate serum/plasma metabolite panel, resolved to explicit GISSMO IDs.

WHY THE IDs ARE HARD-CODED. Name matching is unsafe -- a substring search for
"leucine" hits L-isoleucine, "glutamate" hits N-carbamyl-L-glutamate, "ethanol"
hits ethanolamine, "acetone" hits dihydroxyacetone. Formula matching alone is
also unsafe, because isomers share formulas: C6H12O6 covers glucose AND all four
inositols; C3H7NO2 covers L-alanine, beta-alanine, D-alanine and sarcosine;
C4H8O3 covers both 2- and 3-hydroxybutyrate. Each entry below is therefore
pinned to a specific bmse ID, and `build_basis.py` re-verifies the molecular
formula so a library update cannot silently substitute a different compound.

SELECTION RULES APPLIED
  - the physiological stereoisomer (L- amino acids, L-lactate)
  - pH as close to serum 7.4 as the library offers
  - where several entries are identical in those respects, the lower roi_rmsd

NOT AVAILABLE in GISSMO (checked by name and formula), and therefore absent from
this panel even though they are real serum constituents:
  acetone, urea, mannose, hippurate, malate, 2-oxoglutarate, phosphocholine,
  glycerophosphocholine, methylhistidine.
Acetone and urea in particular are simple singlets and could be added
analytically later if the fit gate shows a residual at 2.22 or ~5.8 ppm.

THIS IS A CANDIDATE LIST, NOT THE FINAL PANEL. Per docs/PI_outline.md section 8,
the panel is pruned empirically: fit all of these to the corpus and keep the ones
whose fitted concentration is reliably non-zero and whose removal measurably
worsens the residual. `fit_gate.py` produces the evidence for that pruning.
"""

# (canonical name, bmse_id, expected molecular formula)
PANEL = [
    ("glucose",             "bmse000015", "C6H12O6"),
    ("L-lactate",           "bmse000208", "C3H6O3"),
    ("L-alanine",           "bmse000028", "C3H7NO2"),
    ("L-valine",            "bmse000052", "C5H11NO2"),
    ("L-isoleucine",        "bmse000041", "C6H13NO2"),
    ("L-leucine",           "bmse000042", "C6H13NO2"),
    ("L-glutamine",         "bmse000038", "C5H10N2O3"),
    ("L-glutamate",         "bmse000037", "C5H9NO4"),
    ("glycine",             "bmse000089", "C2H5NO2"),
    ("citrate",             "bmse000076", "C6H8O7"),
    ("creatine",            "bmse000078", "C4H9N3O2"),
    ("creatinine",          "bmse000155", "C4H7N3O"),
    ("pyruvate",            "bmse000112", "C3H4O3"),
    ("acetate",             "bmse000191", "C2H4O2"),
    ("3-hydroxybutyrate",   "bmse000161", "C4H8O3"),
    ("2-hydroxybutyrate",   "bmse000361", "C4H8O3"),
    ("formate",             "bmse000203", "CH2O2"),
    ("L-histidine",         "bmse000039", "C6H9N3O2"),
    ("L-phenylalanine",     "bmse000045", "C9H11NO2"),
    ("L-tyrosine",          "bmse000051", "C9H11NO3"),
    ("L-threonine",         "bmse000049", "C4H9NO3"),
    ("L-proline",           "bmse000047", "C5H9NO2"),
    ("L-lysine",            "bmse000043", "C6H14N2O2"),
    ("L-methionine",        "bmse000044", "C5H11NO2S"),
    ("taurine",             "bmse000120", "C2H7NO3S"),
    ("choline",             "bmse000285", "C5H14NO"),
    ("betaine",             "bmse000069", "C5H11NO2"),
    ("myo-inositol",        "bmse000103", "C6H12O6"),
    ("succinate",           "bmse000183", "C4H6O4"),
    ("L-asparagine",        "bmse000030", "C4H8N2O3"),
    ("L-aspartate",         "bmse000031", "C4H7NO4"),
    ("L-arginine",          "bmse000029", "C6H14N4O2"),
    ("L-serine",            "bmse000048", "C3H7NO3"),
    ("L-tryptophan",        "bmse000050", "C11H12N2O2"),
    ("L-ornithine",         "bmse000162", "C5H12N2O2"),
    ("trimethylamine",      "bmse000224", "C3H9N"),
    ("TMAO",                "bmse000426", "C3H9NO"),
    ("dimethylglycine",     "bmse000241", "C4H9NO2"),
    ("sarcosine",           "bmse000160", "C3H7NO2"),
    ("glycerol",            "bmse000184", "C3H8O3"),
    ("L-citrulline",        "bmse000032", "C6H13N3O3"),
    ("L-carnitine",         "bmse000211", "C7H15NO3"),
    ("glutathione",         "bmse000185", "C10H17N3O6S"),
]

MISSING_FROM_GISSMO = ["acetone", "urea", "mannose", "hippurate", "malate",
                       "2-oxoglutarate", "phosphocholine", "glycerophosphocholine",
                       "methylhistidine"]
