#!/usr/bin/env python3
"""Collect the G3 realism-gate and diversity numbers into one tracked JSON.

validate_synthetic.py and synthetic_effective_diversity.py print to logs, and
*.log is git-ignored, so the numbers quoted in CONSOLIDATION.md had no tracked
source. This parses those logs into results/phase5/synthetic_realism.json.
Fails loudly if a log is missing or a line cannot be parsed.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "results/phase5"
SETS = {  # name -> (validate log, diversity log or None, file stem for the stratified test)
    "GISSMO + lipids": ("gissmo_fixed_validate.log", "gissmo_fixed_diversity.log", "gissmo_joint_fixed"),
    "GAN v1": ("gan_validate.log", "gan_diversity.log", "gan"),
    "GAN v2": ("gan_v2_validate.log", "gan_v2_diversity.log", "gan_v2"),
    "GAN v2 study-mixing": ("gan_v2_cond_validate.log", "gan_v2_cond_diversity.log", "gan_v2_cond"),
    "VAE prior samples": ("../final/ldm_v2/validate_vae_v2.txt", None, "vae_v2"),
    "Latent diffusion, corpus mix": ("../final/ldm_v2/validate_ldm_v2_proportional.txt",
                                     "../final/ldm_v2/diversity_ldm_v2_proportional.log", "ldm_v2_proportional"),
    "Latent diffusion, even mix": ("../final/ldm_v2/validate_ldm_v2_uniform.txt",
                                   "../final/ldm_v2/diversity_ldm_v2_uniform.log", "ldm_v2_uniform"),
}
STRAT = json.load(open(ROOT / "results/final/realism_stratified.json"))
BLANKED = {"GAN v2": "blank/validate_gan_v2_blank.txt",
           "GAN v2 study-mixing": "blank/validate_gan_v2_cond_blank.txt"}


def grab(text, pat, name):
    m = re.search(pat, text)
    if not m:
        raise SystemExit(f"could not parse {name!r} with {pat!r}")
    return float(m.group(1))


out = {}
for name, (vlog, dlog, stem) in SETS.items():
    v = (P5 / vlog).read_text(); d = (P5 / dlog).read_text() if dlog else None
    r = {"auc": grab(v, r"AUC = ([\d.]+)", vlog),
         "auc_sd": grab(v, r"AUC = [\d.]+ \(\+/- ([\d.]+)\)", vlog),
         "ch3_ch_separation": grab(v, r"synthetic ([\d.]+)\s+expected", vlog),
         "median_rel_wasserstein": grab(v, r"median rel\. Wasserstein ([\d.]+)", vlog),
         "eff_studies": grab(d, r"synthetic set: \d+ spectra, effective studies ([\d.]+)", dlog) if d else None,
         "corpus_eff_studies": 4.09,
         "auc_stratified": STRAT[stem]["auc"][0]}
    if name in BLANKED:
        r["auc_edges_blanked"] = grab((P5 / BLANKED[name]).read_text(), r"AUC = ([\d.]+)", BLANKED[name])
    out[name] = r
    print(f"{name:<30} AUC {r['auc']:.3f} / fair {r['auc_stratified']:.3f}  W {r['median_rel_wasserstein']:.3f}  eff {r['eff_studies']}"
          + (f"  blanked {r['auc_edges_blanked']:.3f}" if "auc_edges_blanked" in r else ""))
json.dump(out, open(P5 / "synthetic_realism.json", "w"), indent=1)
print("wrote results/phase5/synthetic_realism.json")
