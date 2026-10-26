#!/usr/bin/env python3
"""Phase 7 -- the master results tables, generated from the result JSON.

Every number in the consolidation traces to a file under results/, so a rerun
regenerates the tables rather than requiring them to be retyped. Anything that
cannot be read from a result file is not in the tables.

Rebuilt 2026-10-23 for the corrected BrC-T2D labels (n=78). The old Table 1
(Phase 3 episodes, results/phase3/transfer*.json) is superseded by the paired
re-evaluation in results/phase3_fixed/ and is no longer reported.
"""
import json, glob, collections
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/RESULTS_TABLES.md"
TARGETS = ["mtbls563_binary", "mtbls563_3class", "brc_t2d_cancer",
           "brc_t2d_diabetes", "brc_t2d_4class", "barth_case_control"]
NICE = {"brc_t2d_cancer": "BrC-T2D cancer", "mtbls563_binary": "MTBLS563 binary",
        "barth_case_control": "Barth case/control", "brc_t2d_diabetes": "BrC-T2D diabetes",
        "mtbls563_3class": "MTBLS563 3-class", "brc_t2d_4class": "BrC-T2D 4-class",
        "mtbls326_ip3r": "MTBLS326 (excluded: batch confound)"}
L = []


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(ROOT / pat))):
        rows += json.load(open(f))
    if not rows:
        raise SystemExit(f"no rows for {pat}")
    return rows


def ci(d, lo, hi):
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def tally(rows):
    b = sum(r["lo"] > 0 for r in rows); w = sum(r["hi"] < 0 for r in rows)
    return b, w, len(rows) - b - w, float(np.mean([r["diff"] for r in rows]))


def kl(r):
    return {"few-shot": str(r["k_per_class"]), "loocv": "all (LOOCV)",
            "cv10x5": "all (10x5 CV)"}[r["design"]]


# ---- Table 1: the standard-ML bar -------------------------------------------
bar = [r for r in load("results/phase1_classical_bar.json") if r["metric"] == "balanced_accuracy"]
best = {}
for r in bar:
    if r["target"] not in best or r["mean"] > best[r["target"]]["mean"]:
        best[r["target"]] = r
L.append("## Table 1 — The standard-ML bar (Phase 1)\n")
L.append("Best of 8 classical models x 2 bin widths, nested inner CV, balanced accuracy. "
         "Best-of-16 selection biases this upward slightly; the G1 comparisons below use "
         "one fixed classifier (balanced logistic regression on 0.02 ppm bins) instead.\n")
L.append("| task | n | best model | bin (ppm) | balanced accuracy [95% CI] | protocol |")
L.append("|---|---|---|---|---|---|")
for t in TARGETS + ["mtbls326_ip3r"]:
    r = best[t]
    L.append(f"| {NICE[t]} | {r['n']} | {r['model']} | {r['bin_ppm']} | "
             f"{r['mean']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] | {r['protocol']} |")

# ---- Table 2: G1, paired differences ----------------------------------------
g1 = load("results/phase3_fixed/g1_fixed.json")
g1r = load("results/phase3_fixed/g1_fixed_regional16.json")
arms = {"masking (mean pool)": [r for r in g1 if r["objective"] == "masking"],
        "masking (16-region pool)": g1r,
        "jigsaw": [r for r in g1 if r["objective"] == "jigsaw"],
        "joint": [r for r in g1 if r["objective"] == "joint"]}
idx = {a: {(r["target"], r["design"], r["k_per_class"]): r for r in rows} for a, rows in arms.items()}
L.append("\n## Table 2 — Gate G1: pretrained minus standard ML, paired\n")
L.append("Same splits and the same readout (scaled features + balanced logistic regression) "
         "for both arms. Few-shot: 50 episodes, mean paired difference +/- 1.96 SE. "
         "All data: LOOCV with a paired bootstrap (n < 100) or 10x5-fold CV. "
         "Pretrained values are averaged over 5 seeds per split.\n")
L.append("| task | labelled per group | " + " | ".join(arms) + " |")
L.append("|---|---|" + "---|" * len(arms))
for t in TARGETS:
    keys = sorted({k for k in idx["masking (mean pool)"] if k[0] == t},
                  key=lambda k: (k[1] != "few-shot", k[2] or 0))
    for k in keys:
        r0 = idx["masking (mean pool)"][k]
        L.append(f"| {NICE[t]} | {kl(r0)} | " +
                 " | ".join(ci(idx[a][k]["diff"], idx[a][k]["lo"], idx[a][k]["hi"]) if k in idx[a] else "—"
                            for a in arms) + " |")
L.append("\n| arm | pretrained better | standard ML better | too close to call | mean difference |")
L.append("|---|---|---|---|---|")
for a, rows in arms.items():
    b, w, lv, m = tally(rows)
    L.append(f"| {a} | {b} | {w} | {lv} | {m:+.3f} |")

# ---- Table 3: fine-tuning -----------------------------------------------------
ft = load("results/phase5/finetune_?.json")
m = collections.defaultdict(list); fcls = {}
for r in ft:
    m[(r["target"], r["k_per_class"], r["mode"] + ("/rand" if r["random_init"] else ""))].append(r["mean"])
    fcls[(r["target"], r["k_per_class"])] = r["classical"]
L.append("\n## Table 3 — Gate G1 completed: adaptation depth\n")
L.append("Balanced accuracy. All arms read out by the identical probe; they differ only in "
         "whether the encoder was adapted on the support set. 12 episodes, 3 seeds. "
         "This experiment predates the exact-budget fix: k is capped at class size - 1, "
         "and k = all means 80/20 episodes. Compare columns within a row, not with Table 2.\n")
L.append("| task | k | standard ML | frozen | last block | full | frozen, random init |")
L.append("|---|---|---|---|---|---|---|")
for t in TARGETS:
    for k in (5, 10, 20, 0):
        if (t, k) not in fcls:
            continue
        row = [f"{np.mean(m[(t,k,x)]):.3f}" if m.get((t, k, x)) else "—"
               for x in ("head", "last_block", "full", "head/rand")]
        L.append(f"| {NICE[t]} | {'all' if k == 0 else k} | **{fcls[(t,k)]:.3f}** | " + " | ".join(row) + " |")
dd = {mode: [np.mean(m[(t, k, mode)]) - np.mean(m[(t, k, "head")]) for (t, k) in fcls
             if m.get((t, k, mode)) and m.get((t, k, "head"))] for mode in ("last_block", "full", "head/rand")}
L.append(f"\nMean change against the frozen probe: last block {np.mean(dd['last_block']):+.4f}, "
         f"full {np.mean(dd['full']):+.4f}; random init {np.mean(dd['head/rand']):+.4f}.")

# ---- Table 4: G2 scaling ------------------------------------------------------
sc = load("results/phase4/scaling_eval_45.json")
man = {}
for mf in ("manifest.json", "manifest_fixed4000.json"):
    p = ROOT / "rebuild/pretrain/scaling" / mf
    if p.exists():
        for r in json.load(open(p)):
            man[r["subset"]] = r
agg = collections.defaultdict(list); scls = collections.defaultdict(list)
for r in sc:
    agg[(r["k_per_class"], r["subset"], r["seed"])].append(r["mean"])
    scls[r["k_per_class"]].append(r["classical"])
by = collections.defaultdict(list)
for (k, sub, seed), v in agg.items():
    by[(k, sub)].append(np.mean(v))
L.append("\n## Table 4 — Gate G2: corpus scaling, three axes\n")
L.append("Masking, mean pool, balanced accuracy averaged over the 6 tasks and over seeds. "
         "Effective studies = exp(entropy of the study composition).\n")
L.append("| axis | subset | rows | nominal studies | effective | k=10 | k=all |")
L.append("|---|---|---|---|---|---|---|")
order = [("fixed budget", s) for s in ("fixed2000_k02", "fixed2000_k04", "fixed2000_k08",
         "fixed2000_k12", "fixed4000_k02", "fixed4000_k04", "fixed4000_k08", "fixed4000_k12")]
order += [("rows", s) for s in ("rows010", "rows025", "rows050", "rows075")]
order += [("studies", s) for s in ("studies3", "studies6", "studies9")]
for axis, sub in order:
    if (10, sub) not in by:
        continue
    mm = man.get(sub + "_seed0", man.get(sub, {}))
    L.append(f"| {axis} | {sub} | {mm.get('n_rows', '—'):,} | {mm.get('n_studies','—')} | "
             f"{mm.get('eff_studies', '—')} | {np.mean(by[(10,sub)]):.3f} | {np.mean(by[(0,sub)]):.3f} |")
L.append(f"| — | **standard ML** | — | — | — | **{np.mean(scls[10]):.3f}** | **{np.mean(scls[0]):.3f}** |")

px = json.load(open(ROOT / "results/phase4/parity_extrapolation.json"))
L.append("\n### Table 4b — Extrapolating to parity with standard ML\n")
L.append("Gap (pretrained minus standard ML, mean over tasks) regressed on effective studies "
         "across the 45 checkpoints; 95% interval from 5,000 bootstrap resamples of checkpoints. "
         "'none' = the resampled slope never reaches parity.\n")
L.append("| budget | fit | slope | parity at (effective studies) | 95% interval | resamples never reaching parity |")
L.append("|---|---|---|---|---|---|")
for k, res in px.items():
    for fit in ("linear", "log"):
        r = res[fit]; hi = "none" if r["ci95"][1] is None else f"{r['ci95'][1]:.0f}"
        L.append(f"| {k.replace('k=0', 'k=all')} | {fit} | {r['slope']:+.4f} | {r['crossing']:.3g} | "
                 f"{r['ci95'][0]:.0f} to {hi} | {r['share_no_crossing']:.0%} |")

# ---- Table 5: capacity ----------------------------------------------------------
cap = load("results/phase6/capacity_eval.json")
cc = collections.defaultdict(list); ccl = collections.defaultdict(list)
for r in cap:
    cc[(r["params_m"], r["k_per_class"])].append(r["mean"])
    ccl[r["k_per_class"]].append(r["classical"])
L.append("\n## Table 5 — Model size\n")
L.append("Masking, mean pool, balanced accuracy averaged over 6 tasks and 3 seeds.\n")
L.append("| parameters | k=10 | k=all | gap at k=all |")
L.append("|---|---|---|---|")
for p in sorted({p for p, _ in cc}):
    L.append(f"| {p:.2f} M | {np.mean(cc[(p,10)]):.3f} | {np.mean(cc[(p,0)]):.3f} | "
             f"{np.mean(cc[(p,0)]) - np.mean(ccl[0]):+.3f} |")
L.append(f"| **standard ML** | **{np.mean(ccl[10]):.3f}** | **{np.mean(ccl[0]):.3f}** | — |")

# ---- Table 6: G3 realism ---------------------------------------------------------
sr = json.load(open(ROOT / "results/phase5/synthetic_realism.json"))
L.append("\n## Table 6 — Gate G3: is the synthetic data realistic?\n")
L.append("AUC (fair): logistic regression real-vs-synthetic, stratified 5-fold CV, real spectra "
         "drawn fresh (0.5 = indistinguishable). AUC (grouped): the same with folds grouped by "
         "study, the original test; it drops BELOW 0.5 for generators that reproduce each study "
         "closely and flatters distant ones, so the fair column is the one to quote. "
         "Distance: median per-bin Wasserstein / IQR (real-vs-real floor 0.053). "
         "Real corpus effective studies: 4.09.\n")
L.append("| set | AUC (fair) | AUC (grouped) | CH3-CH separation (2.780) | distance | effective studies |")
L.append("|---|---|---|---|---|---|")
for name, r in sr.items():
    eff = f"{r['eff_studies']:.2f}" if r.get("eff_studies") is not None else "—"
    L.append(f"| {name} | **{r['auc_stratified']:.3f}** | {r['auc']:.3f} | {r['ch3_ch_separation']:.3f} | "
             f"{r['median_rel_wasserstein']:.3f} | {eff} |")
mem = json.load(open(ROOT / "results/final/memorization.json"))
L.append("\nMemorisation (each spectrum's highest correlation with any training spectrum, median / "
         "share above 0.999): " + "; ".join(f"{k} {v['median']:.4f} / {v['share_above_0.999']:.1%}"
                                             for k, v in mem.items()) + ".")
reg = json.load(open(ROOT / "results/phase5/gan_auc_by_region_full.json"))
L.append("\nGAN v2 separability by region: " + "; ".join(f"{k} {v:.3f}" for k, v in reg.items()) + ".")

# ---- Table 7: G3 augmentation ------------------------------------------------------
ag = load("results/g3_augment/g3_augment_eval.json")
L.append("\n## Table 7 — Gate G3: does adding GAN v2 spectra to pretraining help?\n")
L.append("realblank = real corpus, edges blanked; realgan = the same + 4,000 GAN v2 spectra; "
         "baseline = Phase 2 masking, unblanked. Masking, mean pool, 3 seeds, same protocol as Table 2.\n")
L.append("| comparison | first better | second better | too close to call | mean difference |")
L.append("|---|---|---|---|---|")
for a, b in [("realgan", "realblank"), ("realgan", "classical"),
             ("realblank", "classical"), ("baseline", "classical")]:
    bb, ww, lv, mm = tally([r for r in ag if r["a"] == a and r["b"] == b])
    L.append(f"| {a} vs {b} | {bb} | {ww} | {lv} | {mm:+.3f} |")

# ---- Table 8: diffusion augmentation curve -----------------------------------
sa = load("results/final/synaug_eval_regional16.json")
L.append("\n## Table 8 — Gate G3: pretraining with 1k / 4k / 16k even-mix diffusion spectra\n")
L.append("Real corpus (8,545) + N latent-diffusion spectra generated with an even study mix "
         "(9.79 effective studies), 3 seeds per N; baseline = Phase 2 masking, 5 seeds. "
         "16-region pooling, same protocol as Table 2. Augmented runs trained 2-3x longer "
         "(synthetic spectra in the early-stopping set), which favours them.\n")
L.append("| comparison | first better | second better | too close to call | mean difference |")
L.append("|---|---|---|---|---|")
for a_, b_ in [("aug1k", "baseline"), ("aug4k", "baseline"), ("aug16k", "baseline"),
               ("baseline", "classical"), ("aug1k", "classical"), ("aug4k", "classical"), ("aug16k", "classical")]:
    bb, ww, lv, mm = tally([r for r in sa if r["a"] == a_ and r["b"] == b_])
    L.append(f"| {a_} vs {b_} | {bb} | {ww} | {lv} | {mm:+.3f} |")

OUT.write_text("# Results tables\n\nGenerated by `code/reporting/build_results_tables.py` "
               "from the files under `results/`. Do not edit by hand.\n\n" + "\n".join(L) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}  ({len(L)} lines)")
