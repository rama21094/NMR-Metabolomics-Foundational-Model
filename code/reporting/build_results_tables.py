#!/usr/bin/env python3
"""Phase 7 -- the master results tables, generated from the result JSON.

Every number in the consolidation traces to a file under results/, so a rerun
regenerates the tables rather than requiring them to be retyped. Anything that
cannot be read from a result file is not in the tables.
"""
import json, glob, collections
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/RESULTS_TABLES.md"
TARGETS = ["brc_t2d_cancer", "mtbls563_binary", "barth_case_control",
           "brc_t2d_diabetes", "mtbls563_3class", "brc_t2d_4class"]
NICE = {"brc_t2d_cancer": "BrC-T2D cancer", "mtbls563_binary": "MTBLS563 binary",
        "barth_case_control": "Barth case/control", "brc_t2d_diabetes": "BrC-T2D diabetes",
        "mtbls563_3class": "MTBLS563 3-class", "brc_t2d_4class": "BrC-T2D 4-class"}
L = []


def load(pat):
    rows = []
    for f in glob.glob(str(ROOT / pat)):
        rows += json.load(open(f))
    return rows


# ---- Table 1: G1, transfer vs classical, per cohort -----------------------
t3 = load("results/phase3/transfer*.json")
d = collections.defaultdict(list); cls = {}
for r in t3:
    rep = r["representation"]
    if rep.startswith("classical"):
        cls[(r["target"], r["k_per_class"])] = r["mean"]
    elif "_seed" in rep:
        d[(r["target"], r["k_per_class"], rep.split("_")[0])].append(r["mean"])

L.append("## Table 1 — Gate G1: frozen-probe transfer vs classical ML\n")
L.append("Balanced accuracy, mean over 50 paired episodes; SSL values are means "
         "over 5 pretraining seeds with the seed sd in brackets.\n")
L.append("| cohort | k | classical | masking | jigsaw | joint |")
L.append("|---|---|---|---|---|---|")
for t in TARGETS:
    for k in (5, 10, 20, 40, 0):
        if (t, k) not in cls:
            continue
        kl = "all" if k == 0 else str(k)
        cells = []
        for obj in ("masking", "jigsaw", "joint"):
            v = d.get((t, k, obj), [])
            cells.append(f"{np.mean(v):.3f} ({np.std(v, ddof=1):.3f})" if v else "—")
        L.append(f"| {NICE[t]} | {kl} | **{cls[(t,k)]:.3f}** | " + " | ".join(cells) + " |")

# ---- Table 2: G1 completed, fine-tuning ----------------------------------
ft = load("results/phase5/finetune_?.json")
if ft:
    m = collections.defaultdict(list); fcls = {}
    for r in ft:
        if r["random_init"]:
            m[(r["target"], r["k_per_class"], r["mode"] + "/rand")].append(r["mean"])
        else:
            m[(r["target"], r["k_per_class"], r["mode"])].append(r["mean"])
        fcls[(r["target"], r["k_per_class"])] = r["classical"]
    L.append("\n## Table 2 — Gate G1 completed: adaptation depth\n")
    L.append("All arms read out by the identical frozen probe; they differ only in "
             "whether the encoder was adapted on the support set. 12 episodes, 3 seeds.\n")
    L.append("| cohort | k | classical | head (frozen) | last block | full | head random-init |")
    L.append("|---|---|---|---|---|---|---|")
    for t in TARGETS:
        for k in (5, 10, 20, 0):
            if (t, k) not in fcls:
                continue
            kl = "all" if k == 0 else str(k)
            row = [f"{np.mean(m[(t,k,x)]):.3f}" if m.get((t, k, x)) else "—"
                   for x in ("head", "last_block", "full", "head/rand")]
            L.append(f"| {NICE[t]} | {kl} | **{fcls[(t,k)]:.3f}** | " + " | ".join(row) + " |")

# ---- Table 3: G2 scaling --------------------------------------------------
sc = load("results/phase4/scaling_eval_45.json")
if sc:
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
    L.append("\n## Table 3 — Gate G2: corpus scaling, three axes\n")
    L.append("Masking, mean over the 6 targets. Effective studies = "
             "exp(entropy of the study composition); nominal counts overstate it.\n")
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
        eff = mm.get("eff_studies", "—")
        L.append(f"| {axis} | {sub} | {mm.get('n_rows', '—'):,} | {mm.get('n_studies','—')} | "
                 f"{eff} | {np.mean(by[(10,sub)]):.3f} | {np.mean(by[(0,sub)]):.3f} |")
    L.append(f"| — | **classical ML** | — | — | — | **{np.mean(scls[10]):.3f}** | "
             f"**{np.mean(scls[0]):.3f}** |")

OUT.write_text("# Results tables\n\nGenerated by `code/reporting/build_results_tables.py` "
               "from the files under `results/`. Do not edit by hand.\n\n" + "\n".join(L) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}  ({len(L)} lines)")
