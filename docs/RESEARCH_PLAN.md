# Research plan: foundation models vs classical ML for NMR metabolomics

Every phase below answers a node of the PI's whiteboard tree (`docs/PI_whiteboard.md`).
Phases are ordered so that each decision gate is reached with the evidence needed
to pass it honestly, including the evidence needed to stop.

**The research question, stated once:** real clinical NMR tasks have 40–150
labelled samples. Does self-supervised pretraining on ~9.5k unlabelled spectra
let such a task be classified better than training on the cohort alone? Small n
is the object of study, not a limitation.

---

## Phase 0 — Freeze the data and state the protocol  (prerequisite)

Everything downstream is meaningless if the dataset or the split protocol moves
mid-experiment.

**0.1 Freeze the corpus and cohorts.** Rebuilt, verified, one ppm axis.

| dataset | rows | role |
|---|---|---|
| corpus: serum 1,961 + plasma 6,739 + workbench 780 | **9,480** | SSL pretraining only |
| MTBLS563 | 142 | evaluation |
| BrC-T2D | 115 | evaluation (4 classes) |
| MTBLS326 | 43 | evaluation |
| Barth | 40 | evaluation |

**0.2 Confirm no leakage.** MTBLS326, MTBLS563 and ST000223 do not appear in the
corpus (checked by study accession over all 9,480 provenance rows). Re-run this
check whenever the corpus changes; it is one query and it protects every result.

**0.3 Record corpus composition honestly.** 18 distinct studies, and MTBLS798
alone is 4,866 rows (51%). The corpus is large in spectra and small in studies.
This is a real limitation on how much *diversity* pretraining can supply, and it
becomes a candidate explanation in Phase 3 if scaling curves plateau early.

**0.4 Fix the evaluation protocol before seeing any result.**
- Repeated stratified k-fold: 10 folds x 10 repeats for BrC-T2D (n=115) and
  MTBLS563 (n=142); LOOCV x 10 seeds for MTBLS326 (n=43) and Barth (n=40).
- Metric: balanced accuracy primary, macro-F1 and AUC secondary. Report
  mean with 95% CI across repeats, never a single split.
- Any hyperparameter chosen on evaluation data is chosen inside a nested inner
  loop, or it is not chosen at all.
- Seeds fixed and recorded; every run writes a JSON result row.

**Deliverable:** frozen arrays, protocol document, leakage check passing.

---

## Phase 1 — The bar: classical ML  (whiteboard: "ML is better — sets bar")

The PI's premise is that classical ML currently wins. That bar must be
re-established on the corrected data before anything claims to beat it.

**1.1 Feature representations**: binned spectra (0.02/0.04 ppm bins), full
resolution with PCA, and a targeted metabolite-quantification vector if the
`code/synthesis` basis fit is sound enough to use.

**1.2 Models**: logistic regression (L1, L2), linear SVM, RBF SVM, random
forest, gradient boosting, PLS-DA. These are the field's standard.

**1.3 Run every cohort under the Phase 0 protocol.**

**Deliverable:** a table of the best classical result per cohort with CIs. This
is the number every later phase is measured against. Existing code:
`code/evaluation/classify_binned_auc_cv.py`, `fewshot_ml_comparison.py`.

---

## Phase 2 — Base models: SSL pretraining  (whiteboard: "masking, jigsaw & joint")

**2.1 Three objectives**, as the PI specified, on the 9,480-row corpus:
- **masking** — reconstruct masked spectral patches (`code/training/transformer.py`)
- **jigsaw** — recover permuted segment order (`train_jigsaw_spectra.py`)
- **joint** — both objectives together (`train_joint_ssl.py`)

**2.2 Held-out corpus split** for pretraining diagnostics, split **by study**,
not by row. Rows from one study are near-duplicates of each other; a random row
split would report a reconstruction loss that flatters the model.

**2.3 Fixed backbone** across the three objectives so the comparison is of
objectives, not capacity. Capacity is varied later, in Phase 5, deliberately.

**2.4 Sanity gates before any downstream use**: reconstruction quality on
held-out studies; embeddings not collapsed (rank, variance); nearest neighbours
in embedding space are chemically sensible rather than study-clustered. A model
that has learned "which study is this" will look excellent here and fail
transfer — check it now, not after.

**Deliverable:** three pretrained checkpoints plus diagnostics.

---

## Phase 3 — Gate G1: does pretraining beat the bar?  ("can we better it")

**3.1 Transfer modes**, in increasing adaptation:
- linear probe on frozen embeddings
- prototype / nearest-centroid few-shot
- fine-tune last block
- full fine-tune

**3.2 Few-shot curves — the core result.** k = 5, 10, 20, 40, all, per class,
with the Phase 0 repeats. Plot SSL against the Phase 1 classical bar on the same
axes. The scientific claim lives or dies here: if SSL wins at small k and
converges at large k, that is the foundation-model argument made concretely.

**3.3 Ablations**: pretrained vs random-init identical architecture (isolates
pretraining from architecture); objective comparison (masking vs jigsaw vs
joint); patch size and pooling; frozen vs fine-tuned depth.

**Gate G1 verdict:**
- **SSL > ML with non-overlapping CIs on ≥2 cohorts** → proceed to Phase 4.
- **SSL ≈ ML** → proceed to Phase 4; the question becomes whether data is the binding constraint.
- **SSL < ML everywhere** → report that classical ML is the better method for this
  task at this data scale. A negative result, honestly established, is publishable.

---

## Phase 4 — Gate G2: is data limiting?  ("need for more data? Is data limiting")

**4.1 Corpus-size scaling.** Pretrain on 10/25/50/75/100% of the corpus and
re-run Phase 3. Subsample **by study**, not by row, and report both axes: rows
and studies. Given that one study is 51% of the corpus, these two curves can
diverge sharply — and if performance tracks *studies* rather than *rows*, the
conclusion is "more diverse data" rather than "more data", which is a different
recommendation to make to the field.

**4.2 Distributions of peak intensities** (the PI's explicit ask). At each
corpus size: intensity distribution per ppm region, how it changes as data is
added, and whether it has converged.

**4.3 The pivotal diagnostic: "have we defined experimental y-distribution at
all x?"** For each ppm position x, do we have enough spectra to characterise the
intensity distribution p(y|x)? Quantify coverage — sample count, distribution
stability under resampling, regions where the corpus is thin or empty.

**Gate G2 verdict:**
- **p(y|x) is defined across the informative range** → synthetic data is
  legitimate. Proceed to Phase 5.
- **p(y|x) is not defined at all x** → generating synthetic data would be
  inventing distributions we have not measured. **Terminal branch: the honest
  conclusion is that more publicly available experimental data is needed.**
  This is the PI's "if no, we end" and it is a real result.

---

## Phase 5 — Gate G3: does synthetic data help?  ("does synthetic data help?")

Only entered if Phase 4 says p(y|x) is characterised.

**5.1 Two generation routes**, as the whiteboard specifies:
- **Linear combinations of metabolite spectra** — physically grounded, using the
  GISSMO/HMDB basis already in `code/synthesis` plus the lipid basis. Concentrations
  sampled from the measured Phase 4 distributions. This route's advantage is that
  every synthetic spectrum is explainable as a real mixture.
- **Generative models** — VAE and diffusion (`code/generative`). More expressive,
  and correspondingly easier to fool ourselves with.

**5.2 Validate the synthetic data before using it.** Do real and synthetic
spectra separate under a discriminator? Do landmark positions and the peak
intensity distributions match the real corpus? Synthetic data that fails these
should not be trained on, whatever it does to the metric.

**5.3 Augment pretraining** with synthetic data at several real:synthetic ratios
and re-run Phase 3.

**Gate G3 verdict:**
- **Synthetic helps** → *"we have our model"*. Consolidate, write up.
- **Synthetic does not help** → Phase 6.

---

## Phase 6 — More parameters  ("if no — more parameters")

Backbone scaling: depth, width, patch size, at fixed data. Existing:
`fig11_backbone_scaling`.

- **Capacity helps** → the constraint was model size; report the scaling result.
- **Capacity does not help** → **terminal branch: a limit on these tasks.**
  Neither more data, nor synthetic data, nor more parameters moves it. Stating
  that clearly, with the scaling curves behind it, is a genuine contribution.

---

## Phase 7 — Consolidation

Results table across every cohort, transfer mode and gate; the five alignment
figures; scaling and few-shot curves; the decision tree annotated with which
branch the evidence actually took; limitations, with corpus study-diversity and
cohort size named explicitly.

---

## Sequencing and dependencies

```
Phase 0  freeze + protocol
   |
Phase 1  classical bar  ------------------+
   |                                      |
Phase 2  pretrain (mask / jigsaw / joint) |
   |                                      |
Phase 3  G1: few-shot transfer vs bar <---+
   |
   +-- SSL loses everywhere ---> report: ML is better (END)
   |
Phase 4  G2: is data limiting? p(y|x) defined?
   |
   +-- p(y|x) undefined -------> report: more public data needed (END)
   |
Phase 5  G3: synthetic data
   |
   +-- helps ------------------> we have our model (END)
   |
Phase 6  more parameters
   |
   +-- helps ------------------> scaling result (END)
   +-- does not ---------------> limit on these tasks (END)
```

Phases 1 and 2 are independent and can run concurrently: Phase 1 needs no GPU,
Phase 2 needs nothing from Phase 1.

## Standing rules

1. **Name the gate.** An experiment that does not serve a node of the tree does
   not get run.
2. **Split by study, not by row**, wherever corpus rows are divided.
3. **No single-split numbers.** Every figure carries uncertainty.
4. **Old results are context, not baseline.** The corpus changed (9,480 vs 9,670
   rows) and the axis changed. Prior numbers are for tracking our own progress
   with the PI, not for publication comparison.
5. **Terminal branches are results.** Three of the five endings above are
   negative, and all three are worth publishing if that is where the evidence goes.
