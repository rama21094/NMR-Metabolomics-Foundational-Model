# Why classical ML outperforms the SSL backbones — analysis and experiment queue

**Status:** analysis complete as of 2026-07-28, on the v4 (cleaned) datasets and the
2026-07-25 SSL checkpoints. Experiments **#1, #2, #3, #5 are done**; #4, #6, #7, #8 queued.

**Purpose of this document.** The v4 benchmark showed logistic regression beating all
three SSL families on all five dataset/label targets. This records *why*, with the
diagnostics that established it, so each follow-up experiment can be run one at a time
without re-deriving the reasoning. Every number here is reproducible from the scripts
named in each section.

---

## 1. TL;DR

1. The headline gap is **not one problem, it is two**, in different proportions per
   dataset: a **representation ceiling** caused by the backbone's patch size, and a
   **head-fitting deficit** in how the classifier is trained.
2. The **masking** family's head is systematically underfit. Replacing it with a
   properly-regularized linear probe on the *identical frozen features* gains
   **+0.12 balanced accuracy on average, on all 5 of 5 targets**. This is free.
3. The backbone tokenizes 131,072 points into **128 patches**, so it cannot represent
   spectral detail finer than 128 positions. LogReg on 1024 bins exploits 8× finer
   resolution, and the bin sweep shows that resolution is exactly what buys accuracy.
4. The classical results are **real, not CV artifacts** — label-permutation nulls give
   p ≤ 0.02 everywhere. They are also **not a dilution confound** — a single global
   intensity scalar explains far less.
5. **A correction to an earlier claim:** the existing Xavier ablation does *not* test
   "pretrained vs random backbone." Read correctly, pretraining **helps**. See §6.

---

## 2. What was compared, and how

| | classical track | SSL track |
|---|---|---|
| features | `binned_abs_area`: absolute integrated area in *N* equal bins (N=1024 default, 256 for `mtbls326_loocv.py`) | frozen/fine-tuned backbone embedding |
| classifier | `StandardScaler` → `LogisticRegression(C=1, max_iter=5000, class_weight="balanced")`, L-BFGS to convergence | pooled embedding → `LayerNorm` → `Dropout` → `Linear`, Adam ~50 epochs, early stopping, class-balanced CE |

Both classifiers are **linear**. That is what makes the swap in §4 a clean experiment.

Pooling per family (each replicated exactly in the probe scripts):

| family | pooling | dim |
|---|---|---|
| masking | `encoded.mean(dim=1)` | 128 |
| jigsaw | per-bin-size transformer pass → mean-pool → concat 4 bin sizes | 768 |
| joint | `encode_spectrum(bin_sizes, include_masked_task=True)` | 960 |

Protocols match the committed runs exactly: LOOCV for Barth (n=37, "Pool" QC rows
excluded) and MTBLS326 (n=42); `StratifiedKFold(10, shuffle=True, random_state=42)` for
BrC-T2D (n=78) and MTBLS563 (n=113, 3-class, "unknown" excluded).

> **Validation gate.** Before drawing any conclusion, the probe harness was checked to
> reproduce the official `summary.csv` LogReg balanced accuracy to 6 decimal places on
> all four datasets: BrC-T2D cancer 0.936842, diabetes 0.828877, MTBLS563 0.720785,
> MTBLS326 1.000000, Barth 0.704969. Without this, none of the comparisons below would
> be interpretable.

---

## 3. Headline result (v4)

Best mode per family, balanced accuracy. Figures:
`results/plots/all_datasets_summary_v4/fig1_balanced_accuracy.png`,
`fig2_roc_auc.png`, `fig3_finetune_depth.png`, `fig4_heatmap_all_models.png`.

| dataset | classical (LR) | masked | jigsaw | joint | gap (LR − best SSL) |
|---|---|---|---|---|---|
| Barth (n=37, LOOCV) | **0.705** | 0.691 | 0.677 | 0.649 | 0.014 |
| MTBLS326 (n=42, LOOCV) | **1.000** | 0.981 | 0.874 | 0.930 | 0.019 |
| MTBLS563 (n=113, 3-class) | **0.721** | 0.558 | 0.550 | 0.500 | 0.163 |
| BrC-T2D cancer (n=78) | **0.937** | 0.796 | 0.782 | 0.757 | 0.141 |
| BrC-T2D diabetes (n=78) | **0.829** | 0.653 | 0.620 | 0.624 | 0.176 |

Consistent SSL ordering: **masked > jigsaw ≈ joint**.

**Do not over-read Barth and MTBLS326.** Their gaps (0.014, 0.019) are half a sample and
one sample respectively, and LOOCV yields no fold variance, so there are no error bars.
Treat those two as ties.

---

## 4. Gap decomposition — representation vs. head

Script: `code/analysis/probe_logreg_advantage.py` →
`results/analysis/logreg_advantage_probe/probe_results.csv`
Figure: `results/plots/all_datasets_summary_v4/fig6_logreg_advantage_probe.png`

Running the **same** LogReg on the frozen masked-SSL embedding isolates the two causes:

| dataset | SSL head (reported) | LogReg on *same* embedding | LogReg @1024 bins | head deficit | representation ceiling |
|---|---|---|---|---|---|
| BrC-T2D cancer | 0.796 | 0.833 | 0.937 | 0.037 | **0.104** |
| BrC-T2D diabetes | 0.653 | **0.810** | 0.829 | **0.157** | 0.019 |
| MTBLS563 | 0.558 | 0.607 | 0.721 | 0.049 | **0.114** |
| Barth | 0.691 | **0.770** | 0.705 | **0.079** | −0.065 (embedding *beats* bins) |
| MTBLS326 | 0.981 | 0.944 | 1.000 | −0.037 | 0.056 |

Reading:
- **Diabetes is a head problem** (89% of the gap). The embedding already supports 0.810.
- **Cancer and MTBLS563 are representation problems** (~70–75% of the gap survives the
  better classifier).
- **Barth's SSL representation is genuinely better than binned features** (0.770 vs
  0.705). The head was throwing that away.

### 4b. The pure classifier test (experiment #2, DONE)

Script: `code/analysis/linear_probe_frozen_embeddings.py` →
`results/analysis/linear_probe_frozen/linear_probe_results.csv`
Figure: `results/plots/all_datasets_summary_v4/fig7_linear_probe_vs_head.png`

LogReg vs the trained MLP head on **identical frozen features** — same backbone, same
pooling, same folds. Any difference is purely how the linear map is fitted.

Δ balanced accuracy (LogReg probe − frozen head):

| dataset | masking | jigsaw | joint |
|---|---|---|---|
| Barth | **+0.115** | +0.093 | +0.022 |
| BrC-T2D cancer | **+0.103** | 0.000 | +0.013 |
| BrC-T2D diabetes | **+0.156** | −0.026 | +0.019 |
| MTBLS326 | **+0.148** | 0.000 | 0.000 |
| MTBLS563 | **+0.077** | −0.024 | −0.006 |
| **mean** | **+0.120** | +0.009 | +0.009 |

**Conclusion: the masking head is underfit on all 5 of 5 targets**, by +0.077 to +0.156.
Jigsaw and joint heads are fine.

Compared against each family's *best fine-tuned* mode (the practical question):

| family | mean Δ | interpretation |
|---|---|---|
| masking | **+0.057** | fine-tuning does *not* close the gap — switch the head |
| jigsaw | −0.022 | fine-tuning beats a frozen probe — keep fine-tuning |
| joint | −0.019 | keep fine-tuning |

**Actionable now:** for the masking family, replace the Adam-trained MLP head with a
converged, L2-regularized linear probe (or tune weight decay / epochs / LR schedule to
match). Expect ≈ +0.06 vs the current best configuration and ≈ +0.12 vs frozen.

---

## 5. Mechanism of the representation ceiling

`patch_size=1024` → 131072/1024 = **128 patches**. The encoder cannot represent detail
finer than 128 spectral positions. Compare LogReg at exactly that resolution:

| | LogReg @128 bins | SSL embedding | LogReg @1024 bins |
|---|---|---|---|
| BrC-T2D cancer | 0.885 | 0.859 (flatten) | 0.937 |
| BrC-T2D diabetes | 0.746 | 0.810 | 0.829 |
| MTBLS563 | between 64-bin 0.592 and 256-bin 0.670 | 0.607–0.621 | 0.721 |

The embedding lands at its own patch resolution. And resolution is what buys accuracy —
the bin sweep (BrC-T2D cancer):

| bins | 16 | 64 | 128 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| bal. acc. | 0.836 | 0.859 | 0.885 | 0.898 | **0.937** | 0.937 |

Monotonic to 1024, then plateau. **The 8× downsampling in the tokenizer is discarding
discriminative signal.** Encouragingly, on diabetes the embedding (0.810) *beats* raw
binning at the same resolution (0.746), so the learned representation does add value —
it is just starved of resolution.

Mean-pooling is a *minor* factor by comparison: flatten (position-preserving) vs
mean-pool changed cancer +0.026 and MTBLS563 +0.014, but *hurt* diabetes −0.018.

---

## 6. Controls — what this is NOT

### Not CV overfitting
200-permutation nulls under the identical protocol, 1024 bins:

| dataset | observed | null mean | null p95 | null max | p |
|---|---|---|---|---|---|
| BrC-T2D cancer | 0.937 | 0.494 | 0.616 | 0.717 | 0.005 |
| BrC-T2D diabetes | 0.829 | 0.498 | 0.619 | 0.674 | 0.005 |
| MTBLS563 | 0.721 | 0.332 | 0.417 | 0.526 | 0.005 |
| MTBLS326 | 1.000 | 0.485 | 0.634 | 0.737 | 0.005 |
| Barth | 0.705 | 0.498 | 0.655 | **0.792** | 0.020 |

p=0.005 is the floor at 200 permutations. **Barth is marginal** — its null reaches 0.792,
*above* the observed 0.705. Treat Barth's classical result as weak evidence.

### Not a global dilution/intensity confound
A single scalar feature:

| dataset | total abs. area | row std | LogReg @1024 bins |
|---|---|---|---|
| BrC-T2D cancer | 0.616 | 0.705 | 0.937 |
| BrC-T2D diabetes | 0.593 | 0.392 | 0.829 |
| MTBLS563 | 0.337 | 0.426 | 0.721 |

There is *some* global component (notably cancer's row_std at 0.705) but it is far below
the full model. The signal is genuinely local spectral structure.

### CORRECTION: the Xavier ablation does not test what it appears to

`--reinit-unfrozen-xavier` calls `reinit_xavier_(layer)` **only inside the unfrozen-layer
loop**. Therefore:
- In `frozen` mode nothing is unfrozen → **nothing is reinitialized** → the "xavier" arm
  is byte-identical to the pretrained arm. Verified: Barth masking 0.526398 in both,
  MTBLS326 0.762963 in both.
- `patch_embedding`, `pos_encoding`, and all still-frozen layers **remain pretrained** in
  every mode.

So it tests "does discarding pretrained weights *in the layers being fine-tuned* hurt?",
not "pretrained vs random backbone." Read mode-by-mode (not max-over-modes),
**pretraining helps**, increasingly with depth — MTBLS326 masked:

| mode | pretrained | reinit |
|---|---|---|
| frozen | 0.763 | 0.763 (identical by construction) |
| +1 layer | 0.896 | 0.830 |
| +2 layers | 0.930 | 0.733 |
| +3 layers | **0.948** | 0.648 |

MTBLS563 at +3 layers: pretrained wins all three families (0.566/0.540/0.463 vs
0.509/0.476/0.414). Barth is noisy and mixed (n=37).

An earlier verbal claim in this project that "pretraining contributes nothing" came from
taking `max()` over four noisy modes per arm and is **retracted**. A true random-backbone
control has now been run — see §6b, which supersedes this section.

### 6b. Experiment #3 — the true random-backbone control (DONE)

`code/evaluation/ssl_linear_probe_eval.py --random-backbone` keeps each architecture but
loads **no pretrained weights anywhere** (patch embedding and positional encoding
included), and the classifier is held fixed at a converged LogReg (C=1) on frozen
features in both arms. Holding the head fixed removes the head-underfitting confound of
§4b, which is what made the original ablation unreadable.
Verified as a genuine control: embeddings differ from pretrained for all three families,
and are reproducible under a fixed seed.

Δ balanced accuracy (pretrained − random init):
`results/linear_probe/exp3_pretrained_vs_random.csv`,
figure `fig8_pretraining_gain.png`

| dataset | masking | jigsaw | joint |
|---|---|---|---|
| Barth | **+0.252** | +0.087 | +0.202 |
| MTBLS326 | **+0.052** | +0.015 | −0.100 |
| BrC-T2D cancer | **+0.063** | −0.028 | −0.077 |
| BrC-T2D diabetes | **+0.171** | −0.067 | −0.023 |
| MTBLS563 | **+0.047** | −0.065 | −0.126 |
| **mean** | **+0.117** | −0.011 | −0.025 |

**This is the single most important result in this document:**

- **Masked pretraining works.** Positive on 5/5 targets, mean +0.117. It is the only
  objective that earns its keep.
- **Jigsaw pretraining is worthless** (mean −0.011, loses to random on 3/5).
- **Joint pretraining is actively harmful** (mean −0.025, loses on 3/5, worst −0.126).
  A *random* joint backbone scores 0.846 on BrC-T2D cancer and 0.911 on MTBLS326 versus
  the pretrained 0.769 and 0.811 — the joint objective is destroying useful structure.

Interpretation caveat: a random transformer is a legitimately strong random-projection
baseline (cf. random-feature kernel methods), so "random wins" means **the objective adds
nothing over a random projection**, not that the architecture is useless.

Consequence for the roadmap: concentrate on the masking objective. Jigsaw and especially
joint pretraining need rethinking or dropping — and experiment #7 (peak-weighted
objective) is now *more* attractive for them, not less.

---

## 7. Open caveats to resolve

- **MTBLS326's perfect 1.000** on 42 samples clears its permutation null (p=0.005), so it
  is not a CV artifact — but permutation does not test batch confounding. Check whether
  the Yes/No label correlates with acquisition date / run order / instrument before
  publishing this number.
- **Barth and MTBLS326 have no error bars** (LOOCV). Their SSL-vs-classical gaps are
  within single-sample noise.
- **BrC-T2D diabetes' gap is weaker than the point estimate suggests** — per-fold IQRs
  overlap substantially (`fig5_brc_t2d_fold_variability.png`).
- **Stale defaults are a live hazard.** `fewshot_benchmark.py` still defaults to
  20260708 checkpoints, and its Barth entry points at
  `aligned_128K_Workbench_Barth_Syndrome.npy` — the **completely unsuppressed** raw file.
  Always pass paths explicitly.

---

## 8. Experiment queue

Ranked by expected payoff per unit effort. Run one at a time.

### ✅ #1 — Gap decomposition (DONE, §4/§5)
```bash
python -u code/analysis/probe_logreg_advantage.py \
  --datasets brc_t2d_cancer brc_t2d_diabetes mtbls563 mtbls326 barth \
  --n-permutations 200 2>&1 | tee results/logreg_advantage_probe.log
python code/plotting/plot_logreg_advantage_probe.py
```

### ✅ #2 — Linear probe on frozen embeddings (DONE, §4b)
```bash
python -u code/analysis/linear_probe_frozen_embeddings.py 2>&1 | tee results/linear_probe_frozen.log
python code/plotting/plot_linear_probe_vs_head.py
```
**Outcome:** masking head underfit by +0.120 mean on 5/5 targets. jigsaw/joint unaffected.

### ✅ #3 — True random-backbone control (DONE, §6b)
```bash
python -u code/evaluation/ssl_linear_probe_eval.py --no-tune-C \
  --output-root results/linear_probe/exp3_pretrained_C1
python -u code/evaluation/ssl_linear_probe_eval.py --no-tune-C --random-backbone \
  --output-root results/linear_probe/exp3_randominit_C1
python code/plotting/plot_pretraining_gain.py
```
**Outcome:** masked pretraining +0.117 on 5/5; jigsaw −0.011; joint −0.025 (harmful).
Still outstanding if wanted: the same three-arm comparison *through the fine-tuning
path* (pretrained / unfrozen-reinit / fully-random), mode by mode. The frozen-feature
version above is cleaner and was sufficient to answer the question.

### #4 — Reduce `patch_size` 1024 → 256 (HIGH; targets the largest gap component)
Directly attacks the resolution ceiling that dominates cancer and MTBLS563. 512 tokens
instead of 128; attention cost grows ~16×, so 256 is the pragmatic choice over 128.
Requires re-pretraining the masking backbone on the v4 corpus.

### ✅ #5 — Linear-probe head as a first-class evaluator (DONE)
```bash
python -u code/evaluation/ssl_linear_probe_eval.py            # nested-CV tuned C
python -u code/evaluation/ssl_linear_probe_eval.py --no-tune-C # fixed C=1 (recommended)
```
`code/evaluation/ssl_linear_probe_eval.py` emits standard `summary.csv` /
`fold_metrics.csv` / `oof_predictions.csv` / `run_config.json` per dataset, reporting the
probe as an **additional** model per family rather than silently replacing the fine-tuned
heads (on MTBLS326 the fine-tuned masking head 0.981 genuinely beats the probe 0.963).

**Outcome:** masking probe beats its best fine-tuned head by **+0.054** (tuned C) /
**+0.057** (fixed C=1).

**Negative finding worth recording: nested-CV tuning of C did not help.** Mean change
across the 15 dataset×family cells was negative — it gained in 4 cells (+0.013..+0.031)
but lost in 7 (down to −0.077). With n=78 at 10-fold the inner CV selects C from ~14
samples per inner fold, so tuning adds variance without reducing bias. **Use fixed
`C=1`** at these sample sizes; the theoretically-cleaner option is empirically worse.

### #6 — Re-run the few-shot benchmark on v4 (MEDIUM)
The previous run used pre-cleaning data and 20260708 checkpoints, and its Barth default
was the unsuppressed raw file — so it needs redoing regardless. This is also where SSL's
*real* advantage should appear: with n=37–113 downstream samples, full-data CV is close to
the ceiling of what is learnable, whereas transfer/few-shot is what pretraining buys.
Pass all paths explicitly:
```bash
python -u code/evaluation/fewshot_benchmark.py --dataset barth \
  --data data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_rowMinMax_v4.npy \
  --metadata data/Barth/Workbench_Barth_Syndrome_metadata.csv \
  --label-column label --exclude-labels Pool \
  --masking-checkpoint models/masked_ssl/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth \
  --jigsaw-checkpoint models/jigsaw/multibin/20260725_085608/multibin_20260725_085608_best.pth \
  --joint-checkpoint models/joint_ssl/joint_ssl_20260725_085627/joint_ssl_20260725_085627_best.pth \
  --output-dir results/fewshot/barth_v4 2>&1 | tee results/fewshot_barth_v4.log
```
(and analogously for `--dataset mtbls326` / `mtbls563` with their v4 `_rowMinMax_v4.npy`
paths; BrC-T2D is not yet supported by this script and would need a loader added.)

### #7 — Peak-weighted pretraining objective (MEDIUM/EXPLORATORY)
Reconstruction MSE on min-max-normalized spectra is dominated by trivially predictable
baseline. A peak-weighted loss (a `topPeakLoss` variant exists in the repo) or a
contrastive objective would push capacity toward peaks. Lower priority than #3/#4 now
that pretraining is known to help rather than being inert.

### #8 — Hybrid features (CHEAP, worth a shot)
Concatenate the SSL embedding with binned areas. They are partly complementary — on
diabetes and Barth the embedding beats same-resolution binning — so the union may exceed
both.

---

## 9. Provenance

- Data: v4 (`*_v4.npy`), built by `code/preprocessing/build_clean_datasets.py` after the
  EDTA cutoff fix (commit `96f6a73`). 7 rows repo-wide retain a dominant EDTA-window peak
  (6 corpus, 1 MTBLS563) — documented residual.
- Checkpoints (all pretrained on the v3 cleaned corpus, 2026-07-25):
  masking `..._085527_..._best.pth`, jigsaw `multibin_20260725_085608_best.pth`,
  joint `joint_ssl_20260725_085627_best.pth`.
- Scripts: `code/analysis/probe_logreg_advantage.py`,
  `code/analysis/linear_probe_frozen_embeddings.py`,
  `code/plotting/plot_all_datasets_summary.py`,
  `code/plotting/plot_brc_t2d_fold_variability.py`,
  `code/plotting/plot_logreg_advantage_probe.py`,
  `code/plotting/plot_linear_probe_vs_head.py`,
  `code/evaluation/ssl_linear_probe_eval.py`,
  `code/plotting/plot_pretraining_gain.py`.
- Figures: `results/plots/all_datasets_summary_v4/fig1..fig8`.
