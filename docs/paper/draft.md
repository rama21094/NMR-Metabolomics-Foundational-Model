# Self-supervised pretraining on public NMR spectra does not beat standard machine learning for small clinical cohorts

**Draft v0.1, 2026-10-23.** Numbers trace to `docs/RESULTS_TABLES.md` (generated from
`results/`) and `docs/CONSOLIDATION.md`. Items marked [TODO] need author input;
[REF] marks a citation to be added. No reference has been filled in from memory.

Authors: [TODO]

---

## Abstract

Foundation models pretrained on large unlabelled collections have transformed
several areas of biology, which raises the question of whether the same recipe
helps metabolomics, where labelled clinical cohorts are typically tens to a few
hundred patients. We pretrained transformer encoders on 8,545 public ¹H CPMG
blood NMR spectra (131,072 points each) with three self-supervised objectives
(masked reconstruction, jigsaw position prediction, and both jointly), and
compared them with standard machine learning on six diagnostic tasks from three
cohorts that were never seen in pretraining (37 to 113 patients). Both arms used
the same classifier and identical splits, at exact few-shot budgets and with
full-data leave-one-out or repeated cross-validation, and every comparison is
reported as a paired difference with a 95% interval. Across 21 comparisons the
best pretrained model was never better than standard ML, was worse in 17 and
statistically level in 4 (mean difference −0.067 in balanced accuracy; −0.048
with position-preserving pooling). Fine-tuning did not help, and models from
1.8 to 23.5 million parameters were behind at every size. In 45 controlled
pretraining runs, neither more spectra nor a more even mix of studies closed the
gap, and the number of studies needed to reach parity could not be extrapolated.
The public corpus, although drawn from 12 studies, has an effective study count
of 4.1. Synthetic spectra from a metabolite basis or from generative adversarial
networks failed a realism test, and adding 4,000 GAN spectra to pretraining made
transfer slightly worse. We also report several silent failure modes, including
a chemical-shift axis defect and a batch-confounded public cohort, that produced
plausible but wrong results. With current public data, standard machine learning
remains the appropriate baseline for small NMR metabolomics cohorts.

---

## 1. Introduction

[TODO: 2–3 paragraphs of context with REFs.]

- Self-supervised pretraining and the "foundation model" recipe in biology:
  masked modelling on sequences and images [REF]; early applications to spectra
  [REF].
- NMR metabolomics: reproducible, quantitative, cheap per sample, but clinical
  studies are small, and public repositories (MetaboLights, Metabolomics
  Workbench) hold thousands of spectra across many studies [REF].
- The appeal: learn spectral structure once from public data, then classify small
  cohorts with few labels.

What this paper does. We ask three questions in order, each with a pre-specified
test:

1. **G1.** Does pretraining on public spectra beat standard machine learning on
   small clinical cohorts?
2. **G2.** If not, is the pretraining data the limit: its volume, its diversity
   of studies, or the model's size?
3. **G3.** If data is the limit, can synthetic spectra supply what is missing?

Contributions.

- A controlled, paired comparison of three self-supervised objectives against
  standard ML, with exact label budgets and intervals, on six tasks.
- Scaling experiments that separate spectrum count from study diversity, using an
  entropy-based effective study count.
- A realism test for synthetic NMR spectra and a direct test of synthetic
  augmentation.
- A record of failure modes that produced plausible, wrong results, as guidance
  for others building on public NMR data.

---

## 2. Data

### 2.1 Pretraining corpus

Public ¹H NMR spectra of blood (serum and plasma) were collected from
MetaboLights and Metabolomics Workbench. Only CPMG experiments were kept, by
upstream selection. [TODO: list of accessions, as a supplementary table.]

All spectra were rebuilt from the raw Bruker processed data onto one fixed axis
from +10.0 to −0.5 ppm (131,072 points), keeping each spectrum's own referencing
offset, enforcing CPMG and ¹H acquisition, and removing near-duplicates
(Pearson r > 0.99999). This yielded 9,480 spectra. The corpus was split by
study, not by spectrum: 8,545 spectra from 12 studies for pretraining, the rest
held out. Each spectrum was scaled by its own minimum and maximum.

The rebuild was necessary because the original pipeline resampled each spectrum
onto its own ppm range and re-referenced it to its rightmost peak, so that a
given index meant a different chemical shift in different studies (Section 6.1).
After the rebuild, the lactate CH₃ doublet sits at 1.325 ppm (true 1.330; 95%
span 0.117 ppm) and the lactate CH₃–CH separation is correct.

**Effective number of studies.** Study composition is highly unequal: one study
contributes 57% of the pretraining spectra, four contribute 8–12% each, and seven
contribute under 1% each. We summarise this as the effective number of studies,
exp(H) where H is the Shannon entropy of the study proportions (the Hill number
of order 1). The pretraining corpus has 12 nominal studies but an effective
4.09. [Figure: study shares, `docs/figures/pi25_study_share.png`.]

### 2.2 Evaluation cohorts

| cohort | source | patients | tasks |
|---|---|---|---|
| MTBLS563 | public | 113 (29 "unknown" excluded) | infection vs surgical control; control vs viral vs bacterial (56/31/26) |
| BrC-T2D | in-house [TODO: ethics statement] | 78 | breast cancer; type 2 diabetes; four combined groups (10/28/20/20) |
| Barth | public (Metabolomics Workbench) | 37 | Barth syndrome vs control (23/14) |

None of these studies is in the pretraining corpus (checked by accession). For
BrC-T2D, the analysis uses a corrected label file listing 78 patients; one
spectrum per patient is used, with repeat acquisitions removed so that a patient
cannot appear in both training and test data.

**Excluded cohort.** MTBLS326 (41 usable samples) was excluded because its
design confounds class with acquisition batch: cases are samples 1–27 and
controls 101–130, recorded as separate blocks. A classifier trained only on
metabolite-free regions (above 9.5 and below −0.5 ppm) separates the classes at
0.726 (p < 0.005), so no method can distinguish disease from batch in this
cohort.

---

## 3. Methods

### 3.1 Encoder and pretraining objectives

All objectives share one encoder. A spectrum is cut into 128 non-overlapping
patches of 1,024 points; each patch is linearly embedded, a sinusoidal position
code is added, and a pre-norm transformer encoder (baseline: width 192, 4 layers,
6 heads, feed-forward 768, about 4.0 million parameters) produces one token per
patch.

- **Masking.** For each spectrum a fraction drawn uniformly from 20–60% of
  patches is hidden. The model reconstructs all 131,072 points. Loss: mean
  squared error on hidden patches plus 0.3 × the error on visible patches.
- **Jigsaw.** The 128 patches are shuffled; for each, the model predicts its
  original position (128-way). Loss: cross-entropy.
- **Joint.** Masked reconstruction (visible-patch weight 0.1) plus a jigsaw term
  over pieces of 256, 512, 1,024 and 2,048 points, with a soft target (a Gaussian
  over position distance) so that near misses cost less.

Training: AdamW, learning rate 10⁻⁴ with warm-up and cosine decay, dropout 0.2,
batch 32, up to 400 epochs with early stopping (patience 50), mixed precision,
5 seeds per objective, on NVIDIA L40S GPUs (about 4.5 GPU-hours per run).
Patch size 1,024 was chosen in an earlier sweep against 256 and 128.

### 3.2 Readout

Pretrained encoders are frozen. Token embeddings are pooled into one vector per
spectrum, either by averaging all 128 tokens (mean pooling) or by averaging
within 16 contiguous groups of 8 tokens and concatenating (16-region pooling,
3,072 features), which keeps coarse chemical-shift position. Features are
standardised and classified with class-balanced logistic regression.

Two learned readouts were also tested. **Attention pooling:** a query vector
scores each standardised token, and the softmax-weighted average feeds a linear
softmax classifier; query and classifier are trained together on the training
patients only (class-balanced cross-entropy, L2 penalty), starting from zero so
that the readout begins at mean pooling. **CLS token:** a learnable token
prepended after positional encoding during pretraining, with no loss of its own
(as in masked autoencoders [REF]), whose final state is read out with the same
logistic regression.

### 3.2b A linear, label-free baseline

To separate "pretraining on this corpus" from "this encoder", PCA was fitted on
the binned pretraining spectra (no labels) and the top k = 5 to 100 components
used as features for the same classifier.

### 3.3 Standard machine learning

Spectra are binned into 0.02 ppm windows over −0.5 to 9.5 ppm, with the water
region (4.45–5.10 ppm) excluded, and each binned spectrum is normalised to unit
total absolute intensity. For the paired comparisons the classifier is the same
balanced logistic regression as in 3.2, so any difference comes from the
features. Separately, to set the bar, eight classical models (L1 and L2 logistic
regression, linear and RBF SVM, random forest, gradient boosting, PLS-DA and LDA)
were tuned with nested cross-validation at two bin widths (Table 1 in the
results tables).

### 3.4 Evaluation protocol

- **Few-shot.** Exactly k = 5, 10, 20 or 40 labelled patients per class for
  training, the rest for testing; a budget is skipped for a task if any class
  would keep fewer than 5 test patients. 50 episodes per budget.
- **Full data.** Leave-one-out cross-validation for cohorts under 100 patients;
  10 repeats of stratified 5-fold cross-validation otherwise.
- **Metric.** Balanced accuracy (mean per-class recall), so chance is 1/number
  of classes regardless of imbalance.
- **Pairing and intervals.** Both arms see identical splits. Pretrained
  predictions are averaged over seeds per split (majority vote under LOOCV).
  Differences are reported with 95% intervals: ±1.96 standard errors over
  episodes or repeats, or a paired bootstrap over patients (2,000 resamples) for
  LOOCV. A comparison is "better" or "worse" when the interval excludes zero, and
  "level" otherwise.
- Split seeds are derived from a stable hash (CRC32) of the task name, so splits
  are identical across runs and processes.

This gives 21 task-by-budget comparisons.

### 3.5 Fine-tuning

The last transformer block, or the whole encoder, was adapted on each episode's
support set, then read out by the same probe as 3.2, against a frozen and a
randomly initialised encoder (12 episodes, 3 seeds).

### 3.6 Scaling experiments

45 masking pretraining runs on subsets of the corpus, along three axes: (i)
number of spectra at constant study mix (858 to 6,410 spectra, effective 4.1
studies); (ii) number of studies (3, 6, 9 studies); (iii) fixed budgets of 2,000
or 4,000 spectra drawn to give 2, 4, 8 or 12 studies (effective 1.7 to 6.9).
Effects were estimated by regressing the mean balanced accuracy over the six
tasks on log spectrum count and on effective studies, with permutation tests.
To extrapolate to parity with standard ML, the gap was regressed on effective
studies (linear, and linear in log), with 5,000 bootstrap resamples of runs.

### 3.7 Model size

Five sizes (width/layers: 96/2, 128/3, 192/4, 320/6, 448/8; 1.82 to 23.52
million parameters), 3 seeds each, patch size fixed.

### 3.8 Synthetic spectra

- **Metabolite basis.** Reference spectra of 87 metabolites from GISSMO [REF],
  plus lipid components (94 in total), placed on the corpus axis, were fitted to
  real spectra by non-negative least squares; synthetic spectra were drawn from
  the fitted coefficient distributions with matched noise.
- **GANs.** A 1D Wasserstein GAN with gradient penalty (weight 10, 5 critic steps
  per generator step). Generator: a 128-dimensional noise vector, a linear layer
  to 512 channels × 64 points, nine blocks of linear upsampling, convolution
  (kernel 9), batch normalisation and LeakyReLU, and a tanh output of 32,768
  points interpolated to 131,072. Critic: nine strided convolutions (kernel 9).
  Real spectra were mean-pooled by 4 for training. v2 added the bounded output,
  an exponential moving average of generator weights (0.999), and checkpoint
  selection by the realism test on a separate real set. A study-conditional
  variant used a projection critic and generated from random blends of two study
  embeddings, weighting studies equally.
- **Realism test.** Logistic regression trained to separate synthetic from real
  binned spectra, with cross-validation grouped by study; AUC 0.5 means
  indistinguishable. Also reported: landmark positions (lactate CH₃–CH
  separation, expected 2.780 ppm), per-window Wasserstein distance divided by the
  real interquartile range (real-vs-real floor 0.053), PCA fitted on real spectra
  only, and per-region classifiers.
- **Augmentation test.** Masking pretraining on the real corpus versus the real
  corpus plus 4,000 GAN v2 spectra, both with spectrum ends (below −0.2 and above
  9.4 ppm) set to zero, 3 seeds each, evaluated as in 3.4.

---

## 4. Results

### 4.1 Pretraining does not beat standard ML (G1)

[Figure 1: paired differences per task and budget, `docs/figures/fig7_G1_fixed.png`.]

| pretraining | better | worse | level | mean difference |
|---|---|---|---|---|
| masking, mean pooling | 0 | 17 | 4 | −0.067 |
| masking, 16-region pooling | 0 | 17 | 4 | −0.048 |
| jigsaw | 0 | 19 | 2 | −0.124 |
| joint | 0 | 19 | 2 | −0.108 |

Masking was the best objective, and in no comparison was any pretrained model
better than standard ML. The level comparisons were all full-data comparisons on
the smaller cohorts, where leave-one-out intervals are wide (0.06 to 0.17 either
side); the closest was BrC-T2D cancer, +0.012 [−0.030, +0.053] with 16-region
pooling. Keeping coarse position in the readout narrowed the mean gap by 0.019,
most at 5 patients per class and on BrC-T2D, but changed no verdict.

Pretraining is not useless: masking improved on the same encoder left untrained
by 0.046. Fine-tuning did not add to it: adapting the last block changed the
frozen result by −0.008, and adapting the whole encoder by 0.000.

**Readouts and corpus.** No readout change closed the gap (Table X). Learned
attention pooling overfitted the small training sets (mean −0.084, worse than
averaging). A CLS token was the weakest readout (−0.095): without a loss of its
own it does not become a useful summary. Pretraining on the full 9,480-spectrum
corpus, which raises the effective study count from 4.1 to 5.9 by adding seven
unseen studies, changed the gap by −0.010 (mean pooling; 3 seeds), within seed
noise. The best readout remains the fixed 16-region pooling of the original
model (−0.048).

| model | corpus (effective studies) | mean pool | 16-region | other |
|---|---|---|---|---|
| masking | 8,545 (4.1) | −0.067 | −0.048 | attention −0.084 |
| masking | 9,480 (5.9) | −0.077 | −0.057 | — |
| masking + CLS | 9,480 (5.9) | −0.071 | −0.051 | CLS −0.095 |

**A linear baseline does better.** PCA fitted on the same unlabelled corpus,
followed by the same classifier, came within −0.018 of binned spectra using 40
components (99.7% of the variance), −0.040 with 10 and −0.073 with 5; 100
components matched binning (+0.004). The 192-dimensional transformer embedding
(−0.067) is therefore worse than a 40-component linear compression of the same
data. Different pretraining seeds learned very similar representations (linear
CKA on standardised embeddings 0.970 between seeds, against 0.870 to an untrained
encoder), so the shortfall is not seed instability.

For reference, the best tuned classical models reached 0.838 (MTBLS563 binary),
0.698 (3-class), 0.930 (BrC-T2D cancer), 0.840 (diabetes), 0.662 (4-class) and
0.825 (Barth).

### 4.2 More of the same data does not close the gap (G2)

[Figure 2: gap against spectrum count and against effective studies,
`docs/figures/pi25_spectra_vs_studies.png`.]

- **Spectra.** At constant study mix, a 7.5-fold increase in spectra gained
  +0.003 per doubling (p = 0.17).
- **Study mix.** At fixed budgets, effective studies did not predict transfer
  (p = 0.19). Pooled over all 45 runs, it reached p = 0.043 at full data and
  p = 0.14 at 10 per class; spectrum count was significant in the pooled model
  (p = 0.003) but small, under 0.01 per standard deviation.
- **Parity.** A linear fit of the gap on effective studies reaches zero at 54
  studies (95% interval 21 to unbounded; 15% of bootstrap resamples never reach
  it). A fit with diminishing returns places it in the hundreds or beyond. Parity
  cannot be extrapolated from the 1.6–6.9 range the corpus allows.

Both effects are an order of magnitude smaller than the 0.06–0.08 gap.

### 4.3 Model size helps up to about 10M parameters, then stops

[Figure 3: gap against model size, `docs/figures/pi25_model_size.png`.]

The gap at full data was −0.082 (1.82M), −0.081 (2.38M), −0.062 (3.99M), −0.041
(10.54M) and −0.047 (23.52M). Smaller models were worse, so the baseline is not
over-parameterised for the pretraining data; larger models did not continue to
improve.

### 4.4 Synthetic spectra fail the realism test (G3)

| set | AUC | AUC, ends blanked | CH₃–CH (ppm) | distance | effective studies |
|---|---|---|---|---|---|
| GISSMO + lipids | 1.000 | — | 2.845 | 4.28 | — |
| GAN v1 | 0.945 | — | 2.788 | 1.19 | 4.71 |
| GAN v2 | 0.832 | 0.740 | 2.787 | 1.03 | 4.38 |
| GAN v2, study-conditional | 0.922 | 0.896 | 2.795 | 1.15 | 8.38 |

The metabolite basis explains a median R² of 0.25 of a real spectrum; the broad
protein and lipid background and unassigned peaks are not modelled. GANs place
peaks correctly and come much closer, but none passes. The study-conditional GAN
is the only generator that spreads beyond the corpus's 4.1 effective studies,
at the cost of realism and training stability.

**What distinguishes GAN spectra.** [Figure 4: PCA and per-region separability,
`docs/figures/pi16_gan_pca.png`, `docs/figures/pi16_gan_region_auc.png`.] On the
first principal components of real spectra, GAN v2 spectra fall inside the real
distribution (3.5% outside the real 95% boundary, against 4.9% for held-out real
spectra) but concentrate in its dense centre and miss the outer groups. 81% of a
GAN spectrum's variation lies outside the 20 leading real components (12% for
real spectra). By region, the far ends of the spectrum, which carry no signal in
real data, separate GAN from real at 0.955; the area around the TSP reference at
0.826; and the metabolite region at 0.669. GAN baselines drift at the ends, and
around TSP they dip below zero more often than real spectra (16% of points
against 7%).

### 4.5 Adding GAN spectra to pretraining makes transfer slightly worse

Against pretraining on real spectra alone, adding 4,000 GAN v2 spectra was better
in 3 comparisons, worse in 10 and level in 8 (mean −0.016). The three gains were
all at 5 patients per class. The augmented runs trained for about twice as many
epochs, because synthetic spectra in the early-stopping set kept the validation
loss falling; this would be expected to favour them.

---

## 5. Discussion

**Why pretraining does not help here.** The clearest evidence is that a linear
PCA of the same unlabelled corpus transfers better than the pretrained
transformer. The spectra are highly redundant (five components explain 90.5% of
the variance of the binned corpus), and most, but not all, of the diagnostic
signal lies in the leading components: 40 components holding 99.7% of the
variance still lose 0.018, which is only recovered by components 50 to 100,
holding 0.15% of the variance. This reconciles two statements that look
contradictory, that the data are "too simple" and that there are "too few"
samples: the unlabelled spectra are simple, so reconstruction is easy and
dominated by high-variance structure, while part of the disease signal sits in
low-variance directions that such an objective has little reason to preserve and
that a few dozen labelled patients cannot recover from a compressed
representation. Three further observations point the same way.
First, the masked-reconstruction task is easy for a reason unrelated to disease:
in preliminary experiments on the pre-rebuild corpus, copying the most similar
training spectrum reconstructed masked regions almost as well as the network
(correlation 0.90 against 0.92), and five principal components held 86% of the
variance. [TODO: rerun this baseline on the rebuilt corpus before submission.]
Lower reconstruction loss did not predict better transfer. Second, the
diagnostic signal in these cohorts lies in small, specific intensity differences
that binned spectra expose directly to a linear classifier. Third, the corpus
is dominated by a few studies, so what the encoder learns is largely how those
studies differ from each other.

**Data, not architecture, is the constraint we could not test.** The scaling
experiments show that more of the same data does not help, and that a more even
study mix does not help measurably within reach. They do not show that many
more independent studies would not help; 4.1 effective studies is too few to
answer that, and the parity extrapolation is unbounded. Public NMR metabolomics
would need many more independent studies with raw data deposited before the
question can be settled.

**Synthetic data cannot stand in for studies.** A generator trained on the
corpus reproduces its composition (4.4 effective studies for GAN v2), so even a
realistic one would add spectra, not studies. Conditioning on study can spread
generation more widely, but at the cost of realism. The specific artefacts, at
the spectrum ends and around the reference peak, suggest that border handling in
the convolutional generator (zero padding) is a first target; we have not tested
this.

**Fragile results.** Several intermediate conclusions in this project changed
when data or code were corrected, and we report them because they are general
risks (Section 6). In particular, the diversity effect in G2 was significant
(p = 0.014) before one cohort's labels were corrected and not after (p = 0.19).

---

## 6. Pitfalls that produced plausible, wrong results

1. **Chemical-shift axis.** Re-referencing each spectrum to its rightmost peak
   and resampling it onto its own range left 11.7% of spectra on a different axis.
   Each spectrum looked normal alone; alignment checks based on cross-correlation
   passed. A landmark test (lactate position and CH₃–CH separation) caught it.
2. **Batch-confounded cohort.** A near-perfect classifier on MTBLS326 reflected
   acquisition order; label-permutation tests cannot detect this, because the
   batch signal is real signal. Per-spectrum min–max scaling amplified it by
   turning a noise-level difference into a feature.
3. **Checkpoint selection.** Choosing the best of 15 checkpoints produced
   apparent wins for pretraining twice; single-run variation between seeds was
   0.045.
4. **Silent numerical failures.** `numpy.interp` on a descending axis returned
   zeros without error; Python's per-process string hash made "identical" splits
   differ; a results join on a wrong key reported zero points instead of failing.
5. **Readout defaults.** Mean pooling was inherited as a default despite earlier
   evidence that it discards position; checked late (Section 4.1).

---

## 7. Limitations

- One modality: ¹H CPMG spectra of blood. Results may differ for other
  experiments or biofluids.
- Cohort sizes of 37–113 patients; several "level" results are "cannot tell".
- One encoder family; other families were not tried.
- The pretraining corpus spans only 1.6–6.9 effective studies.
- One metabolite basis and one GAN family for synthetic data.
- Most experiments used mean pooling; their absolute gaps are probably about
  0.02 pessimistic, though their comparisons are like for like.

---

## 8. Conclusion

With the public NMR data available today, self-supervised pretraining did not
improve on standard machine learning for small clinical cohorts, and neither
more data of the same kind, larger models, nor synthetic spectra changed that.
Standard machine learning with careful, paired evaluation should remain the
baseline. The most useful next step for the field is more independent studies
with raw data deposited, not more spectra from the same studies.

---

## Data and code availability

[TODO] Code: repository link. Public spectra: accession list (supplementary).
Results: every number in this draft is generated from committed files under
`results/` by `code/reporting/build_results_tables.py`. BrC-T2D data: [TODO:
access statement; patient identifiers are not released].

## Figures (current files)

1. G1 paired differences: `docs/figures/fig7_G1_fixed.png`
2. Scaling: `docs/figures/pi25_spectra_vs_studies.png`
3. Model size: `docs/figures/pi25_model_size.png`
4. GAN vs real: `docs/figures/pi16_gan_pca.png`, `docs/figures/pi16_gan_region_auc.png`
5. Supplementary: axis defect `docs/figures/fig5_slim.png`; study shares
   `docs/figures/pi25_study_share.png`; spectrum ends `docs/figures/pi16_gan_edges.png`

[TODO: figures need a consistent journal style; Figure 1 should be regenerated
with the 16-region pooling arm added.]
