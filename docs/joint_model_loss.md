# The joint (masking + jigsaw) model: architecture and loss

For Prof Shyam, following the 2026-10-23 meeting. Everything below is read from
`code/training/train_joint_ssl.py` and the configuration the reported runs used
(`results/phase2/joint_seed*/…/config.json`).

**One correction to the meeting notes:** the joint model is *one shared encoder
with two output heads*, not two backbones fused by a fully connected layer. The
only "fusion" layer combines each token's intensity embedding with its position
embedding at the input.

---

## Notation

- A spectrum $x \in \mathbb{R}^{L}$, $L = 131{,}072$, min-max scaled.
- For a piece size $s$, the spectrum is cut into $n_s = L/s$ contiguous pieces
  $x_1, \dots, x_{n_s} \in \mathbb{R}^{s}$ (no overlap).

## Shared encoder $f_\theta(\cdot\,;\, s, t)$

For piece size $s$ and task $t \in \{\text{mask}, \text{jigsaw}\}$, the token in
slot $j$ is

$$
h^{(0)}_j = \mathrm{Fuse}\big(\big[\, u_j \,;\, p_j \,\big]\big) + P\,e_t + c_s ,
$$

where

- $u_j = \mathrm{Dropout}(\mathrm{LN}(W_s x_j))$ is the intensity embedding
  (one linear projection per piece size), replaced by a learned mask vector
  $u_{\text{mask}}$ when the piece is masked;
- $p_j = E_{\text{slot}}[j] + \phi(j)$ is the position of the *slot*: a learned
  slot embedding plus Fourier features (8 bands);
- $\mathrm{Fuse}$ is Linear($2d \to d$), GELU, LayerNorm, Dropout;
- $P\,e_t$ is a task signal through an 8-dimensional bottleneck ($e_t \in
  \mathbb{R}^{8}$, $P \in \mathbb{R}^{d \times 8}$, initialised small), so the
  encoder cannot easily split into two separate sub-networks;
- $c_s$ is a learned per-piece-size vector.

Then 4 transformer layers (width $d = 192$, 6 heads, feed-forward 768, dropout
0.15) with a learned relative-position bias on attention, and a final LayerNorm:
$h = f_\theta(x; s, t) \in \mathbb{R}^{n_s \times d}$.

## Task 1: masked reconstruction ($s = 1{,}024$, $n = 128$)

1. Draw a ratio $r \sim U(0.2, 0.6)$ per spectrum and mask $\mathrm{round}(r\,n)$
   pieces chosen uniformly: $m_j \in \{0, 1\}$.
2. Encode with masked pieces replaced by $u_{\text{mask}}$:
   $h = f_\theta(x; 1024, \text{mask})$.
3. Reconstruct each piece, with a skip path for visible pieces only:
   $$\hat{x}_j = W_r h_j + 0.3\,(1 - m_j)\, S\, x_j ,$$
   with $S \in \mathbb{R}^{1024 \times 1024}$ initialised to the identity.
4. Per-piece error $\;e_j = \tfrac{1}{1024}\lVert \hat{x}_j - x_j \rVert_2^2$.
5. **Peak weighting.** Only pieces that carry signal are scored:
   $w_j = 1$ if piece $j$ is in the top 17.5% of the spectrum's pieces by
   $\tfrac12 \operatorname{mean}|x_j| + \tfrac12 \max|x_j|$, else $0$.

$$
\mathcal{L}_{\text{mask}} = \frac{\sum_j m_j w_j e_j}{\sum_j m_j w_j},
\qquad
\mathcal{L}_{\text{vis}} = \frac{\sum_j (1 - m_j) w_j e_j}{\sum_j (1 - m_j) w_j}.
$$

## Task 2: jigsaw at four scales ($s \in \{256, 512, 1024, 2048\}$)

1. Each step, the batch is split at random into four groups, one per piece size,
   so every spectrum sees one scale per step and all scales over an epoch.
2. Draw a permutation $\pi$ of $\{1, \dots, n_s\}$ and place piece $x_{\pi(j)}$
   in slot $j$. The label of slot $j$ is its original position $\pi(j)$.
3. Encode without masking, $h = f_\theta(\text{shuffled}; s, \text{jigsaw})$,
   and predict a distribution over positions, $z_j = W^{(s)}_{\text{jig}} h_j \in
   \mathbb{R}^{n_s}$.
4. **Soft, distance-aware target.** Placing a piece one slot away is penalised
   less than placing it far away:
   $$q_j(k) \propto \exp\!\Big(-\frac{(k - \pi(j))^2}{2\sigma^2}\Big),\quad \sigma = 1,
   \qquad q_j \leftarrow (1-\varepsilon)\,q_j + \varepsilon / n_s,\quad \varepsilon = 0.05 .$$
5. Cross-entropy, averaged over slots and spectra, then over the four scales:
   $$\mathcal{L}_{\text{jig}} = \frac{1}{4}\sum_{s} \operatorname{mean}_{j}
   \Big(-\sum_{k} q_j(k)\,\log \operatorname{softmax}(z_j)_k\Big).$$

## Total loss, with scale balancing

The three terms have very different magnitudes (reconstruction about $10^{-4}$,
cross-entropy about 1–5), so each is divided by a running estimate of its own
size before weighting:

$$
\mathcal{L} = \frac{\mathcal{L}_{\text{mask}}}{\bar{s}_{\text{mask}}}
+ 0.1\,\frac{\mathcal{L}_{\text{vis}}}{\bar{s}_{\text{vis}}}
+ 1.0\,\frac{\mathcal{L}_{\text{jig}}}{\bar{s}_{\text{jig}}},
\qquad
\bar{s} \leftarrow 0.98\,\bar{s} + 0.02\,\operatorname{sg}(\mathcal{L}_\bullet),
$$

where $\operatorname{sg}$ stops the gradient, each $\bar{s}$ starts at the first
observed value, and the estimates update on training batches only. In effect
each task contributes in proportion to its *relative* improvement, not its raw
scale. (A coverage regulariser exists in the code but had weight 0.)

Optimiser: AdamW, learning rate $3 \times 10^{-4}$, weight decay 0.05, batch 32,
up to 400 epochs, early stopping on validation loss (patience 50), 5 seeds.

## Embedding used downstream

The frozen encoder is run on the *unshuffled, unmasked* spectrum along both
paths, and each path's tokens are mean-pooled:

$$
z(x) = \big[\, \bar{h}^{\text{jig}}_{256} ;\, \bar{h}^{\text{jig}}_{512} ;\,
\bar{h}^{\text{jig}}_{1024} ;\, \bar{h}^{\text{jig}}_{2048} ;\,
\bar{h}^{\text{mask}}_{1024} \,\big] \in \mathbb{R}^{5 \times 192 = 960},
$$

followed by standardisation and balanced logistic regression, as for every other
model.

---

## How it differs from the masking-only model

| | Masking model | Joint model |
|---|---|---|
| Position encoding | sinusoidal, added | learned slot + Fourier, fused; relative bias in attention |
| Masked-piece loss | all pieces | top 17.5% of pieces by peak score only |
| Visible-piece weight | 0.3 | 0.1 (after scale balancing) |
| Extra task | none | jigsaw at four scales, soft targets |
| Optimiser | AdamW, lr $10^{-4}$, wd 0.01 | AdamW, lr $3 \times 10^{-4}$, wd 0.05 |
| Downstream embedding | 192 | 960 |

Result, for context: joint transferred worse than masking alone (worse than
standard ML in 19 of 21 comparisons against 17 for masking; mean gap −0.108
against −0.067).
