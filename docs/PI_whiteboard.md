# PI whiteboard: the research question tree

Photographed from the whiteboard after the discussion. This is the structure the
project has to answer. Every experiment in `docs/RESEARCH_PLAN.md` traces to a
node here.

> **Comparing foundational models with ML for healthcare applications**
> → classification
>
> **initial trials w/ masking, jigsaw & joint**
> → ML is better — sets bar, can we better it
> ↓
> **need for more data? Is data limiting.**
> → distributions of peak intensities
>   - how they change w/ increasing data
>   - have we defined experimental y-distrib. at all x.
>     - if **yes**, use synthetic data — generative AI, or LC of metabolite spectra
>     - if **no**, we end — more publicly available exptl data needed
>
> **if yes — does synthetic data help?**
>   - if **yes** → we have our model!
>   - if **no** — more parameters
>     - if **no**, then we hit a limit on these tasks

## Reading the tree

Three sequential gates, each with a terminal branch. The project can honestly
end at any of them; what it cannot do is skip one.

| gate | question | if it fails |
|---|---|---|
| **G1** | Can a pretrained model beat classical ML on these cohorts? | ML stands as the better method; report that |
| **G2** | Is performance limited by data quantity? Is the experimental intensity distribution p(y \| x) defined at all x? | If p(y \| x) is not defined everywhere, synthetic data cannot be generated honestly — conclusion: more public experimental data is needed |
| **G3** | Does synthetic data help? If not, do more parameters? | If neither, these tasks have a ceiling — a real and publishable result |

## The governing constraint

The cohorts are small (40–142) **by design, not by accident**. The premise of
the foundation model is that real downstream clinical tasks have very few
labelled samples. The research question is therefore *few-shot transfer*: does
corpus-scale self-supervised pretraining let a small cohort be classified better
than training on that cohort alone? Small n is the object of study. Every
evaluation must report uncertainty accordingly.
