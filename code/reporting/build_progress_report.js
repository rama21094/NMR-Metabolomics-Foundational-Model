const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
        Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
        PageBreak, Footer, PageNumber } = require("docx");
const fs = require("fs");

const NAVY = "21295C", CORAL = "C1435B", GREEN = "1A7A3C", MUTED = "5A6068";
const F = "Calibri", FH = "Cambria";
const PT = (n) => n * 2;

const P = (text, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after === undefined ? 140 : o.after, line: 276 },
  indent: o.indent,
  children: [new TextRun({ text, font: F, size: PT(o.size || 11),
    bold: o.bold, italics: o.italics, color: o.color || "1A1A1A" })],
});
const Rich = (runs, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after === undefined ? 140 : o.after, line: 276 },
  indent: o.indent,
  children: runs.map(r => new TextRun({ text: r.t, font: F, size: PT(o.size || 11),
    bold: r.b, italics: r.i, color: r.c || "1A1A1A" })),
});
const H1 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 },
  children: [new TextRun({ text: t, font: FH, size: PT(15), bold: true, color: NAVY })],
});
const H2 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 110 },
  children: [new TextRun({ text: t, font: FH, size: PT(12.5), bold: true, color: NAVY })],
});
const Bul = (t, o = {}) => new Paragraph({
  bullet: { level: 0 }, spacing: { after: 90, line: 276 },
  children: [new TextRun({ text: t, font: F, size: PT(11), color: "1A1A1A" })],
});
const BulRich = (runs) => new Paragraph({
  bullet: { level: 0 }, spacing: { after: 90, line: 276 },
  children: runs.map(r => new TextRun({ text: r.t, font: F, size: PT(11),
    bold: r.b, italics: r.i, color: r.c || "1A1A1A" })),
});

const cell = (text, { bold, fill, color, align, w } = {}) => new TableCell({
  width: { size: w || 2000, type: WidthType.DXA },
  shading: fill ? { type: ShadingType.CLEAR, fill } : undefined,
  margins: { top: 70, bottom: 70, left: 110, right: 110 },
  children: [new Paragraph({
    alignment: align || AlignmentType.LEFT, spacing: { after: 0 },
    children: [new TextRun({ text, font: F, size: PT(10),
      bold, color: color || "1A1A1A" })],
  })],
});
const table = (header, rows, widths) => new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: widths,
  borders: {
    top: { style: BorderStyle.SINGLE, size: 2, color: "C8D0D8" },
    bottom: { style: BorderStyle.SINGLE, size: 2, color: "C8D0D8" },
    left: { style: BorderStyle.SINGLE, size: 2, color: "C8D0D8" },
    right: { style: BorderStyle.SINGLE, size: 2, color: "C8D0D8" },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "DDE3E9" },
    insideVertical: { style: BorderStyle.SINGLE, size: 2, color: "DDE3E9" },
  },
  rows: [
    new TableRow({ tableHeader: true, children: header.map((h, i) =>
      cell(h, { bold: true, fill: NAVY, color: "FFFFFF", w: widths[i] })) }),
    ...rows.map(r => new TableRow({ children: r.map((c, i) =>
      cell(String(c), { w: widths[i], bold: i === 0 })) })),
  ],
});

const doc = new Document({
  creator: "Shankararama Sharma",
  title: "A foundation model for NMR metabolomics — progress report",
  styles: { default: { document: { run: { font: F, size: PT(11) } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: F,
        size: PT(9), color: MUTED })] })] }) },
    children: [

// ---------------- title ----------------
new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 60 },
  children: [new TextRun({ text: "A Foundation Model for NMR Metabolomics",
    font: FH, size: PT(20), bold: true, color: NAVY })] }),
new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 240 },
  children: [new TextRun({ text: "Progress report — September 2026",
    font: FH, size: PT(13), color: MUTED })] }),
Rich([{ t: "Shankararama Sharma", b: true }, { t: "  ·  Prepared for the co-investigator, "
  + "as a first orientation to the project and its findings to date.", c: MUTED }],
  { after: 260 }),

// ---------------- 1 ----------------
H1("1. Goal and background"),
P("Nuclear magnetic resonance (NMR) spectroscopy of blood serum produces, for each patient "
  + "sample, a single trace that records the concentrations of hundreds of small molecules — "
  + "metabolites — simultaneously. Because metabolite levels shift with disease, these spectra "
  + "are an attractive basis for diagnostic classification: distinguishing patients from "
  + "controls, or one disease state from another."),
P("The obstacle is sample size. Deep neural networks are the standard tool for pattern "
  + "recognition of this kind, but they typically require tens of thousands of labelled "
  + "examples. Clinical metabolomics studies routinely have between 30 and 150. This is the "
  + "gap the project set out to close."),
Rich([{ t: "The strategy is the one that transformed language and vision: ", },
  { t: "self-supervised pretraining", b: true },
  { t: ". A network is first trained on a large pool of ", },
  { t: "unlabelled", i: true },
  { t: " spectra using a task that needs no diagnosis — here, hiding portions of a spectrum "
     + "and asking the model to reconstruct them. The hope is that solving this puzzle forces "
     + "the model to learn the underlying structure of NMR spectra, after which only a small "
     + "labelled dataset is needed to adapt it to a specific clinical question. A model used "
     + "this way is often called a foundation model." }]),
P("We assembled a pretraining corpus of 9,670 serum spectra from public repositories "
  + "(MetaboLights and Metabolomics Workbench), and set aside four independent clinical "
  + "cohorts, entirely unseen during pretraining, for evaluation. The central question is "
  + "simple to state: does this pretraining actually help, compared with conventional "
  + "statistical machine learning applied directly to the same spectra?"),

// ---------------- 2 ----------------
H1("2. Principal finding"),
Rich([{ t: "It does not. ", b: true },
  { t: "Across every evaluation cohort and every labelled-sample budget we tested, ordinary "
     + "logistic regression on binned spectral intensities matched or beat the pretrained "
     + "neural network. The clearest test paired the two methods on identical data splits, so "
     + "that sampling variation cancels: at the smallest labelled-sample budget the difference "
     + "was +0.001 ± 0.016 in balanced accuracy (p = 0.74) — indistinguishable from zero." }]),
P("Two results that initially appeared to favour the neural approach did not survive scrutiny, "
  + "and both failures are instructive:"),
BulRich([{ t: "A single cohort (Barth syndrome) showed the network beating classical machine "
  + "learning by a wide margin. ", },
  { t: "This turned out to be chance.", b: true },
  { t: " Repeating the pretraining with five different random seeds showed that the reported "
     + "figure was the best of the five and sat 1.5 standard deviations above the group mean; "
     + "the other four seeds all fell below the classical baseline." }]),
BulRich([{ t: "A second cohort produced a perfect classification score on 42 samples. ", },
  { t: "This dataset is confounded by experimental design.", b: true },
  { t: " Cases were measured as samples 1–27 and controls as samples 101–130 — two separate "
     + "instrument sessions. Any technical drift therefore predicts the diagnosis perfectly. We "
     + "confirmed the effect is real, not merely possible: regions of the spectrum that contain "
     + "no metabolite signal at all still classify the label at 73% accuracy." }]),
P("With those two removed, the honest record on the remaining cohorts is zero wins for the "
  + "pretrained model."),

// ---------------- 3 ----------------
H1("3. Why it fails — the mechanism"),
P("A negative result is only useful if it comes with an explanation, and the most important "
  + "work of the last month was establishing one."),
Rich([{ t: "The pretraining task is too easy. ", b: true },
  { t: "Reconstruction quality had always been reported without a baseline — the equivalent of "
     + "quoting a classifier's accuracy without saying what chance performance is. When we "
     + "supplied the missing baselines, the picture changed completely." }]),
table(
  ["Method for filling in hidden spectral regions", "Accuracy (correlation)"],
  [["Linear interpolation from neighbouring points", "0.38"],
   ["The average spectrum of the whole corpus", "0.56"],
   ["Copying the most similar other patient's spectrum", "0.90"],
   ["The trained neural network", "0.92"]],
  [7000, 2360]),
P("", { after: 100 }),
Rich([{ t: "Simply copying another patient's spectrum performs almost as well as the network. ",
  b: true },
  { t: "The reason is that the corpus is far more homogeneous than its size suggests: 86% of "
     + "all variation lies in just five underlying dimensions, and the typical spectrum has a "
     + "near-twin elsewhere in the corpus (correlation 0.99). The reconstruction task can be "
     + "solved from population averages alone, so it never compels the model to learn the fine "
     + "chemical detail that distinguishes one patient from another. Disease signal is a small "
     + "perturbation on top of a very strong common pattern, and the training objective is "
     + "dominated by the common pattern." }]),
P("A related measurement reinforced this. Comparing the pretraining corpus against the four "
  + "evaluation cohorts, we found no contamination — no evaluation sample appears in the "
  + "training pool, so the results are a genuine test of transfer. But coverage is poor: within "
  + "the corpus a spectrum's closest match has a correlation of 0.99, whereas for the "
  + "evaluation cohorts the closest available match is only 0.37–0.78. The corpus is narrow "
  + "rather than small. Adding more spectra of the same kind would not address this."),

// ---------------- 4 ----------------
H1("4. Data-quality auditing"),
P("Because one dataset proved to be confounded, we audited all four cohorts on the same "
  + "footing. Two tests were used: whether cases and controls were measured in separate "
  + "instrument sessions, and whether spectral regions containing no metabolite signal can "
  + "nevertheless predict the diagnosis — which would only be possible if a technical "
  + "difference tracks the clinical groups."),
table(
  ["Cohort", "Design balance", "Verdict"],
  [["Barth syndrome", "Interleaved", "Clean — passes both tests"],
   ["MTBLS563", "Partly blocked", "Usable, with a design caveat"],
   ["BrC-T2D (cancer)", "Partly blocked", "Usable, with a design caveat"],
   ["BrC-T2D (diabetes)", "Interleaved", "Ambiguous — see note"],
   ["MTBLS326", "Fully separated", "Not usable"]],
  [2700, 2700, 3960]),
P("", { after: 100 }),
Rich([{ t: "This has methodological value beyond our own project. ", b: true },
  { t: "The standard safeguard in this literature is a label-permutation test, and every one "
     + "of our datasets passes it — including the confounded one. That test cannot detect a "
     + "batch effect, because shuffling the labels destroys the batch structure and the "
     + "biological signal at the same time. A confound has to be attacked directly." }]),
P("The audit also surfaced a defect in our own processing. The intensity-normalisation step we "
  + "had adopted, which rescales each spectrum by its own largest peak, converts an absolute "
  + "intensity difference into a signal-to-noise ratio — and signal-to-noise is a property of "
  + "the measurement session, not the patient. On two cohorts this measurably increased the "
  + "amount of non-biological signal available to a classifier. We tested three alternative "
  + "normalisation schemes; the existing choice remains the best overall, but the caveat is now "
  + "documented and one cohort's result is flagged for re-checking."),

// ---------------- 5 ----------------
H1("5. Current direction: synthetic training data"),
P("If the limitation is that the corpus is too homogeneous, the natural response is to generate "
  + "spectra that are chemically realistic but more diverse than anything the public "
  + "repositories contain. Two approaches are under development."),
H2("5.1 Recombining windows of real spectra"),
P("The first method samples short spectral windows from different patients and stitches them "
  + "together with overlap. We built and evaluated three variants. All of them produce output "
  + "that looks entirely convincing to the eye — and this proved to be the trap. Measured "
  + "against the statistical structure of real data, the naive variant destroys the correlations "
  + "between different regions of the spectrum, while the variant that preserves those "
  + "correlations does so only by copying real spectra almost verbatim."),
Rich([{ t: "The trade-off is itself the finding. ", b: true },
  { t: "We measured that intensities at different points in an NMR spectrum remain correlated "
     + "across patients at essentially all separations — this is the chemical fact that all "
     + "signals from one molecule rise and fall together with its concentration. Consequently "
     + "any recombination that is genuinely novel at long range is also chemically wrong at "
     + "long range. Window-based recombination is useful as local augmentation, not as a "
     + "stand-alone generator." }]),
H2("5.2 Building spectra from known metabolite signatures"),
P("The second and more promising method constructs a spectrum as a weighted sum of the known "
  + "spectral signatures of individual metabolites, with the weights drawn from realistic "
  + "concentration distributions. This preserves chemical coherence by construction — all "
  + "signals belonging to one molecule necessarily scale together — while allowing us to set "
  + "the diversity of the synthetic population deliberately, which is precisely the property "
  + "the real corpus lacks."),
P("We have built the reference library: 43 serum metabolites drawn from GISSMO, a public "
  + "database of experimentally validated NMR signatures, simulated at our instrument's field "
  + "strength. The library is chemically validated — 19 of 21 metabolites with well-established "
  + "literature values match to within 0.03 ppm. Because a metabolite library contains only "
  + "small molecules, we additionally extracted the broad lipoprotein contribution empirically "
  + "from our own data; it accounts for 37.5% of the total signal."),
Rich([{ t: "Before generating anything, we imposed a gate: can this model reproduce a real "
     + "spectrum? At present it explains approximately 50% of the variance, against a target of "
     + "90%. ", b: true },
  { t: "The generator is therefore not yet fit for use, and we have deliberately not proceeded "
     + "to the later stages. The diagnosis is encouraging, however: adding the lipoprotein "
     + "component substantially improved the model's behaviour without raising this ceiling, "
     + "which isolates the remaining problem as one of peak alignment rather than missing "
     + "chemistry." }]),
P("The cause has been identified. Our spectra were aligned to each other using an internal "
  + "reference peak rather than to an absolute chemical standard, so the chemical-shift axis "
  + "carries an arbitrary offset of roughly 0.15–0.3 ppm. This is invisible in any analysis "
  + "confined to our own data — all previous results remain internally consistent — but it "
  + "prevents comparison against an external reference library. The correction requires the "
  + "original instrument parameter files, which are being retrieved."),

// ---------------- 6 ----------------
H1("6. Status and next steps"),
table(
  ["Question", "Status"],
  [["Does self-supervised pretraining beat classical ML?", "Answered: no"],
   ["Why not?", "Answered: the pretraining task is nearly trivial on this corpus"],
   ["Is the amount of data the limitation?", "No — the diversity of the data is"],
   ["Are the evaluation datasets sound?", "Audited; one excluded, two flagged"],
   ["Can synthetic data supply the missing diversity?", "In progress"]],
  [5400, 3960]),
P("", { after: 100 }),
P("Immediate priorities are, first, to recover the original instrument parameters, which will "
  + "resolve the alignment problem and simultaneously allow the data-quality audit to be "
  + "repeated against true acquisition timestamps rather than a proxy; and second, to extend "
  + "the spectral fitting to allow each metabolite's peak positions to shift slightly, as "
  + "established quantification software does. Should the synthetic-data route succeed, the "
  + "pretraining will be repeated on the enlarged corpus and the entire evaluation rerun."),
Rich([{ t: "A note on how this work has been conducted. ", b: true },
  { t: "Several of the findings above are retractions of our own earlier conclusions. Early in "
     + "the project we measured the run-to-run variability of the training procedure and found "
     + "it to be roughly twice what we had assumed; applying that threshold retrospectively "
     + "invalidated several results that had rested on single training runs. Every claim now "
     + "carries an explicit uncertainty, comparisons are made in a paired design wherever "
     + "possible, and results that do not survive are recorded as retracted rather than "
     + "quietly dropped. The negative result reported here is, in our view, considerably more "
     + "secure for it." }], { after: 0 }),

    ],
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("docs/Progress_Report_2026-09.docx", b);
  console.log("wrote docs/Progress_Report_2026-09.docx");
});
