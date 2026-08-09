const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, PageOrientation, TableOfContents, LevelFormat,
  Header, Footer, PageNumber, convertInchesToTwip,
} = require("docx");

const REPO = "/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics";
const FIG = path.join(REPO, "docs/figures");

// US Letter, portrait
const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1080; // 0.75"
const CONTENT_W = PAGE_W - 2 * MARGIN; // 10080 dxa

const C = {
  ink: "1A1A1A", muted: "5C5C5C", rule: "BFBFBF",
  accent: "1F4E79", head: "DCE6F1", zebra: "F4F6F8",
  good: "1B7F4F", bad: "A32020",
};

// ---------- helpers ----------
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120, line: opts.line ?? 276 },
  alignment: opts.align,
  indent: opts.indent,
  children: [new TextRun({
    text, size: opts.size ?? 21, color: opts.color ?? C.ink,
    bold: opts.bold, italics: opts.italics, font: "Calibri",
  })],
});

// rich paragraph from [text, {bold,italics,color}] pairs
const RP = (runs, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120, line: 276 },
  alignment: opts.align,
  indent: opts.indent,
  children: runs.map(([t, o = {}]) => new TextRun({
    text: t, size: o.size ?? opts.size ?? 21, color: o.color ?? C.ink,
    bold: o.bold, italics: o.italics, font: "Calibri",
  })),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 340, after: 160 },
  children: [new TextRun({ text, size: 30, bold: true, color: C.accent, font: "Calibri" })],
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 },
  children: [new TextRun({ text, size: 25, bold: true, color: C.accent, font: "Calibri" })],
});
const H3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text, size: 22, bold: true, color: C.ink, font: "Calibri" })],
});

const BULLET = (text, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  spacing: { after: 80, line: 276 },
  children: [new TextRun({ text, size: 21, color: C.ink, font: "Calibri" })],
});
const NUM = (text, level = 0) => new Paragraph({
  numbering: { reference: "nums", level },
  spacing: { after: 80, line: 276 },
  children: [new TextRun({ text, size: 21, color: C.ink, font: "Calibri" })],
});

const RULE = () => new Paragraph({
  spacing: { before: 60, after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.rule } },
  children: [new TextRun({ text: "", size: 2 })],
});

// Callout box (single-cell shaded table)
const CALLOUT = (label, lines, tone = "accent") => {
  const bg = tone === "bad" ? "FDECEC" : tone === "good" ? "EAF6EF" : "EEF3F9";
  const bar = tone === "bad" ? C.bad : tone === "good" ? C.good : C.accent;
  const kids = [
    new Paragraph({
      spacing: { after: 60 },
      children: [new TextRun({ text: label, bold: true, size: 20, color: bar, font: "Calibri", allCaps: true })],
    }),
    ...lines.map((l, i) => new Paragraph({
      spacing: { after: i === lines.length - 1 ? 0 : 60, line: 264 },
      children: [new TextRun({ text: l, size: 20, color: C.ink, font: "Calibri" })],
    })),
  ];
  return new Table({
    columnWidths: [CONTENT_W],
    width: { size: CONTENT_W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: bg },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: bg },
      right: { style: BorderStyle.SINGLE, size: 2, color: bg },
      left: { style: BorderStyle.SINGLE, size: 18, color: bar },
      insideHorizontal: { style: BorderStyle.NONE },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [new TableRow({
      children: [new TableCell({
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: bg, color: "auto" },
        margins: { top: 140, bottom: 140, left: 180, right: 160 },
        children: kids,
      })],
    })],
  });
};

// Data table. headers: [], rows: [[..]], widths sum to CONTENT_W
const TBL = (headers, rows, widths, opts = {}) => {
  const cell = (txt, o = {}) => new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    children: [new Paragraph({
      spacing: { after: 0, line: 252 },
      alignment: o.align,
      children: [new TextRun({
        text: String(txt), size: o.size ?? 18, bold: o.bold,
        color: o.color ?? C.ink, font: "Calibri",
      })],
    })],
  });
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(h, {
      w: widths[i], bold: true, fill: C.head,
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
    })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((v, ci) => {
      let color = C.ink, bold = false;
      const s = String(v);
      if (opts.boldFirstCol && ci === 0) bold = true;
      if (s.startsWith("**")) { bold = true; }
      const clean = s.replace(/\*\*/g, "");
      if (opts.colorize) {
        if (/^\+/.test(clean) && parseFloat(clean) > 0.0005) color = C.good;
        if (/^−|^-/.test(clean) && parseFloat(clean.replace("−", "-")) < -0.0005) color = C.bad;
      }
      return cell(clean, {
        w: widths[ci], bold, color,
        fill: ri % 2 === 1 ? C.zebra : undefined,
        align: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      });
    }),
  }));
  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: C.rule },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: C.rule },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "E2E2E2" },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...bodyRows],
  });
};

const CAPTION = (text) => new Paragraph({
  spacing: { before: 60, after: 220 },
  keepLines: true,
  children: [new TextRun({ text, size: 17, italics: true, color: C.muted, font: "Calibri" })],
});

// Figure scaled to fit content width
const FIGURE = (file, widthIn = 6.7) => {
  const p = path.join(FIG, file);
  if (!fs.existsSync(p)) return P(`[missing figure: ${file}]`, { italics: true, color: C.bad });
  const dim = require("child_process").execSync(
    `~/anaconda3/envs/NMR/bin/python -c "from PIL import Image; im=Image.open('${p}'); print(im.width, im.height)"`,
    { shell: "/bin/bash" }).toString().trim().split(/\s+/).map(Number);
  const w = widthIn * 96;
  const h = Math.round(w * dim[1] / dim[0]);
  return new Paragraph({
    spacing: { before: 100, after: 40 },
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width: Math.round(w), height: h } })],
  });
};

const SPACER = (n = 120) => new Paragraph({ spacing: { after: n }, children: [new TextRun({ text: "", size: 2 })] });

// ============================================================
const body = [];

// ---------- TITLE PAGE ----------
body.push(SPACER(1400));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: "NMR Metabolomics Foundation Model", size: 52, bold: true, color: C.accent, font: "Calibri" })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 360 },
  children: [new TextRun({ text: "Laboratory Record of Analyses and Decisions", size: 30, color: C.muted, font: "Calibri" })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 100 },
  border: { top: { style: BorderStyle.SINGLE, size: 8, color: C.accent } },
  children: [new TextRun({ text: "", size: 2 })],
}));
body.push(P("Self-supervised representation learning on 1D ¹H-NMR serum and plasma spectra, benchmarked against classical machine learning across five clinical classification targets.",
  { align: AlignmentType.CENTER, size: 22, color: C.muted, after: 400 }));

body.push(TBL(
  ["Record field", "Value"],
  [
    ["Period covered", "up to 29 July 2026"],
    ["Pretraining corpus", "9,670 human serum / plasma ¹H-NMR spectra (131,072 points each)"],
    ["Evaluation targets", "5 — Barth, MTBLS326, MTBLS563, BrC-T2D (cancer), BrC-T2D (diabetes)"],
    ["SSL families", "masked autoencoding, jigsaw, joint (masked + jigsaw)"],
    ["Data version of record", "v4 (water + EDTA suppression corrected, row min-max normalised)"],
    ["Companion markdown", "docs/SSL_vs_classical_analysis.md"],
    ["Status", "Backbone scaling axis exhausted; objective and head remain"],
  ],
  [3000, 7080], { boldFirstCol: true }));

body.push(SPACER(300));
body.push(CALLOUT("How to read this record", [
  "This is a chronological lab notebook, not a paper. It records what we believed at each point, what we did to test it, what the data said, and where a belief was overturned.",
  "Refuted hypotheses are kept in place rather than deleted — several of the most useful conclusions came from predictions that failed.",
  "Every number is reproducible from the scripts named in each section.",
]));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- TOC ----------
body.push(H1("Contents"));
body.push(P("In Word, right-click below and choose \u201cUpdate Field\u201d (or press F9) to populate. It refreshes automatically as new sections are added.",
  { italics: true, color: C.muted, size: 19, after: 200 }));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- 1. STARTING POINT ----------
body.push(H1("1. Where we started"));

body.push(P("The project set out to test a straightforward proposition: that self-supervised pretraining on a large unlabelled corpus of ¹H-NMR serum and plasma spectra would produce a representation that transfers to small clinical classification problems better than classical machine learning applied directly to the spectra."));

body.push(P("The ingredients were in place at the start of this record:"));
body.push(BULLET("A pretraining corpus of 9,670 spectra, each 131,072 points, drawn from MetaboLights and Metabolomics Workbench, all human serum or plasma."));
body.push(BULLET("Three self-supervised objectives already implemented and trained: masked autoencoding (reconstruct hidden patches), jigsaw (recover shuffled bin order), and a joint model combining both."));
body.push(BULLET("Four downstream evaluation datasets, later five targets once BrC-T2D was split into cancer and diabetes labels."));
body.push(BULLET("A classical baseline: integrated absolute area in equal-width bins, fed to logistic regression, SVM-RBF and XGBoost."));

body.push(H2("1.1 The evaluation protocol"));
body.push(P("Sample sizes are small, so the protocol was fixed per dataset and never varied afterwards. This matters: it is what makes numbers comparable across the whole record."));
body.push(TBL(
  ["Target", "n", "Classes", "Cross-validation"],
  [
    ["Barth syndrome", "37", "2 (Case / Control)", "Leave-one-out"],
    ["MTBLS326", "42", "2 (IP3R expressing / control)", "Leave-one-out"],
    ["MTBLS563", "113", "3 (bacterial / viral / control)", "Stratified 10-fold"],
    ["BrC-T2D cancer", "78", "2 (Cancer / No cancer)", "Stratified 10-fold"],
    ["BrC-T2D diabetes", "78", "2 (Diabetes / No diabetes)", "Stratified 10-fold"],
  ],
  [3400, 900, 3180, 2600]));
body.push(CAPTION("Table 1. Evaluation targets and protocols. Barth excludes 3 pooled-QC rows; MTBLS563 excludes 29 rows labelled 'unknown'. Balanced accuracy is the primary metric throughout, because several targets are class-imbalanced."));

body.push(RULE());

// ---------- 2. DATA INTEGRITY ----------
body.push(H1("2. First detour: the data was not what we thought"));

body.push(P("Before any modelling question could be answered, a routine check turned into the largest single piece of work in this record. The question asked was simply whether the water and EDTA suppression steps had actually been applied to every spectrum."));

body.push(H2("2.1 Why suppression matters here"));
body.push(P("Two spectral regions carry artefacts far larger than any metabolite signal. The residual water peak (points 62,500–68,000) and, in some EDTA-anticoagulated plasma samples, the EDTA peak. Both are orders of magnitude taller than real peaks."));
body.push(RP([
  ["The pipeline applies row-wise min-max normalisation. If an artefact peak survives into that step it becomes the row maximum, and every genuine metabolite peak is compressed toward zero. ", {}],
  ["A single unsuppressed artefact therefore corrupts the entire spectrum, not just its own window.", { bold: true }],
]));

body.push(H2("2.2 How we checked, and a false start"));
body.push(P("The first validator asked \"is there a peak here above 3× the local noise?\" and reported that 73.5% of the training corpus had water-region peaks. That result was alarming enough to inspect the spectra directly before reporting it — which showed the test itself was wrong, in two ways:"));
body.push(NUM("A correctly suppressed region is flat, but not exactly zero. After row min-max normalisation a hard-zeroed window maps to a small positive constant, so testing against 0.0 fails."));
body.push(NUM("The EDTA window legitimately contains real, small metabolite peaks in most samples. \"A peak is present\" was never evidence that suppression had been missed."));

body.push(CALLOUT("Method decision", [
  "The correct test for \"was a hard mask applied\" is FLATNESS, not peak absence: peak-to-peak range within the window, relative to the row's own dynamic range (ptp < 1e-6 × row_ptp).",
  "Peak-above-noise was demoted to a secondary diagnostic. This distinction is what made every subsequent suppression audit trustworthy.",
]));

body.push(H2("2.3 What the corrected audit found"));
body.push(P("With the flatness test, the picture was serious and quite different from the first pass:"));
body.push(BULLET("Water suppression was missing in 69.7% of the pretraining corpus."));
body.push(BULLET("It was missing in 100% of the Barth evaluation data — the evaluation scripts pointed at the raw pre-pipeline file, not the pipeline's output."));
body.push(BULLET("EDTA suppression had effectively never fired: 1 row out of 9,670."));

body.push(H2("2.4 Getting EDTA suppression right took three attempts"));
body.push(P("This is worth recording in full because the first two approaches were plausible and both wrong."));
body.push(TBL(
  ["Attempt", "Criterion", "Why it failed"],
  [
    ["1. Dominance ratio", "Peak height relative to others inside the EDTA window", "Confirmed-EDTA and confirmed-Heparin rows had statistically indistinguishable ratios (median ≈1.3 both). Fired on 1 of 9,670 rows."],
    ["2. Local prominence SNR", "Peak prominence above local noise", "Over-fired: suppressed 32 of 33 non-EDTA Heparin rows. Direct visual inspection caught it removing a real 4-line J-coupling multiplet at ~10% of that row's peak scale."],
    ["3. Magnitude vs row max", "Peak height ÷ row's own maximum elsewhere; suppress if ≥ 0.5", "Adopted. Metadata-free, uniform across all datasets, and directly encodes the actual requirement."],
  ],
  [2200, 3100, 4780], { boldFirstCol: true }));
body.push(CAPTION("Table 2. Three EDTA-suppression criteria. The decisive correction came from inspecting a plot: attempt 2 was removing genuine chemistry."));

body.push(CALLOUT("Why criterion 3 is the right question", [
  "The goal was never \"identify EDTA chemically\". It was \"remove peaks that corrupt normalisation\".",
  "A peak comparable in height to other real peaks is harmless, whatever molecule it belongs to. A peak of a totally different magnitude is the problem.",
  "Comparing a candidate peak to the row's own maximum elsewhere expresses exactly that, needs no metadata, and applies identically to every dataset.",
], "good"));

body.push(H2("2.5 A late bug in the same code"));
body.push(P("A follow-up audit found the boundary-cutoff in the accepted detector could exceed the candidate peak's own height when the window edge was noisy. The boundary search then never advanced, width came out zero, and a peak already confirmed dominant went unsuppressed. Capping the cutoff at the baseline-to-peak midpoint fixed it, producing the v4 datasets."));
body.push(TBL(
  ["Dataset", "Missed suppressions", "Of which corrupted normalisation"],
  [
    ["Pretraining corpus (9,670)", "26 (0.3%)", "4 rows (0.04%)"],
    ["Barth (40)", "0", "0"],
    ["MTBLS326 (42)", "5 (11.9%)", "5 (11.9%) — v3 results invalidated"],
    ["MTBLS563 (142)", "48 (33.8%)", "2 (1.4%)"],
    ["BrC-T2D (78)", "12 (15.4%)", "1 (1.3%)"],
  ],
  [3600, 3200, 3280], { boldFirstCol: true }));
body.push(CAPTION("Table 3. Impact of the cutoff bug. A missed suppression only corrupts normalisation when the unsuppressed peak was the row maximum — that distinction is what separates the two columns, and it is why the corpus impact was negligible while MTBLS326's was not."));

body.push(CALLOUT("Decision: do not retrain the SSL models for this", [
  "Only 4 of 9,670 corpus rows had their normalisation scale changed — about 0.004% of all data points, confined to a 2,000-point window.",
  "Retraining three models for that was not justified. The evaluation datasets were rebuilt and the benchmarks re-run instead.",
  "MTBLS326 was the exception that mattered: 11.9% of its spectra were genuinely corrupted, so its earlier results were discarded.",
]));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- 3. HEADLINE ----------
body.push(H1("3. The headline result, and the question it raised"));

body.push(P("With clean v4 data and freshly retrained SSL backbones, the benchmark gave an unambiguous and initially discouraging answer."));
body.push(FIGURE("fig1_balanced_accuracy.png", 6.6));
body.push(CAPTION("Figure 1. Balanced accuracy by model family, best fine-tuning mode per family, across all five targets. Classical logistic regression (blue) leads everywhere."));

body.push(TBL(
  ["Target", "Classical LR", "Masked", "Jigsaw", "Joint", "Gap"],
  [
    ["Barth", "**0.705", "0.691", "0.677", "0.649", "0.014"],
    ["MTBLS326", "**1.000", "0.981", "0.874", "0.930", "0.019"],
    ["MTBLS563", "**0.721", "0.558", "0.550", "0.500", "0.163"],
    ["BrC-T2D cancer", "**0.937", "0.796", "0.782", "0.757", "0.141"],
    ["BrC-T2D diabetes", "**0.829", "0.653", "0.620", "0.624", "0.176"],
  ],
  [2700, 1700, 1420, 1420, 1420, 1420], { boldFirstCol: true }));
body.push(CAPTION("Table 4. Headline v4 benchmark. Consistent family ordering: masked > jigsaw ≈ joint. Barth and MTBLS326 gaps are within one sample and should be read as ties; leave-one-out gives no fold variance, hence no error bars."));

body.push(P("A linear model on binned areas beating three pretrained transformers by up to 18 points needed an explanation, not just a report. The rest of this record is that investigation."));

body.push(H2("3.1 The one asymmetry that made the whole thing tractable"));
body.push(P("Inspecting the two pipelines revealed that the SSL classifier head is:"));
body.push(P("pooled embedding → LayerNorm → Dropout → Linear(d, n_classes)", { align: AlignmentType.CENTER, italics: true, color: C.accent, after: 140 }));
body.push(RP([
  ["That is a ", {}], ["linear classifier", { bold: true }],
  [", fitted by Adam for ~50 epochs with early stopping. The classical track is ", {}],
  ["also", { italics: true }],
  [" a linear classifier, fitted by L-BFGS to convergence with an explicit L2 penalty. Both sides are linear, which means the two tracks can be swapped one component at a time — and every subsequent experiment in this record exploits that.", {}],
]));

body.push(RULE());

// ---------- 4. DECOMPOSITION ----------
body.push(H1("4. Decomposing the gap"));

body.push(P("Two very different explanations predict the same headline. Either the SSL embedding discards the discriminative signal (a representation problem), or it retains the signal but the head cannot extract it from ~70 training samples (a head problem). They are distinguished by running the same logistic regression on the frozen SSL embedding, over the same folds."));

body.push(CALLOUT("Validation gate applied before trusting any of this", [
  "The probe harness was first required to reproduce the official summary.csv logistic-regression balanced accuracy to six decimal places on all five targets: 0.936842, 0.828877, 0.720785, 1.000000, 0.704969.",
  "Without that check, none of the comparisons below would be interpretable.",
], "good"));

body.push(FIGURE("fig6_logreg_advantage_probe.png", 6.5));
body.push(CAPTION("Figure 2. Left: balanced accuracy versus spectral resolution for the classical pipeline, with the SSL embedding and the reported SSL head overlaid, and the backbone's own patch resolution marked. Right: the same quantities against the label-permutation null. Green versus red isolates the classifier; green versus blue isolates the representation."));

body.push(TBL(
  ["Target", "SSL head", "LR on same embedding", "LR @1024 bins", "Head deficit", "Representation ceiling"],
  [
    ["BrC-T2D cancer", "0.796", "0.833", "0.937", "0.037", "**0.104"],
    ["BrC-T2D diabetes", "0.653", "**0.810", "0.829", "**0.157", "0.019"],
    ["MTBLS563", "0.558", "0.607", "0.721", "0.049", "**0.114"],
    ["Barth", "0.691", "**0.770", "0.705", "**0.079", "−0.065"],
    ["MTBLS326", "0.981", "0.944", "1.000", "−0.037", "0.056"],
  ],
  [2400, 1420, 2100, 1600, 1280, 1280], { boldFirstCol: true }));
body.push(CAPTION("Table 5. Gap decomposition. Diabetes is overwhelmingly a head problem (89% of the gap). Cancer and MTBLS563 are representation problems. On Barth the SSL embedding is genuinely better than binned features (0.770 vs 0.705) — the head was discarding a real advantage."));

body.push(H2("4.1 Controls: ruling out the boring explanations"));
body.push(P("Two checks were run before accepting that the classical result was real."));
body.push(TBL(
  ["Target", "Observed", "Null mean", "Null 95th pct", "Null max", "p"],
  [
    ["BrC-T2D cancer", "0.937", "0.494", "0.616", "0.717", "0.005"],
    ["BrC-T2D diabetes", "0.829", "0.498", "0.619", "0.674", "0.005"],
    ["MTBLS563", "0.721", "0.332", "0.417", "0.526", "0.005"],
    ["MTBLS326", "1.000", "0.485", "0.634", "0.737", "0.005"],
    ["Barth", "0.705", "0.498", "0.655", "**0.792", "0.020"],
  ],
  [2500, 1500, 1520, 1620, 1420, 1520], { boldFirstCol: true }));
body.push(CAPTION("Table 6. Label-permutation nulls, 200 draws under the identical protocol. p = 0.005 is the floor at 200 permutations. Barth is marginal — its null reaches 0.792, above the observed 0.705 — so the Barth classical result is weak evidence and is treated as such throughout."));

body.push(P("A single global intensity scalar (total absolute area) reaches only 0.616 / 0.593 / 0.337 on cancer / diabetes / MTBLS563, far below the full model. The signal is genuinely local spectral structure, not a dilution or concentration confound."));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- 5. EXPERIMENT LOG ----------
body.push(H1("5. Experiment log"));
body.push(P("From this point the work became a sequence of single-variable experiments. Each is recorded with the prediction that motivated it and the outcome, including the three that failed."));

body.push(H2("5.1 Experiment #2 — is the head underfitting?"));
body.push(P("Prediction: if the masking head is poorly fitted, a converged L2 logistic regression on identical frozen features should beat it. Everything else — backbone, pooling, folds — held constant, so any difference is attributable purely to how the linear map is fitted."));
body.push(FIGURE("fig7_linear_probe_vs_head.png", 6.5));
body.push(CAPTION("Figure 3. Δ balanced accuracy, logistic-regression probe minus the trained MLP head, on identical frozen features."));

body.push(TBL(
  ["Target", "Masking", "Jigsaw", "Joint"],
  [
    ["Barth", "**+0.115", "+0.093", "+0.022"],
    ["BrC-T2D cancer", "**+0.103", "0.000", "+0.013"],
    ["BrC-T2D diabetes", "**+0.156", "−0.026", "+0.019"],
    ["MTBLS326", "**+0.148", "0.000", "0.000"],
    ["MTBLS563", "**+0.077", "−0.024", "−0.006"],
    ["**Mean", "**+0.120", "+0.009", "+0.009"],
  ],
  [3480, 2200, 2200, 2200], { boldFirstCol: true, colorize: true }));
body.push(CAPTION("Table 7. The masking head is underfit on 5 of 5 targets, by +0.077 to +0.156. Jigsaw and joint heads are fine — and their fine-tuned modes actually beat a frozen probe, so they were left alone. Extending the test to all three families is what revealed this is family-specific rather than a general 'sklearn beats Adam' effect."));

body.push(CALLOUT("Outcome and decision", [
  "Confirmed for masking only: mean +0.120 on frozen features, +0.057 even against the best fine-tuned configuration.",
  "Adopted as a first-class evaluator (code/evaluation/ssl_linear_probe_eval.py), reported as an ADDITIONAL model rather than replacing the fine-tuned heads — on MTBLS326 the fine-tuned head genuinely wins (0.981 vs 0.963), so a silent swap would have lost accuracy there.",
  "Negative side-finding: selecting the L2 strength by nested CV made things WORSE (gained in 4 of 15 cells, lost in 7). At n=78 with 10 folds the inner CV picks C from ~14 samples, adding variance without reducing bias. Fixed C=1 is the better choice.",
], "good"));

body.push(H2("5.2 Experiment #3 — does pretraining contribute anything?"));
body.push(RP([
  ["This began with an error worth recording. An existing \"Xavier ablation\" appeared to show pretraining contributing nothing. Checking the flag revealed it reinitialises ", {}],
  ["only the unfrozen layers", { bold: true }],
  [" — so in frozen mode nothing is reinitialised at all, and the two arms are byte-identical (verified: 0.526398 in both). The patch embedding and positional encoding always stay pretrained. It never tested \"pretrained versus random\".", {}],
]));
body.push(P("Read correctly, mode by mode rather than as a maximum over four noisy modes, pretraining helps and helps more with depth — MTBLS326 masked at +3 layers: 0.948 pretrained versus 0.648 reinitialised. The earlier conclusion was retracted and a genuine control was built: same architecture, no pretrained weights loaded anywhere, with the head held fixed at a converged probe so head quality cannot contaminate the comparison."));

body.push(FIGURE("fig8_pretraining_gain.png", 6.5));
body.push(CAPTION("Figure 4. Δ balanced accuracy, pretrained minus true random initialisation, head held fixed in both arms."));

body.push(TBL(
  ["Target", "Masking", "Jigsaw", "Joint"],
  [
    ["Barth", "**+0.252", "+0.087", "+0.202"],
    ["MTBLS326", "**+0.052", "+0.015", "−0.100"],
    ["BrC-T2D cancer", "**+0.063", "−0.028", "−0.077"],
    ["BrC-T2D diabetes", "**+0.171", "−0.067", "−0.023"],
    ["MTBLS563", "**+0.047", "−0.065", "−0.126"],
    ["**Mean", "**+0.117", "−0.011", "−0.025"],
  ],
  [3480, 2200, 2200, 2200], { boldFirstCol: true, colorize: true }));
body.push(CAPTION("Table 8. Masked pretraining works — positive on 5 of 5 targets. Jigsaw pretraining is worthless and joint pretraining is actively harmful: a random joint backbone scores 0.846 on BrC-T2D cancer and 0.911 on MTBLS326 versus 0.769 and 0.811 pretrained."));

body.push(CALLOUT("Interpretation caveat", [
  "A random transformer is a legitimately strong random-projection baseline, comparable to random-feature kernel methods.",
  "So \"random wins\" means the objective adds nothing OVER a random projection — not that the architecture is useless.",
  "Consequence for the roadmap: concentrate on the masking objective; jigsaw and joint pretraining need rethinking or dropping.",
]));

body.push(H2("5.3 Experiment #4 — the patch-resolution hypothesis, refuted"));
body.push(P("Prediction, and the reasoning behind it: the backbone tokenises 131,072 points into 131072/patch_size tokens. At patch_size = 1024 that is 128 positions. The frozen embedding scored right at what logistic regression achieves on 128 bins, while 1024 bins scored far higher, and the bin sweep showed resolution is exactly what buys accuracy (16→1024 bins: 0.836→0.937 on cancer). Shrinking the patch should therefore lift the ceiling."));

body.push(FIGURE("fig9_patch_size_and_pooling.png", 6.8));
body.push(CAPTION("Figure 5. Left: smaller patches made things worse, on every target — the hypothesis is refuted. Right: the win the same experiment did find."));

body.push(TBL(
  ["Target", "patch 1024", "patch 256", "patch 128"],
  [
    ["Barth", "**0.806", "0.598", "0.655"],
    ["MTBLS326", "**1.000", "0.907", "0.911"],
    ["BrC-T2D cancer", "**0.859", "0.832", "0.768"],
    ["BrC-T2D diabetes", "0.780", "**0.783", "0.738"],
    ["MTBLS563", "**0.618", "0.581", "0.607"],
    ["**Mean Δ vs 1024", "—", "**−0.072", "**−0.077"],
  ],
  [3480, 2200, 2200, 2200], { boldFirstCol: true }));
body.push(CAPTION("Table 9. Patch-size comparison, flatten pooling, all arms read at their true nhead = 4. Zero wins out of five for either smaller patch."));

body.push(CALLOUT("Why the prediction failed — and this is the useful part", [
  "Validation reconstruction loss FELL as patches shrank: 9.26e-5 → 5.56e-5 → 4.36e-5.",
  "A masked 128-point patch is largely interpolable from its immediate neighbours. Shrinking the patch made the pretext task EASIER, not more informative — the model can solve it by local smoothing without learning metabolite structure.",
  "The reasoning error was considering only what the encoder COULD represent, and not whether the task would still force it to learn anything.",
  "Confound recorded: the small-patch models also have ~3× fewer parameters, since the patch embedding and reconstruction head scale with patch size.",
], "bad"));

body.push(H2("5.4 The win that experiment #4 did produce: pooling"));
body.push(P("The masking classifier pools tokens with a mean, which averages away where in the spectrum each token came from. Chemical-shift position is precisely the discriminative information in NMR. Replacing the mean with a position-preserving flattened embedding helped on all five targets."));
body.push(TBL(
  ["Target", "mean-pool", "flatten", "Gain"],
  [
    ["Barth", "0.677", "**0.806", "+0.129"],
    ["BrC-T2D diabetes", "0.687", "**0.780", "+0.093"],
    ["BrC-T2D cancer", "0.782", "**0.859", "+0.077"],
    ["MTBLS326", "0.948", "**1.000", "+0.052"],
    ["MTBLS563", "0.588", "**0.618", "+0.030"],
  ],
  [3480, 2200, 2200, 2200], { boldFirstCol: true, colorize: true }));
body.push(CAPTION("Table 10. Position-preserving pooling versus mean pooling, patch 1024. This is a pooling fix, not a resolution fix — the tokens always carried the position information; the head was throwing it away."));

body.push(H2("5.5 How much position is needed? The pooling sweep"));
body.push(P("Mean-pool and flatten are the two extremes of a spectrum, and they differ in two ways at once: retained positional detail, and feature dimension (128 versus 16,384 — which matters at n = 37–113). Regional pooling separates them: split the tokens into G contiguous groups, mean-pool within each, concatenate. G = 1 is mean-pool; G = number of tokens is flatten."));
body.push(FIGURE("fig10_pooling_sweep.png", 6.8));
body.push(CAPTION("Figure 6. Regional pooling sweep. On 4 of 5 targets the optimum lies strictly between the two extremes. G = 16 matches flatten's accuracy at 2,048 features instead of 16,384."));

body.push(TBL(
  ["Pooling groups G", "1", "2", "4", "8", "16", "32", "64", "128"],
  [
    ["Feature dimension", "128", "256", "512", "1024", "2048", "4096", "8192", "16384"],
    ["**Mean bal. acc.", "0.793", "0.795", "0.799", "0.798", "**0.816", "**0.816", "0.805", "0.813"],
  ],
  [2520, 945, 945, 945, 945, 945, 945, 945, 945], { boldFirstCol: true }));
body.push(CAPTION("Table 11. Mean balanced accuracy across the five targets by group count. G = 1 is mean-pool, G = 128 is flatten. Caveat: G was chosen by inspecting these same numbers, so the exact optimum is indicative — an unbiased estimate requires selecting G by nested cross-validation inside each training fold."));

// ---------- 6. SCALING ----------
body.push(H1("6. Scaling the backbone: four failures"));

body.push(P("With patch 128 and 256 already having failed, two further backbones were pretrained on the v4 corpus: patch 2048 (64 tokens, 5.42M parameters) and a capacity arm holding patch 1024 fixed while raising d_model 128→256, layers 3→6, feed-forward 256→512 (5.17M parameters). Both early-stopped normally."));

body.push(FIGURE("fig11_backbone_scaling.png", 6.8));
body.push(CAPTION("Figure 7. Left: no new backbone beats the original small one, at either pooling. Right: reconstruction loss does not predict downstream utility."));

body.push(TBL(
  ["Backbone", "Params", "Recon loss", "mean-pool", "flatten"],
  [
    ["patch 128", "0.63M", "4.36e-5", "0.745", "0.778"],
    ["patch 256", "0.66M", "5.56e-5", "0.755", "0.779"],
    ["**patch 1024 (original)", "**1.89M", "9.26e-5", "0.802", "**0.888"],
    ["patch 2048", "5.42M", "1.020e-4", "0.818", "0.840"],
    ["patch 1024, d256 L6", "5.17M", "**3.95e-5", "0.814", "0.826"],
    ["Classical logistic regression", "—", "—", "—", "0.881"],
  ],
  [3600, 1400, 1700, 1690, 1690], { boldFirstCol: true }));
body.push(CAPTION("Table 12. Held-out mean balanced accuracy (Barth, MTBLS326, BrC-T2D cancer). Letting each backbone choose its own pooling on the selection subset only gives: original 0.849, patch 2048 0.824, d256 L6 0.817 — the original wins under every pooling."));

body.push(CALLOUT("Two conclusions — as written at the time", [
  "1. Patch 1024 is near-optimal and this is NOT a capacity limit. Patch 2048 carries 2.9× the baseline's parameters and still loses. Four attempts — 128, 256, 2048, and 2.7× capacity — all failed. The backbone axis is exhausted.",
  "2. Capacity compensates for bad pooling but does not beat fixing it. Under mean-pool, accuracy rises perfectly monotonically with parameters (Spearman = +1.00, p < 0.01). Yet the 1.89M model with flatten (0.888) beats every 5M model. The bottleneck was information destroyed by pooling, not model capacity.",
], "bad"));

body.push(CALLOUT("Correction — conclusion 1 does not survive section 7", [
  "The 0.888 reference used throughout this table is a v3-pretrained checkpoint, while all four comparison backbones were pretrained on v4. Section 7 shows that the corpus version alone is worth 0.069, so this table compares two things at once.",
  "Against a proper v4 baseline (0.820), patch 2048 is +0.020 and d256 L6 is +0.006 — but section 7 also establishes a noise floor of 0.020 for a claim of this kind. The honest statement is therefore that patch 1024, patch 2048 and d256 L6 are INDISTINGUISHABLE on one run each.",
  "\"The backbone axis is exhausted\" is not supported by the data. What survives is the narrower claim that shrinking the patch below 1024 hurts (patch 128 −0.042, patch 256 −0.034, both measured v4-against-v4), and conclusion 2, which is a paired within-checkpoint comparison and therefore immune to run-to-run noise.",
], "bad"));

body.push(H2("6.1 Reconstruction loss is not a proxy for downstream utility"));
body.push(P("Across the five backbones, Spearman correlation between validation reconstruction loss and held-out accuracy is +0.60 for flatten and +0.40 for mean-pool — if anything, worse reconstruction goes with better transfer (n = 5, not significant, but the sign is consistent)."));
body.push(P("The starkest case: the d256 L6 model reconstructs 2.3× better than the baseline (3.95e-5 versus 9.26e-5) and transfers worse (0.826 versus 0.888). The same disconnect appeared within a single run, where patch 128's final checkpoint had 2.4% better reconstruction and scored lower downstream on 4 of 5 targets."));

body.push(CALLOUT("Operational rule adopted", [
  "Never select checkpoints, architectures, or epochs on reconstruction loss.",
  "Selection must use a downstream signal — but on a PRE-COMMITTED subset, never on the datasets used for reporting.",
], "good"));

body.push(H2("6.2 On selection bias"));
body.push(P("Raised during review and adopted as standing practice. Comparing many configurations by their downstream cross-validation scores and then quoting the winner inflates the reported number, even though no label information crosses folds — the backbone is pretrained on a disjoint corpus and the probe is fitted on training folds only. It is not leakage, but it is model selection on the reporting set."));
body.push(BULLET("From section 6 onward, configuration choices (pooling G, backbone) are made on a designated selection subset: MTBLS563 + BrC-T2D diabetes."));
body.push(BULLET("Results are reported on the held-out three: Barth, MTBLS326, BrC-T2D cancer."));
body.push(BULLET("Throughout this record, the comparative signs are more trustworthy than the absolute values, because each holds consistently across five independent datasets."));

body.push(RULE());

// ---------- 7. EXPERIMENT #7 ----------
body.push(H1("7. Experiment #7: the pretext objective, and a measurement problem"));

body.push(P("Sections 5 and 6 pointed away from architecture and toward the pretraining objective. Section 5.3 had found something specific: shrinking the patch made reconstruction EASIER while transfer got WORSE, because a lone masked patch bracketed by intact neighbours is largely interpolable — the model can win by local smoothing without learning spectral structure. Two changes follow from that diagnosis, and they are orthogonal, so they were run as a 2×2 factorial."));

body.push(TBL(
  ["Arm", "What is hidden from the encoder", "What the loss is computed over"],
  [
    ["D — reference", "Patches chosen independently at random — scattered singletons", "All 128 patches"],
    ["A — block", "The same count, drawn as contiguous 8-patch spans (8,192 points, ≈0.75 ppm)", "All 128 patches"],
    ["B — peak", "Scattered singletons, as D", "Only the top 25% of patches by magnitude"],
    ["C — both", "Contiguous 8-patch spans", "Only the top 25% by magnitude"],
  ],
  [2100, 4200, 3780], { boldFirstCol: true }));
body.push(CAPTION("Table 16. The four arms. Everything else is held fixed: patch 1024, d_model 128, 3 layers, 4 heads, 1.89M parameters, batch 32, mask ratio 0.20–0.60, identical v4 corpus, patience 200. All four early-stopped normally. Arm D exists because every earlier masking baseline was pretrained on v3 — without a v4 default run the factorial would confound the objective change with the corpus version."));

body.push(P("The peak-weighting function was ported verbatim from the joint family's existing implementation and verified before any GPU time was spent (code/tests/verify_top_peak_loss.py): elementwise agreement with the original on synthetic and real spectra, exact-k selection, per-spectrum rather than per-batch thresholding, bit-identical to the old uniform loss at fraction 1.0, and gradient flow. That verification also produced a caveat worth recording — at patch 1024 each patch spans 1,024 points, so nearly every patch contains some signal and the top 25% of patches hold only ~59% of total absolute intensity (2.4× enrichment). Restricting to 25% therefore discards ~40% of the signal, which is a more aggressive intervention than the \"mostly flat baseline\" framing suggests."));

body.push(H2("7.1 The mechanism fired; the prescription failed"));

body.push(P("Arms D and A optimise the same uniform loss on the same data, so their validation losses are directly comparable. Block masking raised validation loss from 7.10e-5 to 1.00e-4 — 41% harder — and pushed the best epoch out by 450. The task genuinely became harder, exactly as intended. It still lost."));

body.push(FIGURE("fig12_exp7_factorial.png", 6.8));
body.push(CAPTION("Figure 8. Left: no arm reaches classical logistic regression. Middle: the factorial read as main effects, with the baseline uncertainty from section 7.2 drawn as a band. Right: the confound the experiment actually uncovered."));

body.push(TBL(
  ["Effect (flatten pooling)", "Held-out", "Selection", "All five"],
  [
    ["Block masking", "**−0.030", "**−0.034", "−0.032"],
    ["Peak weighting", "+0.011", "+0.007", "+0.009"],
    ["Interaction", "−0.006", "−0.001", "−0.004"],
  ],
  [4200, 2000, 2000, 1880], { boldFirstCol: true }));
body.push(CAPTION("Table 17. Factorial main effects, each averaged over both levels of the other factor. Held-out means: D 0.820, A 0.793, B 0.834, C 0.801, against classical 0.881."));

body.push(CALLOUT("What experiment #7 established about block masking", [
  "Block masking hurts, consistently — negative on both splits and both pooling schemes, and Barth collapses from 0.748 to 0.562.",
  "So section 5.3's diagnosis was correct as a description and wrong as a prescription. Making the pretext task harder is not sufficient. A plausible reading: 0.75 ppm is wide enough that the masked content is genuinely unrecoverable rather than merely non-trivial, so the model learns to predict a conditional mean and loses the sharp local detail the probe reads.",
  "Do not pursue block masking further.",
], "bad"));

body.push(H2("7.2 The real finding: how much does a single run mean?"));

body.push(P("Arm D existed only to remove a confound. It found one far larger than anything the experiment was designed to measure. Comparing it against the v3-pretrained checkpoint that every earlier number in this record was measured against — same architecture, same objective, same hyperparameters, with the configuration block in force on 25 July verified byte-identical to arm D's defaults — the v4 run scored 0.069 LOWER on the held-out mean, down on four of five targets by a consistent 0.057–0.078."));

body.push(P("That admits two readings: either the v4 corpus is worse for pretraining, or run-to-run variance at these sample sizes is around 0.07. Three further runs settled it."));

body.push(FIGURE("fig13_exp7_replicates.png", 6.9));
body.push(CAPTION("Figure 9. Left: three independent runs of one configuration, per target — the within-arm scatter is large, and the v3 reference sits at or above every v4 draw on all three held-out targets. Middle: peak weighting judged without a corpus confound. Right: every effect this record has claimed, against the measured noise floor."));

body.push(TBL(
  ["Target", "Run 1", "Run 2", "Run 3", "sd", "v3 ref"],
  [
    ["Barth", "0.7484", "0.6988", "0.6770", "0.037", "0.8059"],
    ["MTBLS326", "0.9296", "0.9630", "0.9111", "0.026", "1.0000"],
    ["BrC-T2D cancer", "0.7816", "0.8079", "0.8592", "0.040", "0.8592"],
    ["MTBLS563", "0.6283", "0.5505", "0.6331", "0.046", "0.6176"],
    ["BrC-T2D diabetes", "0.7052", "0.7701", "0.7052", "0.037", "0.7654"],
    ["**Held-out mean", "**0.8199", "**0.8232", "**0.8158", "**0.0037", "**0.8884"],
  ],
  [2700, 1600, 1600, 1600, 1200, 1380], { boldFirstCol: true }));
body.push(CAPTION("Table 18. Three independent runs of the identical v4 configuration (unseeded, seed 101, seed 202). The v3 reference is above the entire v4 cluster, so the corpus gap is real."));

body.push(CALLOUT("The 0.0037 is not precision — read the noise floor correctly", [
  "Per-target standard deviation averages 0.035, so a three-target mean should scatter by about 0.035/√3 = 0.020 if the targets were independent. The observed 0.0037 is 5.4× tighter only because Barth falls while cancer rises across the three draws (r = −0.92) and the errors cancel inside the mean.",
  "With three draws that cancellation is luck, not a property to rely on.",
  "Use 0.020 as the floor for a held-out-mean claim, and roughly 0.035 for any single-target claim.",
], "bad"));

body.push(H2("7.3 Peak weighting also fails, once the corpus is matched"));

body.push(P("Arm B's +0.011 was measured against arm D — a v4 checkpoint the corpus had already depressed by 0.069. The clean comparison trains a peak-weighted arm on v3 and reads it against the v3 baseline."));

body.push(TBL(
  ["Target", "Classical", "v3 baseline", "v3 + top-25% (r1)", "(r2)"],
  [
    ["Barth", "0.705", "**0.8059", "0.7484", "0.6910"],
    ["MTBLS326", "1.000", "**1.0000", "0.9630", "0.9630"],
    ["BrC-T2D cancer", "0.937", "0.8592", "0.8461", "0.8842"],
    ["MTBLS563", "0.721", "**0.6176", "0.5928", "0.5949"],
    ["BrC-T2D diabetes", "0.829", "**0.7654", "0.6237", "0.6237"],
    ["**Held-out mean", "0.8807", "**0.8884", "0.8525", "0.8461"],
  ],
  [2700, 1800, 2000, 2200, 1380], { boldFirstCol: true }));
body.push(CAPTION("Table 19. Peak weighting on the matched v3 corpus. −0.039 on the held-out mean (2.0× the noise floor) and −0.083 on the selection subset, driven by a −0.142 collapse on BrC-T2D diabetes."));

body.push(CALLOUT("Experiment #7 is negative on both factors", [
  "Block masking: −0.030, consistent in sign across both splits and both pooling schemes.",
  "Peak weighting: appeared to gain +0.011 against an unmatched reference, and loses 0.039 against a matched one.",
  "The apparent positive was an artifact of the corpus confound. This is why matched references matter more than large effect sizes.",
], "bad"));

body.push(H2("7.4 A correction, and then a correction to the correction"));

body.push(P("A --seed flag was added specifically to make these replicates interpretable, and it was signed off on the basis that two seeded runs produced a bit-identical state dictionary — a verification performed on CPU. The v3 peak arm was then launched twice with the same seed, and the two runs appeared not to match, which was written up here as evidence that seeding is insufficient on GPU. That write-up was wrong."));

body.push(P("The apparent mismatch — best epoch 724 versus 776, validation loss 2.386e-4 versus 2.190e-4, maximum weight difference 5.3e-2 — was measured against a checkpoint that was still being written. The first run was at epoch 724 at the moment of measurement and went on to reach epoch 776; its file timestamp (13:29) is later than the second run's (09:28), which should have been checked and was not. With both runs finished, the two checkpoints are byte-identical: same epoch, same validation loss, and a maximum weight difference of exactly zero."));

body.push(P("So --seed does make this training reproducible, and the planned opt-in --deterministic mode (cudnn.benchmark = False plus deterministic algorithms, at a throughput cost) is unnecessary and has been dropped from the queue. The noise floor established in section 7.2 is unaffected: it comes from three runs with DIFFERENT seeds, which are genuine independent draws — but it is now attributable entirely to seed and initialisation rather than to hardware nondeterminism. The peak-weighting result in section 7.3 used the mid-training value; with the final checkpoint its matched delta moves from −0.039 to −0.042, and the conclusion is unchanged."));

body.push(CALLOUT("Standing rules adopted", [
  "No single-run comparison below 0.04 is reported as an effect.",
  "Either run at least three replicates per arm, or restrict the claim to a paired within-checkpoint comparison — as section 5.4's pooling result is, which is why it remains the most robust positive finding in this record.",
  "Never compare a v3-pretrained checkpoint against a v4-pretrained one.",
  "Never score a checkpoint before confirming its training run finished. Three wrong numbers in this record came from scoring a partially-trained checkpoint — the patch-128 correction in section 5.3, and the same-seed pair above. A finished flag written into the checkpoint at end-of-training, with the evaluators refusing anything without it, would close this off for good.",
], "good"));

body.push(RULE());

// ---------- 8. EXPERIMENT #8 ----------
body.push(H1("8. Experiment #8: localising the corpus effect"));

body.push(P("Section 7.2 established that the v3-to-v4 corpus swap costs 0.069 held-out accuracy at identical configuration — the largest effect in this record — but left the mechanism open. A direct diff of the two corpora narrowed the search considerably: only 164 of 9,670 rows (1.7 per cent) differ between them at all, but each differing row differs almost entirely, because rows are min-max normalised and changing a row's maximum rescales the whole row (99.998 per cent of that row's 131,072 points change; median largest difference 0.34)."));

body.push(P("The mechanism first suspected — that version 4 leaves a residual EDTA artefact that compresses those 164 spectra — was tested directly and refuted. Only 7 of the 164 rows have their row maximum inside the EDTA window in either version, and version 4's rows are if anything slightly brighter outside that window (99.9th percentile 0.969 versus 0.917) — less compressed, not more."));

body.push(CALLOUT("Two ablation arms, built to settle it", [
  "common9506 — the 9,506 rows identical in both corpora, with the 164 differing rows removed. Self-verified after writing: every kept row was re-read from both version 3 and version 4 and confirmed to match exactly.",
  "v3rand9506 control — version 3 with 164 DIFFERENT, always-unchanged rows dropped at random. Same size as common9506, but keeps all 164 special rows. This isolates corpus SIZE from corpus CONTENT as the explanation.",
]));

body.push(TBL(
  ["Arm", "Held-out mean", "Distance to v3 (0.888)", "Distance to v4 (0.820)"],
  [
    ["v3 reference", "0.888", "—", "+0.069"],
    ["v4 (3-run mean)", "0.820", "−0.069", "—"],
    ["common9506", "0.837", "−0.051", "+0.018"],
    ["v3rand9506 control", "0.836", "−0.052", "+0.017"],
  ],
  [3200, 2200, 2600, 2080], { boldFirstCol: true }));
body.push(CAPTION("Table 23. Both ablation arms sit close to the version-4 mean (well inside the 0.020 noise floor) and clearly below version 3 (0.051, outside it)."));

body.push(P("The decisive comparison is common9506 against the control, not either arm against version 3 or version 4 in isolation. If dropping an arbitrary 164 rows lands in the same place as dropping the specific 164 that differ between corpora, the content of those rows has not been shown to matter."));

body.push(FIGURE("fig14_exp8_corpus_subset.png", 6.9));
body.push(CAPTION("Figure 10. Left: both ablation arms land near version 4, not version 3. Middle: the decisive test — common versus control, per target. Right: where that leaves the two live hypotheses."));

body.push(TBL(
  ["Target", "common9506", "control", "|difference|"],
  [
    ["Barth", "0.691", "0.713", "0.022"],
    ["MTBLS326", "0.963", "0.963", "0.000"],
    ["BrC-T2D cancer", "0.858", "0.834", "0.024"],
    ["MTBLS563", "0.591", "0.618", "0.028"],
    ["BrC-T2D diabetes", "0.751", "0.751", "0.000"],
    ["**Held-out mean", "**0.837", "**0.836", "**0.001"],
  ],
  [3200, 2360, 2360, 2160], { boldFirstCol: true }));
body.push(CAPTION("Table 24. Every per-target gap is inside the 0.035 floor; the held-out-mean gap is far inside the 0.020 floor."));

body.push(CALLOUT("Verdict: row content is refuted; corpus size is unconfirmed", [
  "REFUTED — the specific 164 rows carry the effect. common and control are indistinguishable from each other on every target and on the mean. Dropping the rows that differ between corpora gives the same answer as dropping an arbitrary matched-size set.",
  "UNCONFIRMED — corpus size (9,506 versus 9,670, a 1.7 per cent cut) explains the gap. Both arms do sit near version 4 and below version 3, which is consistent with size mattering — but a 1.7 per cent cut producing a roughly 0.05 held-out swing would be disproportionate next to every capacity result in this record: scaling the backbone by up to 2.9x the parameters moved accuracy by at most 0.02 (section 6), and that number is itself now known to be inside the noise floor.",
  "This is n = 1 per ablation arm, the same status every number in section 7 had before replication. At least two more replicates each of common9506 and the control are needed before either hypothesis is reported as established.",
], "bad"));

body.push(P("If a follow-up size sweep (dropping 1, 5 and 10 per cent of version 3 at random) shows accuracy degrading smoothly with corpus size, that would confirm size as at least part of the story. If cuts an order of magnitude larger than 1.7 per cent do not reproduce anything close to a 0.05 swing, the two single runs here were most likely an unlucky draw, and the honest position reverts to: the version 3 versus version 4 gap is real and large, but its cause remains open.", { italics: true, color: C.muted }));

body.push(RULE());

// ---------- 9. INFRA BUGS ----------
body.push(H1("9. Infrastructure defects found along the way"));
body.push(P("Several of these would silently corrupt results, so they are recorded as findings in their own right."));
body.push(TBL(
  ["Defect", "Consequence", "Resolution"],
  [
    ["nhead not recorded in checkpoints", "nn.MultiheadAttention stores in_proj_weight at (3·d_model, d_model) for any head count, so a checkpoint loads silently under the WRONG nhead while reinterpreting trained weights. Training used 4; every eval script defaulted to 8.", "Checkpoints now record the full architecture, read back off the built model. Measured impact was large (diabetes 0.810 vs 0.687) — and counterintuitively the mismatched value performed better."],
    ["Barth eval used the raw file", "100% of Barth evaluation spectra had no water suppression.", "Evaluation paths repointed to pipeline outputs."],
    ["Stale default checkpoint paths", "SSL training scripts defaulted to an older, uncleaned 8,892-row corpus; the few-shot script still defaults to a completely unsuppressed Barth file.", "Training defaults updated; few-shot paths must be passed explicitly."],
    ["tqdm one line per batch", "Progress bars redraw via carriage return, which only works on a TTY. Piping through tee produced one line per batch.", "Bar disabled when stdout is not a terminal; per-epoch summaries unaffected."],
    ["GPU contention unnoticed", "A patch-2048 run shared a GPU with another user's job and ran 17× slower than benchmarked — 53 h projected versus 3.2 h.", "Moved to an idle GPU. Benchmarks are now taken with contention checked."],
  ],
  [2500, 4100, 3480], { boldFirstCol: true }));
body.push(CAPTION("Table 20. Infrastructure defects. The nhead issue is the most dangerous class: it produces wrong numbers with no error and no warning."));

body.push(RULE());

// ---------- 8. WHERE WE STAND ----------
body.push(H1("10. Where we stand"));

body.push(P("Combining the two cheap wins — the converged linear probe head and position-preserving pooling — with no retraining at all:"));
body.push(TBL(
  ["Target", "Reported DNN head", "Probe + flatten", "Classical", "vs Classical"],
  [
    ["Barth", "0.691", "**0.806", "0.705", "**+0.101 SSL wins"],
    ["MTBLS326", "0.981", "**1.000", "1.000", "0.000 tie"],
    ["BrC-T2D cancer", "0.796", "0.859", "**0.937", "−0.078"],
    ["BrC-T2D diabetes", "0.653", "0.783", "**0.829", "−0.046"],
    ["MTBLS563", "0.558", "0.621", "**0.721", "−0.100"],
  ],
  [2500, 2100, 1900, 1600, 1980], { boldFirstCol: true }));
body.push(CAPTION("Table 21. Current best position. Mean improvement of +0.078 over the originally reported numbers, achieved without retraining anything. The SSL-versus-classical record moves from 0 wins / 0 ties / 5 losses to 1 win / 1 tie / 3 losses."));

body.push(H2("10.1 What is established"));
body.push(BULLET("Masked pretraining genuinely works: +0.117 mean over a true random-initialisation control, positive on 5 of 5 targets — comfortably above the 0.020 noise floor."));
body.push(BULLET("Jigsaw pretraining adds nothing over a random projection; joint pretraining is actively harmful."));
body.push(BULLET("The masking head was underfit by ~0.12 on every target. Fixed. This is a paired within-checkpoint comparison and therefore robust."));
body.push(BULLET("Mean pooling was discarding chemical-shift position, worth +0.03 to +0.13. Fixed. Also paired within-checkpoint — the single most reliable positive result in this record."));
body.push(BULLET("Shrinking the patch below 1024 hurts: patch 128 −0.042 and patch 256 −0.034, both measured v4-against-v4."));
body.push(BULLET("Reconstruction loss does not predict — and may anti-correlate with — downstream utility."));
body.push(BULLET("The pretraining corpus version matters more than any architecture or objective change tried so far: v3-pretrained backbones transfer +0.069 better than v4-pretrained ones at identical configuration, established over three replicates."));
body.push(BULLET("Neither of experiment #7's objective changes helps. Block masking loses 0.030 despite provably making the task 41% harder; peak weighting loses 0.039 once judged against a matched corpus."));

body.push(H2("10.2 Open caveats"));
body.push(BULLET("GPU training is not reproducible even with a fixed seed (section 7.4), so any single-run number carries a per-target uncertainty of roughly 0.035. Several comparisons in sections 5 and 6 sit below that and are no longer reported as effects."));
body.push(BULLET("Whether patch 1024, patch 2048 and d256 L6 differ at all is now UNKNOWN — the corrected same-corpus gaps (+0.020 and +0.006) are at or below the noise floor. Section 6's \"the backbone axis is exhausted\" was withdrawn."));
body.push(BULLET("The version 3 versus version 4 corpus gap does NOT localise to the 1.7 per cent of rows that actually differ between them (section 8) — a same-size random-row-drop control lands in the same place as dropping those specific rows. Corpus size is the leading remaining hypothesis but is itself unconfirmed and looks disproportionate for a 1.7 per cent cut. Why the corpus version matters remains the single highest-value open question in this record."));
body.push(BULLET("MTBLS326's perfect score clears its permutation null but permutation does not test batch confounding. Whether the label correlates with acquisition date, run order or instrument has not been checked, and should be before publication."));
body.push(BULLET("Barth and MTBLS326 have no error bars (leave-one-out gives no fold variance); their gaps are within single-sample noise."));
body.push(BULLET("Barth's classical result is marginal — its permutation null reaches 0.792 against an observed 0.705."));
body.push(BULLET("Seven rows repository-wide retain a dominant EDTA-window peak (six in the corpus, one in MTBLS563): four exhaust the peak cap, three sit inside the edge margin."));

body.push(H2("10.3 Next directions"));
body.push(P("The priorities have changed again. Two objective changes have failed on top of four scaling failures, so the objective axis is not the obvious lever. The corpus-version effect is still the largest in the record, but section 8 showed it cannot yet be pinned on the rows that differ — so \"revert to version 3\" remains the right default, but \"and find out why\" is now a size-sweep experiment rather than a row-content one."));
body.push(TBL(
  ["Priority", "Experiment", "Rationale"],
  [
    ["1", "Standardise on version 3; run a corpus-size sweep to localise the effect", "Version 3 still transfers +0.069 better and that recommendation stands regardless of mechanism. Section 8 refuted row content as the cause and left corpus size unconfirmed; dropping 1, 5 and 10 per cent of version 3 at random (multiple seeds) will show whether accuracy degrades smoothly with size or whether the two section-8 single runs were an unlucky draw."],
    ["2", "Seed replicates, and a stale-checkpoint guard", "The v3 reference that the whole corpus effect rests on is a single run, and it is the highest of six comparable v3-family arms (spread 0.081, the other five averaging only 0.011 above v4). Five seeds per corpus resolves whether that effect is real or one lucky draw. The determinism work originally planned here is dropped — seeding does work (section 7.4); what is actually needed is a finished flag on checkpoints so no run is ever scored mid-training again."],
    ["3", "Few-shot benchmark on v3", "Still the one place SSL's value proposition is untested. At n = 37–113 full-data CV is near the learnable ceiling; transfer is where pretraining should pay off."],
    ["4", "Learned attention pooling", "Pooling is the only axis that has ever produced a robust gain, and fixed regional groups are unlikely to be its ceiling. Has trainable parameters, so it must be fitted inside each training fold — a head, not a frozen transform."],
    ["5", "Hybrid features", "Concatenate SSL embedding with binned areas. They are partly complementary — on diabetes and Barth the embedding beats same-resolution binning."],
    ["—", "Batch-confound audit of MTBLS326", "Prerequisite for reporting the perfect score. Until done, MTBLS326 should not count as evidence for anything."],
  ],
  [1200, 3200, 5680], { boldFirstCol: true }));
body.push(CAPTION("Table 22. Ranked next steps, revised after experiment #8. Row content is removed as a live hypothesis for the corpus effect; a size sweep replaces it."));

body.push(CALLOUT("A framing worth considering", [
  "Classical logistic regression on 1024 binned features may simply be near-optimal at these sample sizes.",
  "A defensible contribution is: \"SSL matches classical on 2 of 5 targets and loses on 3, with a rigorous account of why\" — the head deficit, the pooling defect, the failed scaling and objective axes, the reconstruction/utility disconnect, and a measured noise floor that several published-looking effects would not clear are all genuine, well-controlled findings.",
  "That is a stronger result than an unconvincing win. Experiments #7 and #8 strengthen rather than weaken this framing: knowing WHICH plausible interventions fail, and by how much relative to measurement noise, is the substance of the contribution — and knowing that an obvious-looking explanation (the 164 rows) does NOT hold up is itself a finding, not a null result.",
]));

body.push(RULE());

// ---------- 9. REPRODUCIBILITY ----------
body.push(H1("11. Reproducibility"));
body.push(H2("11.1 Data of record (v4)"));
body.push(TBL(
  ["Dataset", "File"],
  [
    ["Pretraining corpus", "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4.npy"],
    ["Barth", "data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_rowMinMax_v4.npy"],
    ["MTBLS326", "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy"],
    ["MTBLS563", "data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy"],
    ["BrC-T2D (revised labels)", "data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_rowMinMax_v4.npy"],
  ],
  [2600, 7480], { boldFirstCol: true }));

body.push(H2("11.2 Key scripts"));
body.push(TBL(
  ["Purpose", "Script"],
  [
    ["Build clean datasets", "code/preprocessing/build_clean_datasets.py"],
    ["EDTA magnitude detector", "code/preprocessing/suppress_edta_magnitude.py"],
    ["Suppression audit", "code/analysis/validate_suppression.py"],
    ["Gap decomposition (#1)", "code/analysis/probe_logreg_advantage.py"],
    ["Linear probe (#2)", "code/analysis/linear_probe_frozen_embeddings.py"],
    ["Probe evaluator + random-backbone control (#3, #5)", "code/evaluation/ssl_linear_probe_eval.py"],
    ["Patch-size comparison (#4)", "code/analysis/compare_patch_sizes.py"],
    ["Pooling sweep", "code/analysis/sweep_pooling.py"],
    ["Peak-weighted loss verification (#7)", "code/tests/verify_top_peak_loss.py"],
    ["Factorial summary (#7)", "code/analysis/summarize_exp7_factorial.py"],
    ["Replicates + noise floor (#7b)", "code/analysis/summarize_exp7_replicates.py"],
    ["Corpus-subset builder (#8)", "code/preprocessing/build_corpus_subset.py"],
    ["Corpus-subset ablation summary (#8)", "code/analysis/summarize_exp8_corpus_subset.py"],
    ["SSL pretraining (masking)", "code/training/trainer_revised.py"],
  ],
  [4200, 5880], { boldFirstCol: true }));

body.push(CALLOUT("Four rules that govern every number in this record", [
  "Corpus: never compare a v3-pretrained checkpoint against a v4-pretrained one. The corpus version alone is worth 0.069 on the held-out mean (section 7.2) — though note the v3 side of that comparison is a single run, which the seed study is now testing.",
  "Noise: a held-out-mean difference below 0.020 is not an effect, and a single-target difference below 0.035 is not an effect, unless supported by at least three replicates per arm or measured as a paired within-checkpoint comparison (section 7.2).",
  "Mechanism: do not assume the corpus effect lives in the 164 rows that differ between versions 3 and 4 — section 8 tested that directly and it did not hold up, and section 8's follow-up size sweep did not support corpus size either. The cause is still open.",
  "Provenance: never score a checkpoint before confirming its training run finished (section 7.4). Three wrong numbers here came from scoring one mid-training.",
], "bad"));

body.push(H2("11.3 BrC-T2D label revision"));
body.push(P("The revised label file changed the diabetes status of exactly four samples; cancer status changed for none. An earlier count of 41 changes was an artefact of comparing a display string (\"Cancer\" versus \"Breast Cancer\"), not the label columns the evaluation reads."));
body.push(TBL(
  ["Sample", "Old diabetes status", "New diabetes status"],
  [
    ["SM58", "Diabetes", "No Diabetes"],
    ["SM35", "No Diabetes", "Diabetes"],
    ["SM23", "Diabetes", "No Diabetes"],
    ["KM70", "No Diabetes", "Diabetes"],
  ],
  [2600, 3740, 3740], { boldFirstCol: true }));
body.push(CAPTION("Table 26. Genuine label changes. Extraction was verified by byte-comparing all 78 extracted spectra against the source array and independently cross-referencing every sample ID — zero mismatches."));

body.push(SPACER(200));
body.push(P("End of record. This document is intended to be extended as work continues; each new experiment should be added to section 5, 6 or 7 with its prediction, outcome, and any decision taken. Note the standing rule from section 7.4: no single-run comparison below 0.04 is reported as an effect.",
  { italics: true, color: C.muted }));

// ============================================================
const doc = new Document({
  creator: "NMR Metabolomics project",
  title: "NMR Metabolomics Foundation Model — Laboratory Record",
  description: "Lab record of SSL vs classical ML analyses on 1H-NMR spectra",
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 460, hanging: 260 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 900, hanging: 260 } } } },
        ],
      },
      {
        reference: "nums",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 460, hanging: 260 } } } },
        ],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21, color: C.ink } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H, orientation: PageOrientation.PORTRAIT },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: C.rule } },
          children: [new TextRun({ text: "NMR Metabolomics — Laboratory Record", size: 16, color: C.muted, font: "Calibri" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: C.muted, font: "Calibri" })],
        })],
      }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(REPO, "docs/NMR_Metabolomics_Lab_Record.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote " + out + " (" + (buf.length / 1024).toFixed(0) + " KB)");
});
