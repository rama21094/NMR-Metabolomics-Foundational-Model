// Group-meeting deck, August 2026 — continues from "GM - 27th Feb 2026".
//   node docs/build_group_meeting_deck.js
// Figures come from docs/gm_figures/ (built by code/plotting/plot_groupmeeting_figures.py,
// which sizes every text element >= 16pt Arial-metric for the PI's readability floor).

const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = path.join(__dirname, "gm_figures");

// ---- palette (matches the figure script exactly) ----
const NAVY = "21295C";   // midnight — dark slides
const DEEP = "065A82";   // deep blue — SSL / primary
const TEAL = "1C7293";   // teal — secondary
const GOLD = "B8860B";   // classical ML
const CORAL = "C1435B";  // retractions / negative
const GREEN = "1A7A3C";  // survives
const INK = "1A1A1A";
const MUTED = "5A6068";
const TINT = "EEF3F7";   // card background
const TINT_BAD = "FBEEF1";
const TINT_OK = "EBF5EE";

const F = "Arial";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.3 x 7.5 in — MUST precede addSlide
pres.author = "Shankararama Sharma";
pres.title = "Foundation Model for NMR Metabolomics — Group Meeting Aug 2026";

// Recorded aspect ratios; any figure used here must be registered so a renamed
// or re-rendered figure fails loudly instead of being silently stretched.
const AR = {
  "gm01_headline.png": 2.3939,
  "gm02_free_wins.png": 2.3939,
  "gm03_pretraining_gain.png": 2.3939,
  "gm04_seed_study.png": 2.3939,
  "gm05_recalibration.png": 2.3939,
  "gm06_fewshot_curves.png": 2.3810,
  "gm07_fewshot_paired.png": 2.3939,
  "gm08_reconstruction.png": 2.7160,
};

/** Place an image centred in a box, preserving aspect ratio. */
function fig(slide, name, { x, y, w, h }) {
  const ar = AR[name];
  if (!ar) throw new Error(`no aspect ratio recorded for ${name}`);
  let dw = w, dh = w / ar;
  if (dh > h) { dh = h; dw = h * ar; }
  slide.addImage({ path: path.join(FIG, name), x: x + (w - dw) / 2, y: y + (h - dh) / 2, w: dw, h: dh });
}

let pageNo = 0;
function slideTitle(slide, text, sub) {
  slide.addText(text, {
    x: 0.55, y: 0.30, w: 12.2, h: 0.72, fontFace: F, fontSize: 30, bold: true,
    color: NAVY, align: "left", margin: 0, valign: "middle",
  });
  if (sub) {
    slide.addText(sub, {
      x: 0.55, y: 1.03, w: 12.2, h: 0.42, fontFace: F, fontSize: 17,
      color: MUTED, align: "left", margin: 0, valign: "middle",
    });
  }
}
function newLight(title, sub) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  if (title) slideTitle(s, title, sub);
  pageNo += 1;
  s.addText(String(pageNo), {
    x: 12.72, y: 6.95, w: 0.45, h: 0.3, fontFace: F, fontSize: 12,
    color: "AAB0B8", align: "right", margin: 0,
  });
  return s;
}
function newDark(color) {
  const s = pres.addSlide();
  s.background = { color: color || NAVY };
  pageNo += 1;
  return s;
}
/** A tinted content card. No edge stripes — background tint only. */
function card(slide, { x, y, w, h, fill, line }) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill || TINT }, rectRadius: 0.08,
    line: line ? { color: line, width: 1 } : { type: "none" },
  });
}
/** The deck's repeated motif: a filled circle holding a number or short glyph. */
function bubble(slide, { x, y, d, color, label, fontSize }) {
  slide.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d, fill: { color }, line: { type: "none" },
  });
  slide.addText(label, {
    x, y, w: d, h: d, fontFace: F, fontSize: fontSize || 18, bold: true,
    color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });
}

/* ============================== 1. TITLE ============================== */
{
  const s = newDark();
  s.addText("Foundation Model for", {
    x: 0.9, y: 1.75, w: 11.5, h: 0.66, fontFace: F, fontSize: 34, color: "9FC4DA", margin: 0,
  });
  s.addText("1D NMR Metabolomics", {
    x: 0.9, y: 2.34, w: 11.5, h: 0.92, fontFace: F, fontSize: 48, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("What we built, what we found, and the three things we got wrong", {
    x: 0.9, y: 3.42, w: 11.5, h: 0.5, fontFace: F, fontSize: 21, color: "CADCFC", margin: 0, italic: true,
  });
  s.addText([
    { text: "Shankararama Sharma", options: { fontSize: 19, bold: true, color: "FFFFFF", breakLine: true } },
    { text: "Group Meeting  ·  August 2026", options: { fontSize: 17, color: "9FC4DA", breakLine: true } },
    { text: "continues from 27 Feb 2026", options: { fontSize: 16, color: "7FA8C4", italic: true } },
  ], { x: 0.9, y: 4.42, w: 11.5, h: 1.3, fontFace: F, margin: 0, lineSpacing: 26 });

  // three headline numbers
  const stats = [
    ["9,670", "spectra pretrained on\n(was 2,146 in Feb)"],
    ["5", "evaluation targets\nacross 4 cohorts"],
    ["15", "numbered experiments\n3 retractions"],
  ];
  stats.forEach(([big, lab], i) => {
    const x = 0.9 + i * 4.0;
    s.addText(big, {
      x, y: 5.85, w: 1.5, h: 0.62, fontFace: F, fontSize: 36, bold: true, color: "6FD3C7", margin: 0,
    });
    s.addText(lab, {
      x: x + 1.6, y: 5.86, w: 2.3, h: 0.62, fontFace: F, fontSize: 14.5, color: "CADCFC",
      margin: 0, valign: "middle", lineSpacing: 16,
    });
  });
  s.addNotes("45 min, questions welcome throughout.\n\n"
    + "TIMING GUIDE (leaves ~8 min slack for interruptions):\n"
    + "  slides 1-4    framing + the arc ................ 6 min\n"
    + "  slides 5-9    what we built ................... 8 min\n"
    + "  slides 10-14  the headline + diagnosis ........ 9 min\n"
    + "  slides 15-21  the three mistakes .............. 10 min  <- the important part\n"
    + "  slides 22-25  the few-shot result ............. 7 min\n"
    + "  slides 26-29  where we stand + discussion ..... 5 min\n\n"
    + "If running long, slide 6 (preprocessing versions) and slide 9 (protocol) are the two "
    + "that can be summarised in one sentence each. Do NOT cut 19, 20, 23 or 24.");
}

/* ============================== 2. THE GOAL ============================== */
{
  const s = newLight("The goal — unchanged since February",
    "Why a foundation model is the right idea for metabolomics");
  const items = [
    ["1", DEEP, "The problem", "Metabolomics cohorts are tiny — 10s to 100s of samples. Deep networks need far more than that to train from scratch."],
    ["2", TEAL, "The idea", "Pretrain once on thousands of UNLABELLED spectra with a self-supervised task. No labels needed — we have plenty of raw spectra."],
    ["3", GOLD, "The payoff", "Transfer that representation to a small labelled cohort. The backbone already knows what NMR spectra look like."],
    ["4", GREEN, "Success criterion", "It must beat classical ML (binned intensities + logistic regression) on small datasets. That is the bar."],
  ];
  items.forEach(([n, col, head, body], i) => {
    const y = 1.62 + i * 1.34;
    card(s, { x: 0.55, y, w: 12.2, h: 1.16 });
    bubble(s, { x: 0.78, y: y + 0.28, d: 0.6, color: col, label: n, fontSize: 19 });
    s.addText(head, {
      x: 1.56, y: y + 0.13, w: 3.0, h: 0.42, fontFace: F, fontSize: 19, bold: true, color: col, margin: 0, valign: "middle",
    });
    s.addText(body, {
      x: 1.56, y: y + 0.52, w: 10.95, h: 0.56, fontFace: F, fontSize: 16.5, color: INK, margin: 0, valign: "top", lineSpacing: 19,
    });
  });
  s.addNotes("This framing has not changed. Everything that follows is about whether we actually "
    + "cleared the bar in item 4 — and the honest answer today is: not yet.");
}

/* ====================== 3. WHERE WE WERE IN FEBRUARY ====================== */
{
  const s = newLight("Where we were on 27 February", "The state of play at the last group meeting");
  card(s, { x: 0.55, y: 1.6, w: 5.95, h: 4.35, fill: TINT_OK });
  s.addText("What was working", {
    x: 0.85, y: 1.78, w: 5.4, h: 0.4, fontFace: F, fontSize: 20, bold: true, color: GREEN, margin: 0,
  });
  s.addText([
    { text: "Masked Spectra Modelling trained end-to-end", options: { bullet: true, breakLine: true } },
    { text: "2,146 CPMG serum spectra from MetaboLights", options: { bullet: true, breakLine: true } },
    { text: "Reconstruction was excellent — MSE ≈ 3×10⁻⁵, R² ≈ 0.98 at 25–35% masking", options: { bullet: true, breakLine: true } },
    { text: "Full preprocessing pipeline: TopSpin → align → water-suppress → dedupe → normalise", options: { bullet: true, breakLine: true } },
    { text: "Three evaluation routes sketched: embeddings+ML, classification head, ML-only", options: { bullet: true } },
  ], {
    x: 0.85, y: 2.16, w: 5.4, h: 3.68, fontFace: F, fontSize: 15.5, color: INK,
    margin: 0, paraSpaceAfter: 15, lineSpacing: 19,
  });

  card(s, { x: 6.8, y: 1.6, w: 5.95, h: 4.35, fill: TINT_BAD });
  s.addText("What was missing", {
    x: 7.1, y: 1.78, w: 5.4, h: 0.4, fontFace: F, fontSize: 20, bold: true, color: CORAL, margin: 0,
  });
  s.addText([
    { text: "Only ONE test cohort (TBI, ~57 samples) — no way to tell general from idiosyncratic", options: { bullet: true, breakLine: true } },
    { text: "No head-to-head against classical ML on identical folds", options: { bullet: true, breakLine: true } },
    { text: "No control asking whether pretraining helped at all vs a random network", options: { bullet: true, breakLine: true } },
    { text: "No error bars anywhere — every arm was a single training run", options: { bullet: true, breakLine: true } },
    { text: "Reconstruction quality was our success metric. It turns out not to be one.", options: { bullet: true } },
  ], {
    x: 7.1, y: 2.16, w: 5.4, h: 3.68, fontFace: F, fontSize: 15.5, color: INK,
    margin: 0, paraSpaceAfter: 9, lineSpacing: 18.5,
  });

  s.addText("Most of the last five months went into fixing the right-hand column.", {
    x: 0.55, y: 6.18, w: 12.2, h: 0.45, fontFace: F, fontSize: 18, bold: true,
    color: NAVY, align: "center", margin: 0,
  });
  s.addNotes("Be upfront: the February deck showed a model that reconstructed beautifully and an "
    + "evaluation that could not support conclusions. The reconstruction quality was real; it just "
    + "turned out to be the wrong thing to optimise.");
}

/* ========================= 4. THE ARC SINCE THEN ========================= */
{
  const s = newLight("What happened since — the arc", "Five months in one picture");
  const steps = [
    ["Scale up", "2,146 → 9,670 spectra\n4 cohorts, 5 targets", DEEP],
    ["Compare\nproperly", "Same folds, same metric,\nclassical vs SSL", TEAL],
    ["Diagnose\nthe gap", "Head deficit vs\nrepresentation ceiling", GOLD],
    ["Try to fix\nthe backbone", "Patch size, capacity,\nobjectives, corpus", CORAL],
    ["Learn to\nmeasure noise", "10 seed replicates.\n3 claims retracted", NAVY],
    ["Test the\npremise", "Few-shot benchmark\non all 5 targets", GREEN],
  ];
  const w = 1.86, gap = 0.19;
  steps.forEach(([head, body, col], i) => {
    const x = 0.55 + i * (w + gap);
    card(s, { x, y: 1.95, w, h: 2.5, fill: TINT });
    bubble(s, { x: x + w / 2 - 0.28, y: 2.14, d: 0.56, color: col, label: String(i + 1), fontSize: 18 });
    s.addText(head, {
      x: x + 0.06, y: 2.82, w: w - 0.12, h: 0.66, fontFace: F, fontSize: 16.5, bold: true,
      color: col, align: "center", margin: 0, valign: "top", lineSpacing: 18,
    });
    s.addText(body, {
      x: x + 0.06, y: 3.52, w: w - 0.12, h: 0.82, fontFace: F, fontSize: 14, color: INK,
      align: "center", margin: 0, valign: "top", lineSpacing: 16,
    });
    if (i < steps.length - 1) {
      s.addShape(pres.ShapeType.rightArrow, {
        x: x + w + 0.015, y: 3.05, w: 0.16, h: 0.18, fill: { color: "B9C2CC" }, line: { type: "none" },
      });
    }
  });
  card(s, { x: 0.55, y: 4.78, w: 12.2, h: 1.5, fill: TINT_OK });
  s.addText("The honest summary of the arc", {
    x: 0.85, y: 4.94, w: 11.6, h: 0.36, fontFace: F, fontSize: 18, bold: true, color: GREEN, margin: 0,
  });
  s.addText("Steps 1–3 worked and produced two real improvements. Step 4 failed on every axis we "
    + "tried. Step 5 revealed that most of step 4's \"results\" were noise. Step 6 tested the "
    + "project's core premise directly — and it did not hold.",
    { x: 0.85, y: 5.34, w: 11.6, h: 0.86, fontFace: F, fontSize: 17, color: INK, margin: 0, lineSpacing: 21 });
  s.addNotes("This is the roadmap for the talk. I will spend most time on 5 and 6 because that is "
    + "where the real learning is.");
}

/* ============================== 5. DATA ============================== */
{
  const s = newLight("Step 1 — more data, and real evaluation cohorts",
    "February had one test set. Now there are five targets across four independent cohorts.");
  card(s, { x: 0.55, y: 1.72, w: 3.7, h: 2.62, fill: TINT });
  s.addText("Pretraining corpus", {
    x: 0.8, y: 1.9, w: 3.2, h: 0.34, fontFace: F, fontSize: 17.5, bold: true, color: DEEP, margin: 0,
  });
  s.addText([
    { text: "2,146", options: { fontSize: 26, bold: true, color: MUTED } },
    { text: "   →   ", options: { fontSize: 20, color: MUTED } },
    { text: "9,670", options: { fontSize: 38, bold: true, color: DEEP } },
  ], { x: 0.8, y: 2.26, w: 3.2, h: 0.66, fontFace: F, margin: 0, valign: "middle" });
  s.addText("unlabelled CPMG serum / plasma spectra\nMetaboLights + Metabolomics Workbench\n131,072 points each, deduplicated", {
    x: 0.8, y: 3.02, w: 3.2, h: 1.2, fontFace: F, fontSize: 14, color: INK, margin: 0, lineSpacing: 17,
  });

  const rows = [
    ["Barth Syndrome", "n = 37", "Case / Control", "LOOCV"],
    ["MTBLS326", "n = 42", "Yes / No (IP3R)", "LOOCV"],
    ["MTBLS563", "n = 113", "3-class diagnosis", "10-fold"],
    ["BrC-T2D — cancer", "n = 78", "Cancer / No cancer", "10-fold"],
    ["BrC-T2D — diabetes", "n = 78", "Diabetes / No", "10-fold"],
  ];
  s.addTable(
    [[
      { text: "Evaluation target", options: { bold: true, color: "FFFFFF", fill: { color: DEEP }, fontSize: 16 } },
      { text: "Size", options: { bold: true, color: "FFFFFF", fill: { color: DEEP }, fontSize: 16 } },
      { text: "Label", options: { bold: true, color: "FFFFFF", fill: { color: DEEP }, fontSize: 16 } },
      { text: "Protocol", options: { bold: true, color: "FFFFFF", fill: { color: DEEP }, fontSize: 16 } },
    ]].concat(rows.map((r, i) => r.map((c, j) => ({
      text: c,
      options: {
        fontSize: 15.5, color: INK, bold: j === 0,
        fill: { color: i % 2 ? "FFFFFF" : "F4F7FA" },
      },
    })))),
    {
      x: 4.5, y: 1.72, w: 8.25, colW: [3.3, 1.25, 2.35, 1.35], rowH: 0.38,
      border: { type: "solid", color: "D8DEE4", pt: 0.75 }, fontFace: F, valign: "middle",
      margin: [0.04, 0.08, 0.04, 0.08],
    });

  card(s, { x: 0.55, y: 4.52, w: 12.2, h: 1.82, fill: TINT_BAD });
  s.addText("Two things to flag about the evaluation set", {
    x: 0.85, y: 4.64, w: 11.6, h: 0.34, fontFace: F, fontSize: 18, bold: true, color: CORAL, margin: 0,
  });
  s.addText([
    { text: "Barth and MTBLS326 use LOOCV, so they have NO fold variance — no error bars. Differences of 0.02 there are one sample.", options: { bullet: true, breakLine: true } },
    { text: "MTBLS326's classical score is a perfect 1.000 on n=42. It clears a permutation null, but we have not yet checked run-order / batch confounding. Until we do, it should not count as evidence.", options: { bullet: true } },
  ], { x: 0.85, y: 5.00, w: 11.6, h: 1.26, fontFace: F, fontSize: 15.5, color: INK, margin: 0, paraSpaceAfter: 7, lineSpacing: 18.5 });
  s.addNotes("Corpus is still small for a foundation model — 9,670 is orders of magnitude below "
    + "where masked pretraining usually starts to pay. That becomes one of my proposed next steps.");
}

/* ====================== 6. PREPROCESSING EVOLUTION ====================== */
{
  const s = newLight("A detour worth mentioning — four preprocessing versions",
    "We found a real bug. Then we found out it did not matter.");
  const vs = [
    ["v1", "Water suppression only\npoints 62,500–68,000 → 0", MUTED],
    ["v2", "EDTA window skipped\n(left largely untouched)", TEAL],
    ["v3", "Uniform magnitude-based\nEDTA suppression", DEEP],
    ["v4", "EDTA cutoff FIXED to the\nbaseline-to-peak midpoint", GREEN],
  ];
  vs.forEach(([tag, body, col], i) => {
    const x = 0.55 + i * 3.10;
    card(s, { x, y: 1.75, w: 2.86, h: 1.62, fill: TINT });
    s.addText(tag, {
      x: x + 0.14, y: 1.9, w: 1.0, h: 0.44, fontFace: F, fontSize: 24, bold: true, color: col, margin: 0,
    });
    s.addText(body, {
      x: x + 0.14, y: 2.38, w: 2.58, h: 0.88, fontFace: F, fontSize: 14.5, color: INK, margin: 0, lineSpacing: 17.5,
    });
    if (i < 3) {
      s.addShape(pres.ShapeType.rightArrow, {
        x: x + 2.89, y: 2.48, w: 0.15, h: 0.17, fill: { color: "B9C2CC" }, line: { type: "none" },
      });
    }
  });

  card(s, { x: 0.55, y: 3.62, w: 5.95, h: 2.62, fill: TINT_OK });
  s.addText("EDTA is a real contaminant problem", {
    x: 0.85, y: 3.78, w: 5.4, h: 0.36, fontFace: F, fontSize: 18, bold: true, color: GREEN, margin: 0,
  });
  s.addText([
    { text: "Blood-collection tubes leave EDTA peaks that can dominate a spectrum and swamp the row normalisation.", options: { bullet: true, breakLine: true } },
    { text: "v3 → v4 fixed a genuine cutoff bug: the suppression was removing too much of the surrounding signal.", options: { bullet: true, breakLine: true } },
    { text: "7 rows repo-wide still retain a dominant EDTA peak — documented, not hidden.", options: { bullet: true } },
  ], { x: 0.85, y: 4.18, w: 5.4, h: 2.0, fontFace: F, fontSize: 15.5, color: INK, margin: 0, paraSpaceAfter: 9, lineSpacing: 19 });

  card(s, { x: 6.8, y: 3.62, w: 5.95, h: 2.62, fill: TINT_BAD });
  s.addText("…but it does not matter downstream", {
    x: 7.1, y: 3.78, w: 5.4, h: 0.4, fontFace: F, fontSize: 18, bold: true, color: CORAL, margin: 0,
  });
  s.addText([
    { text: "We spent two whole experiments trying to explain a v3-vs-v4 downstream difference.", options: { bullet: true, breakLine: true } },
    { text: "With 5 seeds per corpus that difference is −0.014 ± 0.022 — indistinguishable from zero (slide 19).", options: { bullet: true, breakLine: true } },
    { text: "Lesson: fix the preprocessing because it is CORRECT, not because you measured a downstream gain from one run.", options: { bullet: true } },
  ], { x: 7.1, y: 4.28, w: 5.4, h: 1.90, fontFace: F, fontSize: 15, color: INK, margin: 0, paraSpaceAfter: 8, lineSpacing: 18 });
  s.addNotes("Keep this short unless asked. The point is methodological: correctness and measurable "
    + "downstream benefit are different justifications, and we conflated them.");
}

/* =================== 7. ARCHITECTURE + THREE OBJECTIVES =================== */
{
  const s = newLight("The backbone, and the three objectives we tried",
    "Architecture is unchanged from February; two more pretext tasks were added");
  const boxes = [
    ["Input spectrum", "131,072 points"],
    ["Patch split\n+ mask", "1,024-pt patches\n→ 128 tokens\nmask 20–60%"],
    ["Patch embed\n+ position", "Linear → 128-d\n+ positional enc."],
    ["Transformer\nencoder", "3 layers, 4 heads\nff 256, pre-norm"],
    ["Reconstruction\nhead", "MLP → patches"],
    ["Output", "masked regions\nrestored"],
  ];
  const bw = 1.92, bgap = 0.14;
  boxes.forEach(([head, body], i) => {
    const x = 0.55 + i * (bw + bgap);
    const isEnc = i === 3;
    card(s, { x, y: 1.68, w: bw, h: 1.62, fill: isEnc ? "D6E6EF" : TINT, line: isEnc ? DEEP : null });
    s.addText(head, {
      x: x + 0.07, y: 1.79, w: bw - 0.14, h: 0.58, fontFace: F, fontSize: 15.5, bold: true,
      color: isEnc ? DEEP : NAVY, align: "center", margin: 0, valign: "top", lineSpacing: 17,
    });
    s.addText(body, {
      x: x + 0.07, y: 2.4, w: bw - 0.14, h: 0.82, fontFace: F, fontSize: 13.5, color: MUTED,
      align: "center", margin: 0, valign: "top", lineSpacing: 15.5,
    });
    if (i < boxes.length - 1) {
      s.addShape(pres.ShapeType.rightArrow, {
        x: x + bw + 0.005, y: 2.4, w: 0.13, h: 0.15, fill: { color: "9AA6B2" }, line: { type: "none" },
      });
    }
  });
  s.addText("1.89 M parameters  ·  patch 1024  ·  d_model 128  ·  best val reconstruction 9.3×10⁻⁵", {
    x: 0.55, y: 3.38, w: 12.2, h: 0.34, fontFace: F, fontSize: 16, color: "3D4450",
    align: "center", margin: 0, bold: true,
  });

  const objs = [
    ["Masked (MSM)", "Hide 20–60% of patches, reconstruct them.", "The one that works", GREEN, TINT_OK],
    ["Jigsaw", "Shuffle spectral bins, predict the original order.", "No better than random init", CORAL, TINT_BAD],
    ["Joint", "Masked + jigsaw together, shared encoder.", "Actively harmful", CORAL, TINT_BAD],
  ];
  objs.forEach(([head, body, verdict, col, fill], i) => {
    const x = 0.55 + i * 4.13;
    card(s, { x, y: 3.9, w: 3.94, h: 2.35, fill });
    s.addText(head, {
      x: x + 0.2, y: 4.06, w: 3.54, h: 0.38, fontFace: F, fontSize: 19, bold: true, color: NAVY, margin: 0,
    });
    s.addText(body, {
      x: x + 0.2, y: 4.5, w: 3.54, h: 0.9, fontFace: F, fontSize: 16, color: INK, margin: 0, lineSpacing: 19,
    });
    s.addText(verdict, {
      x: x + 0.2, y: 5.55, w: 3.54, h: 0.55, fontFace: F, fontSize: 17, bold: true, color: col, margin: 0, valign: "middle",
    });
  });
  s.addText("(the verdicts come from the random-init control on slide 14)", {
    x: 0.55, y: 6.38, w: 12.2, h: 0.3, fontFace: F, fontSize: 14.5, color: MUTED, align: "center", margin: 0, italic: true,
  });
  s.addNotes("Architecture questions likely here. Patch 1024 = 128 tokens over 131k points. That "
    + "resolution limit becomes a hypothesis we test and refute later.");
}

/* ============ 8. RECONSTRUCTION WORKS — BUT IS THE WRONG METRIC ============ */
{
  const s = newLight("The pretext task itself works very well",
    "Re-measured from February — and this is exactly where the trap was");
  fig(s, "gm08_reconstruction.png", { x: 0.55, y: 1.50, w: 12.2, h: 4.05 });
  s.addText([
    { text: "Across 50 spectra:  ", options: { color: MUTED } },
    { text: "r = 0.940 ± 0.090", options: { bold: true, color: NAVY } },
    { text: "  on the hidden bins only", options: { color: MUTED } },
    { text: "     (whole-spectrum r = 0.979 — but that includes the 75% of bins the model was given)",
      options: { color: MUTED, italic: true } },
  ], {
    x: 0.55, y: 5.62, w: 12.2, h: 0.32, fontFace: F, fontSize: 15, align: "center", margin: 0,
  });

  card(s, { x: 0.55, y: 6.06, w: 12.2, h: 0.94, fill: TINT_BAD });
  s.addText([
    { text: "The trap:  ", options: { bold: true, color: CORAL } },
    { text: "the pretext task is genuinely solved well — and it still tells you almost nothing. "
        + "Across five backbones, BETTER reconstruction went with WORSE transfer; our "
        + "2.3×-better-reconstructing model transferred 0.06 worse. So we never select a checkpoint, "
        + "architecture or epoch on reconstruction loss.", options: { color: INK } },
  ], {
    x: 0.85, y: 6.13, w: 11.6, h: 0.80, fontFace: F, fontSize: 15.5, margin: 0,
    valign: "middle", lineSpacing: 19,
  });
  s.addNotes("This is the single most transferable lesson for anyone else in the group doing SSL: a "
    + "beautiful reconstruction figure tells you almost nothing about whether the features are useful.");
}

/* ======================== 9. EVALUATION PROTOCOL ======================== */
{
  const s = newLight("How we compare now", "The protocol that made everything after this interpretable");
  const left = [
    ["Identical folds", "Both tracks see exactly the same splits — LOOCV for Barth / MTBLS326, stratified 10-fold (seed 42) for the rest."],
    ["Identical metric", "Balanced accuracy everywhere, because several targets are imbalanced (Barth is 14 / 23)."],
    ["Both classifiers linear", "Classical = StandardScaler → LogReg. SSL = frozen embedding → LogReg probe. So a difference is about the FEATURES, not the classifier."],
  ];
  left.forEach(([head, body], i) => {
    const y = 1.62 + i * 1.62;
    card(s, { x: 0.55, y, w: 7.4, h: 1.44 });
    s.addText(head, {
      x: 0.85, y: y + 0.14, w: 6.8, h: 0.36, fontFace: F, fontSize: 18, bold: true, color: DEEP, margin: 0,
    });
    s.addText(body, {
      x: 0.85, y: y + 0.54, w: 6.85, h: 0.82, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
    });
  });

  card(s, { x: 8.25, y: 1.62, w: 4.5, h: 4.86, fill: TINT_OK });
  s.addText("The validation gate", {
    x: 8.55, y: 1.80, w: 3.95, h: 0.38, fontFace: F, fontSize: 19, bold: true, color: GREEN, margin: 0,
  });
  s.addText("Before drawing any conclusion, the harness had to reproduce the committed classical "
    + "numbers to 6 decimal places on all five targets:", {
    x: 8.55, y: 2.24, w: 3.95, h: 1.05, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });
  s.addText([
    { text: "BrC-T2D cancer", options: { breakLine: true } },
    { text: "BrC-T2D diabetes", options: { breakLine: true } },
    { text: "MTBLS563", options: { breakLine: true } },
    { text: "MTBLS326", options: { breakLine: true } },
    { text: "Barth", options: {} },
  ], { x: 8.55, y: 3.42, w: 2.3, h: 1.75, fontFace: F, fontSize: 15, color: NAVY, margin: 0, lineSpacing: 24 });
  s.addText([
    { text: "0.936842", options: { breakLine: true } },
    { text: "0.828877", options: { breakLine: true } },
    { text: "0.720785", options: { breakLine: true } },
    { text: "1.000000", options: { breakLine: true } },
    { text: "0.704969", options: {} },
  ], { x: 10.85, y: 3.42, w: 1.65, h: 1.75, fontFace: "Courier New", fontSize: 15,
       color: NAVY, bold: true, align: "right", margin: 0, lineSpacing: 22 });
  s.addText("Without this, none of the comparisons would mean anything.", {
    x: 8.55, y: 5.32, w: 3.95, h: 1.0, fontFace: F, fontSize: 15.5, bold: true, color: GREEN, margin: 0, lineSpacing: 19,
  });
  s.addNotes("The 6-decimal gate is worth emphasising — it is what lets us attribute differences to "
    + "the features rather than to a plumbing discrepancy.");
}

/* ============================ 10. HEADLINE ============================ */
{
  const s = newLight("Step 2 result — classical ML beat every SSL family",
    "Balanced accuracy, v4 data, best mode per family, identical folds");
  fig(s, "gm01_headline.png", { x: 0.55, y: 1.50, w: 12.2, h: 4.59 });
  card(s, { x: 0.55, y: 6.20, w: 12.2, h: 0.80, fill: TINT_BAD });
  s.addText("Consistent ordering: masked > jigsaw ≈ joint. Two of the gaps are large (0.14–0.18). "
    + "Barth and MTBLS326 are effectively ties — half a sample and one sample respectively.", {
    x: 0.85, y: 6.28, w: 11.6, h: 0.66, fontFace: F, fontSize: 15.5, color: INK, margin: 0, valign: "middle", lineSpacing: 18,
  });
  s.addNotes("This is the result that set the agenda for the next four months. Note it was measured "
    + "with the reported DNN head and mean-pooling — both of which turned out to be fixable.");
}

/* ========================== 11. DECOMPOSITION ========================== */
{
  const s = newLight("The gap is not one problem — it is two",
    "Run the SAME logistic regression on the frozen SSL embedding to separate them");
  card(s, { x: 0.55, y: 1.62, w: 5.95, h: 2.05, fill: TINT });
  s.addText("Head-fitting deficit", {
    x: 0.85, y: 1.78, w: 5.4, h: 0.38, fontFace: F, fontSize: 19, bold: true, color: TEAL, margin: 0,
  });
  s.addText("The SSL head is a linear layer trained by Adam for ~50 epochs on a handful of samples. "
    + "LogReg on the IDENTICAL features fits the same linear map properly (L-BFGS to convergence, "
    + "explicit L2, standardised inputs).", {
    x: 0.85, y: 2.2, w: 5.4, h: 1.35, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });
  card(s, { x: 6.8, y: 1.62, w: 5.95, h: 2.05, fill: TINT });
  s.addText("Representation ceiling", {
    x: 7.1, y: 1.78, w: 5.4, h: 0.38, fontFace: F, fontSize: 19, bold: true, color: GOLD, margin: 0,
  });
  s.addText("Whatever survives the better classifier is the embedding's own limit — information the "
    + "backbone never encoded, which 1,024-bin integrated areas do carry.", {
    x: 7.1, y: 2.2, w: 5.4, h: 1.35, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });

  const rows = [
    ["BrC-T2D cancer", "0.796", "0.833", "0.937", "0.037", "0.104", 1],
    ["BrC-T2D diabetes", "0.653", "0.810", "0.829", "0.157", "0.019", 0],
    ["MTBLS563", "0.558", "0.607", "0.721", "0.049", "0.114", 1],
    ["Barth", "0.691", "0.770", "0.705", "0.079", "−0.065", 0],
    ["MTBLS326", "0.981", "0.944", "1.000", "−0.037", "0.056", 1],
  ];
  const hdr = ["Target", "SSL head", "LogReg on\nsame embedding", "LogReg @\n1024 bins", "Head\ndeficit", "Representation\nceiling"];
  s.addTable(
    [hdr.map(h => ({ text: h, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, fontSize: 15, align: "center" } }))]
      .concat(rows.map((r, i) => r.slice(0, 6).map((c, j) => ({
        text: c,
        options: {
          fontSize: 15.5, bold: j === 0 || j >= 4, align: j === 0 ? "left" : "center",
          color: j === 4 ? (r[6] === 0 ? TEAL : INK) : j === 5 ? (r[6] === 1 ? GOLD : INK) : INK,
          fill: { color: i % 2 ? "FFFFFF" : "F4F7FA" },
        },
      })))),
    {
      x: 0.55, y: 3.9, w: 12.2, colW: [2.75, 1.6, 2.55, 1.9, 1.6, 1.8], rowH: 0.38,
      border: { type: "solid", color: "D8DEE4", pt: 0.75 }, fontFace: F, valign: "middle",
      margin: [0.04, 0.06, 0.04, 0.06],
    });
  s.addText("Diabetes is 89% a head problem — the embedding already supports 0.810. Cancer and MTBLS563 are "
    + "mostly representation. Barth's embedding actually BEATS binned features (0.770 vs 0.705); the head was wasting it.", {
    x: 0.55, y: 6.42, w: 12.2, h: 0.56, fontFace: F, fontSize: 15, color: INK, margin: 0, lineSpacing: 18,
  });
  s.addNotes("The mixed picture is the useful part: there is no single fix, and for one target the "
    + "SSL representation was already better than the classical features.");
}

/* =========================== 12. TWO FREE WINS =========================== */
{
  const s = newLight("Two fixes that cost zero GPU time",
    "Both are PAIRED comparisons — same frozen checkpoint, one thing changed. Both still stand today.");
  fig(s, "gm02_free_wins.png", { x: 0.55, y: 1.50, w: 12.2, h: 4.59 });
  card(s, { x: 0.55, y: 6.20, w: 12.2, h: 0.80, fill: TINT_OK });
  s.addText("Why these two survived everything that came later: they vary a transform on a FIXED "
    + "checkpoint, so they carry no pretraining run-to-run variance at all. That distinction turns out "
    + "to be the whole story of this project.", {
    x: 0.85, y: 6.27, w: 11.6, h: 0.66, fontFace: F, fontSize: 15.5, color: INK, margin: 0, valign: "middle", lineSpacing: 20,
  });
  s.addNotes("Pooling: the backbone makes 128 tokens, one per spectral region. Mean-pooling averages "
    + "them and throws away WHERE the signal was — but chemical shift position is the discriminative "
    + "information in NMR. Keeping position costs nothing.");
}

/* ==================== 13. SCORECARD AFTER FREE FIXES ==================== */
{
  const s = newLight("Where those two fixes left us", "SSL vs classical, after the head and pooling fixes — no retraining");
  const rows = [
    ["Barth", "0.691", "0.806", "0.705", "+0.101", "SSL WINS", GREEN],
    ["MTBLS326", "0.981", "1.000", "1.000", "0.000", "tie", MUTED],
    ["BrC-T2D cancer", "0.796", "0.859", "0.937", "−0.078", "classical", GOLD],
    ["BrC-T2D diabetes", "0.653", "0.783", "0.829", "−0.046", "classical", GOLD],
    ["MTBLS563", "0.558", "0.621", "0.721", "−0.100", "classical", GOLD],
  ];
  s.addTable(
    [["Target", "Reported\nFeb-style head", "Probe +\npooling fix", "Classical\nLogReg", "vs classical", "Verdict"]
      .map(h => ({ text: h, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, fontSize: 15.5, align: "center" } }))]
      .concat(rows.map((r, i) => r.slice(0, 6).map((c, j) => ({
        text: c,
        options: {
          fontSize: 16, bold: j === 0 || j >= 4, align: j === 0 ? "left" : "center",
          color: j >= 4 ? r[6] : INK, fill: { color: i % 2 ? "FFFFFF" : "F4F7FA" },
        },
      })))),
    {
      x: 0.9, y: 1.72, w: 11.5, colW: [2.85, 2.05, 1.95, 1.75, 1.5, 1.4], rowH: 0.45,
      border: { type: "solid", color: "D8DEE4", pt: 0.75 }, fontFace: F, valign: "middle",
      margin: [0.05, 0.08, 0.05, 0.08],
    });

  const stats = [
    ["+0.078", "mean improvement over the\nFebruary-style numbers", DEEP],
    ["0 → 1", "wins against classical\n(was 0 wins / 5 losses)", GREEN],
    ["0 h", "GPU time spent —\nboth fixes are post-hoc", TEAL],
  ];
  stats.forEach(([big, lab, col], i) => {
    const x = 0.9 + i * 3.87;
    card(s, { x, y: 4.72, w: 3.63, h: 1.45, fill: TINT });
    s.addText(big, {
      x: x + 0.15, y: 4.84, w: 3.33, h: 0.58, fontFace: F, fontSize: 30, bold: true, color: col, margin: 0, align: "center",
    });
    s.addText(lab, {
      x: x + 0.15, y: 5.45, w: 3.33, h: 0.62, fontFace: F, fontSize: 14.5, color: INK, margin: 0, align: "center", lineSpacing: 18,
    });
  });
  s.addText("Record moves from 0 wins / 0 ties / 5 losses  →  1 win / 1 tie / 3 losses.", {
    x: 0.9, y: 6.28, w: 11.5, h: 0.4, fontFace: F, fontSize: 17.5, bold: true, color: NAVY, align: "center", margin: 0,
  });
  s.addNotes("Good news slide. But note the two remaining large gaps are the two biggest cohorts — "
    + "the ones with actual error bars.");
}

/* ================= 14. DOES PRETRAINING HELP AT ALL? ================= */
{
  const s = newLight("The control we were missing in February",
    "Same architecture, NO pretrained weights anywhere, classifier held fixed — so any difference is the objective");
  fig(s, "gm03_pretraining_gain.png", { x: 0.55, y: 1.50, w: 12.2, h: 4.59 });
  card(s, { x: 0.55, y: 6.20, w: 5.95, h: 0.80, fill: TINT_OK });
  s.addText("Masked pretraining earns its keep: +0.117 mean, positive on 5 of 5. This is the ONLY "
    + "single-run result in the whole project that clears the noise floor.", {
    x: 0.80, y: 6.26, w: 5.46, h: 0.68, fontFace: F, fontSize: 14.5, color: INK, margin: 0, valign: "middle", lineSpacing: 17,
  });
  card(s, { x: 6.8, y: 6.20, w: 5.95, h: 0.80, fill: TINT_BAD });
  s.addText("Jigsaw and joint add nothing over a RANDOM projection. A random transformer is a strong "
    + "baseline — so this means the objective, not the architecture, is the problem.", {
    x: 7.05, y: 6.26, w: 5.46, h: 0.68, fontFace: F, fontSize: 14.5, color: INK, margin: 0, valign: "middle", lineSpacing: 17,
  });
  s.addNotes("Decision from this: concentrate on masking. We stopped developing jigsaw and joint. "
    + "If asked why random is strong — random-feature kernel methods are a real baseline.");
}

/* ====================== 15. SECTION — MISTAKES ====================== */
{
  const s = newDark(NAVY);
  s.addText("Part 2", {
    x: 1.0, y: 2.0, w: 11.3, h: 0.5, fontFace: F, fontSize: 22, color: "7FA8C4", margin: 0,
  });
  s.addText("Three things we got wrong", {
    x: 1.0, y: 2.55, w: 11.3, h: 1.0, fontFace: F, fontSize: 42, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("and the one experiment that caught them all", {
    x: 1.0, y: 3.6, w: 11.3, h: 0.55, fontFace: F, fontSize: 23, color: "CADCFC", margin: 0, italic: true,
  });
  const ms = [
    ["A control that did not control", "we read an ablation backwards"],
    ["A backwards hypothesis", "finer patches = an easier task"],
    ["An effect that never existed", "and it was our headline result"],
  ];
  ms.forEach(([head, sub], i) => {
    const x = 1.0 + i * 3.8;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 4.66, w: 3.5, h: 1.72, fill: { color: "2E3A73" }, rectRadius: 0.08, line: { type: "none" },
    });
    s.addText(String(i + 1), {
      x: x + 0.2, y: 4.78, w: 0.6, h: 0.42, fontFace: F, fontSize: 23, bold: true, color: "6FD3C7", margin: 0,
    });
    s.addText(head, {
      x: x + 0.2, y: 5.22, w: 3.1, h: 0.62, fontFace: F, fontSize: 16, bold: true, color: "FFFFFF", margin: 0, valign: "top", lineSpacing: 19,
    });
    s.addText(sub, {
      x: x + 0.2, y: 5.88, w: 3.1, h: 0.44, fontFace: F, fontSize: 14, color: "9FC4DA", margin: 0, valign: "top", italic: true,
    });
  });
  s.addNotes("I want to spend real time here — this is the part that changed how I work, and it is "
    + "the part most transferable to everyone else in the group.");
}

/* ==================== 16. MISTAKE 1 — THE XAVIER ABLATION ==================== */
{
  const s = newLight("Mistake 1 — a control that did not control anything",
    "We had an ablation we believed showed 'pretraining does not help'");
  card(s, { x: 0.55, y: 1.65, w: 5.95, h: 2.3, fill: TINT_BAD });
  s.addText("What we thought it tested", {
    x: 0.85, y: 1.82, w: 5.4, h: 0.36, fontFace: F, fontSize: 18.5, bold: true, color: CORAL, margin: 0,
  });
  s.addText("\"Pretrained backbone vs random backbone.\" The numbers looked similar, so the "
    + "reading was: pretraining is not buying us anything.", {
    x: 0.85, y: 2.24, w: 5.4, h: 1.55, fontFace: F, fontSize: 16, color: INK, margin: 0, lineSpacing: 20,
  });
  card(s, { x: 6.8, y: 1.65, w: 5.95, h: 2.3, fill: TINT_OK });
  s.addText("What it actually tested", {
    x: 7.1, y: 1.82, w: 5.4, h: 0.36, fontFace: F, fontSize: 18.5, bold: true, color: GREEN, margin: 0,
  });
  s.addText("It re-initialised only the layers that were being UNFROZEN for fine-tuning, leaving the "
    + "rest pretrained — and the head was underfit in both arms, which masked the difference.", {
    x: 7.1, y: 2.24, w: 5.4, h: 1.55, fontFace: F, fontSize: 16, color: INK, margin: 0, lineSpacing: 20,
  });

  card(s, { x: 0.55, y: 4.2, w: 12.2, h: 1.35, fill: TINT });
  s.addText("The fix", {
    x: 0.85, y: 4.34, w: 2.0, h: 0.36, fontFace: F, fontSize: 18, bold: true, color: DEEP, margin: 0,
  });
  s.addText("Build a genuine control: load NO pretrained weights anywhere (patch embedding and positional "
    + "encoding included), and hold the classifier fixed at a converged LogReg in both arms. That is the "
    + "experiment on the previous slide — and read correctly, masked pretraining HELPS by +0.117.", {
    x: 0.85, y: 4.7, w: 11.6, h: 0.75, fontFace: F, fontSize: 16.5, color: INK, margin: 0, lineSpacing: 20,
  });
  card(s, { x: 0.55, y: 5.70, w: 12.2, h: 1.12, fill: TINT_OK });
  s.addText("Lesson", {
    x: 0.85, y: 5.82, w: 11.6, h: 0.32, fontFace: F, fontSize: 17, bold: true, color: GREEN, margin: 0,
  });
  s.addText("A control has to be checked as carefully as the thing it controls. We now verify explicitly "
    + "that a random-init arm produces genuinely different embeddings before trusting it.", {
    x: 0.85, y: 6.14, w: 11.6, h: 0.56, fontFace: F, fontSize: 16, color: INK, margin: 0, lineSpacing: 19,
  });
  s.addNotes("Worth saying plainly: this one nearly sent us in the wrong direction entirely — we could "
    + "have concluded the whole approach was dead.");
}

/* ==================== 17. MISTAKE 2 — THE PATCH-SIZE IDEA ==================== */
{
  const s = newLight("Mistake 2 — a hypothesis that was exactly backwards",
    "Our best explanation for the representation ceiling, tested properly and refuted");
  card(s, { x: 0.55, y: 1.62, w: 12.2, h: 1.12, fill: TINT });
  s.addText("The hypothesis (very reasonable)", {
    x: 0.85, y: 1.74, w: 5.6, h: 0.34, fontFace: F, fontSize: 17.5, bold: true, color: DEEP, margin: 0,
  });
  s.addText("The backbone turns 131,072 points into 128 tokens, so it cannot represent detail finer than "
    + "128 positions. Classical LogReg uses 1,024 bins — 8× finer. So: shrink the patch, gain resolution.", {
    x: 0.85, y: 2.1, w: 11.6, h: 0.56, fontFace: F, fontSize: 16.5, color: INK, margin: 0, lineSpacing: 20,
  });

  const rows = [
    ["patch 1024 (128 tokens)", "0.806", "1.000", "0.859", "0.780", "0.618", "baseline"],
    ["patch 256 (512 tokens)", "0.598", "0.907", "0.832", "0.783", "0.581", "−0.072"],
    ["patch 128 (1024 tokens)", "0.655", "0.911", "0.768", "0.738", "0.607", "−0.077"],
  ];
  s.addTable(
    [["Backbone", "Barth", "MTBLS326", "BrC cancer", "BrC diab", "MTBLS563", "mean Δ"]
      .map(h => ({ text: h, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, fontSize: 15, align: "center" } }))]
      .concat(rows.map((r, i) => r.map((c, j) => ({
        text: c,
        options: {
          fontSize: 15.5, bold: j === 0 || j === 6, align: j === 0 ? "left" : "center",
          color: j === 6 && i > 0 ? CORAL : INK,
          fill: { color: i === 0 ? "E8F0F5" : "FFFFFF" },
        },
      })))),
    {
      x: 0.55, y: 2.92, w: 12.2, colW: [3.3, 1.4, 1.7, 1.7, 1.5, 1.5, 1.1], rowH: 0.42,
      border: { type: "solid", color: "D8DEE4", pt: 0.75 }, fontFace: F, valign: "middle",
      margin: [0.04, 0.12, 0.04, 0.12],
    });
  s.addText("Finer patches won 0 of 5 targets — both arms were worse.", {
    x: 0.55, y: 4.70, w: 12.2, h: 0.34, fontFace: F, fontSize: 16.5, bold: true, color: CORAL, align: "center", margin: 0,
  });

  card(s, { x: 0.55, y: 5.22, w: 12.2, h: 1.58, fill: TINT_OK });
  s.addText("Why it failed — and this is the interesting part", {
    x: 0.85, y: 5.32, w: 8.0, h: 0.34, fontFace: F, fontSize: 17.5, bold: true, color: GREEN, margin: 0,
  });
  s.addText("Validation reconstruction loss FELL as patches shrank: 9.3×10⁻⁵ → 5.6×10⁻⁵ → 4.4×10⁻⁵. A masked "
    + "128-point patch is largely interpolable from its neighbours, so shrinking the patch made the pretext "
    + "task EASIER, not more informative — the model can solve it by local smoothing without learning any "
    + "metabolite structure. Patch size and mask ratio jointly set task difficulty; we moved the wrong knob.", {
    x: 0.85, y: 5.68, w: 11.6, h: 1.04, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });
  s.addNotes("Also a confound worth admitting: the small-patch models have ~3x fewer parameters, because "
    + "the patch embedding scales with patch size. The falling reconstruction loss is the independent "
    + "evidence for the easier-task explanation.");
}

/* ============ 18. MISTAKE 3 SETUP — THE 'CORPUS EFFECT' ============ */
{
  const s = newLight("Mistake 3 — the big one. An effect that was never there.",
    "How a single number became the largest reported result in the project");
  const steps = [
    ["July", "A new experiment needed a v4-pretrained baseline. We trained one.", DEEP],
    ["The observation", "It scored 0.820 held-out. The v3 checkpoint every earlier number used scored 0.888. Same architecture, same hyperparameters, config verified byte-identical.", TEAL],
    ["The conclusion we drew", "\"The pretraining corpus version is worth +0.069 — larger than every other effect we have measured.\" We even recommended reverting to v3.", CORAL],
    ["What we did next", "Two more experiments (#8, #13) hunting the MECHANISM: was it the 164 differing rows? Was it corpus size? Both came back inconclusive.", GOLD],
  ];
  steps.forEach(([tag, body, col], i) => {
    const y = 1.58 + i * 1.16;
    card(s, { x: 0.55, y, w: 12.2, h: 0.92, fill: i === 2 ? TINT_BAD : TINT });
    bubble(s, { x: 0.78, y: y + 0.18, d: 0.56, color: col, label: String(i + 1), fontSize: 18 });
    s.addText(tag, {
      x: 1.52, y: y + 0.06, w: 2.7, h: 0.36, fontFace: F, fontSize: 17, bold: true, color: col, margin: 0, valign: "middle",
    });
    s.addText(body, {
      x: 4.25, y: y + 0.06, w: 8.25, h: 0.80, fontFace: F, fontSize: 15, color: INK, margin: 0, valign: "middle", lineSpacing: 18,
    });
  });
  card(s, { x: 0.55, y: 6.24, w: 12.2, h: 0.72, fill: NAVY });
  s.addText("Nobody asked the obvious question: how much does the SAME configuration vary between two training runs?", {
    x: 0.85, y: 6.33, w: 11.6, h: 0.56, fontFace: F, fontSize: 16.5, bold: true, color: "FFFFFF", margin: 0, valign: "middle",
  });
  s.addNotes("Pause here. Ask the room what they would have done. The answer we should have reached "
    + "months earlier is: replicate the reference run before believing any of it.");
}

/* ===================== 19. THE SEED STUDY ===================== */
{
  const s = newLight("Experiment #15 — so we finally measured the noise",
    "Ten pretraining runs, five per corpus, identical config differing only by random seed");
  fig(s, "gm04_seed_study.png", { x: 0.55, y: 1.50, w: 12.2, h: 4.59 });
  card(s, { x: 0.55, y: 6.20, w: 12.2, h: 0.80, fill: TINT_BAD });
  s.addText("The 0.888 reference was rank 1 of 5 and sits +1.6 sd above its own distribution. The gap does not "
    + "shrink — it vanishes and marginally reverses. §5f retracted; experiments #8 and #13 were hunting the "
    + "mechanism of a non-effect.", {
    x: 0.85, y: 6.28, w: 11.6, h: 0.66, fontFace: F, fontSize: 15.5, color: INK, margin: 0, valign: "middle", lineSpacing: 17,
  });
  s.addNotes("~45 GPU-hours to learn that our headline result was a sampling artifact. Cheapest "
    + "expensive lesson of the project. Emphasise: this was the student's own suggestion to run seeds.");
}

/* ================ 20. WHAT THE NOISE FLOOR DID ================ */
{
  const s = newLight("The worse news — the noise floor was 2× what we assumed",
    "Every single-run claim in the project, re-scored against the measured spread");
  fig(s, "gm05_recalibration.png", { x: 0.55, y: 1.50, w: 12.2, h: 4.59 });
  card(s, { x: 0.55, y: 6.20, w: 6.05, h: 0.80, fill: TINT_BAD });
  s.addText("Measured sd = 0.045 on the held-out mean; per target up to 0.076 (Barth). We had been using 0.020 — "
    + "which came from three runs whose errors happened to cancel.", {
    x: 0.80, y: 6.26, w: 5.55, h: 0.68, fontFace: F, fontSize: 14.5, color: INK, margin: 0, valign: "middle", lineSpacing: 17,
  });
  card(s, { x: 6.9, y: 6.20, w: 5.85, h: 0.80, fill: TINT_OK });
  s.addText("New standing rule: no single-run difference below 0.09 is an effect. Either ≥5 replicates "
    + "(~23 GPU-h) or a paired comparison — budgeted up front.", {
    x: 7.15, y: 6.26, w: 5.35, h: 0.68, fontFace: F, fontSize: 14.5, color: INK, margin: 0, valign: "middle", lineSpacing: 17,
  });
  s.addNotes("The 0.020 floor is the deeper error: we had explicitly noted at the time that the "
    + "three-run agreement looked like a fluke, called it a fluke in writing, and then adopted it as a "
    + "floor anyway. That is what let three claims through.");
}

/* ============ 21. MISTAKE 4 — PROCESS, AND THE GUARDRAILS ============ */
{
  const s = newLight("A fourth, more mundane failure — and the guardrails",
    "Three wrong numbers in this project came from the same cause");
  card(s, { x: 0.55, y: 1.62, w: 5.95, h: 2.1, fill: TINT_BAD });
  s.addText("Scoring unfinished checkpoints", {
    x: 0.85, y: 1.78, w: 5.4, h: 0.36, fontFace: F, fontSize: 18.5, bold: true, color: CORAL, margin: 0,
  });
  s.addText("Training writes a \"best so far\" checkpoint continuously. Evaluate one mid-run and you get a "
    + "real-looking number for a model that was still training. It happened three times — including once "
    + "when I claimed --seed was broken on GPU, because I compared a mid-training checkpoint against a "
    + "finished one.", {
    x: 0.85, y: 2.18, w: 5.4, h: 1.45, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });
  const guards = [
    ["Provenance stamp", "Checkpoints are written with finished: false and only flipped to true after the training loop exits cleanly."],
    ["Evaluator refuses", "The probe evaluator hard-errors on an unfinished checkpoint and warns on legacy ones with no flag."],
    ["Loud failure on missing data", "Analysis scripts now refuse a results directory missing an expected arm, instead of silently dropping it."],
  ];
  guards.forEach(([head, body], i) => {
    const y = 1.62 + i * 1.42;
    card(s, { x: 6.8, y, w: 5.95, h: 1.26, fill: TINT_OK });
    s.addText(head, {
      x: 7.08, y: y + 0.12, w: 5.4, h: 0.34, fontFace: F, fontSize: 17, bold: true, color: GREEN, margin: 0,
    });
    s.addText(body, {
      x: 7.08, y: y + 0.48, w: 5.42, h: 0.7, fontFace: F, fontSize: 15, color: INK, margin: 0, lineSpacing: 18,
    });
  });
  card(s, { x: 0.55, y: 3.92, w: 5.95, h: 1.86, fill: TINT });
  s.addText("Also worth knowing", {
    x: 0.85, y: 4.06, w: 5.4, h: 0.34, fontFace: F, fontSize: 17, bold: true, color: DEEP, margin: 0,
  });
  s.addText("--seed DOES make training reproducible — two same-seed runs are byte-identical, max|ΔW| = 0. "
    + "That earlier claim is retracted. Reproducibility was never the problem; run-to-run variance across "
    + "DIFFERENT seeds was.", {
    x: 0.85, y: 4.42, w: 5.4, h: 1.28, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });
  s.addText("Every retraction in this project traces to one of two habits: comparing separately-trained "
    + "networks, or trusting a checkpoint we had not verified was finished.", {
    x: 0.55, y: 6.0, w: 12.2, h: 0.7, fontFace: F, fontSize: 17, bold: true, color: NAVY, margin: 0, align: "center", lineSpacing: 21,
  });
  s.addNotes("Mention that all of this is now in a written lab record with the retractions kept in "
    + "place rather than edited out, so the reasoning is auditable.");
}

/* ============= 22. SECTION — THE DECISIVE TEST ============= */
{
  const s = newDark("102C4A");
  s.addText("Part 3", {
    x: 1.0, y: 1.85, w: 11.3, h: 0.5, fontFace: F, fontSize: 22, color: "7FA8C4", margin: 0,
  });
  s.addText("Testing the premise directly", {
    x: 1.0, y: 2.4, w: 11.3, h: 0.95, fontFace: F, fontSize: 42, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("If pretraining helps anywhere, it must help when labels are scarce", {
    x: 1.0, y: 3.42, w: 11.3, h: 0.55, fontFace: F, fontSize: 22, color: "CADCFC", margin: 0, italic: true,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 1.0, y: 4.35, w: 11.3, h: 1.95, fill: { color: "1B3A5C" }, rectRadius: 0.08, line: { type: "none" },
  });
  s.addText("The argument for doing this", {
    x: 1.35, y: 4.52, w: 10.6, h: 0.38, fontFace: F, fontSize: 19, bold: true, color: "6FD3C7", margin: 0,
  });
  s.addText("Everything so far used the FULL dataset. At n=37–113 that is already close to the ceiling of "
    + "what is learnable — the worst possible regime to look for a transfer advantage. Transfer is supposed "
    + "to pay off with few labels. So: sweep the label budget from 2 per class upwards, and find where the "
    + "SSL curve is above the classical one.", {
    x: 1.35, y: 4.95, w: 10.6, h: 1.2, fontFace: F, fontSize: 17, color: "E4EEF7", margin: 0, lineSpacing: 22,
  });
  s.addNotes("This experiment was on the queue since July as #6. It is the one that actually tests what "
    + "we set out to build.");
}

/* ================== 23. FEW-SHOT SETUP + RESULT ================== */
{
  const s = newLight("Experiment #6 — few-shot benchmark, all 5 targets",
    "Every model sees the IDENTICAL support/query draws — 10 episodes per label budget");
  fig(s, "gm06_fewshot_curves.png", { x: 0.55, y: 1.50, w: 12.2, h: 4.62 });
  card(s, { x: 0.55, y: 6.23, w: 12.2, h: 0.77, fill: TINT_BAD });
  s.addText("Masked SSL sits at or below classical in every panel, at every label budget. The premise was "
    + "that SSL wins on the LEFT of these plots — there is no regime where the pretrained backbone wins.", {
    x: 0.85, y: 6.30, w: 11.6, h: 0.64, fontFace: F, fontSize: 15.5, color: INK, margin: 0, valign: "middle", lineSpacing: 19,
  });
  s.addNotes("Note one methodological fix that mattered: the few-shot classifiers still had the OLD "
    + "hardcoded mean-pooling. Left alone, that would have handicapped SSL by 0.03-0.13 in exactly the "
    + "experiment meant to favour it. We ported the pooling fix in first.");
}

/* ================== 24. THE PAIRED ANALYSIS ================== */
{
  const s = newLight("The same data, analysed as a PAIRED comparison",
    "Shared episodes mean the episode-draw variance cancels — far more power than comparing two error bars");
  fig(s, "gm07_fewshot_paired.png", { x: 0.55, y: 1.50, w: 12.2, h: 4.59 });
  card(s, { x: 0.55, y: 6.20, w: 6.05, h: 0.80, fill: TINT_BAD });
  s.addText("Pooled across all five targets at 2 labels/class: +0.0012 ± 0.0162, p = 0.74. Dead even. What "
    + "looked like a low-shot edge is just both methods sitting near chance.", {
    x: 0.80, y: 6.26, w: 5.55, h: 0.68, fontFace: F, fontSize: 14.5, color: INK, margin: 0, valign: "middle", lineSpacing: 17,
  });
  card(s, { x: 6.9, y: 6.20, w: 5.85, h: 0.80, fill: TINT });
  s.addText("And the deficit WIDENS with more labels (negative trend on 4 of 5) — the opposite of what "
    + "transfer learning predicts.", {
    x: 7.15, y: 6.26, w: 5.35, h: 0.68, fontFace: F, fontSize: 14.5, color: INK, margin: 0, valign: "middle", lineSpacing: 16,
  });
  s.addNotes("Explain pairing plainly: for a fixed episode both models saw the same 4 training samples "
    + "and the same test samples, so their difference removes the luck of the draw. Single-episode std is "
    + "0.07-0.15; the paired se is ~0.016.");
}

/* ================== 25. ROBUSTNESS CHECK ================== */
{
  const s = newLight("Before believing a negative result, we tried to break it",
    "The obvious objection, and what happened when we tested it");
  card(s, { x: 0.55, y: 1.68, w: 5.95, h: 2.4, fill: TINT });
  s.addText("The objection", {
    x: 0.85, y: 1.84, w: 5.4, h: 0.36, fontFace: F, fontSize: 18.5, bold: true, color: DEEP, margin: 0,
  });
  s.addText("Our pooling default (regional, G=16) gives 2,048 features. At 2 labels per class that is "
    + "2,048 features for 4 samples. Maybe we handicapped SSL precisely where we claim it has no advantage — "
    + "a lower-dimensional pooling might win.", {
    x: 0.85, y: 2.26, w: 5.4, h: 1.7, fontFace: F, fontSize: 16, color: INK, margin: 0, lineSpacing: 20,
  });

  const rows = [
    ["Barth", "2", "0.609", "0.627", "0.656"],
    ["Barth", "5", "0.564", "0.603", "0.617"],
    ["BrC cancer", "2", "0.543", "0.590", "0.639"],
    ["BrC cancer", "5", "0.597", "0.644", "0.700"],
    ["MTBLS563", "2", "0.338", "0.390", "0.416"],
    ["MTBLS563", "5", "0.338", "0.387", "0.433"],
  ];
  s.addTable(
    [["Target", "labels\n/class", "mean-pool\n(128 feat)", "regional G=4\n(512 feat)", "regional G=16\n(2048 feat)"]
      .map(h => ({ text: h, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, fontSize: 14.5, align: "center" } }))]
      .concat(rows.map((r, i) => r.map((c, j) => ({
        text: c,
        options: {
          fontSize: 15, bold: j === 0 || j === 4, align: j === 0 ? "left" : "center",
          color: j === 4 ? GREEN : INK, fill: { color: i % 2 ? "FFFFFF" : "F4F7FA" },
        },
      })))),
    {
      x: 6.8, y: 1.68, w: 5.95, colW: [1.65, 0.85, 1.15, 1.15, 1.15], rowH: 0.335,
      border: { type: "solid", color: "D8DEE4", pt: 0.75 }, fontFace: F, valign: "middle",
      margin: [0.03, 0.05, 0.03, 0.05],
    });
  card(s, { x: 0.55, y: 4.52, w: 12.2, h: 1.18, fill: TINT_OK });
  s.addText("The check resolves in favour of the conclusion — G=16 wins 6 of 6", {
    x: 0.85, y: 4.63, w: 11.6, h: 0.34, fontFace: F, fontSize: 17.5, bold: true, color: GREEN, margin: 0,
  });
  s.addText("Lower-dimensional poolings are WORSE at every low-support point tested. Even letting SSL pick "
    + "its best pooling post-hoc per point — which inflates SSL — the pooled result is −0.033 ± 0.014, p = 0.006. "
    + "Still negative.", {
    x: 0.85, y: 4.99, w: 11.6, h: 0.64, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });
  card(s, { x: 0.55, y: 5.84, w: 12.2, h: 0.94, fill: TINT });
  s.addText("Also validated: the top of every few-shot curve lands just below that target's known full-data "
    + "value (Barth 0.782 vs 0.705, cancer 0.900 vs 0.937, MTBLS326 0.957 vs 1.000). The harness measures what "
    + "it should.", {
    x: 0.85, y: 5.92, w: 11.6, h: 0.78, fontFace: F, fontSize: 15.5, color: INK, margin: 0, valign: "middle", lineSpacing: 19,
  });
  s.addNotes("I want to be seen trying to break my own negative result. If someone in the room has "
    + "another way it could be an artifact, that is exactly the feedback I want.");
}

/* ================== 26. WHERE WE STAND ================== */
{
  const s = newLight("Where the project stands today", "Everything that survived, and everything that did not");
  const surv = [
    ["Masked pretraining > random init", "+0.117", "2.6 sd, 5/5 targets"],
    ["LogReg probe > trained DNN head", "+0.120", "paired, 5/5"],
    ["Position-preserving pooling", "+0.03…+0.13", "paired, 5/5"],
    ["Same pooling on jigsaw / joint", "+0.079 / +0.049", "paired"],
  ];
  const gone = [
    ["v3 corpus better than v4", "+0.069", "n=5 → −0.014 (0.6 se)"],
    ["\"Backbone scaling exhausted\"", "±0.02", "within noise"],
    ["Patch size 128 / 256 hurt", "−0.04", "within noise"],
    ["Block masking / peak weighting", "−0.03 / +0.01", "within noise"],
    ["\"SSL wins in few-shot\"", "+0.001", "p = 0.74 — refuted"],
  ];
  s.addText("STILL STANDING", {
    x: 0.55, y: 1.62, w: 5.95, h: 0.4, fontFace: F, fontSize: 20, bold: true, color: GREEN, margin: 0,
  });
  s.addText("paired comparisons, plus the one 2 sd result", {
    x: 0.55, y: 2.0, w: 5.95, h: 0.3, fontFace: F, fontSize: 15, color: MUTED, margin: 0, italic: true,
  });
  surv.forEach(([t, d, w], i) => {
    const y = 2.4 + i * 0.94;
    card(s, { x: 0.55, y, w: 5.95, h: 0.82, fill: TINT_OK });
    s.addText(t, { x: 0.8, y: y + 0.08, w: 3.9, h: 0.36, fontFace: F, fontSize: 15.5, color: INK, margin: 0, valign: "middle" });
    s.addText(d, { x: 4.7, y: y + 0.08, w: 1.6, h: 0.36, fontFace: F, fontSize: 15.5, bold: true, color: GREEN, margin: 0, align: "right", valign: "middle" });
    s.addText(w, { x: 0.8, y: y + 0.44, w: 5.5, h: 0.3, fontFace: F, fontSize: 13.5, color: MUTED, margin: 0 });
  });

  s.addText("RETRACTED or WITHIN NOISE", {
    x: 6.8, y: 1.62, w: 5.95, h: 0.4, fontFace: F, fontSize: 20, bold: true, color: CORAL, margin: 0,
  });
  s.addText("everything that rested on a single training run", {
    x: 6.8, y: 2.0, w: 5.95, h: 0.3, fontFace: F, fontSize: 15, color: MUTED, margin: 0, italic: true,
  });
  gone.forEach(([t, d, w], i) => {
    const y = 2.4 + i * 0.755;
    card(s, { x: 6.8, y, w: 5.95, h: 0.65, fill: TINT_BAD });
    s.addText(t, { x: 7.05, y: y + 0.04, w: 3.9, h: 0.32, fontFace: F, fontSize: 15, color: INK, margin: 0, valign: "middle" });
    s.addText(d, { x: 10.95, y: y + 0.04, w: 1.6, h: 0.32, fontFace: F, fontSize: 15, bold: true, color: CORAL, margin: 0, align: "right", valign: "middle" });
    s.addText(w, { x: 7.05, y: y + 0.35, w: 5.5, h: 0.27, fontFace: F, fontSize: 13.5, color: MUTED, margin: 0 });
  });

  card(s, { x: 0.55, y: 6.3, w: 12.2, h: 0.72, fill: NAVY });
  s.addText("The pattern: every surviving result is PAIRED — one fixed checkpoint, one thing varied. Every "
    + "retracted result compared two separately-trained networks.", {
    x: 0.85, y: 6.38, w: 11.6, h: 0.56, fontFace: F, fontSize: 16.5, bold: true, color: "FFFFFF", margin: 0, valign: "middle",
  });
  s.addNotes("If there is one slide to remember, it is this one. The methodological pattern is cleaner "
    + "than any individual result.");
}

/* ================== 27. WHAT I THINK THIS MEANS ================== */
{
  const s = newLight("What I think this means", "Reading the whole picture honestly");
  card(s, { x: 0.55, y: 1.55, w: 12.2, h: 1.42, fill: TINT });
  s.addText("The findings are coherent, not contradictory", {
    x: 0.85, y: 1.66, w: 11.6, h: 0.34, fontFace: F, fontSize: 18, bold: true, color: DEEP, margin: 0,
  });
  s.addText("Masked pretraining DOES learn something real (+0.117 over random init). But it learns less "
    + "discriminative signal than 1,024-bin integrated areas already carry — and reconstruction quality "
    + "anti-correlates with transfer. Meanwhile every axis we pushed on the backbone (patch size, capacity, "
    + "objective, corpus version) came back flat.", {
    x: 0.85, y: 2.02, w: 11.6, h: 0.92, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });

  card(s, { x: 0.55, y: 3.10, w: 5.95, h: 2.06, fill: TINT_BAD });
  s.addText("The most likely cause we have NOT tested", {
    x: 0.85, y: 3.22, w: 5.4, h: 0.58, fontFace: F, fontSize: 17, bold: true, color: CORAL, margin: 0, valign: "top", lineSpacing: 20,
  });
  s.addText("The corpus is 9,670 spectra. That is very small for a foundation model — orders of magnitude "
    + "below where masked pretraining usually starts paying off. Experiment #13 only probed corpus size "
    + "DOWNWARD, which cannot detect an under-training regime.", {
    x: 0.85, y: 3.84, w: 5.4, h: 1.24, fontFace: F, fontSize: 15, color: INK, margin: 0, lineSpacing: 18,
  });
  card(s, { x: 6.8, y: 3.10, w: 5.95, h: 2.06, fill: TINT_OK });
  s.addText("The contribution, reframed", {
    x: 7.1, y: 3.22, w: 5.4, h: 0.36, fontFace: F, fontSize: 17, bold: true, color: GREEN, margin: 0,
  });
  s.addText("This is now a rigorous negative result on 1D NMR SSL: five targets, paired tests, a measured "
    + "noise floor, replicated. The methodology — the 0.045 floor, the paired protocol, the guardrails — is "
    + "arguably a stronger contribution than a backbone that didn't beat logistic regression.", {
    x: 7.1, y: 3.64, w: 5.4, h: 1.44, fontFace: F, fontSize: 15, color: INK, margin: 0, lineSpacing: 18,
  });

  card(s, { x: 0.55, y: 5.32, w: 12.2, h: 1.46, fill: TINT });
  s.addText("What I would NOT recommend", {
    x: 0.85, y: 5.44, w: 11.6, h: 0.34, fontFace: F, fontSize: 17, bold: true, color: NAVY, margin: 0,
  });
  s.addText("More architecture variants. We have tried four patch sizes, two capacities, three objectives and "
    + "two corpora; with the noise floor now known, none of those experiments could have detected an effect "
    + "smaller than 0.09 anyway. Any further single-run backbone tweak is not worth the GPU time.", {
    x: 0.85, y: 5.80, w: 11.6, h: 0.92, fontFace: F, fontSize: 15.5, color: INK, margin: 0, lineSpacing: 19,
  });
  s.addNotes("Be direct here. I would rather bring the group an honest negative with good methodology "
    + "than keep tuning until something crosses a threshold by chance.");
}

/* ================== 28. NEXT STEPS ================== */
{
  const s = newLight("Proposed next steps", "Ranked by what would actually change the answer");
  const items = [
    ["1", GREEN, "Cross-cohort transfer — the strongest untested case",
      "Every evaluation so far is WITHIN a cohort. A foundation model's real selling point is generalising across instruments, sites and batches — exactly where absolute binned intensities should break down and a learned representation might not. Train the probe on one cohort, test on another. Cheap; we have 4 cohorts. If SSL wins anywhere, it is here."],
    ["2", DEEP, "Corpus scaling curve — upward this time",
      "Pretrain at 25 / 50 / 100% of available spectra and see whether downstream utility is still climbing at 9,670. Flat ⇒ the objective is the problem. Rising ⇒ the answer is \"needs 10–100× more data\". Either is publishable and actionable. Needs ≥5 seeds per point (~23 GPU-h each)."],
    ["3", GOLD, "MTBLS326 batch-confound audit",
      "Still outstanding, cheap, and MTBLS326 is one of only two targets where SSL is competitive. Its perfect 1.000 on n=42 is unverified as biology rather than run-order artifact."],
    ["4", TEAL, "Finish jigsaw / joint few-shot arms",
      "Running now. The random-init control already predicts they will be worse than masking, so this is completeness rather than discovery."],
  ];
  items.forEach(([n, col, head, body], i) => {
    const y = 1.55 + i * 1.33;
    card(s, { x: 0.55, y, w: 12.2, h: 1.10, fill: i === 0 ? TINT_OK : TINT });
    bubble(s, { x: 0.78, y: y + 0.3, d: 0.56, color: col, label: n, fontSize: 18 });
    s.addText(head, {
      x: 1.52, y: y + 0.08, w: 10.95, h: 0.36, fontFace: F, fontSize: 17.5, bold: true, color: col, margin: 0, valign: "middle",
    });
    s.addText(body, {
      x: 1.52, y: y + 0.40, w: 10.95, h: 0.66, fontFace: F, fontSize: 14, color: INK, margin: 0, valign: "top", lineSpacing: 17,
    });
  });
  card(s, { x: 0.55, y: 6.82, w: 12.2, h: 0.001, fill: "FFFFFF" });
  s.addNotes("Priority 1 is the one I most want feedback on — it is the last strong argument for the "
    + "backbone, and it is cheap because it needs no retraining.");
}

/* ================== 29. DISCUSSION ================== */
{
  const s = newDark(NAVY);
  s.addText("Discussion", {
    x: 1.0, y: 1.5, w: 11.3, h: 0.95, fontFace: F, fontSize: 42, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("Three things I would like the group's view on", {
    x: 1.0, y: 2.5, w: 11.3, h: 0.5, fontFace: F, fontSize: 21, color: "CADCFC", margin: 0, italic: true,
  });
  const qs = [
    ["Is cross-cohort transfer the right last test?", "It is the one setting where a learned representation should beat absolute binned intensities. If it fails there too, I would call the backbone approach closed for this data scale."],
    ["How much more pretraining data can we realistically get?", "If we cannot get to ~100k spectra, the scaling experiment tells us the ceiling but not how to beat it. Are there cohorts or repositories we are not using?"],
    ["Is a rigorous negative result publishable from this group?", "Five targets, paired protocol, measured noise floor, three self-retractions. I think it is useful to the field — but I would like your read before I write it that way."],
  ];
  qs.forEach(([head, body], i) => {
    const y = 3.14 + i * 1.20;
    s.addShape(pres.ShapeType.roundRect, {
      x: 1.0, y, w: 11.3, h: 1.1, fill: { color: "2E3A73" }, rectRadius: 0.08, line: { type: "none" },
    });
    bubble(s, { x: 1.25, y: y + 0.28, d: 0.55, color: "6FD3C7", label: String(i + 1), fontSize: 18 });
    s.addText(head, {
      x: 1.98, y: y + 0.1, w: 10.1, h: 0.36, fontFace: F, fontSize: 17.5, bold: true, color: "FFFFFF", margin: 0, valign: "middle",
    });
    s.addText(body, {
      x: 1.98, y: y + 0.45, w: 10.1, h: 0.58, fontFace: F, fontSize: 14.5, color: "BBD2E8", margin: 0, valign: "top", lineSpacing: 17,
    });
  });
  s.addText("Thank you — and thanks to Gayatree for the Tirupati TBI samples that started this off.", {
    x: 1.0, y: 6.68, w: 11.3, h: 0.35, fontFace: F, fontSize: 15, color: "7FA8C4", margin: 0, italic: true,
  });
  s.addNotes("Leave plenty of time here. The three questions are genuine — especially #2, since the "
    + "scaling answer depends on data we may not have.");
}

const out = path.join(__dirname, "Group_Meeting_2026-08-10.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out, `(${pageNo} slides)`));
