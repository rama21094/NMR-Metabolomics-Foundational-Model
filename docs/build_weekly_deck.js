// Weekly advisor update deck — covers work since the 2026-07-24 meeting.
// Rebuild:  NODE_PATH=<scratch>/pptxbuild/node_modules node docs/build_weekly_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const REPO = "/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics";
const V4 = "results/plots/all_datasets_summary_v4";
const OLD = "results/plots/all_datasets_summary";
const SAT = "results/analysis/peak_saturation_full_nw/saturation_ratio_summary.png";

// Aspect ratios (w/h), measured from the files — keeps every figure undistorted.
const AR = {
  [`${OLD}/fig1_balanced_accuracy.png`]: 1.59,
  [`${V4}/fig1_balanced_accuracy.png`]: 1.59,
  [`${V4}/fig4_heatmap_all_models.png`]: 1.88,
  [`${V4}/fig6_logreg_advantage_probe.png`]: 0.75,
  [`${V4}/fig7_linear_probe_vs_head.png`]: 2.25,
  [`${V4}/fig8_pretraining_gain.png`]: 2.27,
  [`${V4}/fig9_patch_size_and_pooling.png`]: 2.772,
  [`${V4}/fig10_pooling_sweep.png`]: 2.76,
  [`${V4}/fig11_backbone_scaling.png`]: 2.747,
  [`${V4}/fig12_exp7_factorial.png`]: 2.906,
  [`${V4}/fig13_exp7_replicates.png`]: 2.878,
  [SAT]: 3.36,
  "docs/figures/suppression_train_corpus.png": 0.69,
};

const C = {
  deep: "065A82", teal: "1C7293", mid: "16204A", ink: "1A1A1A",
  muted: "5C6670", card: "F1F5F8", cardWarm: "FBF0E8", line: "D8E0E6",
  good: "1B7F4F", bad: "A32020", warn: "B36A00", white: "FFFFFF",
  softGood: "E8F4EC", softBad: "FBECEC",
  darkCard: "1E2A57", darkCard2: "22305F", darkEdge: "33436F",
  onDark: "CBD6E6", onDarkMute: "9FB2C9",
};
const HEAD = "Cambria", BODY = "Calibri";
const W = 13.3, M = 0.55;
const FLOOR = 6.98; // nothing may extend past this

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Shashank Sharma";
pres.title = "NMR Foundation Models — 15-Day Progress Review";

// ---------- helpers ----------
function titleBar(s, text, kicker) {
  if (kicker) {
    s.addText(kicker, {
      x: M, y: 0.36, w: 11, h: 0.26, margin: 0,
      fontFace: BODY, fontSize: 11.5, bold: true, color: C.teal, charSpacing: 1.2,
    });
  }
  s.addText(text, {
    x: M, y: kicker ? 0.62 : 0.44, w: W - 2 * M, h: 0.55, margin: 0,
    fontFace: HEAD, fontSize: 27, bold: true, color: C.deep, valign: "middle",
  });
}

function caption(s, text, o) {
  s.addText(text, {
    x: o.x, y: o.y, w: o.w, h: o.h, margin: 0,
    fontFace: BODY, fontSize: 9.5, italic: true, color: C.muted,
    valign: "top", lineSpacingMultiple: 0.95,
  });
}

// Places a figure and its caption, and returns the first free y below them.
// Every card row is positioned from this value, so overlap is impossible.
function figBlock(s, rel, o) {
  const ar = AR[rel];
  if (!ar) throw new Error("no aspect ratio recorded for " + rel);
  const h = o.w / ar;
  s.addImage({ path: path.join(REPO, rel), x: o.x, y: o.y, w: o.w, h });
  let next = o.y + h + 0.05;
  if (o.caption) {
    const capH = o.capH || 0.34;
    caption(s, o.caption, { x: o.x, y: next, w: o.w, h: capH });
    next += capH;
  }
  return next + 0.08;
}

function card(s, o) {
  s.addShape(pres.ShapeType.roundRect, {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.06,
    fill: { color: o.fill || C.card },
    line: { color: o.fill || C.card, width: 0.5 },
    shadow: { type: "outer", color: "8899A6", blur: 6, offset: 1, angle: 90, opacity: 0.18 },
  });
}

function noteCard(s, o) {
  card(s, { x: o.x, y: o.y, w: o.w, h: o.h, fill: o.fill });
  let tx = o.x + 0.22;
  if (o.badge) {
    s.addShape(pres.ShapeType.ellipse, {
      x: o.x + 0.2, y: o.y + 0.17, w: 0.4, h: 0.4,
      fill: { color: o.badgeColor || C.deep }, line: { color: o.badgeColor || C.deep, width: 0.5 },
    });
    s.addText(o.badge, {
      x: o.x + 0.2, y: o.y + 0.17, w: 0.4, h: 0.4, margin: 0,
      align: "center", valign: "middle", fontFace: BODY, fontSize: 12, bold: true, color: C.white,
    });
    tx = o.x + 0.72;
  }
  s.addText(o.head, {
    x: tx, y: o.y + 0.16, w: o.w - (tx - o.x) - 0.2, h: 0.42, margin: 0,
    fontFace: BODY, fontSize: o.headSize || 13, bold: true, color: o.headColor || C.ink,
    valign: "middle", lineSpacingMultiple: 0.92,
  });
  if (o.body) {
    s.addText(o.body, {
      x: o.x + 0.22, y: o.y + 0.62, w: o.w - 0.44, h: o.h - 0.76, margin: 0,
      fontFace: BODY, fontSize: o.size || 11.5, color: C.ink, valign: "top",
      lineSpacingMultiple: 1.0,
    });
  }
}

function stat(s, o) {
  card(s, { x: o.x, y: o.y, w: o.w, h: o.h, fill: o.fill || C.card });
  s.addText(o.value, {
    x: o.x, y: o.y + 0.1, w: o.w, h: o.h * 0.46, margin: 0,
    align: "center", valign: "middle", fontFace: HEAD, fontSize: o.size || 32,
    bold: true, color: o.color || C.deep,
  });
  s.addText(o.label, {
    x: o.x + 0.1, y: o.y + o.h * 0.54, w: o.w - 0.2, h: o.h * 0.42, margin: 0,
    align: "center", valign: "top", fontFace: BODY, fontSize: 9.5, color: C.muted,
    lineSpacingMultiple: 0.93,
  });
}

function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i !== items.length - 1 },
  })), {
    x: o.x, y: o.y, w: o.w, h: o.h, margin: 0,
    fontFace: BODY, fontSize: o.size || 12.5, color: o.color || C.ink,
    paraSpaceAfter: o.gap === undefined ? 8 : o.gap, lineSpacingMultiple: 1.02,
    valign: "top",
  });
}

function divider(n, title, sub, points) {
  const s = pres.addSlide();
  s.background = { color: C.mid };
  s.addShape(pres.ShapeType.ellipse, {
    x: M, y: 2.3, w: 1.05, h: 1.05,
    fill: { color: C.teal }, line: { color: C.teal, width: 0.5 },
  });
  s.addText(n, {
    x: M, y: 2.3, w: 1.05, h: 1.05, margin: 0, align: "center", valign: "middle",
    fontFace: HEAD, fontSize: 34, bold: true, color: C.white,
  });
  s.addText(title, {
    x: M + 1.45, y: 2.26, w: 10.6, h: 0.7, margin: 0,
    fontFace: HEAD, fontSize: 33, bold: true, color: C.white, valign: "middle",
  });
  s.addText(sub, {
    x: M + 1.45, y: 3.0, w: 10.4, h: 0.5, margin: 0,
    fontFace: BODY, fontSize: 14.5, color: C.onDark, valign: "top", lineSpacingMultiple: 1.05,
  });
  if (points) {
    bullets(s, points, {
      x: M + 1.45, y: 3.75, w: 10.4, h: 1.5, size: 12.5, gap: 7, color: C.onDarkMute,
    });
  }
  return s;
}

function tbl(s, head, rows, o) {
  const header = head.map((hh, i) => ({
    text: hh,
    options: {
      bold: true, color: C.white, fill: { color: C.deep }, fontSize: o.fs || 11.5,
      align: i === 0 ? "left" : "center", valign: "middle", fontFace: BODY,
    },
  }));
  const body = rows.map((r, ri) => r.map((cellText, ci) => {
    const txt = String(cellText);
    let color = C.ink, bold = false;
    const clean = txt.replace(/\*\*/g, "");
    if (txt.startsWith("**")) bold = true;
    if (o.colorize && ci > 0) {
      if (/^\+/.test(clean)) color = C.good;
      if (/^−/.test(clean)) color = C.bad;
    }
    if (o.tagCol !== undefined && ci === o.tagCol) {
      if (/surv|wins|tie/i.test(clean)) color = C.good;
      else if (/noise/i.test(clean)) color = C.bad;
      else if (/marg/i.test(clean)) color = C.warn;
      else if (/^−/.test(clean)) color = C.bad;
      bold = true;
    }
    return {
      text: clean,
      options: {
        color, bold, fontSize: o.fs || 11.5, fontFace: BODY,
        align: ci === 0 ? "left" : "center", valign: "middle",
        fill: { color: ri % 2 ? "F7FAFC" : C.white },
      },
    };
  }));
  s.addTable([header, ...body], {
    x: o.x, y: o.y, w: o.w, colW: o.colW,
    rowH: o.rowH || 0.3, border: { type: "solid", color: C.line, pt: 0.5 },
    margin: 0.06,
  });
}

// ============================================================
// 1. TITLE
// ============================================================
let s = pres.addSlide();
s.background = { color: C.mid };
s.addText("Comparing Foundation Models with Classical ML", {
  x: M, y: 1.25, w: 11.6, h: 0.4, margin: 0,
  fontFace: BODY, fontSize: 15, bold: true, color: C.teal, charSpacing: 1.4,
});
s.addText("15-Day Progress Review", {
  x: M, y: 1.75, w: 12, h: 0.95, margin: 0,
  fontFace: HEAD, fontSize: 44, bold: true, color: C.white,
});
s.addText("Eight experiments on why classical logistic regression still leads — two cheap wins, six refuted hypotheses, and one measurement result that changes how we read all of them.", {
  x: M, y: 2.82, w: 10.9, h: 0.85, margin: 0,
  fontFace: BODY, fontSize: 15, color: C.onDark, lineSpacingMultiple: 1.1,
});
["8 experiments run", "7 backbones pretrained", "13 figures", "16 commits"].forEach((t, i) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: M + i * 2.28, y: 4.35, w: 2.05, h: 0.5, rectRadius: 0.08,
    fill: { color: C.darkCard }, line: { color: C.darkEdge, width: 1 },
  });
  s.addText(t, {
    x: M + i * 2.28, y: 4.35, w: 2.05, h: 0.5, margin: 0,
    align: "center", valign: "middle", fontFace: BODY, fontSize: 11.5, color: C.onDark,
  });
});
s.addText("Shashank Sharma   ·   Individual meeting, 30 July 2026   ·   covering 24 – 30 July", {
  x: M, y: 6.4, w: 12, h: 0.35, margin: 0,
  fontFace: BODY, fontSize: 12, color: "8A9AB4",
});
s.addNotes("30 min. Plan: 5 min recap + data fix, 10 min diagnosis experiments, 8 min scaling/objective failures, 7 min roadmap assessment and next steps. The most important slide is #18 on run-to-run noise — it changes how we read every other number.");

// ============================================================
// 2. WHERE WE LEFT OFF
// ============================================================
s = pres.addSlide();
titleBar(s, "Where we left off on 24 July", "RECAP");
figBlock(s, `${OLD}/fig1_balanced_accuracy.png`, {
  x: M, y: 1.42, w: 6.0,
  caption: "Presented last meeting: balanced accuracy by model family, pre-cleanup corpus.",
});
noteCard(s, {
  x: 6.9, y: 1.42, w: 5.85, h: 1.66, fill: C.card, badge: "1", badgeColor: C.deep,
  head: "Classical ML set the bar",
  body: "Logistic regression on binned absolute areas beat all three SSL families on every dataset. That result has survived every change since.",
});
noteCard(s, {
  x: 6.9, y: 3.2, w: 5.85, h: 1.66, fill: C.card, badge: "2", badgeColor: C.teal,
  head: "The question you set",
  body: "“ML is better — it sets the bar. Can we better it? And is data the limiting factor?” Parts 1–3 attack the first half; Part 4 the second.",
});
noteCard(s, {
  x: 6.9, y: 4.98, w: 5.85, h: 1.66, fill: C.softBad, badge: "!", badgeColor: C.bad,
  head: "What I did not know then",
  body: "Two of those numbers were measured on data with a suppression bug, and none of them had error bars.",
});
s.addNotes("Recap only, 60 seconds. The bar was set; the question was whether we can beat it. Part 4 returns to the data half of the question.");

// ============================================================
// 3. AGENDA
// ============================================================
s = pres.addSlide();
titleBar(s, "What happened in the last 15 days", "OVERVIEW");
[
  ["1", "Data correctness", "A suppression bug and revised BrC-T2D labels forced a rebuild of every dataset. The benchmark was re-run from scratch.", C.deep],
  ["2", "Why does classical win?", "Decomposed the gap into head-fitting versus representation. Two cheap fixes worked; the resolution hypothesis was refuted.", C.teal],
  ["3", "Scaling and objective", "Four capacity experiments and two pretext-objective changes. All failed — and the last revealed a measurement problem.", C.warn],
  ["4", "Against your roadmap", "Where we sit on your decision tree, and what the peak-intensity data already says about whether more data would help.", C.good],
].forEach((a, i) => {
  noteCard(s, {
    x: M, y: 1.45 + i * 1.36, w: 8.0, h: 1.2,
    badge: a[0], badgeColor: a[3], head: a[1], body: a[2], size: 11,
  });
});
stat(s, { x: 8.9, y: 1.45, w: 3.85, h: 1.5, value: "2 of 8", label: "interventions that improved accuracy", color: C.good });
stat(s, { x: 8.9, y: 3.13, w: 3.85, h: 1.5, value: "+0.078", label: "mean gain from those two — with no retraining", color: C.good });
stat(s, { x: 8.9, y: 4.81, w: 3.85, h: 1.84, value: "0.069", label: "accuracy swing from the pretraining corpus version alone — larger than any model change we tried", color: C.bad, size: 30 });
s.addNotes("Set expectations: most of what I tried did not work, and that is the useful part. The bottom-right number is the headline, built up to in Part 3.");

// ============================================================
// PART 1
// ============================================================
divider("1", "Data correctness first", "Before any modelling conclusion could be trusted, two data problems had to be fixed.", [
  "An EDTA suppression bug that corrupted whole spectra, not just the artefact window",
  "A revised BrC-T2D label file requiring a rebuilt dataset and re-extraction",
]).addNotes("Two minutes. This is why the numbers moved between meetings.");

// ============================================================
// 4. THE SUPPRESSION BUG
// ============================================================
s = pres.addSlide();
titleBar(s, "The suppression bug corrupted entire spectra", "PART 1 · DATA");
figBlock(s, "docs/figures/suppression_train_corpus.png", {
  x: 9.5, y: 1.4, w: 3.25,
  caption: "Corpus-wide suppression audit after the fix.",
});
noteCard(s, {
  x: M, y: 1.4, w: 8.6, h: 1.6, fill: C.softBad, badge: "✗", badgeColor: C.bad,
  head: "Why one unsuppressed peak ruins the whole row",
  body: "Spectra are row-wise min-max normalised. If an EDTA artefact survives suppression and becomes the row maximum, every real peak in that spectrum is compressed toward zero. The damage is global, not local to the artefact window.",
});
noteCard(s, {
  x: M, y: 3.14, w: 8.6, h: 1.6, fill: C.card, badge: "✓", badgeColor: C.good,
  head: "The fix, and how it was verified",
  body: "The suppression cutoff is now capped at the baseline-to-peak midpoint, and detection is magnitude-based and metadata-free. Verification uses a flatness test — “was a hard mask applied” — instead of the peak-above-noise test that had been silently passing.",
});
noteCard(s, {
  x: M, y: 4.88, w: 8.6, h: 1.85, fill: C.card, badge: "→", badgeColor: C.teal,
  head: "Result: dataset version v4",
  body: "All five evaluation targets and the 9,670-spectrum corpus were rebuilt. Revised BrC-T2D labels were extracted and byte-verified against the source array: 4 samples genuinely changed label, and my first report of 41 changes was my own display artefact, which I caught by comparing all 78 rows. 7 rows repo-wide still retain a dominant EDTA peak — documented.",
});
s.addNotes("Key insight: row-wise normalisation turns a local artefact into a global problem. Also own the mistake — my first count of changed labels was wrong and I caught it by byte-comparing all 78 rows.");

// ============================================================
// 5. UPDATED BENCHMARK
// ============================================================
s = pres.addSlide();
titleBar(s, "Re-run on clean data: the bar is unchanged", "PART 1 · BENCHMARK");
figBlock(s, `${V4}/fig1_balanced_accuracy.png`, {
  x: M, y: 1.4, w: 7.1,
  caption: "Balanced accuracy, best fine-tune mode per family, v4 data with revised labels.",
});
card(s, { x: 8.05, y: 1.4, w: 4.7, h: 3.05 });
s.addText("Classical logistic regression leads on all five targets", {
  x: 8.27, y: 1.55, w: 4.26, h: 0.55, margin: 0,
  fontFace: BODY, fontSize: 13.5, bold: true, color: C.ink, lineSpacingMultiple: 0.95,
});
tbl(s, ["Target", "Classical", "Best SSL"], [
  ["Barth", "0.705", "0.691"],
  ["MTBLS326", "1.000", "0.981"],
  ["MTBLS563", "0.721", "0.558"],
  ["BrC-T2D cancer", "0.937", "0.796"],
  ["BrC-T2D diabetes", "0.829", "0.653"],
], { x: 8.27, y: 2.16, w: 4.26, colW: [1.86, 1.2, 1.2], rowH: 0.24, fs: 10 });

noteCard(s, {
  x: 8.05, y: 4.62, w: 4.7, h: 1.2, fill: C.card, badge: "≡", badgeColor: C.teal,
  head: "Both tracks are linear",
  body: "The SSL head is also one linear layer on a pooled embedding, so this isolates the representation.",
  size: 11,
});
noteCard(s, {
  x: 8.05, y: 5.96, w: 4.7, h: 1.02, fill: C.softGood, badge: "▸", badgeColor: C.good,
  head: "Ordering: masked > jigsaw ≈ joint",
  body: "Held through every later experiment.",
  size: 11,
});
s.addNotes("The headline did not change after cleaning. Emphasise that both sides are linear classifiers, so this is a statement about representations, not classifier class.");

// ============================================================
// 6. HEATMAP
// ============================================================
s = pres.addSlide();
titleBar(s, "Every model × every fine-tuning mode", "PART 1 · BENCHMARK");
figBlock(s, `${V4}/fig4_heatmap_all_models.png`, {
  x: 2.0, y: 1.32, w: 9.3, capH: 0.5,
  caption: "Frozen backbone versus unfreezing up to three layers, all families, all five targets. Degenerate cells — a model collapsing to a single predicted class — are now flagged live from the confusion matrix rather than from a stale hardcoded list.",
});
s.addNotes("Do not read every cell. Two points: fine-tuning depth helps the masking family, and several joint/jigsaw cells collapse to single-class prediction, which is why degenerate cells are now auto-flagged.");

// ============================================================
// PART 2
// ============================================================
divider("2", "Why does classical ML win?", "If both classifiers are linear, the gap comes from either the head or the representation. So I measured which.", [
  "Experiment #1 — decompose the gap: head-fitting deficit versus representation ceiling",
  "Experiments #2, #3, #4 — fix the head, test whether pretraining does anything, test the resolution hypothesis",
]).addNotes("The analytical core of the two weeks. About 10 minutes.");

// ============================================================
// 7. GAP DECOMPOSITION
// ============================================================
s = pres.addSlide();
titleBar(s, "Decomposing the gap: head versus representation", "PART 2 · EXPERIMENT #1");
figBlock(s, `${V4}/fig6_logreg_advantage_probe.png`, {
  x: 9.15, y: 1.38, w: 3.6,
  caption: "Bin-resolution sweep and permutation nulls.",
});
noteCard(s, {
  x: M, y: 1.38, w: 8.3, h: 1.5, fill: C.card, badge: "?", badgeColor: C.deep,
  head: "The method",
  body: "Fit the same converged logistic regression on (a) binned absolute areas at many bin counts, (b) the raw spectrum, and (c) the frozen SSL embedding — under each dataset's real CV protocol, with a 200-draw label-permutation null.",
});
tbl(s, ["Target", "Where the gap lives", "Read"], [
  ["BrC-T2D diabetes", "89% head-fitting", "fixable"],
  ["BrC-T2D cancer", "~70–75% representation", "hard"],
  ["MTBLS563", "~70–75% representation", "hard"],
  ["Barth", "embedding 0.770 beats binned 0.705", "already ahead"],
], { x: M, y: 3.02, w: 8.3, colW: [2.5, 3.7, 2.1], rowH: 0.32, fs: 11 });

noteCard(s, {
  x: M, y: 4.92, w: 8.3, h: 1.6, fill: C.softGood, badge: "✓", badgeColor: C.good,
  head: "Why this mattered",
  body: "It split one vague question into two testable ones, and predicted which datasets a fix would flip. Barth was the tell: there the representation was never the problem, so a head fix alone should win it — which is exactly what happened two experiments later.",
});
s.addNotes("The most useful analysis of the two weeks: it turned 'SSL is worse' into two separately measurable deficits, and correctly predicted where the fixes would land.");

// ============================================================
// 8. EXPERIMENT #2
// ============================================================
s = pres.addSlide();
titleBar(s, "Experiment #2: the masking head was underfit", "PART 2 · EXPERIMENT #2");
figBlock(s, `${V4}/fig7_linear_probe_vs_head.png`, {
  x: M, y: 1.4, w: 8.25,
  caption: "Identical frozen features, two ways of fitting the same linear map.",
});
stat(s, { x: 9.05, y: 1.4, w: 3.7, h: 1.5, value: "+0.120", label: "mean gain, masking family", color: C.good });
stat(s, { x: 9.05, y: 3.02, w: 3.7, h: 1.5, value: "5 of 5", label: "targets improved (+0.077 to +0.156)", color: C.good });
noteCard(s, {
  x: 9.05, y: 4.64, w: 3.7, h: 2.0, fill: C.card, badge: "!", badgeColor: C.teal,
  head: "Not a modelling change",
  body: "The head was always a linear layer. Replacing ~50 epochs of Adam on 40 samples with a converged L2 logistic regression is free. Jigsaw and joint were unaffected (+0.009).",
  size: 11,
});
s.addNotes("Cheap win one. Nothing about the model changed — only how the final linear layer was fitted. A pure evaluation-methodology gain.");

// ============================================================
// 9. EXPERIMENT #3
// ============================================================
s = pres.addSlide();
titleBar(s, "Experiment #3: does pretraining contribute anything?", "PART 2 · EXPERIMENT #3");
figBlock(s, `${V4}/fig8_pretraining_gain.png`, {
  x: M, y: 1.4, w: 8.25,
  caption: "Pretrained versus a true random-initialisation control, head held fixed at a converged probe.",
});
noteCard(s, {
  x: 9.05, y: 1.4, w: 3.7, h: 1.28, fill: C.softGood, badge: "✓", badgeColor: C.good,
  head: "Masked: +0.117 on 5 of 5",
  body: "Masked pretraining genuinely works.",
  size: 11,
});
noteCard(s, {
  x: 9.05, y: 2.82, w: 3.7, h: 1.42, fill: C.softBad, badge: "✗", badgeColor: C.bad,
  head: "Jigsaw −0.011 · Joint −0.025",
  body: "A random joint backbone scores 0.846 on cancer versus 0.769 pretrained.",
  size: 11,
});
noteCard(s, {
  x: 9.05, y: 4.38, w: 3.7, h: 2.26, fill: C.cardWarm, badge: "⟲", badgeColor: C.warn,
  head: "A correction I had to make",
  body: "I first read an earlier ablation as “pretraining contributes nothing”. That flag only reinitialises the unfrozen layers — patch embedding and positional encoding stay pretrained — so it never tested the claim. This control loads no pretrained weights anywhere.",
  size: 11,
});
s.addNotes("Two messages: masked pretraining is real and worth keeping; jigsaw and joint are not earning their keep. Also own the earlier misreading — the control had to be rebuilt to actually answer the question.");

// ============================================================
// 10. EXPERIMENT #4
// ============================================================
s = pres.addSlide();
titleBar(s, "Experiment #4: the resolution hypothesis, refuted", "PART 2 · EXPERIMENT #4");
let y = figBlock(s, `${V4}/fig9_patch_size_and_pooling.png`, {
  x: 1.3, y: 1.3, w: 10.7,
  caption: "Left: shrinking the patch does not lift the ceiling — it lowers it. Right: what actually did work.",
});
noteCard(s, {
  x: M, y, w: 6.05, h: FLOOR - y, fill: C.softBad, badge: "✗", badgeColor: C.bad,
  head: "Prediction: finer patches → better. Wrong.",
  body: "Patch 256 −0.072, patch 128 −0.077. Zero wins out of five targets.",
  size: 11,
});
noteCard(s, {
  x: 6.85, y, w: 5.9, h: FLOOR - y, fill: C.cardWarm, badge: "∴", badgeColor: C.warn,
  head: "The mechanism — and my reasoning error",
  body: "Reconstruction loss FELL as patches shrank (9.3e-5 → 4.4e-5): a masked 128-point patch is interpolable from its neighbours. I asked what the encoder could represent, not whether the task still forced it to learn.",
  size: 11,
});
s.addNotes("The most instructive failure: the pretext task got easier, not the representation better. That diagnosis is what motivated experiment #7.");

// ============================================================
// 11. POOLING
// ============================================================
s = pres.addSlide();
titleBar(s, "The one intervention that clearly worked: pooling", "PART 2 · THE REAL WIN");
y = figBlock(s, `${V4}/fig10_pooling_sweep.png`, {
  x: 1.3, y: 1.3, w: 10.7,
  caption: "Regional pooling sweep: G contiguous token groups. G = 1 is mean-pooling, G = 128 is a fully flattened embedding.",
});
noteCard(s, {
  x: M, y, w: 4.0, h: FLOOR - y, fill: C.softGood, badge: "✓", badgeColor: C.good,
  head: "Mean-pooling threw away position",
  body: "Chemical-shift position IS the signal in NMR. Averaging tokens destroys it: +0.030 to +0.129 recovered, on 5 of 5.",
  size: 11,
});
noteCard(s, {
  x: 4.75, y, w: 4.0, h: FLOOR - y, fill: C.card, badge: "◆", badgeColor: C.teal,
  head: "The optimum is in between",
  body: "G = 16 matches a full flatten with 2,048 features instead of 16,384, and beats both extremes on 4 of 5.",
  size: 11,
});
noteCard(s, {
  x: 8.95, y, w: 3.8, h: FLOOR - y, fill: C.card, badge: "★", badgeColor: C.deep,
  head: "Why this one is trustworthy",
  body: "It is measured inside a fixed checkpoint, so it is paired — immune to the run-to-run noise in Part 3.",
  size: 11,
});
s.addNotes("Cheap win two, and the most robust positive result in the project because it is a paired comparison within one checkpoint. Flag that now — it pays off in Part 3.");

// ============================================================
// 12. SCOREBOARD
// ============================================================
s = pres.addSlide();
titleBar(s, "Both cheap wins combined — no retraining required", "PART 2 · WHERE THAT LEAVES US");
tbl(s, ["Target", "Originally reported", "Probe + flatten pooling", "Classical", "Verdict"], [
  ["Barth", "0.691", "**0.806", "0.705", "SSL wins +0.101"],
  ["MTBLS326", "0.981", "**1.000", "1.000", "tie"],
  ["BrC-T2D cancer", "0.796", "0.859", "**0.937", "−0.078"],
  ["BrC-T2D diabetes", "0.653", "0.783", "**0.829", "−0.046"],
  ["MTBLS563", "0.558", "0.621", "**0.721", "−0.100"],
], { x: M, y: 1.45, w: 12.2, colW: [2.9, 2.4, 2.7, 1.9, 2.3], rowH: 0.4, fs: 12, tagCol: 4 });

stat(s, { x: M, y: 4.15, w: 3.9, h: 1.45, value: "+0.078", label: "mean gain over originally reported numbers", color: C.good });
stat(s, { x: 4.7, y: 4.15, w: 3.9, h: 1.45, value: "1 W · 1 T · 3 L", label: "record versus classical, up from 0 W · 0 T · 5 L", color: C.deep, size: 25 });
stat(s, { x: 8.85, y: 4.15, w: 3.9, h: 1.45, value: "0 GPU-hours", label: "both fixes are evaluation-side only", color: C.teal, size: 26 });
noteCard(s, {
  x: M, y: 5.78, w: 12.2, h: 1.2, fill: C.cardWarm, badge: "⚠", badgeColor: C.warn,
  head: "A caveat I want to be explicit about",
  body: "Pooling G was chosen by inspecting these same datasets. From here on, configuration choices are made on a pre-committed selection subset (MTBLS563 + diabetes) and reported on the held-out three.",
  size: 11,
});
s.addNotes("The good-news slide. I flag the selection-bias caveat myself rather than have it asked — from here on I use a pre-committed selection subset.");

// ============================================================
// PART 3
// ============================================================
divider("3", "Scaling and the pretext objective", "Six more interventions. All failed — and the last exposed a problem with how we have been measuring everything.", [
  "Four capacity experiments: patch 128 / 256 / 2048, and a 2.7× wider-and-deeper backbone",
  "Experiment #7: block masking and peak-weighted reconstruction, as a 2×2 factorial",
  "Experiment #7b: replicates — and a noise floor that invalidates several earlier claims",
]).addNotes("Eight minutes. The arc: I ran out of model-side ideas, then found my measurements were noisier than the effects I was chasing.");

// ============================================================
// 13. SCALING
// ============================================================
s = pres.addSlide();
titleBar(s, "Scaling the backbone: four attempts, no gain", "PART 3 · CAPACITY");
y = figBlock(s, `${V4}/fig11_backbone_scaling.png`, {
  x: 1.3, y: 1.3, w: 10.7,
  caption: "Left: no new backbone beats the original 1.89M-parameter model at either pooling. Right: reconstruction loss does not predict transfer.",
});
noteCard(s, {
  x: M, y, w: 3.9, h: FLOOR - y, fill: C.card, badge: "◆", badgeColor: C.teal,
  head: "Capacity is not the bottleneck",
  body: "A 5.42M-parameter model, 2.9× the baseline, still lost. Capacity papers over bad pooling rather than fixing it.",
  size: 11,
});
noteCard(s, {
  x: 4.65, y, w: 3.9, h: FLOOR - y, fill: C.softBad, badge: "✗", badgeColor: C.bad,
  head: "Reconstruction loss misleads",
  body: "Spearman with held-out accuracy is +0.60 — the wrong sign. Rule adopted: never select on reconstruction loss.",
  size: 11,
});
noteCard(s, {
  x: 8.75, y, w: 4.0, h: FLOOR - y, fill: C.cardWarm, badge: "⟲", badgeColor: C.warn,
  head: "This slide is partly withdrawn",
  body: "Its reference is a v3-pretrained checkpoint while the comparisons are v4. Two slides on, that turns out to matter more than the effect.",
  size: 11,
});
s.addNotes("Present as it stood at the time, then flag that I later withdrew the strong conclusion. The honest position is that these backbones are indistinguishable.");

// ============================================================
// 14. EXPERIMENT #7
// ============================================================
s = pres.addSlide();
titleBar(s, "Experiment #7: a harder, better-aligned pretext task", "PART 3 · OBJECTIVE");
y = figBlock(s, `${V4}/fig12_exp7_factorial.png`, {
  x: 1.0, y: 1.3, w: 11.3,
  caption: "2×2 factorial. Identical geometry and corpus; only the pretext task differs. All four arms early-stopped normally.",
});
noteCard(s, {
  x: M, y, w: 3.9, h: FLOOR - y, fill: C.card, badge: "A", badgeColor: C.deep,
  head: "Block masking: −0.030",
  body: "Contiguous 8-patch spans, so neighbour interpolation cannot solve it. It DID make the task 41% harder — and still lost.",
  size: 11,
});
noteCard(s, {
  x: 4.65, y, w: 3.9, h: FLOOR - y, fill: C.card, badge: "B", badgeColor: C.teal,
  head: "Peak weighting: +0.011",
  body: "Loss restricted to the top 25% of patches by magnitude. The best arm here — but it reverses on the next slide.",
  size: 11,
});
noteCard(s, {
  x: 8.75, y, w: 4.0, h: FLOOR - y, fill: C.softBad, badge: "✗", badgeColor: C.bad,
  head: "No arm reaches classical",
  body: "Best held-out mean 0.834 against 0.881. The factorial's real payoff was its reference arm, which exposed the confound overleaf.",
  size: 11,
});
s.addNotes("Emphasise block masking: the mechanism provably fired — validation loss rose 41%, so the task really was harder — and transfer still dropped. A genuine negative result about MAE-style objectives on spectra.");

// ============================================================
// 15. EXPERIMENT #7b — HEADLINE
// ============================================================
s = pres.addSlide();
titleBar(s, "The corpus matters more than the model", "PART 3 · EXPERIMENT #7b");
y = figBlock(s, `${V4}/fig13_exp7_replicates.png`, {
  x: 1.0, y: 1.3, w: 11.3,
  caption: "Left: three independent runs of one configuration. Middle: peak weighting judged against a matched corpus. Right: every claim in the record against the measured noise floor.",
});
noteCard(s, {
  x: M, y, w: 3.9, h: FLOOR - y, fill: C.softBad, badge: "1", badgeColor: C.bad,
  head: "v3 → v4 cost us 0.069",
  body: "Three replicates on v4 land at 0.820 / 0.823 / 0.816. The v3 reference is 0.888 — above the whole cluster. Our EDTA “fix” hurt transfer.",
  size: 11,
});
noteCard(s, {
  x: 4.65, y, w: 3.9, h: FLOOR - y, fill: C.softBad, badge: "2", badgeColor: C.bad,
  head: "Peak weighting actually loses",
  body: "Matched on v3 it scores −0.039 held-out and −0.142 on diabetes. Its +0.011 was an artefact of a corpus-depressed baseline.",
  size: 11,
});
noteCard(s, {
  x: 8.75, y, w: 4.0, h: FLOOR - y, fill: C.cardWarm, badge: "3", badgeColor: C.warn,
  head: "Noise floor: 0.020",
  body: "Per-target run-to-run SD is 0.035. Seeding does not fix it — cuDNN autotuning and mixed precision make GPU training nondeterministic.",
  size: 11,
});
s.addNotes("The most important slide. Three minutes. Two takeaways: our data-cleaning decision has a measurable cost and should be revisited; and we have been reading single-run differences smaller than our own measurement noise.");

// ============================================================
// 16. RECALIBRATION
// ============================================================
s = pres.addSlide();
titleBar(s, "Re-reading every claim against the noise floor", "PART 3 · RECALIBRATION");
tbl(s, ["Claim", "Effect", "vs 0.020 floor", "Status"], [
  ["Pretraining corpus v3 vs v4", "+0.069", "3.4×", "survives"],
  ["Patch 128 vs 1024", "−0.042", "2.1×", "survives"],
  ["Peak weighting (matched corpus)", "−0.039", "2.0×", "survives"],
  ["Patch 256 vs 1024", "−0.034", "1.7×", "marginal"],
  ["Block masking", "−0.030", "1.5×", "marginal"],
  ["Patch 2048 vs 1024", "+0.020", "1.0×", "within noise"],
  ["Peak weighting (unmatched)", "+0.011", "0.5×", "within noise"],
  ["Wider / deeper backbone", "+0.006", "0.3×", "within noise"],
], { x: M, y: 1.4, w: 7.9, colW: [3.5, 1.5, 1.4, 1.5], rowH: 0.335, fs: 11, tagCol: 3, colorize: true });

noteCard(s, {
  x: 8.65, y: 1.4, w: 4.1, h: 1.72, fill: C.softBad, badge: "⟲", badgeColor: C.bad,
  head: "What I have withdrawn",
  body: "“Backbone scaling is exhausted” is not supported. Patch 1024, patch 2048 and the wider model are indistinguishable on one run each.",
  size: 11,
});
noteCard(s, {
  x: 8.65, y: 3.28, w: 4.1, h: 1.72, fill: C.softGood, badge: "✓", badgeColor: C.good,
  head: "What is unaffected",
  body: "The head fix and the pooling win are paired within-checkpoint comparisons, so run-to-run noise cannot explain them. They stand.",
  size: 11,
});
noteCard(s, {
  x: 8.65, y: 5.16, w: 4.1, h: 1.72, fill: C.card, badge: "→", badgeColor: C.teal,
  head: "Standing rule adopted",
  body: "No single-run difference below 0.04 is reported as an effect. Either three or more replicates per arm, or a paired comparison.",
  size: 11,
});
s.addNotes("The uncomfortable slide. Three of eight previously-reported effects are within noise. Better I find this than a reviewer. Note the two positive results survive because they are paired.");

// ============================================================
// PART 4
// ============================================================
divider("4", "Where we are on your roadmap", "Mapping fifteen days of model-side work onto the decision tree you laid out — and what our own data already says about the data question.", [
  "The bar is set, and “can we better it” has been pursued hard: two wins, six failures",
  "The data question already has an answer we have not been using",
]).addNotes("Seven minutes. This is the part I most want your input on.");

// ============================================================
// 17. THE DECISION TREE
// ============================================================
s = pres.addSlide();
titleBar(s, "Your decision tree, with status", "PART 4 · ROADMAP");
const nodes = [
  { x: M, y: 1.3, w: 3.85, h: 1.16, t: "Initial trials — masking, jigsaw, joint", st: "DONE", c: C.good,
    d: "All three pretrained and benchmarked on 5 targets." },
  { x: M, y: 2.58, w: 3.85, h: 1.16, t: "ML is better — sets the bar", st: "DONE", c: C.good,
    d: "Classical LR leads 5/5, confirmed on both v3 and v4." },
  { x: M, y: 3.86, w: 3.85, h: 1.42, t: "Can we better it?", st: "PURSUED HARD", c: C.warn,
    d: "8 interventions: 2 worked (+0.078, evaluation-side), 6 failed. Record now 1 win / 1 tie / 3 losses." },
  { x: M, y: 5.4, w: 3.85, h: 1.58, t: "Is data limiting?", st: "PARTIAL ANSWER", c: C.teal,
    d: "Peak-intensity distributions already measured — next slide. Quantity is not the constraint; version and quality are." },
  { x: 4.72, y: 1.3, w: 3.9, h: 1.72, t: "Peak-intensity distributions vs increasing data", st: "MEASURED", c: C.good,
    d: "60 canonical peaks across 9,670 spectra. Distributions converge at a median of 364 spectra." },
  { x: 4.72, y: 3.14, w: 3.9, h: 1.72, t: "Experimental y-distribution defined at all x?", st: "PARTIAL", c: C.warn,
    d: "Yes at the 60 detected peaks; not across all 131,072 points. This is the open gate." },
  { x: 4.72, y: 4.98, w: 3.9, h: 2.0, t: "Use synthetic data (generative, or metabolite sums)", st: "NOT STARTED", c: C.bad,
    d: "A VAE (60 epochs) and a diffusion U-Net with sampler exist and are smoke-tested, but nothing generated has entered pretraining or evaluation." },
  { x: 8.72, y: 1.3, w: 4.03, h: 1.72, t: "Does synthetic data help?", st: "UNTESTED", c: C.bad,
    d: "The decisive experiment in your tree — and the one we have not run." },
  { x: 8.72, y: 3.14, w: 4.03, h: 1.72, t: "If no → more parameters", st: "ALREADY BURNED", c: C.warn,
    d: "Tested out of order. Four capacity experiments, all at or below the noise floor." },
  { x: 8.72, y: 4.98, w: 4.03, h: 2.0, t: "If no → we have hit a limit on these tasks", st: "PREMATURE", c: C.teal,
    d: "Not concludable yet: the synthetic branch is untried, and n = 37–113 against a 0.035 noise floor may simply be too small to resolve differences." },
];
nodes.forEach((n) => {
  card(s, { x: n.x, y: n.y, w: n.w, h: n.h, fill: C.card });
  s.addShape(pres.ShapeType.roundRect, {
    x: n.x + n.w - 1.72, y: n.y + 0.13, w: 1.55, h: 0.23, rectRadius: 0.1,
    fill: { color: n.c }, line: { color: n.c, width: 0.5 },
  });
  s.addText(n.st, {
    x: n.x + n.w - 1.72, y: n.y + 0.13, w: 1.55, h: 0.23, margin: 0,
    align: "center", valign: "middle", fontFace: BODY, fontSize: 7.5, bold: true, color: C.white,
  });
  s.addText(n.t, {
    x: n.x + 0.15, y: n.y + 0.11, w: n.w - 1.95, h: 0.5, margin: 0,
    fontFace: BODY, fontSize: 10.5, bold: true, color: C.ink, valign: "top",
    lineSpacingMultiple: 0.9,
  });
  s.addText(n.d, {
    x: n.x + 0.15, y: n.y + 0.64, w: n.w - 0.3, h: n.h - 0.76, margin: 0,
    fontFace: BODY, fontSize: 9.5, color: C.muted, valign: "top", lineSpacingMultiple: 0.92,
  });
});
s.addNotes("Walk the left column top to bottom, then middle, then right. Honest summary: fifteen days were spent entirely in the left column, and the middle and right columns are where your tree says the decisive work is.");

// ============================================================
// 18. PEAK SATURATION
// ============================================================
s = pres.addSlide();
titleBar(s, "“Is data limiting?” — what our own corpus says", "PART 4 · THE DATA QUESTION");
y = figBlock(s, SAT, {
  x: 1.75, y: 1.3, w: 9.8,
  caption: "KS saturation ratio per canonical peak: the fraction of the corpus needed before that peak's intensity distribution stops changing. All 60 peaks sit below 0.30; none is unsaturated.",
});
stat(s, { x: M, y, w: 2.95, h: 1.2, value: "364", label: "median spectra to converge, of 9,670", color: C.deep, size: 27 });
stat(s, { x: 3.75, y, w: 2.95, h: 1.2, value: "60 / 60", label: "peaks already under 5% relative SEM", color: C.good, size: 25 });
stat(s, { x: 6.95, y, w: 2.95, h: 1.2, value: "0 / 60", label: "peaks needing more than 9,670", color: C.good, size: 25 });
stat(s, { x: 10.15, y, w: 2.6, h: 1.2, value: "9%", label: "of the corpus is enough", color: C.teal, size: 27 });

const y2 = y + 1.32;
noteCard(s, {
  x: M, y: y2, w: 6.05, h: FLOOR - y2, fill: C.softGood, badge: "→", badgeColor: C.good,
  head: "More of the same data will not help",
  body: "Saturated by a wide margin — more public spectra of this kind should not change them.",
  size: 10.5,
});
noteCard(s, {
  x: 6.85, y: y2, w: 5.9, h: FLOOR - y2, fill: C.cardWarm, badge: "⚠", badgeColor: C.warn,
  head: "Two caveats before we lean on it",
  body: "Covers 60 detected peaks, not all 131,072 points — and was computed on the pre-v3 corpus.",
  size: 10.5,
});
s.addNotes("The bridge from my model-side work to your data question. For quantity the answer looks like no. But the corpus-version result says quality matters a lot — a different and more interesting problem.");

// ============================================================
// 19. THE SYNTHETIC GATE
// ============================================================
s = pres.addSlide();
titleBar(s, "The synthetic-data branch: what exists, what is missing", "PART 4 · NEXT GATE");
card(s, { x: M, y: 1.4, w: 6.05, h: 4.75, fill: C.softGood });
s.addText("Already built", {
  x: M + 0.25, y: 1.55, w: 5.5, h: 0.32, margin: 0,
  fontFace: BODY, fontSize: 14.5, bold: true, color: C.good,
});
bullets(s, [
  "1D VAE (encoder / decoder / loss) trained 60 epochs, with reconstruction previews every 5 epochs",
  "Diffusion U-Net plus a Gaussian diffusion process, written and smoke-tested end to end",
  "generate_spectra.py sampling CLI",
  "A suppression-aware dataset loader, so generated spectra can respect the same artefact windows",
  "Peak extraction and Needleman–Wunsch peak-list alignment, giving canonical positions to condition on",
], { x: M + 0.25, y: 1.98, w: 5.55, h: 4.0, size: 12, gap: 9 });

card(s, { x: 6.85, y: 1.4, w: 5.9, h: 4.75, fill: C.softBad });
s.addText("Missing before the gate can be answered", {
  x: 7.1, y: 1.55, w: 5.4, h: 0.32, margin: 0,
  fontFace: BODY, fontSize: 14.5, bold: true, color: C.bad,
});
bullets(s, [
  "No generated spectrum has ever entered SSL pretraining or evaluation — the pipeline stops at sampling",
  "No fidelity metric: we cannot yet say whether a generated spectrum's intensity distribution matches the experimental one",
  "The linear-combination route (sums of pure metabolite spectra) is not implemented, and is the more physically defensible option",
  "No agreed criterion, set in advance, for “does synthetic data help”",
], { x: 7.1, y: 1.98, w: 5.4, h: 4.0, size: 12, gap: 11 });

noteCard(s, {
  x: M, y: 6.25, w: 12.2, h: 0.73, fill: C.card, badge: "▸", badgeColor: C.teal,
  head: "Low infrastructure debt — the missing pieces are the connection to pretraining and an agreed success criterion.",
  headSize: 11.5,
});
s.addNotes("The infrastructure debt is low; most code exists. What is missing is the connection to pretraining and an agreed fidelity criterion. I would rather agree that criterion with you before spending GPU time than after.");

// ============================================================
// 20. PROPOSAL
// ============================================================
s = pres.addSlide();
s.background = { color: C.mid };
s.addText("PROPOSED NEXT TWO WEEKS", {
  x: M, y: 0.5, w: 11, h: 0.28, margin: 0,
  fontFace: BODY, fontSize: 11.5, bold: true, color: C.teal, charSpacing: 1.3,
});
s.addText("Ranked by evidence, not by novelty", {
  x: M, y: 0.85, w: 12, h: 0.6, margin: 0,
  fontFace: HEAD, fontSize: 30, bold: true, color: C.white,
});
[
  ["1", "Resolve the corpus regression", "Re-read every v4 arm against v3, and diagnose why the harsher v3 suppression pretrains better. Largest measured effect in the project, and it needs no new training.", C.good],
  ["2", "Re-run peak saturation on v3, beyond 60 peaks", "Answers your “at all x” directly, and on the corpus we would actually pretrain on.", C.teal],
  ["3", "Close the synthetic loop, with a criterion agreed first", "Generate, compare intensity distributions against experimental, then pretrain on real + synthetic and evaluate.", C.teal],
  ["4", "Make single runs trustworthy", "Opt-in deterministic mode, plus three replicates as standard for any new arm. Without it every comparison needs triplicates anyway.", C.warn],
].forEach((p, i) => {
  const yy = 1.62 + i * 1.06;
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: yy, w: 8.4, h: 0.94, rectRadius: 0.06,
    fill: { color: C.darkCard }, line: { color: C.darkEdge, width: 1 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: M + 0.18, y: yy + 0.24, w: 0.46, h: 0.46,
    fill: { color: p[3] }, line: { color: p[3], width: 0.5 },
  });
  s.addText(p[0], {
    x: M + 0.18, y: yy + 0.24, w: 0.46, h: 0.46, margin: 0,
    align: "center", valign: "middle", fontFace: BODY, fontSize: 13, bold: true, color: C.white,
  });
  s.addText(p[1], {
    x: M + 0.8, y: yy + 0.09, w: 7.4, h: 0.3, margin: 0,
    fontFace: BODY, fontSize: 12.5, bold: true, color: C.white, valign: "middle",
  });
  s.addText(p[2], {
    x: M + 0.8, y: yy + 0.38, w: 7.4, h: 0.48, margin: 0,
    fontFace: BODY, fontSize: 9.5, color: C.onDarkMute, valign: "top", lineSpacingMultiple: 0.95,
  });
});

s.addShape(pres.ShapeType.roundRect, {
  x: 9.35, y: 1.62, w: 3.4, h: 4.18, rectRadius: 0.06,
  fill: { color: C.darkCard2 }, line: { color: "3B4C7C", width: 1 },
});
s.addText("Three things I would like your view on", {
  x: 9.58, y: 1.8, w: 2.95, h: 0.55, margin: 0,
  fontFace: BODY, fontSize: 13, bold: true, color: C.white, lineSpacingMultiple: 0.95,
});
bullets(s, [
  "Is v3 acceptable to standardise on, given it pretrains better but suppresses artefacts less thoroughly?",
  "What counts as “synthetic data helped” — downstream accuracy only, or distributional fidelity too?",
  "With a 0.035 noise floor at n = 37–113, should we add a larger evaluation cohort before chasing further gains?",
], { x: 9.58, y: 2.45, w: 2.95, h: 3.2, size: 10.5, gap: 10, color: C.onDark });

s.addText("Bottom line: the model-side axes are close to exhausted and now honestly measured. The data-side branch of your tree is where the remaining leverage is — and most of its infrastructure already exists.", {
  x: M, y: 6.1, w: 12.2, h: 0.8, margin: 0,
  fontFace: BODY, fontSize: 13, italic: true, color: C.onDark, lineSpacingMultiple: 1.05,
});
s.addNotes("Land on the three questions — they are the reason for the meeting. Priority 1 costs nothing and has the largest measured effect. Priority 3 needs their judgement before I spend GPU time.");

// ============================================================
pres.writeFile({ fileName: path.join(REPO, "docs/Weekly_Update_2026-07-30.pptx") })
  .then((f) => console.log("Wrote " + f));
