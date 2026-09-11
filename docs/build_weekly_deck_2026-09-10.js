// Weekly PI update, 10 September 2026 — covers the week since 3 September.
//   node docs/build_weekly_deck_2026-09-10.js
// House style, palette and helpers match build_weekly_deck_2026-09-03.js and
// build_group_meeting_deck.js so all three decks read as one series.

const pptxgen = require("pptxgenjs");
const path = require("path");

const FIGDIR = path.join(__dirname, "..", "results", "figures");

const NAVY = "21295C", DEEP = "065A82", TEAL = "1C7293";
const GOLD = "B8860B", CORAL = "C1435B", GREEN = "1A7A3C";
const INK = "1A1A1A", MUTED = "5A6068";
const TINT = "EEF3F7", TINT_BAD = "FBEEF1", TINT_OK = "EBF5EE", TINT_WARN = "FBF4E6";
const F = "Arial";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Shankararama Sharma";
pres.title = "Weekly update — 10 September 2026";

const AR = { "fig_axis_misalignment.png": 1936 / 732 };
function fig(slide, name, { x, y, w, h }) {
  const ar = AR[name];
  if (!ar) throw new Error(`no aspect ratio recorded for ${name}`);
  let dw = w, dh = w / ar;
  if (dh > h) { dh = h; dw = h * ar; }
  slide.addImage({ path: path.join(FIGDIR, name),
    x: x + (w - dw) / 2, y: y + (h - dh) / 2, w: dw, h: dh });
}

let pageNo = 0;
function slideTitle(s, text, sub) {
  s.addText(text, { x: 0.55, y: 0.30, w: 12.2, h: 0.72, fontFace: F, fontSize: 30,
    bold: true, color: NAVY, align: "left", margin: 0, valign: "middle", isTextBox: true });
  if (sub) s.addText(sub, { x: 0.55, y: 1.03, w: 12.2, h: 0.42, fontFace: F, fontSize: 17,
    color: MUTED, align: "left", margin: 0, valign: "middle", isTextBox: true });
}
function newLight(title, sub) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  if (title) slideTitle(s, title, sub);
  pageNo += 1;
  s.addText(String(pageNo), { x: 12.72, y: 6.95, w: 0.45, h: 0.3, fontFace: F,
    fontSize: 12, color: "AAB0B8", align: "right", margin: 0, isTextBox: true });
  return s;
}
function card(s, { x, y, w, h, fill }) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill || TINT },
    rectRadius: 0.08, line: { type: "none" } });
}
function banner(s, y, text, fill) {
  card(s, { x: 0.55, y, w: 12.2, h: 0.62, fill: fill || NAVY });
  s.addText(text, { x: 0.85, y, w: 11.6, h: 0.62, fontFace: F, fontSize: 16, bold: true,
    color: "FFFFFF", margin: 0, valign: "middle", isTextBox: true });
}
function stat(s, { x, y, w, big, label, color }) {
  s.addText(big, { x, y, w, h: 0.78, fontFace: F, fontSize: 46, bold: true,
    color: color || DEEP, align: "center", margin: 0, valign: "middle", isTextBox: true });
  s.addText(label, { x, y: y + 0.76, w, h: 0.62, fontFace: F, fontSize: 13,
    color: MUTED, align: "center", margin: 0, valign: "top", lineSpacing: 15, isTextBox: true });
}

/* ===== 1. The Bruker export landed and answered more than we asked ===== */
{
  const s = newLight("The Bruker export answered more than we asked",
    "Two Stage-0 questions closed; a third turned into a corpus-quality finding");

  const rows = [
    ["Field strength — is one simulated basis enough?", "SETTLED", GREEN, TINT_OK,
     "600 MHz covers 97.8% of rows (700 MHz: 2.2%, one study per pull). The single-field "
     + "GISSMO basis is appropriate, and field is NOT a contributor to the fit-gate ceiling."],
    ["Pulse programme — is the lipid basis well defined?", "SETTLED", GREEN, TINT_OK,
     "cpmgpr* covers 99.7% of rows. The corpus is homogeneously CPMG, so macromolecule signal "
     + "is suppressed corpus-wide and the empirical lipid basis is one object, not an average of two."],
    ["Chemical-shift axis — why does the fit gate stall?", "THE FINDING", CORAL, TINT_BAD,
     "The corpus is not on a common ppm axis. ~40% of rows sit ~800 points — 60 linewidths — "
     + "away from the rest. This is our own preprocessing, and it explains the ceiling."],
  ];
  rows.forEach(([t, tag, col, tint, body], i) => {
    const y = 1.58 + i * 1.44;
    card(s, { x: 0.55, y, w: 12.2, h: 1.26, fill: tint });
    s.addText(t, { x: 0.85, y: y + 0.08, w: 8.2, h: 0.34, fontFace: F, fontSize: 17,
      bold: true, color: INK, margin: 0, valign: "middle", isTextBox: true });
    s.addText(tag, { x: 9.2, y: y + 0.08, w: 3.3, h: 0.34, fontFace: F, fontSize: 16,
      bold: true, color: col, margin: 0, align: "right", valign: "middle", isTextBox: true });
    s.addText(body, { x: 0.85, y: y + 0.44, w: 11.6, h: 0.74, fontFace: F, fontSize: 14,
      color: INK, margin: 0, valign: "top", lineSpacing: 17, isTextBox: true });
  });

  card(s, { x: 0.55, y: 5.92, w: 12.2, h: 1.00, fill: TINT });
  s.addText([
    { text: "Bonus — provenance recovered.  ", options: { bold: true, color: DEEP } },
    { text: "The export's file paths carry MetaboLights accessions, so we now know the corpus "
      + "draws on 11 studies (MTBLS798 alone is 54% of it). The repository never recorded this, "
      + "and it cannot be recovered from the .npy files.", options: { color: INK } },
  ], { x: 0.85, y: 5.96, w: 11.6, h: 0.92, fontFace: F, fontSize: 14,
       margin: 0, valign: "middle", lineSpacing: 17, isTextBox: true });

  s.addNotes("Open by saying the export did its job: the two questions we needed for the "
    + "metabolite basis are closed, both favourably. Then pivot.\n\n"
    + "Caveat to state up front if he asks: my FIRST reading of the referencing question was "
    + "wrong. I tested it via SR = (SF - BF1)*1e6 and concluded referencing varied per spectrum. "
    + "That is invalid -- proc_OFFSET is reported AFTER referencing, so each spectrum's ppm axis "
    + "already absorbs its own SR. MTBLS798 proves it: SR = -81 Hz vs +6 Hz for MTBLS147, yet "
    + "their OFFSETs differ by 0.11 ppm, not the 0.14 ppm SR would imply. The script's verdict "
    + "logic was rewritten around the axis geometry. Better to volunteer this than be caught.\n\n"
    + "Scripts: code/analysis/bruker_param_audit.py. Commit 8cda56a.");
}

/* ===== 2. The finding, confirmed two independent ways ===== */
{
  const s = newLight("Confirmed two independent ways",
    "alignSpectra.py interpolates to a common POINT COUNT and never reads a ppm axis");

  fig(s, "fig_axis_misalignment.png", { x: 0.55, y: 1.50, w: 12.2, h: 3.55 });

  // Card order matches the figure's panel order: measured left, predicted right.
  card(s, { x: 0.55, y: 5.18, w: 5.95, h: 1.18, fill: TINT_BAD });
  s.addText([
    { text: "Left — measured from the spectra.  ", options: { bold: true, color: CORAL } },
    { text: "Cross-correlating 1,200 rows against the corpus median, using no metadata at all: "
      + "50.8% at 0, 32.9% at −600 to −1,000, 6.2% beyond +3,000.", options: { color: INK } },
  ], { x: 0.80, y: 5.22, w: 5.46, h: 1.10, fontFace: F, fontSize: 13.5,
       margin: 0, valign: "middle", lineSpacing: 16, isTextBox: true });

  card(s, { x: 6.80, y: 5.18, w: 5.95, h: 1.18, fill: TINT });
  s.addText([
    { text: "Right — predicted from metadata.  ", options: { bold: true, color: DEEP } },
    { text: "Each row's peaks land at an index set by its own (OFFSET, SW, SF): 0 pts for "
      + "MTBLS798, −720 to −900 for five studies, +5,100 to +8,000 for three that are also "
      + "stretched 1.5−1.7×. Shares: 54% / 42% / 4%.", options: { color: INK } },
  ], { x: 7.05, y: 5.22, w: 5.46, h: 1.10, fontFace: F, fontSize: 13.5,
       margin: 0, valign: "middle", lineSpacing: 16, isTextBox: true });

  banner(s, 6.48, "Scale: a 1.2 Hz linewidth is 13 points here. 63% of rows are displaced by "
    + "more than one linewidth; 41% by more than fifty. These peaks do not overlap at all.");

  s.addNotes("The point of this slide is that the two panels were derived independently. "
    + "The right panel uses only Bruker parameter files and never touches a spectrum. The left "
    + "panel uses only spectra and never touches the metadata. They agree quantitatively.\n\n"
    + "Mechanism in one sentence: align_spectra_to_longest() in code/preprocessing/alignSpectra.py "
    + "interpolates every spectrum to a common number of POINTS, and never reads a chemical-shift "
    + "axis, so studies acquired over different windows put the same metabolite at different "
    + "indices.\n\n"
    + "If he asks why nobody noticed: the spectra look completely normal one at a time. The "
    + "defect only appears when you compare rows from different studies, and nothing in the "
    + "pipeline ever did that explicitly.\n\n"
    + "Script: code/analysis/corpus_axis_misalignment.py. 1,200 rows, band 72,000-84,000, "
    + "clear of the zeroed water window.");
}

/* ===== 3. What it changes ===== */
{
  const s = newLight("What this changes — and what it does not",
    "One puzzle solved, one earlier conclusion needs re-reading, one new open question");

  stat(s, { x: 0.55, y: 1.55, w: 3.90, big: "0.45–0.51",
    label: "fit-gate R² ceiling, now explained.\nNot the solver, not the basis —\nthe design cannot match\npeaks 800 points away.",
    color: GREEN });
  stat(s, { x: 4.70, y: 1.55, w: 3.90, big: "r = 0.991",
    label: "median nearest-neighbour from §19.\nUnder an 800-point offset this\nCANNOT be cross-study — so the\nnear-duplicates are within-study.",
    color: GOLD });
  stat(s, { x: 8.85, y: 1.55, w: 3.90, big: "9,670 → ?",
    label: "the corpus is more fragmented\nthan its row count suggests.\nEffective size is an open\nquestion, not a known number.",
    color: CORAL });

  const items = [
    ["Solved", GREEN, TINT_OK,
     "Three rounds of ridge and bounds tuning on the fit-gate solver were treating a symptom. "
     + "The unconstrained bound of 0.506 is exactly what a design should reach when it fits the "
     + "majority axis and misses the rest."],
    ["Re-read, not retracted", GOLD, TINT_WARN,
     "The negative SSL result stands: copy-a-neighbour still matches the network, the few-shot "
     + "record is still 0 wins / 3 losses, the noise floor is unchanged. But the MECHANISM sharpens "
     + "— the pretext task is easy because near-duplicates exist within studies."],
    ["Newly open", CORAL, TINT_BAD,
     "Does re-pretraining on a correctly aligned corpus change the transfer conclusion? We cannot "
     + "answer this from what we have. It is now the most important question in the queue."],
  ];
  items.forEach(([tag, col, tint, body], i) => {
    const y = 3.62 + i * 1.10;
    card(s, { x: 0.55, y, w: 12.2, h: 0.96, fill: tint });
    s.addText(tag, { x: 0.85, y, w: 2.30, h: 0.96, fontFace: F, fontSize: 15, bold: true,
      color: col, margin: 0, valign: "middle", isTextBox: true });
    s.addText(body, { x: 3.20, y: y + 0.06, w: 9.25, h: 0.84, fontFace: F, fontSize: 13.5,
      color: INK, margin: 0, valign: "middle", lineSpacing: 16, isTextBox: true });
  });

  s.addNotes("Be careful with the middle card. He may hear 'the corpus was broken' as 'so the "
    + "negative result was an artefact'. It is not: copy-a-neighbour beating the network is a "
    + "comparison BETWEEN two methods on the SAME corpus, so a shared defect does not explain the "
    + "gap. What changes is the story of WHY the pretext task is too easy.\n\n"
    + "If he pushes on the third card -- yes, it is genuinely possible that a properly aligned "
    + "corpus trains a better representation. I would not bet on it, because the copy-a-neighbour "
    + "result suggests the objective is the problem rather than the alignment, but I cannot rule "
    + "it out and should not pretend otherwise.\n\n"
    + "Documented in docs/SSL_vs_classical_analysis.md section 21 and docs/PI_outline.md 5h.");
}

/* ===== 4. The decision ===== */
{
  const s = newLight("The decision I need from you", "Rebuild from raw, or patch what we have");

  const opts = [
    ["A — Rebuild from the raw archive", DEEP, TINT, "RECOMMENDED",
     ["Re-interpolate every spectrum onto one ppm grid from (OFFSET, SW, SF).",
      "Deterministic. No fitting, no free parameters, no judgement calls.",
      "Cost: rebuilds the pipeline from raw and invalidates every checkpoint.",
      "Requires the HDD, because the row → study mapping was never recorded."]],
    ["B — Patch the corpus we already have", GOLD, TINT_WARN, "INTERIM",
     ["Assign rows to axis groups by cross-correlation, then shift each group.",
      "Fast, needs no raw data, and recovers most of the gross displacement.",
      "But it is an estimate on damaged data — the stretched 4% cannot be undone.",
      "Good enough to test whether alignment changes anything. Not publishable."]],
  ];
  opts.forEach(([title, col, tint, tag, lines], i) => {
    const x = 0.55 + i * 6.25;
    card(s, { x, y: 1.55, w: 5.95, h: 3.55, fill: tint });
    s.addText(title, { x: x + 0.30, y: 1.68, w: 5.35, h: 0.40, fontFace: F, fontSize: 17,
      bold: true, color: col, margin: 0, valign: "middle", isTextBox: true });
    s.addText(tag, { x: x + 0.30, y: 2.08, w: 5.35, h: 0.28, fontFace: F, fontSize: 12,
      bold: true, color: MUTED, margin: 0, valign: "middle", isTextBox: true });
    s.addText(lines.map((t, j) => ({
      text: t, options: { bullet: true, breakLine: j < lines.length - 1 },
    })), { x: x + 0.30, y: 2.44, w: 5.35, h: 2.50, fontFace: F, fontSize: 13.5,
      color: INK, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
  });

  card(s, { x: 0.55, y: 5.26, w: 12.2, h: 1.10, fill: TINT_OK });
  s.addText([
    { text: "Also for discussion:  ", options: { bold: true, color: GREEN } },
    { text: "(1) does this finding belong in the paper as a methodological point — "
      + "index-based alignment is common and silently wrong; (2) the 778 Workbench spectra have "
      + "no parameter export, so ~8% of the corpus is still uncharacterised; (3) "
      + "data/BrC_T2D/…_metadata_mapping.csv holds patient names and must be de-identified "
      + "before anything is shared.", options: { color: INK } },
  ], { x: 0.85, y: 5.30, w: 11.6, h: 1.02, fontFace: F, fontSize: 14,
       margin: 0, valign: "middle", lineSpacing: 17, isTextBox: true });

  banner(s, 6.48, "My recommendation: rebuild. The defect is in the one coordinate that carries "
    + "chemical meaning, and every downstream conclusion inherits it.", DEEP);

  s.addNotes("Do not start either option before he chooses -- A invalidates every checkpoint we "
    + "have, which is his call, not mine.\n\n"
    + "The honest argument for B first: it is cheap and it answers the scientific question "
    + "(does alignment change transfer?) in days rather than weeks. If B shows no change, A becomes "
    + "much less urgent. The argument for A is that B leaves us with a corpus we cannot defend in "
    + "print, and we would end up doing A anyway.\n\n"
    + "A reasonable compromise if he wants one: run B to answer the question, commit to A before "
    + "publication.\n\n"
    + "On the paper point: the label-permutation null cannot detect batch confounding (from the "
    + "audit work) and index-based alignment is silently wrong -- two transferable methodological "
    + "findings. Both are arguably publishable independent of the main negative result.\n\n"
    + "The PII item is not optional and should not be deferred again.");
}

pres.writeFile({ fileName: path.join(__dirname, "Weekly_Update_2026-09-10.pptx") })
  .then((f) => console.log("wrote " + f));
