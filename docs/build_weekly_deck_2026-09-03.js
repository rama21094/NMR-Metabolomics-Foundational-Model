// Weekly PI update, 3 September 2026 — covers everything since the 10 Aug group meeting.
//   node docs/build_weekly_deck_2026-09-03.js
// House style, palette and helpers deliberately match build_group_meeting_deck.js so the
// two decks read as one series.

const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = path.join(__dirname, "gm_figures");
const PLOT = path.join(__dirname, "..", "results", "plots", "synthetic_windowed");

const NAVY = "21295C", DEEP = "065A82", TEAL = "1C7293";
const GOLD = "B8860B", CORAL = "C1435B", GREEN = "1A7A3C";
const INK = "1A1A1A", MUTED = "5A6068";
const TINT = "EEF3F7", TINT_BAD = "FBEEF1", TINT_OK = "EBF5EE";
const F = "Arial";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Shankararama Sharma";
pres.title = "Weekly update — 3 September 2026";

const AR = {
  "gm09_barth_seeds.png": 2.3810,
  "gm10_recon_baselines.png": 2.3810,
  "gm11_batch_audit.png": 2.3810,
};
function fig(slide, name, { x, y, w, h }, dir) {
  const ar = AR[name];
  if (!ar) throw new Error(`no aspect ratio recorded for ${name}`);
  let dw = w, dh = w / ar;
  if (dh > h) { dh = h; dw = h * ar; }
  slide.addImage({ path: path.join(dir || FIG, name),
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
function bubble(s, { x, y, d, color, label, fontSize }) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color }, line: { type: "none" } });
  s.addText(label, { x, y, w: d, h: d, fontFace: F, fontSize: fontSize || 18, bold: true,
    color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
}

/* ============ 1. WHAT CHANGED — the evidence base shrank ============ */
{
  const s = newLight("Since 10 August: the evidence base got smaller",
    "Both targets where SSL looked competitive turned out not to count");
  const rows = [
    ["Barth — “SSL beats classical, +0.101”", "RETRACTED", CORAL,
     "0.806 was the highest of 5 pretraining seeds, +1.5 sd above its own group mean. The other four score 0.598–0.699, all below classical’s 0.705. Across 20 readings SSL wins 5."],
    ["MTBLS326 — perfect 1.000 on n=42", "UNUSABLE", CORAL,
     "Confounded by design: cases are samples 1–27, controls 101–130 — separate acquisition blocks. Spectral regions containing NO metabolites classify the label at 0.726 (p<0.005)."],
    ["Barth, MTBLS563, BrC-T2D — data quality", "AUDITED", GREEN,
     "All five targets audited. Barth is clean (order AUC 0.58, noise at chance). MTBLS563 and BrC-T2D cancer carry order caveats only. BrC-T2D diabetes is ambiguous."],
  ];
  rows.forEach(([t, tag, col, body], i) => {
    const y = 1.62 + i * 1.60;
    card(s, { x: 0.55, y, w: 12.2, h: 1.40, fill: i === 2 ? TINT_OK : TINT_BAD });
    s.addText(t, { x: 0.85, y: y + 0.10, w: 8.2, h: 0.34, fontFace: F, fontSize: 17,
      bold: true, color: INK, margin: 0, valign: "middle", isTextBox: true });
    s.addText(tag, { x: 9.2, y: y + 0.10, w: 3.3, h: 0.34, fontFace: F, fontSize: 17,
      bold: true, color: col, margin: 0, align: "right", valign: "middle", isTextBox: true });
    s.addText(body, { x: 0.85, y: y + 0.48, w: 11.6, h: 0.82, fontFace: F, fontSize: 14,
      color: INK, margin: 0, valign: "top", lineSpacing: 17, isTextBox: true });
  });
  card(s, { x: 0.55, y: 6.48, w: 12.2, h: 0.62, fill: NAVY });
  s.addText("Honest full-data record on the targets that remain admissible: 0 SSL wins, 3 losses — "
    + "and those are the biggest cohorts, the ones with real error bars.", {
    x: 0.85, y: 6.48, w: 11.6, h: 0.62, fontFace: F, fontSize: 16, bold: true,
    color: "FFFFFF", margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Lead with this. Two results we were relying on are gone, and both went for the same "
    + "reason we already knew about: single-run claims and unaudited data.\n\n"
    + "Barth: experiment #18. The 0.806 came from the original unseeded run, exactly the position "
    + "#15 showed to be selection-biased upward.\n\n"
    + "MTBLS326: experiment #11, the batch audit we had deferred since February. The permutation "
    + "null could never have caught this — permuting labels destroys the batch structure and the "
    + "biology together, so a confounded dataset passes at p<=0.02.\n\n"
    + "This does NOT weaken the few-shot result. It removes two targets that were making SSL look "
    + "better than it is.");
}

/* ============ 2. WHY IT FAILED — the mechanism ============ */
{
  const s = newLight("We now know WHY, not just that it failed",
    "Masked reconstruction is nearly free on this corpus — so the objective cannot bite");
  fig(s, "gm10_recon_baselines.png", { x: 0.55, y: 1.48, w: 12.2, h: 4.30 });
  card(s, { x: 0.55, y: 5.95, w: 5.95, h: 1.05, fill: TINT_BAD });
  s.addText([
    { text: "No learning needed.  ", options: { bold: true, color: CORAL } },
    { text: "Finding the most similar other spectrum and copying its hidden bins scores "
      + "0.900 at 60% masking. The trained network gets 0.921.", options: { color: INK } },
  ], { x: 0.80, y: 5.99, w: 5.46, h: 0.97, fontFace: F, fontSize: 14,
       margin: 0, valign: "middle", lineSpacing: 17, isTextBox: true });
  card(s, { x: 6.80, y: 5.95, w: 5.95, h: 1.05, fill: TINT });
  s.addText([
    { text: "The corpus is narrow, not small.  ", options: { bold: true, color: DEEP } },
    { text: "86% of variance in 5 components; the median spectrum’s nearest neighbour "
      + "correlates at 0.991. And it does not cover our evaluation cohorts (r = 0.37–0.78).",
      options: { color: INK } },
  ], { x: 7.05, y: 5.99, w: 5.46, h: 0.97, fontFace: F, fontSize: 14,
       margin: 0, valign: "middle", lineSpacing: 17, isTextBox: true });
  s.addNotes("This is the most useful thing we have produced since August, because it converts a "
    + "negative result into a mechanism.\n\n"
    + "Every reconstruction claim in the project had been reported WITHOUT a baseline — the same "
    + "error as quoting classifier accuracy with no majority-class rate. With masks matched, "
    + "copy-a-neighbour reaches 0.900 against the network's 0.921.\n\n"
    + "Why: the corpus is close to low-rank and 55% of rows have a neighbour above r=0.99. The "
    + "population prior already answers the pretext task, so masked modelling cannot be forced to "
    + "learn anything disease-discriminative.\n\n"
    + "Separately (#20) we checked the train/eval boundary: NO leakage — zero evaluation spectra "
    + "have a near-duplicate in the corpus — but coverage is poor, r = 0.37-0.78 against 0.99 "
    + "within-corpus.\n\n"
    + "One prediction of ours failed and it is worth admitting: we expected r to be inflated by "
    + "predicting empty baseline. It is not — peak bins 0.879 vs baseline bins 0.855.");
}

/* ============ 3. FOLLOWING YOUR OUTLINE — is data limiting? ============ */
{
  const s = newLight("Your outline: “is data limiting?”",
    "Answered — but the answer splits in two, and only one half was ever measured");
  const items = [
    ["1", GREEN, "Marginal distributions: NOT limiting",
     "Re-ran the 60-peak KS saturation on the v4 corpus under unit-area normalisation. Median N* = 363 spectra; 9,670 is roughly 10× more than needed. For any single peak’s intensity distribution, we have ample data."],
    ["2", CORAL, "The JOINT distribution: this is the constraint",
     "Across-spectra correlation never decays — |r| is 0.95 at 32 points, 0.73 at 1024, and still 0.45 at 6 ppm separation. Removing the top 5 principal components barely dents it, so it is real composition covariance, not one global scale factor."],
    ["3", GOLD, "A trap worth recording",
     "Re-picking the peak panel per normaliser changes which peaks are analysed (only 9 of 60 shared), and would have reversed the conclusion. The panel must be held fixed across conditions — now enforced by a --peaks-from flag."],
  ];
  items.forEach(([n, col, head, body], i) => {
    const y = 1.60 + i * 1.72;
    card(s, { x: 0.55, y, w: 12.2, h: 1.52, fill: TINT });
    bubble(s, { x: 0.82, y: y + 0.44, d: 0.62, color: col, label: n, fontSize: 20 });
    s.addText(head, { x: 1.66, y: y + 0.12, w: 10.8, h: 0.36, fontFace: F, fontSize: 18,
      bold: true, color: col, margin: 0, valign: "middle", isTextBox: true });
    s.addText(body, { x: 1.66, y: y + 0.50, w: 10.8, h: 0.94, fontFace: F, fontSize: 14,
      color: INK, margin: 0, valign: "top", lineSpacing: 17, isTextBox: true });
  });
  s.addNotes("This maps directly onto the whiteboard tree.\n\n"
    + "The marginal half of the question was already partly answered by the peak-saturation work; "
    + "I re-ran it on v4 under unit-area (min-max is the wrong unit for an intensity distribution, "
    + "because its unit is 'this spectrum's tallest peak', which turns out to be 21 different "
    + "chemical positions across 400 spectra). The conclusion replicates: ~360 spectra suffice.\n\n"
    + "The joint half had never been measured and it is where the difficulty lies. Note the "
    + "contrast with the WITHIN-spectrum autocorrelation, which dies by ~1000 points — that is "
    + "lineshape width. Conflating the two would have suggested a 1000-point window is enough.\n\n"
    + "Consequence: no overlap setting makes window-stitching lossless. That is a hard constraint "
    + "on route (b), and it is measured rather than assumed.");
}

/* ============ 4. SYNTHETIC DATA — route (b) built ============ */
{
  const s = newLight("Synthetic data, route (b): overlapping windows",
    "Built and validated. The trade-off it exposes is the result.");
  s.addImage({ path: path.join(PLOT, "fig_synth_methods.png"),
    x: 0.55, y: 1.46, w: 7.55, h: 5.07 });
  const stats = [
    ["independent", "destroys correlation at every scale — 0.198 vs 0.717 even at 0.09 ppm", CORAL],
    ["quilted", "local structure recovered; long-range collapses to 0.051", GOLD],
    ["quilted_base", "long-range preserved (0.401 vs 0.453) — but nn r = 0.982, i.e. copying", TEAL],
  ];
  s.addText("Three variants, 100 spectra each", { x: 8.35, y: 1.52, w: 4.4, h: 0.34,
    fontFace: F, fontSize: 16, bold: true, color: NAVY, margin: 0, isTextBox: true });
  stats.forEach(([t, b, col], i) => {
    const y = 1.98 + i * 1.16;
    card(s, { x: 8.35, y, w: 4.4, h: 1.02, fill: TINT });
    s.addText(t, { x: 8.58, y: y + 0.08, w: 3.95, h: 0.28, fontFace: F, fontSize: 14.5,
      bold: true, color: col, margin: 0, valign: "middle", isTextBox: true });
    s.addText(b, { x: 8.58, y: y + 0.36, w: 3.95, h: 0.60, fontFace: F, fontSize: 12.5,
      color: INK, margin: 0, valign: "top", lineSpacing: 15, isTextBox: true });
  });
  card(s, { x: 8.35, y: 5.52, w: 4.4, h: 1.01, fill: TINT_BAD });
  s.addText([
    { text: "Novelty and long-range fidelity pull against each other. ",
      options: { bold: true, color: CORAL } },
    { text: "If correlation never decays, anything novel at long range is also wrong at long range.",
      options: { color: INK } },
  ], { x: 8.58, y: 5.56, w: 3.95, h: 0.93, fontFace: F, fontSize: 12.5, margin: 0,
       valign: "middle", lineSpacing: 15, isTextBox: true });
  s.addNotes("Worth pausing on the figure: all three variants LOOK like plausible NMR spectra, "
    + "including the one whose statistics are destroyed, because a 50% Hann crossfade hides every "
    + "join. Visual inspection cannot validate a generator here. That is why we scored them "
    + "against the measured correlation profile instead.\n\n"
    + "The trade-off follows directly from the joint-distribution measurement on the previous "
    + "slide, so it is not a tuning failure — it is structural to window sampling.\n\n"
    + "Practical read: route (b) is usable as LOCAL augmentation on top of a coherent base "
    + "spectrum, not as a stand-alone generator. That is what pushed us to build route (c).");
}

/* ============ 5. LC GENERATOR — Stage 0 ============ */
{
  const s = newLight("Synthetic data, route (c): the LC generator",
    "GISSMO basis built and validated at 600 MHz — but the fit gate does not pass yet");
  const left = [
    ["Library", "GISSMO is already installed on NMRbox — 661 compounds with spin-system parameters AND spectra pre-simulated at 19 field strengths. No download, no spin simulation needed.", GREEN],
    ["Basis", "43-metabolite serum panel, each pinned to an explicit compound ID with formula re-verification. 19 of 21 peak positions match literature within 0.03 ppm. Condition number 6.9.", GREEN],
    ["Lipids", "GISSMO holds small molecules only, so the lipoprotein envelope was extracted from our own corpus. It accounts for 37.5% of the signal.", TEAL],
  ];
  left.forEach(([h, b, col], i) => {
    const y = 1.58 + i * 1.30;
    card(s, { x: 0.55, y, w: 6.1, h: 1.16, fill: TINT_OK });
    s.addText(h, { x: 0.80, y: y + 0.08, w: 5.6, h: 0.30, fontFace: F, fontSize: 16,
      bold: true, color: col, margin: 0, valign: "middle", isTextBox: true });
    s.addText(b, { x: 0.80, y: y + 0.40, w: 5.6, h: 0.70, fontFace: F, fontSize: 13,
      color: INK, margin: 0, valign: "top", lineSpacing: 15.5, isTextBox: true });
  });
  card(s, { x: 6.90, y: 1.58, w: 5.85, h: 3.86, fill: TINT_BAD });
  s.addText("The gate: does the model explain real spectra?", {
    x: 7.15, y: 1.70, w: 5.35, h: 0.32, fontFace: F, fontSize: 16, bold: true,
    color: CORAL, margin: 0, valign: "middle", isTextBox: true });
  s.addText("46%", { x: 7.15, y: 2.10, w: 5.35, h: 0.90, fontFace: F, fontSize: 62,
    bold: true, color: CORAL, margin: 0, align: "center", isTextBox: true });
  s.addText("of a real serum spectrum’s variance is explained\n(target: 90%)", {
    x: 7.15, y: 3.02, w: 5.35, h: 0.56, fontFace: F, fontSize: 13.5, color: INK,
    margin: 0, align: "center", lineSpacing: 16, isTextBox: true });
  s.addText([
    { text: "Adding the lipid basis fixed the solver ", options: { bold: true } },
    { text: "(constrained fit 0.11 → 0.45) ", options: { color: MUTED } },
    { text: "but did NOT raise the ceiling (0.47 → 0.51). By elimination, the remaining "
      + "bottleneck is peak ALIGNMENT.", options: { bold: true } },
  ], { x: 7.15, y: 3.68, w: 5.35, h: 1.10, fontFace: F, fontSize: 13.5, color: INK,
       margin: 0, valign: "top", lineSpacing: 16, isTextBox: true });
  card(s, { x: 0.55, y: 5.60, w: 12.2, h: 0.95, fill: TINT });
  s.addText([
    { text: "Root cause, and it is fixable:  ", options: { bold: true, color: DEEP } },
    { text: "our corpus has no 0 ppm reference resonance and was aligned to its own rightmost "
      + "peak, so its ppm axis carries an arbitrary absolute offset of 0.15–0.3 ppm. That is "
      + "invisible within the corpus but blocks any comparison against an external library.",
      options: { color: INK } },
  ], { x: 0.85, y: 5.64, w: 11.6, h: 0.87, fontFace: F, fontSize: 14, margin: 0,
       valign: "middle", lineSpacing: 17, isTextBox: true });
  s.addNotes("Be clear that Stage 0 is a genuine partial result, not a failure.\n\n"
    + "What is solid: the library is local, the basis is built, and it is chemically validated — "
    + "19 of 21 metabolites match literature shifts within 0.03 ppm, and the two apparent misses "
    + "were my reference values quoting a different peak of the same multiplet.\n\n"
    + "What is not: the forward model explains ~46-51% of a real spectrum against a 90% bar.\n\n"
    + "The diagnosis is the useful part. Adding the empirical lipid basis raised the CONSTRAINED "
    + "fit from 0.11 to 0.45 and made the solver well behaved, but the unconstrained CEILING only "
    + "moved 0.47 to 0.51. So the missing variance is not missing components — it is that "
    + "misplaced peaks cannot be fitted by adding basis vectors.\n\n"
    + "The ppm-offset finding is of independent value. Every within-corpus result we have remains "
    + "internally consistent; it only bites when comparing to an external reference.");
}

/* ============ 6. BLOCKED ON + ASKS ============ */
{
  const s = newLight("What I need, and what is next",
    "One input from the instrument archive unblocks four separate open questions");
  card(s, { x: 0.55, y: 1.58, w: 12.2, h: 2.15, fill: TINT_OK });
  s.addText("Ask 1 — the Bruker parameter export", { x: 0.85, y: 1.70, w: 11.6, h: 0.34,
    fontFace: F, fontSize: 18, bold: true, color: GREEN, margin: 0, valign: "middle", isTextBox: true });
  s.addText("The raw archive is tens of GB, but we only need the parameter files: an acqus is 9 kB "
    + "against ~1 MB for a single FID. A standard-library script extracts them on the machine "
    + "holding the disk and produces a few-MB CSV. It unblocks:", {
    x: 0.85, y: 2.06, w: 11.6, h: 0.52, fontFace: F, fontSize: 14, color: INK,
    margin: 0, valign: "top", lineSpacing: 17, isTextBox: true });
  const unlocks = [
    ["SF / SR", "true chemical-shift referencing — the fix for the ceiling above"],
    ["SFO1 / BF1", "field strength per experiment — is the corpus mixed-field?"],
    ["DATE", "real acquisition timestamps — upgrades the batch audit"],
    ["NS / RG", "scans and gain — tests the normalisation-leak question directly"],
  ];
  unlocks.forEach(([k, v], i) => {
    const x = 0.85 + (i % 2) * 5.95, y = 2.66 + Math.floor(i / 2) * 0.48;
    s.addText(k, { x, y, w: 1.55, h: 0.36, fontFace: F, fontSize: 13.5, bold: true,
      color: DEEP, margin: 0, valign: "middle", isTextBox: true });
    s.addText(v, { x: x + 1.60, y, w: 4.05, h: 0.36, fontFace: F, fontSize: 13,
      color: INK, margin: 0, valign: "middle", isTextBox: true });
  });
  const asks = [
    ["2", TEAL, "Lipid basis — done, as you suggested",
     "Extracted empirically from the corpus rather than from a library. Already folded into the fit."],
    ["3", GOLD, "Decision: how far to push route (c)?",
     "Per-peak shift fitting is the clear next step and does not need the archive — it can start now, in parallel."],
  ];
  asks.forEach(([n, col, head, body], i) => {
    const y = 3.95 + i * 1.38;
    card(s, { x: 0.55, y, w: 12.2, h: 1.20, fill: TINT });
    bubble(s, { x: 0.82, y: y + 0.30, d: 0.58, color: col, label: n, fontSize: 19 });
    s.addText(head, { x: 1.62, y: y + 0.10, w: 10.9, h: 0.34, fontFace: F, fontSize: 16.5,
      bold: true, color: col, margin: 0, valign: "middle", isTextBox: true });
    s.addText(body, { x: 1.62, y: y + 0.46, w: 10.9, h: 0.64, fontFace: F, fontSize: 14,
      color: INK, margin: 0, valign: "top", lineSpacing: 17, isTextBox: true });
  });
  card(s, { x: 0.55, y: 6.78, w: 12.2, h: 0.42, fill: NAVY });
  s.addText("Proposed: start per-peak shift fitting now; run the parameter export in parallel.", {
    x: 0.85, y: 6.78, w: 11.6, h: 0.42, fontFace: F, fontSize: 14.5, bold: true,
    color: "FFFFFF", margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Close on the ask, not on the status.\n\n"
    + "The single highest-value thing is the parameter export, because one small CSV resolves four "
    + "questions that are currently each blocked separately. Emphasise that it does NOT mean moving "
    + "the archive — the script reads a few kB per experiment and never opens a FID.\n\n"
    + "If asked what happens if the corpus turns out to be mixed-field: then one 600 MHz basis is "
    + "wrong for part of it, and that alone could account for a chunk of the missing variance in "
    + "the fit gate. Either way we learn something that matters.\n\n"
    + "Per-peak shift fitting can begin immediately without the archive — fitting each spectrum's "
    + "offset as a free parameter would tell us how much of the ~50% ceiling is alignment, before "
    + "the CSV arrives.");
}

const out = path.join(__dirname, "Weekly_Update_2026-09-03.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out, `(${pageNo} slides)`));
