#!/usr/bin/env python3
"""Weekly PI update, 2026-09-25. Plain language, one point per slide."""
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

INK=C(0x1E,0x27,0x61); GREY=C(0x5A,0x5A,0x6E); BODY=C(0x3C,0x3C,0x48)
RED=C(0xC0,0x27,0x2D); BLUE=C(0x2E,0x5E,0x9E); GREEN=C(0x1F,0x6B,0x3B)
CARD_G=C(0xEA,0xF3,0xEC); CARD_R=C(0xFB,0xEC,0xEC); CARD_B=C(0xEA,0xF0,0xF7); CARD_N=C(0xF4,0xF2,0xEA)
F="Arial"; D="docs/figures/"
prs=Presentation(); prs.slide_width=I(13.333); prs.slide_height=I(7.5)
BLANK=prs.slide_layouts[6]; num=[0]

def txt(s,x,y,w,h,runs,size=12,color=BODY,bold=False,italic=False,align=PP_ALIGN.LEFT):
    tb=s.shapes.add_textbox(I(x),I(y),I(w),I(h)); tf=tb.text_frame
    tf.word_wrap=True; tf.margin_left=0; tf.margin_right=0; tf.margin_top=0; tf.margin_bottom=0
    para=tf.paragraphs[0]; para.alignment=align
    if isinstance(runs,str): runs=[(runs,{})]
    for t,o in runs:
        r=para.add_run(); r.text=t; f=r.font
        f.name=F; f.size=Pt(o.get("size",size)); f.bold=o.get("bold",bold)
        f.italic=o.get("italic",italic); f.color.rgb=o.get("color",color)
    return tb

def card(s,x,y,w,h,fill):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,I(x),I(y),I(w),I(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=fill; sh.line.fill.background()
    # python-pptx's shadow.inherit=False still leaves the theme shadow in
    # LibreOffice; an explicit empty effect list is what actually clears it.
    from lxml import etree
    spPr=sh._element.spPr
    for e in spPr.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}effectLst'):
        spPr.remove(e)
    spPr.append(etree.SubElement(spPr,'{http://schemas.openxmlformats.org/drawingml/2006/main}effectLst'))
    try: sh.adjustments[0]=0.04
    except Exception: pass
    return sh

def head(s,title,sub):
    num[0]+=1
    txt(s,0.55,0.28,12.2,0.75,title,size=31,color=INK,bold=True)
    txt(s,0.55,1.05,12.2,0.4,sub,size=14,color=GREY)
    txt(s,12.35,6.98,0.6,0.3,str(num[0]),size=10,color=C(0x9A,0x9A,0xA8),align=PP_ALIGN.RIGHT)

def slide(): return prs.slides.add_slide(BLANK)

def table(s, x, y, w, data, colw, size=12, hl_col=None, hl_row=None, row_h=0.4):
    tb = s.shapes.add_table(len(data), len(data[0]), I(x), I(y), I(w), I(row_h*len(data))).table
    for c, cw in zip(tb.columns, colw): c.width = I(cw)
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tb.cell(ri, ci); cell.text = ""
            pa = cell.text_frame.paragraphs[0]
            pa.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = pa.add_run(); r.text = val; r.font.name = F; r.font.size = Pt(size)
            r.font.bold = (ri == 0) or (hl_row is not None and ri == hl_row)
            r.font.color.rgb = INK if ri == 0 else (RED if ci == hl_col else BODY)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_N if ri == 0 else C(0xFF, 0xFF, 0xFF)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return tb

def note(s, x, y, w, h, fill, title, body, tcol=INK, size=12.5):
    card(s, x, y, w, h, fill)
    txt(s, x+0.28, y+0.14, w-0.56, h-0.2,
        [(title + "  ", {"bold": True, "color": tcol}), (body, {})], size=size)

# ------------------------------------------------------------------ 1 ------
s = slide(); head(s, "Every whiteboard question now has an answer",
                  "Four questions, each tested with real experiments rather than argued")
table(s, 0.55, 1.65, 12.2, [
    ["Question", "What we found", "Answer"],
    ["1. Can a pretrained model beat standard ML on our patient cohorts?",
     "Standard ML wins in all 24 comparisons, even after fine-tuning", "No"],
    ["2. Is the amount of data holding us back?",
     "More spectra stop helping at ~2,000. More studies keep helping", "Not the amount - the variety"],
    ["3. Can synthetic spectra fill the gap?",
     "Neither method produces spectra realistic enough to train on", "Not yet"],
    ["4. Would a bigger model help?",
     "Slightly, then it levels off - still behind standard ML", "Not enough"],
], [4.6, 5.3, 2.3], size=13, hl_col=2, row_h=0.72)
note(s, 0.55, 5.55, 12.2, 1.15, CARD_B, "Bottom line:",
     "with the public data available today, standard ML is the better tool for these small "
     "clinical cohorts. What would change that is data from many more independent studies - "
     "not more spectra from the studies we already have.", size=14)
s.notes_slide.notes_text_frame.text = (
    "Accuracy throughout is balanced accuracy, so an uneven split between patient groups "
    "does not flatter the score. 'Standard ML' is the best of eight classical models "
    "(regularised logistic regression, SVMs, random forest, boosting, PLS-DA, LDA) on binned spectra.")

# ------------------------------------------------------------------ 2 ------
s = slide(); head(s, "1. Standard ML beats the pretrained model on every cohort",
                  "Six patient comparisons, from 5 labelled patients per group up to all of them")
s.shapes.add_picture(D + "fig7_G1_per_cohort_slim.png", I(0.55), I(1.5), width=I(8.2))
txt(s, 0.55, 4.62, 8.2, 0.3, "Red = standard ML.  Blues = our pretrained model, three ways of pretraining it.  Error bars: spread across 5 runs.",
    size=10.5, color=GREY, italic=True)
txt(s, 9.05, 1.55, 3.7, 0.35, "Does retraining the model help?", size=14, color=INK, bold=True)
table(s, 9.05, 1.98, 3.7, [
    ["How much we retrain", "vs frozen"],
    ["Nothing (frozen)", "-"],
    ["Last layer only", "-0.009"],
    ["Whole model", "-0.010"],
], [2.35, 1.35], size=12, row_h=0.42)
txt(s, 9.05, 3.8, 3.7, 1.0, "Retraining on so few patients makes it slightly worse - it memorises "
    "rather than learns.", size=11.5)
note(s, 0.55, 5.25, 6.0, 1.05, CARD_R, "Standard ML wins 24 of 24.",
     "Average margin 0.055 in accuracy, at every number of labelled patients - "
     "including the smallest, where pretraining was supposed to help most.", tcol=RED)
note(s, 6.75, 5.25, 6.0, 1.05, CARD_G, "But pretraining is not useless.",
     "The same model started from scratch does 0.057 worse. It learns something real "
     "from the spectra - just less than standard ML extracts directly.", tcol=GREEN)

# ------------------------------------------------------------------ 3 ------
s = slide(); head(s, "2. Our '12 studies' behave like about 4",
                  "One study supplies more than half of all the spectra")
s.shapes.add_picture(D + "pi25_study_share.png", I(0.55), I(1.5), width=I(7.7))
note(s, 8.55, 1.6, 4.2, 1.75, CARD_B, "Why this matters.",
     "A model learns the variety it sees. Spectra from the same study are very alike, so "
     "5,000 spectra from one study teach it much less than 5,000 spread across many.", size=12.5)
note(s, 8.55, 3.55, 4.2, 2.0, CARD_N, "Effective number of studies.",
     "Counts studies by how much they actually contribute. Twelve equal studies score 12. "
     "Ours - one giant study plus a long tail of tiny ones - scores 4.1. This is the number "
     "that matters on the next slide.", size=12.5)

# ------------------------------------------------------------------ 4 ------
s = slide(); head(s, "2. More studies help a little. More spectra don't",
                  "45 pretraining runs, varying one thing at a time")
s.shapes.add_picture(D + "pi25_spectra_vs_studies.png", I(0.55), I(1.45), width=I(12.2))
note(s, 0.55, 6.18, 12.2, 0.9, CARD_B, "So the limit is variety, not volume.",
     "7x more spectra from the same studies gave no gain. Our corpus can't go beyond about 7 "
     "effective studies, and even there the model is still below standard ML (dashed line).", size=12.5)

# ------------------------------------------------------------------ 5 ------
s = slide(); head(s, "3. Synthetic spectra: not realistic enough to use yet",
                  "We tested both routes the whiteboard names")
s.shapes.add_picture(D + "pi25_synthetic_examples.png", I(0.55), I(1.45), width=I(7.4))
table(s, 8.25, 1.55, 4.5, [
    ["", "GISSMO", "GAN"],
    ["Can a simple test tell it's fake?", "Always (1.00)", "Usually (0.95)"],
    ["Distance from real (lower = better)", "4.3", "1.2"],
    ["Peaks in the right places?", "No", "Yes"],
], [2.2, 1.15, 1.15], size=11.5, row_h=0.55)
txt(s, 8.25, 3.85, 4.5, 0.5, "0.50 on the first row would mean indistinguishable from real.",
    size=10.5, color=GREY, italic=True)
note(s, 0.55, 5.15, 6.0, 1.05, CARD_N, "GISSMO",
     "builds spectra from 87 known metabolites. But those explain only a quarter of a real "
     "serum spectrum, so the rest is missing.", size=12)
note(s, 6.75, 5.15, 6.0, 1.05, CARD_B, "The GAN",
     "learns from real spectra and looks far more realistic - but mostly copies the studies "
     "it was trained on, so it adds volume, not variety.", tcol=BLUE, size=12)

# ------------------------------------------------------------------ 6 ------
s = slide(); head(s, "4. A bigger model helps a little, then levels off",
                  "Five model sizes from 1.8 to 23.5 million parameters, 3 runs each")
s.shapes.add_picture(D + "pi25_model_size.png", I(0.55), I(1.5), width=I(8.3))
table(s, 9.15, 1.6, 3.6, [
    ["Model size", "Accuracy"],
    ["1.8 M", "0.644"],
    ["4.0 M (current)", "0.666"],
    ["10.5 M", "0.678"],
    ["23.5 M", "0.675"],
    ["Standard ML", "0.721"],
], [2.1, 1.5], size=12, hl_row=5, row_h=0.42)
note(s, 0.55, 5.55, 12.2, 0.9, CARD_R, "Gain of about 0.01 per doubling, flat after 10 M.",
     "Closing the gap would take a model hundreds of times larger, and we don't have the "
     "data to train one. We reach the whiteboard's final box: a ceiling at this data scale.",
     tcol=RED, size=12.5)

# ------------------------------------------------------------------ 7 ------
s = slide(); head(s, "Corrections to last week, and what we caught this week",
                  "Each of these produced believable numbers rather than an obvious error")
rows = [
    ("Last week, slide 6: 'diversity beats quantity'",
     "Right direction, wrong reasoning. We counted studies by name (12) rather than by contribution (about 4). "
     "The corrected analysis is slide 4 today."),
    ("Last week: study-count results were missing",
     "A file-naming mismatch silently dropped them from the analysis. Now included."),
    ("This week: our prediction on model size was wrong",
     "We expected smaller models to do better. They do worse. The conclusion still holds - but it now rests on "
     "the experiment, not a guess."),
    ("This week: the first GISSMO spectra were pure noise",
     "A software function silently mishandled our ppm axis direction. Fixed and re-run; the verdict didn't change, the numbers did."),
]
y = 1.6
for t, b in rows:
    card(s, 0.55, y, 12.2, 1.1, CARD_N)
    txt(s, 0.85, y+0.13, 11.6, 0.3, t, size=13.5, color=INK, bold=True)
    txt(s, 0.85, y+0.5, 11.6, 0.55, b, size=11.5)
    y += 1.22

# ------------------------------------------------------------------ 8 ------
s = slide(); head(s, "Where this leaves us, and next steps",
                  "We have a complete, defensible result")
note(s, 0.55, 1.6, 6.0, 1.55, CARD_B, "The result.",
     "Standard ML is the better tool for small clinical NMR cohorts with today's public data. "
     "The limit is the number of independent studies. That is a clear, useful message for the field "
     "about what data to collect.", size=13)
note(s, 6.75, 1.6, 6.0, 1.55, CARD_N, "What would change it.",
     "Data from many more independent studies. A metabolite library covering far more of the serum "
     "spectrum. Cohorts in the hundreds, where retraining has enough patients to learn from.", size=13)
txt(s, 0.55, 3.5, 12.2, 0.35, "Next steps", size=16, color=INK, bold=True)
table(s, 0.55, 3.95, 12.2, [
    ["Step", "Status"],
    ["Improved GAN - more stable, keeps its best version, plus a version that mixes studies", "Running, results in ~12 h"],
    ["Paper draft from the consolidated results", "Structure and tables ready"],
    ["Your feedback: is this the framing you want for the paper?", "Needed"],
], [8.6, 3.6], size=12.5, row_h=0.46)

prs.save("docs/Weekly_Update_2026-09-25.pptx")
print("wrote docs/Weekly_Update_2026-09-25.pptx")
