#!/usr/bin/env python3
"""Weekly PI update, 2026-09-18. Matches the house style of the 2026-09-10 deck."""
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

# ---------------------------------------------------------------- slide 1 ---
s=slide(); head(s,"The rebuild is done, verified, and it answers two gates",
                "Corpus and all four evaluation cohorts now sit on one real chemical-shift axis")
rows=[("Corpus rebuilt and verified","9,480 spectra from 18 studies, reprocessed from raw Bruker. Every spectral-width group peaks at scale 1.00 and carries the correct 2.780 ppm lactate CH3-to-CH separation.","DONE",GREEN,CARD_G),
      ("All four cohorts rebuilt","MTBLS563 142, BrC-T2D 115, MTBLS326 43, Barth 40 - byte-identical ppm axes with the corpus, each verified independently. BrC-T2D recovered 125 spectra the old pipeline discarded.","DONE",GREEN,CARD_G),
      ("Gate G1 - can pretraining beat classical ML?","No. 0 wins, 24 losses, 5 ties across 6 targets x 5 label budgets with 5 seeds. The axis defect was not the reason SSL was losing.","ANSWERED",RED,CARD_R),
      ("Gate G2 - is data limiting?","Quantity no, diversity yes. p(y|x) is pinned down for our 12 studies (99.8% of ppm bins) and does not transfer to a thirteenth (1.5%).","ANSWERED",RED,CARD_R)]
y=1.62
for t,b,badge,bc,fill in rows:
    card(s,0.55,y,12.2,1.22,fill)
    txt(s,0.85,y+0.11,9.0,0.34,t,size=15,color=INK,bold=True)
    txt(s,9.9,y+0.11,2.6,0.34,badge,size=13,color=bc,bold=True,align=PP_ALIGN.RIGHT)
    txt(s,0.85,y+0.49,11.6,0.64,b,size=11.5)
    y+=1.33

# ---------------------------------------------------------------- slide 2 ---
s=slide(); head(s,"The decisive alignment measurement",
                "Where does lactate actually land? The same measurement at each stage")
s.shapes.add_picture(D+"fig5_slim.png",I(0.55),I(1.60),width=I(12.2))
y=5.22
for x,w,fill,lead,lc,rest in [
 (0.55,3.95,CARD_N,"As stored  ",INK,"median 1.618 ppm, 95% span 0.546. Peak positions wander by half a ppm."),
 (4.68,3.95,CARD_R,"After 0 ppm anchoring  ",RED,"span 0.000 - exact agreement, at 1.616 ppm, which is not where lactate is."),
 (8.81,3.94,CARD_B,"After the rebuild  ",BLUE,"median 1.325 against a true 1.330, with 0.117 ppm of genuine inter-study spread.")]:
    card(s,x,y,w,1.30,fill)
    txt(s,x+0.25,y+0.14,w-0.5,1.0,[(lead,{"bold":True,"color":lc}),(rest,{})],size=11.5)
txt(s,0.55,6.70,12.2,0.34,"Agreement between spectra is not the same as correctness. Only the rebuild has both.",
    size=12.5,color=INK,bold=True,italic=True)

# ---------------------------------------------------------------- slide 3 ---
s=slide(); head(s,"Why inspecting 9,000 spectra could not have found it",
                "The defect exists only between rows - each spectrum looks perfectly normal alone")
s.shapes.add_picture(D+"fig3_the_defect.png",I(1.17),I(1.42),width=I(11.0))
card(s,0.55,5.72,12.2,1.0,CARD_N)
txt(s,0.85,5.86,11.6,0.76,[("940 of 9,480 spectra (9.9%) ",{"bold":True}),
 ("were acquired at a materially different sweep width. The old pipeline resampled each onto its own ppm range, so one metabolite line occupied 16 points in one row and 27 in another. Four separate 'common axis' files were written and never used.",{})],size=11.5)

# ---------------------------------------------------------------- slide 4 ---
s=slide(); head(s,"Gate G1: classical ML still wins",
                "And the axis defect was not the reason - this reproduces on correct data")
data=[["k per class","classical","masking","jigsaw","joint"],
      ["5","0.597","0.549","0.498","0.516"],["10","0.653","0.593","0.531","0.548"],
      ["20","0.705","0.633","0.561","0.580"],["40","0.738","0.672","0.587","0.611"],
      ["all","0.729","0.657","0.573","0.607"]]
tb=s.shapes.add_table(len(data),5,I(0.55),I(1.6),I(6.1),I(2.1)).table
for w,cw in zip(tb.columns,[1.35,1.25,1.25,1.15,1.1]): w.width=I(cw)
for ri,row in enumerate(data):
    for ci,val in enumerate(row):
        cell=tb.cell(ri,ci); cell.text=""
        pa=cell.text_frame.paragraphs[0]; pa.alignment=PP_ALIGN.CENTER
        r=pa.add_run(); r.text=val; r.font.name=F; r.font.size=Pt(11.5)
        r.font.bold=(ri==0); r.font.color.rgb=INK if ri==0 else (BLUE if ci==1 else BODY)
        cell.fill.solid(); cell.fill.fore_color.rgb=C(0xFF,0xFF,0xFF) if ri else CARD_N
        cell.vertical_anchor=MSO_ANCHOR.MIDDLE
txt(s,0.55,3.80,6.1,0.3,"Mean balanced accuracy, 6 targets x 5 seeds x 50 paired episodes",size=10,color=GREY,italic=True)
card(s,7.0,1.6,5.75,2.2,CARD_G)
txt(s,7.28,1.74,5.2,0.32,"Pretraining does work",size=15,color=GREEN,bold=True)
txt(s,7.28,2.14,5.2,1.55,"Against a random-init backbone of identical architecture on identical episodes, masking gains +0.081 to +0.121 balanced accuracy - consistent across all six targets, far above the 0.045 noise floor. The pretext task teaches the encoder something real. It is worth less than what a shrinkage LDA extracts from 500 spectral bins.",size=11.5)
card(s,0.55,4.28,12.2,0.98,CARD_R)
txt(s,0.85,4.42,11.6,0.74,[("0 SSL wins / 24 classical wins / 5 ties.  ",{"bold":True,"color":RED}),
 ("Classical leads at every label budget including k=5, so there is no few-shot crossover. Reproducing this on correctly aligned data closes off 'the data was broken' as the explanation.",{})],size=11.5)
card(s,0.55,5.44,12.2,1.16,CARD_N)
txt(s,0.85,5.58,11.6,0.94,[("Two claims withdrawn this week.  ",{"bold":True}),
 ("(1) Effective rank does not predict transfer - I predicted joint would win on that basis, and masking won at every budget. (2) An apparent Barth win was selection bias from reporting the best of 15 checkpoints; every objective's seed mean is below classical. That same error was retracted once before, so the summary now reports seed means and marks best-of-15 as an upper bound.",{})],size=11)

# ---------------------------------------------------------------- slide 5 ---
s=slide(); head(s,"Gate G2: data quantity is not the constraint",
                "Diversity is - and that changes what we ask the field for")
s.shapes.add_picture(D+"fig6_G2_slim.png",I(0.55),I(1.62),width=I(12.2))
card(s,0.55,5.30,6.0,1.30,CARD_B)
txt(s,0.8,5.44,5.5,1.05,[("Within our studies: defined.  ",{"bold":True,"color":BLUE}),
 ("Sampling error falls as 1/sqrt(N) and by N = 4,000 resolves 99.8% of ppm bins. More spectra from these 12 studies would change nothing.",{})],size=11.5)
card(s,6.75,5.30,6.0,1.30,CARD_R)
txt(s,7.0,5.44,5.5,1.05,[("Across studies: not defined.  ",{"bold":True,"color":RED}),
 ("Between-study variation is flat across a 20x range of N and resolves 1.5% of bins. Twelve studies do not determine a thirteenth.",{})],size=11.5)
txt(s,0.55,6.75,12.2,0.34,"A generator trained here would be sound and still reproduce a 12-study composition that does not cover a new cohort.",
    size=12.5,color=INK,bold=True,italic=True)

# ---------------------------------------------------------------- slide 6 ---
s=slide(); head(s,"Does more data, or more diverse data, transfer better?","The scaling sweep - 33 pretraining runs across three machines")
tbl2=[["pretraining corpus","k=10","k=all"],
      ["2,000 rows from 2 studies","0.541","0.600"],
      ["2,000 rows from 4 studies","0.565","0.633"],
      ["2,000 rows from 8 studies","0.565","0.629"],
      ["2,000 rows from 12 studies","0.568","0.627"],
      ["858 rows, all 12 studies","0.557","0.624"],
      ["6,410 rows, all 12 studies","0.571","0.640"],
      ["classical ML","0.658","0.732"]]
t2=s.shapes.add_table(len(tbl2),3,I(0.55),I(1.58),I(6.0),I(2.75)).table
for w,cw in zip(t2.columns,[3.4,1.3,1.3]): w.width=I(cw)
for ri,row in enumerate(tbl2):
    for ci,val in enumerate(row):
        cell=t2.cell(ri,ci); cell.text=""
        pa=cell.text_frame.paragraphs[0]
        pa.alignment=PP_ALIGN.LEFT if ci==0 else PP_ALIGN.CENTER
        r=pa.add_run(); r.text=val; r.font.name=F; r.font.size=Pt(11)
        last = (ri==len(tbl2)-1)
        r.font.bold=(ri==0 or last)
        r.font.color.rgb=INK if ri==0 else (BLUE if last else BODY)
        cell.fill.solid()
        cell.fill.fore_color.rgb=CARD_N if (ri==0 or last) else C(0xFF,0xFF,0xFF)
        cell.vertical_anchor=MSO_ANCHOR.MIDDLE
txt(s,0.55,4.42,6.0,0.3,"Masking, mean over 6 targets; seed sd 0.002-0.021",size=10,color=GREY,italic=True)
card(s,6.85,1.58,5.9,2.75,CARD_N)
txt(s,7.13,1.72,5.35,0.3,"Diversity beats quantity, but both are small",size=14,color=INK,bold=True)
txt(s,7.13,2.10,5.35,2.1,[("At a fixed 2,000-row budget, going from 2 studies to 12 gains ",{}),
 ("+0.027",{"bold":True}),(" at both label budgets - above seed noise, and it saturates after 4 studies.\n\nGoing from 858 rows to 6,410, with all 12 studies present throughout, gains only ",{}),
 ("+0.015",{}),(" and is not monotonic.\n\nSo diversity matters more than quantity, which agrees with the distributional result. But neither closes the ~0.10 gap to classical ML.",{})],size=11.5)
nx=[("What this means for gate G3","A generator trained on this corpus would reproduce a 12-study composition. The scaling result says a 12-study corpus is not much better than a 4-study one for transfer, so synthetic data drawn from it should not be expected to help. Worth one honest test, with a low prior.",CARD_R,RED),
    ("The open question this raises","Saturation at 4 studies may mean the 2,000-row budget is the binding constraint rather than study count. Re-running the fixed-budget axis at 4,000 rows would separate those, and is the next cheap experiment.",CARD_B,BLUE),
    ("Still untested","Fine-tuning rather than frozen probes. Phase 3 froze the backbone deliberately, so G1 is a verdict about representations, not about what the architecture could do if adapted.",CARD_N,INK)]
y=4.60
for t,b,fill,tc in nx:
    card(s,0.55,y,12.2,0.78,fill)
    txt(s,0.85,y+0.08,11.6,0.26,t,size=12.5,color=tc,bold=True)
    txt(s,0.85,y+0.34,11.6,0.42,b,size=10.5)
    y+=0.86

prs.save("docs/Weekly_Update_2026-09-18.pptx")
print("wrote docs/Weekly_Update_2026-09-18.pptx")
