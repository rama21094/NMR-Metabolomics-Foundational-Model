const {Document,Packer,Paragraph,TextRun,HeadingLevel,AlignmentType,Table,TableRow,TableCell,WidthType,ShadingType,BorderStyle}=require('docx');
const fs=require('fs');
const DXA=w=>({size:w,type:WidthType.DXA});
const F="Calibri";
const p=(text,o={})=>new Paragraph({spacing:{after:o.after??100},alignment:o.align,children:[new TextRun({text,font:F,size:o.size??19,bold:o.bold,italics:o.i,color:o.color})]});
const runs=(arr,o={})=>new Paragraph({spacing:{after:o.after??100},children:arr.map(r=>new TextRun({font:F,size:o.size??19,...r}))});
const hdr=t=>new Paragraph({spacing:{before:160,after:80},children:[new TextRun({text:t,font:F,size:21,bold:true,color:"1F3864"})]});
const cell=(t,{b,w,shade,al}={})=>new TableCell({width:DXA(w),shading:shade?{type:ShadingType.CLEAR,fill:shade}:undefined,
  children:[new Paragraph({alignment:al,spacing:{before:40,after:40},children:[new TextRun({text:t,font:F,size:17,bold:b})]})]});
const W=[2100,1500,1500,2100,2100];
const row=(c,o={})=>new TableRow({children:c.map((t,i)=>cell(t,{b:o.b,w:W[i],shade:o.shade,al:i?AlignmentType.CENTER:undefined}))});
const tbl=rows=>new Table({columnWidths:W,width:DXA(W.reduce((a,b)=>a+b,0)),rows});

const doc=new Document({styles:{default:{document:{run:{font:F,size:19}}}},sections:[{
 properties:{page:{size:{width:12240,height:15840},margin:{top:720,bottom:640,left:900,right:900}}},
 children:[
  new Paragraph({spacing:{after:40},children:[new TextRun({text:"Spectral alignment: what the whole-corpus view showed",font:F,size:26,bold:true,color:"1F3864"})]}),
  p("Checking all 9,623 pre-training spectra and the four evaluation cohorts — 15 September 2026",{i:true,size:17,color:"595959",after:160}),

  hdr("Your suggestion, implemented"),
  runs([{text:"You asked that we look at all 9,000+ spectra by eye rather than trust summary statistics. We did, but as a single image instead of a scroll: "},
        {text:"each spectrum is one pixel row",bold:true},
        {text:", so the whole corpus and all four evaluation cohorts fit on one page (Figure 1, attached). A vertical stripe is a metabolite peak; where a stripe steps sideways, those spectra are misaligned. This is both feasible at 9,623 spectra and more sensitive than scrolling, because a 30-point shift is invisible in a single trace but obvious as a step across thousands of stacked rows."}]),

  hdr("Finding 1 — the corpus was aligned by data point, not by chemical shift"),
  runs([{text:"The left panel shows a clear horizontal break part-way down the corpus. The spectra had been put on a common "},
        {text:"point count",bold:true},
        {text:" rather than a common ppm axis, so studies acquired with different parameters were silently offset against each other. Anchoring every spectrum on its near-0 ppm feature fixed it: the interquartile range of displacement fell from "},
        {text:"768 points to 35",bold:true},
        {text:" — about one linewidth. The middle panel is the result, and the right-hand strip shows the reference region itself."}]),

  hdr("Finding 2 — the cohorts were not misaligned; they were on different axes"),
  runs([{text:"Shifting the cohorts onto the corpus never worked, and the reason was not the size of the shift. Their acquisition metadata, which we obtained this week, shows different "},
        {text:"spectral widths",bold:true},
        {text:". A peak then occupies a different number of data points, and no shift can correct that — the spectra must be resampled."}],{after:120}),
  tbl([
    row(["Dataset","Spec. width","Field","Correction applied","Agreement with corpus"],{b:true,shade:"D9E2F3"}),
    row(["Pre-training corpus","20.024 ppm","600 MHz","0 ppm re-referencing","— (reference)"]),
    row(["Barth","12.0308 ppm","950 MHz","resampled x1.664","-0.11  →  +0.64"],{shade:"F2F2F2"}),
    row(["BrC-T2D","20.1587 ppm","800 MHz","resampled x0.993","+0.13  →  +0.26"]),
    row(["MTBLS326","20.02 ppm","800 MHz","-0.08 ppm offset","+0.16  \u2192  +0.68"],{shade:"F2F2F2"}),
    row(["MTBLS563","20.0 ppm","700 MHz","none needed","+0.64"]),
    row(["TBI","unknown","unknown","none — excluded","n/a"],{shade:"F2F2F2"}),
  ]),
  p("Barth was the extreme case: at 950 MHz over a 12 ppm sweep, its peaks are spread 1.66x wider in data points than the corpus's. Resampling turned the worst-agreeing cohort into one of the best. MTBLS326 needed no resampling but sat 0.08 ppm off, because its reference standard is held in a co-axial insert rather than dissolved in the serum; an external standard carries a small susceptibility offset.",{size:17,i:true,after:120}),

  hdr("What the figure still shows as unaligned"),
  runs([{text:"A band of spectra at the top and bottom of the corpus panel remains misaligned, and this is not noise: "},
        {text:"383 spectra from three studies were acquired over a 12-13 ppm sweep instead of 20 ppm",bold:true},
        {text:" (stretch factors 1.54-1.67). They are the same defect as Barth, they are identified, and the correction is the one already validated on Barth. Roughly 4% of the corpus, and fixing them is the next step."}]),

  hdr("What this changes, and what is still open"),
  runs([{text:"The corpus is now internally consistent, and four of five cohorts sit on its axis. This removes a confound that would have depressed every transfer result, so the planned experiments should be re-run on the corrected data. "},
        {text:"Still unresolved: the absolute chemical shift axis.",bold:true},
        {text:" The corpus carries no true reference standard — the feature near 0 ppm is a broad hump, not TSP — so each dataset is internally consistent but their absolute ppm values are not established. This does not affect comparisons within a dataset; it does block quantitative matching against an external reference library."}]),

  hdr("One methodological point worth flagging"),
  runs([{text:"Before the metadata arrived we tried four independent ways to recover the correct axis from the spectra alone. "},
        {text:"All four gave confident answers, and all four were wrong",bold:true},
        {text:" when checked against a case whose answer we later knew — one estimated Barth's scale factor as 0.98 where the truth is 0.60. The dense metabolite region makes a wrong alignment fit almost as well as the right one. The practical conclusion is that acquisition metadata is not optional for this project. We now have it for four of five cohorts; TBI has none, which is why it is excluded rather than corrected."}],{after:60}),
 ]}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync("docs/Alignment_Summary_2026-09-15.docx",b);console.log("wrote docs/Alignment_Summary_2026-09-15.docx");});
