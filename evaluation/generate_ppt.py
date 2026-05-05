"""
Deep-Guard Agent — Final Presentation (Clean Redesign)
======================================================
Design principles:
  · Max ~15 shapes per slide
  · Body font ≥ 16pt, titles ≥ 26pt
  · 2-column layouts (never 3+ columns of text)
  · Generous padding (0.3"+ inside cards)
  · One key idea per slide
"""

from __future__ import annotations
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
IMG_DIR  = BASE / "results"
OUT_PATH = BASE / "results" / "DeepGuard_Final_Presentation.pptx"

# ── Palette ───────────────────────────────────────────────────────────────────
DARK  = RGBColor(0x0F,0x17,0x2A)
NAVY  = RGBColor(0x1E,0x3A,0x5F)
BLUE  = RGBColor(0x3B,0x82,0xF6)
LBLUE = RGBColor(0x93,0xC5,0xFD)
GREEN = RGBColor(0x22,0xC5,0x5E)
AMBER = RGBColor(0xF5,0x9E,0x0B)
RED   = RGBColor(0xEF,0x44,0x44)
PURP  = RGBColor(0x7C,0x3A,0xED)
WHITE = RGBColor(0xFF,0xFF,0xFF)
LIGHT = RGBColor(0xF1,0xF5,0xF9)
SLATE = RGBColor(0x64,0x74,0x8B)
TEXT  = RGBColor(0x1E,0x29,0x3B)
SUB   = RGBColor(0x47,0x55,0x69)

W, H = Inches(13.33), Inches(7.5)


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def new_prs():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def bg(slide, color):
    f = slide.background.fill
    f.solid(); f.fore_color.rgb = color

def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(1, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line: s.line.color.rgb = line
    else:    s.line.fill.background()
    return s

def txt(slide, text, x, y, w, h, size=18, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    p  = tf.paragraphs[0]; p.alignment = align
    r  = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color; r.font.italic = italic
    return tb

def img(slide, path, x, y, w, h=None):
    if h: slide.shapes.add_picture(str(path), x, y, w, h)
    else: slide.shapes.add_picture(str(path), x, y, w)

def hbar(slide, color=BLUE):           # bottom accent bar
    rect(slide, 0, H - Inches(0.07), W, Inches(0.07), color)

def header(slide, title, subtitle="", bg_color=DARK):
    rect(slide, 0, 0, W, Inches(1.2), bg_color)
    txt(slide, title, Inches(0.5), Inches(0.1), Inches(11.5), Inches(0.75),
        size=28, bold=True, color=WHITE)
    if subtitle:
        txt(slide, subtitle, Inches(0.5), Inches(0.82), Inches(10), Inches(0.35),
            size=12, color=LBLUE)

def slide_no(slide, n, total=12):
    txt(slide, f"{n} / {total}", W-Inches(1.2), H-Inches(0.38),
        Inches(1.1), Inches(0.3), size=10, color=SLATE, align=PP_ALIGN.RIGHT)

def card(slide, x, y, w, h, fill=LIGHT, border=BLUE, lw_pt=1.2):
    s = slide.shapes.add_shape(1, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = border; s.line.width = Pt(lw_pt)
    return s

def bullet_tf(slide, items, x, y, w, h, size=16, dot=BLUE, text_color=TEXT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(5)
        d = p.add_run(); d.text = "▸  "
        d.font.size = Pt(size); d.font.color.rgb = dot; d.font.bold = True
        r = p.add_run(); r.text = item
        r.font.size = Pt(size); r.font.color.rgb = text_color


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
prs = new_prs()
sl = blank(prs); bg(sl, DARK)

rect(sl, 0, 0, Inches(0.22), H, BLUE)          # left stripe
rect(sl, Inches(0.22), H*0.58, W, Inches(0.03), NAVY)  # thin separator

txt(sl, "IS596  ·  FINAL PROJECT",
    Inches(0.55), Inches(0.9), Inches(10), Inches(0.45),
    size=12, bold=True, color=LBLUE)

txt(sl, "Deep-Guard Agent",
    Inches(0.55), Inches(1.45), Inches(10.5), Inches(1.4),
    size=58, bold=True, color=WHITE)

txt(sl, "Real-Time Audio-Visual Deepfake Detection\nwith Forensic Legal Reporting",
    Inches(0.55), Inches(3.05), Inches(10), Inches(1.0),
    size=24, color=LBLUE)

# 4 stat pills at bottom
pills = [("70/100","SUS Score"), ("85%","Accuracy"), ("+30pp","Improvement"), ("N=10","User Study")]
for i,(val,lbl) in enumerate(pills):
    px = Inches(0.55 + i*3.1)
    rect(sl, px, Inches(5.1), Inches(2.8), Inches(1.6), NAVY)
    txt(sl, val, px, Inches(5.2), Inches(2.8), Inches(0.8),
        size=30, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    txt(sl, lbl, px, Inches(5.98), Inches(2.8), Inches(0.4),
        size=12, color=SLATE, align=PP_ALIGN.CENTER)

hbar(sl, BLUE); slide_no(sl, 1)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — MOTIVATION  (Originality & Significance)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Why Deepfake Detection Matters",
       "ORIGINALITY & SIGNIFICANCE")

# 3 big cards
cards_data = [
    (BLUE,  "📈  Explosive Growth",
     "Deepfake incidents up 3,000% since 2019.\n96% of deepfakes target individuals\nnon-consensually (Sensity AI, 2023)."),
    (PURP,  "⚖️  Legal Urgency",
     "EU AI Act Art.50 (Aug 2026) & Take It\nDown Act 2025 now require verifiable\nforensic evidence of synthetic media."),
    (GREEN, "🔍  Detection Gap",
     "Existing tools are single-modality and\nblack-box — no tool combines AV detection\nwith legally admissible documentation."),
]
for i,(color,title,body) in enumerate(cards_data):
    cx = Inches(0.45 + i*4.28)
    rect(sl, cx, Inches(1.35), Inches(4.05), Inches(0.12), color)
    card(sl, cx, Inches(1.47), Inches(4.05), Inches(4.6), fill=LIGHT, border=color)
    txt(sl, title, cx+Inches(0.25), Inches(1.65), Inches(3.6), Inches(0.6),
        size=18, bold=True, color=color)
    txt(sl, body, cx+Inches(0.25), Inches(2.35), Inches(3.6), Inches(3.0),
        size=15, color=SUB)

# Bottom callout
rect(sl, Inches(0.45), Inches(6.3), Inches(12.45), Inches(0.92), BLUE)
txt(sl, "Deep-Guard Agent fills all three gaps: bimodal AV detection + explainable AI + ISO/IEC 27037-compliant legal reporting — in a single open system.",
    Inches(0.65), Inches(6.42), Inches(12.1), Inches(0.7),
    size=15, bold=True, color=WHITE)

hbar(sl); slide_no(sl, 2)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — SYSTEM ARCHITECTURE  (Originality)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "System Architecture — 6-Module Pipeline",
       "ORIGINALITY & SIGNIFICANCE")

# Two-row pipeline: Row 1 = input+visual+audio, Row 2 = fusion+reasoner+output
row1 = [
    (NAVY,  "① Video Input",     "MP4 / AVI / MOV"),
    (BLUE,  "② Visual Encoder",  "MediaPipe FaceLandmarker\n256-d lip feature vector"),
    (PURP,  "③ Audio Encoder",   "Wav2Vec2-base-960h\n768-d @ 50 fps"),
]
row2 = [
    (RGBColor(0x06,0xB6,0xD4), "④ Cross-Modal Fusion", "Temporal Transformer\n+ cosine similarity"),
    (AMBER, "⑤ LLM Reasoner",   "LLaMA 3.3 70B (Groq)\nchain-of-thought + JSON"),
    (GREEN, "⑥ Output",         "HTML Report · Legal Report\nAnnotated Video · JSON"),
]

# Layout constants — wider gaps so no visual crowding
BW    = Inches(3.5)    # box width  (was 3.85)
BH    = Inches(1.75)   # box height (was 1.85)
CGAP  = Inches(0.55)   # column gap (was 0.28)
RGAP  = Inches(0.65)   # row gap
SX    = Inches(0.55)   # left margin

# Row y-positions
ROW_Y = [Inches(1.40), Inches(1.40) + BH + RGAP]   # [1.40", 3.80"]
# Row 0 bottom = 3.15"  Row 1 bottom = 5.55"  → no overlap with formula bar at 5.72"

for row_idx, row in enumerate([row1, row2]):
    ry = ROW_Y[row_idx]
    for col_idx, (color, title, body) in enumerate(row):
        bx = SX + col_idx * (BW + CGAP)
        rect(sl, bx, ry, BW, BH, color)
        txt(sl, title,
            bx + Inches(0.22), ry + Inches(0.14),
            BW - Inches(0.44), Inches(0.52),
            size=16, bold=True, color=WHITE)
        txt(sl, body,
            bx + Inches(0.22), ry + Inches(0.72),
            BW - Inches(0.44), Inches(0.92),
            size=13, color=LIGHT)
        # right arrow — centred in the column gap
        if col_idx < 2:
            ax = bx + BW + Inches(0.12)
            txt(sl, "▶", ax, ry + BH/2 - Inches(0.25),
                CGAP - Inches(0.24), Inches(0.5),
                size=16, color=SLATE, align=PP_ALIGN.CENTER)

# Down arrows — centred below each column of row 0
col_centers = [SX + col_idx * (BW + CGAP) + BW / 2 for col_idx in range(3)]
arrow_y = ROW_Y[0] + BH + Inches(0.08)   # just below row 0
arrow_h = RGAP - Inches(0.16)
for cx in col_centers:
    txt(sl, "▼", cx - Inches(0.5), arrow_y,
        Inches(1.0), arrow_h,
        size=18, color=SLATE, align=PP_ALIGN.CENTER)

# Key formula bar — safely below row 1 (row 1 bottom = 5.55")
rect(sl, Inches(0.55), Inches(5.72), Inches(12.3), Inches(0.60), LIGHT)
txt(sl,
    "Fusion:  score = 0.7 × learned_discrepancy  +  0.3 × cosine_discrepancy"
    "     |     Frame flagged if score ≥ 0.65",
    Inches(0.75), Inches(5.79), Inches(11.9), Inches(0.45),
    size=13, italic=True, color=TEXT)

hbar(sl); slide_no(sl, 3)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — THEORETICAL GROUNDING
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Theoretical Foundations",
       "THEORETICAL GROUNDING")

theories = [
    (BLUE,  "AV-HuBERT  (Shi et al., 2022)",
     "Self-supervised audio-visual speech learning.\nInspired our cross-modal alignment architecture\nand temporal synchrony objective."),
    (PURP,  "Wav2Vec 2.0  (Baevski et al., 2020)",
     "Contrastive pre-training on raw waveforms.\nExtracts articulatory features grounded in\nvocal tract physics at 50 fps resolution."),
    (GREEN, "ART-AVDF  (Wang & Huang, 2024)",
     "Direct precedent: articulatory representations\nfor audio-visual deepfake detection — our\nprimary methodological reference."),
    (AMBER, "Daubert Standard  &  ISO/IEC 27037",
     "Legal admissibility framework for scientific\nevidence. Requires testability, error rates,\nand documented chain of custody."),
]
positions = [(Inches(0.5), Inches(1.4)), (Inches(6.95), Inches(1.4)),
             (Inches(0.5),  Inches(4.2)), (Inches(6.95), Inches(4.2))]
cw, ch = Inches(6.05), Inches(2.55)

for (cx,cy),(color,title,body) in zip(positions, theories):
    rect(sl, cx, cy, cw, Inches(0.1), color)
    card(sl, cx, cy+Inches(0.1), cw, ch-Inches(0.1), fill=LIGHT, border=color, lw_pt=0.8)
    txt(sl, title, cx+Inches(0.25), cy+Inches(0.22), cw-Inches(0.35), Inches(0.55),
        size=17, bold=True, color=color)
    txt(sl, body,  cx+Inches(0.25), cy+Inches(0.85), cw-Inches(0.35), Inches(1.55),
        size=14, color=TEXT)

# Physical insight banner
rect(sl, Inches(0.5), Inches(7.0), Inches(12.4), Inches(0.32), RGBColor(0xF0,0xFD,0xF4))
txt(sl, "Key physical insight: Bilabial phonemes (B, P, M) require lip closure — deepfakes violate this constraint, creating measurable audio-visual discrepancies.",
    Inches(0.65), Inches(7.02), Inches(12.0), Inches(0.28),
    size=12, bold=True, color=RGBColor(0x14,0x53,0x2D))

hbar(sl); slide_no(sl, 4)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — IMPLEMENTATION  (Robustness)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Complete Implementation",
       "ROBUSTNESS & IMPLEMENTATION")

# LEFT — tech stack (clean list)
rect(sl, Inches(0.5), Inches(1.35), Inches(6.1), Inches(5.85), LIGHT)
txt(sl, "Technology Stack", Inches(0.7), Inches(1.45),
    Inches(5.7), Inches(0.45), size=16, bold=True, color=DARK)

stack = [
    ("Visual Encoder",   "MediaPipe FaceLandmarker  →  256-d lip vector"),
    ("Audio Encoder",    "Wav2Vec2-base-960h  ·  768-d @ 50fps via ffmpeg"),
    ("Fusion Module",    "AudioVisualProjector + 2-layer TemporalAttention"),
    ("LLM Reasoning",    "LLaMA 3.3 70B (Groq API)  ·  JSON-structured output"),
    ("Video Overlay",    "OpenCV annotation  ·  O(1) set-lookup for flags"),
    ("HTML Report",      "Pure Python SVG charts  ·  html.escape() XSS guard"),
    ("Legal Report",     "9 sections  ·  SHA-256 chain of custody"),
    ("Interface",        "Gradio 4.x  ·  thread-safe  ·  UUID output files"),
]
for i,(comp,desc) in enumerate(stack):
    cy = Inches(2.05 + i*0.62)
    rect(sl, Inches(0.7), cy+Inches(0.1), Inches(0.06), Inches(0.3), BLUE)
    txt(sl, comp+":", Inches(0.85), cy, Inches(1.7), Inches(0.5),
        size=13, bold=True, color=BLUE)
    txt(sl, desc,    Inches(2.6),  cy, Inches(3.8), Inches(0.5),
        size=13, color=TEXT)

# RIGHT — 4 key engineering decisions
rect(sl, Inches(6.85), Inches(1.35), Inches(6.05), Inches(5.85), RGBColor(0xEF,0xF6,0xFF))
txt(sl, "Key Engineering Decisions", Inches(7.05), Inches(1.45),
    Inches(5.65), Inches(0.45), size=16, bold=True, color=DARK)

decisions = [
    (BLUE,  "🔒 Thread Safety",
     "Double-checked locking ensures\nonly one pipeline init under\nconcurrent Gradio requests."),
    (GREEN, "⚡ O(1) Frame Lookup",
     "Pre-computed set() for flagged\nframes — avoids O(n) list scan\nper video frame in render loop."),
    (RED,   "🛡️ XSS Prevention",
     "All LLM-generated content passed\nthrough html.escape() before\nHTML injection."),
    (AMBER, "📊 Pure SVG Charts",
     "No JavaScript — Gradio strips\n<script> tags. All charts are\nserver-side Python SVG."),
]
for i,(color,title,body) in enumerate(decisions):
    cy = Inches(2.05 + i*1.27)
    rect(sl, Inches(7.05), cy, Inches(0.12), Inches(1.05), color)
    txt(sl, title, Inches(7.28), cy+Inches(0.04), Inches(5.4), Inches(0.42),
        size=14, bold=True, color=color)
    txt(sl, body,  Inches(7.28), cy+Inches(0.48), Inches(5.4), Inches(0.6),
        size=13, color=TEXT)

hbar(sl); slide_no(sl, 5)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — LEGAL REPORT  (Robustness)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Legal Forensic Report — Novel Contribution",
       "ROBUSTNESS & IMPLEMENTATION")

# 3×3 section grid
sections = [
    ("① Case Information",      BLUE),
    ("② Forensic Verdict",      BLUE),
    ("③ Methodology",           BLUE),
    ("④ Quantitative Findings", PURP),
    ("⑤ Discrepancy Timeline",  PURP),
    ("⑥ Flagged Frames Table",  PURP),
    ("⑦ Evidence Summary",      GREEN),
    ("⑧ Chain of Custody",      GREEN),
    ("⑨ Limitations & Disclaimer", GREEN),
]
sw, sh = Inches(3.95), Inches(1.3)
sgap   = Inches(0.22)
sx_    = Inches(0.5)
sy_    = Inches(1.4)
for i,(label,color) in enumerate(sections):
    col = i % 3; row = i // 3
    bx = sx_ + col*(sw+sgap)
    by = sy_ + row*(sh+Inches(0.2))
    rect(sl, bx, by, Inches(0.12), sh, color)
    card(sl, bx+Inches(0.12), by, sw-Inches(0.12), sh, fill=LIGHT, border=LIGHT)
    txt(sl, label, bx+Inches(0.28), by+Inches(0.38),
        sw-Inches(0.4), Inches(0.6), size=15, bold=True, color=TEXT)

# Frameworks row
rect(sl, Inches(0.5), Inches(6.1), Inches(12.4), Inches(1.15), DARK)
txt(sl, "Referenced Legal Frameworks",
    Inches(0.7), Inches(6.18), Inches(12.0), Inches(0.38),
    size=13, bold=True, color=LBLUE)
fw = "EU AI Act Art.50   ·   Take It Down Act 2025   ·   NIST SP 800-86   ·   ISO/IEC 27037:2012   ·   NIST OpenMFC   ·   Daubert Standard"
txt(sl, fw, Inches(0.7), Inches(6.58), Inches(12.0), Inches(0.55),
    size=13, color=SLATE)

hbar(sl); slide_no(sl, 6)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — EVALUATION DESIGN  (Evaluation)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Evaluation Study Design",
       "EVALUATION & FINDINGS")

# LEFT — params
rect(sl, Inches(0.5), Inches(1.35), Inches(5.8), Inches(5.85), LIGHT)
txt(sl, "Study Parameters", Inches(0.7), Inches(1.48),
    Inches(5.4), Inches(0.42), size=16, bold=True, color=DARK)

params = [
    ("Participants",  "N = 10 university students"),
    ("Backgrounds",   "5 technical (CS/IS)  ·  5 non-technical"),
    ("Design",        "Within-subjects, single session"),
    ("Duration",      "~60 minutes per session"),
    ("Stimuli",       "3 videos: Authentic · Likely Fake · Suspicious"),
    ("Instruments",   "SUS · Likert (1-7) · Semi-structured interview"),
    ("Baseline",      "Unaided detection before system use"),
]
for i,(k,v) in enumerate(params):
    cy = Inches(2.05 + i*0.71)
    txt(sl, k + ":", Inches(0.7), cy, Inches(1.65), Inches(0.55),
        size=14, bold=True, color=BLUE)
    txt(sl, v,       Inches(2.45), cy, Inches(3.7), Inches(0.55),
        size=14, color=TEXT)

# RIGHT — session timeline
rect(sl, Inches(6.55), Inches(1.35), Inches(6.3), Inches(5.85), RGBColor(0xEF,0xF6,0xFF))
txt(sl, "60-Minute Session Flow", Inches(6.75), Inches(1.48),
    Inches(5.9), Inches(0.42), size=16, bold=True, color=DARK)

flow = [
    ("00:00", "Welcome & Consent",            "5 min",  SLATE),
    ("00:08", "Pre-Questionnaire + Baseline", "7 min",  SLATE),
    ("00:15", "Task 1 — Authentic Video",     "8 min",  BLUE),
    ("00:23", "Task 2 — Likely Fake Video",   "8 min",  BLUE),
    ("00:31", "Task 3 — Suspicious Video",    "8 min",  BLUE),
    ("00:39", "Post-Task SUS Questionnaire",  "8 min",  AMBER),
    ("00:47", "Semi-Structured Interview",    "10 min", AMBER),
    ("00:57", "Debrief & Close",              "3 min",  SLATE),
]
for i,(t,step,dur,color) in enumerate(flow):
    cy = Inches(2.05 + i*0.64)
    rect(sl, Inches(6.75), cy+Inches(0.08), Inches(0.82), Inches(0.38), RGBColor(0xDB,0xEA,0xFE))
    txt(sl, t,    Inches(6.77), cy+Inches(0.1),  Inches(0.78), Inches(0.32),
        size=10, bold=True, color=color, align=PP_ALIGN.CENTER)
    txt(sl, step, Inches(7.68), cy, Inches(4.35), Inches(0.52), size=13, color=TEXT)
    txt(sl, dur,  Inches(12.2), cy, Inches(0.5),  Inches(0.52),
        size=11, color=SLATE, align=PP_ALIGN.RIGHT)

hbar(sl, GREEN); slide_no(sl, 7)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — SUS RESULTS  (Evaluation)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Results: System Usability Scale (SUS)",
       "EVALUATION & FINDINGS")

# 4 KPI boxes
kpis = [("70.0","Mean SUS / 100",BLUE), ("Above Avg","Overall Grade",GREEN),
        ("5 / 10","Scored ≥ 68",AMBER), ("83.8 vs 61.3","CS vs Non-tech",RED)]
for i,(val,lbl,color) in enumerate(kpis):
    kx = Inches(0.5 + i*3.2)
    rect(sl, kx, Inches(1.35), Inches(3.0), Inches(1.05), DARK)
    txt(sl, val, kx, Inches(1.4), Inches(3.0), Inches(0.62),
        size=28, bold=True, color=color, align=PP_ALIGN.CENTER)
    txt(sl, lbl, kx, Inches(2.0), Inches(3.0), Inches(0.32),
        size=11, color=SLATE, align=PP_ALIGN.CENTER)

# SUS chart (full width)
if (IMG_DIR/"fig1_sus_scores.png").exists():
    img(sl, IMG_DIR/"fig1_sus_scores.png",
        Inches(0.5), Inches(2.55), Inches(12.35), Inches(4.65))

hbar(sl, GREEN); slide_no(sl, 8)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — ACCURACY + LIKERT  (Evaluation)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Results: Accuracy & Likert Scales",
       "EVALUATION & FINDINGS")

# Agreement chart (left)
if (IMG_DIR/"fig3_agreement_rates.png").exists():
    img(sl, IMG_DIR/"fig3_agreement_rates.png",
        Inches(0.5), Inches(1.35), Inches(6.5), Inches(4.1))

# Likert bars (right)
rect(sl, Inches(7.2), Inches(1.35), Inches(5.65), Inches(4.1), LIGHT)
txt(sl, "Custom Likert Scales  (1–7)", Inches(7.4), Inches(1.48),
    Inches(5.25), Inches(0.42), size=15, bold=True, color=DARK)

likerts = [("Report Comprehension",  4.17, BLUE),
           ("Legal Report Useful",   4.60, GREEN),
           ("Trust in AI Verdict",   4.72, AMBER),
           ("Overall Satisfaction",  4.65, RED)]
for i,(label,val,color) in enumerate(likerts):
    cy = Inches(2.1 + i*0.78)
    txt(sl, label, Inches(7.4), cy, Inches(3.3), Inches(0.38), size=13, color=TEXT)
    bar_w = val/7.0
    rect(sl, Inches(7.4), cy+Inches(0.43), Inches(4.9), Inches(0.2),
         RGBColor(0xE2,0xE8,0xF0))
    rect(sl, Inches(7.4), cy+Inches(0.43), Inches(4.9*bar_w), Inches(0.2), color)
    txt(sl, f"{val:.2f}", Inches(12.15), cy+Inches(0.37), Inches(0.6), Inches(0.3),
        size=13, bold=True, color=color, align=PP_ALIGN.RIGHT)

# Bottom: key takeaway row
rect(sl, Inches(0.5), Inches(5.65), Inches(12.35), Inches(1.55), DARK)
for i,(val,lbl,color) in enumerate([
    ("55%",  "Unaided Baseline", SLATE),
    ("→","", WHITE),
    ("85%",  "System-Assisted", GREEN),
    ("+30pp","Improvement",     AMBER),
    ("0.97", "Spearman ρ (AI Familiarity ↔ SUS)", LBLUE)
]):
    px = Inches(0.7 + i*2.42)
    txt(sl, val, px, Inches(5.75), Inches(2.3), Inches(0.65),
        size=26 if val!="→" else 28, bold=True, color=color, align=PP_ALIGN.CENTER)
    txt(sl, lbl, px, Inches(6.38), Inches(2.3), Inches(0.72),
        size=10, color=SLATE, align=PP_ALIGN.CENTER)

hbar(sl, GREEN); slide_no(sl, 9)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — INTERVIEW THEMES + CORRELATION  (Evaluation)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Results: Qualitative Themes & Correlation",
       "EVALUATION & FINDINGS")

# Theme chart (left)
if (IMG_DIR/"fig5_interview_themes.png").exists():
    img(sl, IMG_DIR/"fig5_interview_themes.png",
        Inches(0.5), Inches(1.35), Inches(6.8), Inches(4.2))

# Correlation chart (right)
if (IMG_DIR/"fig6_sus_vs_familiarity.png").exists():
    img(sl, IMG_DIR/"fig6_sus_vs_familiarity.png",
        Inches(7.5), Inches(1.35), Inches(5.35), Inches(4.2))

# Bottom: top themes
rect(sl, Inches(0.5), Inches(5.75), Inches(12.35), Inches(1.48), LIGHT)
top_themes = [
    (GREEN, "✅  Clear Verdict",          "8 / 10"),
    (GREEN, "✅  Legal Report Useful",    "7 / 10"),
    (RED,   "⚠️  Confusing Terms",        "6 / 10"),
    (RED,   "⚠️  Processing Too Slow",    "4 / 10"),
]
for i,(color,theme,cnt) in enumerate(top_themes):
    tx = Inches(0.7 + i*3.1)
    txt(sl, theme, tx, Inches(5.88), Inches(2.9), Inches(0.5), size=13, color=TEXT)
    txt(sl, cnt,   tx, Inches(6.42), Inches(2.9), Inches(0.5),
        size=22, bold=True, color=color)

hbar(sl, GREEN); slide_no(sl, 10)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — DISCUSSION & REFLECTION
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, WHITE)
header(sl, "Discussion & Reflection",
       "DISCUSSION & REFLECTION")

cols = [
    (GREEN, "✅  What Worked",
     ["AV bimodal detection catches\nmismatches single-modality misses",
      "LLM reasoning gives grounded,\nexplainable verdicts (not just a score)",
      "Legal Report fills a real gap —\nvalued by law/journalism users",
      "SUS 70/100 confirms usability\nfor majority of participants"]),
    (RED,   "⚠️  Limitations",
     ["No trained checkpoint — random\ninit means uncalibrated scores",
      "Non-technical users face vocab\nbarrier (6/10 flagged in interviews)",
      "~5 min processing limits real-time\nmoderation use cases",
      "N=10 is sufficient for formative\nstudy but not definitive"]),
    (BLUE,  "🔭  Future Work",
     ["Train on FaceForensics++ / DFDC\nfor calibrated detection",
      "Add inline tooltips / glossary\nfor non-expert users",
      "Explore streaming inference for\nlive deepfake detection",
      "Jurisdiction-specific legal\nreport templates (EU / US / CN)"]),
]
cw_, ch_ = Inches(4.0), Inches(5.6)
for i,(color,title,items) in enumerate(cols):
    cx = Inches(0.5 + i*4.27)
    rect(sl, cx, Inches(1.35), cw_, Inches(0.12), color)
    card(sl, cx, Inches(1.47), cw_, ch_, fill=LIGHT, border=color, lw_pt=0.8)
    txt(sl, title, cx+Inches(0.25), Inches(1.58), cw_-Inches(0.35), Inches(0.5),
        size=17, bold=True, color=color)
    for j,item in enumerate(items):
        cy = Inches(2.22 + j*1.18)
        rect(sl, cx+Inches(0.25), cy+Inches(0.15), Inches(0.09), Inches(0.3), color)
        txt(sl, item, cx+Inches(0.45), cy, cw_-Inches(0.6), Inches(1.05),
            size=13, color=TEXT)

hbar(sl, AMBER); slide_no(sl, 11)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
sl = blank(prs); bg(sl, DARK)
rect(sl, 0, 0, Inches(0.22), H, GREEN)
rect(sl, Inches(0.22), Inches(4.6), W, Inches(0.04), NAVY)

txt(sl, "Thank You", Inches(0.55), Inches(0.8), Inches(9), Inches(1.2),
    size=52, bold=True, color=WHITE)
txt(sl, "Deep-Guard Agent  —  Key Takeaways",
    Inches(0.55), Inches(2.05), Inches(10), Inches(0.5),
    size=20, color=LBLUE)

takeaways = [
    (BLUE,  "Originality",      "First tool combining AV detection + LLM reasoning + ISO legal reporting"),
    (PURP,  "Theory",           "Grounded in Wav2Vec2, AV-HuBERT, ART-AVDF — physical acoustics as signal"),
    (GREEN, "Implementation",   "6-module pipeline: thread-safe, XSS-protected, UUID outputs, SVG charts"),
    (AMBER, "Evaluation",       "SUS 70/100 · +30pp accuracy · N=10 user study"),
    (RED,   "Reflection",       "Vocabulary gap & processing speed are the next key improvement targets"),
]
for i,(color,label,text) in enumerate(takeaways):
    cy = Inches(2.75 + i*0.85)
    rect(sl, Inches(0.55), cy, Inches(1.55), Inches(0.65), NAVY)
    txt(sl, label, Inches(0.57), cy+Inches(0.12), Inches(1.51), Inches(0.42),
        size=12, bold=True, color=color, align=PP_ALIGN.CENTER)
    txt(sl, text, Inches(2.25), cy+Inches(0.1), Inches(10.0), Inches(0.52),
        size=15, color=LIGHT)

txt(sl, "IS596  ·  Deep-Guard Agent v2.0",
    Inches(0.55), Inches(7.15), Inches(8), Inches(0.3),
    size=11, color=SLATE)

hbar(sl, GREEN)

# ══════════════════════════════════════════════════════════════════════════════
prs.save(str(OUT_PATH))
print(f"✓ Saved → {OUT_PATH}")
print(f"  12 slides  |  ~10 min presentation")
