"""
Generate a bilingual (EN | ZH) speech-script PDF for Deep-Guard Agent.
Uses only reportlab — no external CLI tools required.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import os, sys

OUT = Path(__file__).parent / "results" / "DeepGuard_Speech_Script.pdf"

# ── Try to register a CJK font so Chinese renders correctly ──────────────────
CJK_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
]
CJK_FONT = "Helvetica"   # fallback
for p in CJK_PATHS:
    if os.path.exists(p):
        try:
            pdfmetrics.registerFont(TTFont("CJK", p))
            CJK_FONT = "CJK"
            break
        except Exception:
            continue

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1E3A5F")
BLUE   = colors.HexColor("#3B82F6")
LBLUE  = colors.HexColor("#DBEAFE")
GREEN  = colors.HexColor("#22C55E")
AMBER  = colors.HexColor("#F59E0B")
RED    = colors.HexColor("#EF4444")
SLATE  = colors.HexColor("#64748B")
LIGHT  = colors.HexColor("#F1F5F9")
WHITE  = colors.white
TEXT   = colors.HexColor("#1E293B")

# ── Styles ────────────────────────────────────────────────────────────────────
SS = getSampleStyleSheet()

def sty(name, parent="Normal", font="Helvetica", size=10,
        leading=14, color=TEXT, bold=False, italic=False,
        align=TA_LEFT, space_before=0, space_after=4, left_indent=0):
    # CJK TTF fonts cannot use the "-Bold"/"-Oblique" suffix convention
    is_cjk = (font == CJK_FONT and CJK_FONT != "Helvetica")
    if is_cjk:
        fname = CJK_FONT   # use as-is; bold handled via <b> tag inline
    else:
        fname = font + ("-BoldOblique" if bold and italic
                        else "-Bold"   if bold
                        else "-Oblique" if italic
                        else "")
    return ParagraphStyle(
        name, parent=SS[parent],
        fontName=fname,
        fontSize=size, leading=leading,
        textColor=color, alignment=align,
        spaceBefore=space_before, spaceAfter=space_after,
        leftIndent=left_indent,
    )

title_sty   = sty("Title2",  size=22, bold=True,  color=WHITE,  align=TA_CENTER, leading=28)
sub_sty     = sty("Sub",     size=11, italic=True, color=LBLUE,  align=TA_CENTER, leading=15)
slide_sty   = sty("Slide",   size=13, bold=True,  color=WHITE)
time_sty    = sty("Time",    size=9,  italic=True, color=SLATE,  align=TA_RIGHT)
en_sty      = sty("En",      font="Helvetica", size=9.5, color=TEXT,  leading=14, space_after=2)
zh_sty      = sty("Zh",      font=CJK_FONT,   size=9.5, color=NAVY,  leading=14, space_after=2)
h3_sty      = sty("H3",      size=10, bold=True,  color=NAVY,  space_before=6)
note_sty    = sty("Note",    size=8.5, italic=True, color=SLATE, leading=12)
tip_sty     = sty("Tip",     size=8.5, color=SLATE, leading=12, left_indent=8)
bold_en     = sty("BoldEn",  font="Helvetica", size=9.5, bold=True, color=WHITE, leading=14)
bold_zh     = sty("BoldZh",  font=CJK_FONT,   size=9.5, color=WHITE, leading=14)

# ── Helper builders ───────────────────────────────────────────────────────────
PW, PH = A4
MARGIN  = 1.8 * cm
COL_W   = (PW - 2 * MARGIN - 0.4 * cm) / 2   # half-page per language

def hr(color=BLUE, thickness=1):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=4)

def spacer(h=6):
    return Spacer(1, h)

def slide_header(number, title, time_budget, color=NAVY):
    """Coloured header bar with slide number, title, and timing."""
    data = [[
        Paragraph(f"Slide {number}  ·  {title}", slide_sty),
        Paragraph(time_budget, time_sty),
    ]]
    t = Table(data, colWidths=[PW - 2*MARGIN - 3*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (0,0),  10),
        ("RIGHTPADDING",  (-1,0), (-1,0), 6),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

def bilingual_table(rows, en_col_w=None, zh_col_w=None):
    """
    rows: list of (en_text, zh_text) tuples.
    Each item can contain <b> markup for bold.
    """
    ecw = en_col_w or COL_W
    zcw = zh_col_w or COL_W
    data = []
    for en, zh in rows:
        e_para = Paragraph(en, en_sty)
        z_para = Paragraph(zh, zh_sty)
        data.append([e_para, z_para])
    t = Table(data, colWidths=[ecw, 0.4*cm + zcw], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F8FAFC")),
        ("BACKGROUND", (1,0), (1,-1), colors.HexColor("#EFF6FF")),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("LINEBELOW", (0,0), (-1,-2), 0.3, colors.HexColor("#E2E8F0")),
        ("LINEAFTER",  (0,0), (0,-1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    return t

def col_labels():
    """Two-column header: 🇬🇧 English | 🇨🇳 中文对照"""
    data = [[
        Paragraph("🇬🇧  English Script", bold_en),
        Paragraph("🇨🇳  中文对照翻译",   bold_zh),
    ]]
    t = Table(data, colWidths=[COL_W, 0.4*cm + COL_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#1E3A5F")),
        ("BACKGROUND", (1,0), (1,-1), colors.HexColor("#1E3A5F")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("FONTCOLOR",     (0,0), (-1,-1), WHITE),
    ]))
    return t

# ── Content definition ────────────────────────────────────────────────────────

SLIDES = [
    # (slide_no, title, time_budget, accent_color, [(en, zh), ...])
    (
        1, "Title · 封面", "~30 sec", NAVY,
        [
            ("Good morning, everyone. Today I'll be presenting <b>Deep-Guard Agent</b> — a real-time audio-visual deepfake detection system with forensic legal reporting, built for the IS596 final project.",
             "大家早上好。今天我将介绍 <b>Deep-Guard Agent</b>——一个结合音视频检测与法证报告生成的实时深度伪造检测系统，这是我在 IS596 课程的期末项目。"),
            ("At the end of this ten minutes, I hope you'll agree that deepfake detection is no longer a research curiosity — it's a legal and societal necessity, and our system takes a meaningful step toward addressing it.",
             "在接下来的十分钟里，我希望能让大家看到：深度伪造检测已不再是学术玩具，而是一个法律与社会层面的迫切需求，而我们的系统正是朝着解决这一需求迈出了重要一步。"),
        ]
    ),
    (
        2, "Motivation · 研究动机", "~60 sec", BLUE,
        [
            ("So why does this matter? Three converging pressures make deepfake detection urgent right now.",
             "为什么这个问题如此重要？有三股力量正在同时施压，使深度伪造检测变得极为迫切。"),
            ("<b>First, the scale of the problem.</b> Deepfake incidents have grown by 3,000 percent since 2019. Ninety-six percent of deepfakes target individuals non-consensually — we're talking about real people having their faces and voices weaponized against them.",
             "<b>第一，问题的规模。</b>自2019年以来，深度伪造事件增长了3000%。96%的深度伪造内容在未经当事人同意的情况下针对个人——这意味着真实的人正在遭受面部和声音被恶意利用的侵害。"),
            ("<b>Second, the legal pressure.</b> The EU AI Act Article 50 and the US Take It Down Act 2025 now <i>require</i> verifiable forensic evidence of synthetic media. Platforms and organizations need tools that can produce court-admissible documentation — not just a score.",
             "<b>第二，法律压力。</b>欧盟《人工智能法》第50条和美国2025年《删除法案》明确要求提供可验证的合成媒体法证证据。平台和机构需要能生成具有法律效力文件的工具——而不仅仅是一个检测分数。"),
            ("<b>Third, the detection gap.</b> Every existing tool is either single-modality — looking at video <i>or</i> audio, not both — or a black box with no explanation. <b>Deep-Guard Agent fills all three gaps in a single open system.</b>",
             "<b>第三，检测空白。</b>现有所有工具，要么是单模态的——只看视频<i>或</i>音频而非同时分析两者——要么是没有解释的黑盒系统。<b>Deep-Guard Agent 在一个开源系统中填补了这三项空白。</b>"),
        ]
    ),
    (
        3, "System Architecture · 系统架构", "~80 sec", colors.HexColor("#0F172A"),
        [
            ("Let me walk you through the six-module pipeline. It flows in two rows.",
             "让我带大家过一遍这个六模块流水线，它分为两行。"),
            ("<b>Row one is the encoding stage.</b> The video enters the Video Input module, then splits: the visual stream goes to the Visual Encoder — MediaPipe FaceLandmarker produces a 256-dimensional lip feature vector. Simultaneously, the Audio Encoder (Wav2Vec2) produces a 768-dimensional representation at 50 fps.",
             "<b>第一行是编码阶段。</b>视频进入视频输入模块后分为两路：视觉流进入视觉编码器，MediaPipe 提取478个面部关键点并压缩为256维唇部特征向量；音频流同步进入音频编码器（Wav2Vec2），以每秒50帧的频率生成768维特征表示。"),
            ("<b>Row two is the reasoning stage.</b> The Cross-Modal Fusion module aligns both streams using a Temporal Transformer, computing a weighted score: 70% learned discrepancy + 30% cosine discrepancy. Frames above 0.65 are flagged.",
             "<b>第二行是推理阶段。</b>跨模态融合模块使用时序 Transformer 对齐两路特征，计算加权差异分数：70% 学习差异 + 30% 余弦差异。分数超过0.65的帧被标记为可疑。"),
            ("The <b>LLM Reasoner</b> — LLaMA 3.3 70B via Groq — then produces a chain-of-thought analysis with a structured JSON verdict. The Output module renders an annotated video, HTML report, legal report, and raw JSON.",
             "<b>LLM 推理器</b>——通过 Groq 调用 LLaMA 3.3 70B——生成思维链分析和结构化 JSON 判决。输出模块渲染标注视频、HTML 分析报告、法证报告和原始 JSON 文件。"),
        ]
    ),
    (
        4, "Theoretical Grounding · 理论基础", "~80 sec", colors.HexColor("#4C1D95"),
        [
            ("The system is grounded in four pillars of existing literature.",
             "该系统建立在四个理论支柱之上。"),
            ("<b>AV-HuBERT</b> (Shi et al., 2022) demonstrated joint audio-visual self-supervised learning. It inspired our cross-modal alignment architecture and the idea that temporal synchrony is a learnable signal.",
             "<b>AV-HuBERT</b>（Shi等，2022）证明了音视频自监督联合学习的可行性，启发了我们的跨模态对齐架构以及时序同步是可学习信号的核心思路。"),
            ("<b>Wav2Vec 2.0</b> (Baevski et al., 2020) gives us a physics-grounded audio representation, pre-trained on 960 hours of real speech. Its feature space captures the physical mechanics of how a human mouth produces sound.",
             "<b>Wav2Vec 2.0</b>（Baevski等，2020）提供了有物理依据的音频表示，在960小时真实人类语音上预训练，其特征空间捕捉了人类口腔产生声音的物理机制。"),
            ("<b>ART-AVDF</b> (Wang &amp; Huang, 2024) is our direct methodological precedent — the closest published work, using articulatory representations specifically for deepfake detection.",
             "<b>ART-AVDF</b>（Wang与Huang，2024）是我们最直接的方法论先例——与我们构建内容最接近的已发表工作，专门使用发音表示进行深度伪造检测。"),
            ("The <b>Daubert Standard + ISO/IEC 27037</b> frames our legal report. Evidence must be testable, have a known error rate, and maintain a chain of custody — a legal requirement in US federal courts. <b>Key physical insight:</b> bilabial phonemes (B, P, M) require full lip closure; deepfake generators consistently violate this — that violation is our detection signal.",
             "<b>Daubert标准 + ISO/IEC 27037</b>构建了我们的法证报告框架。证据必须可验证、具有已知错误率并维护证据保管链——这是美国联邦法院的法律要求。<b>核心物理洞见：</b>双唇音（B、P、M）要求嘴唇完全闭合；深度伪造生成器始终违反这一约束，这个违反就是我们的检测信号。"),
        ]
    ),
    (
        5, "Implementation · 实现细节", "~80 sec", colors.HexColor("#065F46"),
        [
            ("The left panel shows the full eight-component technology stack. I want to highlight four key engineering decisions.",
             "左侧展示了完整的八组件技术栈。我想重点介绍四个关键工程决策。"),
            ("<b>Thread safety.</b> Without protection, concurrent Gradio requests could initialize the pipeline twice. I implemented double-checked locking — checking and acquiring a mutex only when the pipeline is None — so initialization happens exactly once.",
             "<b>线程安全。</b>如果没有保护，并发的 Gradio 请求可能导致流水线被初始化两次。我实现了双重检查锁定——只在流水线为 None 时检查并获取互斥锁——确保初始化只发生一次。"),
            ("<b>O(1) frame lookup.</b> A naïve implementation scans a list per frame — O(n²) overall. I pre-compute a Python set of flagged frame indices before rendering, making each lookup O(1).",
             "<b>O(1)帧查询。</b>朴素实现每帧扫描列表——整体 O(n²)。我在渲染前预计算被标记帧索引的 Python set，使每次查询降为 O(1)。"),
            ("<b>XSS prevention.</b> LLM output could contain HTML injection payloads. Every LLM-generated string passes through Python's html.escape() before being inserted into report templates.",
             "<b>XSS防护。</b>LLM 输出可能包含 HTML 注入攻击载荷。所有来自 LLM 的字符串在注入报告模板之前都经过 Python html.escape() 处理。"),
            ("<b>Pure SVG charts.</b> Gradio strips JavaScript from rendered HTML. All charts are built as server-side Python SVG strings — no JavaScript dependency, no rendering failures.",
             "<b>纯SVG图表。</b>Gradio 会过滤掉渲染 HTML 中的 JavaScript。所有图表都构建为服务端 Python SVG 字符串——无 JavaScript 依赖，无渲染失败。"),
        ]
    ),
    (
        6, "Legal Report · 法证报告", "~50 sec", colors.HexColor("#1D4ED8"),
        [
            ("This is the slide I find most exciting, because nothing like this exists in any other open deepfake detection tool.",
             "这张幻灯片是我最感兴奋的部分，因为没有任何其他开源深度伪造检测工具具备类似功能。"),
            ("The legal report has nine sections in three groups. The <b>blue group</b> (Case Information, Forensic Verdict, Methodology) establishes the what and how. The <b>purple group</b> (Quantitative Findings, Discrepancy Timeline, Flagged Frames Table) provides numerical evidence. The <b>green group</b> (Evidence Summary, Chain of Custody, Limitations) satisfies Daubert and ISO/IEC 27037 transparency requirements.",
             "法证报告包含九个部分，分为三组。<b>蓝色组</b>（案件信息、法证判决、方法论）阐明了做了什么、怎么做的。<b>紫色组</b>（定量发现、差异时间轴、被标记帧表格）提供数值证据。<b>绿色组</b>（证据摘要、证据保管链、局限性声明）满足 Daubert 标准和 ISO/IEC 27037 的透明度要求。"),
            ("Every report includes a SHA-256 hash of the input video file, timestamped at analysis time, proving the file was not tampered with after analysis.",
             "每份报告都包含输入视频文件的 SHA-256 哈希值，并记录分析时间戳，从而证明文件在分析后未被篡改。"),
        ]
    ),
    (
        7, "Evaluation Design · 实验设计", "~40 sec", colors.HexColor("#065F46"),
        [
            ("To evaluate the system, I ran a within-subjects user study with ten university students — five from technical backgrounds (CS, IS) and five from non-technical backgrounds (Law, Journalism, Psychology).",
             "为评估系统，我对10名大学生进行了组内用户研究——5名来自技术背景（CS、IS），5名来自非技术背景（法律、新闻、心理学）。"),
            ("Each 60-minute session began with an unaided baseline task — watching two videos and judging authenticity without the tool. Participants then used the system on three videos of varying difficulty: authentic, clearly fake, and ambiguous. Finally they completed the SUS questionnaire and a semi-structured interview.",
             "每次60分钟的测试以无辅助基线任务开始——在不使用工具的情况下观看两段视频并判断真实性。参与者随后使用系统分析三段难度不一的视频：真实的、明显伪造的和模糊的。最后完成 SUS 问卷和半结构化访谈。"),
        ]
    ),
    (
        8, "SUS Results · 可用性结果", "~55 sec", NAVY,
        [
            ('The <b>mean SUS score was 70.0 out of 100</b>, placing Deep-Guard Agent in the <b>"Above Average"</b> category by the Bangor et al. (2009) adjective scale. Five of ten participants scored above the industry benchmark of 68.',
             '<b>平均SUS分数为70分（满分100）</b>，根据 Bangor 等人（2009）的形容词量表，Deep-Guard Agent 落在<b>"高于平均"</b>区间。10名参与者中有5人超过了行业基准线68分。'),
            ("There is a clear divide along technical background: CS and IS students averaged <b>83.8</b>, while non-technical participants averaged <b>61.3</b>. This 22-point gap is our clearest signal for where to improve the interface.",
             "技术背景带来了明显的分化：CS 和 IS 学生平均 <b>83.8分</b>，而非技术背景参与者平均 <b>61.3分</b>。这22分的差距是我们改进界面方向最清晰的信号。"),
        ]
    ),
    (
        9, "Accuracy & Likert · 准确率与量表", "~55 sec", NAVY,
        [
            ("The accuracy results are the headline finding. In the <b>unaided baseline</b>, participants correctly identified video authenticity <b>55% of the time</b> — essentially chance level, confirming that deepfakes are genuinely hard to detect.",
             "准确率结果是本研究最核心的发现。在<b>无辅助基线条件下</b>，参与者正确判断视频真实性的概率为 <b>55%</b>——基本等同于随机猜测，印证了深度伪造对人类确实极难辨别。"),
            ("With Deep-Guard Agent, accuracy rose to <b>85%</b> — a <b>+30 percentage point improvement</b>. Participants agreed with the system's verdict 85% of the time across all three videos.",
             "使用 Deep-Guard Agent 后，准确率提升至 <b>85%</b>——提升了 <b>30个百分点</b>。参与者在所有三段视频中有85%的概率同意系统的判决。"),
            ("The Likert results show all four constructs between 4.1 and 4.7 on a 1–7 scale. Trust in AI verdict was highest at <b>4.72</b>. Report comprehension was lowest at <b>4.17</b>, suggesting technical terminology remains a barrier.",
             "Likert 量表结果显示，所有四个维度评分在1到7分制的4.1到4.7之间。对 AI 判决的信任度最高，达 <b>4.72</b>；报告可读性最低，为 <b>4.17</b>，表明技术术语对部分用户仍构成障碍。"),
            ("The Spearman correlation between AI familiarity and SUS was <b>ρ = 0.97</b> — a near-perfect monotonic relationship. Users comfortable with AI found the system significantly more usable.",
             "AI 熟悉度与 SUS 分数的 Spearman 相关系数为 <b>ρ = 0.97</b>——几乎完美的单调关系。对 AI 工具熟悉的用户认为该系统显著更易用。"),
        ]
    ),
    (
        10, "Qualitative Themes · 定性主题", "~45 sec", colors.HexColor("#065F46"),
        [
            ("The semi-structured interviews produced four dominant themes.",
             "半结构化访谈揭示了四个主要主题。"),
            ("<b>Positive — Clear Verdict (8/10):</b> Participants said the verdict badge was immediately clear and easy to understand.",
             "<b>正面——判决清晰（8/10）：</b>参与者表示判决标识清晰易懂，可立刻理解。"),
            ("<b>Positive — Legal Report Useful (7/10):</b> Particularly law and journalism participants said they could actually see themselves submitting the report as evidence.",
             "<b>正面——法证报告有用（7/10）：</b>尤其是法律和新闻背景的参与者表示，实际上可以考虑将报告作为证据提交。"),
            ("<b>Concern — Confusing Terms (6/10):</b> Cosine similarity, flagged frames ratio, articulatory discrepancy — these terms confused non-technical users.",
             "<b>负面——术语困惑（6/10）：</b>余弦相似度、标记帧比例、发音差异等术语让非技术背景用户感到困惑。"),
            ("<b>Concern — Processing Speed (4/10):</b> The five-minute processing time was seen as too slow for practical use.",
             "<b>负面——处理速度（4/10）：</b>约五分钟的处理时间被认为对实际使用来说过长。"),
        ]
    ),
    (
        11, "Discussion · 讨论与反思", "~55 sec", colors.HexColor("#92400E"),
        [
            ("<b>What worked well:</b> The bimodal detection catches mismatches that single-modality tools miss. The LLM reasoning layer turns a number into a narrative, making the verdict explainable. The legal report filled a gap participants found immediately valuable.",
             "<b>哪些做得好：</b>双模态检测确实能捕捉到单模态工具遗漏的不匹配。LLM 推理层将数字转化为叙事，使判决可解释。法证报告填补了参与者立刻认可其价值的空白。"),
            ("<b>Limitations:</b> The most honest limitation: the fusion module uses random initialization — no trained checkpoint. Discrepancy scores are not calibrated; they reflect relative patterns, not absolute probabilities. For production, training on FaceForensics++ or DFDC is required.",
             "<b>局限性：</b>最需坦诚说明的是：融合模块使用随机初始化，没有经过训练的权重文件。差异分数未经校准——反映的是相对模式，而非绝对概率。生产系统需要在 FaceForensics++ 或 DFDC 数据集上进行训练。"),
            ("<b>Future work — three priorities:</b> (1) Train the fusion module on a labeled deepfake dataset. (2) Add inline tooltips and a glossary to close the vocabulary gap. (3) Explore streaming inference for live-video use cases.",
             "<b>未来工作——三个优先方向：</b>（1）在标注深度伪造数据集上训练融合模块；（2）添加内联提示和术语表，弥合词汇障碍；（3）探索流式推理以支持实时视频场景。"),
        ]
    ),
    (
        12, "Conclusion · 总结", "~30 sec", colors.HexColor("#064E3B"),
        [
            ("To close: <b>Deep-Guard Agent</b> is, to my knowledge, the first open system combining audio-visual bimodal detection, LLM-powered explainability, and ISO-compliant legal reporting in a single tool.",
             "最后总结：据我所知，<b>Deep-Guard Agent</b> 是第一个将音视频双模态检测、LLM 驱动的可解释性以及符合 ISO 标准的法证报告整合于单一工具的开源系统。"),
            ("The user study confirms a <b>+30 percentage point accuracy improvement</b> over unaided detection, with a SUS score of <b>70</b> indicating above-average usability. The untrained fusion module is a known, addressable engineering challenge — not a fundamental design flaw.",
             "用户研究证实，与无辅助检测相比，准确率提升了 <b>30个百分点</b>，SUS 分数 <b>70分</b>表明可用性高于平均水平。融合模块未经训练是一个已知且可解决的工程挑战——而非根本性的设计缺陷。"),
            ("<b>Thank you. I'm happy to take questions.</b>",
             "<b>谢谢大家，我很乐意回答问题。</b>"),
        ]
    ),
]

TIMING = [
    ("1", "Title · 封面",            "30 sec",  "0:30"),
    ("2", "Motivation · 研究动机",    "60 sec",  "1:30"),
    ("3", "Architecture · 系统架构",  "80 sec",  "2:50"),
    ("4", "Theory · 理论基础",        "80 sec",  "4:10"),
    ("5", "Implementation · 实现",    "80 sec",  "5:30"),
    ("6", "Legal Report · 法证报告",  "50 sec",  "6:20"),
    ("7", "Eval Design · 实验设计",   "40 sec",  "7:00"),
    ("8", "SUS Results · 可用性",     "55 sec",  "7:55"),
    ("9", "Accuracy & Likert",         "55 sec",  "8:50"),
    ("10","Themes · 定性主题",         "45 sec",  "9:35"),
    ("11","Discussion · 讨论",         "55 sec", "10:30"),
    ("12","Conclusion · 总结",         "30 sec", "11:00"),
]

QA = [
    ("Why not use a trained model?",
     "Training requires a labeled dataset and significant GPU time. The architecture is train-ready — the fusion module can be swapped with trained weights whenever data is available.",
     "为何不用训练好的模型？",
     "训练需要标注数据集和大量 GPU 资源。系统架构已具备训练条件——融合模块可随时替换为训练好的权重。"),
    ("Is the legal report actually admissible?",
     "It is designed to meet the structural requirements of Daubert and ISO/IEC 27037. Whether a specific court accepts it depends on jurisdiction and expert testimony.",
     "法证报告真的具有法律效力吗？",
     "它在设计上满足 Daubert 标准和 ISO/IEC 27037 的结构性要求。是否被特定法院采纳取决于司法管辖区和专家证词。"),
    ("Why Gradio and not a custom web app?",
     "Gradio provides a production-quality UI with minimal code, letting me focus engineering effort on the detection pipeline rather than frontend development — the correct trade-off for a research prototype.",
     "为何选择 Gradio 而非自定义网页？",
     "Gradio 以极少的代码提供了生产质量的 UI，使我能将工程精力集中在检测流水线上——这对研究原型是正确的取舍。"),
    ("How does performance scale with video length?",
     "Processing time scales roughly linearly — about 5 minutes per minute of video on CPU. GPU acceleration would reduce this by approximately 10×.",
     "性能如何随视频时长扩展？",
     "处理时间与视频时长大致呈线性关系——CPU 上每分钟视频约需5分钟。GPU 加速可将其缩短约10倍。"),
]

# ── Page template with header/footer ─────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    # top bar
    canvas.setFillColor(NAVY)
    canvas.rect(0, PH - 1.1*cm, PW, 1.1*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(WHITE)
    canvas.drawString(MARGIN, PH - 0.72*cm, "Deep-Guard Agent  ·  IS596 Final Presentation  ·  Speech Script")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(WHITE)
    canvas.drawRightString(PW - MARGIN, PH - 0.72*cm, "🇬🇧 EN  |  🇨🇳 ZH")
    # bottom bar
    canvas.setFillColor(LBLUE)
    canvas.rect(0, 0, PW, 0.8*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(NAVY)
    canvas.drawCentredString(PW/2, 0.27*cm, f"Page {doc.page}")
    canvas.restoreState()

# ── Build story ───────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.0*cm, bottomMargin=1.5*cm,
    )
    story = []

    # ── Cover block ───────────────────────────────────────────────────────────
    cover_data = [[Paragraph("Deep-Guard Agent", title_sty)],
                  [Paragraph("Final Presentation Speech Script", sub_sty)],
                  [Paragraph("IS596  ·  Audio-Visual Deepfake Detection  ·  ~10 Minutes", sub_sty)],
                  [Paragraph("English Script  |  中文对照翻译", sub_sty)],
    ]
    cover = Table(cover_data, colWidths=[PW - 2*MARGIN])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(cover)
    story.append(spacer(10))

    # ── HOW TO USE note ───────────────────────────────────────────────────────
    story.append(Paragraph("How to use this document", h3_sty))
    story.append(Paragraph(
        "Each slide block shows the suggested English script (left, white background) alongside its Chinese translation (right, blue background). "
        "Bold text marks key phrases to emphasize. Italic text marks words to stress in speech. "
        "The timing shown in each slide header is a target — adjust to your natural pace.",
        note_sty))
    story.append(hr())
    story.append(spacer(6))

    # ── Slides ────────────────────────────────────────────────────────────────
    for slide_no, title, time_budget, color, rows in SLIDES:
        block = [
            slide_header(slide_no, title, time_budget, color),
            spacer(4),
            col_labels(),
            bilingual_table(rows),
            spacer(10),
        ]
        story.append(KeepTogether(block[:3]))  # keep header+labels together
        story.append(bilingual_table(rows))
        story.append(spacer(10))

    story.append(PageBreak())

    # ── Timing guide ─────────────────────────────────────────────────────────
    story.append(Paragraph("Timing Guide · 演讲计时参考", h3_sty))
    story.append(spacer(4))
    tdata = [["Slide", "Topic", "Target", "Cumulative"]] + list(TIMING)
    tbl = Table(tdata, colWidths=[1.2*cm, 8.5*cm, 2.2*cm, 2.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("FONTNAME",      (0,1), (-1,-1), CJK_FONT),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LBLUE]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("ALIGN",         (2,0), (-1,-1), "CENTER"),
    ]))
    story.append(tbl)
    story.append(spacer(4))
    story.append(Paragraph(
        "Tip: If running short on time, compress Slides 5 and 10. "
        "If running over, drop the XSS/SVG details on Slide 5 and the Spearman number on Slide 9.",
        tip_sty))
    story.append(spacer(14))

    # ── Q&A ───────────────────────────────────────────────────────────────────
    story.append(Paragraph("Anticipated Q&amp;A · 可能的提问与回答", h3_sty))
    story.append(spacer(4))
    for q_en, a_en, q_zh, a_zh in QA:
        qdata = [[
            Paragraph(f"<b>Q:</b> {q_en}", en_sty),
            Paragraph(f"<b>Q：</b>{q_zh}", zh_sty),
        ],[
            Paragraph(f"<b>A:</b> {a_en}", en_sty),
            Paragraph(f"<b>A：</b>{a_zh}", zh_sty),
        ]]
        qt = Table(qdata, colWidths=[COL_W, 0.4*cm + COL_W])
        qt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E0F2FE")),
            ("BACKGROUND", (0,1), (-1,1), WHITE),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("LINEAFTER",  (0,0), (0,-1), 0.5, colors.HexColor("#CBD5E1")),
            ("LINEBELOW",  (0,0), (-1,-1), 0.3, colors.HexColor("#E2E8F0")),
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ]))
        story.append(qt)
        story.append(spacer(6))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"✓ PDF saved → {OUT}")

if __name__ == "__main__":
    build()
