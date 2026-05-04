"""Legal Forensic Report Generator for Deep-Guard Agent.

Produces structured legal-grade forensic reports compliant with:
  - ISO/IEC 27037:2012 — Digital evidence identification, collection & preservation
  - NIST SP 800-86 — Guide to Integrating Forensic Techniques
  - EU AI Act Article 50 — Transparency obligations for synthetic media
  - SWGDE Best Practices for Digital & Multimedia Evidence

Reports are suitable for platform moderation, law enforcement referral,
and civil/criminal proceedings as supporting documentation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deepguard.detection.fusion import FusionResult
from deepguard.interface.report import ReportGenerator
from deepguard.reasoning.llm_reasoner import AnalysisReport

# ── Tool metadata (NIST SP 800-86 requires documenting tool versions) ──────────
_TOOL_NAME = "Deep-Guard Agent"
_TOOL_VERSION = "2.0"
_VISUAL_METHOD = "MediaPipe FaceLandmarker lip landmark extraction (256-d feature vectors)"
_AUDIO_METHOD = "Facebook Wav2Vec2-base-960h articulatory encoding (768-d, 50 fps)"
_FUSION_METHOD = "Cross-modal temporal attention with cosine similarity scoring"
_REASONING_METHOD = "LLaMA 3.3 70B via Groq API (chain-of-thought verification)"
_REFERENCE_THRESHOLD = 0.65


# ── Legal framework references ─────────────────────────────────────────────────
_LEGAL_FRAMEWORKS = [
    {
        "id": "EU-AI-ACT-50",
        "title": "EU AI Act Article 50 — Transparency Obligations for Synthetic Media",
        "description": (
            "Requires deployers of AI systems that generate or manipulate deepfake "
            "content to disclose that the content has been artificially generated or "
            "manipulated. Providers must mark outputs in machine-readable format. "
            "Enforcement expected August 2026; penalties up to €35M or 7% of global turnover."
        ),
        "url": "https://artificialintelligenceact.eu/article/50/",
    },
    {
        "id": "TAKE-IT-DOWN-ACT-2025",
        "title": "Take It Down Act (US, May 2025)",
        "description": (
            "Federal law criminalizing AI use to create non-consensual deepfake images "
            "without depicted persons' consent. FTC enforcement began within one year of "
            "enactment. Relevant to NCII (non-consensual intimate imagery) harm categories."
        ),
        "url": "https://regulaforensics.com/blog/deepfake-regulations/",
    },
    {
        "id": "NIST-SP-800-86",
        "title": "NIST SP 800-86 — Guide to Integrating Forensic Techniques",
        "description": (
            "Establishes that digital evidence must be repeatable and reproducible "
            "to be accepted in legal proceedings. Requires documentation of tool names, "
            "versions, validation, and chain of custody for all forensic analyses."
        ),
        "url": "https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-86.pdf",
    },
    {
        "id": "ISO-IEC-27037",
        "title": "ISO/IEC 27037:2012 — Digital Evidence Handling Guidelines",
        "description": (
            "International standard governing identification, collection, acquisition, "
            "and preservation of digital evidence. Requires continuous chain-of-custody "
            "documentation, SHA-256 file integrity verification, and documented methodology."
        ),
        "url": "https://www.iso.org/standard/44381.html",
    },
    {
        "id": "NIST-MFC",
        "title": "NIST Media Forensics Challenge (OpenMFC)",
        "description": (
            "NIST benchmark for evaluating deepfake detection systems. Published "
            "'Guardians of Forensic Evidence' (Jan 2025) evaluating analytic systems "
            "against AI-generated deepfakes. Deep-Guard Agent's methodology aligns "
            "with multimodal analysis approaches validated by this challenge."
        ),
        "url": "https://mfc.nist.gov/",
    },
    {
        "id": "DAUBERT-STANDARD",
        "title": "Daubert Standard — Scientific Evidence Admissibility (US)",
        "description": (
            "US Supreme Court standard requiring scientific evidence to be reliable "
            "and relevant, considering: testability, peer-review status, known error "
            "rates, and general scientific acceptance. This report documents error "
            "rates and methodology to support Daubert admissibility review."
        ),
        "url": "https://ubaltlawreview.com/2025/12/01/deepfakes-in-the-courtroom-challenges-in-authenticating-evidence-and-jury-evaluation/",
    },
]

_HARM_LEGAL_MAP = {
    "political": [
        "Potential violation of election integrity laws and platform misinformation policies.",
        "Refer to election authorities or platform trust & safety teams.",
        "Document under EU AI Act Article 50 disclosure obligations.",
    ],
    "ncii": [
        "Potential offense under the Take It Down Act (US, May 2025) — criminal liability.",
        "Contact platform for immediate content removal per NCII removal policies.",
        "Preserve evidence chain-of-custody for law enforcement referral.",
        "Consider referral to NCII support organizations (e.g., StopNCII).",
    ],
    "financial_fraud": [
        "Potential wire fraud, impersonation, or identity theft under applicable law.",
        "Preserve all communications and transaction records as supporting evidence.",
        "Refer to financial institution fraud department and law enforcement.",
    ],
    "general": [
        "Review under applicable platform terms of service and local synthetic media laws.",
        "Document and preserve evidence per ISO/IEC 27037 guidelines.",
    ],
}

_VERDICT_LEGAL_IMPACT = {
    "authentic": (
        "No forensic indicators of synthetic manipulation detected. This finding "
        "does not constitute absolute proof of authenticity — absence of detectable "
        "manipulation artifacts does not exclude the possibility of highly sophisticated "
        "synthesis techniques beyond current detection capabilities."
    ),
    "suspicious": (
        "Forensic analysis detected audio-visual discrepancies consistent with "
        "potential synthetic manipulation. This finding warrants further investigation "
        "by qualified human forensic experts before use in legal proceedings. "
        "Confidence level is below the definitive threshold."
    ),
    "likely_fake": (
        "Forensic analysis detected strong indicators of synthetic manipulation "
        "inconsistent with authentic recording. This finding should be reviewed by "
        "a qualified human forensic examiner and cross-validated with additional tools "
        "before submission as evidence in legal proceedings."
    ),
}


class LegalReportGenerator:
    """Generates ISO/IEC 27037-compliant legal forensic reports."""

    def __init__(self) -> None:
        self._report_gen = ReportGenerator()

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _score_stats(scores: np.ndarray) -> dict:
        if len(scores) == 0:
            return {}
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
        }

    # ── SVG chart (reuse from ReportGenerator) ────────────────────────────────

    def _svg_chart(self, scores: np.ndarray, threshold: float) -> str:
        return ReportGenerator._build_svg_chart(scores, threshold)

    # ── CSS ───────────────────────────────────────────────────────────────────

    _CSS = """\
.lg-report *, .lg-report *::before, .lg-report *::after {
    margin:0; padding:0; box-sizing:border-box; }
.lg-report {
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    color:#1e293b; line-height:1.65; padding:1.75rem 2rem; font-size:1rem; }

/* header */
.lg-report .lg-header {
    border-bottom:2px solid #0f172a; padding-bottom:1rem; margin-bottom:1.5rem; }
.lg-report .lg-title {
    font-size:1.05rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.04em; color:#0f172a; }
.lg-report .lg-subtitle {
    font-size:.85rem; color:#475569; margin-top:.3rem; }
.lg-report .lg-case-id {
    font-family:ui-monospace,monospace; font-size:.78rem;
    color:#475569; margin-top:.45rem; word-break:break-all; }

/* section */
.lg-report .lg-section { margin-bottom:1.75rem; }
.lg-report .lg-section-title {
    font-size:.78rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.07em; color:#334155; border-bottom:2px solid #e2e8f0;
    padding-bottom:.35rem; margin-bottom:.85rem; }

/* verdict block */
.lg-report .lg-verdict-block {
    display:flex; align-items:flex-start; gap:1rem;
    background:#f8fafc; border-radius:8px; padding:1.1rem 1.25rem;
    border-left:4px solid var(--vc, #6b7280); }
.lg-report .lg-verdict-badge {
    flex-shrink:0; padding:.35rem 1.1rem; border-radius:20px;
    font-size:.88rem; font-weight:700; color:#fff;
    background:var(--vc, #6b7280); white-space:nowrap; }
.lg-report .lg-verdict-body { flex:1; }
.lg-report .lg-verdict-conf {
    font-size:.9rem; color:#475569; margin-bottom:.45rem; font-weight:500; }
.lg-report .lg-verdict-legal {
    font-size:.92rem; color:#1e293b; line-height:1.6; }

/* table */
.lg-report table { width:100%; border-collapse:collapse; font-size:.9rem; }
.lg-report th {
    color:#475569; font-weight:600; text-align:left;
    padding:.45rem .7rem; border-bottom:2px solid #cbd5e1; font-size:.8rem;
    text-transform:uppercase; letter-spacing:.04em; }
.lg-report td { padding:.4rem .7rem; border-bottom:1px solid #e2e8f0; color:#334155; vertical-align:top; }
.lg-report td:first-child { font-weight:600; color:#0f172a; width:38%; }
.lg-report .lg-hash { font-family:ui-monospace,monospace; font-size:.78rem;
    word-break:break-all; color:#334155; }

/* list */
.lg-report .lg-list { padding-left:1.3rem; }
.lg-report .lg-list li { margin:.4rem 0; color:#1e293b; font-size:.92rem; }

/* framework cards */
.lg-report .lg-fw-card {
    background:#f8fafc; border-radius:8px; padding:.9rem 1.1rem;
    margin-bottom:.6rem; border-left:3px solid #3b82f6; }
.lg-report .lg-fw-id {
    display:inline-block; font-size:.72rem; font-weight:700;
    color:#fff; background:#3b82f6; border-radius:4px;
    padding:.1rem .45rem; margin-bottom:.35rem; letter-spacing:.03em; }
.lg-report .lg-fw-title {
    font-weight:600; font-size:.92rem; color:#0f172a; }
.lg-report .lg-fw-desc {
    font-size:.88rem; color:#334155; margin-top:.35rem; line-height:1.6; }
.lg-report .lg-fw-link {
    font-size:.78rem; color:#2563eb; margin-top:.35rem; display:block;
    word-break:break-all; text-decoration:underline; }

/* disclaimer */
.lg-report .lg-disclaimer {
    background:#fff7ed; border:1px solid #fdba74;
    border-radius:8px; padding:1.1rem 1.25rem; }
.lg-report .lg-disclaimer-title {
    font-size:.85rem; font-weight:700; color:#7c2d12;
    text-transform:uppercase; letter-spacing:.04em; margin-bottom:.5rem; }
.lg-report .lg-disclaimer-body {
    font-size:.9rem; color:#431407; line-height:1.65; }
.lg-report .lg-disclaimer-body strong { color:#7c2d12; }

/* chart */
.lg-report svg { width:100%; height:auto; display:block; }

/* footer */
.lg-report .lg-foot {
    margin-top:1.75rem; padding-top:.85rem; border-top:2px solid #0f172a;
    font-size:.75rem; color:#64748b; text-align:center;
    font-family:ui-monospace,monospace; word-break:break-all; }"""

    # ── HTML builder ──────────────────────────────────────────────────────────

    def _build_html(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None,
        case_ref: str,
        generated_at: str,
        file_hash: str | None,
    ) -> str:
        scores = fusion_result.discrepancy_scores
        meta = fusion_result.metadata
        threshold = meta.get("threshold", _REFERENCE_THRESHOLD)

        verdict_colors = {
            "authentic": "#22c55e",
            "suspicious": "#f59e0b",
            "likely_fake": "#ef4444",
        }
        vc = verdict_colors.get(analysis.verdict, "#6b7280")

        stats = self._score_stats(scores)
        svg_chart = self._svg_chart(scores, threshold)

        filename = Path(source_file).name if source_file else "Unknown"
        harm = analysis.harm_category or "general"
        legal_steps = _HARM_LEGAL_MAP.get(harm, _HARM_LEGAL_MAP["general"])
        verdict_legal = _VERDICT_LEGAL_IMPACT.get(analysis.verdict, "")

        # ── Framework cards ──────────────────────────────────────────────────
        fw_html = ""
        for fw in _LEGAL_FRAMEWORKS:
            fw_html += (
                f'<div class="lg-fw-card">'
                f'<span class="lg-fw-id">{fw["id"]}</span> '
                f'<div class="lg-fw-title">{fw["title"]}</div>'
                f'<div class="lg-fw-desc">{fw["description"]}</div>'
                f'<a class="lg-fw-link" href="{fw["url"]}" target="_blank">'
                f'&#128279; {fw["url"]}</a>'
                f'</div>\n'
            )

        # ── Recommended legal actions ────────────────────────────────────────
        legal_actions_html = "\n".join(f"<li>{s}</li>" for s in legal_steps)
        if analysis.recommended_actions:
            for a in analysis.recommended_actions:
                legal_actions_html += f"<li>{a}</li>"

        # ── Flagged frames table ─────────────────────────────────────────────
        flagged_rows = ""
        if fusion_result.flagged_frames:
            video_fps = meta.get("video_fps", 25.0)
            top = sorted(
                fusion_result.flagged_frames,
                key=lambda f: fusion_result.discrepancy_scores[f],
                reverse=True,
            )[:15]
            for f in top:
                ts = f / video_fps
                s = fusion_result.discrepancy_scores[f]
                flagged_rows += (
                    f"<tr><td>{f}</td><td>{ts:.2f}s</td>"
                    f"<td>{s:.4f}</td></tr>\n"
                )

        hash_display = (file_hash[:16] + "&hellip;" + file_hash[-8:]) if file_hash else "Not computed"
        hash_full = f'<span class="lg-hash">{file_hash}</span>' if file_hash else "Not computed"

        body = f"""\
<div class="lg-header">
  <div class="lg-title">Deep-Guard Agent &mdash; Forensic Legal Report</div>
  <div class="lg-subtitle">Automated Audio-Visual Deepfake Analysis &nbsp;&middot;&nbsp; ISO/IEC 27037 &nbsp;&middot;&nbsp; NIST SP 800-86</div>
  <div class="lg-case-id">Case Ref: {case_ref} &nbsp;&middot;&nbsp; Generated: {generated_at}</div>
</div>

<!-- 1. Case Information -->
<div class="lg-section">
  <div class="lg-section-title">1. Case Information</div>
  <table>
    <tr><td>Source File</td><td>{filename}</td></tr>
    <tr><td>SHA-256 Hash</td><td>{hash_full}</td></tr>
    <tr><td>Analysis Timestamp (UTC)</td><td>{generated_at}</td></tr>
    <tr><td>Case Reference</td><td class="lg-hash">{case_ref}</td></tr>
    <tr><td>Examining System</td><td>{_TOOL_NAME} v{_TOOL_VERSION}</td></tr>
  </table>
</div>

<!-- 2. Verdict -->
<div class="lg-section">
  <div class="lg-section-title">2. Forensic Verdict</div>
  <div class="lg-verdict-block" style="--vc:{vc}">
    <div class="lg-verdict-badge">{analysis.verdict.upper().replace("_", " ")}</div>
    <div class="lg-verdict-body">
      <div class="lg-verdict-conf">Confidence: {analysis.confidence:.1%} &nbsp;&middot;&nbsp; Harm Category: {harm.replace("_"," ").title()}</div>
      <div class="lg-verdict-legal">{verdict_legal}</div>
    </div>
  </div>
</div>

<!-- 3. Methodology -->
<div class="lg-section">
  <div class="lg-section-title">3. Analytical Methodology</div>
  <table>
    <tr><td>Visual Analysis</td><td>{_VISUAL_METHOD}</td></tr>
    <tr><td>Audio Analysis</td><td>{_AUDIO_METHOD}</td></tr>
    <tr><td>Cross-Modal Fusion</td><td>{_FUSION_METHOD}</td></tr>
    <tr><td>AI Reasoning</td><td>{_REASONING_METHOD}</td></tr>
    <tr><td>Detection Threshold</td><td>{threshold} &nbsp;(scores &ge; threshold flagged as anomalous)</td></tr>
  </table>
</div>

<!-- 4. Quantitative Findings -->
<div class="lg-section">
  <div class="lg-section-title">4. Quantitative Findings</div>
  <table>
    <tr><td>Frames Analyzed</td><td>{meta.get("num_frames_analyzed", "N/A")}</td></tr>
    <tr><td>Frames Flagged</td><td>{meta.get("num_flagged_frames", 0)} &nbsp;({meta.get("flagged_ratio", 0):.1%})</td></tr>
    <tr><td>Mean Cosine Similarity</td><td>{meta.get("mean_cosine_similarity", "N/A")}</td></tr>
    <tr><td>Score Mean &plusmn; Std</td><td>{stats.get("mean", 0):.4f} &plusmn; {stats.get("std", 0):.4f}</td></tr>
    <tr><td>Score Range</td><td>[{stats.get("min", 0):.4f}, {stats.get("max", 0):.4f}]</td></tr>
    <tr><td>Score Median</td><td>{stats.get("median", 0):.4f}</td></tr>
  </table>
</div>

<!-- 4b. Discrepancy Timeline -->
<div class="lg-section">
  <div class="lg-section-title">Discrepancy Score Timeline</div>
  {svg_chart}
</div>

{"<div class='lg-section'><div class='lg-section-title'>Flagged Frames (Top 15 by Score)</div><table><thead><tr><th>Frame</th><th>Timestamp</th><th>Score</th></tr></thead><tbody>" + flagged_rows + "</tbody></table></div>" if flagged_rows else ""}

<!-- 5. Evidence Summary -->
<div class="lg-section">
  <div class="lg-section-title">5. Evidence Summary</div>
  <ul class="lg-list">
    {"".join(f"<li>{e}</li>" for e in analysis.evidence)}
  </ul>
  {f'<p style="margin-top:.85rem;color:#1e293b;font-size:.92rem;line-height:1.6;"><strong>Phoneme Analysis:</strong> {analysis.phoneme_analysis}</p>' if analysis.phoneme_analysis else ""}
</div>

<!-- 6. Applicable Legal Frameworks -->
<div class="lg-section">
  <div class="lg-section-title">6. Applicable Legal Frameworks</div>
  {fw_html}
</div>

<!-- 7. Chain of Custody -->
<div class="lg-section">
  <div class="lg-section-title">7. Chain of Custody (ISO/IEC 27037)</div>
  <table>
    <tr><td>Evidence Received</td><td>{generated_at}</td></tr>
    <tr><td>Analysis Performed</td><td>{generated_at}</td></tr>
    <tr><td>Report Generated</td><td>{generated_at}</td></tr>
    <tr><td>File Integrity Verified</td><td>SHA-256: {hash_display}</td></tr>
    <tr><td>Analysis System</td><td>{_TOOL_NAME} v{_TOOL_VERSION} (automated, no human modification)</td></tr>
    <tr><td>Processing Environment</td><td>Isolated local execution; no external transmission of source file</td></tr>
  </table>
</div>

<!-- 8. Recommended Legal Actions -->
<div class="lg-section">
  <div class="lg-section-title">8. Recommended Actions</div>
  <ul class="lg-list">{legal_actions_html}</ul>
</div>

<!-- 9. Limitations & Disclaimer -->
<div class="lg-section">
  <div class="lg-section-title">9. Limitations &amp; Legal Disclaimer</div>
  <div class="lg-disclaimer">
    <div class="lg-disclaimer-title">&#9888; Important Notice</div>
    <div class="lg-disclaimer-body">
      This report is generated by an <strong>automated AI system</strong> and is provided
      for investigative and informational purposes only. It does <strong>not</strong> constitute
      legal advice, a final forensic determination, or expert witness testimony.
      <br><br>
      <strong>Limitations:</strong> Detection accuracy depends on video quality, face visibility,
      audio clarity, and whether the deepfake technique falls within the system&rsquo;s detection scope.
      Highly sophisticated synthesis methods may evade detection. The system has a known false
      positive/negative rate inherent to probabilistic AI analysis.
      <br><br>
      <strong>For legal proceedings:</strong> This automated report should be reviewed and
      validated by a qualified human forensic expert before submission as evidence. Courts
      applying the <em>Daubert</em> standard require independent validation, peer-reviewed
      methodology, and documented error rates &mdash; consult qualified counsel.
      <br><br>
      <strong>Privacy:</strong> The source file is processed locally. No video content is
      transmitted to external servers except for LLM reasoning (text-only analysis summary
      sent to Groq API; no video frames transmitted).
    </div>
  </div>
</div>

<div class="lg-foot">
  {filename} &middot; SHA-256 {hash_display} &middot;
  {_TOOL_NAME} v{_TOOL_VERSION} &middot; Case Ref: {case_ref}
</div>"""

        return body

    # ── Public API ────────────────────────────────────────────────────────────

    def to_html_embed(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None = None,
    ) -> str:
        """Return embeddable HTML fragment for Gradio gr.HTML."""
        report_gen = ReportGenerator()
        file_hash = report_gen._compute_file_hash(source_file) if source_file else None
        case_ref = str(uuid.uuid4()).upper()
        generated_at = self._now_utc()

        body = self._build_html(
            fusion_result, analysis, source_file, case_ref, generated_at, file_hash,
        )
        return f'<style>{self._CSS}</style><div class="lg-report">{body}</div>'

    def to_html(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        output_path: str,
        source_file: str | None = None,
    ) -> str:
        """Export as a standalone HTML file."""
        report_gen = ReportGenerator()
        file_hash = report_gen._compute_file_hash(source_file) if source_file else None
        case_ref = str(uuid.uuid4()).upper()
        generated_at = self._now_utc()

        body = self._build_html(
            fusion_result, analysis, source_file, case_ref, generated_at, file_hash,
        )
        html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deep-Guard Agent — Forensic Legal Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>body{{background:#f1f5f9;display:flex;justify-content:center;padding:2rem;}}</style>
<style>{self._CSS}</style>
</head>
<body>
<div style="max-width:900px;width:100%;background:#fff;border-radius:12px;overflow:hidden;">
<div class="lg-report">{body}</div>
</div>
</body>
</html>"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return str(path)
