"""Report Generator: produces structured, traceable technical reports.

Exports timestamped evidence usable for platform moderation or legal proceedings.
Supports JSON, plain-text, and HTML output formats.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deepguard.detection.fusion import FusionResult
from deepguard.reasoning.llm_reasoner import AnalysisReport

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates analysis reports in various formats."""

    def __init__(self, include_timestamps: bool = True, include_heatmaps: bool = True):
        self.include_timestamps = include_timestamps
        self.include_heatmaps = include_heatmaps

    def _compute_file_hash(self, file_path: str) -> str | None:
        """Compute SHA-256 hash of a file for chain of custody."""
        try:
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except (OSError, IOError):
            return None

    def _build_report_dict(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None = None,
    ) -> dict:
        """Build a structured report dictionary."""
        report = {
            "report_version": "2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
            "source_file_hash": self._compute_file_hash(source_file) if source_file else None,
            "verdict": analysis.verdict,
            "confidence": analysis.confidence,
            "summary": analysis.summary,
            "evidence": analysis.evidence,
            "harm_category": analysis.harm_category,
            "recommended_actions": analysis.recommended_actions,
            "phoneme_analysis": analysis.phoneme_analysis,
            "confidence_reasoning": analysis.confidence_reasoning,
            "detection_metadata": fusion_result.metadata,
        }

        if self.include_timestamps and len(fusion_result.flagged_frames) > 0:
            video_fps = fusion_result.metadata.get("video_fps", 25.0)
            report["flagged_frames"] = [
                {
                    "frame_index": int(f),
                    "timestamp_sec": round(f / video_fps, 3),
                    "discrepancy_score": float(fusion_result.discrepancy_scores[f]),
                }
                for f in fusion_result.flagged_frames
            ]

        if self.include_heatmaps and len(fusion_result.heatmap) > 0:
            scores = fusion_result.discrepancy_scores
            report["score_statistics"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
            }

        return report

    def to_json(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        output_path: str,
        source_file: str | None = None,
    ) -> str:
        """Export report as JSON."""
        report = self._build_report_dict(fusion_result, analysis, source_file)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return str(path)

    def to_text(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None = None,
    ) -> str:
        """Generate a plain-text report string."""
        file_hash = self._compute_file_hash(source_file) if source_file else None
        video_fps = fusion_result.metadata.get("video_fps", 25.0)

        lines = [
            "=" * 60,
            "DEEP-GUARD AGENT — FORENSIC ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Source: {source_file or 'N/A'}",
        ]
        if file_hash:
            lines.append(f"SHA-256: {file_hash}")

        lines.extend([
            "",
            f"VERDICT: {analysis.verdict.upper()}",
            f"Confidence: {analysis.confidence:.1%}",
        ])

        if analysis.harm_category:
            category_labels = {
                "political": "Political Deepfake (Misinformation)",
                "ncii": "Non-Consensual Intimate Imagery (NCII)",
                "financial_fraud": "Financial Fraud (Impersonation)",
                "general": "General Synthetic Media",
            }
            lines.append(f"Harm Category: {category_labels.get(analysis.harm_category, analysis.harm_category)}")

        lines.extend([
            "",
            "--- Summary ---",
            analysis.summary,
        ])

        if analysis.phoneme_analysis:
            lines.extend([
                "",
                "--- Phoneme Analysis ---",
                analysis.phoneme_analysis,
            ])

        lines.extend(["", "--- Evidence ---"])
        for e in analysis.evidence:
            lines.append(f"  * {e}")

        if analysis.recommended_actions:
            lines.extend(["", "--- Recommended Actions ---"])
            for a in analysis.recommended_actions:
                lines.append(f"  * {a}")

        if analysis.confidence_reasoning:
            lines.extend([
                "",
                "--- Confidence Reasoning ---",
                analysis.confidence_reasoning,
            ])

        meta = fusion_result.metadata
        scores = fusion_result.discrepancy_scores
        lines.extend([
            "",
            "--- Detection Metadata ---",
            f"  Frames analyzed: {meta.get('num_frames_analyzed', 'N/A')}",
            f"  Frames flagged: {meta.get('num_flagged_frames', 'N/A')}",
            f"  Flagged ratio: {meta.get('flagged_ratio', 0):.1%}",
            f"  Threshold: {meta.get('threshold', 'N/A')}",
            f"  Mean cosine similarity: {meta.get('mean_cosine_similarity', 'N/A')}",
        ])

        if len(scores) > 0:
            lines.extend([
                f"  Score range: [{float(np.min(scores)):.3f}, {float(np.max(scores)):.3f}]",
                f"  Score mean: {float(np.mean(scores)):.3f} (std: {float(np.std(scores)):.3f})",
            ])

        # Top flagged frames with timestamps
        if fusion_result.flagged_frames:
            lines.extend(["", "--- Top Flagged Frames ---"])
            top_flagged = sorted(
                fusion_result.flagged_frames,
                key=lambda f: fusion_result.discrepancy_scores[f],
                reverse=True,
            )[:10]
            for f in top_flagged:
                ts = f / video_fps
                lines.append(
                    f"  Frame {f} ({ts:.1f}s): score={fusion_result.discrepancy_scores[f]:.3f}"
                )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    @staticmethod
    def _build_svg_chart(
        scores: np.ndarray,
        threshold: float,
        width: int = 700,
        height: int = 160,
    ) -> str:
        """Build an inline SVG discrepancy timeline chart (no JS needed)."""
        if len(scores) == 0:
            return '<p style="color:#94a3b8;font-size:.85rem;">No score data available.</p>'

        pad_l, pad_r, pad_t, pad_b = 34, 12, 16, 24
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b
        n = len(scores)
        step = pw / max(n - 1, 1)

        # Build polyline points for the score line
        points = []
        for i, s in enumerate(scores):
            x = round(pad_l + i * step, 2)
            y = round(pad_t + (1 - float(s)) * ph, 2)
            points.append(f"{x},{y}")
        polyline = " ".join(points)

        # Build polygon for the area fill (close to bottom)
        first_x = round(pad_l, 2)
        last_x = round(pad_l + (n - 1) * step, 2)
        bottom_y = round(pad_t + ph, 2)
        area_poly = f"{first_x},{bottom_y} {polyline} {last_x},{bottom_y}"

        # Threshold line y
        thresh_y = round(pad_t + (1 - threshold) * ph, 2)

        # Grid lines
        grid_lines = ""
        for v in [0, 0.25, 0.5, 0.75, 1.0]:
            gy = round(pad_t + (1 - v) * ph, 2)
            grid_lines += f'<line x1="{pad_l}" y1="{gy}" x2="{pad_l + pw}" y2="{gy}" stroke="#e2e8f0" stroke-width="1"/>\n'

        svg = f"""\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"
     style="width:100%;height:auto;display:block;" preserveAspectRatio="xMidYMid meet">
  <defs>
    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="rgba(59,130,246,0.22)"/>
      <stop offset="100%" stop-color="rgba(59,130,246,0.02)"/>
    </linearGradient>
  </defs>
  <!-- grid -->
  {grid_lines}
  <!-- threshold -->
  <line x1="{pad_l}" y1="{thresh_y}" x2="{pad_l + pw}" y2="{thresh_y}"
        stroke="#fca5a5" stroke-width="1" stroke-dasharray="5,3"/>
  <text x="{pad_l + pw - 2}" y="{thresh_y - 5}" text-anchor="end"
        fill="#f87171" font-size="9" font-family="Inter,sans-serif" font-weight="500">threshold</text>
  <!-- area -->
  <polygon points="{area_poly}" fill="url(#areaGrad)"/>
  <!-- line -->
  <polyline points="{polyline}" fill="none" stroke="#3b82f6" stroke-width="1.5"
            stroke-linejoin="round" stroke-linecap="round"/>
  <!-- y-axis labels -->
  <text x="{pad_l - 6}" y="{pad_t + 4}" text-anchor="end"
        fill="#94a3b8" font-size="10" font-family="Inter,sans-serif">1</text>
  <text x="{pad_l - 6}" y="{pad_t + ph + 4}" text-anchor="end"
        fill="#94a3b8" font-size="10" font-family="Inter,sans-serif">0</text>
  <!-- x-axis label -->
  <text x="{pad_l + pw / 2}" y="{height - 3}" text-anchor="middle"
        fill="#94a3b8" font-size="10" font-family="Inter,sans-serif">Frame</text>
</svg>"""
        return svg

    def _build_html_body(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None = None,
    ) -> str:
        """Build the report HTML body (no JavaScript required)."""
        report = self._build_report_dict(fusion_result, analysis, source_file)
        video_fps = fusion_result.metadata.get("video_fps", 25.0)

        verdict_colors = {
            "authentic": "#22c55e",
            "suspicious": "#f59e0b",
            "likely_fake": "#ef4444",
        }
        verdict_color = verdict_colors.get(analysis.verdict, "#6b7280")

        scores = fusion_result.discrepancy_scores
        threshold = fusion_result.metadata.get("threshold", 0.65)

        # Build SVG chart (pure markup, no JS)
        svg_chart = self._build_svg_chart(scores, threshold)

        evidence_html = "\n".join(f"<li>{e}</li>" for e in analysis.evidence)
        actions_html = "\n".join(f"<li>{a}</li>" for a in analysis.recommended_actions) if analysis.recommended_actions else "<li>No specific actions recommended</li>"

        flagged_rows = ""
        if fusion_result.flagged_frames:
            top_flagged = sorted(
                fusion_result.flagged_frames,
                key=lambda f: fusion_result.discrepancy_scores[f],
                reverse=True,
            )[:20]
            for f in top_flagged:
                ts = f / video_fps
                s = fusion_result.discrepancy_scores[f]
                flagged_rows += f"<tr><td>{f}</td><td>{ts:.1f}s</td><td>{s:.3f}</td></tr>\n"

        # ── Stat pills ──
        meta = report['detection_metadata']
        stats_html = (
            f'<span class="dg-pill">Frames {meta.get("num_frames_analyzed","?")}</span>'
            f'<span class="dg-pill">Flagged {meta.get("num_flagged_frames","0")}'
            f' ({meta.get("flagged_ratio",0):.0%})</span>'
            f'<span class="dg-pill">Cos-sim {meta.get("mean_cosine_similarity","?"):.3f}</span>'
            if isinstance(meta.get("mean_cosine_similarity"), (int, float))
            else (
                f'<span class="dg-pill">Frames {meta.get("num_frames_analyzed","?")}</span>'
                f'<span class="dg-pill">Flagged {meta.get("num_flagged_frames","0")}'
                f' ({meta.get("flagged_ratio",0):.0%})</span>'
            )
        )

        body = f"""\
  <!-- verdict banner -->
  <div class="dg-banner" style="border-left:4px solid {verdict_color};">
    <div class="dg-verdict-row">
      <span class="dg-badge" style="background:{verdict_color};">
        {analysis.verdict.upper().replace('_',' ')}</span>
      <span class="dg-conf">{analysis.confidence:.0%} confidence</span>
      {f'<span class="dg-harm">{analysis.harm_category}</span>' if analysis.harm_category else ''}
    </div>
    <p class="dg-summary">{analysis.summary}</p>
    <div class="dg-pills">{stats_html}</div>
  </div>

  <!-- timeline chart -->
  <div class="dg-section">
    <h3>Discrepancy Timeline</h3>
    {svg_chart}
  </div>

  <!-- evidence -->
  <div class="dg-section">
    <h3>Evidence</h3>
    <ul class="dg-list">{evidence_html}</ul>
  </div>

  {"<div class='dg-section'><h3>Phoneme Analysis</h3><p>" + analysis.phoneme_analysis + "</p></div>" if analysis.phoneme_analysis else ""}

  <!-- actions -->
  <div class="dg-section">
    <h3>Recommended Actions</h3>
    <ul class="dg-list">{actions_html}</ul>
  </div>

  <!-- flagged frames -->
  {('<div class="dg-section"><h3>Flagged Frames</h3><table><thead><tr>'
    '<th>Frame</th><th>Time</th><th>Score</th></tr></thead><tbody>'
    + flagged_rows + '</tbody></table></div>') if flagged_rows else ''}

  <p class="dg-foot">
    {Path(source_file).name if source_file else ''} &middot;
    {report['generated_at'][:19]}
    {"&middot; SHA-256 " + report['source_file_hash'][:16] + "..." if report.get('source_file_hash') else ""}
  </p>"""

        return body

    _CSS = """\
  .dg-report *, .dg-report *::before, .dg-report *::after {
      margin:0; padding:0; box-sizing:border-box; }
  .dg-report {
      font-family: 'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      color:#1e293b; line-height:1.55; padding:1.5rem 1.75rem; }
  .dg-report h3 {
      font-size:.8rem; font-weight:600; text-transform:uppercase;
      letter-spacing:.06em; color:#64748b; margin-bottom:.6rem; }
  /* verdict banner */
  .dg-report .dg-banner {
      background:#f8fafc; border-radius:10px; padding:1.25rem 1.5rem; margin-bottom:1.25rem; }
  .dg-report .dg-verdict-row { display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }
  .dg-report .dg-badge {
      display:inline-block; padding:.25rem .9rem; border-radius:20px;
      font-size:.82rem; font-weight:700; color:#fff; letter-spacing:.03em; }
  .dg-report .dg-conf { font-size:.85rem; color:#64748b; font-weight:500; }
  .dg-report .dg-harm {
      font-size:.75rem; color:#64748b; background:#f1f5f9;
      padding:.2rem .6rem; border-radius:4px; }
  .dg-report .dg-summary { margin-top:.7rem; font-size:.92rem; color:#334155; }
  .dg-report .dg-pills { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.8rem; }
  .dg-report .dg-pill {
      font-size:.72rem; font-weight:500; color:#475569; background:#e2e8f0;
      padding:.2rem .55rem; border-radius:4px; white-space:nowrap; }
  /* sections */
  .dg-report .dg-section { margin-bottom:1.25rem; }
  .dg-report .dg-section p { font-size:.88rem; color:#475569; }
  .dg-report .dg-list { padding-left:1.2rem; font-size:.88rem; color:#475569; }
  .dg-report .dg-list li { margin:.25rem 0; }
  /* chart */
  .dg-report svg { max-width:100%; height:auto; display:block; }
  /* table */
  .dg-report table { width:100%; border-collapse:collapse; font-size:.82rem; }
  .dg-report th { color:#94a3b8; font-weight:600; text-align:left;
      padding:.4rem .6rem; border-bottom:2px solid #e2e8f0; }
  .dg-report td { padding:.35rem .6rem; border-bottom:1px solid #f1f5f9; color:#475569; }
  /* footer */
  .dg-report .dg-foot {
      margin-top:1.5rem; padding-top:.75rem; border-top:1px solid #e2e8f0;
      font-size:.7rem; color:#94a3b8; text-align:center;
      font-family:ui-monospace,monospace; word-break:break-all; }"""

    def to_html_embed(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None = None,
    ) -> str:
        """Generate an embeddable HTML fragment for Gradio gr.HTML component."""
        body = self._build_html_body(fusion_result, analysis, source_file)
        return (
            f"<style>{self._CSS}</style>"
            f'<div class="dg-report">{body}</div>'
        )

    def to_html(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        output_path: str,
        source_file: str | None = None,
    ) -> str:
        """Export report as a standalone HTML file with embedded styles."""
        body = self._build_html_body(fusion_result, analysis, source_file)
        html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deep-Guard Agent — Forensic Report</title>
<style>{self._CSS}</style>
</head>
<body class="dg-report" style="padding:2rem;">
{body}
</body>
</html>"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return str(path)
