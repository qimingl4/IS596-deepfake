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

    def _build_html_body(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None = None,
    ) -> tuple[str, str, str]:
        """Build shared HTML components. Returns (body_html, score_data_js, threshold)."""
        report = self._build_report_dict(fusion_result, analysis, source_file)
        video_fps = fusion_result.metadata.get("video_fps", 25.0)

        verdict_colors = {
            "authentic": "#22c55e",
            "suspicious": "#f59e0b",
            "likely_fake": "#ef4444",
        }
        verdict_color = verdict_colors.get(analysis.verdict, "#6b7280")

        scores = fusion_result.discrepancy_scores
        score_data_js = json.dumps([round(float(s), 3) for s in scores])
        threshold = fusion_result.metadata.get("threshold", 0.65)

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

        body = f"""\
  <h1>Deep-Guard Agent</h1>
  <p class="dg-meta">Forensic Analysis Report &mdash; {report['generated_at']}</p>
  <p class="dg-meta">Source: {Path(source_file).name if source_file else 'N/A'}</p>
  {"<p class='dg-hash'>SHA-256: " + report['source_file_hash'] + "</p>" if report.get('source_file_hash') else ""}

  <div class="dg-card">
    <span class="dg-verdict" style="background:{verdict_color};">{analysis.verdict.upper().replace('_', ' ')}</span>
    <span class="dg-confidence">Confidence: {analysis.confidence:.1%}</span>
    {f'<p style="margin-top:0.5rem; color:#94a3b8;">Harm Category: {analysis.harm_category}</p>' if analysis.harm_category else ''}
  </div>

  <h2>Summary</h2>
  <div class="dg-card"><p>{analysis.summary}</p></div>

  {"<h2>Phoneme Analysis</h2><div class='dg-card'><p>" + analysis.phoneme_analysis + "</p></div>" if analysis.phoneme_analysis else ""}

  <h2>Evidence</h2>
  <div class="dg-card"><ul>{evidence_html}</ul></div>

  <h2>Recommended Actions</h2>
  <div class="dg-card"><ul>{actions_html}</ul></div>

  <h2>Discrepancy Timeline</h2>
  <div class="dg-card">
    <canvas id="dg-timeline" style="width:100%;height:200px;margin:1rem 0;"></canvas>
  </div>

  <h2>Top Flagged Frames</h2>
  <div class="dg-card">
    <table>
      <thead><tr><th>Frame</th><th>Timestamp</th><th>Score</th></tr></thead>
      <tbody>{flagged_rows if flagged_rows else "<tr><td colspan='3'>No flagged frames</td></tr>"}</tbody>
    </table>
  </div>

  <h2>Detection Metadata</h2>
  <div class="dg-card">
    <table>
      <tr><td>Frames analyzed</td><td>{report['detection_metadata'].get('num_frames_analyzed', 'N/A')}</td></tr>
      <tr><td>Frames flagged</td><td>{report['detection_metadata'].get('num_flagged_frames', 'N/A')}</td></tr>
      <tr><td>Flagged ratio</td><td>{report['detection_metadata'].get('flagged_ratio', 0):.1%}</td></tr>
      <tr><td>Threshold</td><td>{report['detection_metadata'].get('threshold', 'N/A')}</td></tr>
      <tr><td>Mean cosine similarity</td><td>{report['detection_metadata'].get('mean_cosine_similarity', 'N/A')}</td></tr>
    </table>
  </div>

  <div class="dg-footer">
    Generated by Deep-Guard Agent v2.0 &mdash; For forensic and research purposes only.
  </div>"""

        return body, score_data_js, str(threshold)

    _CSS = """\
  .dg-report * { margin: 0; padding: 0; box-sizing: border-box; }
  .dg-report { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f172a; color: #e2e8f0; padding: 2rem; line-height: 1.6;
       border-radius: 12px; }
  .dg-report h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #f8fafc; }
  .dg-report h2 { font-size: 1.2rem; margin: 1.5rem 0 0.5rem; color: #94a3b8;
       border-bottom: 1px solid #334155; padding-bottom: 0.3rem; }
  .dg-report .dg-meta { color: #64748b; font-size: 0.85rem; margin-bottom: 0.3rem; }
  .dg-report .dg-verdict { display: inline-block; padding: 0.4rem 1.2rem; border-radius: 6px;
              font-size: 1.4rem; font-weight: 700; color: white; margin: 0.5rem 0; }
  .dg-report .dg-confidence { font-size: 1.1rem; color: #94a3b8; margin-left: 1rem; }
  .dg-report .dg-card { background: #1e293b; border-radius: 8px; padding: 1.2rem; margin: 0.8rem 0; }
  .dg-report ul { padding-left: 1.5rem; }
  .dg-report li { margin: 0.3rem 0; }
  .dg-report table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
  .dg-report th, .dg-report td { padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid #334155; }
  .dg-report th { color: #94a3b8; font-weight: 600; }
  .dg-report .dg-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #334155;
             color: #475569; font-size: 0.8rem; text-align: center; }
  .dg-report .dg-hash { font-family: monospace; font-size: 0.75rem; color: #64748b; word-break: break-all; }"""

    _CHART_JS = """\
const scores = __SCORES__;
const threshold = __THRESHOLD__;
const canvas = document.getElementById('dg-timeline');
if (canvas) {
  const ctx = canvas.getContext('2d');
  function draw() {
    const W = canvas.width = canvas.offsetWidth * 2;
    const H = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    const w = W / 2, h = H / 2, pad = 30;
    ctx.clearRect(0, 0, w, h);
    const ty = pad + (1 - threshold) * (h - 2 * pad);
    ctx.strokeStyle = '#ef444488'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad, ty); ctx.lineTo(w - pad, ty); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = '#ef4444'; ctx.font = '10px sans-serif';
    ctx.fillText('threshold', w - pad - 50, ty - 4);
    if (scores.length > 0) {
      const step = (w - 2 * pad) / (scores.length - 1 || 1);
      ctx.beginPath();
      scores.forEach((s, i) => {
        const x = pad + i * step, y = pad + (1 - s) * (h - 2 * pad);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.strokeStyle = '#475569'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.lineTo(w - pad, h - pad); ctx.stroke();
    ctx.fillStyle = '#94a3b8'; ctx.font = '10px sans-serif';
    ctx.fillText('1.0', 4, pad + 4); ctx.fillText('0.0', 4, h - pad + 4);
    ctx.fillText('Frames →', w / 2, h - 6);
  }
  draw(); window.addEventListener('resize', draw);
}"""

    def to_html_embed(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None = None,
    ) -> str:
        """Generate an embeddable HTML fragment for Gradio gr.HTML component."""
        body, score_data_js, threshold = self._build_html_body(
            fusion_result, analysis, source_file,
        )
        chart_js = self._CHART_JS.replace("__SCORES__", score_data_js).replace(
            "__THRESHOLD__", threshold
        )
        return (
            f"<style>{self._CSS}</style>"
            f'<div class="dg-report">{body}</div>'
            f"<script>{chart_js}</script>"
        )

    def to_html(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        output_path: str,
        source_file: str | None = None,
    ) -> str:
        """Export report as a standalone HTML file with embedded styles."""
        body, score_data_js, threshold = self._build_html_body(
            fusion_result, analysis, source_file,
        )
        chart_js = self._CHART_JS.replace("__SCORES__", score_data_js).replace(
            "__THRESHOLD__", threshold
        )
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
<script>{chart_js}</script>
</body>
</html>"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return str(path)
