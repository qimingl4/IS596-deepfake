"""Report Generator: produces structured, traceable technical reports.

Exports timestamped evidence usable for platform moderation or legal proceedings.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from deepguard.detection.fusion import FusionResult
from deepguard.reasoning.llm_reasoner import AnalysisReport


class ReportGenerator:
    """Generates analysis reports in various formats."""

    def __init__(self, include_timestamps: bool = True, include_heatmaps: bool = True):
        self.include_timestamps = include_timestamps
        self.include_heatmaps = include_heatmaps

    def _build_report_dict(
        self,
        fusion_result: FusionResult,
        analysis: AnalysisReport,
        source_file: str | None = None,
    ) -> dict:
        """Build a structured report dictionary."""
        report = {
            "report_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
            "verdict": analysis.verdict,
            "confidence": analysis.confidence,
            "summary": analysis.summary,
            "evidence": analysis.evidence,
            "harm_category": analysis.harm_category,
            "recommended_actions": analysis.recommended_actions,
            "detection_metadata": fusion_result.metadata,
        }

        if self.include_timestamps and len(fusion_result.flagged_frames) > 0:
            report["flagged_frames"] = [
                {
                    "frame_index": int(f),
                    "discrepancy_score": float(fusion_result.discrepancy_scores[f]),
                }
                for f in fusion_result.flagged_frames
            ]

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
        lines = [
            "=" * 60,
            "DEEP-GUARD AGENT — FORENSIC ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Source: {source_file or 'N/A'}",
            "",
            f"VERDICT: {analysis.verdict.upper()}",
            f"Confidence: {analysis.confidence:.1%}",
            "",
            "--- Summary ---",
            analysis.summary,
            "",
            "--- Evidence ---",
        ]
        for e in analysis.evidence:
            lines.append(f"  • {e}")

        if analysis.recommended_actions:
            lines.append("")
            lines.append("--- Recommended Actions ---")
            for a in analysis.recommended_actions:
                lines.append(f"  • {a}")

        meta = fusion_result.metadata
        lines.extend([
            "",
            "--- Detection Metadata ---",
            f"  Frames analyzed: {meta.get('num_frames_analyzed', 'N/A')}",
            f"  Frames flagged: {meta.get('num_flagged_frames', 'N/A')}",
            f"  Flagged ratio: {meta.get('flagged_ratio', 0):.1%}",
            f"  Threshold: {meta.get('threshold', 'N/A')}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
