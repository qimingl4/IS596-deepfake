"""Tests for the Report Generator module."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from deepguard.detection.fusion import FusionResult
from deepguard.reasoning.llm_reasoner import AnalysisReport
from deepguard.interface.report import ReportGenerator


def _make_fixtures():
    """Create test fixtures for fusion result and analysis report."""
    fusion = FusionResult(
        discrepancy_scores=np.array([0.2, 0.8, 0.3, 0.9, 0.1]),
        heatmap=np.array([]),
        flagged_frames=[1, 3],
        overall_score=0.46,
        metadata={
            "threshold": 0.65,
            "num_frames_analyzed": 5,
            "num_flagged_frames": 2,
            "flagged_ratio": 0.4,
        },
    )
    analysis = AnalysisReport(
        summary="Test summary.",
        verdict="suspicious",
        confidence=0.46,
        evidence=["Evidence 1", "Evidence 2"],
        harm_category=None,
        recommended_actions=["Action 1"],
        raw_response="Raw LLM response.",
    )
    return fusion, analysis


class TestReportGenerator:
    def test_to_text_contains_verdict(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        text = gen.to_text(fusion, analysis)
        assert "SUSPICIOUS" in text
        assert "Evidence 1" in text

    def test_to_json_creates_valid_file(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        with TemporaryDirectory() as tmp:
            path = gen.to_json(fusion, analysis, f"{tmp}/report.json")
            data = json.loads(Path(path).read_text())
            assert data["verdict"] == "suspicious"
            assert len(data["flagged_frames"]) == 2
