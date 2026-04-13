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
        heatmap=np.array([0.2, 0.8, 0.3, 0.9, 0.1]),
        flagged_frames=[1, 3],
        overall_score=0.46,
        metadata={
            "threshold": 0.65,
            "num_frames_analyzed": 5,
            "num_flagged_frames": 2,
            "flagged_ratio": 0.4,
            "mean_cosine_similarity": 0.72,
            "video_fps": 25.0,
            "audio_fps": 50.0,
        },
    )
    analysis = AnalysisReport(
        summary="Test summary.",
        verdict="suspicious",
        confidence=0.46,
        evidence=["Evidence 1", "Evidence 2"],
        harm_category="political",
        recommended_actions=["Action 1"],
        phoneme_analysis="Bilabial mismatch at frame 3.",
        confidence_reasoning="Based on detection scores.",
        raw_response="Raw LLM response.",
    )
    return fusion, analysis


class TestTextReport:
    def test_contains_verdict(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        text = gen.to_text(fusion, analysis)
        assert "SUSPICIOUS" in text
        assert "Evidence 1" in text

    def test_contains_harm_category(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        text = gen.to_text(fusion, analysis)
        assert "Political" in text

    def test_contains_phoneme_analysis(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        text = gen.to_text(fusion, analysis)
        assert "Bilabial mismatch" in text

    def test_contains_score_stats(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        text = gen.to_text(fusion, analysis)
        assert "Score range" in text

    def test_contains_flagged_frames(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        text = gen.to_text(fusion, analysis)
        assert "Frame 3" in text


class TestJsonReport:
    def test_creates_valid_file(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        with TemporaryDirectory() as tmp:
            path = gen.to_json(fusion, analysis, f"{tmp}/report.json")
            data = json.loads(Path(path).read_text())
            assert data["verdict"] == "suspicious"
            assert len(data["flagged_frames"]) == 2
            assert data["harm_category"] == "political"
            assert data["report_version"] == "2.0"

    def test_flagged_frames_have_timestamps(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        with TemporaryDirectory() as tmp:
            path = gen.to_json(fusion, analysis, f"{tmp}/report.json")
            data = json.loads(Path(path).read_text())
            for ff in data["flagged_frames"]:
                assert "timestamp_sec" in ff
                assert "discrepancy_score" in ff

    def test_score_statistics(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        with TemporaryDirectory() as tmp:
            path = gen.to_json(fusion, analysis, f"{tmp}/report.json")
            data = json.loads(Path(path).read_text())
            assert "score_statistics" in data
            stats = data["score_statistics"]
            assert "mean" in stats
            assert "std" in stats


class TestHtmlReport:
    def test_creates_valid_file(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        with TemporaryDirectory() as tmp:
            path = gen.to_html(fusion, analysis, f"{tmp}/report.html")
            content = Path(path).read_text()
            assert "<!DOCTYPE html>" in content
            assert "SUSPICIOUS" in content
            assert "Deep-Guard Agent" in content

    def test_contains_timeline_chart(self):
        fusion, analysis = _make_fixtures()
        gen = ReportGenerator()
        with TemporaryDirectory() as tmp:
            path = gen.to_html(fusion, analysis, f"{tmp}/report.html")
            content = Path(path).read_text()
            assert "canvas" in content
            assert "timeline" in content
