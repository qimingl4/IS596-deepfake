"""Tests for the Cross-Modal Fusion module."""

import numpy as np
import pytest

from deepguard.detection.fusion import CrossModalFusion


class TestCrossModalFusion:
    def test_align_temporal_matching_lengths(self):
        """When audio and visual have the same duration, alignment preserves length."""
        fusion = CrossModalFusion(audio_dim=8, visual_dim=8)
        audio = np.random.randn(100, 8)  # 100 frames at 50fps = 2s
        visual = np.random.randn(50, 8)  # 50 frames at 25fps = 2s

        a_aligned, v_aligned = fusion.align_temporal(audio, visual, 50.0, 25.0)
        assert len(a_aligned) == len(v_aligned)

    def test_align_temporal_different_durations(self):
        """Alignment uses the shorter duration."""
        fusion = CrossModalFusion(audio_dim=8, visual_dim=8)
        audio = np.random.randn(200, 8)  # 4s at 50fps
        visual = np.random.randn(50, 8)  # 2s at 25fps

        a_aligned, v_aligned = fusion.align_temporal(audio, visual, 50.0, 25.0)
        assert len(a_aligned) == len(v_aligned)
        assert len(a_aligned) == 50  # 2s * 25fps

    def test_analyze_returns_valid_result(self):
        """Full analyze pipeline returns expected structure."""
        fusion = CrossModalFusion(audio_dim=8, visual_dim=8, discrepancy_threshold=0.5)
        audio = np.random.randn(100, 8)
        visual = np.random.randn(50, 8)

        result = fusion.analyze(audio, visual, 50.0, 25.0)
        assert 0 <= result.overall_score <= 1
        assert len(result.discrepancy_scores) > 0
        assert "num_frames_analyzed" in result.metadata
