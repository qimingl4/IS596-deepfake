"""Tests for the Cross-Modal Fusion module."""

import numpy as np
import pytest
import torch

from deepguard.detection.fusion import CrossModalFusion, AudioVisualProjector
from deepguard.detection.visual_encoder import VISUAL_FEATURE_DIM


class TestTemporalAlignment:
    def test_matching_durations(self):
        fusion = CrossModalFusion(audio_dim=8, visual_dim=8)
        audio = np.random.randn(100, 8)  # 100 frames at 50fps = 2s
        visual = np.random.randn(50, 8)  # 50 frames at 25fps = 2s

        a_aligned, v_aligned = fusion.align_temporal(audio, visual, 50.0, 25.0)
        assert len(a_aligned) == len(v_aligned)

    def test_different_durations(self):
        fusion = CrossModalFusion(audio_dim=8, visual_dim=8)
        audio = np.random.randn(200, 8)  # 4s at 50fps
        visual = np.random.randn(50, 8)  # 2s at 25fps

        a_aligned, v_aligned = fusion.align_temporal(audio, visual, 50.0, 25.0)
        assert len(a_aligned) == len(v_aligned)
        assert len(a_aligned) == 50

    def test_empty_features(self):
        fusion = CrossModalFusion(audio_dim=8, visual_dim=8)
        audio = np.zeros((0, 8))
        visual = np.random.randn(50, 8)

        a_aligned, v_aligned = fusion.align_temporal(audio, visual, 50.0, 25.0)
        assert len(a_aligned) == len(v_aligned) == 1


class TestAnalyze:
    def test_valid_result(self):
        fusion = CrossModalFusion(audio_dim=8, visual_dim=8, discrepancy_threshold=0.5)
        audio = np.random.randn(100, 8)
        visual = np.random.randn(50, 8)

        result = fusion.analyze(audio, visual, 50.0, 25.0)
        assert 0 <= result.overall_score <= 1
        assert len(result.discrepancy_scores) > 0
        assert "num_frames_analyzed" in result.metadata
        assert "mean_cosine_similarity" in result.metadata

    def test_default_dimensions(self):
        fusion = CrossModalFusion()
        audio = np.random.randn(100, 768)
        visual = np.random.randn(50, VISUAL_FEATURE_DIM)

        result = fusion.analyze(audio, visual, 50.0, 25.0)
        assert result.discrepancy_scores.shape[0] == 50


class TestAudioVisualProjector:
    def test_forward_shapes(self):
        proj = AudioVisualProjector(audio_dim=16, visual_dim=16, hidden_dim=8)
        a = torch.randn(10, 16)
        v = torch.randn(10, 16)
        a_proj, v_proj, scores = proj(a, v)
        assert a_proj.shape == (10, 8)
        assert v_proj.shape == (10, 8)
        assert scores.shape == (10,)

    def test_scores_in_range(self):
        proj = AudioVisualProjector(audio_dim=16, visual_dim=16, hidden_dim=8)
        a = torch.randn(20, 16)
        v = torch.randn(20, 16)
        _, _, scores = proj(a, v)
        assert (scores >= 0).all() and (scores <= 1).all()


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        fusion = CrossModalFusion(audio_dim=8, visual_dim=8, hidden_dim=4)
        path = str(tmp_path / "model.pt")
        fusion.save_checkpoint(path)

        fusion2 = CrossModalFusion(
            audio_dim=8, visual_dim=8, hidden_dim=4, checkpoint_path=path
        )
        # Verify weights match
        for p1, p2 in zip(
            fusion.projector.parameters(), fusion2.projector.parameters()
        ):
            torch.testing.assert_close(p1, p2)
