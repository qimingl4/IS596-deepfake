"""Tests for the Visual-Lip Encoder module."""

import numpy as np
import pytest

from deepguard.detection.visual_encoder import VisualLipEncoder


class TestVisualLipEncoder:
    def test_compute_lip_aspect_ratio_normal(self):
        """Lip aspect ratio for a typical mouth shape."""
        encoder = VisualLipEncoder.__new__(VisualLipEncoder)
        # Simulate landmarks: wider than tall
        landmarks = np.array([
            [100, 200, 0], [200, 200, 0],  # left-right spread
            [150, 190, 0], [150, 210, 0],  # top-bottom spread
        ])
        ratio = encoder.compute_lip_aspect_ratio(landmarks)
        assert 0 < ratio < 1  # Height < Width for a normal mouth

    def test_compute_lip_aspect_ratio_zero_width(self):
        """Edge case: all landmarks at same x-coordinate."""
        encoder = VisualLipEncoder.__new__(VisualLipEncoder)
        landmarks = np.array([
            [100, 200, 0], [100, 210, 0], [100, 220, 0],
        ])
        ratio = encoder.compute_lip_aspect_ratio(landmarks)
        assert ratio == 0.0
