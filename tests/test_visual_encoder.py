"""Tests for the Visual-Lip Encoder module."""

import numpy as np
import pytest

from deepguard.detection.visual_encoder import (
    VisualLipEncoder,
    VISUAL_FEATURE_DIM,
    ALL_LIP_INDICES,
    FACE_OVAL_INDICES,
)


def _make_encoder():
    """Create an encoder instance without initializing MediaPipe."""
    encoder = VisualLipEncoder.__new__(VisualLipEncoder)
    encoder.temporal_smoothing = 0.0
    encoder._prev_landmarks = None
    return encoder


def _make_landmarks(n=30):
    """Create synthetic lip landmarks."""
    rng = np.random.RandomState(42)
    # Simulate a horizontal mouth shape
    x = np.linspace(100, 200, n)
    y = 200 + 10 * np.sin(np.linspace(0, np.pi, n))
    z = np.zeros(n)
    return np.column_stack([x, y, z])


def _make_all_landmarks():
    """Create synthetic full-face landmarks (478 points)."""
    rng = np.random.RandomState(42)
    return rng.rand(478, 3) * 300 + 50


class TestLipAspectRatio:
    def test_normal_mouth(self):
        encoder = _make_encoder()
        landmarks = np.array([
            [100, 200, 0], [200, 200, 0],
            [150, 190, 0], [150, 210, 0],
        ])
        ratio = encoder.compute_shape_descriptors(landmarks, _make_all_landmarks())["aspect_ratio"]
        assert 0 < ratio < 1

    def test_zero_width(self):
        encoder = _make_encoder()
        landmarks = np.array([
            [100, 200, 0], [100, 210, 0], [100, 220, 0],
        ])
        ratio = encoder.compute_shape_descriptors(landmarks, _make_all_landmarks())["aspect_ratio"]
        assert ratio == 0.0


class TestShapeDescriptors:
    def test_returns_all_keys(self):
        encoder = _make_encoder()
        landmarks = _make_landmarks()
        all_lm = _make_all_landmarks()
        desc = encoder.compute_shape_descriptors(landmarks, all_lm)
        expected_keys = {
            "aspect_ratio", "openness", "asymmetry",
            "eccentricity", "corner_angle", "bilabial_closure",
        }
        assert set(desc.keys()) == expected_keys

    def test_eccentricity_range(self):
        encoder = _make_encoder()
        desc = encoder.compute_shape_descriptors(_make_landmarks(), _make_all_landmarks())
        assert 0 <= desc["eccentricity"] <= 1

    def test_openness_nonnegative(self):
        encoder = _make_encoder()
        desc = encoder.compute_shape_descriptors(_make_landmarks(), _make_all_landmarks())
        assert desc["openness"] >= 0


class TestFeatureVector:
    def test_output_dimension(self):
        encoder = _make_encoder()
        landmarks = _make_landmarks()
        desc = encoder.compute_shape_descriptors(landmarks, _make_all_landmarks())
        vec = encoder.landmarks_to_feature_vector(landmarks, desc)
        assert vec.shape == (VISUAL_FEATURE_DIM,)

    def test_consistent_output(self):
        encoder = _make_encoder()
        landmarks = _make_landmarks()
        desc = encoder.compute_shape_descriptors(landmarks, _make_all_landmarks())
        vec1 = encoder.landmarks_to_feature_vector(landmarks, desc)
        vec2 = encoder.landmarks_to_feature_vector(landmarks, desc)
        np.testing.assert_array_equal(vec1, vec2)


class TestFaceBbox:
    def test_bbox_within_frame(self):
        encoder = _make_encoder()
        all_lm = _make_all_landmarks()
        bbox = encoder.compute_face_bbox(all_lm, (480, 640, 3))
        x, y, w, h = bbox
        assert x >= 0
        assert y >= 0
        assert w > 0
        assert h > 0
