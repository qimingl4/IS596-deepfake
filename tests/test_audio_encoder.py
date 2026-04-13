"""Tests for the Audio-Articulatory Encoder module."""

import numpy as np
import pytest

from deepguard.detection.audio_encoder import (
    AudioArticulatoryEncoder,
    AudioFeatures,
    PhonemeSegment,
    BILABIAL_PHONEMES,
    WAV2VEC2_FRAME_RATE,
)


class TestPhonemeSegment:
    def test_bilabial_detection(self):
        seg = PhonemeSegment(phoneme="B", start_sec=0.0, end_sec=0.1, is_bilabial=True)
        assert seg.is_bilabial is True

    def test_non_bilabial(self):
        seg = PhonemeSegment(phoneme="T", start_sec=0.0, end_sec=0.1, is_bilabial=False)
        assert seg.is_bilabial is False


class TestBilabialPhonemes:
    def test_expected_phonemes(self):
        for p in ["B", "P", "M", "b", "p", "m"]:
            assert p in BILABIAL_PHONEMES

    def test_non_bilabial_excluded(self):
        for p in ["T", "K", "S", "vowel"]:
            assert p not in BILABIAL_PHONEMES


class TestAudioFeatures:
    def test_dataclass_fields(self):
        features = AudioFeatures(
            embeddings=np.zeros((10, 768)),
            phoneme_segments=[],
            sample_rate=16000,
            duration_sec=1.0,
        )
        assert features.frame_rate == WAV2VEC2_FRAME_RATE
        assert features.embeddings.shape == (10, 768)


class TestConstants:
    def test_frame_rate(self):
        assert WAV2VEC2_FRAME_RATE == 50.0
