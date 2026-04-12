"""Audio-Articulatory Encoder: maps audio signals to vocal tract positions.

Self-supervised module using Wav2Vec2 to extract audio features and optionally
Montreal Forced Aligner (MFA) for phoneme-level alignment. Rooted in physical
acoustics — vocal-tract movements obey physical laws that AI cannot easily fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


@dataclass
class AudioFeatures:
    """Extracted audio-articulatory features for a segment."""

    embeddings: np.ndarray         # (T, D) frame-level audio embeddings
    phoneme_segments: list[dict]   # [{phoneme, start_sec, end_sec}, ...]
    sample_rate: int
    duration_sec: float


class AudioArticulatoryEncoder:
    """Extracts articulatory representations from audio using Wav2Vec2."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_embeddings(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract frame-level audio embeddings from a waveform.

        Args:
            waveform: 1D audio signal array.
            sample_rate: Audio sample rate in Hz.

        Returns:
            (T, D) array of audio embeddings at ~20ms frame resolution.
        """
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        outputs = self.model(input_values)
        # last_hidden_state: (1, T, D)
        embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return embeddings

    def load_audio(self, audio_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
        """Load and resample an audio file."""
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return waveform, sr

    def extract_audio_from_video(self, video_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
        """Extract audio track from a video file."""
        waveform, sr = librosa.load(video_path, sr=target_sr, mono=True)
        return waveform, sr

    def process(self, media_path: str, target_sr: int = 16000) -> AudioFeatures:
        """Process an audio or video file and extract articulatory features.

        Args:
            media_path: Path to an audio (.wav) or video (.mp4) file.
            target_sr: Target sample rate.

        Returns:
            AudioFeatures with embeddings and metadata.
        """
        suffix = Path(media_path).suffix.lower()
        if suffix in (".mp4", ".avi", ".mov", ".mkv"):
            waveform, sr = self.extract_audio_from_video(media_path, target_sr)
        else:
            waveform, sr = self.load_audio(media_path, target_sr)

        embeddings = self.extract_embeddings(waveform, sr)
        duration = len(waveform) / sr

        return AudioFeatures(
            embeddings=embeddings,
            phoneme_segments=[],  # Populated by MFA if available
            sample_rate=sr,
            duration_sec=float(duration),
        )
