"""Audio-Articulatory Encoder: maps audio signals to vocal tract positions.

Self-supervised module using Wav2Vec2 to extract audio features and optionally
Montreal Forced Aligner (MFA) for phoneme-level alignment. Rooted in physical
acoustics — vocal-tract movements obey physical laws that AI cannot easily fake.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

logger = logging.getLogger(__name__)

# Bilabial phonemes that require lip closure — key detection signals
BILABIAL_PHONEMES = {"B", "P", "M", "b", "p", "m"}

# Wav2Vec2 produces one frame per 20ms of audio
WAV2VEC2_FRAME_RATE = 50.0  # frames per second


@dataclass
class PhonemeSegment:
    """A single phoneme aligned to a time range."""

    phoneme: str
    start_sec: float
    end_sec: float
    is_bilabial: bool = False


@dataclass
class AudioFeatures:
    """Extracted audio-articulatory features for a segment."""

    embeddings: np.ndarray                  # (T, D) frame-level audio embeddings
    phoneme_segments: list[PhonemeSegment]   # Aligned phonemes from MFA
    sample_rate: int
    duration_sec: float
    frame_rate: float = WAV2VEC2_FRAME_RATE  # Audio feature frame rate


class AudioArticulatoryEncoder:
    """Extracts articulatory representations from audio using Wav2Vec2."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str | None = None,
        use_mfa: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mfa = use_mfa

        logger.info("Loading Wav2Vec2 model: %s on %s", model_name, self.device)
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

    def extract_audio_from_video(
        self, video_path: str, target_sr: int = 16000
    ) -> tuple[np.ndarray, int]:
        """Extract audio track from a video file using ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vn",                    # no video
                    "-acodec", "pcm_s16le",   # WAV PCM
                    "-ar", str(target_sr),     # resample
                    "-ac", "1",               # mono
                    wav_path,
                ],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                # Check if the video simply has no audio track
                if "does not contain any stream" in result.stderr or \
                   "Output file is empty" in result.stderr:
                    logger.warning("Video has no audio track: %s", video_path)
                    return np.zeros(int(target_sr * 0.1), dtype=np.float32), target_sr
                raise RuntimeError(
                    f"ffmpeg failed to extract audio: {result.stderr[-300:]}"
                )

            waveform, sr = librosa.load(wav_path, sr=target_sr, mono=True)
            return waveform, sr
        except FileNotFoundError:
            # ffmpeg not installed — fall back to librosa direct load
            logger.warning("ffmpeg not found, falling back to librosa for audio extraction")
            waveform, sr = librosa.load(video_path, sr=target_sr, mono=True)
            return waveform, sr
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def run_mfa_alignment(
        self, audio_path: str, transcript: str | None = None
    ) -> list[PhonemeSegment]:
        """Run Montreal Forced Aligner for phoneme-level alignment.

        Args:
            audio_path: Path to a .wav file.
            transcript: Optional transcript text. If None, uses MFA's built-in
                        acoustic model for language-independent alignment.

        Returns:
            List of PhonemeSegment with time-aligned phonemes.
        """
        try:
            from montreal_forced_aligner.command_line.mfa import mfa_cli  # noqa: F401
        except ImportError:
            logger.warning("Montreal Forced Aligner not installed, skipping phoneme alignment")
            return []

        segments: list[PhonemeSegment] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            corpus_dir = tmpdir_path / "corpus"
            corpus_dir.mkdir()

            # Copy audio to corpus
            audio_src = Path(audio_path)
            audio_dst = corpus_dir / audio_src.name
            audio_dst.write_bytes(audio_src.read_bytes())

            # Write transcript if provided
            if transcript:
                lab_file = corpus_dir / audio_src.with_suffix(".lab").name
                lab_file.write_text(transcript)

            output_dir = tmpdir_path / "output"
            output_dir.mkdir()

            try:
                cmd = [
                    "mfa", "align",
                    str(corpus_dir),
                    "english_mfa",      # acoustic model
                    "english_mfa",      # dictionary
                    str(output_dir),
                    "--clean",
                ]
                subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)

                # Parse TextGrid output
                segments = self._parse_textgrid(output_dir)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.warning("MFA alignment failed: %s", e)

        return segments

    def _parse_textgrid(self, output_dir: Path) -> list[PhonemeSegment]:
        """Parse MFA TextGrid output into PhonemeSegment list."""
        segments: list[PhonemeSegment] = []

        for tg_file in output_dir.rglob("*.TextGrid"):
            try:
                import textgrid
                tg = textgrid.TextGrid.fromFile(str(tg_file))
                for tier in tg:
                    if tier.name.lower() == "phones":
                        for interval in tier:
                            if interval.mark and interval.mark.strip():
                                phoneme = interval.mark.strip()
                                segments.append(PhonemeSegment(
                                    phoneme=phoneme,
                                    start_sec=float(interval.minTime),
                                    end_sec=float(interval.maxTime),
                                    is_bilabial=phoneme.upper() in BILABIAL_PHONEMES,
                                ))
            except ImportError:
                logger.warning("textgrid package not installed, parsing TextGrid manually")
                segments.extend(self._parse_textgrid_manual(tg_file))
            except Exception as e:
                logger.warning("Failed to parse TextGrid %s: %s", tg_file, e)

        return segments

    def _parse_textgrid_manual(self, tg_path: Path) -> list[PhonemeSegment]:
        """Minimal TextGrid parser without external dependency."""
        segments: list[PhonemeSegment] = []
        text = tg_path.read_text()
        lines = text.split("\n")

        in_phones_tier = False
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if '"phones"' in line.lower():
                in_phones_tier = True
            elif in_phones_tier and line.startswith("xmin"):
                xmin = float(line.split("=")[1].strip())
                i += 1
                xmax = float(lines[i].strip().split("=")[1].strip())
                i += 1
                mark = lines[i].strip().split("=")[1].strip().strip('"')
                if mark:
                    segments.append(PhonemeSegment(
                        phoneme=mark,
                        start_sec=xmin,
                        end_sec=xmax,
                        is_bilabial=mark.upper() in BILABIAL_PHONEMES,
                    ))
            i += 1

        return segments

    def process(
        self,
        media_path: str,
        target_sr: int = 16000,
        transcript: str | None = None,
    ) -> AudioFeatures:
        """Process an audio or video file and extract articulatory features.

        Args:
            media_path: Path to an audio (.wav) or video (.mp4) file.
            target_sr: Target sample rate.
            transcript: Optional transcript for MFA alignment.

        Returns:
            AudioFeatures with embeddings, phoneme segments, and metadata.
        """
        suffix = Path(media_path).suffix.lower()
        if suffix in (".mp4", ".avi", ".mov", ".mkv"):
            waveform, sr = self.extract_audio_from_video(media_path, target_sr)
        else:
            waveform, sr = self.load_audio(media_path, target_sr)

        if len(waveform) == 0:
            logger.error("No audio data extracted from %s", media_path)
            return AudioFeatures(
                embeddings=np.zeros((1, 768)),
                phoneme_segments=[],
                sample_rate=sr,
                duration_sec=0.0,
            )

        embeddings = self.extract_embeddings(waveform, sr)
        duration = len(waveform) / sr

        # Run MFA if enabled
        phoneme_segments: list[PhonemeSegment] = []
        if self.use_mfa:
            # MFA requires a .wav file — export if input is video
            if suffix in (".mp4", ".avi", ".mov", ".mkv"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    import soundfile as sf
                    sf.write(f.name, waveform, sr)
                    phoneme_segments = self.run_mfa_alignment(f.name, transcript)
            else:
                phoneme_segments = self.run_mfa_alignment(media_path, transcript)

            logger.info(
                "MFA alignment: %d phonemes, %d bilabial",
                len(phoneme_segments),
                sum(1 for s in phoneme_segments if s.is_bilabial),
            )

        return AudioFeatures(
            embeddings=embeddings,
            phoneme_segments=phoneme_segments,
            sample_rate=sr,
            duration_sec=float(duration),
            frame_rate=WAV2VEC2_FRAME_RATE,
        )
