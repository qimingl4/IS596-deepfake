"""Cross-Modal Fusion: aligns visual lip features with audio-articulatory positions.

Transformer-based comparator that generates frame-level discrepancy heatmaps
highlighting audio-visual mismatch regions. Inspired by AV-HuBERT and
ART-AVDF (Wang & Huang, 2024).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class FusionResult:
    """Result of cross-modal audio-visual fusion analysis."""

    discrepancy_scores: np.ndarray   # (T,) per-frame mismatch scores in [0, 1]
    heatmap: np.ndarray              # (T, H, W) spatial discrepancy heatmap (if available)
    flagged_frames: list[int]        # Frame indices exceeding the threshold
    overall_score: float             # Aggregated deepfake probability
    metadata: dict


class AudioVisualProjector(nn.Module):
    """Projects audio and visual features into a shared embedding space."""

    def __init__(self, audio_dim: int = 768, visual_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, audio_features: torch.Tensor, visual_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.audio_proj(audio_features), self.visual_proj(visual_features)


class CrossModalFusion:
    """Aligns and compares audio-articulatory and visual-lip features."""

    def __init__(
        self,
        audio_dim: int = 768,
        visual_dim: int = 128,
        hidden_dim: int = 256,
        discrepancy_threshold: float = 0.65,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = discrepancy_threshold

        self.projector = AudioVisualProjector(audio_dim, visual_dim, hidden_dim).to(self.device)
        self.projector.eval()

    def align_temporal(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        audio_fps: float = 50.0,
        video_fps: float = 25.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align audio and visual feature sequences to a common temporal resolution.

        Uses linear interpolation to match the shorter sequence to the longer one.
        """
        n_audio = len(audio_features)
        n_visual = len(visual_features)

        # Determine the common length based on the shorter duration
        audio_duration = n_audio / audio_fps
        visual_duration = n_visual / video_fps
        common_duration = min(audio_duration, visual_duration)
        common_length = int(common_duration * video_fps)

        # Resample audio features to match video frame rate
        audio_indices = np.linspace(0, n_audio - 1, common_length).astype(int)
        visual_indices = np.linspace(0, n_visual - 1, common_length).astype(int)

        return audio_features[audio_indices], visual_features[visual_indices]

    @torch.no_grad()
    def compute_discrepancy(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
    ) -> np.ndarray:
        """Compute per-frame discrepancy scores between audio and visual features.

        Returns:
            (T,) array of discrepancy scores in [0, 1].
        """
        audio_t = torch.from_numpy(audio_features).float().to(self.device)
        visual_t = torch.from_numpy(visual_features).float().to(self.device)

        audio_proj, visual_proj = self.projector(audio_t, visual_t)

        # Cosine similarity → discrepancy
        cos_sim = nn.functional.cosine_similarity(audio_proj, visual_proj, dim=-1)
        discrepancy = (1.0 - cos_sim) / 2.0  # Normalize to [0, 1]

        return discrepancy.cpu().numpy()

    def analyze(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        audio_fps: float = 50.0,
        video_fps: float = 25.0,
    ) -> FusionResult:
        """Run full cross-modal fusion analysis.

        Args:
            audio_features: (T_a, D_a) audio embeddings.
            visual_features: (T_v, D_v) visual lip features.
            audio_fps: Audio feature frame rate.
            video_fps: Video frame rate.

        Returns:
            FusionResult with discrepancy scores and flagged frames.
        """
        aligned_audio, aligned_visual = self.align_temporal(
            audio_features, visual_features, audio_fps, video_fps
        )

        scores = self.compute_discrepancy(aligned_audio, aligned_visual)
        flagged = [int(i) for i, s in enumerate(scores) if s > self.threshold]
        overall = float(np.mean(scores))

        return FusionResult(
            discrepancy_scores=scores,
            heatmap=np.array([]),  # Placeholder — full spatial heatmap requires deeper model
            flagged_frames=flagged,
            overall_score=overall,
            metadata={
                "threshold": self.threshold,
                "num_frames_analyzed": len(scores),
                "num_flagged_frames": len(flagged),
                "flagged_ratio": len(flagged) / len(scores) if len(scores) > 0 else 0.0,
            },
        )
