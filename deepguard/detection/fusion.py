"""Cross-Modal Fusion: aligns visual lip features with audio-articulatory positions.

Transformer-based comparator that generates frame-level discrepancy heatmaps
highlighting audio-visual mismatch regions. Inspired by AV-HuBERT and
ART-AVDF (Wang & Huang, 2024).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deepguard.detection.visual_encoder import VISUAL_FEATURE_DIM

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of cross-modal audio-visual fusion analysis."""

    discrepancy_scores: np.ndarray   # (T,) per-frame mismatch scores in [0, 1]
    heatmap: np.ndarray              # (T, H, W) spatial discrepancy heatmap (if available)
    flagged_frames: list[int]        # Frame indices exceeding the threshold
    overall_score: float             # Aggregated deepfake probability
    metadata: dict


class TemporalAttention(nn.Module):
    """Multi-head temporal attention for capturing phoneme transitions."""

    def __init__(self, hidden_dim: int = 256, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D) -> (1, T, D) -> transformer -> (1, T, D) -> (T, D)
        return self.transformer(x.unsqueeze(0)).squeeze(0)


class AudioVisualProjector(nn.Module):
    """Projects audio and visual features into a shared embedding space
    with temporal attention for capturing cross-frame context."""

    def __init__(
        self,
        audio_dim: int = 768,
        visual_dim: int = VISUAL_FEATURE_DIM,
        hidden_dim: int = 256,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(hidden_dim)

        # Discrepancy head: predicts mismatch score from concatenated features
        self.discrepancy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, audio_features: torch.Tensor, visual_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and compute discrepancy.

        Returns:
            (audio_proj, visual_proj, discrepancy_scores)
        """
        a_proj = self.audio_proj(audio_features)
        v_proj = self.visual_proj(visual_features)

        if self.use_temporal_attention:
            # Apply temporal attention to the concatenated representation
            combined = torch.cat([a_proj, v_proj], dim=-1)
            # Use attention on each projection separately for richer context
            a_proj = self.temporal_attn(a_proj)
            v_proj = self.temporal_attn(v_proj)
            combined = torch.cat([a_proj, v_proj], dim=-1)
        else:
            combined = torch.cat([a_proj, v_proj], dim=-1)

        scores = self.discrepancy_head(combined).squeeze(-1)  # (T,)
        return a_proj, v_proj, scores


class CrossModalFusion:
    """Aligns and compares audio-articulatory and visual-lip features."""

    def __init__(
        self,
        audio_dim: int = 768,
        visual_dim: int = VISUAL_FEATURE_DIM,
        hidden_dim: int = 256,
        discrepancy_threshold: float = 0.65,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = discrepancy_threshold

        self.projector = AudioVisualProjector(
            audio_dim, visual_dim, hidden_dim
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
            logger.info("Loaded fusion checkpoint from %s", checkpoint_path)
        else:
            logger.warning("No fusion checkpoint loaded — using random initialization")

        self.projector.eval()

    def _load_checkpoint(self, path: str):
        """Load model weights from a checkpoint file."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.projector.load_state_dict(state_dict)

    def save_checkpoint(self, path: str):
        """Save current model weights to a checkpoint file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.projector.state_dict(), path)

    def align_temporal(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        audio_fps: float = 50.0,
        video_fps: float = 25.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align audio and visual feature sequences to a common temporal resolution.

        Uses linear interpolation to match sequences to the shorter duration.
        """
        n_audio = len(audio_features)
        n_visual = len(visual_features)

        if n_audio == 0 or n_visual == 0:
            logger.warning("Empty features: audio=%d, visual=%d", n_audio, n_visual)
            dim_a = audio_features.shape[1] if n_audio > 0 else 768
            dim_v = visual_features.shape[1] if n_visual > 0 else VISUAL_FEATURE_DIM
            return np.zeros((1, dim_a)), np.zeros((1, dim_v))

        audio_duration = n_audio / audio_fps
        visual_duration = n_visual / video_fps
        common_duration = min(audio_duration, visual_duration)
        common_length = max(1, int(common_duration * video_fps))

        audio_indices = np.linspace(0, n_audio - 1, common_length).astype(int)
        visual_indices = np.linspace(0, n_visual - 1, common_length).astype(int)

        return audio_features[audio_indices], visual_features[visual_indices]

    @torch.no_grad()
    def compute_discrepancy(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-frame discrepancy scores between audio and visual features.

        Returns:
            Tuple of (discrepancy_scores (T,), cosine_sim (T,))
        """
        audio_t = torch.from_numpy(audio_features).float().to(self.device)
        visual_t = torch.from_numpy(visual_features).float().to(self.device)

        a_proj, v_proj, disc_scores = self.projector(audio_t, visual_t)

        # Also compute cosine similarity as an auxiliary signal
        cos_sim = nn.functional.cosine_similarity(a_proj, v_proj, dim=-1)
        cos_discrepancy = (1.0 - cos_sim) / 2.0

        # Combine learned discrepancy head with cosine-based signal
        combined = 0.7 * disc_scores + 0.3 * cos_discrepancy

        return combined.cpu().numpy(), cos_sim.cpu().numpy()

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

        scores, cos_sim = self.compute_discrepancy(aligned_audio, aligned_visual)
        flagged = [int(i) for i, s in enumerate(scores) if s > self.threshold]
        overall = float(np.mean(scores))

        # Generate per-frame attention-based heatmap (1D temporal heatmap)
        # Full spatial heatmap requires a deeper model with face-region attention
        temporal_heatmap = scores.copy()

        return FusionResult(
            discrepancy_scores=scores,
            heatmap=temporal_heatmap,
            flagged_frames=flagged,
            overall_score=overall,
            metadata={
                "threshold": self.threshold,
                "num_frames_analyzed": len(scores),
                "num_flagged_frames": len(flagged),
                "flagged_ratio": len(flagged) / len(scores) if len(scores) > 0 else 0.0,
                "mean_cosine_similarity": float(np.mean(cos_sim)),
                "audio_fps": audio_fps,
                "video_fps": video_fps,
            },
        )

    def train_step(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single training step for fine-tuning the fusion model.

        Args:
            audio_features: (B, T, D_a) audio feature batch.
            visual_features: (B, T, D_v) visual feature batch.
            labels: (B, T) binary labels (0 = authentic, 1 = fake).
            optimizer: Optimizer instance.

        Returns:
            Batch loss value.
        """
        self.projector.train()
        optimizer.zero_grad()

        loss_total = 0.0
        for i in range(audio_features.size(0)):
            _, _, scores = self.projector(
                audio_features[i].to(self.device),
                visual_features[i].to(self.device),
            )
            loss = nn.functional.binary_cross_entropy(scores, labels[i].to(self.device))
            loss_total += loss

        loss_total = loss_total / audio_features.size(0)
        loss_total.backward()
        optimizer.step()

        self.projector.eval()
        return float(loss_total.item())
