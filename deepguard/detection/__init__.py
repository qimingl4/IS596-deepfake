"""Detection backend — physical-based multi-modal pipeline."""

from deepguard.detection.visual_encoder import VisualLipEncoder
from deepguard.detection.audio_encoder import AudioArticulatoryEncoder
from deepguard.detection.fusion import CrossModalFusion

__all__ = ["VisualLipEncoder", "AudioArticulatoryEncoder", "CrossModalFusion"]
