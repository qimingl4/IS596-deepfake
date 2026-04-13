"""Detection backend — physical-based multi-modal pipeline."""

from deepguard.detection.visual_encoder import VisualLipEncoder, LipFeatures, VISUAL_FEATURE_DIM
from deepguard.detection.audio_encoder import AudioArticulatoryEncoder, AudioFeatures, PhonemeSegment
from deepguard.detection.fusion import CrossModalFusion, FusionResult

__all__ = [
    "VisualLipEncoder",
    "LipFeatures",
    "VISUAL_FEATURE_DIM",
    "AudioArticulatoryEncoder",
    "AudioFeatures",
    "PhonemeSegment",
    "CrossModalFusion",
    "FusionResult",
]
