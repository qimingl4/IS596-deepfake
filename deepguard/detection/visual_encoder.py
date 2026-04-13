"""Visual-Lip Encoder: frame-by-frame tracking of labial movements.

Uses MediaPipe FaceLandmarker (Tasks API) to capture subtle lip shape changes
for bilabial sounds ('b', 'p', 'm') that AI synthesis routinely distorts.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe lip landmark indices (inner + outer lip contour) — 478-point model
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                     308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                     291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
ALL_LIP_INDICES = sorted(set(OUTER_LIP_INDICES + INNER_LIP_INDICES))

# Key bilabial landmarks for phoneme-specific analysis
UPPER_LIP_MID = 13
LOWER_LIP_MID = 14
LEFT_CORNER = 61
RIGHT_CORNER = 291

# Face oval for bounding box
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Fixed feature vector dimension for downstream fusion
VISUAL_FEATURE_DIM = 256

# Default model download location
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_DEFAULT_MODEL_DIR = os.path.expanduser("~/.deepguard/models")
_DEFAULT_MODEL_PATH = os.path.join(_DEFAULT_MODEL_DIR, "face_landmarker.task")


def _ensure_model(model_path: str | None = None) -> str:
    """Ensure the FaceLandmarker model file exists, downloading if needed."""
    path = model_path or _DEFAULT_MODEL_PATH
    if os.path.exists(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("Downloading FaceLandmarker model to %s ...", path)
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
    except Exception:
        # Fallback: try curl
        import subprocess
        subprocess.run(
            ["curl", "-L", "-o", path, _MODEL_URL],
            check=True, capture_output=True,
        )
    logger.info("Model downloaded successfully")
    return path


@dataclass
class LipFeatures:
    """Extracted lip features for a single frame."""

    landmarks: np.ndarray                       # (N, 3) lip landmark coordinates
    face_bbox: tuple[int, int, int, int]        # (x, y, w, h) face bounding box
    descriptors: dict                           # Shape descriptors
    frame_index: int
    confidence: float


class VisualLipEncoder:
    """Extracts lip movement features from video frames using MediaPipe FaceLandmarker."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_path: str | None = None,
        temporal_smoothing: float = 0.0,
    ):
        model_path = _ensure_model(model_path)
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.temporal_smoothing = temporal_smoothing
        self._prev_landmarks: np.ndarray | None = None

    def extract_lip_landmarks(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Extract lip and face landmarks from a single BGR frame.

        Returns:
            Tuple of (lip_landmarks (N, 3), all_face_landmarks (478, 3), confidence)
            or None if no face detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        h, w, _ = frame.shape

        all_landmarks = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in face]
        )

        lip_landmarks = all_landmarks[ALL_LIP_INDICES]

        # Estimate confidence from landmark presence
        presence_scores = [face[i].presence for i in ALL_LIP_INDICES
                           if hasattr(face[i], "presence") and face[i].presence is not None]
        confidence = float(np.mean(presence_scores)) if presence_scores else 0.8

        # Apply temporal smoothing (exponential moving average)
        if self.temporal_smoothing > 0 and self._prev_landmarks is not None:
            alpha = self.temporal_smoothing
            lip_landmarks = alpha * self._prev_landmarks + (1 - alpha) * lip_landmarks
        self._prev_landmarks = lip_landmarks.copy()

        return lip_landmarks, all_landmarks, confidence

    def compute_face_bbox(
        self, all_landmarks: np.ndarray, frame_shape: tuple
    ) -> tuple[int, int, int, int]:
        """Compute face bounding box from face oval landmarks."""
        h, w = frame_shape[:2]
        face_pts = all_landmarks[FACE_OVAL_INDICES]
        x_min = max(0, int(face_pts[:, 0].min()) - 10)
        y_min = max(0, int(face_pts[:, 1].min()) - 10)
        x_max = min(w, int(face_pts[:, 0].max()) + 10)
        y_max = min(h, int(face_pts[:, 1].max()) + 10)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def compute_shape_descriptors(
        self, landmarks: np.ndarray, all_landmarks: np.ndarray
    ) -> dict:
        """Compute lip shape descriptors for deepfake detection."""
        y_coords = landmarks[:, 1]
        x_coords = landmarks[:, 0]

        height = y_coords.max() - y_coords.min()
        width = x_coords.max() - x_coords.min()
        aspect_ratio = float(height / width) if width > 0 else 0.0

        upper_mid = all_landmarks[UPPER_LIP_MID]
        lower_mid = all_landmarks[LOWER_LIP_MID]
        openness = float(np.linalg.norm(upper_mid[:2] - lower_mid[:2]))

        center_x = (x_coords.max() + x_coords.min()) / 2
        left_mask = landmarks[:, 0] < center_x
        right_mask = ~left_mask
        left_vals = landmarks[left_mask, 1]
        right_vals = landmarks[right_mask, 1]
        left_h = float(left_vals.max() - left_vals.min()) if left_mask.any() else 0
        right_h = float(right_vals.max() - right_vals.min()) if right_mask.any() else 0
        asymmetry = float(abs(left_h - right_h))

        pts_2d = landmarks[:, :2] - landmarks[:, :2].mean(axis=0)
        cov = np.cov(pts_2d.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eccentricity = float(1.0 - eigvals[1] / eigvals[0]) if eigvals[0] > 0 else 0.0

        left_corner = all_landmarks[LEFT_CORNER][:2]
        right_corner = all_landmarks[RIGHT_CORNER][:2]
        mid = (upper_mid[:2] + lower_mid[:2]) / 2
        vec_l = left_corner - mid
        vec_r = right_corner - mid
        cos_a = np.dot(vec_l, vec_r) / (np.linalg.norm(vec_l) * np.linalg.norm(vec_r) + 1e-8)
        corner_angle = float(np.arccos(np.clip(cos_a, -1, 1)))

        face_x = all_landmarks[FACE_OVAL_INDICES][:, 0]
        face_width = float(face_x.max() - face_x.min())
        bilabial_closure = float(openness / face_width) if face_width > 0 else 0.0

        return {
            "aspect_ratio": aspect_ratio,
            "openness": openness,
            "asymmetry": asymmetry,
            "eccentricity": eccentricity,
            "corner_angle": corner_angle,
            "bilabial_closure": bilabial_closure,
        }

    def landmarks_to_feature_vector(
        self, landmarks: np.ndarray, descriptors: dict
    ) -> np.ndarray:
        """Convert lip landmarks + descriptors into a fixed-size feature vector."""
        centered = landmarks[:, :2] - landmarks[:, :2].mean(axis=0)
        scale = np.linalg.norm(centered, axis=1).max()
        if scale > 0:
            centered = centered / scale

        flat = centered.flatten()
        desc_vec = np.array([
            descriptors["aspect_ratio"],
            descriptors["openness"],
            descriptors["asymmetry"],
            descriptors["eccentricity"],
            descriptors["corner_angle"],
            descriptors["bilabial_closure"],
        ])
        combined = np.concatenate([flat, desc_vec])

        if len(combined) >= VISUAL_FEATURE_DIM:
            return combined[:VISUAL_FEATURE_DIM]
        return np.pad(combined, (0, VISUAL_FEATURE_DIM - len(combined)))

    def process_video(self, video_path: str) -> list[LipFeatures]:
        """Process a video file and extract per-frame lip features."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            return []

        features: list[LipFeatures] = []
        frame_idx = 0
        self._prev_landmarks = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.extract_lip_landmarks(frame)
            if result is not None:
                lip_landmarks, all_landmarks, confidence = result
                face_bbox = self.compute_face_bbox(all_landmarks, frame.shape)
                descriptors = self.compute_shape_descriptors(lip_landmarks, all_landmarks)
                features.append(LipFeatures(
                    landmarks=lip_landmarks,
                    face_bbox=face_bbox,
                    descriptors=descriptors,
                    frame_index=frame_idx,
                    confidence=confidence,
                ))

            frame_idx += 1

        cap.release()
        logger.info("Processed %d frames, detected face in %d", frame_idx, len(features))
        return features

    def close(self):
        self.landmarker.close()
        self._prev_landmarks = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
