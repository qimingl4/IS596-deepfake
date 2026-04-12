"""Visual-Lip Encoder: frame-by-frame tracking of labial movements.

Uses MediaPipe Face Mesh to capture subtle lip shape changes for bilabial
sounds ('b', 'p', 'm') that AI synthesis routinely distorts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe lip landmark indices (inner + outer lip contour)
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
                     14, 87, 178, 88, 95, 78]
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314,
                     17, 84, 181, 91, 146, 61]


@dataclass
class LipFeatures:
    """Extracted lip features for a single frame."""

    landmarks: np.ndarray       # (N, 3) lip landmark coordinates
    aspect_ratio: float         # lip height / width ratio
    frame_index: int
    confidence: float


class VisualLipEncoder:
    """Extracts lip movement features from video frames using MediaPipe Face Mesh."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def extract_lip_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract lip landmark coordinates from a single BGR frame.

        Returns:
            (N, 3) array of lip landmarks or None if no face detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        lip_indices = list(set(OUTER_LIP_INDICES + INNER_LIP_INDICES))
        landmarks = np.array(
            [[face.landmark[i].x * w, face.landmark[i].y * h, face.landmark[i].z * w]
             for i in lip_indices]
        )
        return landmarks

    def compute_lip_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Compute lip aspect ratio (height / width) from landmarks."""
        y_coords = landmarks[:, 1]
        x_coords = landmarks[:, 0]
        height = y_coords.max() - y_coords.min()
        width = x_coords.max() - x_coords.min()
        return float(height / width) if width > 0 else 0.0

    def process_video(self, video_path: str) -> list[LipFeatures]:
        """Process a video file and extract per-frame lip features.

        Returns:
            List of LipFeatures for each frame where a face was detected.
        """
        cap = cv2.VideoCapture(video_path)
        features: list[LipFeatures] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.extract_lip_landmarks(frame)
            if landmarks is not None:
                aspect_ratio = self.compute_lip_aspect_ratio(landmarks)
                features.append(LipFeatures(
                    landmarks=landmarks,
                    aspect_ratio=aspect_ratio,
                    frame_index=frame_idx,
                    confidence=1.0,
                ))
            frame_idx += 1

        cap.release()
        return features

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
