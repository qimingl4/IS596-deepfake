"""Video Overlay: renders real-time bounding boxes and heatmaps on video frames.

Highlights high-risk frames showing exactly where and when audio-visual
inconsistencies occur. Draws face bounding boxes, lip landmark contours,
and score indicators.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from deepguard.detection.fusion import FusionResult
from deepguard.detection.visual_encoder import LipFeatures

logger = logging.getLogger(__name__)


class VideoOverlay:
    """Renders detection results as visual overlays on video frames."""

    def __init__(
        self,
        heatmap_opacity: float = 0.4,
        bbox_color: tuple[int, int, int] = (0, 255, 0),
        flagged_bbox_color: tuple[int, int, int] = (0, 0, 255),
        highlight_top_k: int = 10,
    ):
        self.heatmap_opacity = heatmap_opacity
        self.bbox_color = bbox_color
        self.flagged_bbox_color = flagged_bbox_color
        self.highlight_top_k = highlight_top_k

    def _scale_factor(self, frame: np.ndarray) -> float:
        """Compute a scale factor based on frame height for adaptive text sizing."""
        return max(frame.shape[0] / 720, 0.5)

    def draw_score_bar(self, frame: np.ndarray, score: float) -> np.ndarray:
        """Draw a discrepancy score indicator bar on the frame."""
        h, w = frame.shape[:2]
        sf = self._scale_factor(frame)
        bar_h = int(24 * sf)
        bar_w = int(w * 0.35)
        margin = int(16 * sf)
        x_start = w - bar_w - margin
        y_start = margin

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 4, y_start - int(22 * sf)),
                       (x_start + bar_w + 4, y_start + bar_h + 4),
                       (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Bar background
        cv2.rectangle(frame, (x_start, y_start), (x_start + bar_w, y_start + bar_h),
                       (80, 80, 80), -1)

        # Filled portion with color gradient
        fill_w = int(bar_w * min(score, 1.0))
        if score < 0.3:
            color = (0, 200, 0)       # Green
        elif score < 0.6:
            color = (0, 180, 255)     # Orange
        else:
            color = (0, 0, 230)       # Red
        cv2.rectangle(frame, (x_start, y_start), (x_start + fill_w, y_start + bar_h),
                       color, -1)

        # Label
        label = f"Mismatch: {score:.1%}"
        font_scale = 0.5 * sf
        cv2.putText(frame, label, (x_start, y_start - int(6 * sf)),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, int(sf)))

        return frame

    def draw_verdict_badge(self, frame: np.ndarray, overall_score: float) -> np.ndarray:
        """Draw an overall verdict badge in the top-left corner."""
        sf = self._scale_factor(frame)
        if overall_score < 0.3:
            label, color = "AUTHENTIC", (0, 200, 0)
        elif overall_score < 0.6:
            label, color = "SUSPICIOUS", (0, 180, 255)
        else:
            label, color = "LIKELY FAKE", (0, 0, 230)

        margin = int(16 * sf)
        font_scale = 0.7 * sf
        thickness = max(1, int(2 * sf))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin - 4, margin - th - 8),
                       (margin + tw + 8, margin + 8), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, label, (margin, margin),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        return frame

    def draw_flagged_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Draw a pulsing 'FLAGGED' warning on frames exceeding the threshold."""
        sf = self._scale_factor(frame)
        h = frame.shape[0]
        font_scale = 0.6 * sf
        thickness = max(1, int(2 * sf))

        cv2.putText(frame, "! FLAGGED", (int(16 * sf), h - int(20 * sf)),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        # Red border for flagged frames
        border = max(2, int(3 * sf))
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                       (0, 0, 255), border)
        return frame

    def draw_face_bbox(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        is_flagged: bool = False,
    ) -> np.ndarray:
        """Draw face bounding box on the frame."""
        x, y, w, h = bbox
        color = self.flagged_bbox_color if is_flagged else self.bbox_color
        thickness = max(1, int(2 * self._scale_factor(frame)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame

    def draw_lip_landmarks(
        self, frame: np.ndarray, landmarks: np.ndarray, is_flagged: bool = False
    ) -> np.ndarray:
        """Draw lip landmark points and contour on the frame."""
        color = (0, 0, 255) if is_flagged else (0, 255, 255)  # Red or cyan
        pts = landmarks[:, :2].astype(np.int32)

        # Draw contour
        if len(pts) > 2:
            hull = cv2.convexHull(pts)
            cv2.drawContours(frame, [hull], 0, color, 1)

        # Draw landmark dots
        radius = max(1, int(2 * self._scale_factor(frame)))
        for pt in pts:
            cv2.circle(frame, tuple(pt), radius, color, -1)

        return frame

    def draw_timestamp(self, frame: np.ndarray, frame_idx: int, fps: float) -> np.ndarray:
        """Draw frame number and timestamp."""
        sf = self._scale_factor(frame)
        h, w = frame.shape[:2]
        seconds = frame_idx / fps if fps > 0 else 0
        label = f"Frame {frame_idx} | {seconds:.1f}s"
        font_scale = 0.4 * sf
        cv2.putText(frame, label, (w - int(180 * sf), h - int(10 * sf)),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
        return frame

    def render_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fusion_result: FusionResult,
        lip_features: LipFeatures | None = None,
        fps: float = 25.0,
    ) -> np.ndarray:
        """Render detection overlay on a single video frame."""
        output = frame.copy()
        scores = fusion_result.discrepancy_scores
        is_flagged = frame_idx in fusion_result.flagged_frames

        if frame_idx < len(scores):
            score = float(scores[frame_idx])
            output = self.draw_score_bar(output, score)

            if is_flagged:
                output = self.draw_flagged_indicator(output)

        # Draw face bounding box and lip landmarks if available
        if lip_features is not None:
            output = self.draw_face_bbox(output, lip_features.face_bbox, is_flagged)
            output = self.draw_lip_landmarks(output, lip_features.landmarks, is_flagged)

        # Verdict badge and timestamp
        output = self.draw_verdict_badge(output, fusion_result.overall_score)
        output = self.draw_timestamp(output, frame_idx, fps)

        return output

    def render_video(
        self,
        video_path: str,
        output_path: str,
        fusion_result: FusionResult,
        lip_features_list: list[LipFeatures] | None = None,
    ) -> str:
        """Render detection overlay on an entire video and save to output_path."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Build a quick lookup from frame_index -> LipFeatures
        lip_lookup: dict[int, LipFeatures] = {}
        if lip_features_list:
            lip_lookup = {lf.frame_index: lf for lf in lip_features_list}

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            lip_feat = lip_lookup.get(frame_idx)
            annotated = self.render_frame(
                frame, frame_idx, fusion_result, lip_feat, fps
            )
            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()
        logger.info("Annotated video saved to %s (%d frames)", output_path, frame_idx)
        return output_path
