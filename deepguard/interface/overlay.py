"""Video Overlay: renders real-time bounding boxes and heatmaps on video frames.

Highlights high-risk frames showing exactly where and when audio-visual
inconsistencies occur.
"""

from __future__ import annotations

import cv2
import numpy as np

from deepguard.detection.fusion import FusionResult


class VideoOverlay:
    """Renders detection results as visual overlays on video frames."""

    def __init__(
        self,
        heatmap_opacity: float = 0.4,
        bbox_color: tuple[int, int, int] = (255, 0, 0),
        highlight_top_k: int = 10,
    ):
        self.heatmap_opacity = heatmap_opacity
        self.bbox_color = bbox_color
        self.highlight_top_k = highlight_top_k

    def draw_score_bar(self, frame: np.ndarray, score: float) -> np.ndarray:
        """Draw a discrepancy score indicator bar on the frame."""
        h, w = frame.shape[:2]
        bar_h, bar_w = 20, int(w * 0.4)
        x_start = w - bar_w - 20
        y_start = 20

        # Background
        cv2.rectangle(frame, (x_start, y_start), (x_start + bar_w, y_start + bar_h),
                       (50, 50, 50), -1)
        # Filled portion
        fill_w = int(bar_w * min(score, 1.0))
        color = (0, 255, 0) if score < 0.3 else (0, 165, 255) if score < 0.6 else (0, 0, 255)
        cv2.rectangle(frame, (x_start, y_start), (x_start + fill_w, y_start + bar_h),
                       color, -1)
        # Label
        label = f"Mismatch: {score:.1%}"
        cv2.putText(frame, label, (x_start, y_start - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def draw_flagged_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Draw a 'FLAGGED' warning on frames exceeding the threshold."""
        cv2.putText(frame, "FLAGGED", (20, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame

    def render_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fusion_result: FusionResult,
    ) -> np.ndarray:
        """Render detection overlay on a single video frame."""
        output = frame.copy()
        scores = fusion_result.discrepancy_scores

        if frame_idx < len(scores):
            score = scores[frame_idx]
            output = self.draw_score_bar(output, float(score))

            if frame_idx in fusion_result.flagged_frames:
                output = self.draw_flagged_indicator(output)

        return output

    def render_video(
        self,
        video_path: str,
        output_path: str,
        fusion_result: FusionResult,
    ) -> str:
        """Render detection overlay on an entire video and save to output_path."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated = self.render_frame(frame, frame_idx, fusion_result)
            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()
        return output_path
