"""Video processing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Basic metadata about a video file."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def get_video_info(video_path: str) -> VideoInfo:
    """Extract basic metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    info = VideoInfo(
        path=video_path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        duration_sec=cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    )
    cap.release()
    return info


def extract_frames(video_path: str, max_frames: int | None = None) -> list[np.ndarray]:
    """Extract frames from a video file.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract (None = all).

    Returns:
        List of BGR frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames
