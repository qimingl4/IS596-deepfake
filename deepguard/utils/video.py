"""Video processing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2


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
    """Extract basic metadata from a video file.

    Raises:
        IOError: If the file cannot be opened by OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = VideoInfo(
        path=video_path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps,
        frame_count=frame_count,
        duration_sec=frame_count / max(fps, 1),
    )
    cap.release()
    return info
