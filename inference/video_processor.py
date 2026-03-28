"""Video processing utilities for PyroSense AI.

Example:
    >>> from inference.video_processor import VideoProcessor
    >>> _ = VideoProcessor(target_fps=10)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from utils.logger import logger


@dataclass
class VideoStats:
    """Aggregated stats from video processing."""

    frames_processed: int
    elapsed_s: float
    avg_fps: float
    avg_inference_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "frames_processed": float(self.frames_processed),
            "elapsed_s": float(self.elapsed_s),
            "avg_fps": float(self.avg_fps),
            "avg_inference_ms": float(self.avg_inference_ms),
        }


class VideoProcessor:
    """Process a video with a per-frame callback and optional writer."""

    def __init__(self, target_fps: Optional[float] = None) -> None:
        """Create processor.

        Example:
            >>> from inference.video_processor import VideoProcessor
            >>> _ = VideoProcessor(target_fps=15)
        """

        self.target_fps = target_fps

    def process(
        self,
        video_path: str,
        *,
        on_frame: Callable[[np.ndarray], tuple[np.ndarray, float]],
        output_path: Optional[str] = None,
    ) -> VideoStats:
        """Process a video.

        The callback receives a BGR frame and returns (annotated_frame, inference_ms).
        If output_path is provided, annotated frames are written to disk.
        """

        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f"OpenCV is required for video processing: {e}") from e

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        writer = None
        if output_path:
            outp = Path(output_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(outp), fourcc, in_fps, (w, h))

        frames = 0
        inf_total = 0.0
        t0 = time.perf_counter()
        min_dt = (1.0 / float(self.target_fps)) if self.target_fps else 0.0
        last_emit = 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                now = time.perf_counter()
                if min_dt and (now - last_emit) < min_dt:
                    continue
                last_emit = now

                annotated, inf_ms = on_frame(frame)
                if writer is not None:
                    writer.write(annotated)
                frames += 1
                inf_total += float(inf_ms)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        elapsed = max(1e-6, time.perf_counter() - t0)
        avg_fps = float(frames / elapsed)
        avg_inf = float(inf_total / max(1, frames))
        logger.info(f"Processed video frames={frames} avg_fps={avg_fps:.2f} avg_inf_ms={avg_inf:.1f}")
        return VideoStats(frames_processed=frames, elapsed_s=elapsed, avg_fps=avg_fps, avg_inference_ms=avg_inf)

