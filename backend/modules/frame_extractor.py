from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2

from config import CONFIG
from utils.file_utils import ensure_dir


logger = logging.getLogger(__name__)


@dataclass
class FrameExtractionConfig:
    """Configuration specific to frame extraction."""

    sampling_rate: int = CONFIG.frame_sampling_rate
    max_frames: int = CONFIG.max_frames
    enable_scene_change: bool = False
    scene_change_threshold: float = 30.0  # higher = less sensitive


class FrameExtractor:
    """Extract frames from a video file at a configurable sampling rate."""

    def __init__(
        self,
        frames_dir: Path,
        config: Optional[FrameExtractionConfig] = None,
    ) -> None:
        self.frames_dir = frames_dir
        self.config = config or FrameExtractionConfig()
        ensure_dir(self.frames_dir)

    def extract_frames(self, video_path: Path) -> List[Path]:
        """
        Extract frames from video, sampling every Nth frame.

        Returns list of saved frame image paths.
        """
        logger.info("Extracting frames from %s", video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frame_paths: List[Path] = []
        frame_idx = 0
        saved_idx = 0
        prev_gray = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample every Nth frame
                if frame_idx % self.config.sampling_rate != 0:
                    frame_idx += 1
                    continue

                # Optional simple scene change detection
                if self.config.enable_scene_change:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = cv2.absdiff(gray, prev_gray)
                        mean_diff = diff.mean()
                        if mean_diff < self.config.scene_change_threshold:
                            frame_idx += 1
                            continue
                    prev_gray = gray

                frame_name = f"frame_{saved_idx:06d}.jpg"
                frame_path = self.frames_dir / frame_name
                success = cv2.imwrite(str(frame_path), frame)
                if not success:
                    logger.warning("Failed to write frame %s", frame_path)
                else:
                    frame_paths.append(frame_path)
                    saved_idx += 1

                frame_idx += 1
                if saved_idx >= self.config.max_frames:
                    logger.info(
                        "Reached max_frames limit (%d); stopping extraction",
                        self.config.max_frames,
                    )
                    break
        finally:
            cap.release()

        logger.info("Extracted %d frames", len(frame_paths))
        return frame_paths

