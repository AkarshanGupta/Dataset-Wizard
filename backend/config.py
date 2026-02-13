"""
Global configuration for the YOLO dataset generator.

CPU-optimized defaults – tweak as needed or override via CLI arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Configuration values for the pipeline (CPU-optimized defaults)."""

    # Frame extraction
    frame_sampling_rate: int = 60  # extract every Nth frame
    max_frames: int = 200  # hard upper bound on processed frames

    # Dataset split
    train_split: float = 0.8

    # Detection
    confidence_threshold: float = 0.05  # Low threshold needed for stylized game graphics
    model_choice: str = "n"  # YOLO model size: n (nano), s (small), m (medium), l (large), x (xlarge)
    class_filter: List[str] | None = None  # e.g., ["person", "train", "car"] or None for all classes

    # Checkpointing
    checkpoint_interval: int = 50

    # System / paths
    output_dir: Path = Path("output")
    temp_dir: Path = Path("tmp")

    # Device (YOLO can use GPU if available)
    device: str = "cpu"


CONFIG = Config()

