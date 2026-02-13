from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

from config import CONFIG
from utils.file_utils import copy_files, ensure_dir


logger = logging.getLogger(__name__)


@dataclass
class DatasetBuildResult:
    """Summary of the built dataset."""

    images_train: int
    images_val: int
    classes: Dict[int, str]


class DatasetBuilder:
    """
    Build YOLO-style dataset directory structure from labeled frames.

    Expects images and corresponding .txt label files to exist in a flat
    directory before splitting into train/val.
    """

    def __init__(
        self,
        base_dir: Path,
        train_split: float,
        class_map: Dict[str, int],
    ) -> None:
        self.base_dir = base_dir
        self.train_split = train_split
        self.class_map = class_map

        self.images_dir = self.base_dir / "images"
        self.labels_dir = self.base_dir / "labels"
        self.train_images_dir = self.images_dir / "train"
        self.val_images_dir = self.images_dir / "val"
        self.train_labels_dir = self.labels_dir / "train"
        self.val_labels_dir = self.labels_dir / "val"

    def build(
        self,
        source_frames_dir: Path,
        source_labels_dir: Path,
    ) -> DatasetBuildResult:
        """Create YOLO dataset layout and data.yaml configuration."""
        ensure_dir(self.base_dir)
        ensure_dir(self.images_dir)
        ensure_dir(self.labels_dir)

        # Collect frames that have labels.
        frame_paths: List[Path] = sorted(source_frames_dir.glob("*.jpg"))
        paired_frames: List[Tuple[Path, Path]] = []
        for img_path in frame_paths:
            label_path = source_labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                # Skip unlabeled frames to keep dataset consistent.
                continue
            paired_frames.append((img_path, label_path))

        if not paired_frames:
            raise RuntimeError("No labeled frames found to build dataset.")

        random.shuffle(paired_frames)
        train_count = int(len(paired_frames) * self.train_split)
        train_pairs = paired_frames[:train_count]
        val_pairs = paired_frames[train_count:]

        train_imgs = [p[0] for p in train_pairs]
        train_labels = [p[1] for p in train_pairs]
        val_imgs = [p[0] for p in val_pairs]
        val_labels = [p[1] for p in val_pairs]

        copy_files(train_imgs, self.train_images_dir)
        copy_files(val_imgs, self.val_images_dir)
        copy_files(train_labels, self.train_labels_dir)
        copy_files(val_labels, self.val_labels_dir)

        # Build data.yaml
        data_yaml = {
            "yamlpath": str(self.base_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {idx: name for name, idx in self.class_map.items()},
        }
        yaml_path = self.base_dir / "data.yaml"
        yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=True), encoding="utf-8")

        logger.info(
            "Dataset built: %d train images, %d val images, %d classes",
            len(train_imgs),
            len(val_imgs),
            len(self.class_map),
        )

        # Ensure no overlap.
        assert not (set(train_imgs) & set(val_imgs))

        return DatasetBuildResult(
            images_train=len(train_imgs),
            images_val=len(val_imgs),
            classes={idx: name for name, idx in self.class_map.items()},
        )

