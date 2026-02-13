from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

from config import CONFIG
from modules.yolo_detector import YoloDetectionResult, get_yolo_detector
from utils.image_utils import resize_max_dim, yolo_normalize_bbox
from utils.yolo_utils import YoloObject, validate_normalized_box, write_yolo_label_file


logger = logging.getLogger(__name__)


@dataclass
class AutoLabelStats:
    """Statistics collected during auto-labeling."""

    frames_processed: int = 0
    objects_detected: int = 0
    errors: int = 0


@dataclass
class AutoLabelCheckpoint:
    """Serializable state for checkpointing."""

    processed_frames: List[str] = field(default_factory=list)
    class_map: Dict[str, int] = field(default_factory=dict)
    stats: AutoLabelStats = field(default_factory=AutoLabelStats)


class AutoLabeler:
    """
    Run YOLO OD on frames sequentially and emit YOLO labels.

    One frame at a time, with ETA reporting.
    """

    def __init__(
        self,
        model_size: str,
        confidence_threshold: float,
        checkpoint: AutoLabelCheckpoint | None = None,
        class_filter: List[str] | None = None,
    ) -> None:
        self.detector = get_yolo_detector(model_size, class_filter)
        self.confidence_threshold = confidence_threshold
        self.checkpoint = checkpoint or AutoLabelCheckpoint()

    @property
    def class_map(self) -> Dict[str, int]:
        """Mapping from class name to YOLO class id."""
        return self.checkpoint.class_map

    @property
    def stats(self) -> AutoLabelStats:
        return self.checkpoint.stats

    def _get_or_create_class_id(self, label: str) -> int:
        if label not in self.class_map:
            self.class_map[label] = len(self.class_map)
        return self.class_map[label]

    def process_frame(
        self,
        frame_path: Path,
        labels_dir: Path,
    ) -> None:
        """
        Process a single frame: run OD and write YOLO label file.

        Skips frames that have already been processed (based on checkpoint).
        """
        if str(frame_path) in self.checkpoint.processed_frames:
            return

        try:
            pil_img = Image.open(frame_path).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to open image %s: %s", frame_path, exc)
            self.stats.errors += 1
            self.checkpoint.processed_frames.append(str(frame_path))
            return

        original_width, original_height = pil_img.size

        # Send the original image to YOLO
        detections: Tuple[YoloDetectionResult, ...] = ()
        try:
            detections = self.detector.detect_objects(
                pil_img,
                confidence_threshold=self.confidence_threshold,
            )
        except Exception as exc:
            logger.exception("Model inference failed for %s: %s", frame_path, exc)
            self.stats.errors += 1
            self.checkpoint.processed_frames.append(str(frame_path))
            return

        yolo_objects: List[YoloObject] = []
        for det in detections:
            class_id = self._get_or_create_class_id(det.label)
            x_c, y_c, w, h = yolo_normalize_bbox(
                det.bbox_xyxy,
                img_width=original_width,
                img_height=original_height,
            )
            if not validate_normalized_box(x_c, y_c, w, h):
                continue
            yolo_objects.append(
                YoloObject(
                    class_id=class_id,
                    x_center=x_c,
                    y_center=y_c,
                    width=w,
                    height=h,
                )
            )

        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / (frame_path.stem + ".txt")
        write_yolo_label_file(label_path, yolo_objects)

        self.stats.frames_processed += 1
        self.stats.objects_detected += len(yolo_objects)
        self.checkpoint.processed_frames.append(str(frame_path))

    def run_on_frames(
        self,
        frame_paths: List[Path],
        labels_dir: Path,
        on_checkpoint: Callable[[AutoLabelCheckpoint], None] | None = None,
        checkpoint_interval: int = CONFIG.checkpoint_interval,
    ) -> AutoLabelCheckpoint:
        """
        Process a list of frames sequentially with progress bar and ETA.

        Optionally invokes on_checkpoint every `checkpoint_interval` frames.
        """
        labels_dir.mkdir(parents=True, exist_ok=True)

        remaining_frames = [
            p for p in frame_paths if str(p) not in self.checkpoint.processed_frames
        ]

        if not remaining_frames:
            logger.info("No new frames to process.")
            return self.checkpoint

        start_time = time.time()
        processed_since_last_ckpt = 0
        with tqdm(
            total=len(remaining_frames),
            desc="Auto-labeling frames",
            unit="frame",
        ) as pbar:
            for idx, frame_path in enumerate(remaining_frames, start=1):
                frame_start = time.time()
                self.process_frame(frame_path, labels_dir=labels_dir)
                pbar.update(1)

                elapsed = time.time() - start_time
                fps = idx / elapsed if elapsed > 0 else 0.0
                remaining = len(remaining_frames) - idx
                eta_seconds = remaining / fps if fps > 0 else 0.0
                pbar.set_postfix(
                    {
                        "fps": f"{fps:.3f}",
                        "eta_s": f"{eta_seconds:.0f}",
                    }
                )

                # Simple guard against runaway per-frame times
                per_frame = time.time() - frame_start
                if per_frame > 60 * 5:  # 5 minutes per frame
                    logger.warning(
                        "Single-frame processing took %.1fs for %s",
                        per_frame,
                        frame_path,
                    )

                processed_since_last_ckpt += 1
                if (
                    on_checkpoint is not None
                    and checkpoint_interval > 0
                    and processed_since_last_ckpt >= checkpoint_interval
                ):
                    on_checkpoint(self.checkpoint)
                    processed_since_last_ckpt = 0

        return self.checkpoint
