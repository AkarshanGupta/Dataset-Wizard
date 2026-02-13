from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2


logger = logging.getLogger(__name__)


def load_image_bgr(path: Path):
    """Load an image with OpenCV in BGR format."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def resize_max_dim(
    image,
    max_width: int = 640,
    max_height: int = 480,
):
    """
    Resize image while preserving aspect ratio so that width <= max_width
    and height <= max_height.
    """
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def yolo_normalize_bbox(
    bbox_xyxy: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert absolute xyxy bbox to YOLO normalized (x_center, y_center, w, h).
    All values are in range [0, 1].
    """
    x_min, y_min, x_max, y_max = bbox_xyxy
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

