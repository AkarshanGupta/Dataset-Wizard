from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class YoloObject:
    """Single YOLO-format detection."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def write_yolo_label_file(path: Path, objects: Iterable[YoloObject]) -> None:
    """
    Write YOLO label file where each line is:
    class_id x_center y_center width height
    """
    lines: List[str] = []
    for obj in objects:
        vals = [
            obj.class_id,
            round(obj.x_center, 6),
            round(obj.y_center, 6),
            round(obj.width, 6),
            round(obj.height, 6),
        ]
        lines.append(" ".join(str(v) for v in vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_normalized_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> bool:
    """Return True if all values lie within [0, 1] and box has positive area."""
    vals: Tuple[float, float, float, float] = (x_center, y_center, width, height)
    if any(v < 0.0 or v > 1.0 for v in vals):
        return False
    if width <= 0.0 or height <= 0.0:
        return False
    return True

