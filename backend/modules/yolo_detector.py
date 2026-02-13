"""
YOLOv8-based object detection for auto-labeling.
Uses pre-trained COCO models to detect objects in frames.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class YoloDetectionResult:
    """Single detection result from YOLO."""

    label: str
    score: float
    bbox_xyxy: Tuple[float, float, float, float]


class YoloDetector:
    """
    YOLO object detector for auto-labeling frames.
    
    Uses pre-trained YOLOv8 models (nano, small, medium, large, xlarge).
    """

    def __init__(self, model_size: str = "n", class_filter: list[str] | None = None) -> None:
        """
        Initialize YOLO detector.
        
        Args:
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 
                       'l' (large), 'x' (xlarge)
            class_filter: List of class names to keep (e.g., ['person', 'train'])
                         If None, keeps all detections
        """
        self.model_size = model_size
        self.model_name = f"yolov8{model_size}.pt"
        self.class_filter = class_filter
        
        logger.info(f"Loading YOLOv8 model: {self.model_name}")
        self.model = YOLO(self.model_name)
        logger.info("YOLOv8 model loaded successfully")
        
        if class_filter:
            logger.info(f"Class filter enabled: {class_filter}")

    def detect_objects(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.3,
    ) -> Tuple[YoloDetectionResult, ...]:
        """
        Run YOLO object detection on a PIL image.
        
        Args:
            image: PIL Image to detect objects in
            confidence_threshold: Minimum confidence score
            
        Returns:
            Tuple of detection results
        """
        # Run inference
        results = self.model(image, conf=confidence_threshold, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                
                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                class_name = result.names[cls]
                
                # Apply class filter if specified
                if self.class_filter and class_name not in self.class_filter:
                    continue
                
                detections.append(
                    YoloDetectionResult(
                        label=class_name,
                        score=conf,
                        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        
        return tuple(detections)


# Global instance cache
_detector_cache = {}


def get_yolo_detector(model_size: str = "n", class_filter: list[str] | None = None) -> YoloDetector:
    """Get or create a cached YOLO detector instance."""
    cache_key = f"{model_size}_{','.join(class_filter) if class_filter else 'all'}"
    if cache_key not in _detector_cache:
        _detector_cache[cache_key] = YoloDetector(model_size, class_filter)
    return _detector_cache[cache_key]
