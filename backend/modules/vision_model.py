from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from config import CONFIG


logger = logging.getLogger(__name__)


FlorenceModelSize = Literal["tiny", "base"]


@dataclass
class DetectionResult:
    """Single detection result from Florence-2."""

    label: str
    score: float
    bbox_xyxy: Tuple[float, float, float, float]


class FlorenceModelManager:
    """
    Manage Florence-2 model & processor lifecycle.

    CPU-only, singleton-style access with local caching and warmup.
    """

    def __init__(self, model_size: FlorenceModelSize = "tiny") -> None:
        self.model_size = model_size
        self.device = CONFIG.device
        torch.set_num_threads(CONFIG.torch_threads)

        self.model_name = self._resolve_model_name(model_size)
        logger.info("Loading Florence-2 model: %s", self.model_name)

        # Get HF token from environment if available
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=hf_token,
        )

        self.model.to(self.device)
        self.model.eval()

        self._warmup_done = False

    @staticmethod
    def _resolve_model_name(model_size: FlorenceModelSize) -> str:
        if model_size == "tiny":
            # Florence-2-base is the smallest available model
            return "microsoft/Florence-2-base"
        if model_size == "base":
            return "microsoft/Florence-2-base"
        # Fallback to base
        return "microsoft/Florence-2-base"

    def warmup(self) -> None:
        """Run a lightweight warmup pass to populate caches."""
        if self._warmup_done:
            return
        try:
            # Create a tiny blank image; we do not care about output.
            img = Image.new("RGB", (64, 64), color=(0, 0, 0))
            _ = self.detect_objects(img, confidence_threshold=0.99)
            logger.info("Model warmup completed.")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Warmup failed: %s", exc)
        finally:
            self._warmup_done = True

    @torch.inference_mode()
    def detect_objects(
        self,
        image: Image.Image,
        confidence_threshold: float,
    ) -> Tuple[DetectionResult, ...]:
        """
        Run Florence-2 object detection on a single PIL image.

        Uses <OD> task prompt and returns detections as a tuple.
        """
        prompt = "<OD>"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
        )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        # post_process_generation is provided via trust_remote_code=True
        processed = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=image.size,
        )

        detections = []
        # Expected Florence-2 OD output structure:
        # { "bboxes": [[x1,y1,x2,y2], ...], "labels": [...], "scores": [...] }
        bboxes = processed.get("bboxes", [])
        labels = processed.get("labels", [])
        scores = processed.get("scores", [])

        for box, label, score in zip(bboxes, labels, scores):
            if score is not None and score < confidence_threshold:
                continue
            try:
                x1, y1, x2, y2 = map(float, box)
            except Exception:
                continue
            detections.append(
                DetectionResult(
                    label=str(label),
                    score=float(score) if score is not None else 0.0,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )

        return tuple(detections)


@lru_cache(maxsize=2)
def get_florence_manager(model_size: FlorenceModelSize) -> FlorenceModelManager:
    """Return a cached FlorenceModelManager instance."""
    manager = FlorenceModelManager(model_size=model_size)
    manager.warmup()
    return manager

