from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from config import CONFIG


logger = logging.getLogger(__name__)

T = TypeVar("T")


class CheckpointManager:
    """
    Save and load checkpoint state to/from disk as JSON.

    The checkpoint file lives under output_dir/checkpoints/.
    """

    def __init__(self, name: str, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or CONFIG.output_dir
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.checkpoint_dir / f"{name}.json"

    def save(self, state: Any) -> None:
        """Serialize a dataclass or plain dict to disk."""
        if is_dataclass(state):
            payload = asdict(state)
        elif isinstance(state, dict):
            payload = state
        else:
            raise TypeError("Checkpoint state must be a dataclass or dict")

        try:
            tmp_path = self.path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.path)
            logger.info("Checkpoint saved: %s", self.path)
        except Exception as exc:  # pragma: no cover - IO errors
            logger.exception("Failed to save checkpoint: %s", exc)

    def load(self, cls: Optional[Type[T]] = None) -> Optional[T]:
        """
        Load checkpoint from disk.

        If cls is provided, it must be a dataclass; we reconstruct it.
        """
        if not self.path.exists():
            return None

        try:
            data: Dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - IO errors
            logger.exception("Failed to read checkpoint: %s", exc)
            return None

        if cls is None:
            return data  # type: ignore[return-value]

        try:
            return cls(**data)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - schema mismatch
            logger.warning("Checkpoint schema mismatch: %s", exc)
            return None

