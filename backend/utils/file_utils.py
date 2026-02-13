from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, List


logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def remove_dir(path: Path) -> None:
    """Remove a directory tree if it exists."""
    if path.exists():
        shutil.rmtree(path)


def copy_files(files: Iterable[Path], dst_dir: Path) -> List[Path]:
    """
    Copy files into destination directory.

    Returns list of new paths.
    """
    ensure_dir(dst_dir)
    copied: List[Path] = []
    for src in files:
        try:
            target = dst_dir / src.name
            shutil.copy2(src, target)
            copied.append(target)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to copy %s -> %s: %s", src, dst_dir, exc)
    return copied

