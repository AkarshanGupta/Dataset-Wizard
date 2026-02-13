from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from yt_dlp import YoutubeDL

from config import CONFIG
from utils.file_utils import ensure_dir


logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a video download operation."""

    video_path: Path
    title: str
    duration: Optional[float]


class VideoDownloader:
    """Download YouTube videos using yt-dlp with basic validation and error handling."""

    def __init__(self, temp_dir: Optional[Path] = None) -> None:
        self.temp_dir = temp_dir or CONFIG.temp_dir
        ensure_dir(self.temp_dir)

    def download(self, url: str) -> DownloadResult:
        """
        Download a YouTube video and return path to the video file.

        Raises RuntimeError on failure.
        """
        if "youtube.com" not in url and "youtu.be" not in url:
            raise ValueError(f"Invalid YouTube URL: {url}")

        logger.info("Starting download: %s", url)

        output_template = str(self.temp_dir / "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "mp4/bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "noprogress": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as exc:  # pragma: no cover - network failures
                logger.exception("Failed to download video: %s", exc)
                raise RuntimeError(f"Failed to download video: {exc}") from exc

        video_id = info.get("id")
        ext = info.get("ext", "mp4")
        video_path = self.temp_dir / f"{video_id}.{ext}"

        if not video_path.exists():
            raise RuntimeError(f"Downloaded video not found at {video_path}")

        result = DownloadResult(
            video_path=video_path,
            title=info.get("title", video_id),
            duration=info.get("duration"),
        )
        logger.info("Downloaded video to %s", video_path)
        return result

