from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from config import CONFIG, Config
from modules.auto_labeler import AutoLabelCheckpoint, AutoLabeler, AutoLabelStats
from modules.checkpoint_manager import CheckpointManager
from modules.dataset_builder import DatasetBuilder
from modules.frame_extractor import FrameExtractionConfig, FrameExtractor
from modules.video_downloader import VideoDownloader
from utils.file_utils import ensure_dir


def setup_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLO dataset from YouTube gameplay using Florence-2.",
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--game", required=True, help="Game name (used in output path)")
    parser.add_argument(
        "--frames",
        type=int,
        default=CONFIG.max_frames,
        help="Maximum number of frames to process (default: config.max_frames)",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base"],
        default=CONFIG.model_choice,
        help="Florence-2 model size (tiny or base).",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=CONFIG.frame_sampling_rate,
        help="Extract every Nth frame (sampling rate).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CONFIG.output_dir),
        help="Base output directory for datasets and checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: process only 10 frames.",
    )
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> Config:
    """Return a Config instance overridden by CLI arguments."""
    cfg = Config(
        frame_sampling_rate=args.frame_rate,
        max_frames=args.frames,
        train_split=CONFIG.train_split,
        confidence_threshold=CONFIG.confidence_threshold,
        model_choice=args.model,
        checkpoint_interval=CONFIG.checkpoint_interval,
        output_dir=Path(args.output_dir),
        temp_dir=CONFIG.temp_dir,
        device=CONFIG.device,
        torch_threads=CONFIG.torch_threads,
    )

    if args.quick_test:
        cfg.max_frames = min(cfg.max_frames, 10)

    return cfg


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_runtime_config(args)

    logger = logging.getLogger("main")
    logger.info("Starting YOLO dataset generation (CPU-only). This will be slow.")

    # Prepare directories
    ensure_dir(cfg.output_dir)
    game_dir = cfg.output_dir / args.game
    frames_dir = game_dir / "frames"
    labels_raw_dir = game_dir / "labels_raw"
    ensure_dir(frames_dir)
    ensure_dir(labels_raw_dir)

    # Checkpoint setup
    checkpoint_name = f"{args.game}"
    ckpt_manager = CheckpointManager(checkpoint_name, output_dir=cfg.output_dir)
    auto_ckpt = None
    if args.resume:
        loaded = ckpt_manager.load(dict)
        if loaded is not None:
            stats_data = loaded.get("stats", {})
            if isinstance(stats_data, dict):
                stats_obj = AutoLabelStats(
                    frames_processed=stats_data.get("frames_processed", 0),
                    objects_detected=stats_data.get("objects_detected", 0),
                    errors=stats_data.get("errors", 0),
                )
            else:
                stats_obj = AutoLabelStats()
            auto_ckpt = AutoLabelCheckpoint(
                processed_frames=loaded.get("processed_frames", []),
                class_map=loaded.get("class_map", {}),
                stats=stats_obj,
            )
            logger.info("Loaded checkpoint with %d processed frames.", len(auto_ckpt.processed_frames))

    # 1. Download video
    downloader = VideoDownloader(temp_dir=cfg.temp_dir)
    download_result = downloader.download(args.url)

    # 2. Extract frames
    fe_config = FrameExtractionConfig(
        sampling_rate=cfg.frame_sampling_rate,
        max_frames=cfg.max_frames,
        enable_scene_change=False,
    )
    extractor = FrameExtractor(frames_dir=frames_dir, config=fe_config)
    frame_paths: List[Path] = extractor.extract_frames(download_result.video_path)

    if not frame_paths:
        raise SystemExit("No frames extracted; aborting.")

    # 3. Auto-label frames sequentially
    if auto_ckpt is None:
        auto_ckpt = AutoLabelCheckpoint(
            processed_frames=[],
            class_map={},
            stats=AutoLabelStats(),
        )

    labeler = AutoLabeler(
        model_size=cfg.model_choice,  # type: ignore[arg-type]
        confidence_threshold=cfg.confidence_threshold,
        checkpoint=auto_ckpt,
    )

    def save_auto_checkpoint(state: AutoLabelCheckpoint) -> None:
        """Helper to persist auto-labeler checkpoint safely."""
        ckpt_manager.save(
            {
                "processed_frames": state.processed_frames,
                "class_map": state.class_map,
                "stats": {
                    "frames_processed": state.stats.frames_processed,
                    "objects_detected": state.stats.objects_detected,
                    "errors": state.stats.errors,
                },
            }
        )

    try:
        auto_ckpt = labeler.run_on_frames(
            frame_paths,
            labels_dir=labels_raw_dir,
            on_checkpoint=save_auto_checkpoint,
            checkpoint_interval=cfg.checkpoint_interval,
        )
    finally:
        # Always save checkpoint, even on errors.
        save_auto_checkpoint(auto_ckpt)

    # 4. Build YOLO dataset structure
    dataset_builder = DatasetBuilder(
        base_dir=game_dir / "dataset",
        train_split=cfg.train_split,
        class_map=auto_ckpt.class_map,
    )
    result = dataset_builder.build(
        source_frames_dir=frames_dir,
        source_labels_dir=labels_raw_dir,
    )

    # Summary report
    logger.info("=== Processing summary ===")
    logger.info("Frames processed: %d", auto_ckpt.stats.frames_processed)
    logger.info("Objects detected: %d", auto_ckpt.stats.objects_detected)
    logger.info("Errors: %d", auto_ckpt.stats.errors)
    logger.info("Train images: %d", result.images_train)
    logger.info("Val images: %d", result.images_val)
    logger.info("Classes: %s", result.classes)
    logger.info("Dataset base directory: %s", (game_dir / "dataset").resolve())
    logger.info("You can now train with Ultralytics YOLO using data.yaml.")


if __name__ == "__main__":
    main()

