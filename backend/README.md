## YOLO Dataset Generator (CPU-Only, Florence-2)

This backend generates a YOLO-format object detection dataset from YouTube gameplay videos using Microsoft Florence‑2 (tiny/base) for auto-labeling. It is designed for **CPU-only** execution and processes frames **sequentially** (no multiprocessing).

### Features

- **YouTube download** with `yt-dlp`
- **Frame extraction** with OpenCV (configurable sampling, optional scene-change filter)
- **Florence‑2 tiny/base** on CPU with caching and warmup
- **Auto-labeling** with `<OD>` prompt → YOLO labels
- **Checkpointing & resume** (JSON checkpoints)
- **YOLO dataset builder** (`images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`)

### Install

```bash
pip install -r requirements.txt
```

Run on CPU only (no GPU required). For best results, close other heavy apps.

### Usage

```bash
python main.py --url "https://youtube.com/watch?v=..." --game "subway_surfers" --frames 200 --model tiny
```

Optional flags:

- `--resume` – resume from last checkpoint
- `--frame-rate 90` – extract every 90th frame
- `--model base` – use Florence‑2-base (slower, more accurate)
- `--output-dir path/to/out` – base output directory
- `--quick-test` – process at most 10 frames (sanity check)

### Output Layout

- `output/<game>/frames/` – extracted frames
- `output/<game>/labels_raw/` – one `.txt` per frame (YOLO labels)
- `output/<game>/dataset/` – final YOLO dataset:
  - `images/train`, `images/val`
  - `labels/train`, `labels/val`
  - `data.yaml`
- `output/checkpoints/<game>.json` – checkpoint for resume

### CPU Performance Notes

- Expected **10–30 seconds per frame** on typical CPUs with Florence‑2‑tiny
- For 200 frames expect **30–90 minutes** total
- Use `--quick-test` first (10 frames), then run full jobs (possibly overnight)

### Training with Ultralytics YOLO

Point YOLO to the generated `data.yaml`, for example:

```bash
yolo detect train data=output/subway_surfers/dataset/data.yaml model=yolov8n.pt
```

