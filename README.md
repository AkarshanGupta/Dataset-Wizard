# YOLO Dataset Generator

A full-stack application for automatically generating YOLO-format object detection datasets from YouTube videos using Microsoft Florence-2 AI model for auto-labeling.

## 🎯 Overview

This project automates the labor-intensive process of creating labeled datasets for object detection. Simply provide a YouTube video URL (e.g., gameplay videos), and the system will:

1. Download the video
2. Extract frames at configurable intervals
3. Automatically detect and label objects using Florence-2 AI
4. Generate a complete YOLO-format dataset with train/val splits

The project includes both a **CLI interface** for power users and a **modern web UI** for easy interaction.

## ✨ Features

### Backend (Python/FastAPI)
- 🎬 YouTube video download with `yt-dlp`
- 🖼️ Intelligent frame extraction with OpenCV
- 🤖 Auto-labeling using Microsoft Florence-2 (tiny/base models)
- 💾 Checkpoint & resume functionality
- 📊 YOLO dataset builder with train/val splits
- 🔌 RESTful API with FastAPI
- 📡 Real-time WebSocket updates
- ⚙️ CPU-optimized (works without GPU)
- 📦 Dataset export (ZIP download)

### Frontend (React/TypeScript)
- 🎨 Modern UI with Shadcn/ui components
- 📝 Intuitive job submission form
- 📈 Real-time job status monitoring
- 🔄 Live progress updates via WebSocket
- 📱 Responsive design with Tailwind CSS
- ⚡ Fast development with Vite

## 🏗️ Architecture

```
┌─────────────────┐      HTTP/WebSocket      ┌──────────────────┐
│                 │ ────────────────────────> │                  │
│  React Frontend │                           │  FastAPI Backend │
│   (TypeScript)  │ <──────────────────────── │     (Python)     │
└─────────────────┘                           └──────────────────┘
                                                       │
                                                       ├─> YouTube Download
                                                       ├─> Frame Extraction
                                                       ├─> Florence-2 AI
                                                       └─> YOLO Dataset
```

## 📋 Prerequisites

### Backend
- Python 3.8+
- pip
- 2GB+ RAM (4GB+ recommended for base model)

### Frontend
- Node.js 16+
- npm or yarn

## Demo 


https://github.com/user-attachments/assets/669f8689-9cae-42ea-aee5-5762ab974728



## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "New folder"
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install
```

## 💻 Usage

### Web Interface (Recommended)

1. **Start the Backend API:**
```bash
cd backend
python app.py
```
The API will be available at `http://localhost:8000`

2. **Start the Frontend:**
```bash
cd frontend
npm run dev
```
The UI will be available at `http://localhost:5173`

3. **Create a Dataset:**
   - Open the web interface
   - Fill in the form:
     - YouTube video URL
     - Game/dataset name
     - Number of frames to process
     - Model size (tiny/base)
     - Frame sampling rate
   - Click "Create Job"
   - Monitor progress in real-time
   - Download the generated dataset

### CLI Interface

For advanced users or automation:

```bash
cd backend
python main.py \
  --url "https://youtube.com/watch?v=VIDEO_ID" \
  --game "game_name" \
  --frames 200 \
  --model tiny \
  --frame-rate 60
```

#### CLI Options:
- `--url` - YouTube video URL (required)
- `--game` - Game/dataset name (required)
- `--frames` - Maximum frames to process (default: 200)
- `--model` - Florence-2 model: `tiny` or `base` (default: tiny)
- `--frame-rate` - Extract every Nth frame (default: 60)
- `--output-dir` - Output directory path
- `--resume` - Resume from last checkpoint
- `--quick-test` - Process only 10 frames for testing

### Quick Test
```bash
python main.py --url "VIDEO_URL" --game "test" --quick-test
```

## ⚙️ Configuration

### Backend Configuration
Edit `backend/config.py` to customize:

```python
frame_sampling_rate = 60          # Extract every Nth frame
max_frames = 200                  # Maximum frames to process
train_split = 0.8                 # Train/validation split ratio
confidence_threshold = 0.05       # Detection confidence threshold
checkpoint_interval = 50          # Save checkpoint every N frames
```

### Frontend Configuration
Edit `frontend/src/services/api.ts` to change API endpoint:

```typescript
const API_BASE_URL = 'http://localhost:8000';
```

## 📁 Project Structure

```
├── backend/
│   ├── app.py                    # FastAPI application
│   ├── main.py                   # CLI interface
│   ├── config.py                 # Configuration
│   ├── requirements.txt          # Python dependencies
│   ├── modules/
│   │   ├── auto_labeler.py      # Florence-2 integration
│   │   ├── dataset_builder.py   # YOLO dataset creation
│   │   ├── frame_extractor.py   # Frame extraction
│   │   ├── video_downloader.py  # YouTube download
│   │   └── ...
│   └── utils/                    # Utility functions
│
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── pages/               # Page components
│   │   ├── services/            # API service
│   │   └── types/               # TypeScript types
│   ├── package.json
│   └── vite.config.ts
│
└── README.md                     # This file
```

## 📤 Output Structure

Generated datasets follow YOLO format:

```
output/
└── <game_name>/
    ├── frames/                   # Extracted frames (PNG)
    ├── labels_raw/              # Raw label files
    ├── dataset/
    │   ├── images/
    │   │   ├── train/          # Training images
    │   │   └── val/            # Validation images
    │   ├── labels/
    │   │   ├── train/          # Training labels
    │   │   └── val/            # Validation labels
    │   └── data.yaml           # Dataset config
    └── checkpoints/             # Resume checkpoints
```

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern web framework
- **Florence-2** - Microsoft's vision AI model
- **OpenCV** - Video/image processing
- **yt-dlp** - YouTube downloader
- **Transformers** - Hugging Face model integration
- **Pillow** - Image manipulation

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Shadcn/ui** - Component library
- **Tanstack Query** - Data fetching
- **Lucide React** - Icons

## ⚡ Performance Notes

- **Florence-2-tiny**: ~10-30 seconds per frame on CPU
- **Florence-2-base**: ~30-60 seconds per frame on CPU
- For 200 frames with tiny model: expect 30-90 minutes total
- GPU acceleration supported if available (auto-detected)
- Checkpoint system allows resuming interrupted jobs

## 🐛 Troubleshooting

### Backend Issues
```bash
# If yt-dlp fails, update it
pip install --upgrade yt-dlp

# For memory issues, reduce batch size in config.py
# Or use fewer frames with --frames 50
```

### Frontend Issues
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check if backend is running
curl http://localhost:8000/health
```

## 📝 API Documentation

Once the backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints:
- `POST /jobs` - Create new job
- `GET /jobs/{job_id}` - Get job status
- `GET /jobs/{job_id}/download` - Download dataset
- `DELETE /jobs/{job_id}` - Delete job
- `WS /ws/{job_id}` - WebSocket for real-time updates

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- Microsoft Florence-2 for the vision AI model
- Ultralytics for YOLO format standards
- Shadcn for the beautiful UI components

---

**Note**: This tool is designed for creating datasets from publicly available videos for research and educational purposes. Please respect copyright and terms of service of video platforms.
