"""
FastAPI backend for YOLO Dataset Generator
"""
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Optional
import asyncio
import uuid
import os
import zipfile
from pathlib import Path
import logging
from datetime import datetime
from enum import Enum

# Import existing modules
from config import Config
from modules.auto_labeler import AutoLabelCheckpoint, AutoLabeler, AutoLabelStats
from modules.dataset_builder import DatasetBuilder
from modules.frame_extractor import FrameExtractionConfig, FrameExtractor
from modules.video_downloader import VideoDownloader
from utils.file_utils import ensure_dir

app = FastAPI(title="YOLO Dataset Generator API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job status enum
class JobStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    LABELING = "labeling"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"

# Job data model
class ProcessingJob:
    def __init__(self, job_id: str, url: str, game_name: str, **kwargs):
        self.id = job_id
        self.url = url
        self.game_name = game_name
        self.max_frames = kwargs.get('max_frames', 200)
        self.model_choice = kwargs.get('model_choice', 'tiny')
        self.frame_rate = kwargs.get('frame_rate', 60)
        self.quick_test = kwargs.get('quick_test', False)
        self.status = JobStatus.PENDING
        self.progress = 0
        self.message = ""
        self.created_at = datetime.now()
        self.updated_at = None
        self.result_path = None
        self.error = None
        self.summary_data = None

# In-memory job storage (resets on server restart - fully local)
jobs: Dict[str, ProcessingJob] = {}

# Request/Response models
class JobRequest(BaseModel):
    url: str
    game_name: str
    max_frames: int = 200
    model_choice: str = "tiny"
    frame_rate: int = 60
    quick_test: bool = False

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

@app.post("/api/jobs")
async def create_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Create a new dataset generation job"""
    job_id = str(uuid.uuid4())
    
    job = ProcessingJob(
        job_id=job_id,
        url=request.url,
        game_name=request.game_name,
        max_frames=request.max_frames,
        model_choice=request.model_choice,
        frame_rate=request.frame_rate,
        quick_test=request.quick_test
    )
    
    jobs[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_dataset_job, job)
    
    return {
        "id": job_id,
        "status": _map_status(job.status.value),
        "message": "Job created successfully"
    }

def _map_status(status_value: str) -> str:
    """Map backend statuses to frontend-expected statuses."""
    if status_value in ("pending",):
        return "pending"
    elif status_value in ("downloading", "extracting", "labeling", "building"):
        return "running"
    elif status_value == "completed":
        return "completed"
    elif status_value == "failed":
        return "failed"
    return status_value

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and progress"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    result = {
        "id": job.id,
        "status": _map_status(job.status.value),
        "message": job.message,
        "progress": job.progress,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "game_name": job.game_name,
    }
    
    # Include summary when completed
    if job.status == JobStatus.COMPLETED and hasattr(job, 'summary_data') and job.summary_data:
        result["summary"] = job.summary_data
    
    return result

@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    return [
        {
            "job_id": job.id,
            "game_name": job.game_name,
            "status": job.status.value,
            "progress": job.progress,
            "created_at": job.created_at
        }
        for job in jobs.values()
    ]

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Clean up local dataset files
    if job.result_path and os.path.exists(job.result_path):
        import shutil
        shutil.rmtree(job.result_path, ignore_errors=True)
    
    del jobs[job_id]
    return {"message": "Job deleted successfully"}

@app.get("/api/jobs/{job_id}/download")
async def download_dataset(job_id: str):
    """Download the generated dataset as a zip file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(status_code=404, detail="Dataset files not found")
    
    # Create zip file
    zip_path = f"output/{job_id}_dataset.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        dataset_dir = Path(job.result_path) / "dataset"
        if dataset_dir.exists():
            for file_path in dataset_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(dataset_dir)
                    zipf.write(file_path, arcname)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=f"{job.game_name}_dataset.zip"
    )

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket for real-time progress updates"""
    await websocket.accept()
    
    try:
        while True:
            if job_id in jobs:
                job = jobs[job_id]
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "message": job.message
                })
                
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    break
            else:
                await websocket.send_json({"error": "Job not found"})
                break
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

async def process_dataset_job(job: ProcessingJob):
    """Background task to process dataset generation"""
    logger = logging.getLogger("dataset_processor")
    
    try:
        logger.info(f"Starting job {job.id} for game: {job.game_name}")
        
        # Build config from job parameters
        cfg = Config(
            frame_sampling_rate=job.frame_rate,
            max_frames=job.max_frames,
            train_split=Config().train_split,
            confidence_threshold=Config().confidence_threshold,
            model_choice=job.model_choice,
            checkpoint_interval=Config().checkpoint_interval,
            output_dir=Path("output"),
            temp_dir=Config().temp_dir,
            device=Config().device,
            class_filter=["person", "train", "car", "truck", "bus"],  # Good for games
        )
        
        if job.quick_test:
            cfg.max_frames = min(cfg.max_frames, 10)
        
        # Prepare directories - Store datasets locally in output folder
        ensure_dir(cfg.output_dir)
        game_dir = cfg.output_dir / job.game_name  # Direct game folder, no job prefix
        frames_dir = game_dir / "frames"
        labels_raw_dir = game_dir / "labels_raw"
        ensure_dir(frames_dir)
        ensure_dir(labels_raw_dir)
        
        # 1. Download video
        job.status = JobStatus.DOWNLOADING
        job.message = "Downloading video..."
        job.updated_at = datetime.now()
        
        downloader = VideoDownloader(temp_dir=cfg.temp_dir)
        download_result = downloader.download(job.url)
        job.progress = 20
        
        # 2. Extract frames
        job.status = JobStatus.EXTRACTING
        job.message = "Extracting frames..."
        job.updated_at = datetime.now()
        
        fe_config = FrameExtractionConfig(
            sampling_rate=cfg.frame_sampling_rate,
            max_frames=cfg.max_frames,
            enable_scene_change=False,
        )
        extractor = FrameExtractor(frames_dir=frames_dir, config=fe_config)
        frame_paths = extractor.extract_frames(download_result.video_path)
        
        if not frame_paths:
            raise Exception("No frames extracted from video")
        
        job.progress = 40
        
        # 3. Auto-label frames
        job.status = JobStatus.LABELING
        job.message = f"Auto-labeling {len(frame_paths)} frames..."
        job.updated_at = datetime.now()
        
        auto_ckpt = AutoLabelCheckpoint(
            processed_frames=[],
            class_map={},
            stats=AutoLabelStats(),
        )
        
        labeler = AutoLabeler(
            model_size=cfg.model_choice,
            confidence_threshold=cfg.confidence_threshold,
            checkpoint=auto_ckpt,
            class_filter=cfg.class_filter,
        )
        
        def progress_callback(current: int, total: int):
            """Update progress during labeling"""
            progress = 40 + int((current / total) * 40)  # 40-80%
            job.progress = progress
            job.message = f"Labeling frame {current}/{total}"
            job.updated_at = datetime.now()
        
        # Run labeling (we'll need to modify run_on_frames to accept progress callback)
        auto_ckpt = labeler.run_on_frames(
            frame_paths,
            labels_dir=labels_raw_dir,
            on_checkpoint=None,
            checkpoint_interval=cfg.checkpoint_interval
        )
        
        job.progress = 80
        
        # 4. Build YOLO dataset
        job.status = JobStatus.BUILDING
        job.message = "Building YOLO dataset structure..."
        job.updated_at = datetime.now()
        
        dataset_builder = DatasetBuilder(
            base_dir=game_dir / "dataset",
            train_split=cfg.train_split,
            class_map=auto_ckpt.class_map,
        )
        result = dataset_builder.build(
            source_frames_dir=frames_dir,
            source_labels_dir=labels_raw_dir,
        )
        
        # Job completed
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.message = f"Dataset ready! {result.images_train} train, {result.images_val} val images"
        job.result_path = str(game_dir)
        job.updated_at = datetime.now()
        
        # Build summary for frontend
        inverted_class_map = {str(v): k for k, v in auto_ckpt.class_map.items()}
        job.summary_data = {
            "frames_processed": auto_ckpt.stats.frames_processed,
            "objects_detected": auto_ckpt.stats.objects_detected,
            "errors": auto_ckpt.stats.errors,
            "train_images": result.images_train,
            "val_images": result.images_val,
            "classes": inverted_class_map,
            "dataset_path": str(game_dir / "dataset"),
        }
        
        logger.info(f"Job {job.id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job.id} failed: {str(e)}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.message = f"Job failed: {str(e)}"
        job.updated_at = datetime.now()

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)