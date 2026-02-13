"""
Gradio Web Interface for Testing YOLO Models
Upload images and see real-time object detection results.

This is a STANDALONE testing tool - separate from the backend API.
"""
import sys
from pathlib import Path
import gradio as gr
from PIL import Image
import numpy as np
from ultralytics import YOLO


class YOLOTester:
    """YOLO model tester with Gradio interface."""
    
    def __init__(self):
        self.model = None
        self.model_path = None
    
    def load_model(self, model_path: str):
        """Load a YOLO model from path."""
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            return f"✅ Model loaded: {Path(model_path).name}"
        except Exception as e:
            return f"❌ Failed to load model: {str(e)}"
    
    def predict(self, image: np.ndarray, confidence: float, iou: float):
        """
        Run YOLO prediction on an image.
        
        Args:
            image: Input image as numpy array
            confidence: Confidence threshold (0-1)
            iou: IoU threshold for NMS (0-1)
        
        Returns:
            Annotated image and detection details text
        """
        if self.model is None:
            return None, "❌ No model loaded! Please load a model first."
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=confidence,
                iou=iou,
                verbose=False
            )
            
            # Get annotated image
            annotated = results[0].plot()  # Draw boxes on image
            
            # Extract detection details
            boxes = results[0].boxes
            detections_text = self._format_detections(boxes, results[0].names)
            
            return annotated, detections_text
            
        except Exception as e:
            return None, f"❌ Prediction failed: {str(e)}"
    
    def _format_detections(self, boxes, class_names):
        """Format detection results as readable text."""
        if len(boxes) == 0:
            return "No objects detected."
        
        lines = [f"🎯 Detected {len(boxes)} objects:\n"]
        lines.append("=" * 50)
        
        for i, box in enumerate(boxes, 1):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            class_name = class_names.get(cls, f"class_{cls}")
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            
            lines.append(f"\n📦 Detection #{i}:")
            lines.append(f"   Class: {class_name}")
            lines.append(f"   Confidence: {conf:.2%}")
            lines.append(f"   Box: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
            lines.append(f"   Size: {int(width)}×{int(height)} pixels")
        
        return "\n".join(lines)


def create_interface():
    """Create and configure the Gradio interface."""
    
    tester = YOLOTester()
    
    # Find available models in common locations
    model_paths = []
    
    # Check for trained models in backend/runs
    runs_dir = Path("backend/runs/detect")
    if runs_dir.exists():
        model_paths.extend(runs_dir.glob("*/weights/best.pt"))
        model_paths.extend(runs_dir.glob("*/weights/last.pt"))
    
    # Check for pre-trained models in backend directory
    backend_dir = Path("backend")
    for model_file in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]:
        model_path = backend_dir / model_file
        if model_path.exists():
            model_paths.append(model_path)
    
    # Convert to strings
    available_models = [str(p) for p in model_paths]
    
    if not available_models:
        available_models = ["backend/yolov8n.pt"]  # Default fallback
    
    with gr.Blocks(title="YOLO Model Tester", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎯 YOLO Model Tester
            
            **Standalone testing tool** for your trained YOLO models.
            Upload an image and see real-time object detection results!
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Model Selection")
                
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    label="Select Model",
                    value=available_models[0] if available_models else None,
                    interactive=True
                )
                
                model_path_input = gr.Textbox(
                    label="Or enter custom model path",
                    placeholder="backend/runs/detect/train/weights/best.pt",
                    lines=1
                )
                
                load_btn = gr.Button("🔄 Load Model", variant="primary")
                model_status = gr.Textbox(label="Status", lines=2, interactive=False)
                
                gr.Markdown("### ⚙️ Detection Settings")
                
                confidence_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.25,
                    step=0.01,
                    label="Confidence Threshold",
                    info="Minimum confidence to show detections"
                )
                
                iou_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.45,
                    step=0.01,
                    label="IoU Threshold",
                    info="IoU threshold for Non-Maximum Suppression"
                )
                
                predict_btn = gr.Button("🚀 Run Detection", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ---
                    ### 💡 Tips
                    - Lower confidence = more detections
                    - Higher IoU = more overlapping boxes
                    - Use your trained models from `backend/runs/detect/`
                    """
                )
        
            with gr.Column(scale=2):
                gr.Markdown("### 📸 Image Input & Results")
                
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400
                )
                
                output_image = gr.Image(
                    label="Detection Results",
                    type="numpy",
                    height=400
                )
                
                detection_details = gr.Textbox(
                    label="Detection Details",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )
        
        # Examples section
        gr.Markdown("### 📚 How to Use")
        gr.Markdown(
            """
            1. **Load Model**: Select or enter path to your `.pt` model
            2. **Upload Image**: Click on image area to upload
            3. **Run Detection**: Click the button to detect objects
            4. **Adjust Settings**: Fine-tune confidence/IoU thresholds
            """
        )
        
        # Event handlers
        def load_model_click(dropdown_value, custom_path):
            """Handle model loading."""
            model_path = custom_path if custom_path else dropdown_value
            if not model_path:
                return "❌ Please select or enter a model path"
            return tester.load_model(model_path)
        
        load_btn.click(
            fn=load_model_click,
            inputs=[model_dropdown, model_path_input],
            outputs=model_status
        )
        
        predict_btn.click(
            fn=tester.predict,
            inputs=[input_image, confidence_slider, iou_slider],
            outputs=[output_image, detection_details]
        )
        
        # Auto-load first model on startup
        demo.load(
            fn=lambda: tester.load_model(available_models[0]) if available_models else "⚠️ No models found - please enter model path",
            outputs=model_status
        )
    
    return demo


def main():
    """Launch the Gradio interface."""
    print("=" * 70)
    print("YOLO MODEL TESTER - GRADIO INTERFACE")
    print("=" * 70)
    print("\n🚀 Starting Gradio interface...")
    print("📝 The interface will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",  # Local only (change to 0.0.0.0 for network access)
        server_port=7860,
        share=False,  # Set to True to get a public URL
        inbrowser=True  # Auto-open in browser
    )


if __name__ == "__main__":
    main()
