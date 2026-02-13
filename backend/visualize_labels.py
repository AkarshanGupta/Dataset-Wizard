"""
Visualize YOLO labels on images to verify detection quality.
Shows bounding boxes drawn on the actual frames.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

sys.path.insert(0, str(Path(__file__).parent))

def yolo_to_pixels(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO normalized coordinates to pixel coordinates."""
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return x1, y1, x2, y2

def draw_labels_on_image(frame_path, label_path, output_path, class_names):
    """Draw bounding boxes from YOLO label file onto image."""
    # Load image
    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    # Colors for different classes
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta"]
    
    # Read label file
    if not Path(label_path).exists():
        print(f"⚠️  No label file: {label_path}")
        return 0
    
    detections = 0
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixels
                x1, y1, x2, y2 = yolo_to_pixels(x_center, y_center, width, height, img_width, img_height)
                
                # Draw box
                color = colors[class_id % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label_text = f"{class_names.get(class_id, 'unknown')}"
                draw.text((x1 + 2, y1 + 2), label_text, fill="white")
                draw.text((x1, y1), label_text, fill=color)
                
                detections += 1
    
    # Save
    img.save(output_path)
    return detections

def main():
    # Class names
    class_names = {
        0: "person",
        1: "train",
        2: "car",
        3: "truck",
        4: "bus",
    }
    
    frames_dir = Path("output/subway_surfers/frames")
    labels_dir = Path("output/subway_surfers/labels_raw")
    output_dir = Path("output/subway_surfers/visualized")
    output_dir.mkdir(exist_ok=True)
    
    if not frames_dir.exists():
        print("❌ No frames directory found!")
        return
    
    # Get all frames
    frame_files = sorted(frames_dir.glob("*.jpg"))[:10]  # First 10 frames
    
    print("=" * 60)
    print("YOLO LABEL VISUALIZATION")
    print("=" * 60)
    
    total_detections = 0
    frames_with_labels = 0
    
    for frame_path in frame_files:
        label_path = labels_dir / (frame_path.stem + ".txt")
        output_path = output_dir / frame_path.name
        
        detections = draw_labels_on_image(frame_path, label_path, output_path, class_names)
        total_detections += detections
        
        if detections > 0:
            frames_with_labels += 1
            print(f"✅ {frame_path.name}: {detections} objects → {output_path.name}")
        else:
            print(f"⚪ {frame_path.name}: No labels")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {len(frame_files)}")
    print(f"Frames with labels: {frames_with_labels}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections/frame: {total_detections/len(frame_files):.1f}")
    print(f"\n📁 Visualizations saved to: {output_dir}")
    print("\nCLASS KEY:")
    for class_id, name in class_names.items():
        print(f"  {class_id}: {name}")

if __name__ == "__main__":
    main()
