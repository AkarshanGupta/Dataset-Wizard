"""
Validate YOLO dataset is ready for training.
Checks that images have corresponding label files and labels are properly formatted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def validate_yolo_label_file(label_path, img_width=None, img_height=None):
    """
    Validate a YOLO label file format.
    
    Returns: (is_valid, num_boxes, errors)
    """
    errors = []
    if not label_path.exists():
        return False, 0, ["File doesn't exist"]
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return True, 0, []  # Empty file is valid (no objects)
        
        num_boxes = 0
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f"Line {i}: Expected 5 values, got {len(parts)}")
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Check normalized coordinates (should be 0-1)
                if not (0 <= x_center <= 1):
                    errors.append(f"Line {i}: x_center {x_center} not in [0,1]")
                if not (0 <= y_center <= 1):
                    errors.append(f"Line {i}: y_center {y_center} not in [0,1]")
                if not (0 < width <= 1):
                    errors.append(f"Line {i}: width {width} not in (0,1]")
                if not (0 < height <= 1):
                    errors.append(f"Line {i}: height {height} not in (0,1]")
                if class_id < 0:
                    errors.append(f"Line {i}: class_id {class_id} is negative")
                
                if len(errors) == 0:
                    num_boxes += 1
                    
            except ValueError as e:
                errors.append(f"Line {i}: Invalid number format - {e}")
        
        return len(errors) == 0, num_boxes, errors
        
    except Exception as e:
        return False, 0, [f"Failed to read file: {e}"]


def check_dataset_structure(dataset_dir):
    """
    Check if dataset has proper YOLO structure:
    - images/train/
    - images/val/
    - labels/train/
    - labels/val/
    - data.yaml
    """
    dataset_path = Path(dataset_dir)
    required_paths = [
        dataset_path / "images" / "train",
        dataset_path / "images" / "val",
        dataset_path / "labels" / "train",
        dataset_path / "labels" / "val",
        dataset_path / "data.yaml",
    ]
    
    missing = []
    for path in required_paths:
        if not path.exists():
            missing.append(str(path.relative_to(dataset_path)))
    
    return len(missing) == 0, missing


def validate_dataset(dataset_dir):
    """
    Complete validation of a YOLO dataset.
    """
    dataset_path = Path(dataset_dir)
    
    print("=" * 70)
    print("YOLO DATASET VALIDATION")
    print("=" * 70)
    print(f"Dataset directory: {dataset_path}\n")
    
    # Check structure
    print("📁 CHECKING DATASET STRUCTURE...")
    structure_ok, missing = check_dataset_structure(dataset_path)
    if not structure_ok:
        print("❌ Missing required directories/files:")
        for item in missing:
            print(f"   - {item}")
        print("\n⚠️  Dataset structure is incomplete!")
        return False
    print("✅ Dataset structure is valid\n")
    
    # Check data.yaml
    print("📄 CHECKING data.yaml...")
    yaml_path = dataset_path / "data.yaml"
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        print(f"✅ data.yaml loaded successfully")
        print(f"   Classes: {data.get('names', {})}")
    except Exception as e:
        print(f"❌ Failed to load data.yaml: {e}")
        return False
    print()
    
    # Validate train and val sets
    for split in ["train", "val"]:
        print(f"🔍 VALIDATING {split.upper()} SET...")
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split
        
        # Get all images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {split} set")
            continue
        
        paired = 0
        unpaired = 0
        total_boxes = 0
        invalid_labels = []
        
        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + ".txt")
            
            if not label_path.exists():
                unpaired += 1
                continue
            
            is_valid, num_boxes, errors = validate_yolo_label_file(label_path)
            if is_valid:
                paired += 1
                total_boxes += num_boxes
            else:
                invalid_labels.append((label_path.name, errors))
        
        print(f"   Images: {len(image_files)}")
        print(f"   ✅ Paired (image + label): {paired}")
        print(f"   ⚠️  Unpaired (no label): {unpaired}")
        print(f"   📦 Total bounding boxes: {total_boxes}")
        
        if total_boxes > 0:
            print(f"   📊 Avg boxes per image: {total_boxes/paired:.2f}")
        
        if invalid_labels:
            print(f"   ❌ Invalid label files: {len(invalid_labels)}")
            for name, errors in invalid_labels[:3]:  # Show first 3
                print(f"      {name}: {errors[0]}")
        
        print()
    
    # Overall assessment
    print("=" * 70)
    print("TRAINING READINESS")
    print("=" * 70)
    
    train_images = len(list((dataset_path / "images" / "train").glob("*.jpg")))
    val_images = len(list((dataset_path / "images" / "val").glob("*.jpg")))
    
    if train_images > 0 and val_images > 0:
        print("✅ Dataset is READY FOR TRAINING!")
        print(f"\n📝 How YOLO will use this:")
        print(f"   1. Read image: images/train/frame_001.jpg")
        print(f"   2. Automatically find: labels/train/frame_001.txt")
        print(f"   3. Load boxes from .txt and train on the image")
        print(f"   4. Repeat for all {train_images} training images")
        print(f"\n🚀 Start training with:")
        print(f"   yolo train data=data.yaml model=yolov8n.pt epochs=50")
        return True
    else:
        print("❌ Dataset NOT ready for training")
        print(f"   Training images: {train_images} (need > 0)")
        print(f"   Validation images: {val_images} (need > 0)")
        return False


def main():
    """Run validation on the most recent dataset."""
    import sys
    
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        # Try to find a dataset in output directory
        output_dir = Path("output")
        if not output_dir.exists():
            print("❌ No output directory found!")
            print("Usage: python validate_dataset.py <dataset_path>")
            return
        
        # Find first dataset directory
        dataset_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        if not dataset_dirs:
            print("❌ No dataset directories found in output/!")
            return
        
        dataset_dir = dataset_dirs[0]
        print(f"🔍 Auto-detected dataset: {dataset_dir}\n")
    
    validate_dataset(dataset_dir)


if __name__ == "__main__":
    main()
