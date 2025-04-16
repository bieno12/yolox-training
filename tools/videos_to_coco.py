import os
import json
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Convert videos to COCO format dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for COCO dataset')
    parser.add_argument('--annotation_dir', type=str, default=None, help='Optional directory containing annotation files (should match video filenames)')
    parser.add_argument('--frame_interval', type=int, default=1, help='Extract frames at this interval (default: 1 = every frame)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training split (default: 0.8)')
    parser.add_argument('--categories', type=str, default='person', help='Comma-separated list of categories (default: person)')
    parser.add_argument('--detector', type=str, default=None, help='Optional object detector to use (yolo, fasterrcnn, or None)')
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Create the necessary directory structure for COCO dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for images
    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "images"), exist_ok=True)
    
    # Create directory for annotations
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    return {
        "train": os.path.join(output_dir, "train", "images"),
        "val": os.path.join(output_dir, "val", "images"),
        "test": os.path.join(output_dir, "test", "images"),
        "annotations": os.path.join(output_dir, "annotations")
    }

def get_video_files(input_dir):
    """Get all video files from the input directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file_path)
    
    return video_files

def initialize_coco_dict(categories):
    """Initialize a COCO format dictionary"""
    category_list = []
    for i, category in enumerate(categories, 1):
        category_list.append({
            "id": i,
            "name": category,
            "supercategory": "object"
        })
    
    return {
        "info": {
            "description": "Video frames converted to COCO format",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "Generated using video-to-coco-converter",
            "date_created": ""
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": category_list
    }

def load_detection_model(detector_name):
    """Load the specified object detection model"""
    if detector_name == 'yolo':
        try:
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return model
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return None
    elif detector_name == 'fasterrcnn':
        try:
            import torchvision
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load Faster R-CNN model: {e}")
            return None
    return None

def detect_objects(frame, model, detector_name, confidence_threshold=0.5):
    """Detect objects in a frame using the specified model"""
    if detector_name == 'yolo':
        results = model(frame)
        detections = []
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= confidence_threshold:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                detections.append({
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'category_id': int(cls) + 1,  # YOLO uses 0-indexed classes
                    'score': float(conf)
                })
        return detections
    
    elif detector_name == 'fasterrcnn':
        import torch
        import torchvision.transforms as T
        
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(frame)
        
        with torch.no_grad():
            prediction = model([img_tensor])
        
        detections = []
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                detections.append({
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'category_id': int(label),
                    'score': float(score)
                })
        return detections
    
    return []

def read_annotation_file(filepath, video_name):
    """Read an annotation file if it exists"""
    # This is a placeholder - you'd need to adapt this to your annotation format
    if not filepath:
        return {}
    
    # Strip extension from video name to match annotation file
    base_name = os.path.splitext(os.path.basename(video_name))[0]
    potential_paths = [
        os.path.join(filepath, f"{base_name}.json"),
        os.path.join(filepath, f"{base_name}.txt"),
        os.path.join(filepath, base_name, "annotations.json"),
        os.path.join(filepath, base_name, "gt.txt")
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            if path.endswith('.json'):
                with open(path, 'r') as f:
                    return json.load(f)
            elif path.endswith('.txt'):
                # Parse text format - assuming format is: frame_id,object_id,x,y,w,h,score,class_id
                annotations = {}
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:  # At minimum need frame_id, x, y, w, h, class
                            frame_id = int(parts[0])
                            if frame_id not in annotations:
                                annotations[frame_id] = []
                            
                            # Parse the rest based on format
                            # This is a generic parser - adjust based on your specific format
                            if len(parts) >= 8:  # MOT format with score and class_id
                                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                                score = float(parts[6]) if len(parts) > 6 else 1.0
                                class_id = int(parts[7]) if len(parts) > 7 else 1
                            else:  # Basic format
                                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                score = float(parts[5]) if len(parts) > 5 else 1.0
                                class_id = 1  # Default to first category
                                
                            annotations[frame_id].append({
                                'bbox': [x, y, w, h],
                                'category_id': class_id,
                                'score': score
                            })
                return annotations
    
    return {}

def process_videos(args):
    """Process videos and convert to COCO format dataset"""
    # Create directory structure
    dirs = create_directory_structure(args.output_dir)
    
    # Get all video files
    video_files = get_video_files(args.input_dir)
    if not video_files:
        print("No video files found in the input directory!")
        return
    
    # Initialize COCO dictionaries for each split
    categories = [cat.strip() for cat in args.categories.split(',')]
    train_coco = initialize_coco_dict(categories)
    val_coco = initialize_coco_dict(categories)
    test_coco = initialize_coco_dict(categories)
    
    # Load detection model if specified
    detector = None
    if args.detector:
        detector = load_detection_model(args.detector)
        if not detector:
            print(f"Warning: Failed to load {args.detector} detector. Proceeding without detection.")
    
    # Initialize counters
    image_id = 0
    annotation_id = 0
    video_id = 0
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_id += 1
        video_name = os.path.basename(video_path)
        video_basename = os.path.splitext(video_name)[0]
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Add video info to COCO dict
        video_info = {
            "id": video_id,
            "file_name": video_name,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count
        }
        
        # Decide on train/val/test split - use video_id as a deterministic way to split
        split_rng = np.random.RandomState(video_id)
        split_value = split_rng.random()
        
        if split_value < args.train_ratio:
            split = "train"
            coco_dict = train_coco
        elif split_value < args.train_ratio + (1 - args.train_ratio) / 2:
            split = "val"
            coco_dict = val_coco
        else:
            split = "test"
            coco_dict = test_coco
        
        coco_dict["videos"].append(video_info)
        
        # Get annotations if available
        annotations = read_annotation_file(args.annotation_dir, video_path)
        
        # Process frames
        frame_idx = 0
        prev_image_id = -1
        
        with tqdm(total=frame_count, desc=f"Extracting frames from {video_name}", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame at specified interval
                if frame_idx % args.frame_interval == 0:
                    # Generate unique image filename
                    image_filename = f"{video_basename}_{frame_idx:06d}.jpg"
                    image_path = os.path.join(dirs[split], image_filename)
                    
                    # Save the frame
                    cv2.imwrite(image_path, frame)
                    
                    # Create image info for COCO format
                    image_id += 1
                    image_info = {
                        "id": image_id,
                        "file_name": os.path.join(split, "images", image_filename),
                        "width": width,
                        "height": height,
                        "video_id": video_id,
                        "frame_id": frame_idx,
                        "prev_image_id": prev_image_id if prev_image_id != -1 else -1,
                        "next_image_id": -1  # Will be updated in the next iteration
                    }
                    
                    # Update previous image's next_image_id
                    if prev_image_id != -1:
                        for img in coco_dict["images"]:
                            if img["id"] == prev_image_id:
                                img["next_image_id"] = image_id
                                break
                    
                    prev_image_id = image_id
                    coco_dict["images"].append(image_info)
                    
                    # Add annotations if available
                    frame_annotations = []
                    if frame_idx in annotations:
                        frame_annotations = annotations[frame_idx]
                    # If no annotations and detector is provided, detect objects
                    elif detector:
                        frame_annotations = detect_objects(frame, detector, args.detector)
                    
                    # Add annotations to COCO dict
                    for ann in frame_annotations:
                        annotation_id += 1
                        x, y, w, h = ann['bbox']
                        
                        # Ensure bbox coordinates are within image bounds
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        coco_annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": ann['category_id'],
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "score": ann.get('score', 1.0)
                        }
                        
                        # Add track_id if available (for tracking datasets)
                        if 'track_id' in ann:
                            coco_annotation["track_id"] = ann['track_id']
                        
                        coco_dict["annotations"].append(coco_annotation)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
    
    # Save COCO JSON files
    print("Saving annotation files...")
    with open(os.path.join(dirs["annotations"], "train.json"), 'w') as f:
        json.dump(train_coco, f)
    
    with open(os.path.join(dirs["annotations"], "val.json"), 'w') as f:
        json.dump(val_coco, f)
    
    with open(os.path.join(dirs["annotations"], "test.json"), 'w') as f:
        json.dump(test_coco, f)
    
    # Create combined annotations for convenience
    combined_coco = initialize_coco_dict(categories)
    combined_coco["images"] = train_coco["images"] + val_coco["images"]
    combined_coco["annotations"] = train_coco["annotations"] + val_coco["annotations"]
    combined_coco["videos"] = train_coco["videos"] + val_coco["videos"]
    
    with open(os.path.join(dirs["annotations"], "combined.json"), 'w') as f:
        json.dump(combined_coco, f)
    
    print(f"Conversion complete! Dataset saved to {args.output_dir}")
    print(f"Train set: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")
    print(f"Val set: {len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations")
    print(f"Test set: {len(test_coco['images'])} images, {len(test_coco['annotations'])} annotations")

if __name__ == "__main__":
    args = parse_args()
    process_videos(args)