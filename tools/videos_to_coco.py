import os
import json
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

def extract_frames(video_path, output_dir, video_id, sample_rate=1):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        video_id: Numeric ID for the video (zero-padded)
        sample_rate: Extract every nth frame
    
    Returns:
        List of dictionaries containing info about extracted frames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return []
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video {video_id:06d}: {video_path}")
    print(f"Total frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")
    
    frames_info = []
    
    frame_id = 0
    frame_index = 0
    
    with tqdm(total=frame_count) as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            if frame_id % sample_rate == 0:
                # Use zero-padded video ID and frame ID in filename
                frame_filename = f"{video_id:06d}_{frame_id:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                frame_info = {
                    "file_name": os.path.join(os.path.basename(output_dir), frame_filename),
                    "id": frame_index + 1,
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "height": height,
                    "width": width,
                    "prev_image_id": frame_index if frame_index > 0 else -1,
                    "next_image_id": frame_index + 2 if frame_id < frame_count - 1 else -1
                }
                frames_info.append(frame_info)
                frame_index += 1
            
            frame_id += 1
            pbar.update(1)
    
    video.release()
    return frames_info

def create_coco_dataset(videos_dir, output_dir, sample_rate=1):
    """
    Create a COCO-style dataset from videos.
    
    Args:
        videos_dir: Directory containing video files
        output_dir: Directory to save the dataset
        sample_rate: Extract every nth frame
    """
    # Create output directories
    frames_dir = os.path.join(output_dir, "test")
    annotations_dir = os.path.join(output_dir, "annotations")
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Initialize COCO format
    coco_data = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{"id": 1, "name": "pedestrian"}]
    }
    
    # Process each video file
    image_count = 0
    video_files = []
    
    # Find video files
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(Path(videos_dir).glob(f'**/*{ext}')))
    
    for video_idx, video_file in enumerate(video_files):
        # Use zero-padded numbers as video IDs, starting from 1
        video_id = video_idx + 1
        
        # Add video info
        coco_data["videos"].append({
            "id": video_id,
            "file_name": f"{video_id:06d}"  # Zero-padded video ID
        })
        
        # Create directory for this video's frames
        video_frames_dir = os.path.join(frames_dir, f"{video_id:06d}")
        
        # Extract frames
        frames_info = extract_frames(str(video_file), video_frames_dir, video_id, sample_rate)
        
        # Update image IDs to ensure they're continuous across videos
        for frame in frames_info:
            frame["id"] += image_count
            if frame["prev_image_id"] != -1:
                frame["prev_image_id"] += image_count
            if frame["next_image_id"] != -1:
                frame["next_image_id"] += image_count
            
            coco_data["images"].append(frame)
        
        image_count += len(frames_info)
    
    # Save annotations
    output_path = os.path.join(annotations_dir, "test.json")
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"Dataset created successfully: {output_path}")
    print(f"Processed {len(video_files)} videos with {image_count} frames total")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert videos to COCO format dataset")
    parser.add_argument("--videos_dir", required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", required=True, help="Directory to save the dataset")
    parser.add_argument("--sample_rate", type=int, default=1, help="Extract every nth frame")
    
    args = parser.parse_args()
    
    create_coco_dataset(args.videos_dir, args.output_dir, args.sample_rate)