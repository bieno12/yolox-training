import os
import json
import cv2
import argparse
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_file_exists(file_path):
    """Check if a file exists and is accessible."""
    if not os.path.isfile(file_path):
        return False
    # Check if file is accessible and not empty
    try:
        if os.path.getsize(file_path) == 0:
            return False
        return True
    except (OSError, IOError):
        return False

def extract_frames(video_path, output_dir, sample_rate=1):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        sample_rate: Extract every nth frame
    
    Returns:
        List of dictionaries containing info about extracted frames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if not verify_file_exists(video_path):
        logger.error(f"Video file not found or not accessible: {video_path}")
        return []
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return []
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_info = []
    
    frame_id = 0
    frame_index = 0
    
    with tqdm(total=frame_count) as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            if frame_id % sample_rate == 0:
                frame_filename = f"{video_name}_{frame_id:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Write frame to file and verify it was written successfully
                success = cv2.imwrite(frame_path, frame)
                if not success:
                    logger.warning(f"Failed to write frame to {frame_path}")
                    frame_id += 1
                    pbar.update(1)
                    continue
                
                # Verify the image was saved correctly
                if not verify_file_exists(frame_path):
                    logger.warning(f"Frame file not created or empty: {frame_path}")
                    frame_id += 1
                    pbar.update(1)
                    continue
                
                frame_info = {
                    "file_name": os.path.join(os.path.basename(output_dir), frame_filename),
                    "id": frame_index + 1,
                    "frame_id": frame_id,
                    "video_id": video_name,
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
    # Check if input directory exists
    if not os.path.isdir(videos_dir):
        logger.error(f"Input directory does not exist: {videos_dir}")
        return
    
    # Create output directories
    frames_dir = os.path.join(output_dir, "images")
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
    video_count = 0
    image_count = 0
    video_files = []
    
    # Find video files
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(Path(videos_dir).glob(f'**/*{ext}')))
    
    if not video_files:
        logger.warning(f"No video files found in {videos_dir}")
        return
    
    for video_file in video_files:
        try:
            video_count += 1
            video_name = video_file.stem
            
            # Add video info
            coco_data["videos"].append({
                "id": video_count,
                "file_name": video_name
            })
            
            # Create directory for this video's frames
            video_frames_dir = os.path.join(frames_dir, video_name)
            
            # Extract frames
            frames_info = extract_frames(str(video_file), video_frames_dir, sample_rate)
            
            if not frames_info:
                logger.warning(f"No frames extracted from {video_file}")
                continue
            
            # Update image IDs to ensure they're continuous across videos
            for frame in frames_info:
                frame["id"] += image_count
                if frame["prev_image_id"] != -1:
                    frame["prev_image_id"] += image_count
                if frame["next_image_id"] != -1:
                    frame["next_image_id"] += image_count
                
                coco_data["images"].append(frame)
            
            image_count += len(frames_info)
            logger.info(f"Extracted {len(frames_info)} frames from {video_name}")
            
        except Exception as e:
            logger.error(f"Error processing video {video_file}: {str(e)}")
    
    # Save annotations
    if image_count == 0:
        logger.error("No images were processed. Cannot create annotation file.")
        return
    
    try:
        output_path = os.path.join(annotations_dir, "test.json")
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        
        logger.info(f"Dataset created successfully: {output_path}")
        logger.info(f"Processed {video_count} videos with {image_count} frames total")
    except Exception as e:
        logger.error(f"Error saving annotation file: {str(e)}")

def create_coco_from_images(images_dir, output_dir):
    """
    Create a COCO-style dataset from existing image files.
    
    Args:
        images_dir: Directory containing image files
        output_dir: Directory to save the dataset
    """
    # Check if input directory exists
    if not os.path.isdir(images_dir):
        logger.error(f"Input directory does not exist: {images_dir}")
        return
    
    # Create output directories
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Initialize COCO format
    coco_data = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{"id": 1, "name": "pedestrian"}]
    }
    
    # Find all subdirectories (considering each as a video)
    video_dirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    
    if not video_dirs:
        # If no subdirectories, process all images in the main directory
        video_dirs = [""]
    
    image_count = 0
    video_count = 0
    
    for video_dir in video_dirs:
        video_count += 1
        video_path = os.path.join(images_dir, video_dir)
        video_name = os.path.basename(video_path)
        
        # Add video info
        coco_data["videos"].append({
            "id": video_count,
            "file_name": video_name
        })
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(video_path).glob(f'*{ext}')))
        
        if not image_files:
            logger.warning(f"No image files found in {video_path}")
            continue
        
        # Sort image files
        image_files.sort()
        
        logger.info(f"Processing {len(image_files)} images from {video_name}")
        
        # Process each image
        frame_index = 0
        valid_images = []
        
        for img_file in tqdm(image_files):
            try:
                # Read image to get dimensions
                img = cv2.imread(str(img_file))
                if img is None:
                    logger.warning(f"Failed to read image: {img_file}")
                    continue
                
                height, width = img.shape[:2]
                
                # Save image info
                rel_path = os.path.relpath(str(img_file), images_dir)
                frame_info = {
                    "file_name": rel_path,
                    "id": image_count + frame_index + 1,
                    "frame_id": frame_index,
                    "video_id": video_count,
                    "height": height,
                    "width": width,
                    "prev_image_id": image_count + frame_index if frame_index > 0 else -1,
                    "next_image_id": image_count + frame_index + 2 if frame_index < len(image_files) - 1 else -1
                }
                
                valid_images.append(frame_info)
                frame_index += 1
                
            except Exception as e:
                logger.error(f"Error processing image {img_file}: {str(e)}")
        
        # Add valid images to dataset
        coco_data["images"].extend(valid_images)
        image_count += len(valid_images)
        logger.info(f"Added {len(valid_images)} valid images from {video_name}")
    
    # Save annotations
    if image_count == 0:
        logger.error("No images were processed. Cannot create annotation file.")
        return
    
    try:
        output_path = os.path.join(annotations_dir, "test.json")
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        
        logger.info(f"Dataset created successfully: {output_path}")
        logger.info(f"Processed {video_count} videos with {image_count} frames total")
    except Exception as e:
        logger.error(f"Error saving annotation file: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert videos to COCO format dataset")
    parser.add_argument("--input_dir", required=True, help="Directory containing video files or image files")
    parser.add_argument("--output_dir", required=True, help="Directory to save the dataset")
    parser.add_argument("--sample_rate", type=int, default=1, help="Extract every nth frame (for video processing)")
    parser.add_argument("--mode", choices=["video", "image"], default="video", 
                        help="Process videos or use existing images")
    
    args = parser.parse_args()
    
    if args.mode == "video":
        create_coco_dataset(args.input_dir, args.output_dir, args.sample_rate)
    else:
        create_coco_from_images(args.input_dir, args.output_dir)