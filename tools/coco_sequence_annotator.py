import os
import json
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def annotate_sequence(sequence_path, gt_path, output_video_path, fps=30):
    """
    Annotate frames from a sequence using ground truth data and create a video.
    
    Args:
        sequence_path: Directory containing sequence frames
        gt_path: Path to ground truth CSV file
        output_video_path: Path to save the output video
        fps: Frames per second for the output video
    """
    # Load ground truth data
    try:
        ground_truth = pd.read_csv(gt_path, header=None,
                                names=['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                                      'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility', '-1'])
    except pd.errors.EmptyDataError:
        print(f"Warning: Ground truth file {gt_path} is empty")
        ground_truth = pd.DataFrame(columns=['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                                           'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility', '-1'])
    except Exception as e:
        print(f"Error reading ground truth file {gt_path}: {e}")
        return
    
    # Get all frames in the sequence directory
    frames = sorted([f for f in os.listdir(sequence_path) if f.endswith(('.jpg', '.png'))])
    
    if not frames:
        print(f"No frames found in {sequence_path}")
        return
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(sequence_path, frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Unable to read frame {first_frame_path}")
        return
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Generate unique colors for object IDs
    colors = {}
    
    # Process each frame
    print(f"Processing {len(frames)} frames from {sequence_path}")
    for frame_file in tqdm(frames, desc="Annotating Frames"):
        # Extract frame number from filename
        # Assuming filename format: {video_id}_{frame_number}.jpg
        try:
            frame_number = int(frame_file.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"Warning: Could not extract frame number from {frame_file}, using file index")
            frame_number = frames.index(frame_file) + 1
        
        # Read the frame
        frame_path = os.path.join(sequence_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Unable to read frame {frame_path}")
            continue
        
        # Filter ground truth data for the current frame
        frame_data = ground_truth[ground_truth['frame_number'] == frame_number]
        
        # Draw annotations
        for _, row in frame_data.iterrows():
            track_id = int(row['track_object_id'])
            x, y, w, h = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_w']), int(row['bbox_h'])
            
            # Handle cases where confidence and visibility might not be present
            confidence = row.get('confidence_score', 1.0)
            class_name = row.get('class', 1)
            visibility = row.get('visibility', 1.0)
            
            # Assign a unique color to each object ID
            if track_id not in colors:
                colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), colors[track_id], 2)
            
            # Put label text
            label = f"ID:{track_id} C:{class_name} V:{visibility:.2f}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[track_id], 2)
        
        # Add frame number to top left corner
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Video saved to {output_video_path}")

def process_coco_dataset(coco_json_path, coco_image_dir, gt_dir, output_dir, fps=30):
    """
    Process a COCO dataset, annotate sequences and create videos.
    
    Args:
        coco_json_path: Path to COCO JSON file
        coco_image_dir: Directory containing image sequences
        gt_dir: Directory containing ground truth files
        output_dir: Directory to save output videos
        fps: Frames per second for output videos
    """
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Group images by video ID
    videos = {}
    for image in coco_data['images']:
        video_id = image.get('video_id')
        if video_id not in videos:
            videos[video_id] = []
        videos[video_id].append(image)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video sequence
    for video_id, images in videos.items():
        print(f"\nProcessing video ID: {video_id}")
        
        # Convert video_id to string and zero-pad if necessary
        video_id_str = f"{video_id:06d}" if isinstance(video_id, int) else str(video_id)
        
        # Find the sequence directory
        sequence_dir = None
        for root, dirs, _ in os.walk(coco_image_dir):
            for d in dirs:
                if d == video_id_str:
                    sequence_dir = os.path.join(root, d)
                    break
            if sequence_dir:
                break
        
        if not sequence_dir:
            print(f"Warning: Could not find sequence directory for video ID {video_id}")
            continue
        
        # Find corresponding ground truth file
        gt_file = None
        for file in os.listdir(gt_dir):
            if file.startswith(f"{video_id_str}") and file.endswith('.csv'):
                gt_file = os.path.join(gt_dir, file)
                break
            elif file == f"gt_{video_id_str}.txt" or file == f"gt_{video_id_str}.csv":
                gt_file = os.path.join(gt_dir, file)
                break
        
        if not gt_file:
            print(f"Warning: No ground truth file found for video ID {video_id}")
            gt_file = os.path.join(gt_dir, f"gt_{video_id_str}.csv")  # Use a placeholder path
            # Create an empty GT file for visualization with just the frames
            with open(gt_file, 'w') as f:
                f.write('')
        
        # Define output video path
        output_video_path = os.path.join(output_dir, f"{video_id_str}.avi")
        
        # Annotate sequence and create video
        annotate_sequence(sequence_dir, gt_file, output_video_path, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO sequences to annotated videos")
    parser.add_argument("--coco_json", required=True, help="Path to COCO JSON file")
    parser.add_argument("--image_dir", required=True, help="Directory containing image sequences")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground truth files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output videos")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output videos")
    
    args = parser.parse_args()
    
    process_coco_dataset(args.coco_json, args.image_dir, args.gt_dir, args.output_dir, args.fps)