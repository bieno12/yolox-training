import os
import json
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def load_ground_truth(gt_path):
    """
    Load ground truth data from CSV or TXT file with flexible handling for different formats.
    
    Args:
        gt_path: Path to ground truth file
    
    Returns:
        DataFrame with ground truth data
    """
    try:
        # First try standard CSV format
        ground_truth = pd.read_csv(gt_path, header=None)
        
        # Determine the number of columns and assign appropriate headers
        num_columns = ground_truth.shape[1]
        
        if num_columns >= 10:
            column_names = ['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                           'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility', 'unused']
        elif num_columns == 9:
            column_names = ['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                           'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility']
        elif num_columns == 8:
            column_names = ['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                           'bbox_w', 'bbox_h', 'confidence_score', 'class']
            ground_truth['visibility'] = 1.0  # Add default visibility
        elif num_columns == 7:
            column_names = ['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                           'bbox_w', 'bbox_h', 'confidence_score']
            ground_truth['class'] = 1  # Add default class
            ground_truth['visibility'] = 1.0  # Add default visibility
        elif num_columns == 6:
            column_names = ['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                           'bbox_w', 'bbox_h']
            ground_truth['confidence_score'] = 1.0  # Add default confidence
            ground_truth['class'] = 1  # Add default class
            ground_truth['visibility'] = 1.0  # Add default visibility
        else:
            print(f"Warning: Unexpected number of columns ({num_columns}) in {gt_path}")
            return pd.DataFrame(columns=['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                                       'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility'])
        
        # Rename columns based on detected format
        ground_truth.columns = column_names[:num_columns] + list(ground_truth.columns[num_columns:])
        
        # Ensure all required columns exist
        for col in ['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']:
            if col not in ground_truth.columns:
                print(f"Error: Required column '{col}' not found in {gt_path}")
                return None
        
        # Convert columns to appropriate types
        ground_truth['frame_number'] = ground_truth['frame_number'].astype(int)
        ground_truth['track_object_id'] = ground_truth['track_object_id'].astype(int)
        ground_truth['bbox_x'] = ground_truth['bbox_x'].astype(float).astype(int)
        ground_truth['bbox_y'] = ground_truth['bbox_y'].astype(float).astype(int)
        ground_truth['bbox_w'] = ground_truth['bbox_w'].astype(float).astype(int)
        ground_truth['bbox_h'] = ground_truth['bbox_h'].astype(float).astype(int)
        
        return ground_truth
    
    except pd.errors.EmptyDataError:
        print(f"Warning: Ground truth file {gt_path} is empty")
        return pd.DataFrame(columns=['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                                   'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility'])
    
    except Exception as e:
        print(f"Error reading ground truth file {gt_path}: {e}")
        print("Attempting to read with different delimiters...")
        
        # Try different delimiters
        for delimiter in [' ', '\t', ',']:
            try:
                ground_truth = pd.read_csv(gt_path, header=None, delimiter=delimiter)
                print(f"Successfully read file with delimiter '{delimiter}'")
                
                # Reconfigure columns as above
                num_columns = ground_truth.shape[1]
                column_names = ['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                              'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility']
                
                ground_truth.columns = column_names[:min(num_columns, len(column_names))]
                
                # Ensure required columns exist
                for col in ['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']:
                    if col not in ground_truth.columns:
                        continue
                
                # Convert columns to appropriate types
                ground_truth['frame_number'] = ground_truth['frame_number'].astype(int)
                ground_truth['track_object_id'] = ground_truth['track_object_id'].astype(int)
                ground_truth['bbox_x'] = ground_truth['bbox_x'].astype(float).astype(int)
                ground_truth['bbox_y'] = ground_truth['bbox_y'].astype(float).astype(int)
                ground_truth['bbox_w'] = ground_truth['bbox_w'].astype(float).astype(int)
                ground_truth['bbox_h'] = ground_truth['bbox_h'].astype(float).astype(int)
                
                return ground_truth
            except:
                continue
        
        # If all attempts fail, return empty DataFrame
        print(f"Failed to read {gt_path} with all attempted delimiters")
        return pd.DataFrame(columns=['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                                   'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility'])

def extract_frame_number(frame_file):
    """
    Extract frame number from filename with multiple possible formats.
    
    Args:
        frame_file: Filename of the frame
    
    Returns:
        Frame number as integer
    """
    try:
        # Try pattern: {video_id}_{frame_number}.jpg
        parts = frame_file.split('_')
        if len(parts) >= 2:
            frame_number = int(parts[1].split('.')[0])  # Get number before file extension
            return frame_number
        
        # Try pattern: frame{number}.jpg or {number}.jpg
        numbers = ''.join(c for c in frame_file if c.isdigit())
        if numbers:
            return int(numbers)
        
        return None
    except (IndexError, ValueError):
        return None

def annotate_sequence(sequence_path, gt_path, output_video_path, fps=30, debug=False):
    """
    Annotate frames from a sequence using ground truth data and create a video.
    
    Args:
        sequence_path: Directory containing sequence frames
        gt_path: Path to ground truth CSV file
        output_video_path: Path to save the output video
        fps: Frames per second for the output video
        debug: Print debug information
    """
    # Load ground truth data with improved handling
    ground_truth = load_ground_truth(gt_path)
    if ground_truth is None or ground_truth.empty:
        print(f"Warning: No usable ground truth data found in {gt_path}")
        if debug:
            print("Creating an empty ground truth dataframe")
        ground_truth = pd.DataFrame(columns=['frame_number', 'track_object_id', 'bbox_x', 'bbox_y', 
                                       'bbox_w', 'bbox_h', 'confidence_score', 'class', 'visibility'])
    elif debug:
        print(f"Loaded ground truth with {len(ground_truth)} annotations")
        print(f"First few entries:\n{ground_truth.head()}")
        print(f"Frame numbers in GT: {sorted(ground_truth['frame_number'].unique())[:10]}...")
    
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
    
    # Create a debug frame to verify annotations if requested
    if debug:
        debug_frame_path = os.path.join(os.path.dirname(output_video_path), f"{os.path.basename(output_video_path)}_debug.jpg")
    
    # Sample frame numbers for debugging
    sample_frames = []
    annotation_count = 0
    
    # Process each frame
    print(f"Processing {len(frames)} frames from {sequence_path}")
    for frame_file in tqdm(frames, desc="Annotating Frames"):
        # Extract frame number from filename
        frame_number = extract_frame_number(frame_file)
        if frame_number is None:
            print(f"Warning: Could not extract frame number from {frame_file}, using file index")
            frame_number = frames.index(frame_file) + 1
        
        if len(sample_frames) < 5:
            sample_frames.append(frame_number)
        
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
            annotation_count += 1
            track_id = int(row['track_object_id'])
            
            try:
                x = int(row['bbox_x'])
                y = int(row['bbox_y'])
                w = int(row['bbox_w'])
                h = int(row['bbox_h'])
                
                # Validate coordinates (don't try to draw outside the image)
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    if debug:
                        print(f"Warning: Bounding box {x},{y},{w},{h} outside image dimensions {width}x{height}")
                    
                    # Clip to image boundaries
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    w = max(1, min(w, width - x))
                    h = max(1, min(h, height - y))
                
                # Assign a unique color to each object ID
                if track_id not in colors:
                    colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                
                # Get optional attributes
                confidence = row.get('confidence_score', 1.0) if 'confidence_score' in row else 1.0
                class_id = row.get('class', 1) if 'class' in row else 1
                visibility = row.get('visibility', 1.0) if 'visibility' in row else 1.0
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[track_id], 2)
                
                # Put label text
                label = f"ID:{track_id} C:{class_id}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[track_id], 2)
                
                # Save debug frame if it's one of the first frames
                if debug and frame_number == sample_frames[0]:
                    cv2.imwrite(debug_frame_path, frame)
                    
            except Exception as e:
                print(f"Error drawing annotation: {e}")
                print(f"Row data: {row}")
        
        # Add frame number to top left corner
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    
    # Print debug information
    if debug or annotation_count == 0:
        print(f"Total annotations drawn: {annotation_count}")
        print(f"Sample frame numbers: {sample_frames}")
        unique_frames_with_data = ground_truth['frame_number'].unique()
        print(f"Unique frames with data: {len(unique_frames_with_data)}")
        if len(unique_frames_with_data) > 0:
            print(f"First few frame numbers with data: {sorted(unique_frames_with_data)[:5]}")
    
    print(f"Video saved to {output_video_path}")

def process_coco_dataset(coco_json_path, coco_image_dir, gt_dir, output_dir, fps=30, debug=False):
    """
    Process a COCO dataset, annotate sequences and create videos.
    
    Args:
        coco_json_path: Path to COCO JSON file
        coco_image_dir: Directory containing image sequences
        gt_dir: Directory containing ground truth files
        output_dir: Directory to save output videos
        fps: Frames per second for output videos
        debug: Print debug information
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
    
    if debug:
        print(f"Found {len(videos)} video sequences in COCO data")
    
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
        
        # If not found with exact match, try partial match (for more flexibility)
        if not sequence_dir:
            for root, dirs, _ in os.walk(coco_image_dir):
                for d in dirs:
                    if video_id_str in d:
                        sequence_dir = os.path.join(root, d)
                        print(f"Found partial match directory: {d}")
                        break
                if sequence_dir:
                    break
        
        if not sequence_dir:
            print(f"Warning: Could not find sequence directory for video ID {video_id}")
            continue
        
        # Find corresponding ground truth file (try multiple common formats)
        gt_file = None
        gt_patterns = [
            f"{video_id_str}.csv", 
            f"{video_id_str}.txt", 
            f"gt_{video_id_str}.csv", 
            f"gt_{video_id_str}.txt",
            f"gt.txt",  # Common format in MOT datasets
            f"gt.csv"
        ]
        
        # First try exact matches
        for pattern in gt_patterns:
            potential_file = os.path.join(gt_dir, pattern)
            if os.path.exists(potential_file):
                gt_file = potential_file
                break
        
        # If not found, try looking in subdirectories
        if not gt_file:
            for root, _, files in os.walk(gt_dir):
                for file in files:
                    if any(pattern in file for pattern in gt_patterns):
                        gt_file = os.path.join(root, file)
                        break
                if gt_file:
                    break
        
        # If still not found, check if there's a 'gt' subdirectory within sequence_dir
        if not gt_file:
            potential_gt_dir = os.path.join(sequence_dir, 'gt')
            if os.path.exists(potential_gt_dir):
                for file in os.listdir(potential_gt_dir):
                    if file.endswith(('.txt', '.csv')):
                        gt_file = os.path.join(potential_gt_dir, file)
                        break
        
        if not gt_file:
            print(f"Warning: No ground truth file found for video ID {video_id}")
            if debug:
                print(f"Tried patterns: {gt_patterns}")
                print(f"In directory: {gt_dir}")
            
            # Create an empty GT file for visualization with just the frames
            gt_file = os.path.join(gt_dir, f"gt_{video_id_str}.csv")  # Use a placeholder path
            with open(gt_file, 'w') as f:
                f.write('')
        else:
            print(f"Using ground truth file: {gt_file}")
        
        # Define output video path
        output_video_path = os.path.join(output_dir, f"{video_id_str}.avi")
        
        # Annotate sequence and create video
        annotate_sequence(sequence_dir, gt_file, output_video_path, fps, debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO sequences to annotated videos")
    parser.add_argument("--coco_json", required=True, help="Path to COCO JSON file")
    parser.add_argument("--image_dir", required=True, help="Directory containing image sequences")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground truth files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output videos")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output videos")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    process_coco_dataset(args.coco_json, args.image_dir, args.gt_dir, args.output_dir, args.fps, args.debug)