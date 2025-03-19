import os
import numpy as np
import argparse
import configparser
import shutil
from pathlib import Path
import time
from tqdm import tqdm  # For progress bars
import random

def get_sequence_info(seq_path):
    seq_info_path = os.path.join(seq_path, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(seq_info_path)
    
    width = int(config['Sequence']['imWidth'])
    height = int(config['Sequence']['imHeight'])
    return width, height

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert MOT20 bbox (x, y, width, height) to YOLO format (x_center, y_center, width, height)
    All values normalized between 0 and 1
    """
    x, y, w, h = bbox
    
    # Calculate center points and normalize
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Normalize width and height
    w = w / img_width
    h = h / img_height
    
    return x_center, y_center, w, h


def process_split(data_path, out_dir, split, sequence=None,
                 start_percentile=0.0, end_percentile=1.0, sample_rate=1,
                 visibility_threshold=0.0, class_mapping=None):
                 
    start_time = time.time()
    
    if not os.path.exists(data_path):
        print(f"Warning: Data path for {split} split does not exist: {data_path}")
        return 0, 0  # Return 0 sequences and 0 images processed
    
    seqs = os.listdir(data_path)
    filtered_objects_count = 0  # Counter for filtered low visibility objects
    total_images = 0
    total_annotations = 0
    
    # Default class mapping if none provided (MOT20 class ID -> YOLO class ID)
    # MOT20 mostly uses class 1 for pedestrians
    if class_mapping is None:
        class_mapping = {1: 0}  # Map pedestrian class (1) to YOLO class 0
    
    # Create output directories
    images_dir = os.path.join(out_dir, split, 'images')
    labels_dir = os.path.join(out_dir, split, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    if sequence:
        seqs = [sequence] if sequence in seqs else []
    
    # Calculate total number of sequences for progress tracking
    valid_seqs = []
    for seq in sorted(seqs):
        if '.DS_Store' in seq:
            continue
        
        seq_path = os.path.join(data_path, seq)
        if not os.path.isdir(seq_path):
            continue
        
        # For test split, we may not have ground truth annotations
        if split == 'test':
            # Add sequence even if gt.txt doesn't exist, we'll handle it later
            valid_seqs.append(seq)
        else:
            # For train split, check for gt.txt
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            if not os.path.isfile(ann_path):
                continue
            valid_seqs.append(seq)
    
    print(f"Found {len(valid_seqs)} valid sequences for {split} split")
    
    # Process each sequence with a progress bar
    for seq_idx, seq in enumerate(valid_seqs):
        print(f"\n[{seq_idx+1}/{len(valid_seqs)}] Processing sequence: {seq} ({split} split)")
        
        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, 'img1')
        ann_path = os.path.join(seq_path, 'gt/gt.txt')
        has_annotations = os.path.isfile(ann_path)
        
        if not has_annotations and split == 'test':
            print(f"  - No ground truth annotations found for test sequence {seq}. Will process images only.")
        elif not has_annotations:
            print(f"  - No ground truth annotations found for sequence {seq}. Skipping.")
            continue
        
        width, height = get_sequence_info(seq_path)
        
        images = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
        num_images = len(images)
        
        start_idx = int(num_images * start_percentile)
        end_idx = int(num_images * end_percentile) - 1
        
        # Calculate number of images to process after applying sample rate
        processed_images_count = len(range(start_idx, end_idx + 1, sample_rate))
        print(f"  - Found {num_images} total images, will process {processed_images_count} images")
        
        # Load all annotations for this sequence if available
        annotations_by_frame = {}
        if has_annotations:
            try:
                print(f"  - Loading annotations from {ann_path}")
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                print(f"  - Loaded {len(anns)} annotations")
                
                # Create a dictionary to store annotations by frame
                for ann in anns:
                    frame_id = int(ann[0])
                    if frame_id not in annotations_by_frame:
                        annotations_by_frame[frame_id] = []
                    annotations_by_frame[frame_id].append(ann)
            except Exception as e:
                print(f"  - Error loading annotations: {e}. Processing without annotations.")
        
        sequence_images = 0
        sequence_annotations = 0
        
        # Process images with progress bar
        pbar = tqdm(total=processed_images_count, desc="  Converting", unit="img")
        
        for i in range(start_idx, end_idx + 1):
            # Apply sample rate
            if (i - start_idx) % sample_rate != 0:
                continue
            
            # Frame number in the dataset (1-indexed)
            frame_idx = i + 1
            
            # Source image path
            img_file = f"{frame_idx:06d}.jpg"
            src_img_path = os.path.join(img_path, img_file)
            
            # Get absolute path for symlink source
            abs_src_img_path = os.path.abspath(src_img_path)
            
            # Destination paths with sequence name to avoid conflicts
            dst_img_path = os.path.join(images_dir, f"{seq}_{img_file}")
            dst_label_path = os.path.join(labels_dir, f"{seq}_{Path(img_file).stem}.txt")
            
            # Create symbolic link instead of copying
            if os.path.exists(dst_img_path):
                os.remove(dst_img_path)  # Remove existing symlink if it exists
            os.symlink(abs_src_img_path, dst_img_path)
            sequence_images += 1
            
            # Create label file if annotations exist for this frame
            frame_ann_count = 0
            if has_annotations and frame_idx in annotations_by_frame:
                frame_anns = annotations_by_frame[frame_idx]
                valid_anns = []
                
                for ann in frame_anns:
                    # Skip if marked to ignore (confidence of 0)
                    if int(ann[6]) == 0:
                        continue
                    
                    # Check visibility if available
                    visibility = ann[8] if len(ann) > 8 else 1.0
                    
                    # Skip objects with visibility below threshold
                    if visibility < visibility_threshold:
                        filtered_objects_count += 1
                        continue
                    
                    # In MOT20, class 1 is typically pedestrian
                    # Map to YOLO class - default is 0 for pedestrians
                    yolo_class = class_mapping.get(1, 0)
                    
                    # Get bbox and convert to YOLO format
                    bbox = ann[2:6]
                    x_center, y_center, w, h = convert_bbox_to_yolo(bbox, width, height)
                    
                    # YOLO format: class_id center_x center_y width height
                    valid_anns.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                    sequence_annotations += 1
                    frame_ann_count += 1
                
                # Write annotations to label file
                with open(dst_label_path, 'w') as f:
                    f.write("\n".join(valid_anns))
            else:
                # Create empty label file
                open(dst_label_path, 'w').close()
            
            # Update progress bar with annotation count
            pbar.set_postfix({"annotations": frame_ann_count})
            pbar.update(1)
        
        pbar.close()
        
        print(f"  - Completed: {sequence_images} images, {sequence_annotations} annotations")
        total_images += sequence_images
        total_annotations += sequence_annotations
    
    elapsed_time = time.time() - start_time
    print(f"\nConversion Summary for {split} split:")
    print(f"  - Total sequences processed: {len(valid_seqs)}")
    print(f"  - Total images: {total_images}")
    print(f"  - Total annotations: {total_annotations}")
    print(f"  - Objects filtered due to low visibility: {filtered_objects_count}")
    print(f"  - Results saved to: {os.path.join(out_dir, split)}")
    print(f"  - Processing time: {elapsed_time:.2f} seconds")
    
    return len(valid_seqs), total_images

def create_train_val_split(out_dir, train_val_ratio):
    """
    Split the train data into train and val subsets
    """
    train_dir = os.path.join(out_dir, 'train')
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    
    val_dir = os.path.join(out_dir, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_labels_dir = os.path.join(val_dir, 'labels')
    
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
    
    # Determine which sequences we have
    sequences = set()
    for img_file in image_files:
        seq = img_file.split('_')[0]
        sequences.add(seq)
    
    # Convert to list and sort for reproducibility
    sequences = sorted(list(sequences))
    
    # Calculate number of sequences for validation
    val_count = max(1, int(len(sequences) * (1 - train_val_ratio)))
    
    # Randomly select sequences for validation
    random.seed(42)  # For reproducibility
    val_sequences = random.sample(sequences, val_count)
    
    print(f"\nCreating train/val split with ratio {train_val_ratio:.2f}:")
    print(f"  - Train sequences: {len(sequences) - val_count}")
    print(f"  - Val sequences: {val_count}")
    print(f"  - Val sequences: {', '.join(val_sequences)}")
    
    # Move selected sequences to validation folder
    val_image_count = 0
    val_annotation_count = 0
    
    for img_file in image_files:
        seq = img_file.split('_')[0]
        if seq in val_sequences:
            # Move image to val folder
            src_img_path = os.path.join(train_images_dir, img_file)
            dst_img_path = os.path.join(val_images_dir, img_file)
            
            # Move label to val folder
            label_file = img_file.replace('.jpg', '.txt')
            src_label_path = os.path.join(train_labels_dir, label_file)
            dst_label_path = os.path.join(val_labels_dir, label_file)
            
            # Check if destination files already exist
            if os.path.exists(dst_img_path):
                os.remove(dst_img_path)
            if os.path.exists(dst_label_path):
                os.remove(dst_label_path)
            
            # Create symlink for image if the source is a symlink
            if os.path.islink(src_img_path):
                target = os.readlink(src_img_path)
                os.symlink(target, dst_img_path)
            else:
                shutil.move(src_img_path, dst_img_path)
            
            # Move label file
            shutil.move(src_label_path, dst_label_path)
            
            val_image_count += 1
            
            # Count annotations in the label file
            with open(dst_label_path, 'r') as f:
                lines = f.readlines()
                val_annotation_count += len([line for line in lines if line.strip()])
    
    print(f"  - Moved {val_image_count} images and {val_annotation_count} annotations to validation set")
    return val_image_count, val_annotation_count

def create_dataset_yaml(out_dir, has_test=False, has_val=False):
    """
    Create dataset.yaml file based on available splits
    """
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    test_dir = os.path.join(out_dir, 'test')
    
    # Determine which splits exist and have data
    has_train = os.path.exists(train_dir) and len(os.listdir(os.path.join(train_dir, 'images'))) > 0
    if has_val:
        has_val = os.path.exists(val_dir) and len(os.listdir(os.path.join(val_dir, 'images'))) > 0
    if has_test:
        has_test = os.path.exists(test_dir) and len(os.listdir(os.path.join(test_dir, 'images'))) > 0
    
    yaml_content = f"path: {os.path.abspath(out_dir)}\n"
    
    if has_train:
        yaml_content += "train: train/images\n"
    
    if has_val:
        yaml_content += "val: val/images\n"
    elif has_train:  # If no val but has train, use train for validation too
        yaml_content += "val: train/images\n"
    
    if has_test:
        yaml_content += "test: test/images\n"
    
    # Always use class 0 for person
    yaml_content += "\nnc: 1\nnames: {0: 'person'}\n"
    
    yaml_path = os.path.join(out_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset configuration: {yaml_path}")
    print(f"  - Available splits: {'train' if has_train else ''}{', val' if has_val else ''}{', test' if has_test else ''}")
    
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="MOT20 to YOLO Format Converter")
    parser.add_argument('--data_path', type=str, default='data/tracking', help="Path to MOT20 dataset")
    parser.add_argument('--sequence', type=str, help="Process a specific sequence only")
    parser.add_argument('--start_percentile', type=float, default=0.0, help="Starting percentage of each video to process (default: 0.0)")
    parser.add_argument('--end_percentile', type=float, default=1.0, help="Ending percentage of each video to process (default: 1.0)")
    parser.add_argument('--sample_rate', type=int, default=1, help="Process every Nth frame (default: 1, which processes all frames)")
    parser.add_argument('--output_dir', type=str, default='data/mot20_yolo', help="Directory to save converted dataset")
    parser.add_argument('--visibility_threshold', type=float, default=0.0, 
                      help="Filter objects with visibility below this threshold (default: 0.0, meaning no filtering)")
    parser.add_argument('--train_val_ratio', type=float, default=0.8, 
                      help="Ratio of training data (default: 0.8, meaning 80%% train, 20%% validation)")
    parser.add_argument('--no_train_val_split', action='store_true', 
                      help="Do not split train data into train and validation sets")
    
    args = parser.parse_args()
    
    # Validate sample_rate
    if args.sample_rate < 1:
        print("Warning: Sample rate must be at least 1. Setting to default value of 1.")
        args.sample_rate = 1
    
    # Validate visibility threshold
    if args.visibility_threshold < 0.0 or args.visibility_threshold > 1.0:
        print("Warning: Visibility threshold must be between 0.0 and 1.0. Setting to default value of 0.0 (no filtering).")
        args.visibility_threshold = 0.0
    
    # Validate train_val_ratio
    if args.train_val_ratio <= 0.0 or args.train_val_ratio > 1.0:
        print("Warning: Train/val ratio must be between 0.0 and 1.0. Setting to default value of 0.8.")
        args.train_val_ratio = 0.8
    
    # Print configuration
    print("\nMOT20 to YOLO Converter")
    print("=======================")
    print(f"Data path: {args.data_path}")
    print(f"Sequence: {args.sequence if args.sequence else 'All'}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Frame range: {args.start_percentile*100:.1f}% to {args.end_percentile*100:.1f}%")
    print(f"Visibility threshold: {args.visibility_threshold}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train/Val ratio: {args.train_val_ratio if not args.no_train_val_split else 1.0}")
    print("=======================\n")
    
    # Create output directory
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Process train data
    train_seqs_count = 0
    train_images_count = 0
    
    train_data_path = os.path.join(args.data_path, 'train')
    if os.path.exists(train_data_path):
        print("\nProcessing train split...")
        train_seqs_count, train_images_count = process_split(
            train_data_path, out_dir, 'train', args.sequence, 
            args.start_percentile, args.end_percentile, args.sample_rate,
            args.visibility_threshold
        )
    else:
        print(f"\nTrain data path not found: {train_data_path}")
    
    # Process test data
    test_seqs_count = 0
    test_images_count = 0
    
    test_data_path = os.path.join(args.data_path, 'test')
    if os.path.exists(test_data_path):
        print("\nProcessing test split...")
        test_seqs_count, test_images_count = process_split(
            test_data_path, out_dir, 'test', args.sequence, 
            args.start_percentile, args.end_percentile, args.sample_rate,
            args.visibility_threshold
        )
    else:
        print(f"\nTest data path not found: {test_data_path}")
    
    # Create train/val split if requested and if train data exists
    val_images_count = 0
    has_val = False
    
    if train_seqs_count > 0 and not args.no_train_val_split and args.train_val_ratio < 1.0:
        _, val_images_count = create_train_val_split(out_dir, args.train_val_ratio)
        has_val = val_images_count > 0
    
    # Create dataset.yaml file
    create_dataset_yaml(out_dir, test_seqs_count > 0, has_val)
    
    # Final summary
    print("\nConversion Complete!")
    print(f"  - Train images: {train_images_count - val_images_count if has_val else train_images_count}")
    if has_val:
        print(f"  - Validation images: {val_images_count}")
    print(f"  - Test images: {test_images_count}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Dataset config: {os.path.join(args.output_dir, 'dataset.yaml')}")
    
    if train_seqs_count == 0 and test_seqs_count == 0:
        print("\nWarning: No sequences were processed. Please check your data paths.")

if __name__ == '__main__':
    main()