#!/usr/bin/env python3
import os
import json
import argparse
import shutil
from collections import defaultdict
import random
from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO format MOT/MOT20 to TorchReID format')
    parser.add_argument('--coco-json', type=str, required=True, help='Path to COCO JSON annotation file')
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing the source images')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for TorchReID dataset')
    parser.add_argument('--dataset-name', type=str, default='mot_custom', help='Name for the dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of data to use for training')
    parser.add_argument('--min-bbox-size', type=int, default=100, help='Minimum bbox height to include')
    parser.add_argument('--min-samples', type=int, default=2, help='Minimum samples per identity to include')
    parser.add_argument('--max-overlap', type=float, default=0.3, help='Maximum allowed IoU overlap with other boxes')
    parser.add_argument('--overlap-threshold', type=float, default=0.5, help='IoU threshold for filtering by visibility')
    parser.add_argument('--visibility-ratio', type=float, default=0.6, help='Minimum ratio of samples with good visibility')
    parser.add_argument('--copy-images', action='store_true', help='Copy images instead of creating crops')
    return parser.parse_args()


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x, y, width, height] format."""
    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area


def process_coco_annotations(coco_json_path, min_bbox_size, min_samples, max_overlap, overlap_threshold, visibility_ratio):
    """Extract relevant information from COCO annotations with visibility analysis."""
    print(f"Processing COCO annotations from {coco_json_path}")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    images_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id for efficient overlap checking
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        if 'bbox' in ann:
            image_annotations[ann['image_id']].append(ann)
    
    # Group annotations by track_id with visibility analysis
    track_annotations = defaultdict(list)
    
    # First pass: collect all detections by track_id
    all_track_detections = defaultdict(list)
    
    for ann in tqdm(coco_data['annotations'], desc="Collecting annotations"):
        if 'track_id' not in ann:
            continue
            
        track_id = ann['track_id']
        image_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, width, height]
        
        # Filter out small detections
        if bbox[3] < min_bbox_size:  # Height is too small
            continue
            
        image_info = images_info[image_id]
        file_name = image_info['file_name']
        
        all_track_detections[track_id].append({
            'image_id': image_id,
            'file_name': file_name,
            'bbox': bbox,
            'annotation_id': ann['id']
        })
    
    # Second pass: calculate visibility for each detection
    print("Calculating visibility for detections...")
    for track_id, detections in tqdm(all_track_detections.items(), desc="Analyzing visibility"):
        for detection in detections:
            image_id = detection['image_id']
            bbox = detection['bbox']
            annotation_id = detection['annotation_id']
            
            # Get all other boxes in this image
            other_boxes = [
                ann['bbox'] for ann in image_annotations[image_id] 
                if ann['id'] != annotation_id
            ]
            
            # Calculate max overlap with any other box
            max_iou = 0.0
            if other_boxes:
                max_iou = max(calculate_iou(bbox, other_box) for other_box in other_boxes)
            
            # Add visibility information
            detection['visibility'] = 1.0 - max_iou
            detection['good_visibility'] = max_iou <= max_overlap
            
            # Add to track annotations if it has good visibility or is below threshold
            if max_iou <= overlap_threshold:
                track_annotations[track_id].append(detection)
    
    # Filter tracks by minimum samples and visibility ratio
    filtered_tracks = {}
    
    for track_id, detections in all_track_detections.items():
        # Count detections with good visibility
        good_visibility_count = sum(1 for det in detections if det['good_visibility'])
        total_detections = len(detections)
        
        # Check if this track meets our criteria
        if (total_detections >= min_samples and 
            good_visibility_count / total_detections >= visibility_ratio):
            
            # Sort detections by visibility (best first)
            sorted_detections = sorted(track_annotations[track_id], 
                                      key=lambda x: x['visibility'], 
                                      reverse=True)
            
            # Only include tracks with at least two detections after filtering
            if len(sorted_detections) >= min_samples:
                filtered_tracks[track_id] = sorted_detections
    
    print(f"Found {len(filtered_tracks)} identities with good visibility")
    return filtered_tracks, images_info


def split_dataset(tracks, train_ratio):
    """Split dataset into train and test sets with test IDs used for both query and gallery."""
    # Get all track_ids
    track_ids = list(tracks.keys())
    random.shuffle(track_ids)
    
    # Calculate splits
    n_tracks = len(track_ids)
    n_train = int(n_tracks * train_ratio)
    
    train_ids = track_ids[:n_train]
    test_ids = track_ids[n_train:]
    
    # Ensure test IDs have at least 2 samples each (for query and gallery)
    valid_test_ids = []
    for track_id in test_ids:
        if len(tracks[track_id]) >= 2:
            valid_test_ids.append(track_id)
    
    print(f"Split dataset: {len(train_ids)} train IDs, {len(valid_test_ids)} test IDs (for both query and gallery)")
    
    return {
        'train': train_ids,
        'test': valid_test_ids
    }


def create_torchreid_dataset(tracks, images_info, splits, args):
    """Create TorchReID dataset structure with proper sample distribution."""
    dataset_dir = os.path.join(args.output_dir, args.dataset_name)
    
    # Create directory structure
    for subset in ['train', 'query', 'gallery']:
        ensure_dir(os.path.join(dataset_dir, 'images', subset))
    
    ensure_dir(os.path.join(dataset_dir, 'splits'))
    
    # Track the mapping between original IDs and new sequential IDs
    id_mapping = {}
    next_id = 0
    
    # Process training identities
    print("Processing train split...")
    train_metadata = []
    
    for track_id in tqdm(splits['train'], desc="Processing train"):
        # Assign a sequential ID
        id_mapping[track_id] = next_id
        sequential_id = next_id
        next_id += 1
        
        # Process all detections for this training identity
        for i, detection in enumerate(tracks[track_id]):
            image_id = detection['image_id']
            file_name = detection['file_name']
            bbox = detection['bbox']  # [x, y, width, height]
            
            # Source image path
            src_img_path = os.path.join(args.images_dir, file_name)
            
            if not os.path.exists(src_img_path):
                print(f"Warning: Image {src_img_path} not found, skipping")
                continue
            
            # Target file name and path
            target_name = f"{sequential_id:04d}_c1s1_{i:06d}_{int(detection['visibility']*100):02d}.jpg"
            target_path = os.path.join(dataset_dir, 'images', 'train', target_name)
            
            # Copy the image or create a crop
            if args.copy_images:
                shutil.copy(src_img_path, target_path)
            else:
                # Create a crop
                try:
                    with Image.open(src_img_path) as img:
                        x, y, w, h = [int(v) for v in bbox]
                        crop = img.crop((x, y, x+w, y+h))
                        crop.save(target_path)
                except Exception as e:
                    print(f"Error processing {src_img_path}: {e}")
                    continue
            
            # Add to metadata (img_path, pid, camid)
            relative_path = os.path.join('images', 'train', target_name)
            train_metadata.append((relative_path, sequential_id, 0))  # Using camid=0 for all training samples
    
    # Save train metadata
    train_meta_path = os.path.join(dataset_dir, 'splits', 'train.txt')
    with open(train_meta_path, 'w') as f:
        for img_path, pid, camid in train_metadata:
            f.write(f'{img_path} {pid} {camid}\n')
    
    print(f"Saved {len(train_metadata)} entries to {train_meta_path}")
    
    # Process test identities (split between query and gallery)
    print("Processing test splits (query and gallery)...")
    query_metadata = []
    gallery_metadata = []
    
    for track_id in tqdm(splits['test'], desc="Processing test"):
        # Assign a sequential ID
        id_mapping[track_id] = next_id
        sequential_id = next_id
        next_id += 1
        
        # Get all detections for this identity
        detections = tracks[track_id]
        
        # Split detections between query and gallery
        # We need at least one sample in each
        split_point = max(1, len(detections) // 2)
        
        # Process query samples (first half) - using camid=0
        for i, detection in enumerate(detections[:split_point]):
            image_id = detection['image_id']
            file_name = detection['file_name']
            bbox = detection['bbox']
            
            src_img_path = os.path.join(args.images_dir, file_name)
            
            if not os.path.exists(src_img_path):
                continue
            
            target_name = f"{sequential_id:04d}_c1s1_{i:06d}_{int(detection['visibility']*100):02d}.jpg"
            target_path = os.path.join(dataset_dir, 'images', 'query', target_name)
            
            if args.copy_images:
                shutil.copy(src_img_path, target_path)
            else:
                try:
                    with Image.open(src_img_path) as img:
                        x, y, w, h = [int(v) for v in bbox]
                        crop = img.crop((x, y, x+w, y+h))
                        crop.save(target_path)
                except Exception as e:
                    print(f"Error processing {src_img_path}: {e}")
                    continue
            
            # Add to query metadata with camid=0
            relative_path = os.path.join('images', 'query', target_name)
            query_metadata.append((relative_path, sequential_id, 0))
        
        # Process gallery samples (second half) - using camid=1 for cross-camera matching
        for i, detection in enumerate(detections[split_point:]):
            image_id = detection['image_id']
            file_name = detection['file_name']
            bbox = detection['bbox']
            
            src_img_path = os.path.join(args.images_dir, file_name)
            
            if not os.path.exists(src_img_path):
                continue
            
            target_name = f"{sequential_id:04d}_c2s1_{i+split_point:06d}_{int(detection['visibility']*100):02d}.jpg"
            target_path = os.path.join(dataset_dir, 'images', 'gallery', target_name)
            
            if args.copy_images:
                shutil.copy(src_img_path, target_path)
            else:
                try:
                    with Image.open(src_img_path) as img:
                        x, y, w, h = [int(v) for v in bbox]
                        crop = img.crop((x, y, x+w, y+h))
                        crop.save(target_path)
                except Exception as e:
                    print(f"Error processing {src_img_path}: {e}")
                    continue
            
            # Add to gallery metadata with camid=1
            relative_path = os.path.join('images', 'gallery', target_name)
            gallery_metadata.append((relative_path, sequential_id, 1))
    
    # Save query metadata
    query_meta_path = os.path.join(dataset_dir, 'splits', 'query.txt')
    with open(query_meta_path, 'w') as f:
        for img_path, pid, camid in query_metadata:
            f.write(f'{img_path} {pid} {camid}\n')
    
    print(f"Saved {len(query_metadata)} entries to {query_meta_path}")
    
    # Save gallery metadata
    gallery_meta_path = os.path.join(dataset_dir, 'splits', 'gallery.txt')
    with open(gallery_meta_path, 'w') as f:
        for img_path, pid, camid in gallery_metadata:
            f.write(f'{img_path} {pid} {camid}\n')
    
    print(f"Saved {len(gallery_metadata)} entries to {gallery_meta_path}")
    
    # Verify that all query IDs appear in gallery
    query_ids = set(pid for _, pid, _ in query_metadata)
    gallery_ids = set(pid for _, pid, _ in gallery_metadata)
    missing_ids = query_ids - gallery_ids
    
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} query IDs are missing from gallery!")
        print(f"Missing IDs: {missing_ids}")
    else:
        print("SUCCESS: All query IDs appear in gallery.")
    
    # Create a report on dataset statistics
    create_dataset_report(tracks, id_mapping, dataset_dir)
    
    return dataset_dir


def create_dataset_report(tracks, id_mapping, dataset_dir):
    """Create a report with dataset statistics."""
    report_path = os.path.join(dataset_dir, 'dataset_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("TorchReID Dataset Statistics\n")
        f.write("===========================\n\n")
        
        f.write(f"Total identities: {len(id_mapping)}\n")
        
        # Calculate samples per identity
        samples_per_id = {id_mapping[track_id]: len(annotations) 
                          for track_id, annotations in tracks.items()}
        
        avg_samples = sum(samples_per_id.values()) / len(samples_per_id)
        min_samples = min(samples_per_id.values())
        max_samples = max(samples_per_id.values())
        
        f.write(f"Samples per identity: avg={avg_samples:.1f}, min={min_samples}, max={max_samples}\n")
        
        # Calculate visibility statistics
        all_visibility = [det['visibility'] for track in tracks.values() for det in track]
        avg_visibility = sum(all_visibility) / len(all_visibility)
        
        f.write(f"Average visibility: {avg_visibility:.2f}\n")
        
        visibility_ranges = {
            '0.8-1.0': sum(1 for v in all_visibility if 0.8 <= v <= 1.0),
            '0.6-0.8': sum(1 for v in all_visibility if 0.6 <= v < 0.8),
            '0.4-0.6': sum(1 for v in all_visibility if 0.4 <= v < 0.6),
            '0.2-0.4': sum(1 for v in all_visibility if 0.2 <= v < 0.4),
            '0.0-0.2': sum(1 for v in all_visibility if 0.0 <= v < 0.2)
        }
        
        f.write("\nVisibility distribution:\n")
        for range_name, count in visibility_ranges.items():
            percentage = count / len(all_visibility) * 100
            f.write(f"  {range_name}: {count} samples ({percentage:.1f}%)\n")
    
    print(f"Dataset report created at {report_path}")


def check_dataset_structure(dataset_dir):
    """Check the integrity of the created dataset."""
    print("\nChecking dataset structure...")
    # Load the split files
    query_pids = set()
    gallery_pids = set()
    query_camids = {}
    gallery_camids = {}
    
    with open(os.path.join(dataset_dir, 'splits', 'query.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pid = int(parts[1])
                camid = int(parts[2])
                query_pids.add(pid)
                query_camids[pid] = query_camids.get(pid, set())
                query_camids[pid].add(camid)
    
    with open(os.path.join(dataset_dir, 'splits', 'gallery.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pid = int(parts[1])
                camid = int(parts[2])
                gallery_pids.add(pid)
                gallery_camids[pid] = gallery_camids.get(pid, set())
                gallery_camids[pid].add(camid)
    
    print(f"Found {len(query_pids)} unique IDs in query")
    print(f"Found {len(gallery_pids)} unique IDs in gallery")
    
    missing = query_pids - gallery_pids
    if missing:
        print(f"ERROR: {len(missing)} query IDs missing from gallery: {missing}")
    else:
        print("SUCCESS: All query IDs appear in gallery!")
    
    # Check if camera IDs are different
    same_camera_count = 0
    for pid in query_pids:
        if pid in gallery_pids:
            q_camids = query_camids[pid]
            g_camids = gallery_camids[pid]
            if q_camids == g_camids and len(q_camids) == 1 and len(g_camids) == 1:
                same_camera_count += 1
    
    if same_camera_count > 0:
        print(f"WARNING: {same_camera_count} IDs have the same camera ID in both query and gallery")
    else:
        print("SUCCESS: All IDs have different camera IDs in query and gallery")


def main():
    args = parse_args()
    
    # Process annotations with visibility analysis
    tracks, images_info = process_coco_annotations(
        args.coco_json, 
        args.min_bbox_size, 
        args.min_samples,
        args.max_overlap,
        args.overlap_threshold,
        args.visibility_ratio
    )
    
    # Split dataset - simplified to ensure identities appear in both query and gallery
    splits = split_dataset(tracks, args.train_ratio)
    
    # Create TorchReID dataset with proper sample distribution and different camera IDs
    dataset_dir = create_torchreid_dataset(tracks, images_info, splits, args)
    
    # Check dataset structure to verify it's valid for evaluation
    check_dataset_structure(dataset_dir)
    
    print(f"Dataset created at: {dataset_dir}")
    
    # Instructions for using with TorchReID
    print("\nTo use this dataset with TorchReID, add the following code:")
    print(f"""
from torchreid.data import ImageDataset
import os
import glob

class {args.dataset_name.capitalize()}Dataset(ImageDataset):
    dataset_dir = '{args.dataset_name}'

    def __init__(self, root='', **kwargs):
        self.root = os.path.abspath(root)
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'images', 'train')
        self.query_dir = os.path.join(self.dataset_dir, 'images', 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'images', 'gallery')

        train = self._process_dir(self.train_dir, is_train=True)
        query = self._process_dir(self.query_dir, is_train=False)
        gallery = self._process_dir(self.gallery_dir, is_train=False)

        super({args.dataset_name.capitalize()}Dataset, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True):
        data = []
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            parts = img_name.split('_')
            pid = int(parts[0])
            camid = 0 if 'c1' in parts[1] else 1  # Extract camera ID from filename
            data.append((img_path, pid, camid))
            
        return data

# Then register your dataset
import torchreid
torchreid.data.register_image_dataset('{args.dataset_name}', {args.dataset_name.capitalize()}Dataset)
    """)
    
    print("\nTo test the dataset directly, run:")
    print("engine.run(test_only=True)")


if __name__ == "__main__":
    main()