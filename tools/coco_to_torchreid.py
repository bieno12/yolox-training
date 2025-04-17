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
    parser.add_argument('--query-ratio', type=float, default=0.15, help='Ratio of test data to use as query')
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
            
            filtered_tracks[track_id] = sorted_detections
    
    print(f"Found {len(filtered_tracks)} identities with good visibility")
    return filtered_tracks, images_info


def split_dataset(tracks, train_ratio, query_ratio):
    """Split dataset into train, query, and gallery sets ensuring all query IDs appear in gallery."""
    # Get all track_ids
    track_ids = list(tracks.keys())
    random.shuffle(track_ids)
    
    # Calculate splits
    n_tracks = len(track_ids)
    n_train = int(n_tracks * train_ratio)
    
    train_ids = track_ids[:n_train]
    test_ids = track_ids[n_train:]
    
    # For test identities, all will be in both query and gallery
    # This ensures query identities appear in gallery
    query_ids = test_ids.copy()
    gallery_ids = test_ids.copy()
    
    splits = {
        'train': train_ids,
        'query': query_ids,
        'gallery': gallery_ids
    }
    
    print(f"Split dataset: {len(train_ids)} train IDs, {len(query_ids)} query IDs, {len(gallery_ids)} gallery IDs")
    return splits


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
    
    # Process each split
    split_metadata = {}
    
    for split_name, track_ids in splits.items():
        print(f"Processing {split_name} split...")
        metadata = []
        
        for track_id in tqdm(track_ids, desc=f"Processing {split_name}"):
            # Assign a sequential ID if not already assigned
            if track_id not in id_mapping:
                id_mapping[track_id] = next_id
                next_id += 1
            
            sequential_id = id_mapping[track_id]
            
            if split_name == 'train':
                # Process all detections for training identities
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
                    target_path = os.path.join(dataset_dir, 'images', split_name, target_name)
                    
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
                    relative_path = os.path.join('images', split_name, target_name)
                    metadata.append((relative_path, sequential_id, 0))  # Using camid=0 for all
            
            else:  # query or gallery
                # For test identities, split samples between query and gallery
                samples = tracks[track_id]
                num_samples = len(samples)
                
                if num_samples < 2:
                    # If only one sample, put a copy in both query and gallery
                    query_samples = 1
                    gallery_samples = 1
                else:
                    # Split samples approximately in half
                    query_samples = max(1, num_samples // 2)
                    gallery_samples = num_samples - query_samples
                
                # Determine which samples go to query and gallery
                if split_name == 'query':
                    # First half of samples go to query
                    target_samples = samples[:query_samples]
                else:  # gallery
                    # Second half of samples go to gallery
                    target_samples = samples[query_samples:]
                
                # Process the samples
                for i, detection in enumerate(target_samples):
                    image_id = detection['image_id']
                    file_name = detection['file_name']
                    bbox = detection['bbox']
                    
                    # Source image path
                    src_img_path = os.path.join(args.images_dir, file_name)
                    
                    if not os.path.exists(src_img_path):
                        print(f"Warning: Image {src_img_path} not found, skipping")
                        continue
                    
                    # Target file name and path
                    target_name = f"{sequential_id:04d}_c1s1_{i:06d}_{int(detection['visibility']*100):02d}.jpg"
                    target_path = os.path.join(dataset_dir, 'images', split_name, target_name)
                    
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
                    
                    # Add to metadata
                    relative_path = os.path.join('images', split_name, target_name)
                    metadata.append((relative_path, sequential_id, 0))
        
        split_metadata[split_name] = metadata
        
        # Save metadata for this split
        metadata_path = os.path.join(dataset_dir, 'splits', f'{split_name}.txt')
        with open(metadata_path, 'w') as f:
            for img_path, pid, camid in metadata:
                f.write(f'{img_path} {pid} {camid}\n')
        
        print(f"Saved {len(metadata)} entries to {metadata_path}")
    
    # Verify that all query identities appear in gallery
    query_pids = set(pid for _, pid, _ in split_metadata['query'])
    gallery_pids = set(pid for _, pid, _ in split_metadata['gallery'])
    missing_pids = query_pids - gallery_pids
    
    if missing_pids:
        print(f"Warning: {len(missing_pids)} query identities missing from gallery")
        print("Adding these missing identities to gallery...")
        
        # Add missing identities to gallery
        gallery_metadata = split_metadata['gallery']
        for pid in missing_pids:
            # Find samples for this PID from query set
            pid_samples = [(path, id, camid) for path, id, camid in split_metadata['query'] if id == pid]
            
            if pid_samples:
                # Use the first sample
                gallery_metadata.append(pid_samples[0])
                
                # Copy the image to gallery
                query_path = os.path.join(dataset_dir, pid_samples[0][0])
                gallery_path = query_path.replace('query', 'gallery')
                
                if os.path.exists(query_path):
                    target_dir = os.path.dirname(gallery_path)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    shutil.copy(query_path, gallery_path)
        
        # Save updated gallery metadata
        metadata_path = os.path.join(dataset_dir, 'splits', 'gallery.txt')
        with open(metadata_path, 'w') as f:
            for img_path, pid, camid in gallery_metadata:
                f.write(f'{img_path} {pid} {camid}\n')
        
        print(f"Updated gallery size: {len(gallery_metadata)}")
    
    # Create a report on dataset statistics
    create_dataset_report(tracks, id_mapping, dataset_dir, split_metadata)
    
    return dataset_dir


def create_dataset_report(tracks, id_mapping, dataset_dir, split_metadata):
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
        
        # Split statistics
        f.write("\nSplit statistics:\n")
        for split_name, metadata in split_metadata.items():
            num_samples = len(metadata)
            num_identities = len(set(pid for _, pid, _ in metadata))
            f.write(f"  {split_name}: {num_samples} samples, {num_identities} identities\n")
        
        # Check identity overlap
        query_pids = set(pid for _, pid, _ in split_metadata['query'])
        gallery_pids = set(pid for _, pid, _ in split_metadata['gallery'])
        common_pids = query_pids.intersection(gallery_pids)
        
        f.write(f"\nIdentity overlap: {len(common_pids)} identities appear in both query and gallery\n")
        f.write(f"All query identities in gallery: {len(query_pids - gallery_pids) == 0}\n")
    
    print(f"Dataset report created at {report_path}")


def analyze_dataset_split(dataset_dir):
    """Analyze the query and gallery identities to find any mismatch."""
    split_query_path = os.path.join(dataset_dir, 'splits', 'query.txt')
    split_gallery_path = os.path.join(dataset_dir, 'splits', 'gallery.txt')
    
    query_pids = set()
    gallery_pids = set()
    
    # Load query identities
    if os.path.exists(split_query_path):
        with open(split_query_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    query_pids.add(int(parts[1]))
    
    # Load gallery identities
    if os.path.exists(split_gallery_path):
        with open(split_gallery_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    gallery_pids.add(int(parts[1]))
    
    # Find missing identities
    missing_pids = query_pids - gallery_pids
    
    print(f"Query identities: {len(query_pids)}")
    print(f"Gallery identities: {len(gallery_pids)}")
    print(f"Missing identities: {len(missing_pids)}")
    if missing_pids:
        print(f"Missing PIDs: {sorted(list(missing_pids))[:10]}...")
    
    return len(missing_pids) == 0


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
    
    # Split dataset
    splits = split_dataset(tracks, args.train_ratio, args.query_ratio)
    
    # Create TorchReID dataset
    dataset_dir = create_torchreid_dataset(tracks, images_info, splits, args)
    
    # Analyze the dataset split
    all_identities_present = analyze_dataset_split(dataset_dir)
    
    print(f"Dataset created at: {dataset_dir}")
    if all_identities_present:
        print("All query identities are present in gallery - dataset is ready for evaluation.")
    else:
        print("Warning: Some query identities are missing from gallery. Use the `combineall=True` option in TorchReID's datamanager.")
    
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
        
        # Directly load data from split files
        train = self._process_split_file(os.path.join(self.dataset_dir, 'splits', 'train.txt'))
        query = self._process_split_file(os.path.join(self.dataset_dir, 'splits', 'query.txt'))
        gallery = self._process_split_file(os.path.join(self.dataset_dir, 'splits', 'gallery.txt'))
        
        # Debug information
        print(f"Train samples: {{len(train)}}")
        print(f"Query samples: {{len(query)}}")
        print(f"Gallery samples: {{len(gallery)}}")
        
        query_pids = set([item[1] for item in query])
        gallery_pids = set([item[1] for item in gallery])
        print(f"Query identities: {{len(query_pids)}}")
        print(f"Gallery identities: {{len(gallery_pids)}}")
        print(f"Missing identities: {{len(query_pids - gallery_pids)}}")
        
        super({args.dataset_name.capitalize()}Dataset, self).__init__(train, query, gallery, **kwargs)

    def _process_split_file(self, filepath):
        '''Process the split file.'''
        if not os.path.exists(filepath):
            print(f"Warning: Split file {{filepath}} does not exist")
            return []
            
        data = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            try:
                img_path, pid, camid = line.split()
                # Convert relative path to absolute
                img_path = os.path.join(self.dataset_dir, img_path)
                
                if not os.path.exists(img_path):
                    print(f"Warning: Image {{img_path}} does not exist")
                    continue
                    
                data.append((img_path, int(pid), int(camid)))
            except Exception as e:
                print(f"Error processing line {{line}}: {{e}}")
                continue
                
        return data

# Then register your dataset
import torchreid
torchreid.data.register_image_dataset('{args.dataset_name}', {args.dataset_name.capitalize()}Dataset)

# Create the data manager with combineall option
datamanager = torchreid.data.ImageDataManager(
    root="data",
    sources="{args.dataset_name}",
    targets="{args.dataset_name}",
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop'],
    combineall=True  # This ensures all query identities are present in gallery
)
    """)


if __name__ == "__main__":
    main()