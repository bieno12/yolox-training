#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

def create_directory(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def coco_to_yolo_format(x, y, width, height, img_width, img_height):
    """
    Convert COCO format (x, y, width, height) to YOLO format (x_center, y_center, width, height) normalized.
    COCO: (x, y) is the top-left corner of the bounding box
    YOLO: (x_center, y_center) is the center of the bounding box, all values normalized
    """
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return x_center, y_center, norm_width, norm_height

def process_coco_annotations(coco_file, dataset_dir, output_dir, split_name):
    """Process COCO format annotations and convert to YOLO."""
    print(f"Processing {split_name} split...")
    
    # Create output directories
    yolo_labels_dir = os.path.join(output_dir, "labels", split_name)
    yolo_images_dir = os.path.join(output_dir, "images", split_name)
    create_directory(yolo_labels_dir)
    create_directory(yolo_images_dir)
    
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category ID to index mapping (YOLO uses sequential class indices)
    category_mapping = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Create a name to class file
    names_file = os.path.join(output_dir, "data.names")
    if not os.path.exists(names_file):
        with open(names_file, 'w') as f:
            for cat in sorted(coco_data['categories'], key=lambda x: category_mapping[x['id']]):
                f.write(f"{cat['name']}\n")
    
    # Create a mapping from image ID to image details
    images_map = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Process each image
    skipped_count = 0
    for image_id, annotations in tqdm(image_annotations.items(), desc=f"Converting {split_name}"):
        image_info = images_map[image_id]
        img_width, img_height = image_info['width'], image_info['height']
        
        # Skip if either dimension is zero
        if img_width == 0 or img_height == 0:
            skipped_count += 1
            continue
        
        # Relative image path (from COCO annotations)
        rel_img_path = image_info['file_name']
        
        # Full path to source image
        src_img_path = os.path.join(dataset_dir, rel_img_path)
        
        # Extract just the filename without any directories
        img_filename = os.path.basename(rel_img_path)
        
        # Target symlink location
        dst_img_path = os.path.join(yolo_images_dir, img_filename)
        
        # Create symlink to the image
        if not os.path.exists(dst_img_path):
            # Check if source exists
            if not os.path.exists(src_img_path):
                print(f"Warning: Source image not found: {src_img_path}")
                continue
                
            try:
                # Create relative symlink for portability
                rel_path = os.path.relpath(src_img_path, os.path.dirname(dst_img_path))
                os.symlink(rel_path, dst_img_path)
            except OSError as e:
                print(f"Warning: Failed to create symlink for {img_filename}: {e}")
                continue
        
        # Create YOLO annotation file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_file_path = os.path.join(yolo_labels_dir, label_filename)
        
        with open(label_file_path, 'w') as f:
            for ann in annotations:
                if 'bbox' not in ann:
                    continue
                
                category_id = ann['category_id']
                yolo_class_id = category_mapping[category_id]
                
                # COCO bounding box format: [x, y, width, height]
                bbox = ann['bbox']
                if len(bbox) < 4:
                    continue
                    
                x, y, width, height = bbox
                
                # Convert to YOLO format
                x_center, y_center, norm_width, norm_height = coco_to_yolo_format(
                    x, y, width, height, img_width, img_height
                )
                
                # Write to file: class_id x_center y_center width height
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"Skipped {skipped_count} images with invalid dimensions")
    return len(image_annotations) - skipped_count

def create_data_yaml(output_dir, class_count, train_count, val_count, test_count):
    """Create a YAML file for YOLO training configuration."""
    yaml_path = os.path.join(output_dir, "data.yaml")
    
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        
        f.write(f"nc: {class_count}  # number of classes\n")
        f.write("names: [")
        
        # Read class names from data.names
        names_file = os.path.join(output_dir, "data.names")
        with open(names_file, 'r') as nf:
            class_names = [line.strip() for line in nf.readlines()]
            class_names_str = ", ".join(f"'{name}'" for name in class_names)
            f.write(f"{class_names_str}]\n\n")
            
        # Add dataset stats    
        f.write(f"# Dataset statistics\n")
        f.write(f"# Train: {train_count} images\n")
        f.write(f"# Validation: {val_count} images\n")
        f.write(f"# Test: {test_count} images\n")

def main():
    parser = argparse.ArgumentParser(description='Convert COCO annotations to YOLO format')
    parser.add_argument('--dataset_dir', required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', required=True, help='Path to output YOLO dataset')
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    
    # Create output directory
    create_directory(output_dir)
    
    # Define annotation files
    annotation_dir = os.path.join(dataset_dir, "annotations")
    train_json = os.path.join(annotation_dir, "train.json")
    val_json = os.path.join(annotation_dir, "val_half.json")
    test_json = os.path.join(annotation_dir, "test.json")
    
    # Check if annotation files exist
    for json_file, split_name in [(train_json, "train"), (val_json, "val"), (test_json, "test")]:
        if not os.path.exists(json_file):
            print(f"Warning: {split_name} annotation file not found: {json_file}")
    
    # Process each split
    train_count = 0
    val_count = 0
    test_count = 0
    
    if os.path.exists(train_json):
        train_count = process_coco_annotations(train_json, dataset_dir, output_dir, "train")
    
    if os.path.exists(val_json):
        val_count = process_coco_annotations(val_json, dataset_dir, output_dir, "val")
    
    if os.path.exists(test_json):
        test_count = process_coco_annotations(test_json, dataset_dir, output_dir, "test")
    
    # Get class count from names file
    names_file = os.path.join(output_dir, "data.names")
    class_count = 0
    if os.path.exists(names_file):
        with open(names_file, 'r') as f:
            class_count = len(f.readlines())
    
    # Create data.yaml for YOLOv5/YOLOv8 compatibility
    create_data_yaml(output_dir, class_count, train_count, val_count, test_count)
    
    print(f"\nConversion complete!")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images: {test_count}")
    print(f"Number of classes: {class_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()