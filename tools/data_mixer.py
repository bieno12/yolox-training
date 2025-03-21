#!/usr/bin/env python3
import json
import os
import argparse
import shutil
from pathlib import Path


def setup_directory_structure(base_dir, output_dir, datasets_to_mix, test_dataset):
    """Create the necessary directory structure and copy validation/test files."""
    print(f"Setting up directory structure in {output_dir}")
    
    # Create output directory and annotations subdirectory
    os.makedirs(os.path.join(base_dir, output_dir, "annotations"), exist_ok=True)
    
    # Base datasets mapping
    datasets = {
        f"{test_dataset}_train": os.path.join(base_dir, f"{test_dataset}/train")
    }
    
    # Add additional datasets based on what's being mixed
    if "crowdhuman_train" in datasets_to_mix:
        datasets["crowdhuman_train"] = os.path.join(base_dir, "crowdhuman/train")
        
    if "crowdhuman_val" in datasets_to_mix:
        datasets["crowdhuman_val"] = os.path.join(base_dir, "crowdhuman/val")
        
    if "cityscapes" in datasets_to_mix:
        datasets["cp_train"] = os.path.join(base_dir, "Cityscapes")
        
    if "ethz" in datasets_to_mix:
        datasets["ethz_train"] = os.path.join(base_dir, "ETHZ")
    
    # Create symlinks for each dataset
    for link_name, target in datasets.items():
        link_path = os.path.join(base_dir, output_dir, link_name)
        if not os.path.exists(link_path):
            try:
                os.makedirs(os.path.dirname(link_path), exist_ok=True)
                os.symlink(target, link_path)
                print(f"Created symlink: {link_path} -> {target}")
            except OSError as e:
                print(f"Warning: Could not create symlink {link_path}: {e}")


def process_mot_dataset(base_dir, dataset_name, output_dir):
    """Process MOT dataset and return its data."""
    json_path = os.path.join(base_dir, dataset_name, "annotations", "train.json")
    print(f"Processing {dataset_name} dataset from {json_path}")
    
    mot_json = json.load(open(json_path, 'r'))
    
    img_list = list()
    for img in mot_json['images']:
        img['file_name'] = f'{dataset_name}_train/' + img['file_name']
        img_list.append(img)
    
    ann_list = list()
    for ann in mot_json['annotations']:
        ann_list.append(ann)
    
    return {
        "images": img_list,
        "annotations": ann_list,
        "videos": mot_json['videos'],
        "categories": mot_json['categories']
    }


def process_additional_dataset(base_dir, dataset_info, max_img, max_ann, max_video):
    """Process additional datasets (CrowdHuman, ETHZ, Cityscapes)."""
    dataset_name = dataset_info["name"]
    json_path = os.path.join(base_dir, dataset_info["path"], "annotations", dataset_info["json"])
    folder_prefix = dataset_info["folder_prefix"]
    file_slice = dataset_info.get("file_slice", None)
    
    print(f"Processing {dataset_name} dataset from {json_path}")
    
    dataset_json = json.load(open(json_path, 'r'))
    
    img_list = []
    img_id_count = 0
    
    for img in dataset_json['images']:
        img_id_count += 1
        
        # Update file name with prefix and optional slicing
        if file_slice:
            img['file_name'] = f'{folder_prefix}/' + img['file_name'][file_slice:]
        else:
            img['file_name'] = f'{folder_prefix}/' + img['file_name']
        
        img['frame_id'] = img_id_count
        img['prev_image_id'] = img['id'] + max_img
        img['next_image_id'] = img['id'] + max_img
        img['id'] = img['id'] + max_img
        img['video_id'] = max_video
        img_list.append(img)
    
    ann_list = []
    for ann in dataset_json['annotations']:
        ann['id'] = ann['id'] + max_ann
        ann['image_id'] = ann['image_id'] + max_img
        ann_list.append(ann)
    
    video_info = {
        'id': max_video,
        'file_name': dataset_info["video_name"]
    }
    
    return {
        "images": img_list,
        "annotations": ann_list,
        "video": video_info
    }


def copy_test_files(base_dir, source_dataset, output_dir):
    """Copy test and validation files from source dataset."""
    source_path = os.path.join(base_dir, source_dataset, "annotations")
    dest_path = os.path.join(base_dir, output_dir, "annotations")
    
    test_files = {
        "val_half.json": "val_half.json",
        "test.json": "test.json"
    }
    
    for src, dst in test_files.items():
        src_file = os.path.join(source_path, src)
        dst_file = os.path.join(dest_path, dst)
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")
        else:
            print(f"Warning: Source file {src_file} not found")


def mix_datasets(base_dir, output_dir, datasets_to_mix, test_dataset):
    """Mix the specified datasets and save the result."""
    # Setup directory structure
    setup_directory_structure(base_dir, output_dir, datasets_to_mix, test_dataset)
    
    # Copy test files from the selected test dataset
    copy_test_files(base_dir, test_dataset, output_dir)
    
    # Process the primary MOT dataset (either mot or mot20)
    base_data = process_mot_dataset(base_dir, test_dataset, output_dir)
    img_list = base_data["images"]
    ann_list = base_data["annotations"]
    video_list = base_data["videos"]
    category_list = base_data["categories"]
    
    print(f"Processed {test_dataset}")
    
    # Define additional datasets with their parameters
    additional_datasets = []
    
    if "crowdhuman_train" in datasets_to_mix:
        additional_datasets.append({
            "name": "CrowdHuman Train",
            "path": "crowdhuman",
            "json": "train.json",
            "folder_prefix": "crowdhuman_train",
            "video_name": "crowdhuman_train",
            "max_img": 10000,
            "max_ann": 2000000,
            "max_video": 10
        })
    
    if "crowdhuman_val" in datasets_to_mix:
        additional_datasets.append({
            "name": "CrowdHuman Val",
            "path": "crowdhuman",
            "json": "val.json",
            "folder_prefix": "crowdhuman_val",
            "video_name": "crowdhuman_val",
            "max_img": 30000,
            "max_ann": 10000000,
            "max_video": 11
        })
    
    if "ethz" in datasets_to_mix:
        additional_datasets.append({
            "name": "ETHZ",
            "path": "ETHZ",
            "json": "train.json",
            "folder_prefix": "ethz_train",
            "file_slice": 5,
            "video_name": "ethz",
            "max_img": 40000,
            "max_ann": 20000000,
            "max_video": 12
        })
    
    if "cityscapes" in datasets_to_mix:
        additional_datasets.append({
            "name": "Cityscapes",
            "path": "Cityscapes",
            "json": "train.json",
            "folder_prefix": "cp_train",
            "file_slice": 11,
            "video_name": "cityperson",
            "max_img": 50000,
            "max_ann": 25000000,
            "max_video": 13
        })
    
    # Process each additional dataset
    for dataset_info in additional_datasets:
        dataset_data = process_additional_dataset(
            base_dir,
            dataset_info,
            dataset_info["max_img"],
            dataset_info["max_ann"],
            dataset_info["max_video"]
        )
        
        # Add the processed data to our lists
        img_list.extend(dataset_data["images"])
        ann_list.extend(dataset_data["annotations"])
        video_list.append(dataset_data["video"])
        
        print(f"Processed {dataset_info['name']}")
    
    # Create the final mixed JSON
    mix_json = {
        'images': img_list,
        'annotations': ann_list,
        'videos': video_list,
        'categories': category_list
    }
    
    # Save the mixed JSON
    output_path = os.path.join(base_dir, output_dir, "annotations", "train.json")
    json.dump(mix_json, open(output_path, 'w'))
    print(f"Created mixed dataset at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mix multiple detection datasets for training.")
    
    parser.add_argument("--base-dir", default="datasets", help="Base directory containing all datasets")
    parser.add_argument("--output-dir", default="mix_det", help="Output directory for the mixed dataset")
    parser.add_argument("--test-dataset", choices=["mot", "mot20"], default="mot", help="Which dataset to use for validation and test")
    
    parser.add_argument("--include", nargs="+", default=[],
                        choices=["crowdhuman_train", "crowdhuman_val", "ethz", "cityscapes"],
                        help="Datasets to include in the mix")
    
    args = parser.parse_args()
    
    print(f"Starting dataset mixing with the following configuration:")
    print(f"  Base directory: {args.base_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Test dataset: {args.test_dataset}")
    print(f"  Datasets to include: {', '.join(args.include)}")
    
    mix_datasets(args.base_dir, args.output_dir, args.include, args.test_dataset)
    
    print("Dataset mixing completed successfully!")


if __name__ == "__main__":
    main()