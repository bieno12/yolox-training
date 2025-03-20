import os
import numpy as np
import json
import argparse
from PIL import Image
import random

def load_func(fpath):
    """Load ODGT annotations from file."""
    print(f'Loading annotations from {fpath}')
    assert os.path.exists(fpath), f"File not found: {fpath}"
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records

def process_split(data_path, out_path, split, subset_percentage=1.0, 
                 random_seed=None, min_visibility=0.0):
    """Process CrowdHuman dataset split with flexible subsetting options."""
    
    # Initialize output structure
    out = {
        'images': [], 
        'annotations': [], 
        'categories': [{'id': 1, 'name': 'person'}]
    }
    
    # Load annotations
    ann_path = os.path.join(data_path, f'annotation_{split}.odgt')
    anns_data = load_func(ann_path)
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Process a random subset based on percentage
    if subset_percentage < 1.0:
        num_images = int(len(anns_data) * subset_percentage)
        selected_anns = random.sample(anns_data, num_images)
    else:
        selected_anns = anns_data
    
    # Process selected annotations
    image_cnt = 0
    ann_cnt = 0
    filtered_boxes = 0
    
    for ann_data in selected_anns:
        image_cnt += 1
        file_path = os.path.join(data_path, f'{split}', f"{ann_data['ID']}.jpg")
        
        # Check if image file exists
        if not os.path.exists(file_path):
            print(f"Warning: Image {file_path} not found, skipping")
            continue
        
        # Get image dimensions
        try:
            im = Image.open(file_path)
            image_info = {
                'file_name': f"{ann_data['ID']}.jpg", 
                'id': image_cnt,
                'height': im.size[1], 
                'width': im.size[0]
            }
            out['images'].append(image_info)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            continue
        
        # Process annotations for this image
        if split != 'test':
            anns = ann_data['gtboxes']
            for i in range(len(anns)):
                # Calculate visibility ratio if both boxes are present
                visibility_ratio = 0.0
                if 'vbox' in anns[i] and 'fbox' in anns[i]:
                    vbox = anns[i]['vbox']
                    fbox = anns[i]['fbox']
                    vis_area = vbox[2] * vbox[3]
                    full_area = fbox[2] * fbox[3]
                    if full_area > 0:
                        visibility_ratio = vis_area / full_area
                
                # Skip if below visibility threshold
                if visibility_ratio < min_visibility:
                    filtered_boxes += 1
                    continue
                
                ann_cnt += 1
                fbox = anns[i]['fbox']
                
                ann = {
                    'id': ann_cnt,
                    'category_id': 1,
                    'image_id': image_cnt,
                    'track_id': -1,
                    'bbox_vis': anns[i]['vbox'] if 'vbox' in anns[i] else fbox,
                    'bbox': fbox,
                    'visibility': visibility_ratio,
                    'area': fbox[2] * fbox[3],
                    'iscrowd': 1 if 'extra' in anns[i] and \
                                   'ignore' in anns[i]['extra'] and \
                                   anns[i]['extra']['ignore'] == 1 else 0
                }
                out['annotations'].append(ann)
    
    # Print statistics
    print(f'Processed {split} split: {len(out["images"])} images ({subset_percentage*100:.1f}% of original), '
          f'{len(out["annotations"])} annotations, {filtered_boxes} boxes filtered by visibility threshold')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save output
    json.dump(out, open(out_path, 'w'))
    print(f'Output saved to {out_path}')

def main():
    """Main function to parse arguments and process data."""
    parser = argparse.ArgumentParser(description="CrowdHuman Dataset Processing Tool")
    
    parser.add_argument('--data_path', type=str, default='datasets/crowdhuman/',
                        help="Base path to CrowdHuman dataset")
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                        required=True, help="Dataset split to process")
    parser.add_argument('--subset_percentage', type=float, default=1.0,
                        help="Percentage of images to include (0.0-1.0)")
    parser.add_argument('--output', type=str, 
                        help="Custom name for the output annotation JSON file")
    parser.add_argument('--random_seed', type=int,
                        help="Random seed for reproducible subsets")
    parser.add_argument('--min_visibility', type=float, default=0.0,
                        help="Filter boxes with visibility ratio below this threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.subset_percentage <= 0.0 or args.subset_percentage > 1.0:
        print("Warning: Subset percentage must be between 0.0 and 1.0. Setting to default value of 1.0.")
        args.subset_percentage = 1.0
    
    if args.min_visibility < 0.0 or args.min_visibility > 1.0:
        print("Warning: Visibility threshold must be between 0.0 and 1.0. Setting to default value of 0.0.")
        args.min_visibility = 0.0
    
    # Set output path
    out_dir = os.path.join(args.data_path, 'annotations')
    if args.output:
        out_filename = args.output
    else:
        subset_tag = f"_{int(args.subset_percentage*100)}pct" if args.subset_percentage < 1.0 else ""
        vis_tag = f"_vis{int(args.min_visibility*100)}" if args.min_visibility > 0.0 else ""
        out_filename = f"{args.split}{subset_tag}{vis_tag}.json"
    
    out_path = os.path.join(out_dir, out_filename)
    
    # Process the split
    process_split(args.data_path, out_path, args.split, 
                 args.subset_percentage, args.random_seed, 
                 args.min_visibility)

if __name__ == '__main__':
    main()