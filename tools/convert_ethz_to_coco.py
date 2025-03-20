import os
import numpy as np
import json
from PIL import Image
import argparse
import random
import tqdm

def load_paths(data_path):
    """Load image and label paths from a text file."""
    with open(data_path, 'r') as file:
        img_files = file.readlines()
        img_files = [x.replace('\n', '') for x in img_files]
        img_files = list(filter(lambda x: len(x) > 0, img_files))
    label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt') for x in img_files]
    return img_files, label_files

def process_split(img_paths, label_paths, out_path, data_root, start_percentile=0.0, end_percentile=1.0, 
                  sample_rate=1, random_seed=None, max_samples=None):
    """Process the dataset and create JSON annotations with flexible sampling options."""
    out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
    
    # Apply percentile filtering
    total_images = len(img_paths)
    start_idx = int(total_images * start_percentile)
    end_idx = int(total_images * end_percentile)
    
    # Select the subset of images based on percentiles
    selected_indices = list(range(start_idx, end_idx))
    
    # Apply sampling rate
    selected_indices = selected_indices[::sample_rate]
    
    # Apply random sampling if specified
    if random_seed is not None:
        random.seed(random_seed)
        if max_samples and max_samples < len(selected_indices):
            selected_indices = random.sample(selected_indices, max_samples)
    elif max_samples and max_samples < len(selected_indices):
        selected_indices = selected_indices[:max_samples]
    
    # Process only the selected images
    image_cnt = 0
    ann_cnt = 0
    processed_images = 0
    skipped_images = 0
    
    # Show progress bar
    print(f"Processing {len(selected_indices)} images...")
    
    # Setup progress bar
    pbar = tqdm.tqdm(total=len(selected_indices), unit="img")
    
    for idx in selected_indices:
        if idx >= len(img_paths):
            skipped_images += 1
            pbar.update(1)
            continue
            
        img_path = img_paths[idx]
        label_path = label_paths[idx]
        
        image_cnt += 1
        processed_images += 1
        
        # Get image dimensions
        try:
            im = Image.open(os.path.join(data_root, img_path))
            image_info = {
                'file_name': img_path, 
                'id': image_cnt,
                'height': im.size[1], 
                'width': im.size[0]
            }
            out['images'].append(image_info)
            
            # Load and process labels
            if os.path.isfile(os.path.join(data_root, label_path)):
                try:
                    labels0 = np.loadtxt(os.path.join(data_root, label_path), dtype=np.float32).reshape(-1, 6)
                    
                    if len(labels0.shape) == 1 and labels0.shape[0] == 6:  # Single detection
                        labels0 = labels0.reshape(1, 6)
                        
                    if len(labels0) > 0:  # Check if the array is not empty
                        # Normalized xywh to pixel xyxy format
                        labels = labels0.copy()
                        labels[:, 2] = image_info['width'] * (labels0[:, 2] - labels0[:, 4] / 2)
                        labels[:, 3] = image_info['height'] * (labels0[:, 3] - labels0[:, 5] / 2)
                        labels[:, 4] = image_info['width'] * labels0[:, 4]
                        labels[:, 5] = image_info['height'] * labels0[:, 5]
                        
                        # Add annotations
                        for i in range(len(labels)):
                            ann_cnt += 1
                            fbox = labels[i, 2:6].tolist()
                            ann = {
                                'id': ann_cnt,
                                'category_id': 1,
                                'image_id': image_cnt,
                                'track_id': int(labels0[i, 1]) if labels0[i, 1] >= 0 else -1,  # Use track ID if available
                                'bbox': fbox,
                                'area': fbox[2] * fbox[3],
                                'iscrowd': 0
                            }
                            out['annotations'].append(ann)
                except Exception as e:
                    print(f"\nError processing label file {label_path}: {e}")
            else:
                pbar.write(f"Warning: Label file not found: {label_path}")
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            skipped_images += 1
        
        # Update progress bar with additional stats
        pbar.set_postfix({"annotations": ann_cnt, "skipped": skipped_images})
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    print(f'\nProcessed {processed_images} images out of {total_images} total images')
    print(f'Created annotations for {len(out["images"])} images and {len(out["annotations"])} objects')
    if skipped_images > 0:
        print(f'Skipped {skipped_images} images due to errors')
    
    # Save the output JSON file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f'Saving annotations to {out_path}...')
    json.dump(out, open(out_path, 'w'))
    print(f'Done!')

def main():
    parser = argparse.ArgumentParser(description="ETHZ Dataset Processing Tool")
    parser.add_argument('--data_root', type=str, default='datasets',
                        help="Root directory of the dataset (default: 'datasets')")
    parser.add_argument('--data_file', type=str, default='data_path/eth.train', 
                        help="Path to the text file containing image paths, relative to data_root")

    parser.add_argument('--output_name', type=str, default='train.json',
                        help="Name of the output JSON file (default: 'train.json')")
    parser.add_argument('--start_percentile', type=float, default=0.0,
                        help="Starting percentile of dataset to process (default: 0.0)")
    parser.add_argument('--end_percentile', type=float, default=1.0,
                        help="Ending percentile of dataset to process (default: 1.0)")
    parser.add_argument('--sample_rate', type=int, default=1,
                        help="Process every Nth image (default: 1, which processes all images)")
    parser.add_argument('--random_seed', type=int,
                        help="Random seed for sampling (default: None, which means no random sampling)")
    parser.add_argument('--max_samples', type=int,
                        help="Maximum number of samples to include (default: None, which means no limit)")
    
    args = parser.parse_args()
    
    # Validate sample_rate
    if args.sample_rate < 1:
        print("Warning: Sample rate must be at least 1. Setting to default value of 1.")
        args.sample_rate = 1
    
    # Determine data paths
    data_root = args.data_root
    data_file_path = os.path.join(data_root, args.data_file)
    
    # Determine output path

    output_dir = os.path.join(data_root, 'ETHZ', 'annotations')
    
    out_path = os.path.join(output_dir, args.output_name)
    
    # Display configuration
    print("Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Data file: {data_file_path}")
    print(f"  Output path: {out_path}")
    print(f"  Start percentile: {args.start_percentile}")
    print(f"  End percentile: {args.end_percentile}")
    print(f"  Sample rate: {args.sample_rate}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Max samples: {args.max_samples}")
    
    # Check if data file exists
    if not os.path.isfile(data_file_path):
        print(f"Error: Data file not found: {data_file_path}")
        return
    
    # Load image and label paths
    print("Loading image and label paths...")
    img_paths, label_paths = load_paths(data_file_path)
    print(f"Found {len(img_paths)} images in data file")
    
    # Process the dataset with the specified parameters
    process_split(
        img_paths, 
        label_paths, 
        out_path,
        data_root,
        start_percentile=args.start_percentile,
        end_percentile=args.end_percentile,
        sample_rate=args.sample_rate,
        random_seed=args.random_seed,
        max_samples=args.max_samples
    )

if __name__ == '__main__':
    main()