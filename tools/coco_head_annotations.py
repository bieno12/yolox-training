#!/usr/bin/env python3
import json
import argparse
import copy
from typing import Dict, List, Any

def add_head_annotations(
    coco_data: Dict[str, Any], 
    head_height_ratio: float = 0.1,
    head_width_ratio: float = 1.0,
    heads_only: bool = False
) -> Dict[str, Any]:
    """
    Add head annotations to COCO format data based on existing pedestrian annotations.
    
    Args:
        coco_data: Original COCO format annotations
        head_height_ratio: The ratio of the original bounding box height to use for the head
        head_width_ratio: The ratio of the original bounding box width to use for the head
        heads_only: If True, only output head annotations
        
    Returns:
        Updated COCO data with added head annotations
    """
    # Create a new dataset to store the results
    result = copy.deepcopy(coco_data)
    
    # Create a head category if it doesn't exist
    head_category_id = None
    for category in result["categories"]:
        if category["name"] == "head":
            head_category_id = category["id"]
            break
    
    if head_category_id is None:
        # Find the maximum category ID to create a new unique ID
        max_category_id = max(cat["id"] for cat in result["categories"]) if result["categories"] else 0
        head_category_id = max_category_id + 1
        result["categories"].append({
            "id": head_category_id,
            "name": "head",
            "supercategory": "person"
        })
    
    # Calculate new annotations for heads
    max_annotation_id = max(ann["id"] for ann in result["annotations"]) if result["annotations"] else 0
    new_annotations = []
    head_annotations = []
    
    for annotation in coco_data["annotations"]:
        # Keep the original annotation if not heads_only
        if not heads_only:
            new_annotations.append(annotation)
        
        # Extract bounding box information
        # COCO bbox format: [x, y, width, height]
        x, y, width, height = annotation["bbox"]
        
        # Calculate head bounding box dimensions
        head_height = height * head_height_ratio
        head_width = width * head_width_ratio
        
        # Calculate the new center-aligned x coordinate if width is reduced
        x_offset = (width - head_width) / 2
        head_x = x + x_offset
        
        # Create the head bounding box
        head_bbox = [head_x, y, head_width, head_height]
        
        # Create a new annotation for the head
        new_annotation_id = max_annotation_id + 1
        max_annotation_id = new_annotation_id
        
        head_annotation = copy.deepcopy(annotation)
        head_annotation.update({
            "id": new_annotation_id,
            "bbox": head_bbox,
            "category_id": head_category_id,
            "area": head_bbox[2] * head_bbox[3]  # width * height
        })
        
        head_annotations.append(head_annotation)
    
    # Update the annotations
    if heads_only:
        result["annotations"] = head_annotations
    else:
        result["annotations"] = new_annotations + head_annotations
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Add head annotations to COCO format pedestrian annotations")
    parser.add_argument("input_file", help="Input COCO JSON file")
    parser.add_argument("output_file", help="Output COCO JSON file with added head annotations")
    parser.add_argument(
        "--head-height-ratio", 
        type=float, 
        default=0.1, 
        help="Ratio of original bounding box height to use for head (default: 0.1)"
    )
    parser.add_argument(
        "--head-width-ratio", 
        type=float, 
        default=1.0, 
        help="Ratio of original bounding box width to use for head (default: 1.0)"
    )
    parser.add_argument(
        "--heads-only", 
        action="store_true", 
        help="Output only the head annotations"
    )
    
    args = parser.parse_args()
    
    # Read the input COCO file
    with open(args.input_file, 'r') as f:
        coco_data = json.load(f)
    
    # Process the annotations
    result = add_head_annotations(
        coco_data, 
        head_height_ratio=args.head_height_ratio,
        head_width_ratio=args.head_width_ratio,
        heads_only=args.heads_only
    )
    
    # Write the output file
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Print summary
    original_count = len(coco_data["annotations"])
    head_count = len(result["annotations"]) - (0 if args.heads_only else original_count)
    
    print(f"Added {head_count} head annotations.")
    print(f"Head dimensions: {args.head_height_ratio*100:.1f}% height, {args.head_width_ratio*100:.1f}% width")
    print(f"Output written to {args.output_file}")
    
    if args.heads_only:
        print(f"Only head annotations included in output (original annotations removed).")
    else:
        print(f"Output contains {len(result['annotations'])} total annotations.")

if __name__ == "__main__":
    main()