import os
import numpy as np
import json
import argparse
import configparser

def get_sequence_info(seq_path):
    seq_info_path = os.path.join(seq_path, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(seq_info_path)
    
    width = int(config['Sequence']['imWidth'])
    height = int(config['Sequence']['imHeight'])
    return width, height

def process_split(data_path, out_path, split, half_video, sequence=None, 
                 start_percentile=0.0, end_percentile=1.0, sample_rate=1, 
                 visibility_threshold=0.0):  # Added visibility threshold parameter
    out = {'images': [], 'annotations': [], 'videos': [],
           'categories': [{'id': 1, 'name': 'pedestrian'}]}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0
    tid_last = -1
    filtered_objects_count = 0  # Counter for filtered low visibility objects
    
    if sequence:
        seqs = [sequence] if sequence in seqs else []
    
    for seq in sorted(seqs):
        if '.DS_Store' in seq:
            continue
        video_cnt += 1  # video sequence number.
        out['videos'].append({'id': video_cnt, 'file_name': seq})
        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, 'img1')
        ann_path = os.path.join(seq_path, 'gt/gt.txt')
        
        width, height = get_sequence_info(seq_path)
        
        images = os.listdir(img_path)
        num_images = len([image for image in images if 'jpg' in image])
        
        start_idx = int(num_images * start_percentile)
        end_idx = int(num_images * end_percentile) - 1
        image_range = [start_idx, end_idx]
        
        valid_image_indices = []
        valid_image_count = 0
        
        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
                
            # Apply sample rate - only process frames at the specified interval
            if (i - image_range[0]) % sample_rate != 0:
                continue
                
            valid_image_indices.append(i)
            valid_image_count += 1
            
            # Calculate the correct previous and next image IDs based on sampled frames
            prev_image_id = -1
            if valid_image_count > 1:
                prev_image_id = image_cnt + valid_image_indices[-2] + 1
                
            next_image_id = -1  # Will be updated for previous frames once we know the next valid frame
            
            image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                          'id': image_cnt + i + 1,
                          'frame_id': valid_image_count,  # Sequential frame ID within the valid sampled frames
                          'prev_image_id': prev_image_id,
                          'next_image_id': next_image_id,  # Will be updated later
                          'video_id': video_cnt,
                          'height': height, 'width': width}
            out['images'].append(image_info)
        
        # Update next_image_id for all frames except the last one
        for j in range(len(out['images']) - valid_image_count, len(out['images']) - 1):
            out['images'][j]['next_image_id'] = out['images'][j + 1]['id']
        
        print('{}: {} images processed out of {} total (sample rate: {})'.format(
            seq, valid_image_count, num_images, sample_rate))

        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        
        for i in range(anns.shape[0]):
            frame_id = int(anns[i][0])
            if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                continue
                
            # Skip annotations for frames that were skipped due to sampling
            if (frame_id - 1 - image_range[0]) % sample_rate != 0:
                continue
                
            track_id = int(anns[i][1])
            cat_id = int(anns[i][7])
            
            # Check visibility (column 8 in MOT16 format contains visibility ratio)
            # MOT16 format: [frame_id, track_id, bb_left, bb_top, bb_width, bb_height, confidence, class_id, visibility]
            visibility = anns[i][8] if anns[i].shape[0] > 8 else 1.0
            
            # Skip objects with visibility below threshold
            if visibility < visibility_threshold:
                filtered_objects_count += 1
                continue
                
            ann_cnt += 1
            if not (int(anns[i][6]) == 1):  # whether ignore.
                continue
            if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                continue
            category_id = 1  # pedestrian(non-static)
            if not track_id == tid_last:
                tid_curr += 1
                tid_last = track_id
            
            ann = {'id': ann_cnt,
                   'category_id': category_id,
                   'image_id': image_cnt + frame_id,
                   'track_id': tid_curr,
                   'bbox': anns[i][2:6].tolist(),
                   'conf': float(anns[i][6]),
                   'iscrowd': 0,
                   'visibility': float(visibility),  # Add visibility to annotation
                   'area': float(anns[i][4] * anns[i][5])}
            out['annotations'].append(ann)
        
        image_cnt += num_images
        print(tid_curr, tid_last)
    
    print('Processed {} split: {} images, {} annotations, {} objects filtered due to low visibility (sample rate: {}, visibility threshold: {})'.format(
        split, len(out['images']), len(out['annotations']), filtered_objects_count, sample_rate, visibility_threshold))
    json.dump(out, open(out_path, 'w'))

def main():
    parser = argparse.ArgumentParser(description="MOT16 Data Processing Tool")
    parser.add_argument('--data_dir', type=str, default='data/tracking', help="dataset directory")
    parser.add_argument('--split', type=str, choices=['train', 'test'], required=True, help="Dataset split to process")
    parser.add_argument('--half_video', action='store_true', help="Process only half of each video")
    parser.add_argument('--sequence', type=str, help="Process a specific sequence only")
    parser.add_argument('--start_percentile', type=float, default=0.0, help="Starting percentage of each video to process (default: 0.0)")
    parser.add_argument('--end_percentile', type=float, default=1.0, help="Ending percentage of each video to process (default: 1.0)")
    parser.add_argument('--sample_rate', type=int, default=1, help="Process every Nth frame (default: 1, which processes all frames)")
    parser.add_argument('--output', type=str, help="Custom name for the output annotation JSON file")
    parser.add_argument('--visibility_threshold', type=float, default=0.0, 
                        help="Filter objects with visibility below this threshold (default: 0.0, meaning no filtering)")
    
    args = parser.parse_args()
    
    # Validate sample_rate
    if args.sample_rate < 1:
        print("Warning: Sample rate must be at least 1. Setting to default value of 1.")
        args.sample_rate = 1
    
    # Validate visibility threshold
    if args.visibility_threshold < 0.0 or args.visibility_threshold > 1.0:
        print("Warning: Visibility threshold must be between 0.0 and 1.0. Setting to default value of 0.0 (no filtering).")
        args.visibility_threshold = 0.0
    
    data_path = os.path.join(args.data_dir, args.split)
    out_filename = args.output if args.output else '{}.json'.format(args.split)
    os.makedirs(os.path.join(args.data_dir, 'annotations'), exist_ok=True)
    out_path = os.path.join(args.data_dir, 'annotations', out_filename)
    
    process_split(data_path, out_path, args.split, args.half_video, args.sequence, 
                 args.start_percentile, args.end_percentile, args.sample_rate,
                 args.visibility_threshold)

if __name__ == '__main__':
    main()