import os
import numpy as np
import json
import cv2
import argparse

def process_split(data_path, out_path, split, half_video, sequence=None, percentile=1.0):
    out = {'images': [], 'annotations': [], 'videos': [],
           'categories': [{'id': 1, 'name': 'pedestrian'}]}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0
    tid_last = -1
    
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
        images = os.listdir(img_path)
        num_images = len([image for image in images if 'jpg' in image])
        
        if half_video:
            image_range = [0, num_images // 2]
        else:
            image_range = [0, int(num_images * (percentile)) - 1]

        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            img = cv2.imread(os.path.join(data_path, '{}/img1/{:06d}.jpg'.format(seq, i + 1)))
            height, width = img.shape[:2]
            image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                          'id': image_cnt + i + 1,
                          'frame_id': i + 1 - image_range[0],
                          'prev_image_id': image_cnt + i if i > 0 else -1,
                          'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                          'video_id': video_cnt,
                          'height': height, 'width': width}
            out['images'].append(image_info)
        print('{}: {} images processed'.format(seq, num_images))

        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        
        for i in range(anns.shape[0]):
            frame_id = int(anns[i][0])
            if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                continue
            track_id = int(anns[i][1])
            cat_id = int(anns[i][7])
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
                    'area': float(anns[i][4] * anns[i][5])}
            out['annotations'].append(ann)
        image_cnt += num_images
        print(tid_curr, tid_last)
    print('Processed {} split: {} images, {} annotations'.format(split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))

def main():
    parser = argparse.ArgumentParser(description="MOT16 Data Processing Tool")
    parser.add_argument('--split', type=str, choices=['train', 'test'], required=True, help="Dataset split to process")
    parser.add_argument('--half_video', action='store_true', help="Process only half of each video")
    parser.add_argument('--sequence', type=str, help="Process a specific sequence only")
    parser.add_argument('--percentile', type=float, default=1.0, help="Percentage of each video to process (default: 1.0)")
    parser.add_argument('--output', type=str, help="Custom name for the output annotation JSON file")
    
    args = parser.parse_args()
    
    data_path = os.path.join('data/tracking', args.split)
    out_filename = args.output if args.output else '{}.json'.format(args.split)
    out_path = os.path.join('data/tracking/annotations', out_filename)
    
    process_split(data_path, out_path, args.split, args.half_video, args.sequence, args.percentile)

if __name__ == '__main__':
    main()
