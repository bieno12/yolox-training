#!/usr/bin/env python
import pandas as pd
import cv2
import numpy as np
import torch
import torchreid
from torchreid.models import build_model
from torchreid.utils import load_pretrained_weights
from collections import OrderedDict
from pathlib import Path
import os
import pickle
import argparse
from tqdm import tqdm
import torchvision
from shapely.geometry import box
from shapely import ops

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        # path compression
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # union by rank
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank.setdefault(rx, 0) < self.rank.setdefault(ry, 0):
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

class EmbeddingComputer:
    def __init__(self, model_ckpt, grid_off, max_batch=1024):
        self.model = None
        self.model_ckpt = model_ckpt
        self.crop_size = (128, 384)
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.grid_off = grid_off
        self.max_batch = max_batch

        # Only used for the general ReID model (not FastReID)
        self.normalize = False

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def get_horizontal_split_patches(self, image, bbox, tag, idx, viz=False):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.shape[2:]

        bbox = np.array(bbox)
        bbox = bbox.astype(np.int32)
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
            # Faulty Patch Correction
            bbox[0] = np.clip(bbox[0], 0, None)
            bbox[1] = np.clip(bbox[1], 0, None)
            bbox[2] = np.clip(bbox[2], 0, image.shape[1])
            bbox[3] = np.clip(bbox[3], 0, image.shape[0])

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        ### TODO - Write a generalized split logic
        split_boxes = [
            [x1, y1, x1 + w, y1 + h / 3],
            [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
            [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
        ]

        split_boxes = np.array(split_boxes, dtype="int")
        patches = []
        for ix, patch_coords in enumerate(split_boxes):
            if isinstance(image, np.ndarray):
                im1 = image[patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2], :]

                if viz:
                    dirs = "./viz/{}/{}".format(tag.split(":")[0], tag.split(":")[1])
                    Path(dirs).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(dirs, "{}_{}.png".format(idx, ix)),
                        im1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255,
                    )
                patch = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, self.crop_size, interpolation=cv2.INTER_LINEAR)
                patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
                patch = patch.unsqueeze(0)
                patches.append(patch)
            else:
                im1 = image[:, :, patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2]]
                patch = torchvision.transforms.functional.resize(im1, (256, 128))
                patches.append(patch)

        patches = torch.cat(patches, dim=0)
        return patches

    def compute_embedding(self, img, bbox, tag, return_crops=False):
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs

        if self.model is None:
            self.initialize_model()

        # Generate all of the patches
        crops = []
        if self.grid_off:
            # Basic embeddings
            h, w = img.shape[:2]
            results = np.round(bbox).astype(np.int32)
            results[:, 0] = results[:, 0].clip(0, w)
            results[:, 1] = results[:, 1].clip(0, h)
            results[:, 2] = results[:, 2].clip(0, w)
            results[:, 3] = results[:, 3].clip(0, h)

            crops = []
            for p in results:
                crop = img[p[1] : p[3], p[0] : p[2]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
                if self.normalize:
                    crop /= 255
                    crop -= np.array((0.485, 0.456, 0.406))
                    crop /= np.array((0.229, 0.224, 0.225))
                crop = torch.as_tensor(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crops.append(crop)
        else:
            # Grid patch embeddings
            for idx, box in enumerate(bbox):
                crop = self.get_horizontal_split_patches(img, box, tag, idx)
                crops.append(crop)
        crops = torch.cat(crops, dim=0)

        # Create embeddings and l2 normalize them
        embs = []
        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx : idx + self.max_batch]
            batch_crops = batch_crops.cuda()
            with torch.no_grad():
                batch_embs = self.model(batch_crops)
            embs.extend(batch_embs)
        embs = torch.stack(embs)

        if not self.grid_off:
            embs = embs.reshape(bbox.shape[0], -1, embs.shape[-1])
        embs = embs.cpu().numpy()

        self.cache[tag] = embs
        if return_crops:
            return embs, crops
        return embs

    def initialize_model(self):
        return self._get_general_model()
    
    def _get_general_model(self):
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2268, loss="triplet", pretrained=False)
        sd = torch.load(self.model_ckpt, weights_only=False)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        self.model = model
        self.crop_size = (128, 256)
        self.normalize = True

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)

def add_visibility_score(df: pd.DataFrame, margin: float = 0) -> pd.DataFrame:
    """
    Return a copy of `df` with a new column `visibility_score` in [0,1].
    Any overlapping bbox whose top y-coordinate + margin is still above
    the current bbox's top will be ignored (treated as behind).
    
    Params:
      df     – DataFrame with columns ['frame','x','y','w','h',…].
      margin – number of pixels: other.y + margin < current.y → skip that other.
    """
    # 1) Copy & build geometry
    df2 = df.copy().reset_index(drop=True)
    df2['geometry'] = df2.apply(
        lambda r: box(r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']),
        axis=1
    )

    # 2) Compute visibility per frame
    vis_scores = {}
    for frame_id, group in tqdm(df2.groupby('frame'), desc='Computing visibility scores'):
        for idx, row in group.iterrows():
            geom      = row.geometry
            top_cur   = row['y']
            inters    = []
            # test only those not "behind" current
            for jdx, other in group.iterrows():
                if jdx == idx:
                    continue
                top_other = other['y']
                # if other is too far above → skip
                if top_other + margin < top_cur:
                    continue
                if not geom.intersects(other.geometry):
                    continue
                inters.append(geom.intersection(other.geometry))

            if inters:
                overlapped     = ops.unary_union(inters)
                overlapped_area = overlapped.area
            else:
                overlapped_area = 0.0

            area  = geom.area
            score = ((area - overlapped_area) / area) if area > 0 else 0.0
            vis_scores[idx] = max(0.0, score)

    # 3) Attach scores back
    df2['visibility_score'] = df2.index.map(vis_scores).fillna(1.0)

    # 4) Clean up
    return df2.drop(columns=['geometry'])

def get_top_visible_bboxes(df: pd.DataFrame, n: int= 5, margin: float = 0) -> pd.DataFrame:
    """
    For each track_id in `df`, compute visibility scores 
    and return the top `n` bboxes per track with the highest visibility.

    Params:
      df     – DataFrame with ['frame','track_id','x','y','w','h',…].
      n      – number of bboxes to keep per track.
    Returns:
      DataFrame with the same columns as `df` plus `visibility_score`, 
      containing at most `n` rows per track_id.
    """
    # needs the previous function in scope:
    scored = df.copy()

    # pick top-n by visibility_score within each track
    topn = (
        scored
        .groupby('track_id', group_keys=False)
        .apply(lambda g: g.nlargest(n, 'visibility_score'))
        .reset_index(drop=True)
    )
    return topn

def compute_embeddings(df: pd.DataFrame, embedding_computer, frame_path_template: str) -> pd.DataFrame:
    # 1) get top‑n visible boxes per track
    topn = df.copy()

    records = []
    # 2) compute embeddings frame by frame
    for frame_id, group in tqdm(topn.groupby('frame'), desc='Computing embeddings'):
        img_path = frame_path_template.format(frame_id)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load frame image: {img_path}")

        # prepare [x1,y1,x2,y2] array
        boxes = group[['x','y','w','h']].to_numpy()
        xyxy = np.stack([
            boxes[:,0],
            boxes[:,1],
            boxes[:,0] + boxes[:,2],
            boxes[:,1] + boxes[:,3]
        ], axis=1)

        embeddings = embedding_computer.compute_embedding(
            image, xyxy, f"frame_{frame_id}", return_crops=False
        )

        # 3) collect per‑bbox record
        for (_, row), emb in zip(group.iterrows(), embeddings):
            records.append({
                'track_id':        row['track_id'],
                'frame':           row['frame'],
                'x':               row['x'],
                'y':               row['y'],
                'w':               row['w'],
                'h':               row['h'],
                'visibility_score': row['visibility_score'],
                'embedding':       emb
            })

    # 4) return DataFrame
    return pd.DataFrame.from_records(records)

def compute_track_similarity(means, stds, track_ids):
    # Number of tracks
    n_tracks = len(means)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_tracks, n_tracks))
    
    # Compute pairwise similarities
    for i in range(n_tracks):
        for j in range(i, n_tracks):
            # Get mean and std for both tracks
            mean_i, std_i = means[i], stds[i]
            mean_j, std_j = means[j], stds[j]
            
            # Compute cosine similarity between means
            mean_similarity = 1 / (1 + np.linalg.norm(mean_i - mean_j))
            
            # Combine similarity components (adjust weights as needed)
            combined_similarity = 0.7 * mean_similarity 
            
            # Fill symmetric matrix
            similarity_matrix[i, j] = combined_similarity
            similarity_matrix[j, i] = combined_similarity
    
    # Create DataFrame with track IDs
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=track_ids,
        columns=track_ids
    )
    
    return similarity_df

def merge_tracks_by_similarity_fast(df: pd.DataFrame, similarity_df: pd.DataFrame, thresh: float = 0.03) -> pd.DataFrame:
    """
    Fast merging of track IDs using Union-Find + a single vectorized map.
    """
    uf = UnionFind()

    # 1) Union all pairs with similarity >= thresh
    pairs = similarity_df.loc[
        similarity_df['similarity_score'] >= thresh,
        ['track_id', 'most_similar_track_id']
    ].itertuples(index=False, name=None)

    for a, b in pairs:
        uf.union(a, b)

    # 2) Build a map from every original track_id in df to its root
    unique_ids = df['track_id'].unique()
    root_map = {tid: uf.find(tid) for tid in unique_ids}

    # 3) Apply the map in one go
    df_copy = df.copy()
    df_copy['track_id'] = df_copy['track_id'].map(root_map)

    return df_copy

def add_interpolated_bboxes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each track_id, looks for frame-gaps >1 and linearly
    interpolates x,y,w,h between the end of one segment and
    the start of the next. Returns a new df with those rows added.
    """
    out_chunks = []
    # work group-by-group
    for tid, grp in df.groupby('track_id', sort=False):
        grp = grp.sort_values('frame')
        out_chunks.append(grp)
        # walk pairs of consecutive detections
        for (_, prev), (_, nxt) in zip(grp.iloc[:-1].iterrows(),
                                       grp.iloc[1:].iterrows()):
            f0, f1 = int(prev.frame), int(nxt.frame)
            gap = f1 - f0 - 1
            if gap <= 0:
                continue
            # linear interp params
            for step in range(1, gap+1):
                alpha = step / (gap+1)
                new_frame = f0 + step
                interp = {
                    'track_id': tid,
                    'frame':    new_frame,
                    'x':        prev.x + alpha*(nxt.x - prev.x),
                    'y':        prev.y + alpha*(nxt.y - prev.y),
                    'w':        prev.w + alpha*(nxt.w - prev.w),
                    'h':        prev.h + alpha*(nxt.h - prev.h),
                }
                # copy any other columns verbatim from prev (or nxt)
                for col in grp.columns:
                    if col in interp or col in ('track_id','frame','x','y','w','h'):
                        continue
                    interp[col] = prev[col]
                out_chunks.append(pd.DataFrame([interp]))
    # concatenate & resort
    result = pd.concat(out_chunks, ignore_index=True)
    result = result.sort_values(['track_id','frame']).reset_index(drop=True)
    return result

def process_tracks(input_file, output_file, reid_model_path, frame_path_template, similarity_threshold=0.4, min_track_duration=5, top_n_visible=20):
    """
    Main processing function to improve tracking results using ReID
    """
    print(f"Processing tracking file: {input_file}")
    
    # Define column names and load data
    columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'class', 'visibility', '1', '2']
    df = pd.read_csv(input_file, header=None, names=columns)
    print(f"Loaded {len(df)} detections with {df['track_id'].nunique()} tracks")
    
    # Calculate track durations
    track_durations = df.groupby('track_id')['frame'].agg(['min', 'max'])
    track_durations['duration'] = track_durations['max'] - track_durations['min'] + 1
    
    # Filter short tracks
    valid_track_ids = track_durations[track_durations['duration'] >= min_track_duration].index
    filtered_df = df[df['track_id'].isin(valid_track_ids)]
    print(f"Filtered to {len(filtered_df)} detections with {filtered_df['track_id'].nunique()} tracks")
    
    # Sort by track_id and frame
    filtered_df = filtered_df.sort_values(by=['track_id', 'frame'])
    
    # Compute visibility scores
    result_with_visibility = add_visibility_score(filtered_df, margin=10)
    print("Computed visibility scores for all detections")
    
    # Get top visible bboxes
    most_visible = get_top_visible_bboxes(result_with_visibility, n=top_n_visible)
    print(f"Selected {len(most_visible)} most visible detections for embedding computation")
    
    # Initialize embedding computer
    embedding_computer = EmbeddingComputer(
        model_ckpt=reid_model_path,
        grid_off=True,
        max_batch=1024,
    )
    
    # Compute embeddings
    embeddings_df = compute_embeddings(most_visible, embedding_computer, frame_path_template)
    print(f"Computed embeddings for {len(embeddings_df)} detections")
    
    # Group by track_id and aggregate embeddings
    grouped_embeddings = embeddings_df.groupby('track_id').agg({
        'embedding': lambda x: np.stack(x.values),
    }).reset_index()
    
    # Calculate mean and std for each track's embeddings
    means = []
    stds = []
    for _, row in grouped_embeddings.iterrows():
        track_embeddings = row['embedding']
        mean_emb = np.mean(track_embeddings, axis=0)
        std_emb = np.std(track_embeddings, axis=0)
        means.append(mean_emb)
        stds.append(std_emb)
    
    # Convert to numpy arrays
    means = np.array(means)
    stds = np.array(stds)
    
    # Compute similarity matrix
    similarity_df = compute_track_similarity(means, stds, grouped_embeddings['track_id'])
    print("Computed track similarity matrix")
    
    # Find most similar tracks
    most_similar_tracks = []
    for track_id in similarity_df.index:
        similarities = similarity_df.loc[track_id].copy()
        similarities[track_id] = -1
        most_similar_id = similarities.idxmax()
        most_similar_score = similarities.max()
        
        most_similar_tracks.append({
            'track_id': track_id,
            'most_similar_track_id': most_similar_id,
            'similarity_score': most_similar_score
        })
    
    most_similar_df = pd.DataFrame(most_similar_tracks)
    
    # Get the set of frames for each track_id
    track_frames = filtered_df.groupby('track_id')['frame'].apply(set)
    
    # Filter mutually exclusive track_ids
    mutually_exclusive_tracks = most_similar_df[
        most_similar_df.apply(
            lambda row: track_frames[row['track_id']].isdisjoint(track_frames[row['most_similar_track_id']]),
            axis=1
        )
    ]
    
    # Ensure unique pairs by sorting track_id and most_similar_track_id
    mutually_exclusive_tracks['sorted_pair'] = mutually_exclusive_tracks.apply(
        lambda row: tuple(sorted((row['track_id'], row['most_similar_track_id']))), axis=1
    )
    
    # Drop duplicates based on the sorted pairs
    mutually_exclusive_tracks = mutually_exclusive_tracks.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])
    print(f"Found {len(mutually_exclusive_tracks)} mutually exclusive track pairs")
    
    # Merge tracks based on similarity
    updated_df = merge_tracks_by_similarity_fast(filtered_df, mutually_exclusive_tracks, thresh=similarity_threshold)
    print(f"Merged tracks: original={filtered_df['track_id'].nunique()}, merged={updated_df['track_id'].nunique()}")
    
    # Add interpolated bboxes
    final_df = add_interpolated_bboxes(updated_df)
    print(f"Added interpolated bboxes, final detection count: {len(final_df)}")
    
    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False, header=False)
    print(f"Results saved to {output_file}")
    
    # Return statistics
    return {
        "original_tracks": filtered_df['track_id'].nunique(),
        "merged_tracks": final_df['track_id'].nunique(),
        "track_reduction_percentage": 100 * (1 - final_df['track_id'].nunique() / filtered_df['track_id'].nunique())
    }

def main():
    parser = argparse.ArgumentParser(description='Track merging using ReID embeddings')
    parser.add_argument('--input', '-i', required=True, help='Input tracking file path')
    parser.add_argument('--output', '-o', required=True, help='Output file path for merged tracks')
    parser.add_argument('--reid-model', '-m', required=True, help='Path to the ReID model checkpoint')
    parser.add_argument('--frame-path', '-f', default='data/tracking/test/01/img1/{:06d}.jpg', 
                      help='Template for frame image paths with {:06d} placeholder for frame number')
    parser.add_argument('--similarity-threshold', '-t', type=float, default=0.4, 
                      help='Similarity threshold for track merging')
    parser.add_argument('--min-track-duration', '-d', type=int, default=5, 
                      help='Minimum track duration to consider')
    parser.add_argument('--top-visible', '-v', type=int, default=20, 
                      help='Number of top visible bboxes to use per track')
    
    args = parser.parse_args()
    
    # Process tracks
    stats = process_tracks(
        args.input, 
        args.output, 
        args.reid_model,
        args.frame_path,
        args.similarity_threshold,
        args.min_track_duration,
        args.top_visible
    )
    
    # Print statistics
    print("\nProcessing complete!")
    print(f"Original tracks: {stats['original_tracks']}")
    print(f"Merged tracks: {stats['merged_tracks']}")
    print(f"Track reduction: {stats['track_reduction_percentage']:.2f}%")

if __name__ == "__main__":
    main()