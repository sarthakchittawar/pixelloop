"""
Compute exact 3D point matches between frames using hash-based matching,
then apply FPS sampling to matched subsets.

Usage:
    python compute_exact_matches_fps.py --session_folder <path> [options]
"""

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from pathlib import Path
import habitat_sim
import copy


def load_metadata(session_folder):
    """Load metadata.json from session folder."""
    with open(os.path.join(session_folder, "metadata.json"), "r") as f:
        return json.load(f)


def load_poses(session_folder, num_frames):
    """Load camera poses from poses.txt."""
    poses = {}
    with open(os.path.join(session_folder, "poses_odom.txt"), "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            # frame_idx = int(parts[0]) // 5
            frame_idx = int(parts[0])
            position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            quaternion = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            poses[frame_idx] = (position, quaternion)
    return poses


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2],
    ])


def backproject_depth_to_3d(depth_map, K, position, quaternion):
    """Backproject depth map to 3D world coordinates."""
    H, W = depth_map.shape
    v, u = np.mgrid[0:H, 0:W]
    
    # Filter valid depth
    valid = (depth_map > 0.1) & (depth_map < 10.0) & np.isfinite(depth_map)
    if not np.any(valid):
        return None
    
    # Get valid pixels and depths
    u_valid = u[valid]
    v_valid = v[valid]
    depth_valid = depth_map[valid]
    
    # Back-project to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u_valid - cx) * depth_valid / fx
    Y = (v_valid - cy) * depth_valid / fy
    Z = depth_valid
    points_cam = np.stack([X, Y, Z], axis=-1)
    
    # Transform to world coordinates
    R = quaternion_to_rotation_matrix(quaternion)
    points_world = (R @ points_cam.T).T + position
    pixels = np.stack([u_valid, v_valid], axis=-1)
    
    return {
        "points_3d": points_world,
        "pixels": pixels,
        "depths": depth_valid,
    }


def farthest_point_sampling_fast(points, k):
    """
    Greedy FPS without full distance matrix.
    Complexity: O(N * K). Memory: O(N).
    """
    n_points, _ = points.shape
    if n_points <= k:
        return np.arange(n_points)
    
    # Initialize distances to infinity
    min_dists = np.full(n_points, np.inf)
    selected_indices = np.zeros(k, dtype=int)
    
    # Random start
    current_idx = np.random.randint(n_points)
    selected_indices[0] = current_idx
    
    for i in range(1, k):
        # Compute distance from the last selected point to all others
        dist_new = cdist(points[current_idx:current_idx+1], points, 'euclidean').flatten()
        
        # Update the running minimum distance for every point
        min_dists = np.minimum(min_dists, dist_new)
        
        # Pick the point furthest from the current set
        current_idx = np.argmax(min_dists)
        selected_indices[i] = current_idx
        
    return selected_indices


def uniform_sampling(points, stride):
    """
    Sample points uniformly with a given stride.
    
    Args:
        points: (N, 3) array of 3D points
        stride: Integer stride for sampling (e.g., stride=2 takes every 2nd point)
    
    Returns:
        Array of selected indices
    """
    n_points = points.shape[0]
    selected_indices = np.arange(0, n_points, stride)
    return selected_indices


def random_sampling(points, k):
    """
    Sample k points randomly without replacement.
    
    Args:
        points: (N, 3) array of 3D points
        k: Number of points to sample
    
    Returns:
        Array of selected indices
    """
    n_points = points.shape[0]
    if n_points <= k:
        return np.arange(n_points)
    
    selected_indices = np.random.choice(n_points, size=k, replace=False)
    return selected_indices


def find_exact_matches_hash_full(points_i, pixels_i, points_j, pixels_j):
    """
    Find exact 3D point matches using hash set on full point clouds.
    
    Args:
        points_i: (N, 3) array of 3D points in frame i
        pixels_i: (N, 2) array of pixel coordinates in frame i
        points_j: (M, 3) array of 3D points in frame j
        pixels_j: (M, 2) array of pixel coordinates in frame j
    
    Returns:
        matched_indices_i: Indices of matched points in frame i
        matched_indices_j: Indices of matched points in frame j
    """
    # Build hash map for frame_j
    point_map_j = {}
    for idx_j, pt in enumerate(points_j):
        pt_key = tuple(np.round(pt, decimals=2))
        if pt_key not in point_map_j:
            point_map_j[pt_key] = []
        point_map_j[pt_key].append(idx_j)
    
    # Find matches in frame_i
    matched_i = []
    matched_j = []
    
    for idx_i, pt_i in enumerate(points_i):
        pt_key = tuple(np.round(pt_i, decimals=2))
        if pt_key in point_map_j:
            # Take first match if multiple exist
            idx_j = point_map_j[pt_key][0]
            matched_i.append(idx_i)
            matched_j.append(idx_j)
    
    return np.array(matched_i), np.array(matched_j)


def apply_fps_to_matches(points_i, pixels_i, colors_i, points_j, pixels_j, colors_j,
                          matched_indices_i, matched_indices_j, target_points=250,
                          sampling_method='fps', stride=1):
    """
    Apply sampling to matched points to get a subset.
    
    Args:
        points_i, pixels_i, colors_i: Full data for frame i
        points_j, pixels_j, colors_j: Full data for frame j
        matched_indices_i: Indices of matched points in frame i
        matched_indices_j: Indices of matched points in frame j
        target_points: Number of points to sample (used for fps and random)
        sampling_method: 'fps', 'uniform', or 'random'
        stride: Stride for uniform sampling (ignored for fps and random)
    
    Returns:
        Dictionary with sampled frame data and matches
    """
    if len(matched_indices_i) == 0:
        return None
    
    # Get matched point subsets
    matched_points_i = points_i[matched_indices_i]
    matched_pixels_i = pixels_i[matched_indices_i]
    matched_colors_i = colors_i[matched_indices_i]
    
    matched_points_j = points_j[matched_indices_j]
    matched_pixels_j = pixels_j[matched_indices_j]
    matched_colors_j = colors_j[matched_indices_j]
    
    # Apply sampling based on method
    if sampling_method == 'fps':
        if len(matched_points_i) > target_points:
            # FPS on frame i matched points
            sample_indices = farthest_point_sampling_fast(matched_points_i, target_points)
        else:
            sample_indices = np.arange(len(matched_points_i))
    elif sampling_method == 'uniform':
        # Uniform sampling with stride
        sample_indices = uniform_sampling(matched_points_i, stride)
    elif sampling_method == 'random':
        # Random sampling
        if len(matched_points_i) > target_points:
            sample_indices = random_sampling(matched_points_i, target_points)
        else:
            sample_indices = np.arange(len(matched_points_i))
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Get sampled points
    sampled_points_i = matched_points_i[sample_indices]
    sampled_pixels_i = matched_pixels_i[sample_indices]
    sampled_colors_i = matched_colors_i[sample_indices]
    
    sampled_points_j = matched_points_j[sample_indices]
    sampled_pixels_j = matched_pixels_j[sample_indices]
    sampled_colors_j = matched_colors_j[sample_indices]
    
    # Create match list in the expected format
    matches = []
    for idx in range(len(sampled_points_i)):
        matches.append({
            'idx_i': idx,
            'pixel_j': tuple(sampled_pixels_j[idx])
        })
    
    return {
        'frame_i': {
            'points_3d': sampled_points_i,
            'pixels': sampled_pixels_i,
            'colors': sampled_colors_i
        },
        'frame_j': {
            'points_3d': sampled_points_j,
            'pixels': sampled_pixels_j,
            'colors': sampled_colors_j
        },
        'matches': matches,
        'num_total_matches': len(matched_indices_i),
        'num_sampled': len(sampled_points_i)
    }


def load_pointmap(pointmap_dir, frame_idx):
    """Load a single pointmap from disk."""
    p = os.path.join(pointmap_dir, f"{frame_idx:05d}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


def load_oracle_loops(oracle_loops_path):
    """Load oracle loop closure pairs from text file.
    
    Args:
        oracle_loops_path: Path to text file with space-separated frame pairs
        
    Returns:
        List of (frame_i, frame_j) tuples
    """
    if oracle_loops_path is None or not os.path.exists(oracle_loops_path):
        return []
    
    oracle_pairs = []
    with open(oracle_loops_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                frame_i = int(parts[0])
                frame_j = int(parts[1])
                oracle_pairs.append((frame_i, frame_j))
    
    return oracle_pairs

def project_3d_to_2d(point_3d_world, position, quaternion, K):
    """
    Project a 3D world point to 2D pixel coordinates.
    
    Args:
        point_3d_world: (3,) array of 3D point in world coordinates
        position: (3,) camera position in world
        quaternion: (4,) camera orientation quaternion
        K: (3, 3) camera intrinsics matrix
    
    Returns:
        (u, v) pixel coordinates as integers
    """
    # Transform from world to camera coordinates
    R = quaternion_to_rotation_matrix(quaternion)
    point_cam = R.T @ (point_3d_world - position)
    
    # Project to pixel coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    X, Y, Z = point_cam
    u = int(fx * X / Z + cx)
    v = int(fy * Y / Z + cy)
    
    return u, v

def compute_exact_matches_fps(session_folder, window_size=7, target_fps_points=250, 
                               min_matches=50, output_filename="exact_match_pairs_fps.pkl",
                               oracle_loops_path=None, sampling_method='fps', stride=1,
                               pointmap_dir=None):
    """
    Main function to compute exact matches with sampling.
    
    Args:
        session_folder: Path to session folder containing pointmaps
        window_size: Number of frames forward to match
        target_fps_points: Target number of points after sampling (for fps and random)
        min_matches: Minimum matches required before sampling
        output_filename: Name of output pickle file
        oracle_loops_path: Optional path to oracle loops file with additional frame pairs
        sampling_method: Sampling method to use ('fps', 'uniform', 'random')
        stride: Stride for uniform sampling (ignored for fps and random)
        pointmap_dir: Optional path to pointmap directory (default: session_folder/gt_pointmaps_fov90)
    
    Returns:
        Path to saved pickle file
    """
    print(f"Loading session data from: {session_folder}")
    
    # Load metadata and poses
    metadata = load_metadata(session_folder)
    K = np.array(metadata["camera_intrinsics"]["matrix"])
    num_frames = metadata["num_frames"]
    poses = load_poses(session_folder, num_frames)
    
    print(f"Loaded {num_frames} frames")
    print(f"Camera intrinsics: {K}")
    
    # Compute global bounds for color normalization
    print("Computing global bounds for RGB color mapping...")
    if pointmap_dir is None:
        pointmap_dir = os.path.join(session_folder, "gt_pointmaps_fov90")
    print(f"Using pointmap directory: {pointmap_dir}")
    if not os.path.isdir(pointmap_dir):
        raise FileNotFoundError(f"Missing pointmap_dir: {pointmap_dir}. Run backprojection first.")
    
    all_points = []
    for frame_idx in tqdm(range(num_frames), desc="Loading pointmaps for bounds"):
        pm = load_pointmap(pointmap_dir, frame_idx)
        if pm is None:
            continue
        valid = np.isfinite(pm[..., 0]) & np.isfinite(pm[..., 1]) & np.isfinite(pm[..., 2])
        if np.any(valid):
            all_points.append(pm[valid])
    
    all_points_stacked = np.vstack(all_points)
    bounds_min = np.min(all_points_stacked, axis=0)
    bounds_max = np.max(all_points_stacked, axis=0)
    print(f"Bounds: {bounds_min} to {bounds_max}")
    
    # Load oracle loops if provided
    oracle_pairs = load_oracle_loops(oracle_loops_path)
    if oracle_pairs:
        print(f"\nLoaded {len(oracle_pairs)} oracle loop closure pairs from: {oracle_loops_path}")
    
    # Compute exact matches with sampling
    if sampling_method == 'uniform':
        print(f"\nComputing exact matches (window_size={window_size}, sampling=uniform, stride={stride})...")
    else:
        print(f"\nComputing exact matches (window_size={window_size}, sampling={sampling_method}, target_points={target_fps_points})...")
    exact_match_pairs_fps = []
    image_size = None
    N = num_frames
    
    # Collect all pairs to process: window-based + oracle
    pairs_to_process = set()
    
    # Add window-based pairs
    for i in range(N):
        start = i + 1
        end = min(i + window_size + 1, N)
        for j in range(start, end):
            pairs_to_process.add((i, j))
    
    # Add oracle pairs
    for i, j in oracle_pairs:
        if i < N and j < N and i < j:  # Ensure valid and ordered
            pairs_to_process.add((i, j))
    
    # Convert to sorted list for consistent processing
    pairs_to_process = sorted(list(pairs_to_process))
    num_window_pairs = sum(min(i + window_size + 1, N) - (i + 1) for i in range(N))
    print(f"Total pairs to process: {len(pairs_to_process)} (window: {num_window_pairs}, oracle: {len(oracle_pairs)})")
    
    for i, j in tqdm(pairs_to_process, desc='Exact matching + FPS'):
        # Load full pointmap for frame i
        pm_i = load_pointmap(pointmap_dir, i)
        if pm_i is None:
            continue
        
        # Get image dimensions (store first time)
        if image_size is None:
            height, width = pm_i.shape[0], pm_i.shape[1]
            image_size = (height, width)
        
        valid_i = np.isfinite(pm_i[..., 0]) & np.isfinite(pm_i[..., 1]) & np.isfinite(pm_i[..., 2])
        if not np.any(valid_i):
            continue
        
        vv_i, uu_i = np.where(valid_i)
        points_i = pm_i[valid_i]
        pixels_i = np.stack([uu_i, vv_i], axis=-1)
        colors_i = ((points_i - bounds_min) / (bounds_max - bounds_min + 1e-6) * 255).clip(0, 255).astype(np.uint8)
        
        # Load full pointmap for frame j
        pm_j = load_pointmap(pointmap_dir, j)
        if pm_j is None:
            continue
        
        valid_j = np.isfinite(pm_j[..., 0]) & np.isfinite(pm_j[..., 1]) & np.isfinite(pm_j[..., 2])
        if not np.any(valid_j):
            continue
        
        vv_j, uu_j = np.where(valid_j)
        points_j = pm_j[valid_j]
        pixels_j = np.stack([uu_j, vv_j], axis=-1)
        colors_j = ((points_j - bounds_min) / (bounds_max - bounds_min + 1e-6) * 255).clip(0, 255).astype(np.uint8)
        
        # Find exact matches on full clouds
        matched_i, matched_j = find_exact_matches_hash_full(points_i, pixels_i, points_j, pixels_j)
        
        # Skip if too few matches
        if len(matched_i) < min_matches:
            continue
        
        # Apply sampling to matched points
        result = apply_fps_to_matches(
            points_i, pixels_i, colors_i,
            points_j, pixels_j, colors_j,
            matched_i, matched_j,
            target_points=target_fps_points,
            sampling_method=sampling_method,
            stride=stride
        )
        
        if result is not None:
            exact_match_pairs_fps.append({
                'frame_i': {**result['frame_i'], 'frame_idx': i},
                'frame_j': {**result['frame_j'], 'frame_idx': j},
                'matches': result['matches'],
                'num_total_matches': result['num_total_matches'],
                'num_sampled': result['num_sampled']
            })
    
    print(f"\nFound {len(exact_match_pairs_fps)} pairs with exact matches (after {sampling_method} sampling) in window size {window_size}.")

    # Show statistics
    if len(exact_match_pairs_fps) > 0:
        total_matches_before = [pair['num_total_matches'] for pair in exact_match_pairs_fps]
        num_sampled = [pair['num_sampled'] for pair in exact_match_pairs_fps]
        
        print(f"\nStatistics:")
        print(f"  Pairs found: {len(exact_match_pairs_fps)}")
        print(f"  Total matches before sampling:")
        print(f"    Total: {sum(total_matches_before)}")
        print(f"    Average per pair: {np.mean(total_matches_before):.1f}")
        print(f"    Min: {np.min(total_matches_before)}, Max: {np.max(total_matches_before)}")
        print(f"  After {sampling_method} sampling:")
        print(f"    Total: {sum(num_sampled)}")
        print(f"    Average per pair: {np.mean(num_sampled):.1f}")
        print(f"    Min: {np.min(num_sampled)}, Max: {np.max(num_sampled)}")
    else:
        print("WARNING: No exact matches found!")
        print("This suggests point clouds don't have overlapping exact coordinates.")
    
    # Save results
    output_dir = os.path.join(session_folder, "gt_matches")
    os.makedirs(output_dir, exist_ok=True)
    
    # Structure output as dict with image_size and matches
    output_data = {
        'image_size': image_size if image_size is not None else (480, 640),  # default fallback
        'matches': exact_match_pairs_fps
    }
    
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)
    
    print(f"\nSaved {len(exact_match_pairs_fps)} exact match pairs to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute exact 3D point matches with sampling (FPS, uniform, or random)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--session_folder",
        type=str,
        required=True,
        help="Path to session folder containing pointmaps/"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="Number of frames forward to match"
    )
    parser.add_argument(
        "--target_fps_points",
        type=int,
        default=1000,
        help="Target number of points after sampling (used for fps and random methods)"
    )
    parser.add_argument(
        "--min_matches",
        type=int,
        default=10,
        help="Minimum matches required before sampling"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="matches_160x120.pkl",
        help="Name of output pickle file"
    )
    parser.add_argument(
        "--oracle_loops",
        type=str,
        default=None,
        help="Path to oracle loops text file with additional frame pairs to match"
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        choices=['fps', 'uniform', 'random'],
        default='fps',
        help="Sampling method to use: 'fps' (farthest point sampling), 'uniform' (stride-based), or 'random'"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for uniform sampling (only used when sampling_method='uniform')"
    )
    parser.add_argument(
        "--pointmap_dir",
        type=str,
        default=None,
        help="Path to pointmap directory (default: session_folder/gt_pointmaps_fov90). "
             "Use e.g. scene_dir/mapanything_pointmaps_fov90 for MapAnything pointmaps."
    )
    
    args = parser.parse_args()
    
    # Print configuration block
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"  Session folder:      {args.session_folder}")
    print(f"  Window size:         {args.window_size}")
    print(f"  Target FPS points:   {args.target_fps_points}")
    print(f"  Min matches:         {args.min_matches}")
    print(f"  Output filename:     {args.output_filename}")
    print(f"  Oracle loops:        {args.oracle_loops}")
    print(f"  Sampling method:     {args.sampling_method}")
    print(f"  Stride:              {args.stride}")
    print(f"  Pointmap dir:        {args.pointmap_dir or '(default: gt_pointmaps_fov90)'}")
    print("="*80 + "\n")

    # Run computation
    output_path = compute_exact_matches_fps(
        session_folder=args.session_folder,
        window_size=args.window_size,
        target_fps_points=args.target_fps_points,
        min_matches=args.min_matches,
        output_filename=args.output_filename,
        oracle_loops_path=args.oracle_loops,
        sampling_method=args.sampling_method,
        stride=args.stride,
        pointmap_dir=args.pointmap_dir,
    )
    
    print(f"\n✓ Done! Results saved to: {output_path}")

if __name__ == "__main__":
    main()
