"""
Generate Oracle Loop Closures based on co-visibility between image pairs.

This script:
1. Computes co-visibility between all image pairs using depth maps and camera intrinsics
2. Filters out nearby frames (within ±window_size of query)
3. Avoids redundant loop closures by checking if nearby frames already have LCs to similar targets
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import numba
from numba import jit, prange

BASE_DIR = "."
sys.path.append(BASE_DIR)

from utils import getK_fromParams


def load_depth_map(depth_path: str, depth_mode: str = "png") -> np.ndarray:
    """
    Load depth map from file.
    
    Args:
        depth_path: Path to depth file
        depth_mode: "png" or "npy"
    
    Returns:
        Depth map in meters as numpy array
    """
    if depth_mode == "png":
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth_meters = depth_raw * 0.001  # Convert from mm to meters
    else:  # "npy"
        depth_meters = np.load(depth_path).astype(np.float64)
    
    return depth_meters


def get_pixel_3d_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth map to 3D points in camera coordinates.
    
    Args:
        depth: Depth map (H, W)
        K: Camera intrinsic matrix (3, 3)
    
    Returns:
        3D points array of shape (H, W, 3)
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Compute 3D coordinates
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    pts3d = np.stack([X, Y, Z], axis=-1)
    return pts3d


def load_camera_poses(scene_dir: str) -> np.ndarray:
    """
    Load camera poses from agent_states.npy or poses file.
    
    Args:
        scene_dir: Path to scene directory
    
    Returns:
        Array of camera poses (N, 4, 4)
    
    Raises:
        FileNotFoundError: If no pose file is found
    """
    import quaternion
    
    poses_path = os.path.join(scene_dir, "poses_odom.txt")
    if os.path.exists(poses_path):
        poses_data = np.loadtxt(poses_path)
        # Assuming format: tx ty tz qx qy qz qw per line
        poses = []
        for row in poses_data:
            pos = row[1:4]
            q = np.quaternion(row[7], row[4], row[5], row[6])  # w, x, y, z
            R = quaternion.as_rotation_matrix(q)
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pos
            poses.append(T)
        
        return np.array(poses)
    
    raise FileNotFoundError(f"No camera poses found in {scene_dir}. "
                            f"Expected agent_states.npy or poses.txt")


def compute_covisibility(
    pts3d_src: np.ndarray,
    depth_src: np.ndarray,
    depth_tgt: np.ndarray,
    K: np.ndarray,
    T_src_to_tgt: np.ndarray,
    depth_thresh: float = 0.1,
    valid_depth_min: float = 0.01,
    valid_depth_max: float = 100.0,
    subsample: int = 4
) -> Tuple[int, int, float]:
    """
    Compute co-visibility between two images.
    
    Projects 3D points from source image to target image and checks
    how many points are visible in both views.
    
    Args:
        pts3d_src: 3D points from source image (H, W, 3)
        depth_src: Depth map of source image (H, W)
        depth_tgt: Depth map of target image (H, W)
        K: Camera intrinsic matrix (3, 3)
        T_src_to_tgt: Transformation from source to target camera (4, 4)
        depth_thresh: Threshold for depth consistency check
        valid_depth_min: Minimum valid depth value
        valid_depth_max: Maximum valid depth value
        subsample: Subsample factor for faster computation
    
    Returns:
        Tuple of (num_covisible_points, num_valid_src_points, covisibility_ratio)
    """
    H, W = depth_src.shape
    
    # Subsample for efficiency
    pts3d_sub = pts3d_src[::subsample, ::subsample]
    depth_src_sub = depth_src[::subsample, ::subsample]
    
    # Flatten
    pts3d_flat = pts3d_sub.reshape(-1, 3)
    depth_src_flat = depth_src_sub.reshape(-1)
    
    # Filter valid depth points
    valid_mask = (depth_src_flat > valid_depth_min) & (depth_src_flat < valid_depth_max)
    pts3d_valid = pts3d_flat[valid_mask]
    
    if len(pts3d_valid) == 0:
        return 0, 0, 0.0
    
    num_valid_src = len(pts3d_valid)
    
    # Transform to target camera frame
    pts3d_homo = np.hstack([pts3d_valid, np.ones((len(pts3d_valid), 1))])
    pts3d_tgt = (T_src_to_tgt @ pts3d_homo.T).T[:, :3]
    
    # Filter points behind camera
    in_front = pts3d_tgt[:, 2] > valid_depth_min
    pts3d_tgt = pts3d_tgt[in_front]
    
    if len(pts3d_tgt) == 0:
        return 0, num_valid_src, 0.0
    
    # Project to target image
    pts2d_tgt = (K @ pts3d_tgt.T).T
    pts2d_tgt = pts2d_tgt[:, :2] / pts2d_tgt[:, 2:3]
    
    # Check if points are within image bounds
    u = pts2d_tgt[:, 0]
    v = pts2d_tgt[:, 1]
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    pts2d_valid = pts2d_tgt[in_bounds]
    pts3d_tgt_valid = pts3d_tgt[in_bounds]
    
    if len(pts2d_valid) == 0:
        return 0, num_valid_src, 0.0
    
    # Check depth consistency
    u_int = pts2d_valid[:, 0].astype(int)
    v_int = pts2d_valid[:, 1].astype(int)
    
    depth_tgt_sampled = depth_tgt[v_int, u_int]
    projected_depth = pts3d_tgt_valid[:, 2]
    
    # Valid if depth is consistent (not occluded)
    depth_valid = (depth_tgt_sampled > valid_depth_min) & (depth_tgt_sampled < valid_depth_max)
    depth_consistent = np.abs(depth_tgt_sampled - projected_depth) < depth_thresh
    
    num_covisible = np.sum(depth_valid & depth_consistent)
    covisibility_ratio = num_covisible / num_valid_src if num_valid_src > 0 else 0.0
    
    return int(num_covisible), num_valid_src, covisibility_ratio


@jit(nopython=True, parallel=True, cache=True)
def compute_bidirectional_covisibility_numba(
    pts3d_i: np.ndarray,
    pts3d_j: np.ndarray,
    depth_i: np.ndarray,
    depth_j: np.ndarray,
    K: np.ndarray,
    T_i_to_j: np.ndarray,
    valid_depth_min: float = 0.1,
    valid_depth_max: float = 10.0,
    depth_thresh: float = 0.01,
    subsample: int = 1
) -> tuple:
    """Numba-accelerated bidirectional co-visibility computation."""
    H, W = depth_i.shape
    H_sub = H // subsample
    W_sub = W // subsample
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    num_valid_i = 0
    num_valid_j = 0
    num_common = 0
    
    # Count valid points in j
    for v in range(H_sub):
        for u in range(W_sub):
            d = depth_j[v * subsample, u * subsample]
            if d > valid_depth_min and d < valid_depth_max:
                num_valid_j += 1
    
    # Process points from i
    for v in range(H_sub):
        for u in range(W_sub):
            vs = v * subsample
            us = u * subsample
            d_i = depth_i[vs, us]
            
            if d_i <= valid_depth_min or d_i >= valid_depth_max:
                continue
            
            num_valid_i += 1
            
            # Get 3D point in camera i frame
            X_i = pts3d_i[vs, us, 0]
            Y_i = pts3d_i[vs, us, 1]
            Z_i = pts3d_i[vs, us, 2]
            
            # Transform to camera j frame
            X_j = T_i_to_j[0, 0] * X_i + T_i_to_j[0, 1] * Y_i + T_i_to_j[0, 2] * Z_i + T_i_to_j[0, 3]
            Y_j = T_i_to_j[1, 0] * X_i + T_i_to_j[1, 1] * Y_i + T_i_to_j[1, 2] * Z_i + T_i_to_j[1, 3]
            Z_j = T_i_to_j[2, 0] * X_i + T_i_to_j[2, 1] * Y_i + T_i_to_j[2, 2] * Z_i + T_i_to_j[2, 3]
            
            if Z_j <= valid_depth_min:
                continue
            
            # Project to image j
            u_proj = fx * X_j / Z_j + cx
            v_proj = fy * Y_j / Z_j + cy
            
            u_int = int(u_proj)
            v_int = int(v_proj)
            
            if u_int < 0 or u_int >= W or v_int < 0 or v_int >= H:
                continue
            
            # Check depth at projected location
            d_j = depth_j[v_int, u_int]
            if d_j <= valid_depth_min or d_j >= valid_depth_max:
                continue
            
            # Get 3D point in camera j frame from image j
            X_j2 = pts3d_j[v_int, u_int, 0]
            Y_j2 = pts3d_j[v_int, u_int, 1]
            Z_j2 = pts3d_j[v_int, u_int, 2]
            
            # Compute distance
            dist = np.sqrt((X_j - X_j2)**2 + (Y_j - Y_j2)**2 + (Z_j - Z_j2)**2)
            
            if dist < depth_thresh:
                num_common += 1
    
    return num_common, num_valid_i, num_valid_j


def compute_relative_transform(T_i: np.ndarray, T_j: np.ndarray) -> np.ndarray:
    """Compute relative transform from camera i to camera j."""
    T_opencv_to_habitat = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)
    
    T_i_adjusted = T_i @ T_opencv_to_habitat
    T_j_adjusted = T_j @ T_opencv_to_habitat
    T_i_to_j = np.linalg.inv(T_j_adjusted) @ T_i_adjusted
    return T_i_to_j.astype(np.float64)


def compute_pose_difference(T_i: np.ndarray, T_j: np.ndarray) -> Tuple[float, float]:
    """
    Compute translation distance and rotation angle between two poses.
    
    Args:
        T_i: Camera-to-world transform for image i (4, 4)
        T_j: Camera-to-world transform for image j (4, 4)
    
    Returns:
        Tuple of (translation_distance, rotation_angle_degrees)
    """
    # Extract positions
    pos_i = T_i[:3, 3]
    pos_j = T_j[:3, 3]
    translation_dist = np.linalg.norm(pos_i - pos_j)
    
    # Extract rotations and compute relative rotation
    R_i = T_i[:3, :3]
    R_j = T_j[:3, :3]
    R_rel = R_i.T @ R_j
    
    # Compute rotation angle from trace
    # trace(R) = 1 + 2*cos(theta)
    trace = np.trace(R_rel)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    rotation_angle_rad = np.arccos(cos_angle)
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    
    return translation_dist, rotation_angle_deg


def plot_loop_closures(
    scene_dir: str,
    loop_closures: List[Tuple[int, int]],
    lc_scores: Dict[Tuple[int, int], float],
    lc_common_points: Dict[Tuple[int, int], int],
    output_dir: str,
    max_plots: int = 50
) -> None:
    """
    Generate visualization plots for loop closures.
    
    Args:
        scene_dir: Path to scene directory
        loop_closures: List of (query_idx, target_idx) pairs
        lc_scores: Dictionary mapping pairs to co-visibility scores
        lc_common_points: Dictionary mapping pairs to number of common points
        output_dir: Directory to save plots
        max_plots: Maximum number of plots to generate
    """
    img_dir = os.path.join(scene_dir, "images_fov90")
    vis_dir = os.path.join(output_dir, "lc_visualizations")

    import shutil
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get image files
    img_files = natsorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    
    # Limit number of plots
    lcs_to_plot = loop_closures
    
    for idx, (q, t) in enumerate(tqdm(lcs_to_plot, desc="Generating visualizations")):
        # Load images
        img_q_path = os.path.join(img_dir, img_files[q])
        img_t_path = os.path.join(img_dir, img_files[t])
        
        img_q = cv2.imread(img_q_path)
        img_t = cv2.imread(img_t_path)
        
        if img_q is None or img_t is None:
            continue
        
        img_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB)
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        
        # Get scores
        key = (q, t) if (q, t) in lc_scores else (t, q)
        score = lc_scores.get(key, 0.0)
        common_pts = lc_common_points.get(key, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(img_q)
        axes[0].set_title(f"Query: Frame {q}")
        axes[0].axis('off')
        
        axes[1].imshow(img_t)
        axes[1].set_title(f"Target: Frame {t}")
        axes[1].axis('off')
        
        fig.suptitle(f"Loop Closure: {q} ↔ {t}\n"
                     f"Common Points: {common_pts}, Co-visibility: {score:.3f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"lc_{idx:04d}_{q}_{t}.png"), dpi=100)
        plt.close(fig)


def compute_bidirectional_covisibility(
    pts3d_i: np.ndarray,
    pts3d_j: np.ndarray,
    depth_i: np.ndarray,
    depth_j: np.ndarray,
    K: np.ndarray,
    T_i: np.ndarray,
    T_j: np.ndarray,
    depth_thresh: float = 0.01,
    valid_depth_min: float = 0.1,
    valid_depth_max: float = 10.0,
    subsample: int = 1
) -> Tuple[int, int, int]:
    """
    Compute bidirectional co-visibility between two images based on common 3D points.
    
    For each pixel in image i, project its 3D point to image j. Then check if the
    3D point at that pixel location in image j matches (in world coordinates).
    
    Args:
        pts3d_i: 3D points from image i in camera coordinates (H, W, 3)
        pts3d_j: 3D points from image j in camera coordinates (H, W, 3)
        depth_i: Depth map of image i (H, W)
        depth_j: Depth map of image j (H, W)
        K: Camera intrinsic matrix (3, 3)
        T_i: Camera-to-world transform for image i (4, 4)
        T_j: Camera-to-world transform for image j (4, 4)
        depth_thresh: Threshold for 3D point matching (in meters)
        valid_depth_min: Minimum valid depth value
        valid_depth_max: Maximum valid depth value
        subsample: Subsample factor for faster computation
    
    Returns:
        Tuple of (num_common_points, num_valid_i, num_valid_j)
    """
    H, W = depth_i.shape

    T_i_to_j = compute_relative_transform(T_i, T_j)
    
    num_common, num_valid_i, num_valid_j = compute_bidirectional_covisibility_numba(
        pts3d_i, pts3d_j, depth_i, depth_j, K, T_i_to_j,
        valid_depth_min=valid_depth_min, valid_depth_max=valid_depth_max,
        depth_thresh=depth_thresh, subsample=subsample
    )
    
    return num_common, num_valid_i, num_valid_j


def generate_oracle_loop_closures(
    scene_dir: str,
    output_path: str,
    width: int = 320,
    height: int = 240,
    hfov: int = 90,
    depth_mode: str = "png",
    covisibility_thresh: float = 0.1,
    min_common_points: int = 100,
    window_size: int = 3,
    subsample: int = 1,
    max_lc_per_image: int = 3,
    max_translation: float = 10.0,
    max_rotation: float = 180.0,
    verbose: bool = True,
    visualize: bool = True,
    max_vis_plots: int = 50
) -> List[Tuple[int, int]]:
    """
    Generate oracle loop closures for a scene.
    
    Args:
        scene_dir: Path to scene directory containing images and depth
        output_path: Path to save the output loop closure pairs
        width: Image width
        height: Image height
        hfov: Horizontal field of view in degrees
        depth_mode: "png" or "npy"
        covisibility_thresh: Minimum co-visibility ratio to consider as loop closure
        min_common_points: Minimum number of common 3D points for a valid LC
        window_size: Size of window to mask around query (±window_size)
        subsample: Subsample factor for co-visibility computation
        max_lc_per_image: Maximum number of loop closures per query image
        max_translation: Maximum translation distance (meters) between LC pairs
        max_rotation: Maximum rotation angle (degrees) between LC pairs
        verbose: Print progress information
        visualize: Whether to generate visualization plots
        max_vis_plots: Maximum number of visualization plots to generate
    
    Returns:
        List of loop closure pairs (query_idx, target_idx)
    """
    # Setup paths
    img_dir = os.path.join(scene_dir, "images_fov90")
    depth_dir = os.path.join(scene_dir, "images_depth_fov90")
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(depth_dir):
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    
    # Get image list
    img_files = natsorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    num_images = len(img_files)
    
    if verbose:
        print(f"Found {num_images} images in {img_dir}")
    
    # Get camera intrinsics
    K = getK_fromParams(hfov, width, height)
    
    # Load camera poses (required)
    poses = load_camera_poses(scene_dir)
    if verbose:
        print(f"Loaded {len(poses)} camera poses")
    
    # Determine depth file extension
    depth_ext = ".png" if depth_mode == "png" else ".npy"
    
    # Preload all depth maps
    if verbose:
        print("Loading depth maps...")
    
    depth_maps = []
    pts3d_maps = []
    for i in tqdm(range(num_images), desc="Loading depth", disable=not verbose):
        depth_path = os.path.join(depth_dir, f"{i:05d}{depth_ext}")
        if not os.path.exists(depth_path):
            # Try alternative naming
            depth_path = os.path.join(depth_dir, f"{i:04d}{depth_ext}")
        
        if os.path.exists(depth_path):
            depth = load_depth_map(depth_path, depth_mode)
            if depth.shape != (height, width):
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            print(f"Warning: Depth not found for image {i}")
            depth = np.zeros((height, width), dtype=np.float64)
        
        depth_maps.append(depth)
        pts3d = get_pixel_3d_points(depth, K)
        pts3d_maps.append(pts3d)
    
    # Track which images already have loop closures (and their LC targets)
    lc_assigned: Dict[int, Set[int]] = defaultdict(set)
    
    # Store all candidate loop closures with their scores
    all_candidates: List[Tuple[int, int, int, float]] = []
    
    # Store scores for visualization
    lc_scores: Dict[Tuple[int, int], float] = {}
    lc_common_points: Dict[Tuple[int, int], int] = {}
    
    # Stats for filtering
    num_skipped_pose = 0
    
    if verbose:
        print("Computing co-visibility matrix...")
        print(f"  Max translation: {max_translation:.2f} meters")
        print(f"  Max rotation: {max_rotation:.1f} degrees")
    
    # Compute co-visibility for all pairs
    for i in tqdm(range(num_images), desc="Computing co-visibility", disable=not verbose):
        for j in range(i + 1, num_images):
            # Skip if within window (sequential frames)
            if abs(j - i) <= window_size:
                continue
            
            # Check translation and rotation thresholds
            trans_dist, rot_angle = compute_pose_difference(poses[i], poses[j])
            
            # Skip if pose difference exceeds thresholds
            if trans_dist > max_translation or rot_angle > max_rotation:
                num_skipped_pose += 1
                continue
            
            # Compute bidirectional co-visibility
            num_common, num_valid_i, num_valid_j = compute_bidirectional_covisibility(
                pts3d_maps[i], pts3d_maps[j],
                depth_maps[i], depth_maps[j],
                K, poses[i], poses[j],
                subsample=subsample
            )
            
            # Compute co-visibility ratio (relative to smaller valid point set)
            min_valid = min(num_valid_i, num_valid_j)
            covis_ratio = num_common / min_valid if min_valid > 0 else 0.0
            
            # Check both thresholds: minimum common points AND ratio
            if num_common >= min_common_points and covis_ratio >= covisibility_thresh:
                all_candidates.append((i, j, num_common, covis_ratio))
                lc_scores[(i, j)] = covis_ratio
                lc_common_points[(i, j)] = num_common
    
    if verbose:
        print(f"Skipped {num_skipped_pose} pairs due to pose thresholds")
        print(f"Found {len(all_candidates)} candidate pairs above thresholds "
              f"(min_common_points={min_common_points}, covis_thresh={covisibility_thresh})")
    
    # Sort candidates by number of common points (descending), then by ratio
    all_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    # Greedily select loop closures
    final_loop_closures: List[Tuple[int, int]] = []
    
    for query_idx, target_idx, num_common, covis_ratio in all_candidates:
        # Check if this pair should be skipped due to redundancy
        skip = False
        
        # Get the window around query and target
        query_window = set(range(max(0, query_idx - window_size), 
                                  min(num_images, query_idx + window_size + 1)))
        target_window = set(range(max(0, target_idx - window_size), 
                                   min(num_images, target_idx + window_size + 1)))
        
        # Check if any image in query's window already has LC to target's window
        for q in query_window:
            if q in lc_assigned:
                if lc_assigned[q] & target_window:
                    skip = True
                    break
        
        if skip:
            continue
        
        # Check if query already has max LCs
        if len(lc_assigned[query_idx]) >= max_lc_per_image:
            continue
        
        # Add this loop closure
        final_loop_closures.append((query_idx, target_idx))
        lc_assigned[query_idx].add(target_idx)
        lc_assigned[target_idx].add(query_idx)  # Symmetric
    
    if verbose:
        print(f"Selected {len(final_loop_closures)} non-redundant loop closures")
    
    # Sort by query frame index (ascending)
    final_loop_closures.sort(key=lambda x: (x[0], x[1]))
    
    # Save to file
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for q, t in final_loop_closures:
            f.write(f"{q} {t}\n")
    
    if verbose:
        print(f"Saved loop closures to {output_path}")
    
    # Generate visualizations
    if visualize and len(final_loop_closures) > 0:
        if verbose:
            print("Generating visualizations...")
        plot_loop_closures(
            scene_dir=scene_dir,
            loop_closures=final_loop_closures,
            lc_scores=lc_scores,
            lc_common_points=lc_common_points,
            output_dir=output_dir if output_dir else scene_dir,
            max_plots=max_vis_plots
        )
    
    return final_loop_closures


def main():
    parser = argparse.ArgumentParser(
        description="Generate oracle loop closures based on co-visibility"
    )
    
    # Make scene_dir and batch_scenes mutually exclusive
    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument(
        "--scene_dir", type=str, default=None,
        help="Path to scene directory containing images/ and images_depth/"
    )
    scene_group.add_argument(
        "--batch_scenes", type=str, default=None,
        help="Path to directory containing multiple scenes to process"
    )
    
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: scene_dir/oracle_loops2.txt)"
    )
    parser.add_argument(
        "--width", type=int, default=320,
        help="Image width"
    )
    parser.add_argument(
        "--height", type=int, default=240,
        help="Image height"
    )
    parser.add_argument(
        "--hfov", type=int, default=90,
        help="Horizontal field of view in degrees"
    )
    parser.add_argument(
        "--depth_mode", type=str, default="png", choices=["png", "npy"],
        help="Depth file format"
    )
    parser.add_argument(
        "--covisibility_thresh", type=float, default=0.0,
        help="Minimum co-visibility ratio to consider as loop closure"
    )
    parser.add_argument(
        "--min_common_points", type=int, default=1000,
        help="Minimum number of common 3D points for a valid LC"
    )
    parser.add_argument(
        "--window_size", type=int, default=3,
        help="Size of window to mask around query (±window_size)"
    )
    parser.add_argument(
        "--subsample", type=int, default=1,
        help="Subsample factor for co-visibility computation"
    )
    parser.add_argument(
        "--max_lc_per_image", type=int, default=3,
        help="Maximum number of loop closures per query image"
    )
    parser.add_argument(
        "--max_translation", type=float, default=1.0,
        help="Maximum translation distance (meters) between LC pairs"
    )
    parser.add_argument(
        "--max_rotation", type=float, default=90.0,
        help="Maximum rotation angle (degrees) between LC pairs"
    )
    parser.add_argument(
        "--no_visualize", action="store_true",
        help="Disable visualization generation"
    )
    parser.add_argument(
        "--max_vis_plots", type=int, default=50,
        help="Maximum number of visualization plots to generate"
    )
    
    args = parser.parse_args()
    
    if args.batch_scenes:
        # Process multiple scenes
        scene_dirs = sorted([
            os.path.join(args.batch_scenes, d) 
            for d in os.listdir(args.batch_scenes) 
            if os.path.isdir(os.path.join(args.batch_scenes, d))
        ])
        
        print(f"Processing {len(scene_dirs)} scenes...")
        
        for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
            output_path = os.path.join(scene_dir, "oracle_loops.txt")
            try:
                generate_oracle_loop_closures(
                    scene_dir=scene_dir,
                    output_path=output_path,
                    width=args.width,
                    height=args.height,
                    hfov=args.hfov,
                    depth_mode=args.depth_mode,
                    covisibility_thresh=args.covisibility_thresh,
                    min_common_points=args.min_common_points,
                    window_size=args.window_size,
                    subsample=args.subsample,
                    max_lc_per_image=args.max_lc_per_image,
                    max_translation=args.max_translation,
                    max_rotation=args.max_rotation,
                    verbose=False,
                    visualize=not args.no_visualize,
                    max_vis_plots=args.max_vis_plots
                )
            except Exception as e:
                print(f"Error processing {scene_dir}: {e}")
    else:
        # Process single scene
        output_path = args.output or os.path.join(args.scene_dir, "oracle_loops.txt")
        
        generate_oracle_loop_closures(
            scene_dir=args.scene_dir,
            output_path=output_path,
            width=args.width,
            height=args.height,
            hfov=args.hfov,
            depth_mode=args.depth_mode,
            covisibility_thresh=args.covisibility_thresh,
            min_common_points=args.min_common_points,
            window_size=args.window_size,
            subsample=args.subsample,
            max_lc_per_image=args.max_lc_per_image,
            max_translation=args.max_translation,
            max_rotation=args.max_rotation,
            verbose=True,
            visualize=not args.no_visualize,
            max_vis_plots=args.max_vis_plots
        )


if __name__ == "__main__":
    main()