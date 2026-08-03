"""
Geometry and Image Processing Utilities

Provides utility functions for:
- Point cloud sampling (FPS)
- Image resizing for ViT models
- 2D to 3D projections
- Mask processing
"""

import numpy as np
import torch
import cv2
import torchvision.transforms as tfm
import os

try:
    import open3d as o3d
except ImportError:
    o3d = None


def farthest_point_sampling_o3d(points, n_samples):
    """
    Farthest Point Sampling using Open3D (most robust)

    Args:
        points: numpy array of shape (N, 3) - 3D coordinates
        n_samples: number of points to sample

    Returns:
        indices: numpy array of sampled point indices
    """
    if o3d is None:
        raise ImportError("open3d is required for farthest_point_sampling_o3d")
    
    if len(points) <= n_samples:
        return np.arange(len(points))

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Apply farthest point sampling
    sampled_pcd = pcd.farthest_point_down_sample(n_samples)

    # Find indices of sampled points in original array
    sampled_points = np.asarray(sampled_pcd.points)

    # Find closest matches (should be exact for numerical precision)
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, indices = tree.query(sampled_points)

    return indices


def resize_to_divisible(img: torch.Tensor, divisible_by: int = 14) -> torch.Tensor:
    """Resize to be divisible by a factor. Useful for ViT based models.

    Args:
        img (torch.Tensor): img as tensor, in (*, H, W) order
        divisible_by (int, optional): factor to make sure img is divisible by. Defaults to 14.

    Returns:
        torch.Tensor: img tensor with divisible shape
    """
    h, w = img.shape[-2:]

    divisible_h = round(h / divisible_by) * divisible_by
    divisible_w = round(w / divisible_by) * divisible_by
    img = tfm.functional.resize(img, [divisible_h, divisible_w], antialias=True)

    return img


def pixel_to_camera_3d(y, x, depth, K):
    """
    Projects a 2D pixel (y, x) with depth to 3D camera coordinates using intrinsics K.

    Args:
        y (float or int): Pixel row coordinate.
        x (float or int): Pixel column coordinate.
        depth (float): Depth value at the pixel.
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        np.ndarray: 3D point (x, y, z) in camera coordinates.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z])


def get_mask_centroid(object_mask):
    """
    Get centroid of binary mask using OpenCV moments.

    Args:
        object_mask: Binary mask (0s and 1s) of shape (H, W)

    Returns:
        (cx, cy): Centroid coordinates as (x, y) pixel coordinates
        Returns None if mask is empty
    """
    # Ensure mask is uint8
    if object_mask.dtype != np.uint8:
        object_mask = object_mask.astype(np.uint8)

    # Calculate moments
    moments = cv2.moments(object_mask)

    # Check if mask is not empty
    if moments["m00"] == 0:
        return None

    # Calculate centroid
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return (cx, cy)


def get_goal_info(episode_path: str, task_type: str = "original") -> tuple:
    """
    Get goal info from episode folder.
    
    Looks for goal information in the following order:
    1. episode.npy for goal_object_id
    2. obs_g.npy for goal semantic mask
    3. images_sem/ folder for semantic masks
    
    Args:
        episode_path: Path to episode directory (contains images/, episode.npy, etc.)
        task_type: Task type for goal parsing ("original", "alt_goal", "alt_goal_v2")
    
    Returns:
        (goal_img_idx, goal_mask, goal_instance_id)
    """
    # Get goal instance ID
    episode_data_path = os.path.join(episode_path, "episode.npy")
    if os.path.exists(episode_data_path):
        episode = np.load(episode_data_path, allow_pickle=True)[()]
        # Handle both dict and object types
        if isinstance(episode, dict):
            goal_instance_id = episode['goal_object_id']
        else:
            goal_instance_id = getattr(episode, 'goal_object_id', None)
            if goal_instance_id is None:
                # Try alternative attribute names
                goal_instance_id = getattr(episode, 'object_id', None)
    else:
        # Parse from folder name: scene_id_goal_instance_id_
        goal_instance_id = int(episode_path.rstrip('/').split('_')[-2])
    
    # Get goal image index (last image in the map)
    images_dir = os.path.join(episode_path, "images")
    goal_img_idx = len(os.listdir(images_dir)) - 1
    
    # Get goal mask
    goal_filepath = os.path.join(episode_path, "obs_g.npy")
    if os.path.exists(goal_filepath):
        # Direct goal observation
        obs_g = np.load(goal_filepath, allow_pickle=True)[()]
        instance_mask = obs_g.get('semantic_sensor', obs_g)
    else:
        # Load from semantic images folder
        semantic_filepath = os.path.join(episode_path, f"images_sem/{goal_img_idx:05d}.npy")
        if os.path.exists(semantic_filepath):
            semantic_mask = np.load(semantic_filepath, allow_pickle=True)
            instance_mask = semantic_mask == goal_instance_id
        else:
            raise FileNotFoundError(
                f"Could not find goal mask at {goal_filepath} or {semantic_filepath}"
            )
    
    # Handle binary vs instance mask
    unique_vals = np.unique(instance_mask)
    if instance_mask.dtype == bool or set(unique_vals).issubset({False, True, 0, 1}):
        goal_mask = instance_mask.astype(np.uint8)
    else:
        goal_mask = (instance_mask == int(goal_instance_id)).astype(np.uint8)
    
    return goal_img_idx, goal_mask, goal_instance_id

def get_goal_info_new(episode_path: str, task_type: str = "original") -> tuple:
    """
    Get goal info from episode folder.
    
    Looks for goal information in the following order:
    1. goal_mask_{img_idx}.png files in the episode directory
    Args:
        episode_path: Path to episode directory (contains images/, episode.npy, etc.)
        task_type: Task type for goal parsing ("original", "alt_goal", "alt_goal_v2")
    Returns:
        (goal_img_idx, goal_mask)
    """
    goal_mask_files = [f for f in os.listdir(episode_path) if f.startswith("goal_mask_") and f.endswith(".png")]
    if not goal_mask_files:
        raise FileNotFoundError(f"No goal_mask_*.png files found in {episode_path}")
    # Extract image indices from filenames
    img_indices = [int(f.split('_')[2].split('.')[0]) for f in goal_mask_files]
    # Get the last image index
    goal_img_idx = max(img_indices)
    # Load the corresponding goal mask
    goal_mask_path = os.path.join(episode_path, f"goal_mask_{goal_img_idx:05d}.png")
    goal_mask = cv2.imread(goal_mask_path, cv2.IMREAD_GRAYSCALE)
    if goal_mask is None:
        raise FileNotFoundError(f"Could not load goal mask from {goal_mask_path}")
    # Binarize the mask
    _, goal_mask = cv2.threshold(goal_mask, 127, 1, cv2.THRESH_BINARY)
    return goal_img_idx, goal_mask