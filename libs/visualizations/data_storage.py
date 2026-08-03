"""
Visualization Data Storage Module

Manages collection and storage of visualization data during navigation.
Implements hybrid file structure: raw data per-step, visualizations per-type.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any
from omegaconf import DictConfig
import logging

logger = logging.getLogger("[VisualizationDataCollector]")


class VisualizationDataCollector:
    """
    Single-class solution for managing all visualization data storage.
    
    Implements hybrid file structure:
    - Raw data organized per-step in step_data/
    - Visualizations organized per-type in visualizations/
    
    Features:
    - Lazy directory creation
    - Format-agnostic saving based on config
    - Handles both per-step and episode-level data
    """
    
    def __init__(
        self,
        episode_results_path: Path,
        cfg: DictConfig,
        start_position: np.ndarray,
        goal_position: np.ndarray
    ):
        """
        Initialize data collector with episode metadata.
        
        Args:
            episode_results_path: Root path for episode results
            cfg: Hydra config with visualization settings
            start_position: Agent start position (3,)
            goal_position: Goal position (3,)
        """
        self.episode_results_path = Path(episode_results_path)
        self.cfg = cfg
        self.enabled = cfg.visualization.save_raw_data.enabled  # Master toggle for raw data
        
        # Episode metadata (stored as instance variables - no separate class needed)
        self.start_position = start_position
        self.goal_position = goal_position
        self.steps_saved = []
        
        # Directory paths (created lazily)
        self.step_data_dir = self.episode_results_path / "step_data"
        self.vis_dir = self.episode_results_path / "visualizations"
        
        # Track if directories have been created
        self._dirs_created = False
        
        logger.info(f"VisualizationDataCollector initialized: enabled={self.enabled}, path={episode_results_path}")
    
    def _ensure_dirs_created(self):
        """Lazily create directory structure on first save."""
        if self._dirs_created or not self.enabled:
            return
        
        self.step_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created step_data directory: {self.step_data_dir}")
        self._dirs_created = True
    
    def save_topdown_data(
        self, 
        sim,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        meters_per_pixel: float = 0.025
    ):
        """
        Save topdown base map and metadata for offline rendering.
        
        This should be called once at episode start. Saves:
        - topdown_base_map.png: The navigable area image
        - topdown_metadata.json: Dimensions, bounds, and positions
        
        Args:
            sim: Habitat simulator with pathfinder
            start_position: Agent start position [x, y, z]
            goal_position: Goal position [x, y, z]
            meters_per_pixel: Map resolution (default 0.025)
        """
        if not self.enabled:
            return
        
        from habitat.utils.visualizations import maps
        
        self._ensure_dirs_created()
        
        # Create topdown directory
        topdown_dir = self.episode_results_path / "topdown_data"
        topdown_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base topdown map
        height = start_position[1]
        top_down_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        
        # Recolor: 0=navigable->white, 1=border->gray, 2=obstacle->black
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        tdv = recolor_map[top_down_map]
        
        # Save base map as PNG
        base_map_path = topdown_dir / "topdown_base_map.png"
        cv2.imwrite(str(base_map_path), cv2.cvtColor(tdv, cv2.COLOR_RGB2BGR))
        
        # Get pathfinder bounds for coordinate conversion
        bounds = sim.pathfinder.get_bounds()
        
        # Helper to convert numpy types to native Python types
        def to_native(val):
            if hasattr(val, 'tolist'):
                return val.tolist()
            elif hasattr(val, 'item'):
                return val.item()
            return val
        
        # Save metadata (convert all numpy types to native Python)
        metadata = {
            'map_height': int(tdv.shape[0]),
            'map_width': int(tdv.shape[1]),
            'meters_per_pixel': float(meters_per_pixel),
            'start_position': [float(x) for x in start_position],
            'goal_position': [float(x) for x in goal_position],
            'slice_height': float(height),
            'pathfinder_bounds': {
                'lower': [float(x) for x in bounds[0]],
                'upper': [float(x) for x in bounds[1]]
            }
        }
        
        metadata_path = topdown_dir / "topdown_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved topdown data: base_map {tdv.shape}, metadata to {topdown_dir}")
    
    def save_step_data(
        self,
        step: int,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        pts3d: Optional[np.ndarray] = None,
        costmap: Optional[np.ndarray] = None,
        matches_data: Optional[Dict[str, Any]] = None,
        velocity: Optional[float] = None,
        theta: Optional[float] = None,
        waypoints: Optional[np.ndarray] = None,
        agent_state: Optional[dict] = None,
        collided: Optional[bool] = None,
        distance_to_goal: Optional[float] = None,
        care_data: Optional[Dict[str, Any]] = None
    ):
        """
        Save all data for a single navigation step.
        
        Args:
            step: Step index
            rgb: RGB image (H, W, 3) uint8
            depth: Depth map (H, W) float32 in meters
            pts3d: 3D point cloud (H, W, 3) float32
            costmap: Distance-to-goal costmap (H, W) float32
            matches_data: Dict with keys: 
                - qry_img_idx: query image index
                - closest_map_img_idx: closest/best match map image index
                - localized_img_idxs: candidate submap indices
                - qry_mkpts: (N, 2) query pixels
                - ref_mkpts: (N, 2) reference pixels
                - confidences: (N,) match confidences
            velocity: Linear velocity
            theta: Angular velocity
            waypoints: Controller waypoints (original, before CARE adjustment)
            agent_state: Dict with 'position' (3,) and 'rotation' (4,) quaternion
            collided: Whether collision occurred this step
            distance_to_goal: Current distance to goal in meters
            care_data: CARE collision avoidance data dict with keys:
                - waypoints_adjusted: (K, D) adjusted waypoints
                - obstacles: (M, 2) top-down obstacle points
                - theta_rot: rotation angle applied
                - k_star: waypoint index with max repulsive force
                - frep_all: (K, 2) repulsive force vectors
                - v, omega: computed velocity commands
        """
        if not self.enabled:
            return
        
        self._ensure_dirs_created()
        
        # Create step directory
        step_dir = self.step_data_dir / f"step_{step:04d}"
        step_dir.mkdir(exist_ok=True)
        
        # Save based on config toggles
        if rgb is not None and self.cfg.visualization.save_raw_data.rgb:
            self._save_rgb(step_dir, rgb)
        
        if depth is not None and self.cfg.visualization.save_raw_data.depth:
            self._save_depth_png(step_dir, depth)
        
        if pts3d is not None and self.cfg.visualization.save_raw_data.pts3d:
            self._save_pts3d(step_dir, pts3d)
        
        if costmap is not None and self.cfg.visualization.save_raw_data.costmaps:
            self._save_costmap(step_dir, costmap)
        
        if matches_data is not None and self.cfg.visualization.save_raw_data.matches:
            self._save_matches_npz(step_dir, matches_data)
        
        if waypoints is not None and self.cfg.visualization.save_raw_data.waypoints:
            self._save_waypoints(step_dir, waypoints)
        
        if velocity is not None and theta is not None and self.cfg.visualization.save_raw_data.control_outputs:
            self._save_control_outputs(step_dir, velocity, theta)
        
        if agent_state is not None and self.cfg.visualization.save_raw_data.agent_state:
            self._save_agent_state(step_dir, agent_state)
        
        if collided is not None and self.cfg.visualization.save_raw_data.collision:
            self._save_collision(step_dir, collided)
        
        if distance_to_goal is not None and self.cfg.visualization.save_raw_data.distance_to_goal:
            self._save_distance_to_goal(step_dir, distance_to_goal)
        
        if care_data is not None and getattr(self.cfg.visualization.save_raw_data, 'care_data', False):
            self._save_care_npz(step_dir, care_data)
        
        self.steps_saved.append(step)
        logger.debug(f"Saved step {step} data to {step_dir}")
    
    def save_episode_metadata(self, **extra_metadata):
        """
        Save episode-level metadata to metadata.json.
        
        Args:
            **extra_metadata: Additional metadata fields to save
        """
        if not self.enabled:
            return
        
        metadata = {
            'start_position': self.start_position.tolist(),
            'goal_position': self.goal_position.tolist(),
            'total_steps': len(self.steps_saved),
            'steps_saved': self.steps_saved,
            **extra_metadata
        }
        
        metadata_path = self.episode_results_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved episode metadata to {metadata_path}")
    
    # ========== Internal Save Methods ==========
    
    def _save_rgb(self, step_dir: Path, rgb: np.ndarray):
        """Save RGB image as PNG."""
        path = step_dir / "rgb.png"
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    
    def _save_depth_png(self, step_dir: Path, depth_meters: np.ndarray):
        """
        Save depth as PNG (uint16, millimeters).
        
        Converts from meters (float32) to millimeters (uint16) for storage.
        Compatible with standard depth PNG format.
        """
        path = step_dir / "depth.png"
        # Convert meters to millimeters, clip to uint16 range
        depth_mm = (depth_meters * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(path), depth_mm)
    
    def _save_pts3d(self, step_dir: Path, pts3d: np.ndarray):
        """Save 3D point cloud as NPY."""
        path = step_dir / "pts3d.npy"
        np.save(path, pts3d)
    
    def _save_costmap(self, step_dir: Path, costmap: np.ndarray):
        """Save costmap as NPY."""
        path = step_dir / "costmap.npy"
        np.save(path, costmap)
    
    def _save_matches_npz(self, step_dir: Path, matches_data: Dict[str, Any]):
        """
        Save match data as NPZ file.
        
        Expected matches_data keys:
        - qry_img_idx: int
        - closest_map_img_idx: int (closest/best match map image)
        - closest_map_img_idx: int (closest/best match)
        - localized_img_idxs: array of candidate indices (submap)
        - qry_mkpts: (N, 2) array
        - ref_mkpts: (N, 2) array
        - confidences: (N,) array
        """
        path = step_dir / "matches.npz"
        np.savez(
            path,
            qry_img_idx=matches_data.get('qry_img_idx', -1),
            closest_map_img_idx=matches_data['closest_map_img_idx'],
            localized_img_idxs=matches_data.get('localized_img_idxs', np.array([])),
            qry_mkpts=matches_data['qry_mkpts'],
            ref_mkpts=matches_data['ref_mkpts'],
            confidences=matches_data['confidences']
        )
    
    def _save_waypoints(self, step_dir: Path, waypoints: np.ndarray):
        """Save waypoints as NPY."""
        path = step_dir / "waypoints.npy"
        np.save(path, waypoints)
    
    def _save_control_outputs(self, step_dir: Path, velocity: float, theta: float):
        """Save control outputs as text files."""
        velocity_path = step_dir / "velocity.txt"
        theta_path = step_dir / "theta.txt"
        
        with open(velocity_path, 'w') as f:
            f.write(f"{velocity}\n")
        
        with open(theta_path, 'w') as f:
            f.write(f"{theta}\n")
    
    def _save_agent_state(self, step_dir: Path, agent_state: dict):
        """
        Save agent state (position and rotation) as NPY.
        
        Args:
            step_dir: Directory to save to
            agent_state: Dict with 'position' (3,) and 'rotation' (4,) quaternion [w, x, y, z]
        """
        path = step_dir / "agent_state.npy"
        np.save(path, agent_state, allow_pickle=True)
    
    def _save_collision(self, step_dir: Path, collided: bool):
        """Save collision status as text file."""
        with open(step_dir / "collided.txt", 'w') as f:
            f.write("1" if collided else "0")
    
    def _save_distance_to_goal(self, step_dir: Path, distance_to_goal: float):
        """Save distance to goal as text file."""
        with open(step_dir / "distance_to_goal.txt", 'w') as f:
            f.write(f"{distance_to_goal:.6f}")
    
    def _save_care_npz(self, step_dir: Path, care_data: Dict[str, Any]):
        """
        Save CARE collision avoidance data as NPZ file.
        
        Args:
            step_dir: Directory to save to
            care_data: Dict with CARE outputs:
                - waypoints_adjusted: (K, D) adjusted waypoints
                - obstacles: (M, 2) top-down obstacle points (x_fwd, y_left)
                - theta_rot: rotation angle applied
                - k_star: waypoint index with max repulsive force
                - frep_all: (K, 2) repulsive force vectors
                - v, omega: computed velocity commands
        """
        path = step_dir / "care_data.npz"
        # Filter out None values before saving
        data_to_save = {k: v for k, v in care_data.items() if v is not None}
        np.savez(path, **data_to_save)


def load_depth_png(depth_png_path: Path) -> np.ndarray:
    """
    Load depth from PNG (uint16, millimeters) and convert to meters.
    
    Args:
        depth_png_path: Path to depth PNG file
        
    Returns:
        Depth map in meters (float32)
    """
    depth_raw = cv2.imread(str(depth_png_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Failed to read depth PNG: {depth_png_path}")
    
    depth_meters = depth_raw.astype(np.float32) * 0.001
    return depth_meters


def load_matches_npz(matches_npz_path: Path) -> Dict[str, Any]:
    """
    Load match data from NPZ file.
    
    Args:
        matches_npz_path: Path to matches NPZ file
        
    Returns:
        Dict with keys: qry_img_idx, closest_map_img_idx, 
                       localized_img_idxs, qry_mkpts, ref_mkpts, confidences
    """
    data = np.load(matches_npz_path)
    return {
        'qry_img_idx': int(data.get('qry_img_idx', -1)),
        'closest_map_img_idx': int(data['closest_map_img_idx']),
        'localized_img_idxs': data.get('localized_img_idxs', np.array([])),
        'qry_mkpts': data['qry_mkpts'],
        'ref_mkpts': data['ref_mkpts'],
        'confidences': data['confidences']
    }


def load_care_npz(care_npz_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load CARE collision avoidance data from NPZ file.
    
    Args:
        care_npz_path: Path to care_data.npz file
        
    Returns:
        Dict with CARE data, or None if file doesn't exist.
        Keys may include: waypoints_adjusted, obstacles, theta_rot, 
                         k_star, frep_all, v, omega
    """
    if not care_npz_path.exists():
        return None
    
    data = np.load(care_npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}
