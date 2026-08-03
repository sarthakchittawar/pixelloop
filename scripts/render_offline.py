#!/usr/bin/env python3
"""
Offline visualization renderer.

Generates visualizations from saved episode data without needing the simulator.

Usage:
    # Single episode
    python scripts/render_offline.py --episode_path /path/to/episode_results
    
    # All episodes in experiment directory
    python scripts/render_offline.py --experiment_dir /path/to/experiment_results
    
    # With custom config
    python scripts/render_offline.py --episode_path /path/to/results --config configs/visualization/visualization_offline.yaml
"""
import argparse
import json
import sys
from pathlib import Path
import os

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.visualizations.vis_renderer import VisualizationRenderer

# Default config - everything enabled
DEFAULT_VIS_CONFIG = {
    'render_visualizations': {
        'enabled': True,
        'online': True,
        'rgb_images': False,
        'depth_images': False,
        'costmap_heatmaps': False,
        'match_vis': False,
        'topdown_trajectory': True,
        'topdown_meters_per_pixel': 0.025,
        'combined_step_vis': False,
        'waypoint_topdown_vis': False,
        'care_vis': False,
    },
    'offline_vis': {
        'create_videos': True,
        'video_fps': 5,
        'videos_only': False,
    }
}

# Default base directory for mapping episodes (can be overridden via --mapping_base_dir)
DEFAULT_MAPPING_EPISODES_BASE_DIR = "/scratch2/public_scratch/toponav/indoor-topo-loc/datasets/hm3d-0.2/benchmarking"
MAPPING_EPISODE_NAME = "1W61QJVDBqe"

class AgentStateOffline:
    """Wraps saved state dict to match AgentState interface."""
    def __init__(self, state_dict):
        self.position = np.array(state_dict['position'])
        self.rotation = np.array(state_dict['rotation'])


class EpisodeDataLoader:
    """
    Loads saved episode data and provides it step-by-step.
    Mimics online inference data flow for offline visualization.
    """
    
    def __init__(self, episode_results_path: Path, mapping_img_dir: Path = None):
        self.episode_path = Path(episode_results_path)
        self.step_data_dir = self.episode_path / "step_data"
        self.topdown_dir = self.episode_path / "topdown_data"
        
        # Mapping images directory for loading reference images
        self._mapping_img_dir = mapping_img_dir
        
        # Discover all steps
        self._step_dirs = sorted(self.step_data_dir.glob("step_*"))
        self._num_steps = len(self._step_dirs)
        
        # Trajectory history (builds up as we iterate)
        self._trajectory_history = []
        
        # Episode-level data (loaded once)
        self._topdown_metadata = None
        self._start_position = None
        self._goal_position = None
        self._pathfinder_bounds = None
        self._topdown_base_map = None
        
        self._load_episode_metadata()
    
    def set_mapping_img_dir(self, mapping_img_dir: Path):
        """Set the mapping images directory for loading reference images."""
        self._mapping_img_dir = Path(mapping_img_dir) if mapping_img_dir else None
    
    def _load_episode_metadata(self):
        """Load episode-level data from topdown_metadata.json."""
        metadata_path = self.topdown_dir / "topdown_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Topdown metadata not found: {metadata_path}")
        
        with open(metadata_path) as f:
            meta = json.load(f)
        
        self._start_position = np.array(meta['start_position'])
        self._goal_position = np.array(meta['goal_position'])
        self._pathfinder_bounds = (
            np.array(meta['pathfinder_bounds']['lower']),
            np.array(meta['pathfinder_bounds']['upper'])
        )
        self._topdown_metadata = meta
        
        # Load base map
        base_map_path = self.topdown_dir / "topdown_base_map.png"
        if base_map_path.exists():
            self._topdown_base_map = cv2.cvtColor(cv2.imread(str(base_map_path)), cv2.COLOR_BGR2RGB)
    
    @property
    def num_steps(self) -> int:
        return self._num_steps
    
    @property
    def start_position(self) -> np.ndarray:
        return self._start_position
    
    @property
    def goal_position(self) -> np.ndarray:
        return self._goal_position
    
    @property
    def pathfinder_bounds(self) -> tuple:
        return self._pathfinder_bounds
    
    @property
    def trajectory_history(self) -> list:
        """Current trajectory history (grows as steps are loaded)."""
        return self._trajectory_history
    
    def get_step_data(self, step: int) -> dict:
        """
        Load and return all data for a single step.
        Also updates internal trajectory history.
        
        Returns dict with keys matching render_step_visualizations() args.
        """
        step_dir = self.step_data_dir / f"step_{step:04d}"
        
        # Load per-step data
        rgb = self._load_rgb(step_dir)
        depth = self._load_depth(step_dir)
        costmap = self._load_costmap(step_dir)
        agent_state = self._load_agent_state(step_dir)
        waypoints = self._load_waypoints(step_dir)
        matches_data = self._load_matches(step_dir)
        care_data = self._load_care_data(step_dir)
        
        # Extract adjusted waypoints from care_data if available
        waypoints_adjusted = None
        if care_data is not None:
            waypoints_adjusted = care_data.get('waypoints_adjusted')
        
        # Load reference image if mapping dir is set and matches data available
        ref_img, ref_img_idx = self._load_ref_img(matches_data)
        
        # Extract agent pose from agent_state
        agent_position = agent_state.position if agent_state is not None else None
        agent_rotation = agent_state.rotation if agent_state is not None else None
        
        # Update trajectory history
        if agent_state is not None:
            self._trajectory_history.append(agent_state)
        
        return {
            'step': step,
            'rgb': rgb,
            'depth': depth,
            'costmap': costmap,
            'waypoints': waypoints,
            'waypoints_adjusted': waypoints_adjusted,
            'care_data': care_data,
            'matches_data': matches_data,
            'ref_img': ref_img,
            'ref_img_idx': ref_img_idx,
            'trajectory_history': list(self._trajectory_history),  # Copy
            'start_position': self._start_position,
            'goal_position': self._goal_position,
            'agent_position': agent_position,
            'agent_rotation': agent_rotation,
        }
    
    def _load_rgb(self, step_dir: Path):
        path = step_dir / "rgb.png"
        if path.exists():
            return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return None
    
    def _load_costmap(self, step_dir: Path):
        path = step_dir / "costmap.npy"
        return np.load(path) if path.exists() else None
    
    def _load_depth(self, step_dir: Path):
        """Load depth as float32 in meters from PNG (stored as uint16 mm).
        
        Falls back to computing depth from pts3d if depth.png is not available.
        """
        from libs.visualizations.data_storage import load_depth_png
        
        # Try loading depth.png first
        path = step_dir / "depth.png"
        if path.exists():
            return load_depth_png(path)
        
        # Fallback: compute depth from pts3d
        pts3d_path = step_dir / "pts3d.npy"
        if pts3d_path.exists():
            pts3d = np.load(pts3d_path)
            Z = pts3d[..., 2]
            depth_from_pts3d = np.where(Z > 0, Z, 0.0).astype(np.float32)
            return depth_from_pts3d
        
        return None
    
    def _load_agent_state(self, step_dir: Path):
        path = step_dir / "agent_state.npy"
        if path.exists():
            state_dict = np.load(path, allow_pickle=True).item()
            return AgentStateOffline(state_dict)
        return None
    
    def _load_waypoints(self, step_dir: Path):
        path = step_dir / "waypoints.npy"
        return np.load(path) if path.exists() else None
    
    def _load_matches(self, step_dir: Path):
        path = step_dir / "matches.npz"
        if path.exists():
            data = np.load(path)
            # Handle closest_map_img_idx - could be scalar, 0-d array, or 1-d array
            closest_idx = None
            if 'closest_map_img_idx' in data:
                idx_arr = data['closest_map_img_idx']
                if idx_arr.ndim == 0:
                    closest_idx = int(idx_arr.item())  # 0-d array
                else:
                    closest_idx = int(idx_arr[0])  # 1-d array
            
            # Keys from data_storage: qry_mkpts, ref_mkpts, confidences
            return {
                'qry_mkpts': data.get('qry_mkpts'),       # (N, 2) query keypoints
                'ref_mkpts': data.get('ref_mkpts'),       # (N, 2) reference keypoints
                'confidences': data.get('confidences'),   # (N,) confidence scores
                'closest_map_img_idx': closest_idx
            }
        return None
    
    def _load_care_data(self, step_dir: Path):
        """Load CARE collision avoidance data if available."""
        from libs.visualizations.data_storage import load_care_npz
        return load_care_npz(step_dir / "care_data.npz")
    
    def _load_ref_img(self, matches_data: dict):
        """Load reference image based on matches data."""
        if not self._mapping_img_dir or not matches_data:
            return None, None
        
        ref_img_idx = matches_data.get('closest_map_img_idx')
        if ref_img_idx is None:
            return None, None
        
        ref_img_path = self._mapping_img_dir / f"{ref_img_idx:05d}.jpg"
        
        if ref_img_path.exists():
            ref_img = cv2.cvtColor(cv2.imread(str(ref_img_path)), cv2.COLOR_BGR2RGB)
            return ref_img, ref_img_idx
        
        return None, ref_img_idx
    
    def __iter__(self):
        """Iterate over all steps."""
        self._trajectory_history = []  # Reset on new iteration
        for step in range(self._num_steps):
            yield self.get_step_data(step)


def render_episode(episode_path: Path, vis_cfg, episode_name: str = None, mapping_base_dir: Path = None, verbose: bool = True, videos_only: bool = False) -> bool:
    """
    Render visualizations for a single episode.
    
    Args:
        episode_path: Path to episode results directory
        vis_cfg: Visualization config
        mapping_base_dir: Base directory for mapping episodes (for ref images)
        verbose: Print progress
        videos_only: If True, skip frame rendering and only create videos
    
    Returns True on success, False on failure.
    """
    try:
        loader = EpisodeDataLoader(episode_path)
        if loader.num_steps == 0:
            if verbose:
                print(f"  Skipping (no steps found): {episode_path.name}")
            return False
        
        if verbose:
            print(f"  Processing {episode_path.name} ({loader.num_steps} steps)...")
        
        # Determine mapping episode images directory and set on loader
        # Episode result folder is like: "wcojb4TFT35_0000000_bed_17__learnt_topological_pixelwise"
        # Mapping episode folder is like: "wcojb4TFT35_0000000_bed_17_" (with trailing underscore)
        print(f"{mapping_base_dir = } | {os.path.exists(mapping_base_dir)}")
        if mapping_base_dir:
            # Extract mapping episode name (before double underscore, which becomes single)
            # "abc_xyz__method" -> split on "__" gives "abc_xyz" but folder is "abc_xyz_"
            if episode_name is not None:
                mapping_episode_name = episode_name
            else:
                episode_name = episode_path.name
                if "__" in episode_name:
                    # Split on "__" and add trailing underscore
                    mapping_episode_name = episode_name.split("__")[0] + "_"
                else:
                    mapping_episode_name = episode_name.rsplit("_", 2)[0]  # Fallback
            
            print(f"{mapping_episode_name = }")
            
            mapping_episode_path = Path(mapping_base_dir) / mapping_episode_name / "images_fov90"
            print(f"{mapping_episode_path = } | {mapping_episode_path.exists() = }")
            
            if verbose:
                print(f"    Mapping base dir: {mapping_base_dir}")
                print(f"    Mapping episode name: {mapping_episode_name}")
                print(f"    Mapping images path: {mapping_episode_path}")
                print(f"    Exists: {mapping_episode_path.exists()}")
            
            if mapping_episode_path.exists():
                loader.set_mapping_img_dir(mapping_episode_path)
                if verbose:
                    print(f"    -> Set mapping images directory")
        
        # Create renderer
        renderer = VisualizationRenderer(
            vis_cfg=vis_cfg,
            episode_results_path=episode_path,
            output_subdir="visualizations_offline"
        )
        
        # Initialize offline mode
        renderer.init_offline_mode(loader.pathfinder_bounds)
        
        # Load topdown base map if available
        if loader._topdown_base_map is not None:
            renderer._topdown_base_map = loader._topdown_base_map
            renderer._topdown_dims = (loader._topdown_base_map.shape[0], loader._topdown_base_map.shape[1])
            renderer._start_position = loader.start_position
            renderer._goal_position = loader.goal_position
        
        # Process each step - loader now includes ref_img and ref_img_idx
        if not videos_only:
            desc = f"    {episode_path.name[:30]}..." if len(episode_path.name) > 30 else f"    {episode_path.name}"
            for step_data in tqdm(loader, total=loader.num_steps, desc=desc, leave=False):
                renderer.render_step_visualizations(**step_data)
            
            if verbose:
                print(f"    Done ({loader.num_steps} frames)")
        else:
            if verbose:
                print(f"    Skipping frame rendering (videos_only mode)")
        
        # Create videos if enabled
        offline_vis_cfg = vis_cfg.get('offline_vis', {})
        if offline_vis_cfg.get('create_videos', True):
            fps = offline_vis_cfg.get('video_fps', 5)
            video_results = renderer.create_all_videos(fps=fps)
            if verbose:
                created = sum(1 for v in video_results.values() if v)
                print(f"    Created {created} videos")
        
        return True
    
    except Exception as e:
        import traceback
        if verbose:
            print(f"  Error processing {episode_path.name}: {e}")
            traceback.print_exc()
        return False


def find_episode_dirs(experiment_dir: Path) -> list:
    """
    Find all episode directories in an experiment directory.
    An episode directory is identified by having a topdown_data subdirectory.
    """
    episode_dirs = []
    for subdir in sorted(experiment_dir.iterdir()):
        if subdir.is_dir():
            # Check if it's an episode directory (has topdown_data or step_data)
            if (subdir / "topdown_data").exists() or (subdir / "step_data").exists():
                episode_dirs.append(subdir)
    return episode_dirs


def main():
    parser = argparse.ArgumentParser(description="Offline visualization renderer")
    parser.add_argument("--episode_path", type=Path, default=None, 
                        help="Path to single episode results")
    parser.add_argument("--experiment_dir", type=Path, default=None,
                        help="Path to experiment directory (processes all episodes)")
    parser.add_argument("--config", type=Path, default=None, 
                        help="Optional visualization config file")
    parser.add_argument("--mapping_base_dir", type=Path, default=None,
                        help=f"Base directory for mapping episodes (default: {DEFAULT_MAPPING_EPISODES_BASE_DIR})")
    parser.add_argument("--videos_only", action="store_true",
                        help="Skip frame rendering, only create videos from existing frames")
    args = parser.parse_args()
    
    # Validate arguments
    if args.episode_path is None and args.experiment_dir is None:
        print("Error: Must provide either --episode_path or --experiment_dir")
        return 1
    
    if args.episode_path is not None and args.experiment_dir is not None:
        print("Error: Cannot provide both --episode_path and --experiment_dir")
        return 1
    
    # Load config
    if args.config and args.config.exists():
        print(f"Loading config from: {args.config}")
        vis_cfg = OmegaConf.load(args.config)
    else:
        print("Using default config (all visualizations enabled)")
        vis_cfg = OmegaConf.create(DEFAULT_VIS_CONFIG)
    
    # Determine mapping base directory
    mapping_base_dir = args.mapping_base_dir or Path(DEFAULT_MAPPING_EPISODES_BASE_DIR)
    if not mapping_base_dir.exists():
        print(f"Warning: Mapping base dir does not exist: {mapping_base_dir}")
        print("  Reference images will not be loaded. Use --mapping_base_dir to specify.")
        mapping_base_dir = None
    else:
        print(f"Using mapping base dir: {mapping_base_dir}")
    
    # Single episode mode
    if args.episode_path:
        if not args.episode_path.exists():
            print(f"Error: Episode path does not exist: {args.episode_path}")
            return 1
        
        print(f"Processing single episode: {args.episode_path}")
        videos_only = args.videos_only or vis_cfg.get('offline_vis', {}).get('videos_only', False)
        success = render_episode(args.episode_path, vis_cfg, MAPPING_EPISODE_NAME, mapping_base_dir, videos_only=videos_only)
        if success:
            print(f"\nDone! Visualizations saved to: {args.episode_path / 'visualizations_offline'}")
            return 0
        else:
            return 1
    
    # Experiment directory mode
    if args.experiment_dir:
        if not args.experiment_dir.exists():
            print(f"Error: Experiment directory does not exist: {args.experiment_dir}")
            return 1
        
        print(f"Scanning experiment directory: {args.experiment_dir}")
        episode_dirs = find_episode_dirs(args.experiment_dir)
        
        if not episode_dirs:
            print("No episode directories found")
            return 1
        
        print(f"Found {len(episode_dirs)} episodes\n")
        
        success_count = 0
        error_count = 0
        skip_count = 0
        
        videos_only = args.videos_only or vis_cfg.get('offline_vis', {}).get('videos_only', False)
        
        for episode_path in episode_dirs:
            result = render_episode(episode_path, vis_cfg, mapping_base_dir, videos_only=videos_only)
            if result:
                success_count += 1
            else:
                error_count += 1
        
        print(f"\n{'='*50}")
        print(f"Summary: {success_count} succeeded, {error_count} failed/skipped")
        print(f"Output saved to: <episode>/visualizations_offline/")
        
        return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
