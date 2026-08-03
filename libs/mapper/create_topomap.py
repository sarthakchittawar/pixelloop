import os
import time
import numpy as np
import cv2
import json
import scipy
from tqdm import tqdm
from typing import Tuple
import networkx as nx
from itertools import combinations
# import open3d as o3d
from natsort import natsorted
import blosc2
import pickle
import sys
from pathlib import Path
import shutil
from enum import Enum
from PIL import Image
from dataclasses import dataclass

from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
import torch
import torchvision.transforms as tfm

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

# Add third-party libraries to path
BASE_DIR = Path(__file__).parent.parent.parent
MAST3R_PATH = BASE_DIR / "libs" / "matcher" / "mast3r"

# Add to Python path if not already there
if str(MAST3R_PATH) not in sys.path:
    sys.path.insert(0, str(MAST3R_PATH))

# Import geometry utilities
from libs.common.geometry_utils import (
    farthest_point_sampling_o3d,
    resize_to_divisible,
    pixel_to_camera_3d,
    get_mask_centroid
)

from libs.common.graph_utils import load_compressed_graph_chunked

# Import timing manager
from libs.timing_manager import timing_manager, time_function

# Import our MASt3R wrapper
from libs.mast3r_utils import MASt3RInference

# Import matchers
from libs.matcher.mast3r_matcher import Mast3rMatcher
from libs.matcher.superpoint_matcher import SuperPointMatcher

# MASt3R specific imports for loop closure
try:
    from mast3r.retrieval.processor import Retriever
    from mast3r.retrieval.graph import farthest_point_sampling as mast3r_fps
    from mast3r.retrieval.model import extract_local_features
    RETRIEVAL_AVAILABLE = True
except ImportError:
    print("Warning: MASt3R retrieval not available. Loop closure features will be limited.")
    RETRIEVAL_AVAILABLE = False

class NodeCullingMode(Enum):
    NONE = "NONE"
    FPS = "FPS"

class EdgeCullingMode(Enum):
    NONE = "NONE"
    EMST_SINGLE = "EMST_SINGLE"
    DELAUNAY_3D = "DELAUNAY_3D"

@dataclass
class CostmapData:
    costmaps: np.ndarray  # shape: (N_images, H, W)
    metadata: dict

    @staticmethod
    def from_npz(npz_path):
        """Load costmap data from NPZ file"""
        data = np.load(npz_path, allow_pickle=True)
        costmaps = data['costmaps']
        metadata = json.loads(data['metadata'].item())
        return CostmapData(costmaps=costmaps, metadata=metadata)
    
    def get_costmap(self):
        return self.costmaps
    
    def get_metadata(self):
        return self.metadata

def read_oracle_lc_pairs(txt_path, num_imgs):
    """Read oracle loop closure pairs from text file"""
    pairs = set()
    with open(txt_path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = map(int, line.replace(",", " ").split())
            if a == b:
                continue
            if 0 <= a < num_imgs and 0 <= b < num_imgs:
                pairs.add(tuple(sorted((a, b))))
            else:
                print(f"[WARN] Skipping invalid LC pair at line {ln}: {a}, {b}")
    return list(pairs)

class MapTopological3DPoints:
    def __init__(self, img_dir: str, out_dir: str, cfg: DictConfig):
        self.cfg = cfg
        # self.cfg.update(cfg)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        print("\n" + "="*80)
        print("INITIALIZING TOPOLOGICAL MAPPER")
        print("="*80)
        print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

        # Directory paths
        self.img_dir = Path(img_dir)
        self.scene_dir = self.img_dir.parent
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Image configuration
        self.W = cfg.image.width
        self.H = cfg.image.height
        self.hfov = cfg.image.hfov
        self.device = cfg.model.device
        self.img_match_window_size = cfg.graph.inter_image_match_window_size

        # Load and sort image names
        self.img_names = natsorted(os.listdir(self.img_dir))
        self.img_paths = [self.img_dir / img_name for img_name in self.img_names]
        print(f"Found {len(self.img_paths)} images in {self.img_dir}")
        
        # Apply subsampling
        self.img_paths = self._subsample_images()
        print(f"After subsampling, {len(self.img_paths)} images will be used.")

        # optionally copy them (useful when original images are in a different dir)
        if cfg.processing.copy_images:
            self._copy_images_to_output_dir()

        # init other variables
        self.G, self.nodeID_to_imgRegionIdx = None, None
        self.inter_image_edges = {}
        self.intra_image_edges = {}
        self.pixel_to_node_id = {}  # Mapping for sparse graph
        self.force_recompute_graph = cfg.processing.force_recompute_graph
        self.recreate_graphs = cfg.processing.get("recreate_graphs", False)

        # Depth source: "mast3r" (default), "gt", or "mapanything"
        self.depth_source = cfg.model.get("depth_source", "mast3r")
        # Backward compat: if use_gt_depth is set, override depth_source
        if cfg.model.get("use_gt_depth", False) and self.depth_source == "mast3r":
            self.depth_source = "gt"

        # Output file paths
        self.pc_npz_path = self.out_dir / f"nodes_{self.depth_source}_points.npz"
        self.graph_intra_path = self.out_dir / "graph_intra_edges.pickle"
        self.graph_inter_path = self.out_dir / "graph_just_inter_edges.pickle"
        self.graph_path = self.out_dir / "graph_mast3r_intra_edges_with_weights.pickle"

        # GT matches configuration
        self.use_gt_matches = cfg.model.get("use_gt_matches", False)
        self.gt_depth_mode = cfg.model.get("gt_depth_mode", "raw")
        
        if self.use_gt_matches:
            self._setup_gt_mode()
        elif self.depth_source == "mapanything":
            self._setup_mapanything_mode()
        else:
            if self.depth_source == "gt":
                self._setup_gt_mode()
            # Load MASt3R model using wrapper (for depth estimation)
            self.model_path = cfg.model.path
            self.mast3r_match_subsample = self.cfg.model.subsample_or_initxy1
            self.mast3r = MASt3RInference(model_path=self.model_path, device=self.device)
            
            # Initialize retriever for loop closure if enabled
            if cfg.graph.get("enable_loop_closure", False) and RETRIEVAL_AVAILABLE:
                retrieval_path = cfg.model.get("retrieval_path", None)
                if retrieval_path:
                    self.retriever = Retriever(retrieval_path, backbone=self.mast3r.model, device=self.device)
                else:
                    print("Warning: Loop closure enabled but no retrieval_path provided")
                    self.retriever = None
            else:
                self.retriever = None
        
            # Initialize the matcher for 2D-2D matching (separate from depth estimation)
            self._init_matcher()
    
    def _setup_mapanything_mode(self):
        """Setup MapAnything depth mode.
        
        No MASt3R model needed for depth. Matcher is still initialized
        for 2D-2D matching unless use_gt_matches is True.
        """
        print("\n" + "="*50)
        print("MAPANYTHING DEPTH MODE ENABLED")
        print("="*50 + "\n")

        self.model = None  # No MASt3R model needed for depth
        self.mast3r = None
        self.retriever = None

        # Determine GT matches path for mapanything LC mode
        lc_mode = self.cfg.graph.get("loop_closure_mode", "oracle")
        gt_matches_filename = self.cfg.model.get("gt_matches_filename", None)
        if gt_matches_filename:
            self.gt_matches_path = self.scene_dir / "gt_matches" / gt_matches_filename
        else:
            self.gt_matches_path = self.scene_dir / "gt_matches" / f"matches_fov90_{lc_mode}.pkl"

        # Validate that MapAnything pointmaps exist
        cam_dir = self.scene_dir / "mapanything_pointmaps_cam_fov90"
        if cam_dir.exists():
            num_files = len([f for f in os.listdir(cam_dir) if f.endswith(".npy")])
            print(f"  Found {num_files} MapAnything camera-frame pointmaps in {cam_dir}")
        else:
            print(f"  WARNING: {cam_dir} not found. Will attempt to load from mapanything_outputs/ at runtime.")

        # Initialize matcher for 2D-2D matching (if not using GT matches)
        if not self.use_gt_matches:
            self._init_matcher()

    def _setup_gt_mode(self):
        """Setup ground-truth matches mode"""
        print("\n" + "="*50)
        print("GT MODE ENABLED")
        print(f"  gt_depth_mode: {self.gt_depth_mode}")
        print("="*50 + "\n")
        
        self.model = None  # No MASt3R model needed
        gt_matches_filename = self.cfg.model.get("gt_matches_filename", None)
        if gt_matches_filename:
            self.gt_matches_path = self.scene_dir / "gt_matches" / gt_matches_filename
        else:
            self.gt_matches_path = self.scene_dir / "gt_matches" / f"matches_fov90_{self.cfg.graph.loop_closure_mode if self.cfg.graph.enable_loop_closure else 'oracle'}.pkl"
        
        # Determine original resolution from sample depth file
        if self.gt_depth_mode == "png":
            depth_dir = self.scene_dir / "images_depth_fov90"
            sample_depth_path = depth_dir / "00000.png"
            
            if sample_depth_path.exists():
                sample_depth = cv2.imread(str(sample_depth_path), cv2.IMREAD_UNCHANGED)
                if sample_depth is not None:
                    self.gt_orig_H, self.gt_orig_W = sample_depth.shape
                    print(f"GT original resolution (PNG): {self.gt_orig_W}×{self.gt_orig_H}, target: {self.W}×{self.H}")
                else:
                    self.gt_orig_H, self.gt_orig_w = 480, 640
            else:
                self.gt_orig_H, self.gt_orig_w = 480, 640
        else:  # "raw" mode
            depth_dir = self.scene_dir / "images_depth_fov90"
            sample_depth_path = depth_dir / "00000.npy"
            
            if sample_depth_path.exists():
                sample_depth = np.load(str(sample_depth_path))
                self.gt_orig_H, self.gt_orig_w = sample_depth.shape
                print(f"GT original resolution (NPY): {self.gt_orig_w}×{self.gt_orig_H}, target: {self.W}×{self.H}")
            else:
                self.gt_orig_H, self.gt_orig_w = 480, 640
    
    def _init_matcher(self):
        """Initialize the 2D-2D matcher based on configuration"""
        matcher_cfg = self.cfg.get("map_matcher", None)
        
        # Fall back to using MASt3R inference if no matcher config
        if matcher_cfg is None:
            print("No matcher config found, using MASt3R.get_matches() for matching")
            self.matcher = None
            self.matcher_name = "mast3r_inference"
            return
        
        matcher_name = matcher_cfg.get("name", "mast3r")
        self.matcher_name = matcher_name
        
        print(f"\nInitializing matcher: {matcher_name}")
        
        if matcher_name == "mast3r":
            # Fall back to model.subsample_or_initxy1 for mast3r matcher
            subsample = matcher_cfg.get("subsample_or_initxy1", self.cfg.model.subsample_or_initxy1)
            self.matcher = Mast3rMatcher(
                resize_w=matcher_cfg.get("resize_w", self.W),
                resize_h=matcher_cfg.get("resize_h", self.H),
                geometric_verification=matcher_cfg.get("geometric_verification", True),
                subsample_or_initxy1=subsample,
                device=self.device
            )
        elif matcher_name == "superpoint":
            self.matcher = SuperPointMatcher(
                resize_w=matcher_cfg.get("resize_w", self.W),
                resize_h=matcher_cfg.get("resize_h", self.H),
                geometric_verification=matcher_cfg.get("geometric_verification", True),
                max_keypoints=matcher_cfg.get("max_keypoints", 2048),
                keypoint_threshold=matcher_cfg.get("keypoint_threshold", 0.005),
                device=self.device
            )
        else:
            raise ValueError(f"Unknown matcher: {matcher_name}. Supported: 'mast3r', 'superpoint'")
        
        print(f"  resize: {matcher_cfg.get('resize_w', self.W)}x{matcher_cfg.get('resize_h', self.H)}")
        print(f"  geometric_verification: {matcher_cfg.get('geometric_verification', True)}")
    
    def _get_common_filename_params(self) -> str:
        """Generate common filename parameters used across all graph/costmap filenames.
        
        Returns:
            String containing: {w}x{h}_EC_{ec_mode}_NC_{nc_mode}_NCF_{nc_factor}_{dist}_BS_{block_str}_{match_feature_str}
        """
        # Image and graph configuration
        ec_mode = self.cfg.graph.edge_culling_mode
        nc_mode = self.cfg.graph.node_culling_mode
        nc_factor = self.cfg.graph.node_culling_factor
        w, h = self.cfg.image.width, self.cfg.image.height
        mast3r_subsample = self.cfg.model.subsample_or_initxy1
        
        # MASt3R matcher params
        dist = self.cfg.model.dist
        if dist is None:
            dist = "NONE"
        else:
            dist = str.upper(self.cfg.model.dist)

        block_size = self.cfg.model.block_size
        match_feature_type = self.cfg.model.match_feature_type
        
        # Format match feature type
        if match_feature_type == "descriptor":
            match_feature_str = "DESC"
        elif match_feature_type == "pointmap":
            match_feature_str = "PTS3D"
        else:
            match_feature_str = match_feature_type  # fallback
        
        # Format block size
        if block_size is None:
            block_str = "NONE"
        else:
            block_str = str(block_size)

        if block_size is None and dist.lower() == 'dot' and mast3r_subsample == 8:
            # return the old naming when using the default values
            base = f"{w}x{h}_EC_{ec_mode}_NC_{nc_mode}_NCF_{nc_factor}"
        else:
            base = f"{w}x{h}_EC_{ec_mode}_NC_{nc_mode}_NCF_{nc_factor}_{dist}_BS_{block_str}_{match_feature_str}_SUB{mast3r_subsample}"

        # Append depth source if non-default
        depth_source = self.cfg.model.get("depth_source", "mast3r")
        if depth_source not in ("mast3r", None):
            base += f"_DEPTH_{depth_source}"

        return base
    
    def _get_base_graph_filename(self):
        """Generate filename for base graph (without goal)"""
        common_params = self._get_common_filename_params()
        lc_enabled = self.cfg.graph.get("enable_loop_closure", False)
        
        filename = f"graph_base_{common_params}"
        
        if lc_enabled:
            lc_mode = self.cfg.graph.get("loop_closure_mode", "subsample")
            filename += f"_LC_{lc_mode}"
        
        return filename + ".pkl"
    
    def _get_goal_graph_filename(self, goal_img_idx: int = None):
        """Generate filename for graph with goal distances"""
        common_params = self._get_common_filename_params()
        lc_enabled = self.cfg.graph.get("enable_loop_closure", False)
        
        filename = f"graph_with_distances_to_goal_{common_params}"
        
        if lc_enabled:
            lc_mode = self.cfg.graph.get("loop_closure_mode", "subsample")
            filename += f"_LC_{lc_mode}"
        
        if goal_img_idx is not None:
            filename += f"_goalImg{goal_img_idx}"
        
        return filename + ".pkl"
    
    def _get_costmap_filename(self, metadata: dict = None):
        """Generate descriptive filename for costmaps"""
        common_params = self._get_common_filename_params()

        filename = f"costmaps_{common_params}"

        if self.cfg.graph.get("enable_loop_closure", False):
            lc_mode = self.cfg.graph.get("loop_closure_mode", "subsample")
            filename += f"_LC_{lc_mode}"

        if metadata and 'goal_img_idx' in metadata:
            filename += f"_goalImg{metadata['goal_img_idx']}"
        
        return filename

    def _subsample_images(self) -> list:
        """Subsample images based on configuration"""
        start_idx = self.cfg.processing.subsample_start_idx
        end_idx = self.cfg.processing.subsample_end_idx
        step = self.cfg.processing.subsample_step

        return self.img_paths[start_idx:end_idx:step]
    
    def _copy_images_to_output_dir(self):
        """Copy images to output directory"""
        out_img_dir = self.out_dir / "images_fov90"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file extension from first image
        extension = Path(self.img_paths[0]).suffix
        
        print(f"Copying {len(self.img_paths)} images to {out_img_dir}")
        for i, img_path in enumerate(self.img_paths):
            output_path = out_img_dir / f"{i:04d}{extension}"
            shutil.copy2(img_path, output_path)
    
    @time_function("create_map_topo")
    def create_map_topo(self):
        """Main method to create topological map"""
        # Step 1: Compute or load point clouds
        if self.recreate_graphs or (not self.pc_npz_path.exists())  or self.force_recompute_graph:
            pc_dict = self.compute_and_save_point_clouds(save_as_npz=True)
        else:
            print(f"Using Precomputed point clouds from {self.pc_npz_path}")
            pc_dict = np.load(self.pc_npz_path)
            npz_file = np.load(self.pc_npz_path)
            pc_dict = {key: npz_file[key] for key in npz_file.files}
            npz_file.close()
        
        # Step 2: Create base graph with nodes
        self.G = self.create_base_graph_with_nodes(pc_dict)
        print(f"\nGraph just after creation: {self.G}")

        # Step 3: Add inter-image edges (including loop closures if enabled)
        if self.recreate_graphs or not self.graph_inter_path.exists() or self.force_recompute_graph:
            self.G = self.add_inter_image_edges_to_graph()
        
        # Step 4: Add intra-image edges
        if self.recreate_graphs or not self.graph_intra_path.exists() or self.force_recompute_graph:
            self.G = self.add_intra_image_edges_to_graph()
    
    @time_function("compute_and_save_point_clouds")
    def compute_and_save_point_clouds(self, save_as_npz: bool = True) -> dict:
        """Compute 3D point clouds for all images"""
        if self.depth_source == "mapanything":
            return self._compute_mapanything_point_clouds(save_as_npz)
        if self.use_gt_matches or self.depth_source == "gt":
            return self._compute_gt_point_clouds(save_as_npz)
        
        # Normal MASt3R mode
        pc_dict = {}
        for img_idx, img_path in enumerate(
            tqdm(self.img_paths, desc="Computing MASt3R point clouds")
        ):
            # Infer MASt3R model (self-to-self for point cloud extraction)
            pts3d = self.mast3r.get_pts3d(img_path, resize=(self.H, self.W))
            
            # Store using string path as key
            pc_dict[str(img_path)] = pts3d
        
        # Optionally save to disk
        if save_as_npz:
            np.savez_compressed(self.pc_npz_path, **pc_dict)
        
        return pc_dict
    
    def _compute_gt_point_clouds(self, save_as_npz: bool = True) -> dict:
        depth_dir = self.scene_dir / "images_depth_fov90"
        depth_ext = ".png" if self.gt_depth_mode == "png" else ".npy"

        # Intrinsics from FOV
        fov_x = self.hfov
        fx = self.W / (2.0 * np.tan(np.radians(fov_x / 2.0)))
        fy = fx
        cx = self.W / 2.0
        cy = self.H / 2.0

        # Pixel grid (compute once, reuse per frame)
        u = np.arange(self.W, dtype=np.float32)
        v = np.arange(self.H, dtype=np.float32)
        u, v = np.meshgrid(u, v)  # (H, W)

        pc_dict = {}
        for frame_idx, img_path in enumerate(tqdm(self.img_paths, desc=f"Loading GT depth ({self.gt_depth_mode})")):
            depth_path = depth_dir / f"{frame_idx:05d}{depth_ext}"
            if not depth_path.exists():
                print(f"Warning: Depth file not found: {depth_path}")
                continue
            
            # Load depth based on format
            if depth_ext == ".png":
                depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_raw is None:
                    print(f"Warning: Failed to read PNG depth: {depth_path}")
                    continue
                depth = depth_raw.astype(np.float32) * 0.001  # mm to meters
            else:
                depth = np.load(str(depth_path)).astype(np.float32)
            
            # Resize if needed
            if depth.shape != (self.H, self.W):
                depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

            # Backproject to 3D
            pts3d = np.zeros((self.H, self.W, 3), dtype=np.float32)
            pts3d[:, :, 0] = (u - cx) * depth / fx  # X
            pts3d[:, :, 1] = (v - cy) * depth / fy  # Y
            pts3d[:, :, 2] = depth                   # Z

            pc_dict[str(img_path)] = pts3d

        print(f"Loaded {len(pc_dict)} GT depth maps from {depth_dir} ({self.gt_depth_mode} format)")
        if save_as_npz:
            np.savez_compressed(self.pc_npz_path, **pc_dict)
        return pc_dict

    def _compute_mapanything_point_clouds(self, save_as_npz: bool = True) -> dict:
        """
        Load camera-frame 3D pointmaps from MapAnything predictions.

        Tries mapanything_pointmaps_cam_fov90/{idx:05d}.npy first (pre-extracted),
        falls back to loading pred_{idx:04d}.pt from mapanything_outputs/.
        """
        cam_dir = self.scene_dir / "mapanything_pointmaps_cam_fov90"
        use_npy = cam_dir.exists()
        ma_dir = self.scene_dir / "mapanything_outputs"

        pc_dict = {}
        for frame_idx, img_path in enumerate(
            tqdm(self.img_paths, desc="Loading MapAnything camera-frame pointmaps")
        ):
            pts3d = None

            if use_npy:
                npy_path = cam_dir / f"{frame_idx:05d}.npy"
                if npy_path.exists():
                    pts3d = np.load(str(npy_path)).astype(np.float32)
                else:
                    print(f"Warning: MapAnything cam pointmap not found: {npy_path}")

            if pts3d is None:
                # Fallback: load from pred_*.pt
                pt_path = ma_dir / f"pred_{frame_idx:04d}.pt"
                if pt_path.exists():
                    pred = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    pm = pred["pts3d_cam"]
                    if isinstance(pm, torch.Tensor):
                        pm = pm.numpy()
                    if pm.ndim == 4:
                        pm = pm[0]
                    # Apply mask
                    if "mask" in pred:
                        mask = pred["mask"]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.numpy()
                        mask = mask.squeeze()
                        if mask.dtype != bool:
                            mask = mask > 0.5
                        pm[~mask] = 0.0
                    pts3d = pm.astype(np.float32)
                else:
                    print(f"Warning: No MapAnything data for frame {frame_idx}")
                    pts3d = np.zeros((self.H, self.W, 3), dtype=np.float32)

            # Replace NaN with 0 (graph code expects finite values)
            pts3d = np.nan_to_num(pts3d, nan=0.0)

            # Resize if needed
            if pts3d.shape[:2] != (self.H, self.W):
                pts3d_resized = np.zeros((self.H, self.W, 3), dtype=np.float32)
                for c in range(3):
                    pts3d_resized[:, :, c] = cv2.resize(
                        pts3d[:, :, c], (self.W, self.H),
                        interpolation=cv2.INTER_NEAREST
                    )
                pts3d = pts3d_resized

            pc_dict[str(img_path)] = pts3d

        print(f"Loaded {len(pc_dict)} MapAnything camera-frame pointmaps")
        if save_as_npz:
            np.savez_compressed(self.pc_npz_path, **pc_dict)
        return pc_dict
    
    def pixel_coord_to_global_node_id(self, img_idx, px, py):
        """Convert pixel coordinates to global node ID"""
        H, W = self.H, self.W

        # node id of the first node belonging this image
        img_st_node_id = img_idx * (H * W)
        node_id = img_st_node_id + (py * W + px)
        return node_id
    
    def collect_loop_closure_match_pairs(self):
        print("Collecting loop-closure match pairs...")
        loop_match_pairs = []

        mode = self.cfg.graph.get("loop_closure_mode", "oracle")

        if mode == "oracle":
            oracle_path = self.scene_dir / "oracle_loops.txt"
            loop_pairs = read_oracle_lc_pairs(
                oracle_path, len(self.img_paths)
            )

            for img_i, img_j in tqdm(loop_pairs, desc="LC matching"):
                matches_i, matches_j = self.get_mast3r_matches(
                    self.img_paths[img_i],
                    self.img_paths[img_j],
                )

                matches_per_pair = []
                for k in range(len(matches_i)):  # iterate all matches
                    pi = (img_i, int(matches_i[k][0]), int(matches_i[k][1]))
                    pj = (img_j, int(matches_j[k][0]), int(matches_j[k][1]))
                    matches_per_pair.append((pi, pj))

                matches_per_pair = matches_per_pair[::self.cfg.graph.node_culling_factor]
                loop_match_pairs.extend(matches_per_pair)
        elif mode == "seqvlad":
            path = self.scene_dir / "seqvlad_loops_ufm.txt"
            loop_pairs = read_oracle_lc_pairs(
                path, len(self.img_paths)
            )
            for img_i, img_j in tqdm(loop_pairs, desc="LC matching"):
                matches_i, matches_j = self.get_mast3r_matches(
                    self.img_paths[img_i],
                    self.img_paths[img_j],
                )

                matches_per_pair = []
                for k in range(len(matches_i)):  # iterate all matches
                    pi = (img_i, int(matches_i[k][0]), int(matches_i[k][1]))
                    pj = (img_j, int(matches_j[k][0]), int(matches_j[k][1]))
                    matches_per_pair.append((pi, pj))

                matches_per_pair = matches_per_pair[::self.cfg.graph.node_culling_factor]
                loop_match_pairs.extend(matches_per_pair)
        elif mode == "mapanything":
            ma_path = self.scene_dir / "mapanything_loops.txt"
            loop_pairs = read_oracle_lc_pairs(
                ma_path, len(self.img_paths)
            )
            for img_i, img_j in tqdm(loop_pairs, desc="LC matching (mapanything)"):
                matches_i, matches_j = self.get_mast3r_matches(
                    self.img_paths[img_i],
                    self.img_paths[img_j],
                )

                matches_per_pair = []
                for k in range(len(matches_i)):
                    pi = (img_i, int(matches_i[k][0]), int(matches_i[k][1]))
                    pj = (img_j, int(matches_j[k][0]), int(matches_j[k][1]))
                    matches_per_pair.append((pi, pj))

                matches_per_pair = matches_per_pair[::self.cfg.graph.node_culling_factor]
                loop_match_pairs.extend(matches_per_pair)
        else:
            raise NotImplementedError(f"Loop closure mode '{mode}' not supported")

        print(f"Collected {len(loop_match_pairs)} LC match pairs")
        return loop_match_pairs


    @time_function("create_base_graph_with_nodes")
    def create_base_graph_with_nodes(self, pc_dict):
        """
        Creates a Sparse graph based on mast3r matches
        """
        G = nx.Graph()
        G.graph['cfg'] = OmegaConf.to_container(self.cfg, resolve=True)
        
        # Collect all match pairs while preserving correspondence
        all_match_pairs = []
        num_matches_per_pair = []  # Track number of matches per image pair for statistics
        nc_factor = self.cfg.graph.node_culling_factor
        
        print("First pass: collecting all match pairs...")
        img_st_idx = 0
        img_end_idx = len(self.img_paths)
        match_window_size = self.img_match_window_size
        
        for i in tqdm(range(img_st_idx, img_end_idx), desc="Collecting match pairs"):
            for j in range(i + 1, min(i + 1 + match_window_size, img_end_idx)): 
                try:
                    matches_im0, matches_im1 = self.get_mast3r_matches(
                        self.img_paths[i], self.img_paths[j]
                    )
                    
                    num_matches = len(matches_im0)
                    num_matches_per_pair.append(num_matches)
                    
                    matches_per_pair = []
                    # Store each match as a pair (preserves correspondence)
                    for k in range(num_matches):
                        pixel_i = (i, int(matches_im0[k][0]), int(matches_im0[k][1]))  # (img_idx, px, py)
                        pixel_j = (j, int(matches_im1[k][0]), int(matches_im1[k][1]))  # (img_idx, px, py)
                        
                        match_pair = (pixel_i, pixel_j)
                        matches_per_pair.append(match_pair)
                        # all_match_pairs.append(match_pair)
                    
                    # Sample match pairs (preserves correspondence)
                    matches_per_pair = matches_per_pair[::nc_factor]
                    all_match_pairs.extend(matches_per_pair)
                                
                except Exception as e:
                    print(f"Error getting matches between {i} and {j}: {e}")
                    continue

        # ---- NEW: collect LC match pairs ----
        if self.cfg.graph.get("enable_loop_closure", False):
            loop_match_pairs = self.collect_loop_closure_match_pairs()
        else:
            loop_match_pairs = []

        
        print(f"Found {len(all_match_pairs)} match pairs across all images")
        if len(num_matches_per_pair) > 0:
            avg_matches = np.mean(num_matches_per_pair)
            min_matches = np.min(num_matches_per_pair)
            max_matches = np.max(num_matches_per_pair)
            print(f"Matches per image pair - Avg: {avg_matches:.1f}, Min: {min_matches}, Max: {max_matches}")
        
        sampled_match_pairs = all_match_pairs
        
        print(f"Sampled {len(sampled_match_pairs)} match pairs (every {nc_factor}th pair)")
        
        # Extract unique pixels from sampled match pairs
        unique_pixels = set()
        for pair in sampled_match_pairs:
            unique_pixels.update(pair)

        for pair in loop_match_pairs:
            unique_pixels.update(pair)

        print(f"Extracted {len(unique_pixels)} unique pixels from sampled pairs")
        
        # Create nodes
        pixel_to_node_id = {}
        for img_idx, px, py in tqdm(unique_pixels, desc="Creating DA nodes"):
            # Node ID calculation
            node_id = self.pixel_coord_to_global_node_id(img_idx, px, py)
            
            key = str(self.img_paths[img_idx])
            pcd = pc_dict[key]  # (H, W, 3)
            
            # Get 3D coordinate for this pixel
            coord_3d = pcd[py, px]  # Note: pcd is indexed as [y, x]
            
            # Create node
            node_attrs = {
                "map": [img_idx, py * self.W + px],  # [image_idx, pixel_index]
                "coord_mast3r": coord_3d,
                "pixel": [px, py],
                "type": "da"
            }
            
            G.add_node(node_id, **node_attrs)
            pixel_to_node_id[(img_idx, px, py)] = node_id
            # node_id_to_pixel[node_id] = (img_idx, px, py)
        
        # Store mappings
        self.pixel_to_node_id = pixel_to_node_id
        self.sampled_match_pairs = sampled_match_pairs  # Store for guaranteed edge creation
        self.loop_match_pairs = loop_match_pairs
        # self.node_id_to_pixel = node_id_to_pixel
        
        print(f"Created sparse graph with {G.number_of_nodes()} DA nodes")
        print(f"Stored {len(self.sampled_match_pairs)} match pairs for edge creation")
        print(f"Stored {len(self.loop_match_pairs)} loop-closure match pairs for edge creation")
        
        # Compute nodes per image statistics
        nodes_per_image = {}
        for img_idx, px, py in unique_pixels:
            nodes_per_image[img_idx] = nodes_per_image.get(img_idx, 0) + 1
        
        if len(nodes_per_image) > 0:
            counts = list(nodes_per_image.values())
            avg_nodes = np.mean(counts)
            min_nodes = np.min(counts)
            max_nodes = np.max(counts)
            print(f"DA nodes per image - Avg: {avg_nodes:.1f}, Min: {min_nodes}, Max: {max_nodes}")
        
        return G

    @time_function("add_inter_image_edges_to_graph")
    def add_inter_image_edges_to_graph(self):
        da_edges = []

        print(f"Adding Inter-Image edges from {len(self.sampled_match_pairs)} stored match pairs...")
        
        for pair in tqdm(self.sampled_match_pairs, desc="Creating DA edges from match pairs"):
            # pixel_i = (img_idx_i, px_i, py_i)
            pixel_i, pixel_j = pair
            
            # Get node IDs for both pixels in the pair
            node_i = self.pixel_coord_to_global_node_id(pixel_i[0], pixel_i[1], pixel_i[2])
            node_j = self.pixel_coord_to_global_node_id(pixel_j[0], pixel_j[1], pixel_j[2])
            
            # Both nodes should exist since we created them from sampled pairs
            if node_i in self.G.nodes and node_j in self.G.nodes:
                da_edges.append((int(node_i), int(node_j), {'edge_type': 'da', 'weight': 0}))
            else:
                print(f"Warning: Missing nodes {node_i} or {node_j} for pair {pair}")
        
        print(f"Created {len(da_edges)} DA edges from stored pairs")

        if self.cfg.graph.get("enable_loop_closure", False):
            for pair in tqdm(self.loop_match_pairs, desc="Creating LC edges from loop match pairs"):
                pixel_i, pixel_j = pair
                
                node_i = self.pixel_coord_to_global_node_id(pixel_i[0], pixel_i[1], pixel_i[2])
                node_j = self.pixel_coord_to_global_node_id(pixel_j[0], pixel_j[1], pixel_j[2])
                
                if node_i in self.G.nodes and node_j in self.G.nodes:
                    da_edges.append((int(node_i), int(node_j), {'edge_type': 'da_loop', 'weight': 0}))
                else:
                    print(f"Warning: Missing nodes {node_i} or {node_j} for LC pair {pair}")
            print(f"Added {len(self.loop_match_pairs)} loop closure edges")

        # Add DA edges to graph
        self.G.add_edges_from(da_edges)
        print(f"\n\nNumber of nodes and edges: {len(self.G.nodes())}, {self.G.number_of_edges()}")

        return self.G

    @time_function("add_intra_image_edges_to_graph")
    def add_intra_image_edges_to_graph(self):
        """
        Connect DA nodes within the same image using spatial relationships
        """
        # Group DA nodes by image
        da_nodes_per_img = {}
        for node_id in self.G.nodes():
            node = self.G.nodes[node_id]
            img_id = node['map'][0]
            
            if img_id not in da_nodes_per_img:
                da_nodes_per_img[img_id] = []
            da_nodes_per_img[img_id].append(node_id)
        print(f"Adding intra-image edges for {len(da_nodes_per_img)} images")

        # adding DA intra-image edges
        for img_id in tqdm(
            da_nodes_per_img.keys(), desc="connecting da nodes intra edges"
        ):
            da_nodes = da_nodes_per_img[img_id]
            edge_culling_mode = EdgeCullingMode(self.cfg.graph.edge_culling_mode)
            
            # Choose edge creation method
            if edge_culling_mode == EdgeCullingMode.EMST_SINGLE:
                edges = self._create_emst_edges(da_nodes, img_id)
            elif edge_culling_mode == EdgeCullingMode.DELAUNAY_3D:
                edges = self._create_delaunay_3d_edges(da_nodes, img_id)
            else:
                edges = self._create_complete_graph_edges(da_nodes, img_id)
        
            # Add all intra-image edges to the graph
            self.G.add_edges_from(edges)

        print(
            f"Final graph has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges"
        )

        return self.G
    
    # def _create_emst_edges(self, da_nodes: list, img_id: int) -> list:
    #     """Create edges using Euclidean Minimum Spanning Tree"""
    #     if len(da_nodes) <= 1:
    #         return []
        
    #     # Get 3D coordinates
    #     coords = np.array([
    #         self.G.nodes[nid]["coord_mast3r"] 
    #         for nid in da_nodes
    #     ])
        
    #     # Compute distance matrix
    #     dist_matrix = np.linalg.norm(
    #         coords[:, None, :] - coords[None, :, :], 
    #         axis=2
    #     )

    #     # === DEBUG: Essential checks ===
    #     # print(
    #     #     f"EMST DEBUG img {img_id}: nodes={len(da_nodes)}, dist_matrix min/max={dist_matrix.min():.3f}/{dist_matrix.max():.3f}, has_nan/inf={np.isnan(dist_matrix).any()}/{np.isinf(dist_matrix).any()}"
    #     # )
        
    #     # Compute MST
    #     mst = minimum_spanning_tree(dist_matrix).toarray()
        
    #     # Build edge list with attributes
    #     edges = []
    #     for i in range(len(da_nodes)):
    #         for j in range(len(da_nodes)): 
    #             if i != j and mst[i, j] > 0:
    #                 edge_weight = dist_matrix[i, j]
    #                 edges.append((
    #                     da_nodes[i], 
    #                     da_nodes[j], 
    #                     {
    #                         'edge_type': 'da_intra',
    #                         'weight': edge_weight,
    #                     }
    #                 ))
        
    #     # # Check connectivity using NetworkX
    #     # mst_graph = nx.Graph()
    #     # mst_graph.add_nodes_from(range(len(da_nodes)))
    #     # mst_graph.add_edges_from(
    #     #     [
    #     #         (i, j)
    #     #         for i in range(len(da_nodes))
    #     #         for j in range(len(da_nodes))
    #     #         if i != j and mst[i, j] > 0
    #     #     ]
    #     # )
    #     # is_connected = nx.is_connected(mst_graph)

    #     # print(
    #     #     f"EMST edges: {len(da_da_intra_edges)}, expected: {len(da_nodes) - 1}, connected: {is_connected}"
    #     # )
        
    #     # print(f"Created {len(edges)} EMST edges for image {img_id} "
    #     #     f"(expected {len(da_nodes) - 1})")
        
    #     return edges
    @time_function("create_emst_edges")
    def _create_emst_edges(self, da_nodes: list, img_id: int) -> list:
        """Create edges using Euclidean Minimum Spanning Tree"""
        if len(da_nodes) <= 1:
            return []

        # Get 3D coordinates
        coords = np.array([
            self.G.nodes[nid]["coord_mast3r"]
            for nid in da_nodes
        ])

        # Compute distance matrix
        # GPU distance matrix
        coords_t = torch.from_numpy(coords).float().to(self.device)
        dist_matrix = torch.cdist(coords_t, coords_t).cpu().numpy()
        
        # Old CPU version
        # dist_matrix = scipy.spatial.distance.cdist(coords, coords)

        # Compute MST
        mst = minimum_spanning_tree(dist_matrix)

        # Extract edges directly from the sparse matrix — no Python loop
        coo = mst.tocoo()
        da_nodes_arr = np.array(da_nodes)

        src = da_nodes_arr[coo.row]
        dst = da_nodes_arr[coo.col]
        weights = coo.data

        edges = [
            (int(s), int(d), {'edge_type': 'da_intra', 'weight': float(w)})
            for s, d, w in zip(src, dst, weights)
        ]

        return edges
    
    def _create_delaunay_3d_edges(self, da_nodes: list, img_id: int) -> list:
        """Create edges using 3D Delaunay triangulation"""
        if len(da_nodes) <= 3:
            print(f"Not enough DA nodes for 3D Delaunay in image {img_id}")
            return []
        
        # Get 3D coordinates
        coords = np.array([
            self.G.nodes[nid]["coord_mast3r"] 
            for nid in da_nodes
        ])

        # Compute distance matrix
        dist_matrix = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :], 
            axis=2
        )

        try:
            tri = Delaunay(coords)
            
            # Extract unique edges from simplices using a dict keyed by sorted node pair
            edges_dict = {}
            for simplex in tri.simplices:
                # Each simplex is a tetrahedron (4 vertices)
                for i in range(4):
                    for j in range(i + 1, 4):
                        a, b = simplex[i], simplex[j]
                        node_a = da_nodes[a]
                        node_b = da_nodes[b]
                        edge_key = (min(node_a, node_b), max(node_a, node_b))
                        if edge_key not in edges_dict:
                            edges_dict[edge_key] = (
                                edge_key[0],
                                edge_key[1],
                                {'edge_type': 'da_intra', 'weight': float(dist_matrix[a, b])}
                            )
            
            edges = list(edges_dict.values())

            return edges
            
        except Exception as e:
            print(f"3D Delaunay failed for image {img_id}: {e}")
            return []

    def _create_complete_graph_edges(self, da_nodes: list, img_id: int) -> list:
        """Create complete graph (all-to-all connections) with distance-based weights"""
        if len(da_nodes) <= 1:
            print(f"Only one DA node in image {img_id}, no intra-image edges needed")
            return []
        
        # Get 3D coordinates
        coords = np.array([
            self.G.nodes[nid]["coord_mast3r"] 
            for nid in da_nodes
        ])
        
        # Compute pairwise distance matrix
        dist_matrix = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :], 
            axis=2
        )
        
        # Create all-to-all edges with weights
        edges = [
            (da_nodes[i], da_nodes[j], {'edge_type': 'da_intra', 'weight': dist_matrix[i, j]})
            for i, j in combinations(range(len(da_nodes)), 2)
        ]
        
        # print(f"Created {len(edges)} complete graph edges for image {img_id}")
        return edges

    def get_goal_from_episode(self):
        """
        Infer goal from episode folder using semantic mask and centroid.
        
        Uses get_goal_info() to find the goal image and mask, then computes
        the centroid to get goal pixel coordinates.
        """
        from libs.common.geometry_utils import get_goal_info, get_goal_info_new, get_mask_centroid
        
        task_type = self.cfg.goal.get("task_type", "original")
        # goal_img_idx, goal_mask, goal_instance_id = get_goal_info(
        #     str(self.scene_dir), 
        #     task_type
        # )
        goal_img_idx, goal_mask = get_goal_info_new(
            str(self.scene_dir), 
            task_type
        )
        
        # Resize mask if needed
        if goal_mask.shape != (self.H, self.W):
            print(f"Resizing goal mask from {goal_mask.shape} to ({self.H}, {self.W})")
            goal_mask = cv2.resize(goal_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        
        # Get centroid
        centroid = get_mask_centroid(goal_mask)
        if centroid is None:
            raise ValueError(f"Goal mask is empty in episode {self.scene_dir}")
        
        goal_px, goal_py = centroid
        
        print(f"Inferred goal from episode: img_idx={goal_img_idx}, pixel=({goal_px}, {goal_py})")
        
        # Store as instance variables for compute_distances_to_goal_node
        self.inferred_goal_img_idx = goal_img_idx
        self.inferred_goal_px = goal_px
        self.inferred_goal_py = goal_py
        self.inferred_goal_mask = goal_mask
        
        return goal_img_idx, goal_px, goal_py

    @time_function("compute_distances_to_goal_node")
    def compute_distances_to_goal_node(
        self,
        goal_img_idx: int = None,
        goal_px: int = None,
        goal_py: int = None
    ):
        """
        Compute distances from all nodes to a goal node.
        
        Args:
            goal_img_idx: Goal image index (optional, uses config/inferred if None)
            goal_px: Goal pixel x coordinate (optional)
            goal_py: Goal pixel y coordinate (optional)
            
        Returns:
            Costmaps array if compute_costmaps is enabled, else None
        """
        # Determine goal coordinates: explicit args > inferred > config
        if goal_img_idx is None:
            if hasattr(self, 'inferred_goal_img_idx'):
                goal_img_idx = self.inferred_goal_img_idx
                goal_px = self.inferred_goal_px
                goal_py = self.inferred_goal_py
            else:
                goal_img_idx = self.cfg.goal.image_idx
                goal_px = self.cfg.goal.pixel_x
                goal_py = self.cfg.goal.pixel_y

        print(f"Goal: img_idx={goal_img_idx}, pixel=({goal_px}, {goal_py})")

        # Load point cloud data if needed
        if os.path.exists(self.pc_npz_path):
            pc_dict = np.load(self.pc_npz_path)
        else:
            print(f"Point cloud NPZ not found at {self.pc_npz_path}, recomputing point clouds...")
            pc_dict = self.compute_and_save_point_clouds(save_as_npz=False)
        
        # Calculate expected goal node ID
        expected_goal_node_id = self.pixel_coord_to_global_node_id(goal_img_idx, goal_px, goal_py)
        
        # Add or find goal node
        if expected_goal_node_id in self.G.nodes:
            print(f"Goal node {expected_goal_node_id} already exists")
            goal_node_id = expected_goal_node_id
        else:
            # Add the goal node to the graph
            goal_node_id = self.add_goal_node(goal_img_idx, goal_px, goal_py, pc_dict)
            print(f"Added goal node {goal_node_id} at pixel ({goal_px}, {goal_py}) in image {goal_img_idx}")
        
        # Store goal info in graph (snake_case)
        self.G.graph['goal_img_idx'] = goal_img_idx
        self.G.graph['goal_node_coords'] = (goal_px, goal_py)
        self.G.graph['goal_node_id'] = goal_node_id
        
        # Connect DA nodes to goal
        self.connect_da_to_goal_node(goal_img_idx, goal_node_id)
        
        # Run Dijkstra from goal to all nodes
        path_lengths = self.get_single_source_paths(self.G, source_node=goal_node_id, weight='weight')
        self.all_path_lengths = path_lengths
        self.G.graph['all_path_lengths'] = {'weight': path_lengths}

        # Compute costmaps if enabled
        if self.cfg.goal.get('compute_costmaps', True):
            print(f"\nComputing distance-to-goal costmaps for all images...")
            img_costmaps = self.compute_all_image_costmaps(pc_dict)
            metadata = {
                'goal_img_idx': goal_img_idx,
                'goal_pixel': (goal_px, goal_py),
                'goal_node_id': goal_node_id,
                'cfg': OmegaConf.to_container(self.cfg, resolve=True),
                'goal_coord_3d': self.G.nodes[goal_node_id]['coord_mast3r'].tolist(),
                'image_paths': [str(path) for path in self.img_paths],
                'shape': list(img_costmaps.shape),
            }
            self.save_costmaps(img_costmaps, metadata)
            return img_costmaps

        return None

    def add_goal_node(self, goal_img_idx, goal_px, goal_py, pc_dict):
        """Add goal node to the sparse graph"""
        key = str(self.img_paths[goal_img_idx])
        pcd = pc_dict[key]
        coord_3d = pcd[goal_py, goal_px]
        
        # Use OLD node ID calculation instead of sequential
        goal_node_id = self.pixel_coord_to_global_node_id(goal_img_idx, goal_px, goal_py)
        
        goal_attrs = {
            "map": [goal_img_idx, goal_py * self.W + goal_px],
            "coord_mast3r": coord_3d,
            "pixel": [goal_px, goal_py],
            "type": "goal"
        }
        
        self.G.add_node(goal_node_id, **goal_attrs)
        
        # Update mapping
        self.pixel_to_node_id[(goal_img_idx, goal_px, goal_py)] = goal_node_id
        
        return goal_node_id

    def connect_da_to_goal_node(self, img_idx, goal_node_id):
        """
        Connect goal node to DA nodes in the same image (sparse graph version)
        """
        goal_node = self.G.nodes[goal_node_id]
        
        # Find all DA nodes in the same image
        img_da_node_ids = []
        for node_id, node_data in self.G.nodes(data=True):
            if (node_data['map'][0] == img_idx and 
                node_data.get('type') == 'da'):
                img_da_node_ids.append(node_id)
        
        print(f"Found {len(img_da_node_ids)} DA nodes in image {img_idx}")
        
        # if goal node is da node then it's already connected to all other da nodes of the image
        if goal_node["type"] == "da":
            print(f"{goal_node_id = } is a DA node")
            self.G.nodes[goal_node_id]["type"] = "goal"

            # Update the edgeType for all edges between goal_node_id and DA nodes in the image
            for da_node_id in img_da_node_ids:
                if self.G.has_edge(goal_node_id, da_node_id):
                    self.G.edges[goal_node_id, da_node_id]["edge_type"] = "goal_da_intra"
            return

        # updating the goal node type
        self.G.nodes[goal_node_id]["type"] = "goal"
        pts3d_goal_node = goal_node["coord_mast3r"]

        # Compute distances using cdist
        da_coords = np.array([self.G.nodes[da_node_id]['coord_mast3r'] for da_node_id in img_da_node_ids])
        goal_coord_2d = pts3d_goal_node.reshape(1, -1)
        distances = cdist(goal_coord_2d, da_coords, metric='euclidean')[0]
        
        edge_weights = {da_node_id: distances[i] for i, da_node_id in enumerate(img_da_node_ids)}

        # Add edges
        self.G.add_edges_from(
            [
                (
                    goal_node_id,
                    da_node_id,
                    {
                        "edge_type": "goal_da_intra",
                        "weight": edge_weights[da_node_id],
                    },
                )
                for da_node_id in img_da_node_ids
            ]
        )
        
        print(f"Connected goal node {goal_node_id} to {len(img_da_node_ids)} DA nodes")
        return self.G

    @time_function("get_single_source_paths")
    def get_single_source_paths(self, G, source_node, weight=None, maxVal=1e6):
        """
        Compute shortest path lengths from a single source node to all other nodes.

        Args:
            G: NetworkX graph
            source_node: Source node ID
            weight: Edge weight attribute to use (e.g., 'margin')
            maxVal: Value to use for unreachable nodes

        Returns:
            dict: Dictionary mapping target nodes to their shortest path lengths from source_node
        """
        # Use NetworkX's single_source_dijkstra_path_length
        path_lengths = nx.single_source_dijkstra_path_length(
            G, source_node, weight=weight
        )

        # Fill in unreachable nodes with maxVal
        for node in G.nodes():
            if node not in path_lengths:
                path_lengths[node] = maxVal

        return path_lengths

    def compute_all_image_costmaps(self, pc_dict):
        """Compute costmaps for all images"""
        img_costmaps = []
        num_images = len(self.img_paths)
        for i in tqdm(range(0, num_images), desc="computing non-da to goal distances"):
            pts3d = pc_dict[str(self.img_paths[i])]
            costmap = self.compute_single_image_costmap(i, pts3d)
            img_costmaps.append(costmap)
        
        img_costmaps = np.stack(img_costmaps, axis=0)
        return img_costmaps
    
    def compute_single_image_costmap(self, img_idx, pts3d, max_dist=1e6):
        """
        Compute distance-to-goal costmap for a single image using GPU acceleration.
        
        Args:
            img_idx: Image index
            pts3d: Point cloud of shape (H, W, 3)
            max_dist: Maximum distance for unreachable/invalid pixels
            
        Returns:
            np.ndarray: Costmap of shape (H, W) with distance to goal for each pixel
        """
        H, W = self.H, self.W
        
        costmap = np.full((H, W), max_dist, dtype=np.float32)
        
        pts3d_flat = pts3d.reshape(H * W, 3)  # (H*W, 3)
        
        # Step 1: Collect DA and Non-DA pixel information
        da_pixel_indices = []
        da_distances = [] 
        nonda_pixel_indices = []
        
        for y in range(H):
            for x in range(W):
                node_id = self.pixel_coord_to_global_node_id(img_idx, x, y)
                linear_idx = y * W + x
                
                if node_id in self.G.nodes:
                    node_type = self.G.nodes[node_id].get('type', 'unknown')
                    
                    if node_type in ['da', 'goal']:
                        da_pixel_indices.append(linear_idx)
                        da_distances.append(self.all_path_lengths[node_id])
                        costmap[y, x] = self.all_path_lengths[node_id]
                    else:
                        nonda_pixel_indices.append(linear_idx)
                else:
                    nonda_pixel_indices.append(linear_idx)
        
        # Convert to numpy arrays
        da_pixel_indices = np.array(da_pixel_indices, dtype=np.int32)
        da_distances = np.array(da_distances, dtype=np.float32)
        nonda_pixel_indices = np.array(nonda_pixel_indices, dtype=np.int32)
        
        # print(f"  Image {img_idx}: {len(da_pixel_indices)} DA pixels, {len(nonda_pixel_indices)} Non-DA pixels")
        
        if len(da_pixel_indices) == 0:
            print(f"  Warning: No DA nodes in image {img_idx}, returning max distances")
            return costmap
        
        if len(nonda_pixel_indices) == 0:
            return costmap
        
        # Step 2: distance computation for Non-DA pixels
        nonda_distances = self.compute_nonda_distances(
            pts3d_flat, 
            nonda_pixel_indices, 
            da_pixel_indices,
            da_distances,
            max_dist
        )
        
        # Step 3: Fill costmap with Non-DA distances
        nonda_y = nonda_pixel_indices // W
        nonda_x = nonda_pixel_indices % W
        costmap[nonda_y, nonda_x] = nonda_distances
        
        return costmap

    def compute_nonda_distances( self,
        pts3d_flat: np.ndarray,  # (H*W, 3)
        nonda_indices: np.ndarray,  # (N_nonda,)
        da_indices: np.ndarray,  # (N_da,)
        da_distances: np.ndarray,  # (N_da,)
        max_dist: float = 1e6
    ) -> np.ndarray:
        """
        Compute distances for Non-DA pixels
        
        Args:
            pts3d_flat: All 3D points in image (H*W, 3)
            nonda_indices: Indices of Non-DA pixels
            da_indices: Indices of DA pixels
            da_distances: Distance-to-goal for DA pixels (N_da,)
            max_dist: Maximum distance for invalid pixels
            
        Returns:
            np.ndarray: Distances for Non-DA pixels (N_nonda,)
        """
        device = torch.device(self.device)
        
        # Get 3D coordinates
        nonda_pts3d = pts3d_flat[nonda_indices]  # (N_nonda, 3)
        da_pts3d = pts3d_flat[da_indices]  # (N_da, 3)
        
        # Convert to torch tensors
        nonda_pts3d = torch.from_numpy(nonda_pts3d).float().to(device)  # (N_nonda, 3)
        da_pts3d = torch.from_numpy(da_pts3d).float().to(device)  # (N_da, 3)
        da_distances = torch.from_numpy(da_distances).float().to(device)  # (N_da,)
        
        # Check for invalid depths (z < 0)
        nonda_valid = nonda_pts3d[:, 2] >= 0  # (N_nonda,)
        da_valid = da_pts3d[:, 2] >= 0  # (N_da,)
        
        # Compute pairwise 3D distances: (N_nonda, N_da)
        # Broadcasting: (N_nonda, 1, 3) - (1, N_da, 3) -> (N_nonda, N_da, 3)
        diff = nonda_pts3d.unsqueeze(1) - da_pts3d.unsqueeze(0)  # (N_nonda, N_da, 3)
        euclidean_dists = torch.norm(diff, dim=2)  # (N_nonda, N_da)
        
        # Add DA distances to goal
        total_dists = euclidean_dists + da_distances.unsqueeze(0)  # (N_nonda, N_da)
        
        # Mask out invalid DA points (set to max_dist)
        total_dists[:, ~da_valid] = max_dist
        
        # Find minimum distance to goal via any DA point
        min_dists, _ = torch.min(total_dists, dim=1)  # (N_nonda,)
        
        # Set invalid Non-DA points to max_dist
        min_dists[~nonda_valid] = max_dist
        
        # Move back to CPU and convert to numpy
        nonda_distances = min_dists.cpu().numpy()
        
        return nonda_distances

    def save_costmaps(self, costmaps: np.ndarray, metadata: dict, filename=None):
        """
        Save costmaps to compressed NPZ file.
        
        Args:
            costmaps: Array of shape (N_images, H, W)
            filename: Output filename
        """
        if filename is None:
            filename = self._get_costmap_filename(metadata)

        save_path = self.out_dir / filename
        
        np.savez_compressed(
            save_path,
            costmaps=costmaps,
            metadata=json.dumps(metadata)
        )
        
        print(f"✓ Saved costmaps to {save_path}")
    
    @staticmethod
    def load_costmap_file(filepath):
        """
        Load a costmap .npz file and return (costmap, metadata).
        Args:
            filepath (str or Path): Path to the .npz file.
        Returns:
            costmap (np.ndarray): The costmap array.
            metadata (dict): Metadata dictionary.
        """
        data = np.load(filepath, allow_pickle=True)
        costmap = data["costmaps"]
        metadata = json.loads(data["metadata"].item())
        return costmap, metadata

    @time_function("get_mast3r_matches")
    def get_mast3r_matches(self, img0_path, img1_path):
        """Get matches between two images using configured matcher (MASt3R, SuperPoint, or GT)"""
        if self.use_gt_matches:
            return self._get_gt_matches(img0_path, img1_path)
        
        # Use the configured matcher if available
        if self.matcher is not None:
            # Load images and run matcher
            img0 = self.matcher.load_image(img0_path, resize=(self.H, self.W))
            img1 = self.matcher.load_image(img1_path, resize=(self.H, self.W))
            result = self.matcher(img0, img1)
            # Return inlier matches (post-RANSAC if geometric_verification enabled)
            return result["inliers0"], result["inliers1"]
        else:
            # Fall back to MASt3R inference wrapper
            return self.mast3r.get_matches(
                img0_path, img1_path, 
                resize=(self.H, self.W),
                subsample=self.cfg.model.subsample_or_initxy1,
                dist=self.cfg.model.dist,
                block_size=self.cfg.model.block_size,
                match_feature_type=self.cfg.model.match_feature_type
            )
    
    def _get_gt_matches(self, img0_path, img1_path):
        """Get ground-truth matches between two images"""
        if not hasattr(self, 'gt_matches'):
            with open(self.gt_matches_path, "rb") as f:
                gt_matches_dict = pickle.load(f)
                self.gt_matches_image_size = gt_matches_dict['image_size']
                self.gt_matches = gt_matches_dict['matches']
        
        # Extract frame indices
        img0_idx = self.img_paths.index(img0_path)
        img1_idx = self.img_paths.index(img1_path)
        
        scale_y = self.H / self.gt_matches_image_size[0]
        scale_x = self.W / self.gt_matches_image_size[1]
        
        # Find matching pair
        for pair in self.gt_matches:
            fi_idx = pair['frame_i']['frame_idx']
            fj_idx = pair['frame_j']['frame_idx']
            
            if (fi_idx == img0_idx and fj_idx == img1_idx):
                matches_im0 = np.round(pair['frame_i']['pixels'] * [scale_x, scale_y]).astype(int)
                matches_im1 = np.round(np.array([m['pixel_j'] for m in pair['matches']]) * [scale_x, scale_y]).astype(int)
                
                # Clip to valid bounds
                matches_im0[:, 0] = np.clip(matches_im0[:, 0], 0, self.W - 1)
                matches_im0[:, 1] = np.clip(matches_im0[:, 1], 0, self.H - 1)
                matches_im1[:, 0] = np.clip(matches_im1[:, 0], 0, self.W - 1)
                matches_im1[:, 1] = np.clip(matches_im1[:, 1], 0, self.H - 1)
                
                return matches_im0, matches_im1
            elif (fi_idx == img1_idx and fj_idx == img0_idx):
                matches_im1 = np.round(pair['frame_i']['pixels'] * [scale_x, scale_y]).astype(int)
                matches_im0 = np.round(np.array([m['pixel_j'] for m in pair['matches']]) * [scale_x, scale_y]).astype(int)
                
                # Clip to valid bounds
                matches_im0[:, 0] = np.clip(matches_im0[:, 0], 0, self.W - 1)
                matches_im0[:, 1] = np.clip(matches_im0[:, 1], 0, self.H - 1)
                matches_im1[:, 0] = np.clip(matches_im1[:, 0], 0, self.W - 1)
                matches_im1[:, 1] = np.clip(matches_im1[:, 1], 0, self.H - 1)
                
                return matches_im0, matches_im1
        
        # No match found
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    def update_goal_node_only(self, goal_img_idx, goal_node_idx=None, goal_coords=None):
        """Fast update of goal node without rebuilding graph"""
        t_start = time.time()
        
        # Remove old goal node edges
        if 'goal_node_id' in self.G.graph:
            old_goal_id = self.G.graph['goal_node_id']
            edges_to_remove = [(u, v) for u, v in self.G.edges() 
                                if (v == old_goal_id or u == old_goal_id)]
            self.G.remove_edges_from(edges_to_remove)
            print(f"Removed {len(edges_to_remove)} old goal edges")

            # Reset old goal node type
            if old_goal_id in self.G.nodes:
                self.G.nodes[old_goal_id]['type'] = 'da'
        
        # Determine new goal node ID
        if goal_node_idx is None:
            if goal_coords is None:
                raise ValueError("Either goal_node_idx or goal_coords must be provided")
            cx, cy = goal_coords
            goal_node_idx = self.pixel_coord_to_global_node_id(goal_img_idx, cx, cy)
        else:
            cx, cy = self.G.nodes[goal_node_idx]['pixel']
        
        # Update graph attributes
        self.G.graph['goal_img_idx'] = goal_img_idx
        self.G.graph['goal_node_coords'] = (cx, cy)
        self.G.graph['goal_node_id'] = goal_node_idx
        
        # Update node type
        if goal_node_idx in self.G.nodes:
            self.G.nodes[goal_node_idx]['type'] = 'goal'
        
        # Reconnect DA nodes to new goal
        t_connect = time.time()
        self.connect_da_to_goal_node(goal_img_idx, goal_node_idx)
        print(f"Connected DA nodes to new goal: {time.time() - t_connect:.3f}s")
        
        # Recompute distances
        t_dijkstra = time.time()
        path_lengths = self.get_single_source_paths(self.G, source_node=goal_node_idx, weight='weight')
        self.all_path_lengths = path_lengths
        self.G.graph['all_path_lengths'] = {'weight': path_lengths}
        print(f"Recomputed distances: {time.time() - t_dijkstra:.3f}s")
        
        print(f"Total goal update time: {time.time() - t_start:.3f}s")
        return self.G

    def update_loop_closures_only(self, lc_txt_path, recompute_paths=True):
        """Fast update of loop-closure edges without rebuilding graph"""
        t_start = time.time()
        
        # Remove old LC edges
        old_lc_edges = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if d.get("edge_type") == "da_loop"
        ]
        self.G.remove_edges_from(old_lc_edges)
        print(f"Removed {len(old_lc_edges)} old LC edges")
        
        # Read oracle LC pairs
        lc_pairs = read_oracle_lc_pairs(lc_txt_path, len(self.img_paths))
        print(f"Loaded {len(lc_pairs)} LC image pairs")
        
        # Add new LC edges
        new_lc_edges = self._create_loop_edges_from_pairs(lc_pairs)
        self.G.add_edges_from(new_lc_edges)
        print(f"Added {len(new_lc_edges)} new LC edges")
        
        # Recompute paths if requested
        if recompute_paths and "goal_node_id" in self.G.graph:
            goal = self.G.graph["goal_node_id"]
            print("Recomputing shortest paths from goal...")
            t_dij = time.time()
            path_lengths = self.get_single_source_paths(self.G, goal, weight="weight")
            self.G.graph["all_path_lengths"] = {"weight": path_lengths}
            self.all_path_lengths = path_lengths
            print(f"Dijkstra took {time.time() - t_dij:.3f}s")
        
        print(f"LC update done in {time.time() - t_start:.3f}s")
        return self.G

    @time_function("save_compressed_graph_chunked")
    def save_compressed_graph_chunked(self, path, graph=None):
        """
        Simple and reliable compression using blosc2 with manual chunking for large data
        """
        if graph is None:
            graph = self.G

        t_start = time.time()
        graph_data = self.decompose_graph_data(graph)

        # Serialize
        serialized = pickle.dumps(graph_data, protocol=pickle.HIGHEST_PROTOCOL)
        original_size = len(serialized)

        # Compress with chunking support
        compressed_path = f"{path}.b2s"
        BLOSC2_MAX_SIZE = 2000000000
        
        if original_size <= BLOSC2_MAX_SIZE:
            # Single compression
            compressed = blosc2.compress(serialized, codec=blosc2.Codec.ZSTD, clevel=9)
            
            with open(compressed_path, "wb") as f:
                f.write(b"SINGLE")
                f.write(len(compressed).to_bytes(8, "little"))
                f.write(compressed)
            
            compressed_size = len(compressed)
        else:
            # Chunked compression
            chunk_size = 1024 * 1024 * 1024  # 1GB chunks
            num_chunks = (len(serialized) + chunk_size - 1) // chunk_size
            compressed_chunks = []
            
            print(f"Data too large, using {num_chunks} chunks...")
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(serialized))
                chunk = serialized[start_idx:end_idx]
                compressed_chunk = blosc2.compress(chunk, codec=blosc2.Codec.ZSTD, clevel=9)
                compressed_chunks.append(compressed_chunk)
            
            # Save chunked data
            with open(compressed_path, "wb") as f:
                f.write(b"CHUNKS")
                f.write(num_chunks.to_bytes(4, "little"))
                f.write(original_size.to_bytes(8, "little"))
                
                for chunk in compressed_chunks:
                    f.write(len(chunk).to_bytes(4, "little"))
                
                for chunk in compressed_chunks:
                    f.write(chunk)
            
            compressed_size = sum(len(chunk) for chunk in compressed_chunks)

        compression_ratio = (1 - compressed_size / original_size) * 100
        print(f"Saved to: {compressed_path} | Total Time: {time.time() - t_start:.3f}s | Compression: {compression_ratio:.1f}%")

        return compressed_path

    def decompose_graph_data(self, graph=None):
        """Decompose graph into arrays for better compression"""
        t_start = time.time()

        if graph is None:
            graph = self.G
        
        graph_data = {"directed": graph.is_directed(), "graph_attrs": dict(graph.graph)}
        
        # Decompose nodes
        nodes_list = list(graph.nodes(data=True))
        node_ids = []
        node_maps = []
        node_coords = []
        node_pixels = []
        node_types = []

        print(f"Processing {len(nodes_list)} nodes...")
        for node_id, attrs in nodes_list:
            node_ids.append(node_id)

            # Extract specific attributes
            node_maps.append(attrs.get("map", [0, 0]))
            node_coords.append(
                attrs.get("coord_mast3r", np.array([0, 0, 0], dtype=np.float64))
            )
            node_pixels.append(attrs.get("pixel", [0, 0]))
            node_types.append(attrs.get("type", "unknown"))

        # Convert to numpy arrays for better compression
        graph_data["node_ids"] = np.array(node_ids)
        graph_data["node_maps"] = np.array(node_maps, dtype=np.int32)
        graph_data["node_coords"] = np.array(node_coords, dtype=np.float64)
        graph_data["node_pixels"] = np.array(node_pixels, dtype=np.int32)
        graph_data["node_types"] = np.array(node_types, dtype="U20")  # String array

        # Decompose edges and their attributes
        edges_list = list(graph.edges(data=True))
        edge_sources = []
        edge_targets = []
        edge_types = []
        edge_weights = []

        print(f"Processing {len(edges_list)} edges...")
        for source, target, attrs in edges_list:
            edge_sources.append(source)
            edge_targets.append(target)
            edge_types.append(attrs.get("edge_type", "unknown"))
            edge_weights.append(attrs.get("weight", 0.0))

        # Convert to numpy arrays
        graph_data["edge_sources"] = np.array(edge_sources, dtype=np.int32)
        graph_data["edge_targets"] = np.array(edge_targets, dtype=np.int32)
        graph_data["edge_types"] = np.array(edge_types, dtype="U10")  # String array
        graph_data["edge_weights"] = np.array(edge_weights, dtype=np.float64)

        # Handle allPathLengths specially (convert dict to arrays)
        if "all_path_lengths" in graph_data["graph_attrs"]:
            path_lengths = graph_data["graph_attrs"]["all_path_lengths"]["weight"]
            path_nodes = np.array(list(path_lengths.keys()))
            path_distances = np.array(list(path_lengths.values()), dtype=np.float64)

            graph_data["path_nodes"] = path_nodes
            graph_data["path_distances"] = path_distances
            del graph_data["graph_attrs"]["all_path_lengths"]
            print(f"Converted {len(path_lengths)} path lengths to arrays")

        print(f"Data preparation took: {time.time() - t_start:.3f}s")
        return graph_data

    def load_base_graph_and_add_goal(self):
        """
        Load existing base graph, add goal from config, compute distances, save new files.
        
        This method is used for update_graph mode to avoid rebuilding the entire graph.
        """
        # Load base graph
        base_graph_path = self.scene_dir / self.cfg.scenes.costmap_dirname / self.cfg.goal.base_graph_path
        if not base_graph_path.exists():
            raise FileNotFoundError(f"Base graph not found: {base_graph_path}")
        
        # Get goal from config and compute distances
        goal_img_idx = self.cfg.goal.image_idx
        goal_px = self.cfg.goal.pixel_x
        goal_py = self.cfg.goal.pixel_y
        
        graph_filename = self._get_goal_graph_filename(goal_img_idx)
        if not self.recreate_graphs and os.path.exists(self.out_dir / graph_filename):
            print(f"Goal graph already exists at {self.out_dir / graph_filename}, skipping update.")
            return self.G
        
        print(f"Loading base graph from: {base_graph_path}")
        self.G = load_compressed_graph_chunked(str(base_graph_path))
        
        self.compute_distances_to_goal_node(goal_img_idx, goal_px, goal_py)
        
        # Save graph with goal
        self.save_compressed_graph_chunked(str(self.out_dir / graph_filename))
        
        print(f"✓ Updated graph saved with goal node")
        return self.G

def make_topo_map(scene_dir: Path, img_dir: Path, out_dir: Path, cfg: DictConfig) -> nx.Graph:
    """
    Create topological map for a single scene.
    
    Args:
        scene_dir: Scene directory path
        img_dir: Images directory path
        out_dir: Output directory path
        cfg: Hydra configuration
        
    Returns:
        NetworkX graph with distances to goal node
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING SCENE: {scene_dir.name}")
    print(f"{'='*80}\n")

    # Get goal mode (default to "config" for backward compatibility)
    goal_mode = cfg.goal.get("mode", "config")
    print(f"Goal mode: {goal_mode}")

    # Initialize mapper
    start_time = time.time()
    
    # Enable timing manager
    timing_manager.enable()
    
    topo_map = MapTopological3DPoints(str(img_dir), str(out_dir), cfg)

    base_filename = topo_map._get_base_graph_filename()
    
    # Handle update_graph mode separately (doesn't create new graph)
    if goal_mode == "update_graph":
        print("\nMode: UPDATE_GRAPH - Loading base graph and adding goal")
        topo_map.load_base_graph_and_add_goal()
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✓ SCENE COMPLETE in {total_time:.2f}s")
        print(f"{'='*80}\n")
        return topo_map.G
    elif goal_mode == "none":
        if not topo_map.recreate_graphs and os.path.exists(out_dir / (base_filename+'.b2s')):
            print(f"Base graph already exists at {out_dir / (base_filename+'.b2s')}, skipping map creation.")
            return None

    # For all other modes: create topological map first
    print(f"\nMode: {goal_mode.upper()} - Creating topological map...")
    t1 = time.time()
    topo_map.create_map_topo()
    print(f"✓ Created topological map in {time.time() - t1:.2f}s")
    print(f"  Graph: {topo_map.G.number_of_nodes()} nodes, {topo_map.G.number_of_edges()} edges")

    # Save base graph (always, before adding goal distances)
    print("\nSaving base graph...")
    if cfg.compression.enabled:
        topo_map.save_compressed_graph_chunked(str(out_dir / base_filename))
    else:
        with open(out_dir / base_filename, 'wb') as f:
            pickle.dump(topo_map.G, f)

    # Compute goal distances (for config and episode modes)
    if goal_mode == "episode":
        print("\nInferring goal from episode folder...")
        topo_map.get_goal_from_episode()
        
        print("Computing distances to goal node...")
        t2 = time.time()
        topo_map.compute_distances_to_goal_node()
        print(f"✓ Computed distances in {time.time() - t2:.2f}s")
        
        # Save goal graph
        print("\nSaving goal graph...")
        goal_filename = topo_map._get_goal_graph_filename()
        if cfg.compression.enabled:
            topo_map.save_compressed_graph_chunked(str(out_dir / goal_filename))
        else:
            with open(out_dir / goal_filename, 'wb') as f:
                pickle.dump(topo_map.G, f)

    elif goal_mode == "config":
        print("\nComputing distances to goal node...")
        t2 = time.time()
        topo_map.compute_distances_to_goal_node()
        print(f"✓ Computed distances in {time.time() - t2:.2f}s")
        
        # Save goal graph
        print("\nSaving goal graph...")
        goal_filename = topo_map._get_goal_graph_filename()
        if cfg.compression.enabled:
            topo_map.save_compressed_graph_chunked(str(out_dir / goal_filename))
        else:
            with open(out_dir / goal_filename, 'wb') as f:
                pickle.dump(topo_map.G, f)

    elif goal_mode == "none":
        # Base graph already saved, nothing more to do
        pass

    else:
        raise ValueError(f"Unknown goal mode: {goal_mode}. Expected: none, config, episode, update_graph")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ SCENE COMPLETE in {total_time:.2f}s")
    print(f"{'='*80}\n")

    # Print timing summary and save to CSV
    timing_manager.print_all_function_summaries()
    csv_path = out_dir / "timing_analysis.csv"
    timing_manager.save_function_timings_csv(str(csv_path))
    
    # Clear timing data for next scene (in multi-scene mode)
    timing_manager.clear_all_data()

    return topo_map.G

def get_scene_list(cfg: DictConfig) -> list:
    """
    Get list of scene paths based on config.
    
    Priority:
    1. scene_list_file (if provided)
    2. start_idx/end_idx/step filtering
    3. Single scene_name (when multi_scene=false)
    """
    base_dir = Path(cfg.scenes.base_dir)
    
    if not cfg.scenes.multi_scene:
        # Single scene mode
        scene_path = base_dir / cfg.scenes.scene_name
        print(f"Single scene mode: {scene_path.name}")
        return [scene_path]
    
    # Multi-scene mode: check txt file first
    list_file = cfg.scenes.get("scene_list_file")
    if list_file and Path(list_file).exists():
        with open(list_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        scenes = [base_dir / name for name in names if (base_dir / name).exists()]
        print(f"Multi-scene mode (from file): {len(scenes)} scenes")
        return scenes
    
    # Multi-scene mode: use start/end/step
    all_scenes = natsorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda x: x.name)
    
    start = cfg.scenes.get("start_idx", 0)
    end = cfg.scenes.get("end_idx", -1)
    step = cfg.scenes.get("step", 1)
    
    if end == -1:
        end = len(all_scenes)
    
    scenes = all_scenes[start:end:step]
    print(f"Multi-scene mode: {len(scenes)} scenes (indices {start}:{end}:{step})")
    return scenes


@hydra.main(version_base=None, config_path="../../configs/mapper", config_name="mapper_config")
def main(cfg: DictConfig):
    print("\n" + "="*80)
    print("TOPOLOGICAL MAP CREATION")
    print("="*80)
    print(f"\nUsing configuration from: {cfg}")
    print("="*80 + "\n")

    # Set environment variable for base directory
    os.environ['BASE_DIR'] = str(BASE_DIR)

    # Get scene list
    scenes = get_scene_list(cfg)
    
    if len(scenes) == 0:
        raise ValueError("No scenes found to process")
    
    # Disable goal computation for multi-scene (base graph only)
    if cfg.scenes.multi_scene:
        print("Multi-scene mode: disabling goal computation (base graph only)")
    
    # Process each scene
    results = {}
    base_out_dir = cfg.scenes.get("base_out_dir", None)
    
    for scene_num, scene_dir in enumerate(tqdm(scenes, desc="Processing scenes", unit="scene")):
        img_dir = scene_dir / "images_fov90"
        
        # Determine output directory
        if base_out_dir is not None:
            out_dir = Path(base_out_dir) / scene_dir.name / cfg.scenes.costmap_dirname
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = scene_dir / cfg.scenes.costmap_dirname
            out_dir.mkdir(parents=True, exist_ok=True)
        
        if not img_dir.exists():
            print(f"⚠ Skipping {scene_dir.name}: no images_fov90/ folder")
            results[scene_dir.name] = False
            continue
        
        print(f"\nScene: {scene_dir.name}")
        print(f"Images: {img_dir}")
        print(f"Output: {out_dir}\n")
        
        graph = make_topo_map(scene_dir, img_dir, out_dir, cfg)
        results[scene_dir.name] = graph is not None
    
    # Summary
    successful = sum(results.values())
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE: {successful}/{len(results)} scenes processed successfully")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()