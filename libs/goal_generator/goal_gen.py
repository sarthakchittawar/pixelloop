"""
Goal Generator for Visual Navigation

Coordinates localization and planning to generate pixel-wise goal masks.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import logging

from libs.localizer.loc_topo import LocalizeTopological
from libs.planner.plan_topo import PlanTopological
from libs.common.utils_sim import depth_to_3d_points

logger = logging.getLogger("[GoalGenerator]")


class GoalGenerator:
    """
    Generates pixel-wise goal masks by coordinating localization and planning.
    
    This class orchestrates the following pipeline:
    1. Localize query image against map images
    2. Get path lengths for matched reference pixels from costmap
    3. Compute path lengths for all query pixels (matched + unmatched)
    4. Return goal mask (H x W cost map)
    """
    
    def __init__(
        self,
        H: int,
        W: int,
        localizer: LocalizeTopological,
        planner: PlanTopological,
        cfg: DictConfig
    ):
        """
        Initialize goal generator.
        
        Args:
            H: Image height
            W: Image width
            localizer: Localization module
            planner: Planning module
            cfg: Hydra configuration
        """
        self.H = H
        self.W = W
        self.localizer = localizer
        self.planner = planner
        self.cfg = cfg
        
        # Config parameters
        self.max_pl = 1e6
        # self.max_pl = cfg.goal_generator.get('max_path_length', 1e6)
        
        # State tracking
        self.iteration = 0
        self.goal_mask_default = np.full((H, W), self.max_pl, dtype=np.float32)
        self.goal_mask = self.goal_mask_default.copy()
        
        logger.info(f"GoalGenerator initialized: H={H}, W={W}, max_pl={self.max_pl}")
    
    def get_goal_mask(
        self,
        qry_img: np.ndarray,
        qry_depth: np.ndarray,
        qry_pts3d: Optional[np.ndarray] = None,
        intrinsics: Optional[torch.Tensor] = None,
        candidate_img_indices: np.ndarray = None,
        return_vis_data: bool = False
    ):
        """
        Compute pixel-wise goal mask for query image.
        
        Args:
            qry_img: Query RGB image (H, W, 3)
            qry_depth: Query depth map (H, W)
            qry_pts3d: Optional pre-computed 3D points (H, W, 3)
            intrinsics: Camera intrinsics for depth projection
            candidate_img_indices: Candidate map image indices
            return_vis_data: If True, return (goal_mask, vis_data) tuple
            
        Returns:
            goal_mask: Path length for each pixel (H, W)
            OR
            (goal_mask, vis_data): If return_vis_data=True, also returns dict with:
                - qry_mkpts: (N, 2) query matched pixels
                - ref_mkpts: (N, 2) reference matched pixels  
                - confidences: (N,) match confidences
                - localized_img_idx: localized reference image index
        """
        # Compute 3D points if not provided
        self.iteration += 1
        logger.info(f"--- Goal Mask Iteration {self.iteration} ---")
        
        if qry_pts3d is None:
            if intrinsics is None:
                raise ValueError("Must provide either qry_pts3d or intrinsics")
            qry_pts3d = depth_to_3d_points(qry_depth, intrinsics)
        
        # Step 1: Localize - find matches between query and map
        # qry_mkpts: (N, 2), ref_mkpts: (N, 3), confidences: (N, )
        qry_mkpts, ref_mkpts, confidences = self.localizer.localize(
            qry_img, candidate_img_indices
        )
        
        # Handle localization failure
        if len(qry_mkpts) == 0:
            logger.warning(f"Localization lost or no matches found")
            if return_vis_data:
                empty_vis_data = {
                    'qry_mkpts': np.array([]).reshape(0, 2),
                    'ref_mkpts': np.array([]).reshape(0, 2),
                    'confidences': np.array([]),
                    'localized_img_idx': -1
                }
                return self.goal_mask_default.copy(), empty_vis_data
            return self.goal_mask_default.copy()
        
        logger.info(f"Found {len(qry_mkpts)} matches, {ref_mkpts.shape = }")
        
        # Step 2: Get reference path lengths from costmap
        # ref_mkpts_pls: (N, )
        ref_mkpts_pls = self.planner.get_matched_pixel_pathlengths(ref_mkpts)
        
        # Step 3: Compute query path lengths for all pixels
        # (H, W)
        qry_pls = self.planner.get_query_pathlengths(
            qry_mkpts, ref_mkpts_pls, qry_pts3d
        )
        
        # Update state
        self.goal_mask = qry_pls
        
        # Log goal mask statistics
        valid_mask = qry_pls < self.max_pl
        if np.any(valid_mask):
            min_pl = qry_pls[valid_mask].min()
            max_pl = qry_pls[valid_mask].max()
            mean_pl = qry_pls[valid_mask].mean()
            logger.info(f"Goal mask stats: valid_pixels={valid_mask.sum()}, min={min_pl:.2f}, max={max_pl:.2f}, mean={mean_pl:.2f}")
        else:
            logger.warning("No valid pixels in goal mask!")
        
        # Return with visualization data if requested
        if return_vis_data:
            # Extract reference pixel coordinates (remove img_idx from ref_mkpts)
            # ref_mkpts is (N, 3) where columns are [img_idx, px, py]
            ref_mkpts_pixels = ref_mkpts[:, 1:3]  # (N, 2) - just [px, py]
            
            vis_data = {
                'qry_mkpts': qry_mkpts,  # (N, 2)
                'ref_mkpts': ref_mkpts_pixels,  # (N, 2)
                'confidences': confidences,  # (N,)
                'localized_img_idx': self.localizer.localized_img_idx
            }
            return self.goal_mask, vis_data
        
        return self.goal_mask