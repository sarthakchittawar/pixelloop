import os
import numpy as np
from natsort import natsorted
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging

logger = logging.getLogger("[Localizer]")

from libs.matcher.mast3r_matcher import Mast3rMatcher, BaseMatcher
from omegaconf import OmegaConf, DictConfig

class LocalizeTopological:
    def __init__(self, map_img_paths: Path, H: int, W: int, matcher: BaseMatcher, cfg: DictConfig):
        self.map_img_paths = map_img_paths
        self.H, self.W = H, W
        self.matcher = matcher
        self.cfg = cfg
        self.lost = False

        # Iterator bounds for candidate selection
        self.localizer_iter_lb = 0
        self.localizer_iter_ub = len(map_img_paths)
        self.localized_img_idx = 0
        self.greedy_propeller = cfg.greedy_propeller  # If True, only move forward along map
        self.subsample_ref = cfg.subsample_ref

        # History for voting
        self.matched_img_history = []
        self.history_size = cfg.get("history_size", 8)
        self.ref_imgs_tried = np.array([])
        
        # Relocalization related parameters
        self.try_relocalize = not cfg.get("use_gt_localization", False)
        self.reloc_dia_default = 2 * cfg.loc_radius
        self.reloc_rad_add = 2 * cfg.reloc_rad_add
        # self.reloc_dia_max = 2 * cfg.reloc_rad_max
        self.reloc_dia = self.reloc_dia_default
        self.reloc_exhausted = False
        self.min_num_matches = cfg.min_num_matches
        
        logger.info(f"LocalizeTopological initialized: H={H}, W={W}, num_map_images={len(map_img_paths)}")
    
    def update_localizer_iter_lb(self):
        """Update search window based on current localized position."""
        if self.greedy_propeller:
            if self.localized_img_idx > self.localizer_iter_lb:
                self.localizer_iter_lb = self.localized_img_idx
        else:
            self.localizer_iter_lb = max(0, self.localized_img_idx - self.reloc_dia // 2)

    def get_ref_img_indices(self):
        """Get candidate reference image indices."""
        lb = self.localizer_iter_lb
        ub = min(lb + self.reloc_dia, self.localizer_iter_ub)
        return np.arange(lb, ub)[::self.subsample_ref]
    
    def relocalize(self, qry_img, candidate_indices):
        """Expand search and retry localization."""
        self.reloc_dia += 2 * self.reloc_rad_add
        logger.info(f"Relocalizing with expanded radius: {self.reloc_dia}")
        return self.localize(qry_img, candidate_indices)

    
    def localize(self, qry_img, candidate_indices):
        """
        Localize query image using pixel-level matching.
        
        Args:
            qry_img: Query image (H, W, 3)
            candidate_indices: candidate image indices to match against
            
        Returns:
            qry_mkpts: (N, 2) query matched keypoints
            ref_mkpts: (N, 3) reference matched keypoints (img_idx, px, py)
            confidences: match confidences
        """
        start_time = time.time()

        if candidate_indices is None:
            self.update_localizer_iter_lb()
            logger.warning("No candidate indices provided, Auto-Genering candidate map img indices")
            candidate_indices = self.get_ref_img_indices()
        
        if len(candidate_indices) == 0:
            self.lost = True
            logger.warning("No candidate indices provided, marking as lost")
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3), []
        
        logger.info(f"Localizing against {len(candidate_indices)} candidate images: {candidate_indices}")

        # Getting the list of candidate map images
        ref_img_list = [self.map_img_paths[ref_idx] for ref_idx in candidate_indices]

        # Match against all candidates map images
        # matches in (x, y) order
        match_pairs, confidences, _ = self.matcher.match_one_to_many(qry_img, ref_img_list)
        logger.info(f"Found {len(match_pairs)} total matches across {len(candidate_indices)} images")

        # Query and Reference image matched keypoints
        qry_mkpts = []
        ref_mkpts = []
        total_matches = 0
        for i, ref_idx in enumerate(candidate_indices):
            mkpts1, mkpts2 = match_pairs[i]  # (N, 2), (N, 2)
            num_matches = mkpts2.shape[0]
            total_matches += num_matches
            
            logger.debug(f"Image {ref_idx}: {num_matches} matches")

            # (ref_img_idx, ref_px, ref_py) per match
            img_idx_col = np.full(num_matches, ref_idx)
            mkpts2_with_img_idx = np.column_stack((img_idx_col, mkpts2))  # (N, 3)
            qry_mkpts.append(mkpts1)
            ref_mkpts.append(mkpts2_with_img_idx)
        
        if len(qry_mkpts) == 0 or total_matches <= self.min_num_matches:
            self.lost = True
            logger.warning(f"Lost! {total_matches} matches < {self.min_num_matches}")
            
            # Recursive relocalization if lost
            if self.try_relocalize:
                untried = np.setdiff1d(candidate_indices, self.ref_imgs_tried)
                if len(untried) == 0 or self.reloc_dia > 2 * self.cfg.reloc_rad_max:
                    logger.info("Relocalization exhausted")
                    self.reloc_exhausted = True
                else:
                    self.ref_imgs_tried = np.append(self.ref_imgs_tried, candidate_indices)
                    return self.relocalize(qry_img, untried)
            
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3), []
        
        # Found enough matches - not lost
        self.lost = False
    
        # Reset state after successful localization or exhausted relocalization
        self.reloc_dia = self.reloc_dia_default
        self.ref_imgs_tried = np.array([])
        self.reloc_exhausted = False

        qry_mkpts = np.concatenate(qry_mkpts, axis=0)  # (N, 2)
        ref_mkpts = np.concatenate(ref_mkpts, axis=0)  # (N, 3)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Localization complete: {total_matches} total matches in {elapsed_time:.3f}s")

        # Update localized position via voting
        matched_imgs = ref_mkpts[:, 0].astype(int)
        self.matched_img_history.append(matched_imgs)
        if len(self.matched_img_history) > self.history_size:
            self.matched_img_history = self.matched_img_history[-self.history_size:]
        
        all_matched = np.concatenate(self.matched_img_history)
        votes = np.bincount(all_matched)
        self.localized_img_idx = int(np.argmax(votes))
        logger.info(f"Localized to image index {self.localized_img_idx}")

        return qry_mkpts, ref_mkpts, confidences