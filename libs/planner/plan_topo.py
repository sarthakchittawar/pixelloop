import numpy as np
import torch
import logging
from omegaconf import OmegaConf, DictConfig

from libs.mapper.create_topomap import CostmapData

logger = logging.getLogger("[Planner]")

class PlanTopological:
    def __init__(self, H: int, W: int, costmap_data: CostmapData, device: str, cfg: DictConfig):
        self.H = H
        self.W = W
        self.max_pl = 1e6 # put this in config later
        self.device = device
        self.costmap_data = costmap_data
        self.cfg = cfg
        
        logger.info(f"PlanTopological initialized: H={H}, W={W}, device={device}")

    def get_matched_pixel_pathlengths(self, ref_mkpts):
        """
        Takes in a list of (N, 3) reference image matches where each match is of the form: (ref_img_id, px, py)

        Returns:
            - ref_match_pathlengths (N, ): array containing the pathlengths for each of the matched ref image pixels
        """
        # Get the pre-computed costmaps for all the reference images
        ref_costmaps = self.costmap_data.get_costmap() # (N, H, W)

        # Getting the reference image pixel costs from the reference image costmaps
        ref_img_ids, ref_px, ref_py = ref_mkpts[:, 0].astype(int), ref_mkpts[:, 1].astype(int), ref_mkpts[:, 2].astype(int)
        ref_mkpts_pathlengths = ref_costmaps[ref_img_ids, ref_py, ref_px] # (N,)
        
        logger.debug(f"Matched {len(ref_mkpts)} reference pixels, min_pl={ref_mkpts_pathlengths.min():.2f}, max_pl={ref_mkpts_pathlengths.max():.2f}")

        return ref_mkpts_pathlengths
    
    def get_query_pathlengths(self, qry_mkpts, qry_mkpts_pathlengths, qry_pts3d):
        """
        qry_mkpts: (N, 2)
        qry_mkpts_pathlengths: (N, )
        qry_pts3d: (N, 3)
        """
        H, W = self.H, self.W
        pathlengths = np.full((H, W), self.max_pl, dtype=np.float32)
        qry_pts3d_flat = qry_pts3d.reshape(-1, 3) # (H*W, 3)

        # Get linear pixel index for the matched query pixels
        px = qry_mkpts[:, 0].astype(int)
        py = qry_mkpts[:, 1].astype(int)
        matched_pixel_indices = py * W + px

        # In case the same pixel appears multiple times, keep the min path-length
        np.minimum.at(pathlengths, (py, px), qry_mkpts_pathlengths)

        # Get the unique matched query pixel indices and their corresponding path lengths
        matched_pixel_indices = np.unique(matched_pixel_indices)
        matched_pathlengths = pathlengths.ravel()[matched_pixel_indices]

        # Compute unmatched pixels using a mask
        full_mask = np.ones(H * W, dtype=bool)
        full_mask[matched_pixel_indices] = False
        unmatched_pixel_indices = np.nonzero(full_mask)[0]

        # Path-length computation for unmatched query pixels
        unmatched_pathlengths = self.get_unmatched_query_pathlengths(
            qry_pts3d_flat,
            unmatched_pixel_indices,
            matched_pixel_indices,
            matched_pathlengths)

        # Fill pathlengths array with the unmatched query pixel distances
        pathlengths_flat = pathlengths.ravel()
        pathlengths_flat[unmatched_pixel_indices] = unmatched_pathlengths
        pathlengths = pathlengths_flat.reshape(H, W)
        
        logger.debug(f"Query pathlengths: matched={len(matched_pixel_indices)}, unmatched={len(unmatched_pixel_indices)}, min={pathlengths.min():.2f}, max={pathlengths.max():.2f}")
        
        return pathlengths

    def get_unmatched_query_pathlengths(
        self, 
        pts3d_flat, 
        unmatched_pixel_indices,
        matched_pixel_indices, 
        matched_pathlengths):

        device = torch.device(self.device)
        max_pl = self.max_pl

        # Get 3D coordinates
        unmatched_pts3d = pts3d_flat[unmatched_pixel_indices]  # (N_unmatched, 3)
        matched_pts3d = pts3d_flat[matched_pixel_indices]  # (N_matched, 3)
        
        # Convert to torch tensors
        unmatched_pts3d = torch.from_numpy(unmatched_pts3d).float().to(device)  # (N_unmatched, 3)
        matched_pts3d = torch.from_numpy(matched_pts3d).float().to(device)  # (N_matched, 3)
        matched_pathlengths = torch.from_numpy(matched_pathlengths).float().to(device)  # (N_matched,)
        
        # Check for invalid depths (z < 0)
        unmatched_valid = unmatched_pts3d[:, 2] >= 0  # (N_unmatched,)
        matched_valid = matched_pts3d[:, 2] >= 0  # (N_matched,)
        
        # Compute pairwise 3D distances: (N_unmatched, N_matched)
        # Broadcasting: (N_unmatched, 1, 3) - (1, N_matched, 3) -> (N_unmatched, N_matched, 3)
        diff = unmatched_pts3d.unsqueeze(1) - matched_pts3d.unsqueeze(0)  # (N_unmatched, N_matched, 3)
        euclidean_dists = torch.norm(diff, dim=2)  # (N_unmatched, N_matched)
        
        # Add matched pathlengths to goal
        total_dists = euclidean_dists + matched_pathlengths.unsqueeze(0)  # (N_unmatched, N_matched)
        
        # Mask out invalid matched points (set to max_pl)
        total_dists[:, ~matched_valid] = max_pl
        
        # Find minimum distance to goal via any matched point
        min_dists, _ = torch.min(total_dists, dim=1)  # (N_unmatched,)
        
        # Set invalid Non-matched points to max_pl
        min_dists[~unmatched_valid] = max_pl
        
        # Move back to CPU and convert to numpy
        unmatched_pathlengths = min_dists.cpu().numpy()
        
        return unmatched_pathlengths 