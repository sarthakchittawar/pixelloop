"""
SuperPoint + LightGlue matcher implementation.

Uses the official LightGlue repository: https://github.com/cvg/LightGlue
Install with: pip install git+https://github.com/cvg/LightGlue.git
"""
from __future__ import annotations
import numpy as np
import torch
import hashlib
from pathlib import Path
from typing import Optional, Dict

from .mast3r_matcher import BaseMatcher
from .match_utils import to_numpy

# LightGlue imports
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd  # remove batch dimension
except ImportError:
    raise ImportError(
        "LightGlue is not installed. Install with:\n"
        "pip install git+https://github.com/cvg/LightGlue.git"
    )


class SuperPointMatcher(BaseMatcher):
    """
    Matcher using SuperPoint for keypoint detection and LightGlue for matching.
    
    SuperPoint extracts sparse keypoints with descriptors.
    LightGlue performs learned feature matching between keypoint sets.
    """
    
    def __init__(
        self,
        resize_w: int = 320,
        resize_h: int = 240,
        device: str = 'cuda',
        max_keypoints: int = 2048,
        keypoint_threshold: float = 0.005,
        consistent_keypoints: bool = False,
        **kwargs
    ):
        """
        Args:
            resize_w: Width to resize images to
            resize_h: Height to resize images to  
            device: Device to run inference on ('cuda' or 'cpu')
            max_keypoints: Maximum number of keypoints to extract per image
            keypoint_threshold: Detection threshold for SuperPoint keypoints
            consistent_keypoints: If True, cache keypoints per image so the same 
                keypoints are used regardless of which image pair is being matched.
                This ensures 3D point consistency across different match pairs.
            **kwargs: Additional args passed to BaseMatcher (ransac params, etc.)
        """
        super().__init__(device, **kwargs)
        
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.resize = (self.resize_h, self.resize_w)
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.consistent_keypoints = consistent_keypoints
        
        # Feature cache for consistent keypoints mode
        self._feature_cache: Dict[str, dict] = {}
        
        # Initialize SuperPoint extractor
        self.extractor = SuperPoint(
            max_num_keypoints=max_keypoints,
            detection_threshold=keypoint_threshold,
        ).eval().to(device)
        
        # Initialize LightGlue matcher (for SuperPoint features)
        self.matcher = LightGlue(
            features='superpoint',
        ).eval().to(device)
        
    def preprocess(self, img: torch.Tensor):
        """
        Preprocess image for SuperPoint.
        
        LightGlue expects RGB images in [0, 1] range with shape (C, H, W).
        Returns the image and original shape for coordinate rescaling.
        """
        _, h, w = img.shape
        orig_shape = (h, w)
        
        # LightGlue expects images in [0, 1] range - already the case from load_image
        # No need to normalize like mast3r does
        return img, orig_shape
    
    def _compute_image_hash(self, img: torch.Tensor) -> str:
        """Compute a hash for an image tensor to use as cache key."""
        # Use a fast hash of the image data
        img_bytes = img.cpu().numpy().tobytes()
        return hashlib.md5(img_bytes).hexdigest()

    def _extract_features(self, img: torch.Tensor, img_key: Optional[str] = None) -> dict:
        """
        Extract SuperPoint features from a single image.
        
        Args:
            img: Image tensor (C, H, W) in [0, 1] range
            img_key: Optional cache key. If provided and consistent_keypoints=True,
                     will use cached features if available.
            
        Returns:
            dict with keys: keypoints, descriptors, scores
        """
        # Check cache if consistent_keypoints is enabled
        if self.consistent_keypoints and img_key is not None:
            if img_key in self._feature_cache:
                return self._feature_cache[img_key]
        
        # SuperPoint expects (B, C, H, W)
        img_batch = img.unsqueeze(0)
        
        with torch.no_grad():
            feats = self.extractor.extract(img_batch)
        
        # Cache features if consistent_keypoints is enabled
        if self.consistent_keypoints and img_key is not None:
            self._feature_cache[img_key] = feats
        
        return feats

    def clear_feature_cache(self):
        """Clear the feature cache. Call this when switching to a new scene/dataset."""
        self._feature_cache.clear()

    def get_all_keypoints(self, img: torch.Tensor) -> np.ndarray:
        """
        Get ALL detected keypoints for an image (not just matched ones).
        
        This is useful for visualizing or using consistent keypoints.
        
        Args:
            img: Image tensor (C, H, W) in [0, 1] range
            
        Returns:
            Keypoints array (N, 2) in pixel coordinates
        """
        img_key = self._compute_image_hash(img) if self.consistent_keypoints else None
        feats = self._extract_features(img, img_key=img_key)
        kpts = rbd(feats)['keypoints']
        return to_numpy(kpts)
    
    def _forward(self, img0: torch.Tensor, img1: torch.Tensor):
        """
        Core matching logic using SuperPoint + LightGlue.
        
        Args:
            img0: First image tensor (C, H, W)
            img1: Second image tensor (C, H, W)
            
        Returns:
            Tuple of:
                mkpts0: np.ndarray (N, 2) - matched keypoints in img0
                mkpts1: np.ndarray (N, 2) - matched keypoints in img1
                matches_conf: np.ndarray (N,) - confidence per match
                desc0: torch.Tensor - descriptors for img0
                desc1: torch.Tensor - descriptors for img1
                desc0_conf: torch.Tensor - descriptor confidences (keypoint scores)
                desc1_conf: torch.Tensor - descriptor confidences (keypoint scores)
                conf0: torch.Tensor (H, W) - spatial confidence map
                conf1: torch.Tensor (H, W) - spatial confidence map
        """
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)
        
        # Compute cache keys if consistent_keypoints is enabled
        img0_key = self._compute_image_hash(img0) if self.consistent_keypoints else None
        img1_key = self._compute_image_hash(img1) if self.consistent_keypoints else None
        
        # Extract features from both images (uses cache if available)
        feats0 = self._extract_features(img0, img_key=img0_key)
        feats1 = self._extract_features(img1, img_key=img1_key)
        
        # Match features using LightGlue
        with torch.no_grad():
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        
        # Remove batch dimension
        feats0 = rbd(feats0)  # {keypoints: (N, 2), descriptors: (N, D), scores: (N,)}
        feats1 = rbd(feats1)
        matches01 = rbd(matches01)  # {matches: (M, 2), scores: (M,)}
        
        # Get keypoints and descriptors
        kpts0 = feats0['keypoints']  # (N0, 2)
        kpts1 = feats1['keypoints']  # (N1, 2)
        desc0 = feats0['descriptors']  # (N0, D)
        desc1 = feats1['descriptors']  # (N1, D)
        scores0 = feats0['keypoint_scores']  # (N0,)
        scores1 = feats1['keypoint_scores']  # (N1,)
        
        # Get matches
        matches = matches01['matches']  # (M, 2) indices into kpts0, kpts1
        match_scores = matches01['scores']  # (M,) confidence scores
        
        # Extract matched keypoint coordinates
        valid_mask = matches[:, 0] >= 0  # Filter out unmatched (-1)
        matches = matches[valid_mask]
        match_scores = match_scores[valid_mask]
        
        if len(matches) > 0:
            mkpts0 = kpts0[matches[:, 0]]  # (M, 2)
            mkpts1 = kpts1[matches[:, 1]]  # (M, 2)
        else:
            mkpts0 = torch.empty((0, 2), device=self.device)
            mkpts1 = torch.empty((0, 2), device=self.device)
            match_scores = torch.empty((0,), device=self.device)
        
        # Convert to numpy
        mkpts0 = to_numpy(mkpts0)
        mkpts1 = to_numpy(mkpts1)
        matches_conf = to_numpy(match_scores)
        
        # Rescale coordinates from resized image to original
        H0, W0 = img0.shape[-2:]
        H1, W1 = img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)
        
        # Create spatial confidence maps (sparse - scatter keypoint scores to grid)
        # These are placeholder zero tensors since SuperPoint produces sparse keypoints
        conf0 = torch.zeros((H0, W0), device=self.device)
        conf1 = torch.zeros((H1, W1), device=self.device)
        
        return mkpts0, mkpts1, matches_conf, desc0, desc1, scores0, scores1, conf0, conf1

    def match_one_to_many(self, src, tgts):
        """
        Match a source image against multiple target images.
        
        Optimized to extract features from source only once.
        
        Args:
            src: Source image path (str/Path) or numpy array (H, W, 3)
            tgts: List of target image paths
            
        Returns:
            correspondences: List of [mkpts_src, mkpts_tgt] pairs
            confidences: List of confidence arrays
            images: List of loaded images as numpy arrays
        """
        # Load and preprocess source image
        image1 = self.load_image(src, resize=self.resize)
        images = [image1.cpu().numpy()]
        
        correspondences = []
        confidences = []
        
        for image_path2 in tgts:
            image2 = self.load_image(image_path2, resize=self.resize)
            images.append(image2.cpu().numpy().copy())
            
            # Use the parent class forward() which calls _forward() and does RANSAC
            result = self(image1, image2)
            
            mkpts1, mkpts2 = result["inliers0"], result["inliers1"]
            correspondences.append([mkpts1, mkpts2])
            confidences.append(result["inliers_conf"])
        
        return correspondences, confidences, images
    
    def matchPair_imgPixelwise_multi(self, qryImg, refImgList):
        """
        Match query image against reference images and return pixel-wise match pairs.
        
        Args:
            qryImg: Query image as numpy array (H, W, 3)
            refImgList: List of reference images
            
        Returns:
            matchPairs: List of arrays with [qryNodeIdx, refNodeIdx] per match
            confidences: List of confidence arrays
        """
        matchPairs = []
        correspondences, images, confidences = self.match_one_to_many(qryImg, refImgList)

        H, W = qryImg.shape[:2]

        for i in range(len(refImgList)):
            mkpts1, mkpts2 = correspondences[i]

            x_i, y_i = mkpts1[:, 0], mkpts1[:, 1]
            x_j, y_j = mkpts2[:, 0], mkpts2[:, 1]
            qryNodesInds = (y_i * W + x_i).astype(int)
            refNodesInds = (y_j * W + x_j).astype(int)

            matchPairs.append(np.column_stack([qryNodesInds, refNodesInds]))
        
        return matchPairs, confidences
