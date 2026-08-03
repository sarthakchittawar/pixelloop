"""
MASt3R Inference Module

Provides a unified interface for MASt3R model operations including:
- Model loading
- 3D point cloud inference (self-matching)
- Image pair matching
- Depth conversion utilities
"""

import sys
import copy
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import torch
import torchvision.transforms as tfm
from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)

# Add MASt3R to path
BASE_DIR = Path(__file__).parent.parent.parent
MAST3R_PATH = BASE_DIR / "libs" / "matcher" / "mast3r"
if str(MAST3R_PATH) not in sys.path:
    sys.path.insert(0, str(MAST3R_PATH))

# Default model path (inside the mast3r repo)
DEFAULT_MODEL_PATH = MAST3R_PATH / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


class MASt3RInference:
    """
    Unified interface for MASt3R model operations.
    
    Provides simple methods for:
    - get_pts3d(): Get 3D point cloud from single image (self-matching)
    - get_matches(): Get 2D-2D matches between two images
    - infer(): Raw inference on image pair
    
    Example:
        mast3r = MASt3RInference(device="cuda")
        pts3d = mast3r.get_pts3d(rgb_image)
        matches0, matches1 = mast3r.get_matches(img0, img1)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", initialized_model: Optional[torch.nn.Module] = None):
        """
        Initialize MASt3R model.
        
        Args:
            model_path: Path to model weights. Uses default if None.
            device: Compute device ("cuda" or "cpu")
            initialized_model: Optional pre-initialized model instance
        """
        from mast3r.model import AsymmetricMASt3R
        
        self.device = device
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if initialized_model is not None:
            logger.info("Using provided initialized MASt3R model")
            self.model = initialized_model.to(device)

        else:
            logger.info(f"Loading MASt3R model from {self.model_path}")
            self.model = AsymmetricMASt3R.from_pretrained(str(self.model_path)).to(device)
            logger.info("MASt3R model loaded successfully")
        
        # Image normalization
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    def _load_image(self, img: Union[np.ndarray, str, Path], 
                    resize: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Load and preprocess a single image."""
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        img_tensor = tfm.ToTensor()(img)
        if resize is not None:
            img_tensor = tfm.Resize(resize, antialias=True)(img_tensor)
        return img_tensor
    
    def _preprocess(self, img: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Normalize and prepare image for inference."""
        _, h, w = img.shape
        orig_shape = (h, w)
        img = self.normalize(img).unsqueeze(0)
        return img, orig_shape
    
    def _prepare_images(self, img0, img1, 
                        resize: Optional[Tuple[int, int]] = None) -> list:
        """Prepare image pair for MASt3R inference."""
        img0_tensor = self._load_image(img0, resize)
        img1_tensor = self._load_image(img1, resize)
        
        img0_tensor, _ = self._preprocess(img0_tensor)
        img1_tensor, _ = self._preprocess(img1_tensor)
        
        images = [
            {"img": img0_tensor.to(self.device), "idx": 0, "instance": 0, 
             "true_shape": np.int32([img0_tensor.shape[-2:]])},
            {"img": img1_tensor.to(self.device), "idx": 1, "instance": 1, 
             "true_shape": np.int32([img1_tensor.shape[-2:]])}
        ]
        return images
    
    def infer(self, img0, img1, resize: Optional[Tuple[int, int]] = None) -> dict:
        """
        Run MASt3R inference on image pair.
        
        Args:
            img0: First image (np.ndarray, path, or Path)
            img1: Second image (np.ndarray, path, or Path)
            resize: Optional resize dimensions (H, W)
            
        Returns:
            dict: Raw MASt3R output containing view1, view2, pred1, pred2
        """
        from dust3r.inference import inference
        
        images = self._prepare_images(img0, img1, resize)
        
        with torch.no_grad():
            output = inference([tuple(images)], self.model, self.device, 
                             batch_size=1, verbose=False)
        return output
    
    def get_pts3d(self, img, resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get 3D point cloud from single image using self-matching.
        
        Args:
            img: Input image (np.ndarray, path, or Path)
            resize: Optional resize dimensions (H, W)
            
        Returns:
            pts3d: (H, W, 3) numpy array of 3D points
        """
        # Self-matching: use same image for both inputs
        output = self.infer(img, copy.deepcopy(img) if isinstance(img, np.ndarray) else img, resize)
        pts3d = output["pred1"]["pts3d"][0].cpu().numpy()
        return pts3d
    
    def get_matches(self, img0, img1, 
                    resize: Optional[Tuple[int, int]] = None,
                    subsample: int = 8,
                    dist: str = "dot",
                    block_size: int = 2**13,
                    match_feature_type: str = "descriptor",
                    border: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D-2D matches between two images.
        
        Args:
            img0: First image
            img1: Second image
            resize: Optional resize dimensions (H, W)
            subsample: Subsampling factor for fast_reciprocal_NNs
            border: Border pixels to ignore
            
        Returns:
            matches_im0: (N, 2) array of match coordinates in img0 [x, y]
            matches_im1: (N, 2) array of match coordinates in img1 [x, y]
        """
        from mast3r.fast_nn import fast_reciprocal_NNs
        
        output = self.infer(img0, img1, resize)
        
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]
        
        desc1 = pred1["desc"].squeeze(0).detach()
        desc2 = pred2["desc"].squeeze(0).detach()

        pts3d_1 = pred1['pts3d'].squeeze(0).detach()
        pts3d_2 = pred2['pts3d_in_other_view'].squeeze(0).detach()

        if match_feature_type == 'pointmap':
            # (H, W, 3)
            feat1, feat2 = pts3d_1, pts3d_2
        else:
            # (H, W, 24)
            feat1, feat2 = desc1, desc2

        if dist is None:
            # use the KD Tree implementation
            matches_im0, matches_im1 = fast_reciprocal_NNs(
                feat1, feat2,
                subsample_or_initxy1=subsample,
            )
        else:
            # Find 2D-2D matches
            matches_im0, matches_im1 = fast_reciprocal_NNs(
                feat1, feat2,
                subsample_or_initxy1=subsample,
                device=self.device,
                dist=dist,
                block_size=block_size,
            )
        
        # Filter border matches
        H0, W0 = view1["true_shape"][0]
        H1, W1 = view2["true_shape"][0]
        
        valid = (
            (matches_im0[:, 0] >= border) & (matches_im0[:, 0] < int(W0) - border) &
            (matches_im0[:, 1] >= border) & (matches_im0[:, 1] < int(H0) - border) &
            (matches_im1[:, 0] >= border) & (matches_im1[:, 0] < int(W1) - border) &
            (matches_im1[:, 1] >= border) & (matches_im1[:, 1] < int(H1) - border)
        )
        
        return matches_im0[valid], matches_im1[valid]
    
    @staticmethod
    def pts3d_to_depth(pts3d: np.ndarray) -> np.ndarray:
        """
        Convert pts3d to depth map using Z channel.
        
        Args:
            pts3d: (H, W, 3) point cloud
            
        Returns:
            depth: (H, W) depth map
        """
        Z = pts3d[..., 2]
        return np.where(Z > 0, Z, 0.0).astype(np.float32)

