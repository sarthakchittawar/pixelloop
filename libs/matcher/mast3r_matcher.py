from __future__ import annotations
import sys
from pathlib import Path
import os
import torchvision.transforms as tfm
import py3_wget
import numpy as np
import cv2
import torch
import numpy as np
from PIL import Image
import warnings
from typing import Tuple

from .match_utils import to_numpy, resize_to_divisible
from .match_utils import to_normalized_coords, to_px_coords, to_numpy

sys.path.append(str(Path(__file__).parent.joinpath("mast3r")))
MAST3R_ROOT = Path(__file__).parent.joinpath("mast3r")
MAST3R_WEIGHTS_PATH = MAST3R_ROOT / 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference

class BaseMatcher(torch.nn.Module):
    """
    This serves as a base class for all matchers. It provides a simple interface
    for its sub-classes to implement, namely each matcher must specify its own
    __init__ and _forward methods. It also provides a common image_loader and
    homography estimator
    """

    # OpenCV default ransac params
    DEFAULT_RANSAC_ITERS = 2000
    DEFAULT_RANSAC_CONF = 0.95
    DEFAULT_REPROJ_THRESH = 3

    def __init__(self, device="cpu", **kwargs):
        super().__init__()
        self.device = device

        self.ransac_iters = kwargs.get("ransac_iters", BaseMatcher.DEFAULT_RANSAC_ITERS)
        self.ransac_conf = kwargs.get("ransac_conf", BaseMatcher.DEFAULT_RANSAC_CONF)
        self.ransac_reproj_thresh = kwargs.get("ransac_reproj_thresh", BaseMatcher.DEFAULT_REPROJ_THRESH)
        self.geometric_verification = kwargs.get("geometric_verification", True)

    @staticmethod
    def image_loader(path: str | Path, resize: int | Tuple, rot_angle: float = 0):
        warnings.warn(
            "`image_loader` is replaced by `load_image` and will be removed in a future release.",
            DeprecationWarning,
        )
        return BaseMatcher.load_image(path, resize, rot_angle)

    @staticmethod
    def load_image(path: str | Path, resize: int | Tuple = None, rot_angle: float = 0) -> torch.Tensor:
        if isinstance(resize, int):
            resize = (resize, resize)
        if isinstance(path, (str, Path)):
            img = Image.open(path).convert("RGB")
        else:
            img = path
        img = tfm.ToTensor()(img)
        if resize is not None:
            img = tfm.Resize(resize, antialias=True)(img)
        img = tfm.functional.rotate(img, rot_angle)
        return img

    def rescale_coords(
        self,
        pts: np.ndarray | torch.Tensor,
        h_orig: int,
        w_orig: int,
        h_new: int,
        w_new: int,
    ) -> np.ndarray:
        """Rescale kpts coordinates from one img size to another

        Args:
            pts (np.ndarray | torch.Tensor): (N,2) array of kpts
            h_orig (int): height of original img
            w_orig (int): width of original img
            h_new (int): height of new img
            w_new (int): width of new img

        Returns:
            np.ndarray: (N,2) array of kpts in original img coordinates
        """
        return to_px_coords(to_normalized_coords(pts, h_new, w_new), h_orig, w_orig)

    @staticmethod
    def find_homography(
        points1: np.ndarray | torch.Tensor,
        points2: np.ndarray | torch.Tensor,
        reproj_thresh: int = DEFAULT_REPROJ_THRESH,
        num_iters: int = DEFAULT_RANSAC_ITERS,
        ransac_conf: float = DEFAULT_RANSAC_CONF,
    ):
        assert points1.shape == points2.shape
        assert points1.shape[1] == 2
        points1, points2 = to_numpy(points1), to_numpy(points2)

        H, inliers_mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, reproj_thresh, ransac_conf, num_iters)
        assert inliers_mask.shape[1] == 1
        inliers_mask = inliers_mask[:, 0]
        return H, inliers_mask.astype(bool)

    def process_matches(self, mkpts0: np.ndarray, mkpts1: np.ndarray, mkpts_conf: np.ndarray = None):
        if len(mkpts0) < 4:
            return 0, None, mkpts0, mkpts1, mkpts_conf

        H, inliers_mask = self.find_homography(
            mkpts0,
            mkpts1,
            self.ransac_reproj_thresh,
            self.ransac_iters,
            self.ransac_conf,
        )
        inlier_mkpts0 = mkpts0[inliers_mask]
        inlier_mkpts1 = mkpts1[inliers_mask]
        num_inliers = int(inliers_mask.sum())

        if mkpts_conf is not None:
            inlier_mkpts_conf = mkpts_conf[inliers_mask]
            return num_inliers, H, inlier_mkpts0, inlier_mkpts1, inlier_mkpts_conf

        return num_inliers, H, inlier_mkpts0, inlier_mkpts1, None

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Image preprocessing for each matcher. Some matchers require grayscale, normalization, etc.
        Applied to each input img independently

        Default preprocessing is none

        Args:
            img (torch.Tensor): input image (before preprocessing)

        Returns:
            img (torch.Tensor): img after preprocessing
        """
        return img

    @torch.inference_mode()
    def forward(self, img0: torch.Tensor | str | Path, img1: torch.Tensor | str | Path) -> dict:
        """
        All sub-classes implement the following interface:

        Parameters
        ----------
        img0 : torch.tensor (C x H x W) | str | Path
        img1 : torch.tensor (C x H x W) | str | Path

        Returns
        -------
        dict with keys: ['num_inliers', 'H', 'mkpts0', 'mkpts1', 'inliers0', 'inliers1', 'kpts0', 'kpts1', 'desc0', 'desc1']

        num_inliers : int, number of inliers after RANSAC, i.e. num(inliers0)
        H : np.array (3 x 3), the homography matrix to map mkpts0 to mkpts1
        mkpts0 : np.ndarray (N x 2), keypoints from img0 that match mkpts1 (pre-RANSAC)
        mkpts1 : np.ndarray (N x 2), keypoints from img1 that match mkpts0 (pre-RANSAC)
        inliers0 : np.ndarray (N x 2), filtered mkpts0 that fit the H model (post-RANSAC mkpts)
        inliers1 : np.ndarray (N x 2), filtered mkpts1 that fit the H model (post-RANSAC mkpts)
        inliers_conf : np.ndarray (N,), confidence scores for inlier matches
        desc0 : np.ndarray (N x 2), all descriptors from img0
        desc1 : np.ndarray (N x 2), all descriptors from img1
        desc0_conf : np.ndarray (N,), confidence scores for descriptors from img0
        desc1_conf : np.ndarray (N,), confidence scores for descriptors from img1
        """
        # Take as input a pair of images (not a batch)
        if isinstance(img0, (str, Path)):
            img0 = BaseMatcher.load_image(img0)
        if isinstance(img1, (str, Path)):
            img1 = BaseMatcher.load_image(img1)

        assert isinstance(img0, torch.Tensor)
        assert isinstance(img1, torch.Tensor)

        img0 = img0.to(self.device)
        img1 = img1.to(self.device)

        # self._forward() is implemented by the children modules
        mkpts0, mkpts1, mkpts_conf, desc0, desc1, desc0_conf, desc1_conf, conf0, conf1 = self._forward(img0, img1)

        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        if self.geometric_verification:
            num_inliers, H, inliers0, inliers1, inlier_conf = self.process_matches(mkpts0, mkpts1, mkpts_conf)
        else:
            num_inliers, H, inliers0, inliers1, inlier_conf = len(mkpts0), None, mkpts0, mkpts1, mkpts_conf

        return {
            "num_inliers": num_inliers,
            "H": H,
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "inliers0": inliers0,
            "inliers1": inliers1,
            "inliers_conf": inlier_conf,
            "desc0": to_numpy(desc0),
            "desc1": to_numpy(desc1),
            "desc0_conf": to_numpy(desc0_conf),
            "desc1_conf": to_numpy(desc1_conf),
            "conf0": to_numpy(conf0),
            "conf1": to_numpy(conf1),
        }


class Mast3rMatcher(BaseMatcher):
    model_path = MAST3R_WEIGHTS_PATH
    VIT_PATCH_SIZE = 16

    def __init__(self, resize_w=320, resize_h=240, device='cuda', *args, **kwargs):
        super().__init__(device, **kwargs)

        self.device = device
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.resize = (self.resize_h, self.resize_w)
        self.subsample_or_initxy1 = kwargs.get("subsample_or_initxy1", 8)

        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.verbose = False

        self.download_weights()

        self.model = AsymmetricMASt3R.from_pretrained(self.model_path).to(device)

    @staticmethod
    def download_weights():
        url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

        if not os.path.isfile(Mast3rMatcher.model_path):
            print("Downloading Master(ViT large)... (takes a while)")
            py3_wget.download_file(url, Mast3rMatcher.model_path)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w

        img = resize_to_divisible(img, self.VIT_PATCH_SIZE)

        img = self.normalize(img).unsqueeze(0)

        return img, orig_shape
    
    @staticmethod
    def fast_reciprocal_NNs_with_conf(A, B, confA, confB, subsample_or_initxy1=8, device='cuda', **matcher_kw):
        """
        Fast reciprocal nearest neighbors with confidence scores.
        Single-direction matching (A ↔ B only).
        """
        H1, W1 = A.shape[:2]
        H2, W2 = B.shape[:2]
        
        # Get matches
        xy1, xy2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample_or_initxy1, 
                                        ret_xy=True, device=device, **matcher_kw)
        
        # Compute flat indices from xy coordinates
        # xy1 is (N, 2) array with [x, y]
        idx1 = xy1[:, 1] * W1 + xy1[:, 0]
        idx2 = xy2[:, 1] * W2 + xy2[:, 0]
        
        # Get confidences
        c1 = confA.ravel()[idx1]
        c2 = confB.ravel()[idx2]
        conf = np.minimum(c1, c2)

        if type(conf) is torch.Tensor:
            conf = conf.cpu().numpy()

        return (xy1, xy2, conf)

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        img_pair = [
            {"img": img0, "idx": 0, "instance": 0, "true_shape": np.int32([img0.shape[-2:]])},
            {"img": img1, "idx": 1, "instance": 1, "true_shape": np.int32([img1.shape[-2:]])},
        ]
        output = inference([tuple(img_pair)], self.model, self.device, batch_size=1, verbose=False)
        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1, desc2 = pred1["desc"].squeeze(0).detach(), pred2["desc"].squeeze(0).detach()
        desc1_conf, desc2_conf = pred1["desc_conf"].squeeze(0).detach(), pred2["desc_conf"].squeeze(0).detach()
        conf1, conf2 = pred1["conf"].squeeze(0).detach(), pred2["conf"].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1, matches_conf = Mast3rMatcher.fast_reciprocal_NNs_with_conf(
            desc1, desc2, desc1_conf, desc2_conf, subsample_or_initxy1=self.subsample_or_initxy1, device=self.device, dist="dot", block_size=2**13
        )

        # ignore small border around the edge
        H0, W0 = view1["true_shape"][0]
        valid_matches_im0 = (
            (matches_im0[:, 0] >= 3)
            & (matches_im0[:, 0] < int(W0) - 3)
            & (matches_im0[:, 1] >= 3)
            & (matches_im0[:, 1] < int(H0) - 3)
        )

        H1, W1 = view2["true_shape"][0]
        valid_matches_im1 = (
            (matches_im1[:, 0] >= 3)
            & (matches_im1[:, 0] < int(W1) - 3)
            & (matches_im1[:, 1] >= 3)
            & (matches_im1[:, 1] < int(H1) - 3)
        )

        valid_matches = valid_matches_im0 & valid_matches_im1
        mkpts0, mkpts1 = matches_im0[valid_matches], matches_im1[valid_matches]
        matches_conf = matches_conf[valid_matches]

        # duster sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, matches_conf, desc1, desc2, desc1_conf, desc2_conf, conf1, conf2

    def matchPair_imgPixelwise_multi(self, qryImg, refImgList):
        matchPairs = []
        correspondences, images, confidences = self.match_one_to_many(qryImg, refImgList)

        # # mean confidence based filtering
        # mean_conf = np.mean([np.mean(conf) for conf in confidences])
        # for i in range(len(confidences)):
        #     conf = confidences[i]
        #     mask = conf >= mean_conf
        #     correspondences[i][0] = correspondences[i][0][mask]
        #     correspondences[i][1] = correspondences[i][1][mask]
        #     confidences[i] = conf[mask]

        H, W = qryImg.shape[:2]

        for i in range(len(refImgList)):
            mkpts1, mkpts2 = correspondences[i]

            # TODO : right now I am assumming order of nodes based on pixel coordinates
            x_i, y_i = mkpts1[:, 0], mkpts1[:, 1]
            x_j, y_j = mkpts2[:, 0], mkpts2[:, 1]
            qryNodesInds = (y_i * W + x_i).astype(int) # (N,)
            refNodesInds = (y_j * W + x_j).astype(int) # (N,)

            matchPairs.append(np.column_stack([qryNodesInds, refNodesInds]))
        
        return matchPairs, confidences

    def match_one_to_many(self, src, tgts):
        image1 = self.load_image(src, resize=self.resize)
        images = [image1.cpu().numpy()]
        correspondences = []
        confidences = []
        for i, image_path2 in enumerate(tgts):
            image2 = self.load_image(image_path2, resize=self.resize)
            images.append(image2.cpu().numpy().copy())
            result = self(image1, image2)
            # result.keys() = ["num_inliers", "H", "mkpts0", "mkpts1", "inliers0", "inliers1", "kpts0", "kpts1", "desc0", "desc1"]
            # print(f"match_one_to_many: {result.keys() = }")
            num_inliers, H, mkpts1, mkpts2 = (
                result["num_inliers"],
                result["H"],
                result["inliers0"],
                result["inliers1"],
                # result["inlier_kpts0"],
                # result["inlier_kpts1"],
            )
            correspondences.append([mkpts1, mkpts2])
            confidences.append(result["inliers_conf"])

            # if self.out_dir is not None:
            #     dict_path = self.out_dir / f"output_{i}.torch"
            #     output_dict = {
            #         "num_inliers": num_inliers,
            #         "H": H,
            #         "mkpts0": mkpts1,
            #         "mkpts1": mkpts2,
            #         "img0_path": None,
            #         "img1_path": None,
            #         "matcher": self.matcher_name,
            #         "n_kpts": self.n_kpts,
            #         "im_size": self.resize,
            #         "conf0": results["conf0"],
            #         "conf1": results["conf1"]
            #     }
            #     torch.save(output_dict, dict_path)
        return correspondences, confidences, images
