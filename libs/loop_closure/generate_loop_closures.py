#!/usr/bin/env python3
"""
Sequence-based candidate loop closures with SeqVLAD.

Pipeline
--------
1. Extract SeqVLAD descriptors (sliding window) for every image in a scene.
2. Build a similarity matrix (cosine).
3. Select loop-closure candidates based on similarity threshold.
4. MASt3R pose estimation: FM-RANSAC inlier check + PnP on MASt3R's metric ``pts3d``
   pointmap.  Pairs are kept when ``||tvec|| ≤ max_translation`` (metres) and
   rotation ≤ ``max_rotation`` degrees.
5. UFM covisibility filter: retain pairs where min(fwd, bwd) covisibility ≥ threshold
   (fraction of pixels with covisibility score > 0.9).
6. Deduplicate (NMS) and save.
"""

from __future__ import annotations

import argparse
import collections.abc as container_abcs
import multiprocessing
import os
import ssl
import sys
import threading
import types
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import quaternion
import torch
import torch.nn as nn
import torch.nn.modules.linear as linear
import torchvision.transforms as transforms
from natsort import natsorted
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Compatibility shims for older package code
# ---------------------------------------------------------------------------
if not hasattr(linear, "_LinearWithBias"):

    class _LinearWithBias(nn.Linear):
        pass

    linear._LinearWithBias = _LinearWithBias

if "torch._six" not in sys.modules:
    _torch_six = types.ModuleType("torch._six")
    _torch_six.PY3 = True
    _torch_six.string_classes = (str, bytes)
    _torch_six.int_classes = (int,)
    _torch_six.container_abcs = container_abcs
    sys.modules["torch._six"] = _torch_six

# Optional: ASMK for kernel-based similarity
try:
    import faiss

    try:
        faiss.StandardGpuResources()
    except AttributeError:
        import asmk.index

        class _FaissCpuL2Index(asmk.index.FaissL2Index):
            def __init__(self, gpu_id):
                super().__init__()
                self.gpu_id = gpu_id

            def _faiss_index_flat(self, dim):
                return faiss.IndexFlatL2(dim)

        asmk.index.FaissGpuL2Index = _FaissCpuL2Index

    from asmk import asmk_method

    ASMK_AVAILABLE = True
except ImportError as _asmk_err:
    ASMK_AVAILABLE = False

# Optional: MASt3R matcher for geometric verification (stage 2)
_MAST3R_NAV_PATH = "/home/onyx/work_dirs/sarthak/mast3rnav/mast3r-nav"
try:
    if _MAST3R_NAV_PATH not in sys.path:
        sys.path.append(_MAST3R_NAV_PATH)
    from libs.matcher.mast3r_matcher import Mast3rMatcher  # type: ignore

    MAST3R_AVAILABLE = True
except ImportError:
    MAST3R_AVAILABLE = False

# Optional: UFM for covisibility-based loop-closure filtering
_UFM_PATH = "/home/onyx/work_dirs/sarthak/mast3rnav/mast3r-nav/libs/UFM"
try:
    if _UFM_PATH not in sys.path:
        sys.path.append(_UFM_PATH)
    from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement


    UFM_AVAILABLE = True
except ImportError as _ufm_err:
    UFM_AVAILABLE = False
    print(_ufm_err)

# Local project imports
sys.path.append('/home/onyx/work_dirs/sarthak/mast3rnav/mast3r-nav/libs/loop_closure')
from tvg.models import TVGNet
from tvg.utils import parse_arguments

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = (
    "/home/onyx/work_dirs/sarthak/mast3rnav/mast3r-nav/"
    "checkpoints/msls_cct384_tr8fz1__seqvlad_seq5.pth"
)
DEFAULT_BASE_DIR = (
    "/scratch2/public_scratch/sarthak/datasets/mast3nav/benchmarking/"
)
DEFAULT_SAVE_FILE = "seqvlad_loops_ufm.txt"

IMAGE_META = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# Global CUDNN benchmark for faster conv operations
torch.backends.cudnn.benchmark = True


# ===================================================================
# UFM pose-estimation helpers
# ===================================================================


def load_ufm_model(device: torch.device) -> nn.Module:
    """
    Load a UFM model on ``device``.

    torch.compile is intentionally NOT used — on RTX A4000 it consumes
    ~12 GB of VRAM for CUDA graphs while giving negligible speedup.
    Without it we can use larger batch sizes and multiple GPUs.
    """
    if not UFM_AVAILABLE:
        raise ImportError("UFM is not available. Add the UFM repo to sys.path.")
    ufm = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine-336")
    import types as _types
    from uniflowmatch.models import ufm as _ufm_module
    ufm.forward = _types.MethodType(_ufm_module.UniFlowMatchClassificationRefinement.forward, ufm)

    ufm.eval()
    ufm = ufm.to(device)
    print(f"UFM model loaded on {device}")
    return ufm


def load_ufm_models_multi_gpu(
    gpu_ids: Optional[List[int]] = None,
) -> List[nn.Module]:
    """
    Load one UFM model per GPU for parallel inference.

    Parameters
    ----------
    gpu_ids : list of CUDA device indices.  ``None`` → use all GPUs.

    Returns
    -------
    List of UFM models, one per GPU.
    """
    if not UFM_AVAILABLE:
        print("UFM not available — skipping multi-GPU load")
        return []

    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    models: List[nn.Module] = []
    for gid in gpu_ids:
        dev = torch.device(f"cuda:{gid}")
        models.append(load_ufm_model(dev))

    print(f"Loaded {len(models)} UFM model(s) on GPU(s) {gpu_ids}")
    return models


# ===================================================================
# Batched UFM covisibility
# ===================================================================


def batch_ufm_covisibility(
    ufm_models: Union[nn.Module, List[nn.Module]],
    pairs: List[Tuple[int, int]],
    img_paths: List[str],
    batch_size: int = 48,
    resize: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Compute UFM covisibility scores for a list of image pairs.

    No depth maps or pose estimation are performed.  For every pair the
    UFM covisibility head predicts a per-pixel visibility mask; the score
    is the fraction of pixels marked covisible (> 0.5).

    Supports multi-GPU parallelism when *ufm_models* is a list.

    Parameters
    ----------
    ufm_models : single model or list of models (one per GPU)
    pairs : list of (img_i, img_j) index tuples
    img_paths : image file paths indexed by image number
    batch_size : pairs per GPU batch
    resize : (W, H) to pre-resize images on CPU, or None for full-res

    Returns
    -------
    dict mapping (i, j) -> covisibility_score for every input pair
    """
    if len(pairs) == 0:
        return {}

    if isinstance(ufm_models, nn.Module):
        ufm_models = [ufm_models]

    n_gpus = len(ufm_models)
    devices = [next(m.parameters()).device for m in ufm_models]
    n_pairs = len(pairs)

    # Treat pairs as edges with dummy score for compatibility
    edges = [(i, j, 0.0) for i, j in pairs]

    # --- Pre-load unique images -----------------------------------------------
    unique_indices: set[int] = {i for i, j in pairs} | {j for i, j in pairs}
    resize_tag = f", resized to {resize[0]}×{resize[1]}" if resize else ""
    print(f"Pre-loading {len(unique_indices)} unique images{resize_tag} …")

    img_tensors: dict[int, torch.Tensor] = {}
    for idx in tqdm(sorted(unique_indices), desc="Loading images"):
        img = cv2.imread(img_paths[idx])
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize is not None:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
        img_tensors[idx] = torch.from_numpy(img)

    shapes = {tuple(t.shape) for t in img_tensors.values()}
    if len(shapes) > 1:
        print(f"WARNING: mixed image shapes {shapes} — falling back to batch_size=1")
        batch_size = 1

    covis_arr = np.zeros(n_pairs, dtype=np.float32)

    def _run_chunk(gpu_idx: int, chunk_start: int, chunk_end: int) -> None:
        model = ufm_models[gpu_idx]
        device = devices[gpu_idx]
        chunk = edges[chunk_start:chunk_end]
        n = len(chunk)
        if n == 0:
            return

        xfer_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )

        def _make_batch(s, e):
            b = chunk[s:e]
            src = torch.stack([img_tensors[i] for i, *_ in b]).pin_memory()
            tgt = torch.stack([img_tensors[j] for _, j, *_ in b]).pin_memory()
            return src, tgt

        b0_end = min(batch_size, n)
        next_src, next_tgt = _make_batch(0, b0_end)

        for start in tqdm(
            range(0, n, batch_size),
            desc=f"UFM-covis gpu{device.index if hasattr(device, 'index') else 0}",
        ):
            end = min(start + batch_size, n)
            cur_src, cur_tgt = next_src, next_tgt

            next_start = start + batch_size
            if next_start < n:
                next_end = min(next_start + batch_size, n)
                next_src, next_tgt = _make_batch(next_start, next_end)
                if xfer_stream is not None:
                    with torch.cuda.stream(xfer_stream):
                        next_src = next_src.to(device, non_blocking=True)
                        next_tgt = next_tgt.to(device, non_blocking=True)

            cur_src = cur_src.to(device, non_blocking=True)
            cur_tgt = cur_tgt.to(device, non_blocking=True)

            with torch.inference_mode():
                result = model.predict_correspondences_batched(
                    source_image=cur_src,
                    target_image=cur_tgt,
                )

            covis_masks = result.covisibility.mask.cpu().numpy()  # (B, H, W)
            for bi in range(end - start):
                covis_arr[chunk_start + start + bi] = float(
                    np.mean(covis_masks[bi] > 0.9)
                )

            if xfer_stream is not None and next_start < n:
                xfer_stream.synchronize()

    print(
        f"Running UFM covisibility on {n_pairs} pairs "
        f"(batch_size={batch_size}, {n_gpus} GPU(s)) …"
    )
    chunk_size = (n_pairs + n_gpus - 1) // n_gpus
    if n_gpus == 1:
        _run_chunk(0, 0, n_pairs)
    else:
        with ThreadPoolExecutor(max_workers=n_gpus) as ex:
            futs = [
                ex.submit(
                    _run_chunk,
                    gi,
                    gi * chunk_size,
                    min((gi + 1) * chunk_size, n_pairs),
                )
                for gi in range(n_gpus)
            ]
            for f in futs:
                f.result()

    return {(i, j): float(covis_arr[k]) for k, (i, j) in enumerate(pairs)}


# ===================================================================
# Model loading
# ===================================================================


def load_model(
    checkpoint_path: str,
    device: torch.device,
    arch: str = "cct384",
    seq_length: int = 5,
    trunc_te: int = 8,
    freeze_te: int = 1,
) -> Tuple[TVGNet, argparse.Namespace]:
    """Load a trained SeqVLAD model from a checkpoint."""
    ssl._create_default_https_context = ssl._create_unverified_context

    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    args = parse_arguments()
    sys.argv = original_argv

    args.arch = arch
    args.aggregation = "seqvlad"
    args.pooling = "none"
    args.seq_length = seq_length
    args.img_shape = [120, 160]
    args.features_dim = 256
    args.device = device
    args.trunc_te = trunc_te
    args.freeze_te = freeze_te

    model = TVGNet(args)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = OrderedDict(
        {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded  —  arch={arch}  trunc_te={trunc_te}  "
          f"aggregation=seqvlad  seq_length={seq_length}")
    return model, args


# ===================================================================
# Image / descriptor helpers
# ===================================================================


def prepare_image_sequence(
    image_paths: List[str],
    img_size: Tuple[int, int] = (384, 384),
) -> torch.Tensor:
    """Load and transform a sequence of images into a stacked tensor."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_META["mean"], std=IMAGE_META["std"]),
    ])
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors)


def extract_descriptor(
    model: nn.Module,
    images_batch: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run forward pass and return the descriptor as a numpy array."""
    images_batch = images_batch.to(device)
    with torch.no_grad():
        descriptor = model(images_batch)
    return descriptor.cpu().numpy()


# ===================================================================
# ASMK similarity
# ===================================================================


def compute_similarity_matrix_asmk(
    descriptors_matrix: np.ndarray,
    asmk_params: Optional[dict] = None,
) -> np.ndarray:
    """Compute an image-to-image similarity matrix with ASMK."""
    if not ASMK_AVAILABLE:
        raise ImportError("ASMK is not available. Install faiss and asmk.")

    n_images = len(descriptors_matrix)

    # Heuristic codebook size
    if n_images < 128:
        cb = 64
    elif n_images < 256:
        cb = 128
    elif n_images < 512:
        cb = 256
    elif n_images < 1024:
        cb = 512
    elif n_images < 2048:
        cb = "1k"
    elif n_images < 4096:
        cb = "2k"
    elif n_images < 8192:
        cb = "4k"
    else:
        cb = "8k"

    print(f"ASMK: {n_images} images, codebook size {cb}")

    if asmk_params is None:
        asmk_params = {
            "index": {"gpu_id": 0},
            "train_codebook": {"codebook": {"size": cb}},
            "build_ivf": {
                "kernel": {"binary": True},
                "ivf": {"use_idf": False},
                "quantize": {"multiple_assignment": 1},
                "aggregate": {},
            },
            "query_ivf": {
                "quantize": {"multiple_assignment": 5},
                "aggregate": {},
                "search": {"topk": None},
                "similarity": {"similarity_threshold": 0.0, "alpha": 3.0},
            },
        }

    feat = descriptors_matrix
    ids = np.arange(n_images, dtype=np.int32)

    asmk = asmk_method.ASMKMethod.initialize_untrained(asmk_params)
    asmk = asmk.train_codebook(feat)
    asmk_dataset = asmk.build_ivf(feat, ids)
    _metadata, _query_ids, ranks, ranked_scores = asmk_dataset.query_ivf(feat, ids)

    scores = np.empty_like(ranked_scores)
    scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
    return scores


# ===================================================================
# Core loop-closure detection
# ===================================================================


def _top_k_symmetric(
    sim_matrix: np.ndarray,
    row_idx: int,
    k: int = 1,
    m: int = 3,
    exclude_range: int = 50,
) -> List[int]:
    """
    Find the *k* most similar images for ``row_idx``, excluding its
    neighbourhood and applying a symmetry check over the top-*m* candidates.
    """
    sim_values = sim_matrix[row_idx].copy()
    sim_values[row_idx] = -np.inf

    lo = max(0, row_idx - exclude_range)
    hi = min(len(sim_values), row_idx + exclude_range + 1)
    sim_values[lo:hi] = -np.inf

    if m == -1:
        top_m = np.argsort(sim_values)[::-1]
    else:
        top_m = np.argsort(sim_values)[-m:][::-1]

    top_m = [i for i in top_m if sim_values[i] >= 0.0]

    symmetry_info = []
    for idx in top_m:
        diff = abs(sim_matrix[row_idx, idx] - sim_matrix[idx, row_idx])
        symmetry_info.append((idx, diff))
    symmetry_info.sort(key=lambda x: x[1])

    if k == -1:
        selected = symmetry_info
    else:
        selected = symmetry_info[:k]

    return sorted(
        [x[0] for x in selected],
        key=lambda i: sim_matrix[row_idx, i],
        reverse=True,
    )


def get_seqvlad_loop_closures(
    img_dir: str,
    model: nn.Module,
    device: torch.device,
    *,
    window_size: int = 5,
    k: int = -1,
    m: int = -1,
    exclude_range: int = 50,
    similarity_method: str = "cosine",
    sim_threshold: float = 0.4,
) -> Tuple[List[Tuple[int, int]], np.ndarray, List[int]]:
    """
    Detect loop-closure candidate pairs using SeqVLAD descriptors.

    Returns
    -------
    (loop_closure_pairs, sim_matrix, valid_indices)
    """
    img_names = natsorted(os.listdir(img_dir))
    img_paths = [os.path.join(img_dir, n) for n in img_names]
    n_images = len(img_paths)
    half_window = window_size // 2
    valid_indices = list(range(n_images))

    print(f"Found {n_images} images in {img_dir}")

    # --- 1. Extract descriptors -----------------------------------------------
    descriptors = []
    for center in tqdm(valid_indices, desc="Extracting seqVLAD descriptors"):
        win_indices = [
            max(0, min(n_images - 1, center + off))
            for off in range(-half_window, half_window + 1)
        ]
        win_paths = [img_paths[i] for i in win_indices]
        images = prepare_image_sequence(win_paths, img_size=(384, 384))
        desc = extract_descriptor(model, images, device)
        descriptors.append(desc.squeeze())

    descriptors_matrix = np.vstack(descriptors)
    print(f"Descriptors matrix: {descriptors_matrix.shape}")

    # --- 2. Similarity matrix -------------------------------------------------
    if similarity_method == "cosine":
        print("Computing cosine similarity …")
        sim_matrix = cosine_similarity(descriptors_matrix)
    elif similarity_method == "asmk":
        sim_matrix = compute_similarity_matrix_asmk(descriptors_matrix)
    else:
        raise ValueError(f"Unknown similarity method: {similarity_method}")

    print(f"Similarity range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")

    # --- 3. Candidate selection ------------------------------------------------
    edges: List[Tuple[int, int, float]] = []
    existing = set()
    for mi in tqdm(range(n_images), desc="Finding loop closures"):
        img_i = valid_indices[mi]
        for mj in _top_k_symmetric(sim_matrix, mi, k=k, m=m, exclude_range=exclude_range):
            img_j = valid_indices[mj]
            if sim_matrix[mi, mj] < sim_threshold:
                continue
            if (img_j, img_i) not in existing:
                edges.append((img_i, img_j, float(sim_matrix[mi, mj])))
                existing.add((img_i, img_j))

    print(f"Found {len(edges)} loop-closure candidates ({similarity_method})")

    lc_pairs = [(i, j) for i, j, _ in edges]
    lc_pairs.sort()
    return lc_pairs, sim_matrix, valid_indices


# ===================================================================
# Stage 2 – MASt3R pose filtering + geometric verification
# ===================================================================


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Rotation angle in degrees from a 3×3 rotation matrix."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _load_image_cv2(
    path: str,
    resize: Optional[Tuple[int, int]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Read an image with OpenCV and return a ``(3, H, W)`` float tensor."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if resize is not None:
        img = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32") / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return torch.from_numpy(img).permute(2, 0, 1).to(device)


def compute_mast3r_pose_and_inliers(
    img1_path: str,
    img2_path: str,
    matcher: "Mast3rMatcher",
    device: str = "cuda",
    max_depth: float = 10.0,
) -> dict:
    """
    Run MASt3R inference to get:
      1. 2-D correspondences + FM-RANSAC inlier check.
      2. Relative pose via PnP using MASt3R's ``pts3d`` pointmap as 3-D world
         reference — no external depth files or explicit intrinsics required.

    DUSt3R/MASt3R output layout
    ---------------------------
    ``pred1["pts3d"]``               (H1,W1,3) — metric 3-D world points for
                                     cam1 pixels, expressed in frame 1
                                     (cam1 = identity / origin).
    ``pred2["pts3d_in_other_view"]``  (H2,W2,3) — metric 3-D world points for
                                     cam2 pixels, ALSO expressed in frame 1.
    ``pred2["pts3d"]``               NOT present.

    Why PnP
    -------
    Both ``pred1["pts3d"]`` and ``pred2["pts3d_in_other_view"]`` are expressed
    in the same frame 1, so a point-cloud alignment (e.g. Procrustes) between
    them yields a near-identity rotation.  The correct approach is PnP:
    use ``pred1["pts3d"]`` as the 3-D world reference and img2 pixel
    coordinates as 2-D observations; cam1 is the origin so the resulting
    (R, t) is directly the relative pose.

    ``f ≈ max(H2, W2)`` is DUSt3R's internal focal-length heuristic.
    Because DUSt3R predicts **metric** depth, ``||tvec||`` is in metres.

    Translation outputs
    -------------------
    ``translation_m``    — ``||tvec||`` in metres (primary pipeline filter).
    ``translation_ratio`` — ``||tvec|| / median_z_scene`` (dimensionless; kept
                            for legacy comparison only, not used for filtering).

    Returns
    -------
    dict with keys:
        ``inlier_ratio``    — FM-RANSAC inlier fraction.
        ``n_inliers``       — absolute FM-RANSAC inlier count.
        ``translation_m``   — ``||tvec||`` in metres (``np.inf`` on failure).
        ``translation_ratio`` — dimensionless ratio (``np.inf`` on failure).
        ``rotation_deg``    — rotation angle in degrees (``np.inf`` on failure).
    """
    from dust3r.inference import inference  # local import — only in worker processes

    _fail = {
        "inlier_ratio": 0.0,
        "n_inliers": 0,
        "translation_ratio": np.inf,
        "translation_m": np.inf,
        "rotation_deg": np.inf,
    }

    # --- Run MASt3R inference (keeps pts3d which _forward discards) ----------
    img1 = _load_image_cv2(img1_path, resize=(matcher.resize_h, matcher.resize_w), device=device)
    img2 = _load_image_cv2(img2_path, resize=(matcher.resize_h, matcher.resize_w), device=device)

    img1_pre, img1_orig_shape = matcher.preprocess(img1)
    img2_pre, img2_orig_shape = matcher.preprocess(img2)

    img_pair = [
        {"img": img1_pre, "idx": 0, "instance": 0, "true_shape": np.int32([img1_pre.shape[-2:]])},
        {"img": img2_pre, "idx": 1, "instance": 1, "true_shape": np.int32([img2_pre.shape[-2:]])},
    ]
    output = inference([tuple(img_pair)], matcher.model, device, batch_size=1, verbose=False)
    pred1, pred2 = output["pred1"], output["pred2"]
    view1, view2 = output["view1"], output["view2"]

    desc1 = pred1["desc"].squeeze(0).detach()
    desc2 = pred2["desc"].squeeze(0).detach()
    desc1_conf = pred1["desc_conf"].squeeze(0).detach()
    desc2_conf = pred2["desc_conf"].squeeze(0).detach()

    # DUSt3R convention: pred1["pts3d"] — (H1,W1,3) metric world points for
    # cam1 pixels in frame 1 (cam1 = origin).  pred2 has no "pts3d" key; only
    # "pts3d_in_other_view" (cam2 pixels, also in frame 1).
    # PnP uses pred1["pts3d"] as 3-D world reference and img2 pixel coords as
    # 2-D observations, recovering cam2's pose relative to cam1.
    pts3d_world = pred1["pts3d"].squeeze(0).detach().cpu().numpy()  # (H1, W1, 3) in frame 1

    # 2-D matches in ViT-preprocessed space
    matches_im0, matches_im1, _ = matcher.fast_reciprocal_NNs_with_conf(
        desc1, desc2, desc1_conf, desc2_conf,
        subsample_or_initxy1=matcher.subsample_or_initxy1,
        device=device, dist="dot", block_size=2**13,
    )

    # Border mask (same as _forward)
    H0, W0 = view1["true_shape"][0]
    H1, W1 = view2["true_shape"][0]
    valid_0 = (
        (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) &
        (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
    )
    valid_1 = (
        (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) &
        (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
    )
    valid = valid_0 & valid_1
    matches_im0 = matches_im0[valid]  # (N, 2) in preprocessed px space
    matches_im1 = matches_im1[valid]

    if len(matches_im0) < 8:
        return _fail

    # Rescale to original image space for FM-RANSAC
    pts0 = matcher.rescale_coords(matches_im0, *img1_orig_shape, *img1_pre.shape[-2:])
    pts1 = matcher.rescale_coords(matches_im1, *img2_orig_shape, *img2_pre.shape[-2:])

    # --- FM-RANSAC inlier check ----------------------------------------------
    F, mask = cv2.findFundamentalMat(
        pts0.astype(np.float32), pts1.astype(np.float32),
        cv2.FM_RANSAC, ransacReprojThreshold=1.0,
    )
    if mask is None:
        return _fail
    fm_mask = mask.ravel().astype(bool)
    n_inliers = int(fm_mask.sum())
    inlier_ratio = n_inliers / len(pts0)

    # --- 3-D pose via PnP on MASt3R pts3d -----------------------------------
    translation_ratio = np.inf
    translation_m = np.inf
    rotation_deg = np.inf
    try:
        # 3-D world points: pred1["pts3d"] at img1 FM-inlier match positions
        u0 = np.clip(matches_im0[fm_mask, 0], 0, pts3d_world.shape[1] - 1).astype(int)
        v0 = np.clip(matches_im0[fm_mask, 1], 0, pts3d_world.shape[0] - 1).astype(int)
        pts_3d = pts3d_world[v0, u0]  # (M, 3) in frame 1

        # 2-D observations: img2 pixel coords (original resolution)
        pts_2d = pts1[fm_mask].astype(np.float32)  # already rescaled to img2_orig_shape

        # Depth filter
        depth_valid = (pts_3d[:, 2] > 0) & (pts_3d[:, 2] < max_depth)
        pts_3d = pts_3d[depth_valid]
        pts_2d = pts_2d[depth_valid]

        if len(pts_3d) >= 6:
            # Intrinsics for cam2: DUSt3R assumes f ~ max(H, W), principal point = centre
            H2_orig, W2_orig = img2_orig_shape
            H2_pre, W2_pre = img2_pre.shape[-2:]
            f2 = float(max(H2_orig, W2_orig))
            cx2, cy2 = W2_orig / 2.0, H2_orig / 2.0
            K2 = np.array([[f2, 0.0, cx2], [0.0, f2, cy2], [0.0, 0.0, 1.0]], dtype=np.float64)

            median_z_scene = float(np.median(pts_3d[:, 2]))
            if median_z_scene > 0:
                ret, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                    pts_3d.astype(np.float64),
                    pts_2d.astype(np.float64),
                    K2, None,
                    reprojectionError=8.0,
                    confidence=0.99,
                    iterationsCount=1000,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                if ret and rvec is not None:
                    R2, _ = cv2.Rodrigues(rvec)
                    rotation_deg = _rotation_angle_deg(R2)
                    translation_m = float(np.linalg.norm(tvec))
                    translation_ratio = translation_m / median_z_scene
    except Exception as _e:
        print(f"  [pose] {Path(img1_path).name} ↔ {Path(img2_path).name}: {_e}")

    return {
        "inlier_ratio": inlier_ratio,
        "n_inliers": n_inliers,
        "translation_ratio": translation_ratio,
        "translation_m": translation_m,
        "rotation_deg": rotation_deg,
    }


# ---------------------------------------------------------------------------
# Persistent worker pool for MASt3R — one process = one loaded model
# ---------------------------------------------------------------------------

_worker_matcher: Optional["Mast3rMatcher"] = None  # set by _worker_init in each spawned process
_worker_device: str = "cuda:0"                      # set by _worker_init


def _worker_init(device_queue) -> None:
    """
    Pool initializer: called once per spawned process.
    Pops a device string from *device_queue* and loads MASt3R into the
    module-level ``_worker_matcher`` global.  Because the queue has exactly
    one entry per worker slot, every worker gets its own assigned device.
    """
    global _worker_matcher, _worker_device
    _worker_device = device_queue.get()
    _worker_matcher = Mast3rMatcher(device=_worker_device)
    print(f"  [worker pid={os.getpid()}] MASt3R loaded on {_worker_device}")


def _mast3r_worker_fn(args: tuple) -> List[dict]:
    """
    Top-level worker function — must be module-level so multiprocessing
    'spawn' can pickle it.  Uses the pre-loaded ``_worker_matcher`` global
    (set once by ``_worker_init``) so the model is never reloaded between scenes.
    Sends a sentinel (1) into *progress_queue* after every completed pair.
    """
    (
        pairs_chunk, scene_dir_str,
        img_subdir, img_ext,
        progress_queue,
    ) = args
    results = []
    for idx1, idx2 in pairs_chunk:
        img1 = os.path.join(scene_dir_str, img_subdir, f"{idx1:05d}.{img_ext}")
        img2 = os.path.join(scene_dir_str, img_subdir, f"{idx2:05d}.{img_ext}")
        info = compute_mast3r_pose_and_inliers(
            img1, img2, _worker_matcher, device=_worker_device,
        )
        results.append({
            "pair": (idx1, idx2),
            "inliers": info["n_inliers"],
            "inlier_ratio": info["inlier_ratio"],
            "translation_ratio": info["translation_ratio"],
            "translation_m": info["translation_m"],
            "rotation_deg": info["rotation_deg"],
        })
        progress_queue.put(1)
    return results


def parallel_mast3r_pose_filter(
    pairs: List[Tuple[int, int]],
    scene_dir: Path,
    pool,                          # persistent multiprocessing.Pool
    manager,                       # persistent multiprocessing.Manager
    n_workers: int,
    max_translation: float = 3.0,
    max_rotation: float = 90.0,
    img_subdir: str = "images_fov90",
    img_ext: str = "jpg",
    desc: str = "MASt3R pose filter",
) -> List[dict]:
    """
    Run MASt3R matching + 3-D pose estimation on every pair using a
    **persistent** worker pool.  The pool is created once in ``main()``
    and reused for every scene, so MASt3R models are loaded only once per
    worker process regardless of how many scenes are processed.

    Pairs are kept when both ``||tvec||`` (metres) ≤ *max_translation* and
    rotation ≤ *max_rotation*.  The translation bound is a generous sanity
    cap (default 3 m) that rejects degenerate PnP solutions while passing
    all plausible nearby-view pairs.

    Parameters
    ----------
    pairs           : (img_i, img_j) candidate pairs.
    scene_dir       : root directory of the scene.
    pool            : a ``multiprocessing.Pool`` whose workers were
                      initialised with ``_worker_init`` (MASt3R pre-loaded).
    manager         : the ``multiprocessing.Manager`` used to create queues.
    n_workers       : number of worker processes in *pool*.
    max_translation : metres threshold on PnP ``||tvec||`` (default 3.0 m).
    max_rotation    : degrees threshold (default 90.0).
    img_subdir      : sub-directory that holds RGB images.
    img_ext         : image file extension (default ``jpg``).
    desc            : label used in progress messages.

    Returns
    -------
    List of dicts (one per *kept* pair) with keys:
        ``pair``, ``inliers``, ``inlier_ratio``,
        ``translation_ratio``, ``translation_m``, ``rotation_deg``.
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        return []

    # Round-robin distribution across n_workers
    chunks = [pairs[i::n_workers] for i in range(n_workers)]

    progress_queue = manager.Queue()

    job_args = [
        (
            chunks[i], str(scene_dir),
            img_subdir, img_ext,
            progress_queue,
        )
        for i in range(n_workers)
        if chunks[i]  # skip empty chunks for small pair counts
    ]
    n_active = len(job_args)
    print(f"{desc}: {n_pairs} pairs across {n_active} worker processes …")

    # Drive a per-pair tqdm bar from the main process by reading the queue
    # in a background thread while workers run.
    pbar = tqdm(total=n_pairs, desc=desc, unit="pair")

    def _progress_reader():
        completed = 0
        while completed < n_pairs:
            try:
                progress_queue.get(timeout=2.0)
                pbar.update(1)
                completed += 1
            except Exception:
                pass  # timeout — loop and check again

    reader = threading.Thread(target=_progress_reader, daemon=True)
    reader.start()

    chunks_out = pool.map(_mast3r_worker_fn, job_args)

    reader.join()
    pbar.close()

    all_results = [r for chunk in chunks_out for r in chunk]

    kept = [
        r for r in all_results
        if r["translation_m"] <= max_translation and r["rotation_deg"] <= max_rotation
    ]
    n_pruned = n_pairs - len(kept)
    print(
        f"MASt3R pose filter (trans_m<={max_translation}m, rot<={max_rotation}°): "
        f"{len(kept)} kept, {n_pruned} pruned"
    )
    return kept


def deduplicate_pairs(
    pairs_with_scores: List[dict],
    window: int = 3,
) -> List[Tuple[int, int]]:
    """Keep only the best pair per spatial window.

    Pairs are ranked by (covisibility DESC, inlier_ratio DESC, inliers DESC)
    so within each window the pair with the highest UFM covisibility score
    is retained.
    """
    sorted_pairs = sorted(
        pairs_with_scores,
        key=lambda r: (-r.get("covisibility", 0.0), -r["inlier_ratio"], -r["inliers"]),
    )
    kept: List[Tuple[int, int]] = []
    for r in sorted_pairs:
        # inlier check is not needed
        # if r["inlier_ratio"] <= 0.7 or r["inliers"] <= 100:
        #     continue
        a, b = r["pair"]
        if any(abs(a - ka) <= window and abs(b - kb) <= window for ka, kb in kept):
            continue
        kept.append((a, b))
    return sorted(kept)


# ===================================================================
# Main entry point
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SeqVLAD + MASt3R + UFM loop-closure pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                        help="Root directory containing per-scene subdirectories.")
    parser.add_argument("--save-file", default=DEFAULT_SAVE_FILE,
                        help="Output filename written inside each scene directory.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to the SeqVLAD model checkpoint.")
    # SeqVLAD
    parser.add_argument("--window-size", type=int, default=5,
                        help="Sliding-window size for SeqVLAD descriptor extraction.")
    parser.add_argument("--exclude-range", type=int, default=3,
                        help="Frames within this range of the query are excluded as candidates.")
    parser.add_argument("--similarity-method", default="cosine",
                        choices=["cosine", "asmk"],
                        help="Similarity metric for candidate selection.")
    parser.add_argument("--sim-threshold", type=float, default=0.4,
                        help="Minimum similarity score for a pair to be a candidate.")
    parser.add_argument("-k", type=int, default=-1,
                        help="Top-k matches per query (-1 = all above threshold).")
    parser.add_argument("-m", type=int, default=-1,
                        help="Top-m candidates checked for symmetry (-1 = all).")
    # MASt3R pose filter
    parser.add_argument("--matchers-per-gpu", type=int, default=8,
                        help="Persistent MASt3R worker processes per GPU.")
    parser.add_argument("--max-translation", type=float, default=3.0,
                        help="Max PnP ||tvec|| in metres to keep a pair (sanity cap; default 3.0 m).")
    parser.add_argument("--max-rotation", type=float, default=90.0,
                        help="Max relative rotation in degrees to keep a pair.")
    # UFM covisibility
    parser.add_argument("--covis-threshold", type=float, default=0.1,
                        help="Minimum min(fwd, bwd) UFM covisibility score.")
    parser.add_argument("--ufm-batch-size", type=int, default=16,
                        help="Batch size for UFM covisibility inference.")
    # NMS
    parser.add_argument("--nms-window", type=int, default=3,
                        help="Spatial window for pair deduplication (NMS).")

    script_args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_dir = script_args.base_dir
    save_file = script_args.save_file
    print(f"Processing scenes in {base_dir} …")

    # --- Load models ----------------------------------------------------------
    model, args = load_model(script_args.checkpoint, device)

    # MASt3R — create a persistent pool so workers load the model once and
    # reuse it for every scene.  Uses a Manager Queue to assign each worker
    # its own CUDA device on startup via _worker_init.
    if not MAST3R_AVAILABLE:
        raise RuntimeError(
            "MASt3R is not available. Check that libs/matcher/mast3r_matcher.py "
            "is importable and all dependencies are installed."
        )
    MATCHERS_PER_GPU = script_args.matchers_per_gpu
    n_gpus = torch.cuda.device_count()
    gpu_ids = list(range(n_gpus)) if n_gpus > 0 else [None]
    matcher_devices = [
        f"cuda:{gpu_ids[i % n_gpus]}" if n_gpus > 0 else "cpu"
        for i in range(len(gpu_ids) * MATCHERS_PER_GPU)
    ]
    n_matchers = len(matcher_devices)
    print(
        f"Starting {n_matchers} persistent MASt3R worker process(es) "
        f"({MATCHERS_PER_GPU} per GPU, GPU(s): {gpu_ids}) — loading models once …"
    )

    _ctx = multiprocessing.get_context("spawn")
    _manager = _ctx.Manager()
    _device_queue = _manager.Queue()
    for d in matcher_devices:
        _device_queue.put(d)

    _pool = _ctx.Pool(
        processes=n_matchers,
        initializer=_worker_init,
        initargs=(_device_queue,),
    )
    print(f"  {n_matchers} MASt3R worker processes ready.")

    # UFM — for covisibility scoring only
    ufm_models: List[nn.Module] = []
    if UFM_AVAILABLE:
        ufm_models = load_ufm_models_multi_gpu(
            list(range(torch.cuda.device_count())) if torch.cuda.device_count() > 0 else None
        )
    else:
        print("WARNING: UFM not available — covisibility filter will be skipped.")

    COVIS_THRESHOLD = script_args.covis_threshold

    benchmark_root = Path(base_dir)
    try:
        for scene_dir in sorted(d for d in benchmark_root.iterdir() if d.is_dir()):
            print(f"\n{'='*60}\nScene: {scene_dir.name}\n{'='*60}")
            img_dir = str(scene_dir / "images_fov90")
            if not os.path.isdir(img_dir):
                print(f"  Skipping — no images_fov90 directory found.")
                continue

            # --- Stage 1: SeqVLAD candidate selection ------------------------
            candidates, sim_matrix, _ = get_seqvlad_loop_closures(
                img_dir=img_dir,
                model=model,
                device=device,
                window_size=script_args.window_size,
                k=script_args.k,
                m=script_args.m,
                exclude_range=script_args.exclude_range,
                similarity_method=script_args.similarity_method,
                sim_threshold=script_args.sim_threshold,
            )
            print(f"Stage 1 — SeqVLAD candidates: {len(candidates)}")

            # --- Stage 2: MASt3R pose filter ---------------------------------
            filtered = parallel_mast3r_pose_filter(
                candidates,
                scene_dir,
                pool=_pool,
                manager=_manager,
                n_workers=n_matchers,
                max_translation=script_args.max_translation,
                max_rotation=script_args.max_rotation,
                desc=f"MASt3R {scene_dir.name}",
            )
            print(
                f"Stage 2 — MASt3R pose filter: {len(filtered)} kept "
                f"(from {len(candidates)})"
            )

            # --- Stage 3: UFM covisibility filter ----------------------------
            # Free any cached allocations from MASt3R before UFM runs in the
            # main process — workers hold their model weights but we can still
            # reclaim fragmented/reserved-but-unallocated blocks.
            torch.cuda.empty_cache()
            if ufm_models:
                filtered_pairs = [r["pair"] for r in filtered]
                img_paths = sorted(
                    str(p) for p in (scene_dir / "images_fov90").glob("*.jpg")
                )
                # Run UFM both ways (i→j and j→i) in a single batched call,
                # then use min(forward, backward) as the conservative score.
                both_directions = list(
                    dict.fromkeys(
                        filtered_pairs + [(j, i) for i, j in filtered_pairs]
                    )
                )
                covis_dict = batch_ufm_covisibility(
                    ufm_models,
                    both_directions,
                    img_paths,
                    batch_size=script_args.ufm_batch_size,
                )
                def _min_covis(pair):
                    i, j = pair
                    fwd = covis_dict.get((i, j), 0.0)
                    bwd = covis_dict.get((j, i), 0.0)
                    return min(fwd, bwd)

                pre_covis = len(filtered)
                filtered = [
                    {**r, "covisibility": _min_covis(r["pair"])}
                    for r in filtered
                    if _min_covis(r["pair"]) >= COVIS_THRESHOLD
                ]
                print(
                    f"Stage 3 — UFM covisibility min(fwd,bwd) (>={COVIS_THRESHOLD}): "
                    f"{len(filtered)} kept, {pre_covis - len(filtered)} pruned"
                )
            else:
                print("Stage 3 — UFM not available, skipping covisibility filter.")
                filtered = [{**r, "covisibility": 0.0} for r in filtered]

            # --- Stage 4: Deduplicate and save -------------------------------
            pairs_to_save = deduplicate_pairs(filtered, window=script_args.nms_window)
            out_path = scene_dir / save_file
            with open(out_path, "w") as f:
                for idx1, idx2 in pairs_to_save:
                    f.write(f"{idx1} {idx2}\n")
            print(
                f"Stage 4 — Saved {len(pairs_to_save)} pairs "
                f"(from {len(filtered)} pre-dedup) -> {out_path}"
            )
    finally:
        print("Shutting down MASt3R worker pool …")
        _pool.close()
        _pool.join()
        _manager.shutdown()
        print("Pool shut down.")


if __name__ == "__main__":
    main()