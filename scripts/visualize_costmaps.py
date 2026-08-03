#!/usr/bin/env python3
"""
Mapping Costmap Visualization Script

Visualizes costmaps from the converted sparse graph format as heatmaps,
optionally overlaid on RGB images, and generates a video.

Usage:
    # Single episode
    python scripts/visualize_costmaps.py --episode_path /path/to/episode_maps
    
    # With custom mapping RGB images directory
    python scripts/visualize_costmaps.py --episode_path /path/to/maps --mapping_img_dir /path/to/images
    
    # Multi-episode mode
    python scripts/visualize_costmaps.py --base_dir /path/to/converted_maps
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
from glob import glob

# =============================================================================
# CONFIGURABLE VARIABLES
# =============================================================================

# Single/Multi mode
SINGLE_EPISODE = True

# For single episode mode
EPISODE_PATH = "/scratch2/public_scratch/toponav/indoor-topo-loc/datasets/hm3d-0.2/benchmarking/CETmJJqkhcK"  # Path to converted maps directory

# For multi-episode mode
BASE_DIR = "data/hm3d-0.2/benchmarking"
# BASE_DIR = "/scratch2/public_scratch/vanshg/mast3r-nav/hm3d_val_sparse_from_dense"
EPISODE_LIST_FILE = "episodes_removing_blacklist.txt"
EPISODE_START_IDX = 0
EPISODE_END_IDX = -1
EPISODE_STEP = 1

# Mapping images (optional, for RGB overlay)
MAPPING_IMG_EPISODE_DIR = "/scratch2/public_scratch/toponav/indoor-topo-loc/datasets/hm3d-0.2/benchmarking/CETmJJqkhcK/images_fov90"
MAPPING_IMAGES_BASE_DIR = "data/hm3d-0.2/benchmarking"

# Visualization settings
COSTMAP_GLOB_PATTERN = "costmaps_320x240_EC_EMST_SINGLE_NC_NONE_NCF_1_NONE_BS_NONE_PTS3D_SUB4_LC_oracle_goalImg65.npz"  # Pattern to find costmap files
VIS_OUTPUT_SUBDIR = "costmap_vis_LC_goal_img65"
VIDEO_FPS = 5
COLORMAP = "turbo"  # turbo, viridis, jet, etc.
OVERLAY_ALPHA = 0.6  # Alpha for overlay on RGB
SHOW_COLORBAR = True
DPI = 150

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def load_costmaps(costmap_path: Path) -> tuple:
    """
    Load costmaps from NPZ file.
    
    Returns:
        costmaps: np.ndarray of shape (N_images, H, W)
        metadata: dict with goal info, config, etc.
    """
    data = np.load(costmap_path, allow_pickle=True)
    costmaps = data['costmaps']
    metadata = json.loads(data['metadata'].item())
    return costmaps, metadata


def create_combined_costmap_visualization(
    costmap: np.ndarray,
    rgb: np.ndarray,
    colormap: str = "turbo",
    overlay_alpha: float = 0.6,
    title: str = None,
    dpi: int = 150
) -> np.ndarray:
    """
    Create a combined 3-panel visualization: RGB | Costmap | Overlayed Costmap.
    
    Args:
        costmap: Single costmap (H, W)
        rgb: RGB image (H, W, 3) - required for this visualization
        colormap: Matplotlib colormap name
        overlay_alpha: Alpha for overlay panel
        title: Optional overall title
        dpi: Output DPI (default 150 to match reference)
        
    Returns:
        Visualization as numpy array (H, W*3+, 3)
    """
    if rgb is None:
        raise ValueError("RGB image is required for combined visualization")
    
    H, W = costmap.shape
    
    # Resize RGB if needed
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))
    
    # Mask invalid values for costmap
    max_valid = 1e5
    heatmap = np.where(costmap >= max_valid, np.nan, costmap)
    valid_costs = heatmap[~np.isnan(heatmap)]
    
    if valid_costs.size > 0:
        vmin, vmax = np.percentile(valid_costs, [5, 95])
        cost_min = float(valid_costs.min())
        cost_max = float(valid_costs.max())
        cost_median = float(np.median(valid_costs))
    else:
        vmin, vmax = 0, 1
        cost_min, cost_max, cost_median = 0, 0, 0
    
    # Create figure with 3 panels - matching reference: figsize=(20, 5.5), dpi=150
    fig = plt.figure(figsize=(20, 5.5), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # Add overall title if provided (matching reference: fontsize=12, fontweight='bold', y=0.98)
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    
    # GridSpec: equal width for all 3 panels (matching reference: wspace=0.08, top=0.93)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.08, top=0.93)
    
    # Get colormap
    cmap_obj = plt.get_cmap(colormap).copy()
    cmap_obj.set_bad(color='white')
    
    # Panel 1: RGB Image (matching reference: fontsize=10)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(rgb)
    ax_rgb.axis('off')
    ax_rgb.set_title("RGB", fontsize=10)
    
    # Panel 2: Costmap Heatmap (matching reference: fontsize=9 for stats title)
    ax_costmap = fig.add_subplot(gs[0, 1])
    im = ax_costmap.imshow(heatmap, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax_costmap.axis('off')
    ax_costmap.set_title(
        f"Costmap\nMin:{cost_min:.1f} Max:{cost_max:.1f} Med:{cost_median:.1f}",
        fontsize=9
    )
    # Add colorbar (matching reference: fraction=0.046, pad=0.02, labelsize=7)
    cbar = plt.colorbar(im, ax=ax_costmap, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    
    # Panel 3: Overlayed Costmap on RGB
    ax_overlay = fig.add_subplot(gs[0, 2])
    ax_overlay.imshow(rgb)
    im_overlay = ax_overlay.imshow(heatmap, cmap=cmap_obj, vmin=vmin, vmax=vmax, alpha=overlay_alpha)
    ax_overlay.axis('off')
    ax_overlay.set_title("Overlay", fontsize=10)
    
    # Tight layout matching reference: pad=1.0, w_pad=2.0, rect=[0, 0, 1, 0.96]
    plt.tight_layout(pad=1.0, w_pad=2.0, rect=[0, 0, 1, 0.96])
    
    # Convert to numpy array
    fig.canvas.draw()
    vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return vis_img


def create_costmap_heatmap(
    costmap: np.ndarray,
    rgb: np.ndarray = None,
    colormap: str = "turbo",
    overlay_alpha: float = 0.6,
    show_colorbar: bool = True,
    title: str = None,
    dpi: int = 100
) -> np.ndarray:
    """
    Create a costmap heatmap visualization.
    
    Args:
        costmap: Single costmap (H, W)
        rgb: Optional RGB image to overlay on (H, W, 3)
        colormap: Matplotlib colormap name
        overlay_alpha: Alpha for overlay
        show_colorbar: Whether to add colorbar
        title: Optional title
        dpi: Output DPI
        
    Returns:
        Visualization as numpy array (H, W, 3)
    """
    H, W = costmap.shape
    
    # Mask invalid values
    max_valid = 1e5  # Threshold for "unreachable"
    heatmap = np.where(costmap >= max_valid, np.nan, costmap)
    valid_costs = heatmap[~np.isnan(heatmap)]
    
    if valid_costs.size > 0:
        vmin, vmax = np.percentile(valid_costs, [2, 98])
    else:
        vmin, vmax = 0, 1
    
    # Create figure
    if show_colorbar:
        fig, ax = plt.subplots(figsize=(W / dpi + 0.8, H / dpi), dpi=dpi)
    else:
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    
    fig.patch.set_facecolor('white')
    
    # Get colormap
    cmap_obj = plt.get_cmap(colormap).copy()
    cmap_obj.set_bad(color='white')
    
    # Overlay on RGB if provided
    if rgb is not None:
        # Resize RGB if needed
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))
        ax.imshow(rgb)
        im = ax.imshow(heatmap, cmap=cmap_obj, vmin=vmin, vmax=vmax, alpha=overlay_alpha)
    else:
        im = ax.imshow(heatmap, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=10, pad=5)
    
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance to Goal', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    plt.tight_layout(pad=0.5)
    
    # Convert to numpy array
    fig.canvas.draw()
    vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return vis_img


def create_video_from_frames(
    frame_dir: Path,
    output_path: Path,
    fps: int = 5,
    pattern: str = "*.png"
) -> bool:
    """
    Create video from image frames.
    
    Returns:
        True if successful, False otherwise
    """
    # Find all frames
    frame_paths = natsorted(glob(str(frame_dir / pattern)))
    
    if not frame_paths:
        print(f"  No frames found with pattern {pattern}")
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"  Could not read first frame")
        return False
    
    H, W = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    for frame_path in tqdm(frame_paths, desc="Creating video", leave=False):
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    
    out.release()
    
    return True


def find_costmap_file(episode_dir: Path, pattern: str = COSTMAP_GLOB_PATTERN) -> Path:
    """Find costmap file in episode directory."""
    map_dir = episode_dir / "topo_map_outputs"
    matches = list(map_dir.glob(pattern))
    if matches:
        return matches[0]
    return None


def get_mapping_images_dir(episode_name: str, mapping_base_dir: Path) -> Path:
    """
    Determine mapping images directory from episode name.
    
    Episode: wcojb4TFT35_0000000_bed_17_
    Mapping: wcojb4TFT35_0000000_bed_17_/images
    """
    mapping_dir = mapping_base_dir / episode_name / "images"
    if mapping_dir.exists():
        return mapping_dir
    return None


def load_mapping_images(mapping_img_dir: Path, num_images: int) -> list:
    """Load RGB images from mapping directory."""
    images = []
    
    if mapping_img_dir is None or not mapping_img_dir.exists():
        print(f"MAPPING IMAGING DIR DOES NOT EXIST")
        return [None] * num_images
    
    # Try common extensions
    for ext in ['.jpg', '.png', '.jpeg']:
        img_paths = natsorted(glob(str(mapping_img_dir / f"*{ext}")))
        if img_paths:
            break
    
    if not img_paths:
        return [None] * num_images
    
    for i in range(num_images):
        if i < len(img_paths):
            img = cv2.imread(img_paths[i])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            images.append(None)
    
    return images


def visualize_single_episode(
    episode_dir: Path,
    mapping_img_dir: Path = None,
    output_subdir: str = VIS_OUTPUT_SUBDIR,
    video_fps: int = VIDEO_FPS,
    colormap: str = COLORMAP,
    overlay_alpha: float = OVERLAY_ALPHA,
    show_colorbar: bool = SHOW_COLORBAR,
    costmap_pattern: str = COSTMAP_GLOB_PATTERN,
    use_combined: bool = False
) -> bool:
    """
    Visualize costmaps for a single episode.
    
    Args:
        episode_dir: Directory containing costmap NPZ file
        mapping_img_dir: Optional directory with RGB images for overlay
        output_subdir: Name of output subdirectory
        video_fps: Video frame rate
        colormap: Matplotlib colormap name
        overlay_alpha: Alpha for RGB overlay
        show_colorbar: Whether to show colorbar
        costmap_pattern: Glob pattern to find costmap files
        use_combined: Use 3-panel combined visualization (RGB | Costmap | Overlay)
    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing: {episode_dir.name}")
    
    # Find costmap file
    costmap_path = find_costmap_file(episode_dir, pattern=costmap_pattern)
    if costmap_path is None:
        print(f"  ✗ No costmap file found")
        return False
    
    print(f"  Costmap: {costmap_path.name}")
    
    # Load costmaps
    try:
        costmaps, metadata = load_costmaps(costmap_path)
    except Exception as e:
        print(f"  ✗ Failed to load costmaps: {e}")
        return False
    
    num_images, H, W = costmaps.shape
    print(f"  Shape: {costmaps.shape} ({num_images} images, {H}x{W})")
    
    # Goal info
    goal_img_idx = metadata.get('goal_img_idx')
    goal_pixel = metadata.get('goal_pixel')
    print(f"  Goal: image {goal_img_idx}, pixel {goal_pixel}")
    
    # Create output directory
    vis_dir = episode_dir / output_subdir
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"{mapping_img_dir = }  | {os.path.exists(str(mapping_img_dir)) = }")
    
    # Load mapping images if available
    if mapping_img_dir and mapping_img_dir.exists():
        print(f"  Loading RGB images from: {mapping_img_dir}")
        rgb_images = load_mapping_images(mapping_img_dir, num_images)
        has_rgb = any(img is not None for img in rgb_images)
    else:
        rgb_images = [None] * num_images
        has_rgb = False
    
    print(f"  RGB overlay: {'Yes' if has_rgb else 'No'}")
    print(f"  Visualization mode: {'Combined 3-panel' if use_combined else 'Single'}")
    
    # Generate visualizations
    print(f"  Generating {num_images} costmap visualizations...")
    
    for i in tqdm(range(num_images), desc="  Rendering", leave=False):
        costmap = costmaps[i]
        rgb = rgb_images[i] if has_rgb else None
        
        # Create title
        title = f"Frame {i:04d}"
        if i == goal_img_idx:
            title += " [GOAL IMAGE]"
        
        # Generate visualization based on mode
        if use_combined and rgb is not None:
            vis_img = create_combined_costmap_visualization(
                costmap=costmap,
                rgb=rgb,
                colormap=colormap,
                overlay_alpha=overlay_alpha,
                title=title
            )
        else:
            vis_img = create_costmap_heatmap(
                costmap=costmap,
                rgb=rgb,
                colormap=colormap,
                overlay_alpha=overlay_alpha,
                show_colorbar=show_colorbar,
                title=title
            )
        
        # Save frame
        frame_path = vis_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    # Create video
    print(f"  Creating video (fps={video_fps})...")
    video_path = vis_dir / "costmap_video.mp4"
    video_success = create_video_from_frames(vis_dir, video_path, fps=video_fps)
    
    if video_success:
        print(f"  ✓ Saved video: {video_path}")
    else:
        print(f"  ✗ Failed to create video")
    
    print(f"  ✓ Saved {num_images} frames to: {vis_dir}")
    return True


def get_episode_list(
    base_dir: Path,
    episode_list_file: str = None,
    start_idx: int = 0,
    end_idx: int = -1,
    step: int = 1
) -> list:
    """
    Get list of episode directories.
    
    Priority:
    1. episode_list_file (if provided and exists)
    2. List all directories in base_dir with start/end/step filtering
    """
    # Try list file first
    if episode_list_file and Path(episode_list_file).exists():
        with open(episode_list_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(names)} episodes from {episode_list_file}")
        
        # Apply range filtering
        if end_idx == -1:
            end_idx = len(names)
        names = names[start_idx:end_idx:step]
        
        # Convert to full paths
        episodes = [base_dir / name for name in names if (base_dir / name).exists()]
        print(f"Found {len(episodes)} existing episode directories")
        return episodes
    
    # Fall back to listing directories
    if not base_dir.exists():
        return []
    
    all_dirs = natsorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    if end_idx == -1:
        end_idx = len(all_dirs)
    
    return all_dirs[start_idx:end_idx:step]


def main():
    parser = argparse.ArgumentParser(description="Visualize mapping costmaps")
    parser.add_argument("--episode_path", type=Path, default=EPISODE_PATH if EPISODE_PATH else None,
                        help="Path to single episode maps directory")
    parser.add_argument("--base_dir", type=Path, default=BASE_DIR,
                        help="Base directory for multi-episode mode")
    parser.add_argument("--mapping_img_dir", type=Path, default=MAPPING_IMG_EPISODE_DIR,
                        help="Directory with RGB images for overlay")
    parser.add_argument("--mapping_base_dir", type=Path, default=MAPPING_IMAGES_BASE_DIR,
                        help="Base directory to find mapping images (looks for episode_name/images)")
    parser.add_argument("--list-file", type=str, default=EPISODE_LIST_FILE,
                        help="Text file with episode names (one per line)")
    parser.add_argument("--start", type=int, default=EPISODE_START_IDX)
    parser.add_argument("--end", type=int, default=EPISODE_END_IDX)
    parser.add_argument("--step", type=int, default=EPISODE_STEP)
    parser.add_argument("--fps", type=int, default=VIDEO_FPS, help="Video FPS")
    parser.add_argument("--colormap", type=str, default=COLORMAP)
    parser.add_argument("--no-colorbar", action="store_true")
    parser.add_argument("--with-overlay", action="store_true", help="Overlay costmap on RGB images")
    parser.add_argument("--costmap-pattern", type=str, default=COSTMAP_GLOB_PATTERN,
                        help="Glob pattern to find costmap files")
    parser.add_argument("--output-subdir", type=str, default=VIS_OUTPUT_SUBDIR,
                        help="Output subdirectory name for visualizations")
    parser.add_argument("--combined", action="store_true",
                        help="Use combined 3-panel visualization (RGB | Costmap | Overlay)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("COSTMAP VISUALIZATION")
    print("="*60)
    
    # Determine mode
    if args.episode_path:
        # Single episode mode
        episode_path = Path(args.episode_path)
        if not episode_path.exists():
            print(f"Error: Episode path does not exist: {episode_path}")
            return 1
        
        # Determine mapping images dir
        if args.mapping_img_dir:
            mapping_img_dir = Path(args.mapping_img_dir)
        elif args.mapping_base_dir:
            mapping_img_dir = get_mapping_images_dir(episode_path.name, Path(args.mapping_base_dir))
        else:
            # Try default base dir
            mapping_img_dir = get_mapping_images_dir(episode_path.name, Path(MAPPING_IMAGES_BASE_DIR))
        
        if not args.with_overlay and not args.combined:
            mapping_img_dir = None
        
        success = visualize_single_episode(
            episode_dir=episode_path,
            mapping_img_dir=mapping_img_dir,
            video_fps=args.fps,
            colormap=args.colormap,
            show_colorbar=not args.no_colorbar,
            costmap_pattern=args.costmap_pattern,
            output_subdir=args.output_subdir,
            use_combined=args.combined
        )
        
        return 0 if success else 1
    
    elif args.base_dir:
        # Multi-episode mode
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            print(f"Error: Base directory does not exist: {base_dir}")
            return 1
        
        list_file = args.list_file if args.list_file else EPISODE_LIST_FILE
        episodes = get_episode_list(base_dir, list_file, args.start, args.end, args.step)
        
        if not episodes:
            print("No episodes found")
            return 1
        
        print(f"Processing {len(episodes)} episodes...")
        
        mapping_base_dir = Path(args.mapping_base_dir) if args.mapping_base_dir else Path(MAPPING_IMAGES_BASE_DIR)
        
        success_count = 0
        fail_count = 0
        
        for episode_dir in episodes:
            # Get mapping images dir for this episode
            if args.with_overlay or args.combined:
                mapping_img_dir = get_mapping_images_dir(episode_dir.name, mapping_base_dir)
            else:
                mapping_img_dir = None
            
            success = visualize_single_episode(
                episode_dir=episode_dir,
                mapping_img_dir=mapping_img_dir,
                video_fps=args.fps,
                colormap=args.colormap,
                show_colorbar=not args.no_colorbar,
                use_combined=args.combined
            )
            
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        print("\n" + "="*60)
        print(f"Summary: {success_count} succeeded, {fail_count} failed")
        print(f"Output: <episode>/{VIS_OUTPUT_SUBDIR}/")
        
        return 0 if fail_count == 0 else 1
    
    else:
        print("Error: Must provide either --episode_path or --base_dir")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
