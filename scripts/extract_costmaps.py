#!/usr/bin/env python3
"""
Extract costmaps from NPZ files and save as separate .npy files indexed by frame ID.

Supports batch processing across multiple scenes with noLC, oracleLC, and seqvladLC variants.

Usage:
    # Single file extraction
    python extract_costmaps.py single <input_npz_path> <output_dir>
    
    # Batch processing for all scenes and goals
    python extract_costmaps.py batch <scenes_base_dir> [--topo-dir topo_map_outputs]
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# LC mode configurations
LC_MODES = {
    "noLC": "",           # No loop closure suffix
    "oracleLC": "_LC_oracle",
    "seqvladLC": "_LC_seqvlad",
}

# Costmap filename pattern (captures LC mode and goal index)
COSTMAP_PATTERN = re.compile(
    r"costmaps_(\d+x\d+)_EC_([A-Z_]+)_NC_([A-Z_]+)_NCF_(\d+)(_LC_[a-z]+)?_goalImg(\d+)\.npz"
)


def save_costmap_as_image(
    costmap: np.ndarray,
    output_path: Path,
    cmap: str = 'turbo',
    dpi: int = 100
):
    """
    Save a costmap as an image with colorbar showing absolute costs.
    
    Args:
        costmap: 2D numpy array of costmap values
        output_path: Path to save the image
        cmap: Matplotlib colormap name
        dpi: Image resolution
    """
    vmin, vmax = costmap.min(), costmap.max()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(costmap, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'Costmap - {output_path.stem}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cost')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def extract_costmaps(
    npz_path: str,
    output_dir: str,
    verbose: bool = True,
    save_images: bool = False,
    cmap: str = 'turbo',
    image_dpi: int = 100
):
    """
    Extract costmaps from NPZ file and save as individual .npy files.
    
    Args:
        npz_path: Path to the input NPZ file containing costmaps
        output_dir: Directory to save individual costmap .npy files
        verbose: Whether to print progress information
        save_images: Whether to also save costmaps as images with colorbar
        cmap: Matplotlib colormap for images (default: turbo)
        image_dpi: DPI for saved images (default: 100)
        
    Returns:
        Number of frames extracted
    """
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NPZ file
    if verbose:
        print(f"Loading costmaps from: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    costmaps = data['costmaps']  # Shape: (N_images, H, W)
    
    # Parse metadata if available
    metadata = None
    if 'metadata' in data:
        try:
            metadata = json.loads(data['metadata'].item())
        except (json.JSONDecodeError, AttributeError):
            metadata = None
    
    n_frames, height, width = costmaps.shape
    
    if verbose:
        print(f"Costmaps shape: {costmaps.shape}")
        print(f"  - Number of frames: {n_frames}")
        print(f"  - Height: {height}")
        print(f"  - Width: {width}")
        if metadata:
            print(f"  - Goal image index: {metadata.get('goal_img_idx', 'N/A')}")
            print(f"  - Goal pixel: {metadata.get('goal_pixel', 'N/A')}")
    
    # Save each frame as a separate .npy file
    if verbose:
        print(f"\nSaving {n_frames} costmap files to: {output_dir}")
    
    # Create images subdirectory if saving images
    if save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  - Saving images as relative costs with colorbar (cmap={cmap}, dpi={image_dpi})")
    
    for frame_idx in range(n_frames):
        frame_costmap = costmaps[frame_idx]  # Shape: (H, W)
        output_path = output_dir / f"{frame_idx:05d}.npy"
        np.save(output_path, frame_costmap)
        
        # Save as image with colorbar (relative costs, per-frame min/max)
        if save_images:
            image_path = images_dir / f"{frame_idx:05d}.png"
            save_costmap_as_image(
                frame_costmap,
                image_path,
                cmap=cmap,
                dpi=image_dpi
            )
    
    # Optionally save metadata as JSON
    if metadata:
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        if verbose:
            print(f"Saved metadata to: {metadata_path}")
    
    if verbose:
        print(f"\n✓ Successfully extracted {n_frames} costmaps to {output_dir}")
    
    return n_frames


def get_goal_indices_from_json(scene_dir: Path) -> list:
    """
    Extract goal image indices from goal_info.json.
    
    Args:
        scene_dir: Path to scene directory
        
    Returns:
        List of goal image indices
    """
    goal_info_path = scene_dir / "goal_info.json"
    
    if not goal_info_path.exists():
        return []
    
    with open(goal_info_path, 'r') as f:
        goal_info = json.load(f)
    
    return [g["goal_image_index"] for g in goal_info]


def get_goal_indices_from_costmaps(topo_dir: Path) -> set:
    """
    Extract goal indices from existing costmap filenames.
    
    Args:
        topo_dir: Path to topo_map_outputs directory
        
    Returns:
        Set of goal image indices found
    """
    goal_indices = set()
    
    for npz_file in topo_dir.glob("costmaps_*.npz"):
        match = COSTMAP_PATTERN.match(npz_file.name)
        if match:
            goal_idx = int(match.group(6))
            goal_indices.add(goal_idx)
    
    return goal_indices


def find_costmap_file(topo_dir: Path, goal_idx: int, lc_mode: str) -> Path | None:
    """
    Find costmap file for a given goal and LC mode.
    
    Args:
        topo_dir: Path to topo_map_outputs directory
        goal_idx: Goal image index
        lc_mode: LC mode key (noLC, oracleLC, seqvladLC)
        
    Returns:
        Path to costmap file or None if not found
    """
    lc_suffix = LC_MODES[lc_mode]
    
    # Search for matching file
    pattern = f"costmaps_*{lc_suffix}_goalImg{goal_idx}.npz"
    matches = list(topo_dir.glob(pattern))
    
    if matches:
        return matches[0]
    
    return None


def process_scene(
    scene_dir: Path,
    topo_subdir: str = "topo_map_outputs",
    verbose: bool = True
) -> dict:
    """
    Process a single scene: extract costmaps for all goals and LC modes.
    
    Args:
        scene_dir: Path to scene directory
        topo_subdir: Subdirectory containing topo map outputs
        verbose: Whether to print progress
        
    Returns:
        Dictionary with processing results
    """
    topo_dir = scene_dir / topo_subdir
    
    if not topo_dir.exists():
        if verbose:
            print(f"⚠ Skipping {scene_dir.name}: {topo_subdir}/ not found")
        return {"scene": scene_dir.name, "status": "skipped", "reason": "no topo dir"}
    
    # Get goal indices (try JSON first, then scan files)
    goal_indices = get_goal_indices_from_json(scene_dir)
    if not goal_indices:
        goal_indices = sorted(get_goal_indices_from_costmaps(topo_dir))
    
    if not goal_indices:
        if verbose:
            print(f"⚠ Skipping {scene_dir.name}: no goals found")
        return {"scene": scene_dir.name, "status": "skipped", "reason": "no goals"}
    
    if verbose:
        print(f"\nProcessing {scene_dir.name}")
        print(f"  Goals: {goal_indices}")
    
    results = {
        "scene": scene_dir.name,
        "status": "processed",
        "goals": {},
    }
    
    # Process each goal and LC mode
    for goal_idx in goal_indices:
        results["goals"][goal_idx] = {}
        
        for lc_mode in LC_MODES.keys():
            costmap_file = find_costmap_file(topo_dir, goal_idx, lc_mode)
            
            if costmap_file is None:
                if verbose:
                    print(f"  ⚠ Missing: goal{goal_idx}/{lc_mode}")
                results["goals"][goal_idx][lc_mode] = "missing"
                continue
            
            # Output directory: scene_dir/costmaps_npy/goal{idx}/{lc_mode}/
            output_dir = scene_dir / "costmaps_npy" / f"goal{goal_idx}" / lc_mode
            
            try:
                n_frames = extract_costmaps(
                    npz_path=str(costmap_file),
                    output_dir=str(output_dir),
                    verbose=False
                )
                results["goals"][goal_idx][lc_mode] = n_frames
                if verbose:
                    print(f"  ✓ goal{goal_idx}/{lc_mode}: {n_frames} frames")
            except Exception as e:
                results["goals"][goal_idx][lc_mode] = f"error: {e}"
                if verbose:
                    print(f"  ✗ goal{goal_idx}/{lc_mode}: {e}")
    
    return results


def batch_process_scenes(
    scenes_base_dir: str,
    topo_subdir: str = "topo_map_outputs",
    scene_list_file: str = None,
    verbose: bool = True
) -> list:
    """
    Process all scenes in a directory.
    
    Args:
        scenes_base_dir: Base directory containing scene folders
        topo_subdir: Subdirectory name for topo map outputs
        scene_list_file: Optional file with list of scene names to process
        verbose: Whether to print progress
        
    Returns:
        List of processing results for each scene
    """
    base_dir = Path(scenes_base_dir)
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Scenes directory not found: {base_dir}")
    
    # Get list of scenes
    if scene_list_file and Path(scene_list_file).exists():
        with open(scene_list_file, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
        scenes = [base_dir / name for name in scene_names if (base_dir / name).is_dir()]
    else:
        # Find all scene directories (exclude config files)
        scenes = sorted([
            p for p in base_dir.iterdir()
            if p.is_dir() and not p.name.endswith('.json')
        ])
    
    if not scenes:
        raise ValueError(f"No scene directories found in {base_dir}")
    
    print(f"\n{'='*80}")
    print(f"BATCH COSTMAP EXTRACTION")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Topo subdirectory: {topo_subdir}")
    print(f"Number of scenes: {len(scenes)}")
    print(f"LC modes: {list(LC_MODES.keys())}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for scene_dir in tqdm(scenes, desc="Processing scenes"):
        result = process_scene(
            scene_dir=scene_dir,
            topo_subdir=topo_subdir,
            verbose=verbose
        )
        all_results.append(result)
    
    # Summary
    processed = sum(1 for r in all_results if r["status"] == "processed")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Processed: {processed}/{len(scenes)} scenes")
    print(f"Skipped: {skipped}/{len(scenes)} scenes")
    print(f"{'='*80}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Extract costmaps from NPZ files and save as .npy files"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single file extraction
    single_parser = subparsers.add_parser(
        "single",
        help="Extract costmaps from a single NPZ file"
    )
    single_parser.add_argument(
        "npz_path",
        type=str,
        help="Path to the input NPZ file containing costmaps"
    )
    single_parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save individual costmap .npy files"
    )
    single_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    single_parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also save costmaps as images with colorbar"
    )
    single_parser.add_argument(
        "--cmap",
        type=str,
        default="turbo",
        help="Matplotlib colormap for images (default: turbo)"
    )
    single_parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for saved images (default: 100)"
    )
    
    # Batch processing
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process all scenes for noLC, oracleLC, seqvladLC"
    )
    batch_parser.add_argument(
        "scenes_base_dir",
        type=str,
        help="Base directory containing scene folders"
    )
    batch_parser.add_argument(
        "--topo-dir",
        type=str,
        default="topo_map_outputs",
        help="Subdirectory containing topo map outputs (default: topo_map_outputs)"
    )
    batch_parser.add_argument(
        "--scene-list",
        type=str,
        default=None,
        help="Optional file with list of scene names to process"
    )
    batch_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-scene progress output"
    )
    
    args = parser.parse_args()
    
    if args.command == "single":
        extract_costmaps(
            npz_path=args.npz_path,
            output_dir=args.output_dir,
            verbose=not args.quiet,
            save_images=args.save_images,
            cmap=args.cmap,
            image_dpi=args.dpi
        )
    elif args.command == "batch":
        batch_process_scenes(
            scenes_base_dir=args.scenes_base_dir,
            topo_subdir=args.topo_dir,
            scene_list_file=args.scene_list,
            verbose=not args.quiet
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
