#!/usr/bin/env python3
"""
Evaluate loop closure quality through costmap comparison against navmesh ground truth.

Compares predicted costmaps (noLC, oracleLC, seqvladLC) against navmesh GT costmaps
and computes IoU metrics for top p% lowest cost pixels.

Usage:
    python eval_costmaps_lc.py --scenes-dir data/hm3d-0.2/benchmarking --scene-list episodes_removing_blacklist.txt
"""

import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import pandas as pd
from natsort import natsorted
import cv2


# LC modes to evaluate
LC_MODES = ["noLC", "oracleLC", "seqvladLC"]

# Percentiles for IoU evaluation (top p% lowest cost pixels)
DEFAULT_PERCENTILES = [5, 15, 30, 50]


def load_costmap(path):
    """Load a costmap from file (assuming .npy format)."""
    return np.load(path)


def local_normalize(costmap):
    """Locally normalize costmap to [0, 1] range."""
    min_val = costmap.min()
    max_val = costmap.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(costmap)
    return (costmap - min_val) / (max_val - min_val)


def compute_top_p_iou(pred_costmap, gt_costmap, p=10):
    """
    Compute IoU between predicted and GT costmaps for top p% of pixels
    ordered ascendingly (i.e., lowest cost pixels).
    
    Args:
        pred_costmap: Predicted costmap (H, W)
        gt_costmap: Ground truth costmap (H, W)
        p: Percentage of pixels to consider (e.g., 10 for top 10%)
    
    Returns:
        iou: Intersection over Union for top p% lowest cost pixels
    """
    # Flatten costmaps
    pred_flat = pred_costmap.flatten()
    gt_flat = gt_costmap.flatten()
    
    # Get number of pixels to consider
    n_pixels = len(pred_flat)
    n_top = int(n_pixels * p / 100)
    
    if n_top == 0:
        return 0.0
    
    # Get indices of top p% lowest cost pixels for each costmap
    pred_top_indices = np.argpartition(pred_flat, n_top)[:n_top]
    gt_top_indices = np.argpartition(gt_flat, n_top)[:n_top]
    
    # Convert to sets for IoU computation
    pred_set = set(pred_top_indices)
    gt_set = set(gt_top_indices)
    
    # Compute IoU
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    
    iou = intersection / union if union > 0 else 0.0
    
    return iou


def find_gt_costmap_dir(scene_dir: Path, goal_idx: int) -> Path | None:
    """
    Find the navmesh costmap directory for a given goal.
    
    Tries different naming conventions:
    - navmesh_costmaps_goal077/arrays
    - navmesh_costmaps_goal77/arrays
    """
    # Try 3-digit padded format first
    gt_dir = scene_dir / f"navmesh_costmaps_goal{goal_idx:03d}" / "arrays"
    if gt_dir.exists():
        return gt_dir
    
    # Try non-padded format
    gt_dir = scene_dir / f"navmesh_costmaps_goal{goal_idx}" / "arrays"
    if gt_dir.exists():
        return gt_dir
    
    # Try without /arrays subdirectory (in case they're directly in the folder)
    gt_dir = scene_dir / f"navmesh_costmaps_goal{goal_idx:03d}"
    if gt_dir.exists() and any(gt_dir.glob("*.npy")):
        return gt_dir
    
    gt_dir = scene_dir / f"navmesh_costmaps_goal{goal_idx}"
    if gt_dir.exists() and any(gt_dir.glob("*.npy")):
        return gt_dir
    
    return None


def evaluate_single_goal(
    pred_dir: Path,
    gt_dir: Path,
    percentiles: list
) -> dict:
    """
    Evaluate a single goal's costmaps against GT.
    
    Args:
        pred_dir: Directory containing predicted costmaps (.npy files)
        gt_dir: Directory containing GT costmaps (.npy files)
        percentiles: List of percentiles to evaluate
    
    Returns:
        Dictionary with per-frame IoUs and aggregate statistics
    """
    # Get list of prediction files
    pred_files = natsorted(list(pred_dir.glob("*.npy")))
    
    if len(pred_files) == 0:
        return None
    
    # Store per-frame IoUs for each percentile
    per_frame_ious = {p: [] for p in percentiles}
    
    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        
        if not gt_file.exists():
            continue
        
        # Load costmaps
        pred_costmap = load_costmap(pred_file)
        gt_costmap = load_costmap(gt_file)
        
        # Handle shape mismatch by resizing GT to match prediction
        if pred_costmap.shape != gt_costmap.shape:
            # Resize GT to match prediction shape using bilinear interpolation
            gt_costmap = cv2.resize(
                gt_costmap.astype(np.float32),
                (pred_costmap.shape[1], pred_costmap.shape[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR
            )
        
        # Locally normalize
        pred_norm = local_normalize(pred_costmap)
        gt_norm = local_normalize(gt_costmap)
        
        # Compute IoU for each percentile
        for p in percentiles:
            iou = compute_top_p_iou(pred_norm, gt_norm, p=p)
            per_frame_ious[p].append(iou)
    
    # Compute aggregate statistics
    results = {
        'n_frames': len(per_frame_ious[percentiles[0]]),
        'per_percentile': {}
    }
    
    for p in percentiles:
        values = np.array(per_frame_ious[p])
        if len(values) > 0:
            results['per_percentile'][p] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'values': values.tolist()
            }
        else:
            results['per_percentile'][p] = {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'values': []
            }
    
    return results


def evaluate_scene(
    scene_dir: Path,
    percentiles: list,
    verbose: bool = True
) -> dict:
    """
    Evaluate all goals and LC modes for a single scene.
    
    Args:
        scene_dir: Path to scene directory
        percentiles: List of percentiles to evaluate
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results for all goals and LC modes
    """
    costmaps_npy_dir = scene_dir / "costmaps_npy"
    
    if not costmaps_npy_dir.exists():
        return None
    
    # Find all goal directories
    goal_dirs = natsorted([
        d for d in costmaps_npy_dir.iterdir()
        if d.is_dir() and d.name.startswith("goal")
    ])
    
    if not goal_dirs:
        return None
    
    results = {}
    
    for goal_dir in goal_dirs:
        goal_name = goal_dir.name  # e.g., "goal77"
        goal_idx = int(goal_name.replace("goal", ""))
        
        # Find GT directory
        gt_dir = find_gt_costmap_dir(scene_dir, goal_idx)
        if gt_dir is None:
            if verbose:
                print(f"  ⚠ No GT found for {goal_name}")
            continue
        
        results[goal_idx] = {}
        
        for lc_mode in LC_MODES:
            pred_dir = goal_dir / lc_mode
            
            if not pred_dir.exists():
                if verbose:
                    print(f"  ⚠ Missing {goal_name}/{lc_mode}")
                continue
            
            eval_result = evaluate_single_goal(pred_dir, gt_dir, percentiles)
            
            if eval_result:
                results[goal_idx][lc_mode] = eval_result
                if verbose:
                    n_frames = eval_result['n_frames']
                    mean_5 = eval_result['per_percentile'][percentiles[0]]['mean']
                    print(f"  ✓ {goal_name}/{lc_mode}: {n_frames} frames, "
                          f"top-{percentiles[0]}% mean IoU = {mean_5:.4f}")
    
    return results


def batch_evaluate(
    scenes_base_dir: str,
    scene_list_file: str = None,
    percentiles: list = None,
    output_csv: str = None,
    output_json: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Batch evaluate all scenes and generate results table.
    
    Args:
        scenes_base_dir: Base directory containing scene folders
        scene_list_file: Optional file with list of scene names
        percentiles: List of percentiles to evaluate
        output_csv: Path to save CSV results
        output_json: Path to save JSON results
        verbose: Whether to print progress
    
    Returns:
        DataFrame with all results
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES
    
    base_dir = Path(scenes_base_dir)
    
    # Get list of scenes
    if scene_list_file and Path(scene_list_file).exists():
        with open(scene_list_file, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
        scenes = [base_dir / name for name in scene_names if (base_dir / name).is_dir()]
    else:
        scenes = natsorted([
            p for p in base_dir.iterdir()
            if p.is_dir() and not p.name.endswith('.json')
        ])
    
    print(f"\n{'='*80}")
    print("COSTMAP LC EVALUATION")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Number of scenes: {len(scenes)}")
    print(f"LC modes: {LC_MODES}")
    print(f"Percentiles: {percentiles}")
    print(f"{'='*80}\n")
    
    # Collect all results
    all_results = {}
    rows = []
    
    for scene_dir in tqdm(scenes, desc="Evaluating scenes"):
        scene_name = scene_dir.name
        
        if verbose:
            print(f"\nProcessing {scene_name}")
        
        scene_results = evaluate_scene(scene_dir, percentiles, verbose=verbose)
        
        if scene_results:
            all_results[scene_name] = scene_results
            
            # Build rows for DataFrame
            for goal_idx, goal_data in scene_results.items():
                for lc_mode, lc_data in goal_data.items():
                    row = {
                        'Scene': scene_name,
                        'Goal Image': goal_idx,
                        'LC Mode': lc_mode,
                        'N Frames': lc_data['n_frames']
                    }
                    
                    for p in percentiles:
                        pdata = lc_data['per_percentile'][p]
                        row[f'Min {p}% Mean'] = pdata['mean']
                        row[f'Min {p}% Median'] = pdata['median']
                        row[f'Min {p}% Std'] = pdata['std']
                    
                    rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns
    col_order = ['Scene', 'Goal Image', 'LC Mode', 'N Frames']
    for p in percentiles:
        col_order.extend([f'Min {p}% Mean', f'Min {p}% Median', f'Min {p}% Std'])
    df = df[col_order]
    
    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS TABLE")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    # Compute overall statistics per LC mode
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS BY LC MODE")
    print(f"{'='*80}")
    
    for lc_mode in LC_MODES:
        lc_df = df[df['LC Mode'] == lc_mode]
        if len(lc_df) == 0:
            continue
        
        print(f"\n{lc_mode}:")
        for p in percentiles:
            mean_col = f'Min {p}% Mean'
            mean_of_means = lc_df[mean_col].mean()
            print(f"  Top {p}% - Mean of Means: {mean_of_means:.4f}")
    
    # Save outputs
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved CSV to {output_csv}")
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Saved JSON to {output_json}")
    
    return df


def create_excel_table(
    df: pd.DataFrame,
    output_path: str,
    percentiles: list = None
):
    """
    Create a formatted Excel table similar to the reference image.
    
    Args:
        df: DataFrame with results
        output_path: Path to save Excel file
        percentiles: List of percentiles used
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES
    
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils.dataframe import dataframe_to_rows
        from openpyxl.formatting.rule import ColorScaleRule
    except ImportError:
        print("openpyxl not installed, skipping Excel output")
        return
    
    # Pivot the data to have LC modes as separate column groups
    pivot_rows = []
    
    scenes_goals = df[['Scene', 'Goal Image']].drop_duplicates()
    
    for _, row in scenes_goals.iterrows():
        scene = row['Scene']
        goal = row['Goal Image']
        
        pivot_row = {'Scene': scene, 'Goal Image': goal}
        
        for lc_mode in LC_MODES:
            lc_data = df[(df['Scene'] == scene) & 
                         (df['Goal Image'] == goal) & 
                         (df['LC Mode'] == lc_mode)]
            
            if len(lc_data) == 0:
                continue
            
            lc_row = lc_data.iloc[0]
            for p in percentiles:
                pivot_row[f'{lc_mode}_Min{p}%_Mean'] = lc_row[f'Min {p}% Mean']
                pivot_row[f'{lc_mode}_Min{p}%_Median'] = lc_row[f'Min {p}% Median']
                pivot_row[f'{lc_mode}_Min{p}%_Std'] = lc_row[f'Min {p}% Std']
        
        pivot_rows.append(pivot_row)
    
    pivot_df = pd.DataFrame(pivot_rows)
    pivot_df.to_excel(output_path, index=False)
    print(f"✓ Saved Excel to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate loop closure quality through costmap comparison"
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        required=True,
        help="Base directory containing scene folders"
    )
    parser.add_argument(
        "--scene-list",
        type=str,
        default=None,
        help="Optional file with list of scene names to process"
    )
    parser.add_argument(
        "--percentiles",
        type=int,
        nargs='+',
        default=DEFAULT_PERCENTILES,
        help=f"Percentiles to evaluate (default: {DEFAULT_PERCENTILES})"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="costmap_lc_evaluation.csv",
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output JSON file for detailed results"
    )
    parser.add_argument(
        "--output-excel",
        type=str,
        default=None,
        help="Optional output Excel file for formatted table"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-scene progress output"
    )
    
    args = parser.parse_args()
    
    df = batch_evaluate(
        scenes_base_dir=args.scenes_dir,
        scene_list_file=args.scene_list,
        percentiles=args.percentiles,
        output_csv=args.output_csv,
        output_json=args.output_json,
        verbose=not args.quiet
    )
    
    if args.output_excel:
        create_excel_table(df, args.output_excel, args.percentiles)


if __name__ == "__main__":
    main()
