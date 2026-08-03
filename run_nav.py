"""
run_nav.py - Navigation Test Script for mast3r-nav

A clean, Hydra-based script with comprehensive timing, metrics, and visualization.
"""

import os
import sys
import time
import logging
import random
import json
import re
from pathlib import Path
from typing import Optional, Tuple
import signal
import atexit

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
import cv2
import csv

from libs.matcher.mast3r_matcher import Mast3rMatcher
from libs.matcher.superpoint_matcher import SuperPointMatcher

# Seed everything for reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything()

# Suppress habitat-sim logs
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

from libs.experiments import task_setup
from libs.logger import default_logger
from libs.mast3r_utils import MASt3RInference
from libs.experiments.episode_utils import check_if_stuck, find_shortest_path
from libs.visualizations import VisualizationDataCollector, VisualizationRenderer
from libs.timing_manager import timing_manager, time_function
import habitat_sim

# Setup logging
default_logger.setup_logging(level=logging.INFO, console=True)
logger = logging.getLogger("[RunNav]")

# Global variable for timing data on exit
GLOBAL_RESULTS_PATH = None

# ==============================================================================
# Exit Handlers for Timing Data
# ==============================================================================

def save_timing_on_exit():
    """Save comprehensive timing data when program exits"""
    if GLOBAL_RESULTS_PATH is not None and timing_manager.enabled:
        try:
            print("\nSaving comprehensive timing analysis before exit...")
            timing_manager.print_all_function_summaries()
            timing_manager.save_function_timings_csv(GLOBAL_RESULTS_PATH)
            timing_manager.create_hierarchical_report(GLOBAL_RESULTS_PATH)
            timing_manager.save_step_timing_breakdown(GLOBAL_RESULTS_PATH)
            print(f"Comprehensive timing data saved to:")
            print(f"  CSV: {GLOBAL_RESULTS_PATH}/function_timing_analysis.csv")
            print(f"  Hierarchical: {GLOBAL_RESULTS_PATH}/hierarchical_timing_analysis.txt")
            print(f"  Step Breakdown: {GLOBAL_RESULTS_PATH}/step_timing_breakdown.txt")
        except Exception as e:
            print(f"Error saving timing data: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals"""
    print(f"\nReceived signal {signum}. Saving timing data before exit...")
    save_timing_on_exit()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(save_timing_on_exit)

# ==============================================================================
# Utility Functions
# ==============================================================================

def split_observations(observations: dict) -> tuple:
    """
    Extract RGB, depth, and semantic from simulator observations.
    
    Returns:
        rgb: (H, W, 3) uint8 array
        depth: (H, W) float32 array
        semantic: (H, W) int32 array or None
    """
    rgb = observations["color_sensor"][:, :, :3]  # Drop alpha channel
    depth = observations["depth_sensor"]
    semantic = observations.get("semantic_sensor", None)
    return rgb, depth, semantic

def compute_episode_metrics(episode, results_csv_path: Path) -> dict:
    """
    Compute comprehensive metrics for a completed episode.
    
    Args:
        episode: Episode instance
        results_csv_path: Path to results.csv file
        
    Returns:
        dict with metrics: spl, soft_spl, path_length, collisions, etc.
    """
    metrics = {
        'spl': 0.0,
        'soft_spl': 0.0,
        'path_length': 0.0,
        'avg_collisions': 0.0,
        'shortest_path_length': episode.distance_to_final_goal,
        'remain_distance': episode.distance_to_goal,
        'success': False
    }
    
    # Compute Soft SPL
    if metrics['shortest_path_length'] > 0:
        metrics['soft_spl'] = max(0, (1 - metrics['remain_distance'] / metrics['shortest_path_length']))
    
    # Compute SPL and path metrics from CSV if available
    if results_csv_path.exists():
        try:
            results_csv = results_csv_path.read_text().splitlines()
            
            # Extract positions for path length
            x_pos = [float(r.split(',')[1]) for r in results_csv[1:]]
            z_pos = [float(r.split(',')[3]) for r in results_csv[1:]]
            xz = np.array(list(zip(x_pos, z_pos)))
            metrics['path_length'] = np.linalg.norm(xz[1:] - xz[:-1], axis=1).sum() + metrics['remain_distance']
            
            # Compute SPL
            if metrics['path_length'] > 0:
                metrics['spl'] = metrics['shortest_path_length'] / max(metrics['shortest_path_length'], metrics['path_length'])
                metrics['soft_spl'] *= metrics['spl']
            
            # Compute average collisions
            collisions = [int(r.split(',')[-1]) for r in results_csv[1:]]
            metrics['avg_collisions'] = np.mean(collisions) if collisions else 0.0
        except Exception as e:
            logger.warning(f"Error computing metrics from CSV: {e}")
    
    # Adjust for success
    if episode.success_status is None or 'success' in episode.success_status.lower():
        metrics['soft_spl'] = 1.0  # Full credit for successful navigation
        metrics['success'] = True
    else:
        metrics['spl'] = 0.0
    
    return metrics

def save_metrics_to_csv(metrics: dict, csv_path: Path, success_status: str):
    """Save episode metrics to CSV file"""
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['metric', 'value'])
        csvwriter.writerow(['spl', f"{metrics['spl']:.4f}"])
        csvwriter.writerow(['soft_spl', f"{metrics['soft_spl']:.4f}"])
        csvwriter.writerow(['avg_collisions', f"{metrics['avg_collisions']:.2f}"])
        csvwriter.writerow(['path_length', f"{metrics['path_length']:.4f}"])
        csvwriter.writerow(['shortest_path_length', f"{metrics['shortest_path_length']:.4f}"])
        csvwriter.writerow(['remain_distance', f"{metrics['remain_distance']:.4f}"])
        csvwriter.writerow(['success_status', success_status if success_status else 'success'])

def quat_to_euler(w, x, y, z):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

# ==============================================================================
# Episode Runner
# ==============================================================================

def run_episode(cfg: DictConfig, episode_path: Path, episode_results_path: Path, 
                mast3r_model=None, iteration_idx: int = 0, start_idx: int = -1, initialized_matcher=None) -> dict:
    """
    Run a single navigation episode.
    
    Args:
        cfg: Hydra config
        episode_path: Path to episode data
        episode_results_path: Path to save results
        mast3r_model: Optional pre-loaded MASt3R model
        iteration_idx: Iteration number (for multiple random starts)
        start_idx: Start state index (-1 for random)
    
    Returns:
        dict with episode results (success_status, steps, distance_to_goal, metrics)
    """
    # Construct scene path.
    # Supports episode folders like "3UDjdrwcqMb.basis_20260401_..." where
    # the true scene id may be in either the folder name prefix or parent folder.
    hm3d_root_path = Path(cfg.hm3d_root_path)
    episode_dir_name = episode_path.name
    scene_id_candidates = [
        episode_path.parent.name,
        episode_dir_name.split(".")[0],
        episode_dir_name.split("_")[0].split(".")[0],
    ]
    # Preserve order, drop duplicates/empties.
    scene_id_candidates = list(dict.fromkeys([s for s in scene_id_candidates if s]))

    glb_candidates = []
    for scene_id in scene_id_candidates:
        glb_candidates.extend(sorted(hm3d_root_path.rglob(f"*{scene_id}*basis.glb")))

    if len(glb_candidates) == 0:
        raise ValueError(
            "Could not find scene GLB under hm3d_root_path. "
            f"hm3d_root_path={hm3d_root_path}, episode_path={episode_path}, "
            f"scene_id_candidates={scene_id_candidates}"
        )

    scene_glb_path = str(glb_candidates[0])

    logger.info(f"Scene GLB PATH: {scene_glb_path} | {os.path.exists(scene_glb_path) = }")
    
    # Create episode runner
    logger.info(f"Initializing episode: {episode_path} (iteration {iteration_idx})")
    
    # Pre-load models if needed
    preload_data = {}
    if hasattr(cfg, 'preload_models') and cfg.preload_models:
        # This would contain controller, segmentor, etc.
        pass
    
    episode = task_setup.Episode(
        cfg=cfg,
        episode_path=episode_path,
        scene_glb_path=scene_glb_path,
        episode_results_path=episode_results_path,
        preload_data=preload_data,
        start_idx=start_idx,
        initialized_matcher=initialized_matcher
    )
    
    # Setup logging directories
    episode.setup_logging()
    
    # Initialize visualization system if enabled
    data_collector = None
    vis_renderer = None
    if cfg.visualization.save_raw_data.enabled:
        start_pos = episode.agent.get_state().position
        goal_pos = episode.final_goal_position
        
        data_collector = VisualizationDataCollector(
            episode_results_path=episode_results_path,
            cfg=cfg,
            start_position=start_pos,
            goal_position=goal_pos
        )
        
        vis_renderer = VisualizationRenderer(
            vis_cfg=cfg.visualization,
            episode_results_path=episode_results_path
        )
        
        # Save topdown base map
        meters_per_pixel = getattr(cfg.visualization.render_visualizations, 'topdown_meters_per_pixel', 0.025)
        data_collector.save_topdown_data(
            sim=episode.sim,
            start_position=np.array(start_pos),
            goal_position=np.array(goal_pos),
            meters_per_pixel=meters_per_pixel
        )
        
        logger.info("Visualization system initialized")
    
    # Navigation loop parameters
    step = 0
    max_steps = cfg.max_steps
    stuck_threshold = cfg.get("stuck_threshold", 0.01)
    stuck_window = cfg.get("stuck_window", 15)
    pts3d_source = cfg.get("pts3d_source", "gt_depth")
    
    # Collision avoidance parameters
    collision_avoidance_mode = False
    collision_recovery = cfg.get("collision_recovery", False)
    recovery_cooldown = 0  # Steps to skip stuck detection after recovery
    
    # Track last known vis data so recovery steps can reuse them
    last_matches_data = None
    last_waypoints = None
    last_vis_data = None
    
    logger.info(f"Starting navigation loop (max_steps={max_steps}, pts3d_source={pts3d_source})")
    
    pbar = tqdm(total=max_steps, desc=f"Navigation (iter {iteration_idx})", leave=False)
    
    while step < max_steps:
        timing_manager.start_timer("navigation_step")
        step_start = time.time()
        
        # Check for stuck condition (skip if recently recovered)
        if recovery_cooldown > 0:
            recovery_cooldown -= 1
        elif check_if_stuck(episode.agent_state_history, stuck_threshold, stuck_window):
            logger.warning(f"Episode failed: Agent stuck (little movement in last {stuck_window} steps)")
            episode.success_status = "stuck_no_movement"
            collision_avoidance_mode = True  # Try recovery
        
        # Check if goal reached
        if episode.is_done():
            logger.info(f"Episode succeeded at step {step}!")
            break
        
        # === 1. PERCEPTION ===
        observations = episode.sim.get_sensor_observations()
        rgb, depth_gt, semantic = split_observations(observations)
        robot_pose = episode.agent.get_state()
        
        # === 2. 3D RECONSTRUCTION ===
        if pts3d_source == "mast3r" and mast3r_model is not None:
            timing_manager.start_timer("mast3r_inference")
            pts3d = mast3r_model.get_pts3d(rgb)
            depth = None
            timing_manager.end_timer("mast3r_inference")
            logger.debug(f"Step {step}: Got pts3d from MASt3R, shape={pts3d.shape}")
        else:
            # Use GT depth
            pts3d = None
            depth = depth_gt
        
        # === 3. COLLISION RECOVERY (if needed) ===
        if collision_avoidance_mode and collision_recovery:
            logger.info("Attempting collision recovery by rotating...")

            if True:
                # ── Real-world recovery ────────────────────────────────────────────
                # Use the topdown depth map (same projection as CARE) to decide
                # whether to turn left or right: turn away from the side with more
                # nearby obstacles, then rotate exactly 45° in place.
                logger.info("Real-world collision recovery: using depth-based direction selection...")
                from libs.collision_avoidance.care.care import ConstructTopDownObstacleMap, DEFAULT_PARAMS as CARE_PARAMS

                recovery_depth = depth if depth is not None else depth_gt

                # Default: turn right if no depth is available
                direction = -1  # +1 = left, -1 = right

                if recovery_depth is not None:
                    try:
                        intrinsics = {
                            "fx": episode.agent_intrinsics[0, 0].item(),
                            "fy": episode.agent_intrinsics[1, 1].item(),
                            "cx": episode.agent_intrinsics[0, 2].item(),
                            "cy": episode.agent_intrinsics[1, 2].item(),
                        }
                        obstacles = ConstructTopDownObstacleMap(recovery_depth, intrinsics, CARE_PARAMS)
                        if obstacles is not None and len(obstacles) > 0:
                            # obstacles[:, 1] is y_left in robot frame:
                            #   positive y_left → obstacle is on the LEFT side
                            #   negative y_left → obstacle is on the RIGHT side
                            n_left  = int((obstacles[:, 1] >  0).sum())
                            n_right = int((obstacles[:, 1] <  0).sum())
                            logger.info(
                                f"Topdown depth: {n_left} left obstacles, {n_right} right obstacles"
                            )
                            # Turn toward the side with fewer obstacles (more free space)
                            direction = 1 if n_left <= n_right else -1
                        else:
                            logger.warning("No obstacles in topdown depth map, defaulting to right turn")
                    except Exception as e:
                        logger.warning(f"Depth-based direction selection failed: {e}, defaulting to right turn")
                else:
                    logger.warning("No depth available for recovery direction, defaulting to right turn")

                logger.info(f"Recovery direction: {'left' if direction > 0 else 'right'}")

                # Rotate exactly 45° (π/4 rad) at 0.2 rad per step → ~8 steps, capped at 50
                target_angle = np.radians(45)
                angle_accumulated = 0.0
                step_theta = 0.2  # rad per action step
                max_recovery_steps = 50

                for _ in range(max_recovery_steps):
                    episode.velocity_control = 0.0
                    episode.theta_control = step_theta * direction
                    episode.execute_action()

                    angle_accumulated += step_theta

                    # Save visualization data during recovery
                    if cfg.visualization.save_raw_data.enabled and data_collector is not None:
                        agent_state_obj = episode.agent.get_state()
                        agent_state = {
                            'position': np.array(agent_state_obj.position),
                            'rotation': np.array([
                                agent_state_obj.rotation.w, agent_state_obj.rotation.x,
                                agent_state_obj.rotation.y, agent_state_obj.rotation.z
                            ])
                        }
                        data_collector.save_step_data(
                            step=step,
                            rgb=rgb,
                            depth=recovery_depth,
                            pts3d=None,
                            costmap=getattr(episode, 'goal_mask', None),
                            matches_data=last_matches_data,
                            velocity=episode.velocity_control,
                            theta=episode.theta_control,
                            waypoints=last_waypoints,
                            agent_state=agent_state,
                            collided=getattr(episode, 'collided', False),
                            distance_to_goal=episode.distance_to_goal
                        )

                        if vis_renderer is not None and vis_renderer.online_render:
                            recovery_ref_img = None
                            recovery_ref_img_idx = None
                            ref_img_path = None
                            if last_matches_data is not None:
                                recovery_ref_img_idx = last_matches_data.get('closest_map_img_idx')
                                if recovery_ref_img_idx is not None:
                                    ref_img_path = episode.map_img_paths[recovery_ref_img_idx]
                                    recovery_ref_img = cv2.imread(str(ref_img_path))[:, :, ::-1]

                            vis_renderer.render_step_visualizations(
                                step=step,
                                rgb=rgb,
                                costmap=getattr(episode, 'goal_mask', None),
                                matches_data=last_matches_data,
                                ref_img_path=ref_img_path,
                                sim=None,
                                trajectory_history=episode.agent_state_history,
                                start_position=np.array(episode.start_position),
                                goal_position=np.array(episode.final_goal_position),
                                waypoints=last_waypoints,
                                ref_img=recovery_ref_img,
                                ref_img_idx=recovery_ref_img_idx,
                                agent_position=agent_state['position'],
                                agent_rotation=agent_state['rotation']
                            )

                    episode.log_results(step, final=False)
                    pbar.update(1)
                    step += 1

                    if angle_accumulated >= target_angle:
                        logger.info(
                            f"Real-world recovery: completed {np.degrees(angle_accumulated):.1f}° turn"
                        )
                        break
                else:
                    logger.warning("Real-world recovery rotation did not reach 45° within max steps")

            else:
                # ── Simulation recovery ────────────────────────────────────────────
                # Use geodesic path to find heading toward goal, then rotate toward it.
                _, geodesic_path = find_shortest_path(
                    episode.sim, episode.agent.get_state().position, episode.final_goal_position
                )
                if len(geodesic_path) >= 2:
                    next_pos = geodesic_path[1]
                    direction_vector = np.array(next_pos) - np.array(episode.agent.get_state().position)
                    heading_to_goal = np.arctan2(direction_vector[2], direction_vector[0])
                    current_rotation = episode.agent.get_state().rotation
                    # Compute agent heading from forward vector (consistent with heading_to_goal)
                    forward_vec = habitat_sim.utils.quat_rotate_vector(
                        current_rotation, np.array([0, 0, -1.0])
                    )
                    agent_heading = np.arctan2(forward_vec[2], forward_vec[0])
                    angle_diff = heading_to_goal - agent_heading
                    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
                    direction = 1 if angle_diff > 0 else -1

                    max_recovery_steps = 50  # Safety limit
                    for _ in range(max_recovery_steps):
                        episode.velocity_control = 0.0
                        episode.theta_control = 0.2 * direction  # Turn in direction of goal path
                        episode.execute_action()

                        # Get new observation
                        observations = episode.sim.get_sensor_observations()
                        rgb, depth_gt, _ = split_observations(observations)
                        robot_pose = episode.agent.get_state()

                        # Update distance to goal for logging
                        episode.is_done()

                        # Save visualization data during recovery to avoid jumps
                        # Reuse last known matches/waypoints so files are consistent
                        if cfg.visualization.save_raw_data.enabled and data_collector is not None:
                            agent_state_obj = episode.agent.get_state()
                            agent_state = {
                                'position': np.array(agent_state_obj.position),
                                'rotation': np.array([agent_state_obj.rotation.w, agent_state_obj.rotation.x,
                                                      agent_state_obj.rotation.y, agent_state_obj.rotation.z])
                            }
                            data_collector.save_step_data(
                                step=step,
                                rgb=rgb,
                                depth=depth_gt,
                                pts3d=None,
                                costmap=getattr(episode, 'goal_mask', None),
                                matches_data=last_matches_data,
                                velocity=episode.velocity_control,
                                theta=episode.theta_control,
                                waypoints=last_waypoints,
                                agent_state=agent_state,
                                collided=episode.collided,
                                distance_to_goal=episode.distance_to_goal
                            )

                            # Render combined visualization during recovery
                            if vis_renderer is not None and vis_renderer.online_render:
                                # Get ref image from last known match
                                recovery_ref_img = None
                                recovery_ref_img_idx = None
                                if last_matches_data is not None:
                                    recovery_ref_img_idx = last_matches_data.get('closest_map_img_idx')
                                    if recovery_ref_img_idx is not None:
                                        ref_img_path = episode.map_img_paths[recovery_ref_img_idx]
                                        recovery_ref_img = cv2.imread(str(ref_img_path))[:, :, ::-1]

                                vis_renderer.render_step_visualizations(
                                    step=step,
                                    rgb=rgb,
                                    costmap=getattr(episode, 'goal_mask', None),
                                    matches_data=last_matches_data,
                                    ref_img_path=ref_img_path if recovery_ref_img_idx is not None else None,
                                    sim=episode.sim,
                                    trajectory_history=episode.agent_state_history,
                                    start_position=np.array(episode.start_position),
                                    goal_position=np.array(episode.final_goal_position),
                                    waypoints=last_waypoints,
                                    ref_img=recovery_ref_img,
                                    ref_img_idx=recovery_ref_img_idx,
                                    agent_position=agent_state['position'],
                                    agent_rotation=agent_state['rotation']
                                )

                        # Log results during recovery
                        episode.log_results(step, final=False)
                        pbar.update(1)
                        step += 1

                        # break if current heading is now aligned with heading to goal
                        curr_forward = habitat_sim.utils.quat_rotate_vector(
                            robot_pose.rotation, np.array([0, 0, -1.0])
                        )
                        curr_heading = np.arctan2(curr_forward[2], curr_forward[0])
                        angle_moved = curr_heading - agent_heading
                        angle_moved = (angle_moved + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
                        angle_moved = abs(angle_moved)
                        if angle_moved >= min(abs(angle_diff), np.radians(45)):  # If we've turned enough towards the goal direction
                            logger.info(f"Completed recovery rotation, angle moved: {np.degrees(angle_moved):.2f} degrees")
                            break
                    else:
                        logger.warning("Recovery rotation did not converge within max steps")
                else:
                    logger.warning("Geodesic path too short for collision recovery")

            # Always exit recovery mode after attempting recovery
            collision_avoidance_mode = False
            episode.success_status = None  # Clear stuck status so navigation can continue
            recovery_cooldown = 5  # Skip stuck detection for a few steps to allow for recovery effect

            # Skip the rest of this iteration — next loop will do normal nav
            timing_manager.end_timer("navigation_step")
            continue
        
        # === 4. GOAL GENERATION ===
        timing_manager.start_timer("goal_generation")
        try:
            if cfg.visualization.save_raw_data.enabled and data_collector is not None:
                goal_mask, vis_data = episode.get_goal(
                    rgb=rgb, depth=depth, pose=robot_pose, pts3d=pts3d, return_vis_data=True
                )
            else:
                goal_mask = episode.get_goal(rgb=rgb, depth=depth, pose=robot_pose, pts3d=pts3d)
                vis_data = None
        except Exception as e:
            logger.error(f"Error in goal generation at step {step}: {e}")
            collision_avoidance_mode = True  # Trigger collision avoidance
        timing_manager.end_timer("goal_generation")
        
        # === 5. CONTROL ===
        timing_manager.start_timer("control_prediction")
        episode.get_control_signal(step, rgb, depth)
        timing_manager.end_timer("control_prediction")
        
        # === 6. COLLISION AVOIDANCE (CARE-based, if enabled) ===
        care_data = None
        waypoints_adjusted = None
        if cfg.get("use_care_collision_avoidance", False):
            try:
                from libs.collision_avoidance.care.care import care_step, DEFAULT_PARAMS
                
                if hasattr(episode.goal_controller, 'controller_logs') and episode.goal_controller.controller_logs:
                    waypoints = episode.goal_controller.controller_logs[-1].get("action_pred")
                    
                    if waypoints is not None:
                        intrinsics = {
                            "fx": episode.agent_intrinsics[0, 0].item(),
                            "fy": episode.agent_intrinsics[1, 1].item(),
                            "cx": episode.agent_intrinsics[0, 2].item(),
                            "cy": episode.agent_intrinsics[1, 2].item()
                        }
                        
                        out = care_step(rgb, depth if depth is not None else depth_gt, 
                                       waypoints, intrinsics, DEFAULT_PARAMS)
                        
                        # Capture all CARE outputs for storage and visualization
                        waypoints_adjusted = out['adjusted_waypoints']
                        care_data = {
                            'waypoints_adjusted': waypoints_adjusted,
                            'obstacles': out.get('obstacles'),
                            'theta_rot': out.get('theta_rot'),
                            'k_star': out.get('k_star'),
                            'frep_all': out.get('frep_all'),
                            'v': out.get('v'),
                            'omega': out.get('omega'),
                        }
                        
                        # Update control based on adjusted waypoints
                        wp_index = getattr(episode.goal_controller, 'waypoint_index', 5)
                        wp = waypoints_adjusted[wp_index][:2]
                        
                        w = np.arctan2(wp[-1], wp[-2])
                        w = np.clip(w, -0.1, 0.1)
                        v = min(wp[0]/100, 0.05)
                        
                        episode.velocity_control = v
                        episode.theta_control = -w
                        
                        logger.debug(f"CARE adjusted: v={v:.3f}, w={w:.3f}, theta_rot={out.get('theta_rot', 0):.3f}")
            except Exception as e:
                logger.warning(f"CARE collision avoidance failed: {e}")
        
        # === 7. EXECUTE ACTION ===
        timing_manager.start_timer("action_execution")
        episode.execute_action()
        timing_manager.end_timer("action_execution")
        
        # === 8. SAVE VISUALIZATION DATA ===
        if cfg.visualization.save_raw_data.enabled and data_collector is not None and vis_data is not None:
            timing_manager.start_timer("save_visualization_data")
            
            # Prepare matches data
            matches_data = {
                'qry_img_idx': step,
                'closest_map_img_idx': vis_data['closest_map_img_idx'],
                'localized_img_idxs': vis_data.get('localized_img_idxs', np.array([])),
                'qry_mkpts': vis_data['qry_mkpts'],
                'ref_mkpts': vis_data['ref_mkpts'],
                'confidences': vis_data['confidences']
            }
            last_matches_data = matches_data  # Track for recovery steps
            
            # Extract waypoints from controller
            waypoints = None
            if hasattr(episode.goal_controller, 'controller_logs') and episode.goal_controller.controller_logs:
                last_log = episode.goal_controller.controller_logs[-1]
                if 'action_pred' in last_log:
                    waypoints = last_log['action_pred']
            last_waypoints = waypoints  # Track for recovery steps
            
            # Agent state
            agent_state_obj = episode.agent.get_state()
            agent_state = {
                'position': np.array(agent_state_obj.position),
                'rotation': np.array([agent_state_obj.rotation.w, agent_state_obj.rotation.x, 
                                      agent_state_obj.rotation.y, agent_state_obj.rotation.z])
            }
            
            # Save data
            data_collector.save_step_data(
                step=step,
                rgb=rgb,
                depth=depth if pts3d_source != "mast3r" else None,
                pts3d=pts3d,
                costmap=episode.goal_mask,
                matches_data=matches_data,
                velocity=episode.velocity_control,
                theta=episode.theta_control,
                waypoints=waypoints,
                agent_state=agent_state,
                collided=episode.collided,
                distance_to_goal=episode.distance_to_goal,
                care_data=care_data
            )
            
            # Online rendering (if enabled)
            if vis_renderer is not None and vis_renderer.online_render:
                ref_img_path = episode.map_img_paths[vis_data['closest_map_img_idx']]
                ref_img = cv2.imread(str(ref_img_path))[:, :, ::-1]  # BGR to RGB
                vis_renderer.render_step_visualizations(
                    step=step,
                    rgb=rgb,
                    costmap=episode.goal_mask,
                    matches_data=matches_data,
                    ref_img_path=ref_img_path,
                    sim=episode.sim,
                    trajectory_history=episode.agent_state_history,
                    start_position=np.array(episode.start_position),
                    goal_position=np.array(episode.final_goal_position),
                    waypoints=waypoints,
                    waypoints_adjusted=waypoints_adjusted,
                    care_data=care_data,
                    ref_img=ref_img,
                    ref_img_idx=vis_data['closest_map_img_idx'],
                    agent_position=agent_state['position'],
                    agent_rotation=agent_state['rotation']
                )
            
            timing_manager.end_timer("save_visualization_data")
        
        # === 9. LOG RESULTS ===
        timing_manager.start_timer("logging")
        episode.log_results(step, final=False)
        timing_manager.end_timer("logging")
        
        # Progress update
        step_time = time.time() - step_start
        pbar.set_postfix({
            "dist": f"{episode.distance_to_goal:.2f}m",
            "v": f"{episode.velocity_control:.3f}",
            "w": f"{episode.theta_control:.3f}",
            "t": f"{step_time:.2f}s"
        })
        pbar.update(1)
        
        timing_manager.end_timer("navigation_step")
        
        # Print timing breakdown if benchmarking
        if cfg.get("benchmark_functions", False) and step % 10 == 0:
            timing_manager.print_last_function_timings()
        
        step += 1
    
    pbar.close()

    # === STEP FREQUENCY & VRAM SUMMARY ===
    nav_step_times = timing_manager.function_timings.get("navigation_step", [])
    if nav_step_times:
        mean_s = float(np.mean(nav_step_times))
        total_s = float(np.sum(nav_step_times))
        hz = 1.0 / mean_s if mean_s > 0 else 0.0
        logger.info(
            f"Step frequency: {hz:.2f} Hz  "
            f"(mean {mean_s * 1000:.1f} ms/step, "
            f"{len(nav_step_times)} steps, "
            f"total {total_s:.2f} s)"
        )
    else:
        # Fallback: derive from wall time when timing_manager is disabled
        if step > 0 and 'step_start' in dir():
            pass  # step_start is local to the loop; no data available without timing_manager
        logger.info(
            "Step frequency not available — set benchmark_functions=true to enable timing_manager"
        )
    if torch.cuda.is_available():
        vram_alloc_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        vram_reserv_gb = torch.cuda.max_memory_reserved() / 1024 ** 3
        logger.info(
            f"Peak VRAM: {vram_alloc_gb:.3f} GB allocated, "
            f"{vram_reserv_gb:.3f} GB reserved"
        )

    # === FINALIZE ===
    if step >= max_steps:
        episode.success_status = "exceeded_steps"
        logger.warning(f"Episode exceeded max steps ({max_steps})")
    
    # Log final results
    episode.log_results(step, final=True)
    
    # Compute comprehensive metrics
    results_csv_path = episode_results_path / "results.csv"
    metrics = compute_episode_metrics(episode, results_csv_path)
    
    # Save metrics
    metrics_csv_path = episode_results_path / "metrics.csv"
    save_metrics_to_csv(metrics, metrics_csv_path, episode.success_status)
    logger.info(f"Metrics: SPL={metrics['spl']:.4f}, Soft SPL={metrics['soft_spl']:.4f}, "
               f"Collisions={metrics['avg_collisions']:.2f}")
    
    # Save episode visualization metadata
    if cfg.visualization.save_raw_data.enabled and data_collector is not None:
        data_collector.save_episode_metadata(
            success_status=episode.success_status,
            total_distance=episode.distance_to_final_goal,
            final_distance_to_goal=episode.distance_to_goal
        )
    
    # # Generate visualizations if enabled
    # if cfg.get("create_visualizations", False):
    #     try:
    #         from libs.visualizations.create_heatmaps import create_heatmap_videos
    #         from libs.visualizations.create_waypoints import create_waypoint_videos
    #         from libs.visualizations.create_matcher_vis import create_matcher_combined_videos
            
    #         logger.info("Generating visualizations from saved data...")
    #         src_dirs = [str(episode_results_path)]
            
    #         create_heatmap_videos(src_dirs)
    #         create_waypoint_videos(src_dirs)
            
    #         logger.info("Visualizations generated successfully")
    #     except Exception as e:
    #         logger.warning(f"Error generating visualizations: {e}")
    
    # Results summary
    results = {
        "success_status": episode.success_status,
        "steps": step,
        "distance_to_goal": episode.distance_to_goal,
        "distance_to_final_goal": episode.distance_to_final_goal,
        "metrics": metrics
    }
    
    # Close simulator
    episode.sim.close()
    
    return results

# ==============================================================================
# Episode Discovery
# ==============================================================================

def get_episode_list(cfg: DictConfig) -> list:
    """
    Get list of episode paths based on config.
    
    Priority order:
    1. If episode_list_file exists: use episodes from file
    2. If episode_list is non-empty: use only those specific episodes
    3. If multi_episode is true: get episodes from episodes_dir with filtering
    4. Otherwise: use single episode_path
    """
    from natsort import natsorted
    
    # Single episode mode
    if not cfg.multi_episode:
        episode_path = Path(cfg.episode_path)
        if not episode_path.exists():
            raise ValueError(f"Episode path does not exist: {episode_path}")
        return [episode_path]
    
    # Multi-episode mode
    episodes_dir = Path(cfg.episodes_dir) if cfg.get("episodes_dir") else None
    if not episodes_dir or not episodes_dir.exists():
        raise ValueError(f"Episodes directory does not exist: {episodes_dir}")
    
    # Priority 1: episode_list_file
    episode_list_file = cfg.get("episode_list_file", None)
    if episode_list_file:
        file_path = Path(episode_list_file)
        if not file_path.exists():
            raise ValueError(f"Episode list file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            episode_names = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(episode_names)} episodes from {file_path}")
        
        episodes = []
        for ep_name in episode_names:
            ep_path = episodes_dir / ep_name
            if ep_path.exists():
                episodes.append(ep_path)
            else:
                logger.warning(f"Episode not found: {ep_path}")
        
        return natsorted(episodes, key=lambda x: x.name)
    
    # Priority 2: episode_list array
    episode_list = cfg.get("episode_list", [])
    if episode_list and len(episode_list) > 0:
        logger.info(f"Using episode_list with {len(episode_list)} episodes")
        
        episodes = []
        for ep_name in episode_list:
            ep_path = episodes_dir / ep_name
            if ep_path.exists():
                episodes.append(ep_path)
            else:
                logger.warning(f"Episode not found: {ep_path}")
        
        return natsorted(episodes, key=lambda x: x.name)
    
    # Priority 3: All episodes with filtering
    all_episodes = [p for p in episodes_dir.iterdir() if p.is_dir()]
    all_episodes = natsorted(all_episodes, key=lambda x: x.name)
    
    logger.info(f"Found {len(all_episodes)} episodes in {episodes_dir}")
    
    # Apply start/end filtering
    start_idx = cfg.get("episode_start_idx", 0)
    end_idx = cfg.get("episode_end_idx", -1)
    
    if start_idx > 0:
        all_episodes = all_episodes[start_idx:]
    if end_idx > 0:
        all_episodes = all_episodes[:end_idx - start_idx]
    
    logger.info(f"After filtering (start={start_idx}, end={end_idx}): {len(all_episodes)} episodes")
    
    # Apply blacklist
    blacklist = cfg.get("episode_blacklist", [])
    if blacklist:
        episodes = [ep for ep in all_episodes 
                   if not any(bl in ep.name for bl in blacklist)]
        logger.info(f"After blacklist filtering: {len(episodes)} episodes")
    else:
        episodes = all_episodes
    
    return episodes


def _extract_goal_idx_from_string(text: Optional[str]) -> Optional[int]:
    """Extract goal image index from a string containing 'goalImg<idx>' if present."""
    if not text:
        return None
    m = re.search(r"goalImg(\d+)", str(text))
    return int(m.group(1)) if m else None


def _get_goal_idx_from_goal_info(episode_path: Path, cfg: DictConfig) -> Optional[int]:
    """Read goal_info.json and return selected goal_image_index."""
    goal_info_path = episode_path / "goal_info.json"
    if not goal_info_path.exists():
        return None

    with open(goal_info_path, "r") as f:
        goal_info = json.load(f)

    if isinstance(goal_info, dict) and "goal_image_index" in goal_info:
        return int(goal_info["goal_image_index"])

    if isinstance(goal_info, list) and len(goal_info) > 0:
        goal_info_idx = int(cfg.get("goal_info_idx", 0))
        goal_info_idx = max(0, min(goal_info_idx, len(goal_info) - 1))
        return int(goal_info[goal_info_idx]["goal_image_index"])

    return None


def _resolve_costmap_file_for_goal(episode_path: Path, cfg: DictConfig, goal_idx: int) -> Optional[Path]:
    """Resolve a costmap npz matching goalImg<goal_idx> from common benchmark directories."""
    # If an explicit path already matches, keep it.
    explicit = cfg.get("costmap_file_path", None)
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists() and _extract_goal_idx_from_string(explicit_path.name) == goal_idx:
            return explicit_path

    search_dirs = []

    costmap_base_dir = cfg.get("costmap_base_dir", None)
    costmap_dirname = cfg.get("costmap_dirname", "topo_map_outputs")
    if costmap_base_dir:
        search_dirs.append(Path(costmap_base_dir) / episode_path.name / costmap_dirname)

    search_dirs.append(episode_path / costmap_dirname)
    search_dirs.append(episode_path.parent / costmap_dirname)

    for d in search_dirs:
        if not d.exists():
            continue
        candidates = sorted(d.glob(f"*goalImg{goal_idx}.npz"))
        if candidates:
            return candidates[0]

    return None


def _get_goal_idx_for_episode(episode_path: Path, cfg: DictConfig) -> Optional[int]:
    """
    Determine goal index per episode with priority:
    1) cfg.goal_image_index
    2) goalImg suffix in cfg.costmap_file_path / cfg.costmap_filename
    3) episode_path/goal_info.json (benchmark mode)
    """
    if cfg.get("goal_image_index", None) is not None:
        return int(cfg.goal_image_index)

    idx = _extract_goal_idx_from_string(cfg.get("costmap_file_path", None))
    if idx is not None:
        return idx

    idx = _extract_goal_idx_from_string(cfg.get("costmap_filename", None))
    if idx is not None:
        return idx

    return _get_goal_idx_from_goal_info(episode_path, cfg)

# ==============================================================================
# Main Entry Point
# ==============================================================================

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point (shared by both Hydra and custom config)."""
    global GLOBAL_RESULTS_PATH
    
    # Enable timing if requested
    if cfg.get("benchmark_functions", False):
        timing_manager.enable()
        logger.info("Function-level benchmarking enabled")
    
    # Print config
    logger.info("=" * 60)
    logger.info("mast3r-nav Navigation Test")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    initialized_model = None
    matcher = None

    # Initialize matcher based on config
    if not cfg.model.use_gt_matches or not cfg.model.use_gt_depth:
        matcher_name = cfg.matcher.get("name", "mast3r")
        
        if matcher_name == "mast3r":
            matcher = Mast3rMatcher(
                resize_w=cfg.matcher.resize_w,
                resize_h=cfg.matcher.resize_h,
                geometric_verification=cfg.matcher.geometric_verification,
                subsample_or_initxy1=cfg.matcher.subsample_or_initxy1,
                device=cfg.device
            )
            # Reuse mast3r model for depth estimation if using mast3r matcher
            initialized_model = matcher.model
        elif matcher_name == "superpoint":
            matcher = SuperPointMatcher(
                resize_w=cfg.matcher.resize_w,
                resize_h=cfg.matcher.resize_h,
                geometric_verification=cfg.matcher.geometric_verification,
                max_keypoints=cfg.matcher.max_keypoints,
                keypoint_threshold=cfg.matcher.keypoint_threshold,
                device=cfg.device
            )
            # SuperPoint doesn't share model with mast3r depth - initialized_model stays None
        else:
            raise ValueError(f"Unknown matcher: {matcher_name}. Supported: 'mast3r', 'superpoint'")
    
    # Load MASt3R model if needed for depth estimation
    mast3r_model = None
    if cfg.get("pts3d_source", "gt_depth") == "mast3r":
        logger.info("Loading MASt3R model for 3D reconstruction...")
        if initialized_model is not None:
            logger.info("  -> Reusing model from Mast3rMatcher")
        mast3r_model = MASt3RInference(device=cfg.device, initialized_model=initialized_model)
    
    # Initialize results directory
    results_path = task_setup.init_results_dir_and_save_cfg(cfg, default_logger)
    logger.info(f"Results will be saved to: {results_path}")
    GLOBAL_RESULTS_PATH = results_path
    
    # Get episodes to process
    try:
        episodes = get_episode_list(cfg)
    except ValueError as e:
        logger.error(str(e))
        return
    
    logger.info(f"Processing {len(episodes)} episode(s)")
    
    # Results tracking
    results_summary = {
        'total_episodes': 0,
        'successful_episodes': 0,
        'failed_episodes': 0,
        'exceeded_steps': 0,
        'stuck': 0,
        'success_rate': 0.0,
        'episode_results': [],
        'spl_scores': [],
        'soft_spl_scores': [],
        'successful_spl_scores': [],
        'collision_scores': []
    }
    
    # Process each episode
    for ei, episode_path in enumerate(tqdm(episodes, desc="Processing Episodes")):
        episode_name = episode_path.parts[-1]

        # Benchmark convenience: infer goal index and matching costmap from files.
        goal_idx = _get_goal_idx_for_episode(episode_path, cfg)
        if goal_idx is not None:
            logger.info(f"Using goal image index {goal_idx} for episode {episode_name}")

            auto_resolve_costmap = bool(cfg.get("auto_resolve_costmap_from_goal_info", True))
            if auto_resolve_costmap:
                resolved_costmap = _resolve_costmap_file_for_goal(episode_path, cfg, goal_idx)
                if resolved_costmap is not None:
                    with open_dict(cfg):
                        cfg.costmap_file_path = str(resolved_costmap)
                    logger.info(f"Resolved costmap_file_path from goal index: {resolved_costmap}")
                else:
                    logger.warning(
                        f"Could not auto-resolve costmap for goalImg{goal_idx} in common directories. "
                        "Falling back to existing costmap settings."
                    )

        # Get number of random iterations
        num_iterations = cfg.get("num_random_iterations", 1)
        # start_indices = cfg.get("start_indices", None)  # List of specific start indices
        if cfg.get("start_indices", None) is not None and len(cfg.start_indices) > 0:
            start_indices = cfg.start_indices
            logger.info(f"Using specified start indices from config: {start_indices}")
        else:
            if cfg.get("start_state_mode", "random") == "fixed_idx":
                # search for start_state.json in scene dir
                start_state_file = episode_path / "start_states.json"
                start_states_per_goal = {}
                if start_state_file.exists():
                    with open(start_state_file, 'r') as f:
                        start_states_per_goal = json.load(f)
                        start_states_per_goal = {int(k): v for k, v in start_states_per_goal.items()}
                    logger.info(f"Loaded start states from {start_state_file}")
                else:
                    raise ValueError(f"Start states file not found: {start_state_file}")

                # Prefer already resolved goal index; fallback to parsing costmap names.
                if goal_idx is None:
                    goal_idx = _extract_goal_idx_from_string(cfg.get("costmap_filename", ""))
                if goal_idx is None:
                    goal_idx = _extract_goal_idx_from_string(cfg.get("costmap_file_path", ""))

                if goal_idx is None:
                    raise ValueError(
                        "Could not determine goal index for fixed start mode. "
                        "Provide goal_image_index, costmap with goalImg suffix, or goal_info.json."
                    )

                start_indices = start_states_per_goal.get(goal_idx, [])
                if len(start_indices) == 0:
                    logger.warning(f"No start indices found in start_states.json for goal {goal_idx}")
            else:
                start_indices = []
        
        if start_indices:
            num_iterations = len(start_indices)
            logger.info(f"Using {num_iterations} fixed start indices: {start_indices}")
        else:
            logger.info(f"Running {num_iterations} iteration(s) per episode with random starts")
        
        # Run multiple iterations per episode
        for iter_idx in range(num_iterations):
            logger.info("=" * 60)
            logger.info(f"Episode {ei+1}/{len(episodes)}, Iteration {iter_idx+1}/{num_iterations}: {episode_name}")
            logger.info("=" * 60)
            
            results_summary['total_episodes'] += 1
            
            # Create iteration-specific results directory
            if num_iterations > 1:
                episode_results_dir = f"{episode_name}_{cfg.controller.name}_{cfg.goal_source}_iter{iter_idx:03d}"
            else:
                episode_results_dir = f"{episode_name}_{cfg.controller.name}_{cfg.goal_source}"
            
            episode_results_path = results_path / episode_results_dir
            episode_results_path.mkdir(exist_ok=True, parents=True)
            
            # Determine start index
            if start_indices:
                start_idx = start_indices[iter_idx]
            else:
                start_idx = -1  # Random
            
            try:
                # Run episode
                results = run_episode(
                    cfg, episode_path, episode_results_path, 
                    mast3r_model, iteration_idx=iter_idx, start_idx=start_idx, initialized_matcher=matcher
                )
                
                # Track results
                results['episode_name'] = episode_name
                results['iteration'] = iter_idx
                results_summary['episode_results'].append(results)
                
                # Track success/failure
                if results['success_status'] == 'success':
                    results_summary['successful_episodes'] += 1
                elif results['success_status'] == 'exceeded_steps':
                    results_summary['exceeded_steps'] += 1
                    results_summary['failed_episodes'] += 1
                elif 'stuck' in results['success_status']:
                    results_summary['stuck'] += 1
                    results_summary['failed_episodes'] += 1
                else:
                    results_summary['failed_episodes'] += 1
                
                # Track metrics
                metrics = results['metrics']
                results_summary['spl_scores'].append(metrics['spl'])
                results_summary['soft_spl_scores'].append(metrics['soft_spl'])
                results_summary['collision_scores'].append(metrics['avg_collisions'])
                
                if metrics['success']:
                    results_summary['successful_spl_scores'].append(metrics['spl'])
                
                # Save individual episode results
                results_file = episode_results_path / "results_summary.txt"
                with open(results_file, "w") as f:
                    f.write(f"episode_name: {episode_name}\n")
                    f.write(f"iteration: {iter_idx}\n")
                    f.write(f"success_status: {results['success_status']}\n")
                    f.write(f"steps: {results['steps']}\n")
                    f.write(f"distance_to_goal: {results['distance_to_goal']:.4f}\n")
                    f.write(f"distance_to_final_goal: {results['distance_to_final_goal']:.4f}\n")
                    f.write(f"spl: {metrics['spl']:.4f}\n")
                    f.write(f"soft_spl: {metrics['soft_spl']:.4f}\n")
                
                logger.info(f"Episode {episode_name} (iter {iter_idx}): {results['success_status']} "
                           f"(steps={results['steps']}, dist={results['distance_to_goal']:.2f}m, "
                           f"SPL={metrics['spl']:.4f}, Soft SPL={metrics['soft_spl']:.4f})")
                
            except Exception as e:
                logger.error(f"Error processing episode {episode_name} iteration {iter_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                results_summary['failed_episodes'] += 1
                results_summary['episode_results'].append({
                    'episode_name': episode_name,
                    'iteration': iter_idx,
                    'success_status': f'error: {str(e)}',
                    'steps': 0,
                    'distance_to_goal': float('nan'),
                    'distance_to_final_goal': float('nan'),
                    'metrics': {
                        'spl': 0.0,
                        'soft_spl': 0.0,
                        'path_length': 0.0,
                        'avg_collisions': 0.0
                    }
                })
            
            # Save timing data for this iteration
            if cfg.get("benchmark_functions", False):
                timing_manager.save_function_timings_csv(episode_results_path)
                timing_manager.create_hierarchical_report(episode_results_path)
                timing_manager.save_step_timing_breakdown(episode_results_path)
                timing_manager.clear_all_data()  # Clear for next iteration
    
    # === FINAL SUMMARY ===
    
    # Calculate success rate
    results_summary['success_rate'] = (
        results_summary['successful_episodes'] / results_summary['total_episodes'] * 100
        if results_summary['total_episodes'] > 0 else 0
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Final Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total Episodes: {results_summary['total_episodes']}")
    logger.info(f"Successful: {results_summary['successful_episodes']}")
    logger.info(f"Failed: {results_summary['failed_episodes']}")
    logger.info(f"  - Exceeded Steps: {results_summary['exceeded_steps']}")
    logger.info(f"  - Stuck: {results_summary['stuck']}")
    logger.info(f"Success Rate: {results_summary['success_rate']:.2f}%")
    
    if results_summary['spl_scores']:
        logger.info(f"Mean SPL: {np.mean(results_summary['spl_scores']):.4f}")
        logger.info(f"Mean Soft SPL: {np.mean(results_summary['soft_spl_scores']):.4f}")
        logger.info(f"Mean Collisions: {np.mean(results_summary['collision_scores']):.2f}")
        
        if results_summary['successful_spl_scores']:
            logger.info(f"Successful Episodes SPL: {np.mean(results_summary['successful_spl_scores']):.4f} "
                       f"(min: {np.min(results_summary['successful_spl_scores']):.4f}, "
                       f"max: {np.max(results_summary['successful_spl_scores']):.4f})")
    
    logger.info("=" * 60)
    
    # Save overall results summary
    summary_file = results_path / "results_summary.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_name", "iteration", "success_status", "steps", 
                        "distance_to_goal", "distance_to_final_goal", "spl", "soft_spl"])
        
        for ep_result in results_summary['episode_results']:
            metrics = ep_result.get('metrics', {})
            writer.writerow([
                ep_result['episode_name'],
                ep_result.get('iteration', 0),
                ep_result['success_status'],
                ep_result['steps'],
                f"{ep_result['distance_to_goal']:.4f}",
                f"{ep_result['distance_to_final_goal']:.4f}",
                f"{metrics.get('spl', 0):.4f}",
                f"{metrics.get('soft_spl', 0):.4f}"
            ])
    
    logger.info(f"Results summary saved to: {summary_file}")
    
    # Save aggregated metrics
    metrics_file = results_path / "metrics_summary.txt"
    with open(metrics_file, "w") as f:
        f.write(f"total_episodes: {results_summary['total_episodes']}\n")
        f.write(f"successful_episodes: {results_summary['successful_episodes']}\n")
        f.write(f"failed_episodes: {results_summary['failed_episodes']}\n")
        f.write(f"exceeded_steps: {results_summary['exceeded_steps']}\n")
        f.write(f"stuck: {results_summary['stuck']}\n")
        f.write(f"success_rate: {results_summary['success_rate']:.2f}\n")
        
        if results_summary['spl_scores']:
            f.write(f"mean_spl: {np.mean(results_summary['spl_scores']):.4f}\n")
            f.write(f"mean_soft_spl: {np.mean(results_summary['soft_spl_scores']):.4f}\n")
            f.write(f"mean_collisions: {np.mean(results_summary['collision_scores']):.2f}\n")
            
            if results_summary['successful_spl_scores']:
                f.write(f"successful_mean_spl: {np.mean(results_summary['successful_spl_scores']):.4f}\n")
                f.write(f"successful_min_spl: {np.min(results_summary['successful_spl_scores']):.4f}\n")
                f.write(f"successful_max_spl: {np.max(results_summary['successful_spl_scores']):.4f}\n")
    
    logger.info(f"Metrics summary saved to: {metrics_file}")
    
    # Final timing report
    if cfg.get("benchmark_functions", False):
        save_timing_on_exit()


if __name__ == "__main__":
    main()