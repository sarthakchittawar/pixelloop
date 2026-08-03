import os
import sys

# IMPORTANT: Set habitat-sim env vars BEFORE importing habitat_sim
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import numpy as np
from pathlib import Path
from natsort import natsorted
import json
import pickle
import torch
import cv2
from datetime import datetime
from typing import Tuple
import torchvision.transforms as tfm
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib.colors import Normalize

import habitat_sim
from habitat.utils.visualizations import maps
import magnum as mn

from libs.simulation.habitat_utils import get_sim_agent
from libs.experiments.episode_utils import (
    pick_random_start_state,
    select_trajectory_start_state,
    calculate_path_distance,
    find_shortest_path,
    initialize_results,
    write_results,
    write_final_meta_results
)
from libs.mapper.create_topomap import CostmapData
from libs.matcher.mast3r_matcher import Mast3rMatcher
from libs.matcher.superpoint_matcher import SuperPointMatcher
from libs.localizer.loc_topo import LocalizeTopological
from libs.planner.plan_topo import PlanTopological
from libs.goal_generator.goal_gen import GoalGenerator
from libs.common.utils_sim import build_intrinsics, apply_velocity
from libs.control.learnt_controller import ObjRelLearntController

import logging
logger = logging.getLogger("[Task Setup]")

def _quat_to_heading(q):
    """Convert quaternion to heading angle"""
    R = np.array(mn.Quaternion(q.imag, q.real).to_matrix())
    R_bc = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    R = R @ R_bc
    return np.arctan2(R[0,2], R[2,2])

def _sim_to_grid(sim, tdv_dims, pos):
    """Convert simulation position to grid coordinates"""
    return maps.to_grid(pos[2], pos[0], tdv_dims, pathfinder=sim.pathfinder)

class Episode:
    def __init__(self, cfg: DictConfig, episode_path, scene_glb_path, episode_results_path, preload_data={}, start_idx=0, initialized_matcher=None):

        self.cfg = cfg
        self.steps = 0
        self.device = cfg.device
        self.H = self.cfg.sim.height
        self.W = self.cfg.sim.width

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.episode_path = Path(episode_path)
        self.localization_episode_path = self.episode_path
        self._agent_states_cache = {}
        self.episode_results_path = Path(episode_results_path)
        logger.info(f"Running {self.episode_path=}...")

        self.scene_glb_path = scene_glb_path
        self.preload_data = preload_data
        self.episode_img_dir = episode_path / 'images_fov90'
        self.closest_map_img_idx = None
        self.localizedImgInds = None
        
        self.results_folder_path = Path(episode_results_path)
        self.episode_vis_dir = self.results_folder_path / "vis"

        self.start_idx = start_idx
        self.loc_radius = self.cfg.localizer.loc_radius
        self.subsample_ref = self.cfg.localizer.subsample_ref

        # Resolve costmap path with priority:
        # 1) cfg.costmap_file_path (direct file override)
        # 2) {costmap_base_dir}/{episode_name}/{costmap_dirname}/{costmap_filename}
        # 3) {episode_path}/{costmap_dirname}/{costmap_filename}
        costmap_file_override = cfg.get("costmap_file_path", None)
        if costmap_file_override:
            self.costmap_file_path = Path(costmap_file_override)
        else:
            costmap_base_dir = cfg.get("costmap_base_dir", None)
            costmap_filename = cfg.costmap_filename
            costmap_dirname = cfg.costmap_dirname
            episode_name = self.episode_path.name

            if costmap_base_dir:
                self.costmap_file_path = Path(costmap_base_dir) / episode_name / costmap_dirname / costmap_filename
            else:
                self.costmap_file_path = self.episode_path / costmap_dirname / costmap_filename
        
        if not self.costmap_file_path.exists():
            raise FileNotFoundError(f"Costmap file not found: {self.costmap_file_path}")
        
        logger.info(f"Loading costmap from: {self.costmap_file_path}")

        # Getting the Mapping costmap data
        self.costmap_data = CostmapData.from_npz(self.costmap_file_path)
        costmap_metadata = self.costmap_data.get_metadata()

        self.map_img_paths = []
        self.uses_external_map_images = False
        self.episode_global_map_indices = []
        image_paths_costmap = costmap_metadata['image_paths']
        for global_idx, image_path in enumerate(image_paths_costmap):
            # Multi-folder map creation stores original image paths across folders,
            # while legacy maps only need episode-local image names.
            candidate_path = Path(image_path)
            if candidate_path.exists():
                self.map_img_paths.append(str(candidate_path))
                if self.episode_path in candidate_path.parents:
                    self.episode_global_map_indices.append(global_idx)
                else:
                    self.uses_external_map_images = True
            else:
                img_name = os.path.basename(image_path)
                map_img_path = str(self.episode_img_dir / img_name)
                self.map_img_paths.append(map_img_path)

            # Initialize mapping for active localization episode.
            self._set_localization_episode(self.episode_path)
            self._prepare_global_pose_lookup()

        if self.uses_external_map_images and self.cfg.localizer.use_gt_localization:
            logger.info(
                "Using GT/pose localization with multi-folder map. "
                "Localization indices will be mapped from episode-local frames to global map indices."
            )

        self.method = self.cfg.controller.name

        self.init_controller_params()

        self.setup_sim_agent()
        self.ready_agent()

        # robot intrinsics in simulator
        self.agent_intrinsics = build_intrinsics(
            image_width=self.W,
            image_height=self.H,
            field_of_view_radians_u=self.hfov_radians,
            device=self.device
        )

        if initialized_matcher is not None and self.cfg.matcher.name == "mast3r":
            self.matcher = initialized_matcher
        else:
            self.matcher = None

        # Get the goal generator
        self.get_goal_generator()

        # Set the controller
        self.set_controller()
        self.vis_img_default = np.zeros((self.H, self.W, 3)).astype(np.uint8)
        
        # Tracking for absolute goal mask scale
        self.goal_mask_absolute_max = None
    
    def init_controller_params(self):
        self.fov_deg = self.cfg.sim.hfov if 'robohop' in self.method.lower() else 79
        self.hfov_radians = np.pi * self.fov_deg / 180

        # controller params
        self.time_delta = 0.1
        self.theta_control = np.nan
        self.velocity_control = np.nan

        self.pid_steer_values = [.25, 0, 0] if self.method.lower(
        ) == 'robohop+' else []
        self.discrete_action = -1
        self.controller_logs = None
    
    def set_controller(self):
        method_name = self.cfg.controller.name
        self.collided = None
        controller_cfg = self.cfg.controller

        if method_name == 'learnt':
            goal_controller = ObjRelLearntController(
                config=controller_cfg.config_file,
                goal_source=self.cfg.goal_source,
                boost_final_goal=controller_cfg.boost_final_goal   
            )
            goal_controller.reset_params()
            goal_controller.dirname_vis_episode = self.episode_vis_dir
        else:
            raise NotImplementedError("Other controller methods have not been implemented yet")
        
        self.goal_controller = goal_controller

    def setup_sim_agent(self):
        # Note: MAGNUM_LOG and HABITAT_SIM_LOG are set at module level before import
        sim_cfg = self.cfg.sim

        # Initialize Habitat Sim and Agent
        self.sim, self.agent, self.vel_control = get_sim_agent(
            scene_path=self.scene_glb_path,
            update_nav_mesh=sim_cfg.update_nav_mesh,
            width=sim_cfg.width,
            height=sim_cfg.height,
            hfov=sim_cfg.hfov,
            sensor_height=sim_cfg.sensor_height,
        )
        self.sim.agents[0].agent_config.sensor_specifications[1].normalize_depth = True

        # create and configure a new VelocityControl structure
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True
        self.vel_control = vel_control
    
    def ready_agent(self, goal_init_flag=True):

        # 1. Load agent trajectory
        agent_states_path = self.episode_path / 'agent_states.npy'
        if agent_states_path.exists():
            self.agent_states = np.load(str(agent_states_path), allow_pickle=True)
        else:
            odom_path = self.episode_path / 'poses_odom.txt'
            if not odom_path.exists():
                raise FileNotFoundError(
                    f"Agent states file not found: {agent_states_path} and odometry fallback not found: {odom_path}"
                )
            logger.info(f"agent_states.npy missing, loading trajectory from odometry: {odom_path}")
            self.agent_states = self._load_agent_states_from_odometry(odom_path)

        if len(self.agent_states) == 0:
            raise ValueError(f"No valid agent states loaded for episode: {self.episode_path}")

        self.agent_positions_in_map = np.array([state.position for state in self.agent_states])

        # 2. Set goal based on task type
        self._set_goal_state()

        # 3. Select and set start state
        self._set_start_state()

        # 4. calculate distance metric
        self.distance_to_final_goal = calculate_path_distance(
            self.sim,
            self.start_position,
            self.final_goal_position,
        )
        self.agent_state_history = []

        logger.info(f"Agent ready: task_type={self.cfg.task_type}, "
            f"reverse={self.cfg.reverse}, "
            f"start_idx={self.start_idx}, "
            f"Start Position={self.start_position}, "
            f"Goal Position={self.final_goal_position}, "
            f"goal_distance={self.distance_to_final_goal:.2f}m")

    def _load_agent_states_from_odometry(self, odom_path: Path):
        """Build Habitat AgentState trajectory from poses_odom.txt (x y z qx qy qz qw)."""
        states = []
        with open(odom_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                state = habitat_sim.AgentState()
                state.position = np.array([x, y, z], dtype=np.float32)
                state.rotation = np.quaternion(qw, qx, qy, qz)
                states.append(state)

        return np.array(states, dtype=object)

    def _load_agent_states_for_episode(self, episode_dir: Path):
        """Load trajectory states for an episode directory, with cache."""
        episode_dir = Path(episode_dir)
        cache_key = str(episode_dir.resolve())
        if cache_key in self._agent_states_cache:
            return self._agent_states_cache[cache_key]

        agent_states_path = episode_dir / 'agent_states.npy'
        if agent_states_path.exists():
            states = np.load(str(agent_states_path), allow_pickle=True)
        else:
            odom_path = episode_dir / 'poses_odom.txt'
            if not odom_path.exists():
                raise FileNotFoundError(
                    f"Neither agent_states.npy nor poses_odom.txt found in {episode_dir}"
                )
            states = self._load_agent_states_from_odometry(odom_path)

        self._agent_states_cache[cache_key] = states
        return states

    def _set_localization_episode(self, episode_dir: Path):
        """Set active episode for GT/pose localization and refresh local->global mapping."""
        self.localization_episode_path = Path(episode_dir)
        self.episode_global_map_indices = []
        for global_idx, image_path in enumerate(self.map_img_paths):
            if self.localization_episode_path in Path(image_path).parents:
                self.episode_global_map_indices.append(global_idx)

    def _get_local_frame_idx_from_map_path(self, map_img_path: Path) -> int:
        """Infer local frame index from image filename, with natural-sort fallback."""
        try:
            return int(map_img_path.stem)
        except ValueError:
            img_files = natsorted([p.name for p in map_img_path.parent.iterdir() if p.is_file()])
            return img_files.index(map_img_path.name)

    def _prepare_global_pose_lookup(self):
        """Build global pose arrays aligned with map_img_paths for global odometry localization."""
        n = len(self.map_img_paths)
        self.global_pose_positions = np.full((n, 3), np.nan, dtype=np.float32)
        self.global_pose_quats = np.full((n, 4), np.nan, dtype=np.float32)  # qx,qy,qz,qw
        self.global_pose_valid_indices = []

        for global_idx, path_str in enumerate(self.map_img_paths):
            map_img_path = Path(path_str)
            if map_img_path.parent.name != 'images_fov90':
                continue

            episode_dir = map_img_path.parent.parent
            try:
                local_idx = self._get_local_frame_idx_from_map_path(map_img_path)
            except Exception:
                continue

            try:
                states = self._load_agent_states_for_episode(episode_dir)
            except Exception:
                continue

            if local_idx < 0 or local_idx >= len(states):
                continue

            st = states[local_idx]
            self.global_pose_positions[global_idx] = np.array(st.position, dtype=np.float32)
            q = st.rotation
            self.global_pose_quats[global_idx] = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
            self.global_pose_valid_indices.append(global_idx)

        self.global_pose_valid_indices = np.array(self.global_pose_valid_indices, dtype=np.int32)
        logger.info(
            f"Prepared global pose lookup: {len(self.global_pose_valid_indices)}/{len(self.map_img_paths)} indices valid"
        )

    def _resolve_start_state_from_global_index(self, global_idx: int):
        """
        Resolve a global map index to an AgentState by locating the corresponding
        episode folder and local frame index from map image paths.
        """
        if global_idx < 0 or global_idx >= len(self.map_img_paths):
            raise ValueError(
                f"Global start index {global_idx} out of range for map of size {len(self.map_img_paths)}"
            )

        map_img_path = Path(self.map_img_paths[global_idx])
        # .../<episode_dir>/images_fov90/<frame>.jpg
        if map_img_path.parent.name != 'images_fov90':
            raise ValueError(f"Could not resolve episode dir from map image path: {map_img_path}")

        source_episode_dir = map_img_path.parent.parent
        try:
            local_idx = int(map_img_path.stem)
        except ValueError:
            # Fallback if names are not zero-padded integers.
            img_files = natsorted([p.name for p in map_img_path.parent.iterdir() if p.is_file()])
            local_idx = img_files.index(map_img_path.name)

        source_states = self._load_agent_states_for_episode(source_episode_dir)
        if local_idx >= len(source_states):
            raise ValueError(
                f"Resolved local index {local_idx} out of range for {source_episode_dir} "
                f"(num_states={len(source_states)})"
            )

        return source_states[local_idx], source_episode_dir, local_idx, source_states

    def _resolve_agent_state_from_global_index(self, global_idx: int):
        """
        Resolve a global map frame index to an AgentState and source episode metadata.
        """
        if global_idx < 0 or global_idx >= len(self.map_img_paths):
            raise ValueError(
                f"Global index {global_idx} out of range for map size {len(self.map_img_paths)}"
            )

        map_img_path = Path(self.map_img_paths[global_idx])
        if map_img_path.parent.name != 'images_fov90':
            raise ValueError(f"Could not resolve episode dir from map image path: {map_img_path}")

        source_episode_dir = map_img_path.parent.parent
        local_idx = self._get_local_frame_idx_from_map_path(map_img_path)
        source_states = self._load_agent_states_for_episode(source_episode_dir)

        if local_idx < 0 or local_idx >= len(source_states):
            raise ValueError(
                f"Resolved local index {local_idx} out of range for {source_episode_dir} "
                f"(num_states={len(source_states)})"
            )

        return source_states[local_idx], source_episode_dir, local_idx, source_states
    
    def _set_goal_state(self):
        """Set goal state based on task type"""
        if self.cfg.reverse:
            self._set_reverse_goal()
        elif self.cfg.task_type in ['alt_goal', 'alt_goal_v2']:
            raise NotImplementedError("Alt goal task not implemented yet.")
            # self._set_alt_goal()
        else:
            self._set_topological_goal()
    
    def _set_reverse_goal(self):
        """Set goal for reverse navigation task."""
        self.final_goal_state = self.agent_states[0]
        self.final_goal_position = self.final_goal_state.position
        self.final_goal_image_idx = len(self.agent_states) - 1

        logger.debug(f"Reverse goal set: image_idx={self.final_goal_image_idx}")
    
    # TODO: Fix this function later
    def _set_alt_goal(self):
        # TODO : This function needs to be fixed later
        metadata_path = self.episode_path / 'alt_goal_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f'Alt goal metadata not found: {metadata_path}')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.final_goal_image_idx = metadata['goal_image_idx']
        goal_instance_id = metadata['goal_instance_id']
        
        # Find instance position in scene
        instance_position = None
        for instance in self.sim.semantic_scene.objects:
            if instance.semantic_id == goal_instance_id:
                instance_position = instance.aabb.center
                break
        
        if instance_position is None:
            raise ValueError(f'Goal instance {goal_instance_id} not found in scene')
        
        # Snap to navigable surface at average floor height
        avg_floor_height = self.agent_positions_in_map[:, 1].mean()
        instance_position = np.array(instance_position, dtype=np.float32)
        instance_position[1] = avg_floor_height
        self.final_goal_position = self.sim.pathfinder.snap_point(instance_position)
        self.final_goal_state = None
        
        logger.debug(f"Alt goal set: instance_id={goal_instance_id}, "
                    f"image_idx={self.final_goal_image_idx}")

    def _set_topological_goal(self):
        """Set goal for topological graph-based navigation"""
        metadata = self.costmap_data.get_metadata()

        # get final goal position from costmap metadata
        self.final_goal_image_idx = metadata['goal_img_idx']
        self.goal_node_idx = metadata['goal_node_id']
        self.goal_px, self.goal_py = metadata['goal_pixel']
        self.goal_coord_3d = np.array(metadata['goal_coord_3d'], dtype=np.float32) # (3, )

        self.final_goal_state = None
        self.goal_episode_path = None
        self.goal_local_idx = None

        # In multi-folder mode, treat goal image index as global and resolve state accordingly.
        if self.uses_external_map_images:
            try:
                goal_state, goal_episode_dir, goal_local_idx, _ = self._resolve_agent_state_from_global_index(self.final_goal_image_idx)
                self.final_goal_state = goal_state
                self.goal_episode_path = goal_episode_dir
                self.goal_local_idx = goal_local_idx
            except Exception as e:
                logger.warning(f"Could not resolve global goal state for idx={self.final_goal_image_idx}: {e}")

        # Select goal position method
        goal_method = getattr(self.cfg, 'goal_position_method', 'trajectory')
        if goal_method == "trajectory":
            # Use agent position at goal image (simple, reliable)
            if self.final_goal_state is not None:
                self.final_goal_position = np.array(self.final_goal_state.position, dtype=np.float32)
            else:
                self.final_goal_position = np.array(
                    self.agent_states[self.final_goal_image_idx].position, dtype=np.float32
                )
        else:  # "projection"
            # Project from 3D coords (may fail if not navigable)
            try:
                self.final_goal_position = self._compute_goal_position_from_3d_coords()
                self.final_goal_position = np.array(self.final_goal_position, dtype=np.float32)
            except Exception as e:
                print(f"[WARN] Could not compute projected goal position: {e}. Using trajectory position instead.")
                if self.final_goal_state is not None:
                    self.final_goal_position = np.array(self.final_goal_state.position, dtype=np.float32)
                else:
                    self.final_goal_position = np.array(
                        self.agent_states[self.final_goal_image_idx].position, dtype=np.float32
                    )
    
    def _compute_goal_position_from_3d_coords(self):
        """Project goal from image pixel + depth to navigable 3d position."""
        if self.goal_coord_3d.size < 3:
            raise ValueError(f"Invalid goal 3D coordinates for node {self.goal_node_idx}: {self.goal_coord_3d}")

        # goal_coord_3d is (x_px, y_px, z_depth). Use depth as a proxy for goal distance and
        # place the goal along the agent's forward direction at the goal image.
        depth = float(self.goal_coord_3d[2]) 

        # Get camera pose at goal image
        if self.final_goal_state is not None:
            goal_agent_state = self.final_goal_state
        else:
            goal_agent_state = self.agent_states[self.final_goal_image_idx]
        camera_pos = np.array(goal_agent_state.position, dtype=np.float32)
        q = goal_agent_state.rotation

        # Extract forward direction from quaternion
        # Convert numpy-quaternion (w,x,y,z) to scipy format (x,y,z,w)
        q_scipy = np.array([q.x, q.y, q.z, q.w])
        R_mat = R.from_quat(q_scipy).as_matrix()
        forward = -R_mat[:, 2]
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Project and find navigable point
        search_depth = depth
        while search_depth > 0.1:
            candidate = camera_pos + forward * search_depth
            candidate[1] = camera_pos[1]  # Maintain floor height
            
            if self.sim.pathfinder.is_navigable(candidate):
                return self.sim.pathfinder.snap_point(candidate)
            
            search_depth -= 0.1
        
        raise ValueError(f'No navigable position found for node {self.goal_node_idx}')

    def _set_start_state(self):
        """
        Select and set the agent's starting state based on start_state_mode.
        
        Modes:
            - "random": Sample random navigable points with distance constraints
            - "trajectory": Select from recorded trajectory at target distance from goal
            - "fixed_idx": Use specific trajectory index (from start_idx config)
        """
        # mode = getattr(self.cfg, 'start_state_mode', 'random')
        if self.cfg.start_state_mode == "trajectory":
            mode = "trajectory"
        elif self.start_idx == -1:
            mode = "random"
        elif self.start_idx >= 0:
            mode = "fixed_idx"
        
        if mode == "random":
            # Random start state with distance constraints
            start_state = pick_random_start_state(
                sim=self.sim,
                cfg=self.cfg,
                final_goal_position=self.final_goal_position,
                agent_positions_in_map=self.agent_positions_in_map,
                max_tries=100
            )
            logger.debug(f"Random start state selected")
            
        elif mode == "trajectory":
            # Select from recorded trajectory at target distance from goal
            start_state = select_trajectory_start_state(
                sim=self.sim,
                cfg=self.cfg,
                agent_states=self.agent_states,
                goal_position=self.final_goal_position
            )
            logger.debug(f"Trajectory-based start state selected")
            
        elif mode == "fixed_idx":
            # Use specific trajectory index
            force_global_start_indices = bool(self.cfg.get("force_global_start_indices", self.uses_external_map_images))
            if self.start_idx < len(self.agent_states) and not force_global_start_indices:
                start_state = self.agent_states[self.start_idx]
                logger.debug(f"Fixed index start state: idx={self.start_idx}")
            elif self.uses_external_map_images:
                # In multi-folder mode, allow fixed index in global map space.
                start_state, source_episode_dir, local_idx, source_states = self._resolve_start_state_from_global_index(self.start_idx)
                self._set_localization_episode(source_episode_dir)
                # Use source episode trajectory for pose/gt localization.
                self.agent_states = source_states
                self.agent_positions_in_map = np.array([state.position for state in self.agent_states])
                logger.info(
                    f"Mapped global start_idx={self.start_idx} -> "
                    f"episode={source_episode_dir.name}, local_idx={local_idx}"
                )
            else:
                raise ValueError(f"start_idx {self.start_idx} out of range "
                               f"(trajectory has {len(self.agent_states)} states)")
            
        else:
            raise ValueError(f"Unknown start_state_mode: {mode}. "
                           f"Expected: random, trajectory, fixed_idx")
        
        if start_state is None:
            raise ValueError(f'Could not find valid start state for {self.episode_path}')
        
        self.agent.set_state(start_state)
        self.start_position = start_state.position
        
        logger.debug(f"Start state set: mode={mode}, position={self.start_position}")
        
        return start_state
    
    def get_goal_generator(self):

        # setup the matcher
        if self.cfg.matcher.name == "mast3r":
            if self.matcher is None:
                self.matcher = Mast3rMatcher(
                    resize_w=self.cfg.matcher.resize_w,
                    resize_h=self.cfg.matcher.resize_h,
                    geometric_verification=self.cfg.matcher.geometric_verification,
                    subsample_or_initxy1=self.cfg.matcher.subsample_or_initxy1,
                    device=self.device
                )
        elif self.cfg.matcher.name == "superpoint":
            if self.matcher is None:
                self.matcher = SuperPointMatcher(
                    resize_w=self.cfg.matcher.resize_w,
                    resize_h=self.cfg.matcher.resize_h,
                    geometric_verification=self.cfg.matcher.geometric_verification,
                    max_keypoints=self.cfg.matcher.max_keypoints,
                    keypoint_threshold=self.cfg.matcher.keypoint_threshold,
                    device=self.device
                )
        else:
            raise ValueError(f"Unknown matcher: {self.cfg.matcher.name}")
        logger.info(f"Matcher set: {self.cfg.matcher.name}")
        
        # setup the localizer
        if self.cfg.localizer.name == "topological":
            self.localizer = LocalizeTopological(
                map_img_paths=self.map_img_paths,
                H=self.H,
                W=self.W,
                matcher=self.matcher,
                cfg=self.cfg.localizer
            )
        else:
            raise ValueError(f"Unknown localizer: {self.cfg.localizer.name}")

        # setup the planner
        if self.cfg.planner.name == "topological":
            self.planner = PlanTopological(
                H=self.H,
                W=self.W,
                costmap_data=self.costmap_data,
                device=self.device,
                cfg=self.cfg.planner
            )
        else:
            raise ValueError(f"Unknown planner: {self.cfg.planner.name}")

        # setup the goal generator finally
        if self.cfg.goal_source == "topological_pixelwise":
            self.goal_generator = GoalGenerator(
                H=self.H,
                W=self.W,
                localizer=self.localizer,
                planner=self.planner,
                cfg=self.cfg
            )
        else:
            raise ValueError(f"Unknown goal generator: {self.cfg.goal_generator.name}")
    
    def get_goal(self, rgb, depth, pose=None, pts3d=None, return_vis_data=False):
        """
        Get goal mask for the current observation.
        
        Args:
            rgb: RGB observation (H, W, 3)
            depth: Depth map (H, W)
            pose: Agent pose (optional)
            pts3d: 3D points (H, W, 3) (optional)
            return_vis_data: If True, return (goal_mask, vis_data) tuple
            
        Returns:
            goal_mask: Distance-to-goal costmap (H, W)
            OR
            (goal_mask, vis_data): If return_vis_data=True, dict contains match data
        """
        # Getting the closest reference image to the particular query image
        if self.cfg.localizer.use_gt_localization:
            if pose is not None:
                localized_img_idxs, closest_map_img_idx = self.get_closest_map_img_from_odometry(
                    pose, self.localization_episode_path
                )
            else:
                localized_img_idxs, closest_map_img_idx = self.get_gt_closest_map_img()
                closest_map_img_idx = self._select_min_median_path_length(localized_img_idxs)[0]
        else:
            localized_img_idxs, closest_map_img_idx = self.get_visual_closest_map_img(rgb)
        
        # Store for logging
        self.localizedImgInds = localized_img_idxs
        self.closest_map_img_idx = closest_map_img_idx
        
        # Get goal mask
        result = self.goal_generator.get_goal_mask(
            qry_img=rgb,
            qry_depth=depth,
            qry_pts3d=pts3d,
            intrinsics=self.agent_intrinsics,
            candidate_img_indices=[closest_map_img_idx],
            return_vis_data=return_vis_data
        )

        if return_vis_data:
            self.goal_mask, vis_data = result
            self.control_input_learnt = self.goal_mask
            self.control_input_robohop = self.goal_mask
            # Add localization metadata
            vis_data['localized_img_idxs'] = localized_img_idxs  # Candidate submap
            vis_data['closest_map_img_idx'] = closest_map_img_idx  # Best match from submap
            return self.goal_mask, vis_data
        else:
            self.goal_mask = result
            self.control_input_learnt = self.goal_mask
            self.control_input_robohop = self.goal_mask
            return self.goal_mask

    def get_gt_closest_map_img(self):
        """Get ground truth closest map images based on position"""
        use_global_pose_loc = self.cfg.localizer.get("global_pose_localization", self.uses_external_map_images)
        if use_global_pose_loc and len(self.global_pose_valid_indices) > 0:
            qry_pos = np.array(self.agent.get_state().position, dtype=np.float32)
            poses = self.global_pose_positions[self.global_pose_valid_indices]
            dists = np.linalg.norm(poses - qry_pos, axis=1)
            top_k = 2 * self.loc_radius
            order = np.argsort(dists)[:top_k]
            closest_idxs = [int(self.global_pose_valid_indices[i]) for i in order][::self.subsample_ref]
            closest_idx = int(self.global_pose_valid_indices[int(np.argmin(dists))])
            logger.info(f"Top K closest idxs (global): {closest_idxs = }")
            return closest_idxs, closest_idx

        dists = np.linalg.norm(
            self.agent_positions_in_map - self.agent.get_state().position, axis=1)
        
        top_k = 2 * self.loc_radius
        closest_local = np.argsort(dists)[:top_k]
        closest_local = sorted(closest_local)[::self.subsample_ref]

        if self.episode_global_map_indices and len(self.episode_global_map_indices) >= len(dists):
            closest_idxs = [self.episode_global_map_indices[i] for i in closest_local]
            closest_idx = self.episode_global_map_indices[int(np.argmin(dists))]
        else:
            closest_idxs = closest_local
            closest_idx = int(np.argmin(dists))

        logger.info(f"Top K closest idxs: {closest_idxs = }")
        return closest_idxs, closest_idx

    def get_closest_map_img_from_odometry(self, odom_pose, episode_path, 
                                          position_weight=1.0, rotation_weight=1.0):
        """
        Find closest map image using odometry pose (x,y,z,qx,qy,qz,qw).
        Combines translation and rotation distances.
        
        Returns:
            closest_idx: Index of closest image
            localized_img_inds: List of candidate indices (topK, subsampled)
        """
        self._set_localization_episode(Path(episode_path))

        use_global_pose_loc = self.cfg.localizer.get("global_pose_localization", self.uses_external_map_images)
        if use_global_pose_loc and len(self.global_pose_valid_indices) > 0:
            candidate_global_idxs = self.global_pose_valid_indices
            poses = self.global_pose_positions[candidate_global_idxs]
            quats = self.global_pose_quats[candidate_global_idxs]
        else:
            odom_file = Path(episode_path) / 'poses_odom.txt'
            if not odom_file.exists():
                raise FileNotFoundError(f"{odom_file} does not exist.")

            # Load odometry poses for current episode only.
            poses = []
            quats = []
            with open(odom_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    poses.append([x, y, z])
                    quats.append([qx, qy, qz, qw])

            poses = np.array(poses)
            quats = np.array(quats)

        # pos_query = np.array(odom_pose[:3])
        # quat_query = np.array(odom_pose[3:7])
        pos_query = np.array([odom_pose.position[0], odom_pose.position[1], odom_pose.position[2]])
        q = odom_pose.rotation
        quat_query = np.array([q.x, q.y, q.z, q.w])

        # Translation distances
        trans_dists = np.linalg.norm(poses - pos_query, axis=1)

        # Rotation distances (quaternion angle)
        def quat_angle(q1, q2):
            dot = np.abs(np.sum(q1 * q2, axis=-1))
            dot = np.clip(dot, -1.0, 1.0)
            return 2 * np.arccos(dot)

        rot_dists = quat_angle(quats, quat_query)
        total_dists = position_weight * trans_dists + rotation_weight * rot_dists

        closest_local_idx = int(np.argmin(total_dists))
        
        # Get topK and subsample
        topK = 2 * self.loc_radius
        sorted_local_idxs = np.argsort(total_dists)[:topK]
        localized_local_inds = sorted(sorted_local_idxs.tolist())[::self.subsample_ref]

        if use_global_pose_loc and len(self.global_pose_valid_indices) > 0:
            localized_img_inds = [int(candidate_global_idxs[i]) for i in localized_local_inds]
            closest_idx = int(candidate_global_idxs[closest_local_idx])
        # Map local episode indices to global map indices for multi-folder maps.
        elif self.episode_global_map_indices:
            if len(self.episode_global_map_indices) < len(total_dists):
                raise ValueError(
                    "Not enough episode-global map index mappings for odometry localization: "
                    f"mappings={len(self.episode_global_map_indices)}, odom_poses={len(total_dists)}"
                )
            localized_img_inds = [self.episode_global_map_indices[i] for i in localized_local_inds]
            closest_idx = self.episode_global_map_indices[closest_local_idx]
        else:
            localized_img_inds = localized_local_inds
            closest_idx = closest_local_idx

        # max_rot_threshold = np.deg2rad(90)  # 90 degrees
        # localized_img_inds = [idx for idx in localized_img_inds
        #                       if rot_dists[idx] <= max_rot_threshold]
                
        # Optional reranking: by default this keeps previous behavior, but can be
        # disabled to use strict nearest-pose localization.
        rerank_with_costmap = self.cfg.localizer.get("odom_rerank_with_costmap", True)
        if rerank_with_costmap:
            # localized_img_inds = self._select_min_median_path_length(localized_img_inds)
            localized_img_inds = self._select_min_bottom_percent_median_path_length(localized_img_inds, percent=15)
            closest_idx = localized_img_inds[0]
        
        return localized_img_inds, closest_idx
    
    def get_visual_closest_map_img(self, rgb):
        """Get closest map image using visual matching"""
        current_image = rgb[:, :, :3]
        match_scores = []

        # Match against global map image indices directly.
        candidate_idxs = list(range(len(self.map_img_paths)))
        
        for idx in candidate_idxs[::5]:
            map_image = cv2.imread(str(self.map_img_paths[idx]))[:, :, ::-1]

            # Ensure positive strides
            qry_img = self.matcher.load_image(current_image.copy())
            ref_img = self.matcher.load_image(map_image.copy())
            result = self.matcher(qry_img, ref_img)
            inlier_count = result['num_inliers']
            logger.info(f"MASt3R Found {inlier_count} inliers with Reference Image {idx}")

            match_scores.append({'idx': idx, 'inliers': inlier_count})

        # Sort by inliers
        top_k = 2 * self.loc_radius
        sorted_matches = sorted(match_scores, key=lambda x: x['inliers'], reverse=True)[:top_k]

        best_goal_idx = sorted_matches[0]['idx']
        localized_img_idxs = [match['idx'] for match in sorted_matches][::self.subsample_ref]

        localized_img_idxs = self._select_min_median_path_length(localized_img_idxs)
        closest_idx = localized_img_idxs[0]
        
        return localized_img_idxs, closest_idx

    def _select_min_median_path_length(self, candidate_img_indices):
        """Select candidate with minimum median path length to goal"""
        min_median_path_length = 100
        best_ref_img_idx = None
        img_costmaps = self.costmap_data.get_costmap()
        
        for ref_img_idx in candidate_img_indices:
            img_pls = img_costmaps[ref_img_idx]
            median_path_length = np.median(img_pls)
            logger.info(f"Image {ref_img_idx} has {median_path_length} median path length")
            
            if median_path_length < min_median_path_length:
                min_median_path_length = median_path_length
                best_ref_img_idx = ref_img_idx
        
        if best_ref_img_idx is not None:
            logger.info(f"Selected Image {best_ref_img_idx} with {min_median_path_length} median path length")
            return [best_ref_img_idx]
        else:
            logger.warning(f"No good matches found, using 0th index: {candidate_img_indices[0]}")
            return [candidate_img_indices[0]]
        
    def _select_min_bottom_percent_median_path_length(self, candidate_indices, percent=5):
        """
        For each candidate reference image, gather all node path-lengths belonging to that image,
        take the bottom `percent` smallest values, compute their median, and select the image with
        the minimum median of that bottom set.
        Returns a single-element list [best_ref_img_id] (consistent with _select_min_median_path_length).
        """
        min_median_path_length = 100
        best_ref_img_idx = None
        img_costmaps = self.costmap_data.get_costmap()
        
        for ref_img_idx in candidate_indices:
            img_pls = img_costmaps[ref_img_idx]
            k = max(1, int(len(img_pls) * percent / 100))
            bottom_k_pls = np.partition(img_pls, k)[:k]
            median_path_length = np.median(bottom_k_pls)
            # logger.info(f"Image {ref_img_idx} has {median_path_length} median of bottom {percent}% path lengths")
            
            if median_path_length < min_median_path_length:
                min_median_path_length = median_path_length
                best_ref_img_idx = ref_img_idx
        
        if best_ref_img_idx is not None:
            logger.info(f"Selected Image {best_ref_img_idx} with {min_median_path_length} median of bottom {percent}% path lengths")
            return [best_ref_img_idx]
        else:
            logger.warning(f"No good matches found, using 0th index: {candidate_indices[0]}")
            return [candidate_indices[0]]
    
    def get_control_signal(self, step, rgb, depth):
        """Get control signal from controller"""
        control_method = self.cfg.controller.name
        if control_method == 'learnt':
            if self.control_input_learnt[0] is None or self.control_input_learnt[1] is None:
                self.velocity_control, self.theta_control, self.vis_img = 0, 0, self.vis_img_default.copy()
            else:
                self.velocity_control, self.theta_control, self.vis_img = self.goal_controller.predict(
                    rgb, self.control_input_learnt)
            
            self.controller_logs = self.goal_controller.controller_logs
            # NOTE: In simulation, theta is NOT negated here.
            # It's only negated for real robot (env != 'sim').
            # The negation for sim happens in execute_action with steer=-self.theta_control
        else:
            raise NotImplementedError(f"{control_method} is not available...")
    
    def execute_action(self):
        """Execute control action in simulator"""
        control_method = self.cfg.controller.name
        if control_method == 'learnt':
            self.agent, self.sim, self.collided = apply_velocity(
                vel_control=self.vel_control,
                agent=self.agent,
                sim=self.sim,
                velocity=self.velocity_control,
                steer=-self.theta_control,  # opposite y axis
                time_step=self.time_delta
            )  # will add velocity control once steering is working
        else:
            raise NotImplementedError("Other controller methods task not implemented yet.")
        
        self.agent_state_history.append(self.agent.get_state())
    
    def is_done(self):
        """Check if goal is reached"""
        done = False
        current_robot_state = self.agent.get_state()  # world coordinates
        self.distance_to_goal = find_shortest_path(
            self.sim, p1=current_robot_state.position, p2=self.final_goal_position)[0]
        if self.distance_to_goal <= self.cfg.goal_distance_threshold:
            logger.info(
                f'\nWinner! dist to goal: {self.distance_to_goal:.6f}\n')
            self.success_status = 'success'
            done = True
        elif self.cfg.goal_position_method != 'trajectory' and self.distance_to_goal <= self.cfg.goal_distance_threshold * 2:  # close enough to check goal image success
            # check if distance to goal image is within threshold
            distance_to_goal_image = np.linalg.norm(
                current_robot_state.position - self.agent_states[self.final_goal_image_idx].position)
            if distance_to_goal_image <= self.cfg.goal_distance_threshold / 2:  # smaller threshold for image-based success
                logger.info(
                    f'\nWinner by reaching goal image! dist to goal image: {distance_to_goal_image:.6f}\n')
                self.success_status = 'success'
                done = True
        return done
    
    def setup_logging(self):
        """Initialize logging files and directories"""
        self.episode_metadata_filepath = self.episode_results_path / 'metadata.txt'
        self.episode_results_csv = self.episode_results_path / 'results.csv'

        # Initialize results files
        initialize_results(
            metadata_file=self.episode_metadata_filepath,
            results_csv=self.episode_results_csv,
            method=self.cfg.controller.name,
            goal_source=self.cfg.goal_source,
            max_steps=self.cfg.max_steps,
            goal_distance_threshold=self.cfg.goal_distance_threshold,
            pid_steer_values=self.pid_steer_values,
            hfov_degrees=self.fov_deg,
            time_delta=self.time_delta,
            velocity_control=self.velocity_control,
            goal_position=self.final_goal_position,
        )

        # Initialize results dictionary for accumulating per-step data
        results_dict_keys = [
            "step",
            "distance_to_goal",
            "velocity_control",
            "theta_control",
            "collided",
            "discrete_action",
            "agent_states",
            "controller_logs",
        ]
        self.results_dict = {k: [] for k in results_dict_keys}

        # Create visualization directories
        if self.cfg.goal_source.lower() == 'topological_pixelwise':
            self.dirname_observed_rgb = self.episode_results_path / 'observed_rgb'
            self.dirname_goal_masks = self.episode_results_path / 'goal_masks'
            self.path_grid_match_viz = self.episode_results_path / 'grid_match_viz'

            self.dirname_observed_rgb.mkdir(exist_ok=True, parents=True)
            self.dirname_goal_masks.mkdir(exist_ok=True, parents=True)
            self.path_grid_match_viz.mkdir(exist_ok=True, parents=True)

    def log_results(self, step: int, final: bool = False) -> None:
        """Log per-step or final results to files"""
        if not final:
            # Write per-step results to CSV
            write_results(
                results_csv=self.episode_results_csv,
                step=step,
                current_robot_state=self.agent.get_state() if self.agent is not None else None,
                distance_to_goal=self.distance_to_goal,
                velocity_control=self.velocity_control,
                theta_control=self.theta_control,
                collided=self.collided,
                discrete_action=self.discrete_action
            )
            
            # Accumulate results
            results_dict_curr = {
                "step": step,
                "distance_to_goal": self.distance_to_goal,
                "velocity_control": self.velocity_control,
                "theta_control": self.theta_control,
                "collided": self.collided,
                "discrete_action": self.discrete_action,
                "agent_states": self.agent.get_state() if self.agent is not None else None,
                "controller_logs": self.controller_logs[-1] if self.controller_logs is not None and len(self.controller_logs) > 0 else None,
            }
            self.update_results_dict(results_dict_curr)
        else:
            # Write final metadata
            write_final_meta_results(
                metadata_file=self.episode_metadata_filepath,
                success_status=self.success_status,
                final_distance=self.distance_to_goal,
                step=step,
                distance_to_final_goal=self.distance_to_final_goal
            )
            
            # Save accumulated results
            np.savez(
                self.episode_results_path / 'results_dict.npz',
                **self.results_dict
            )
    
    def update_results_dict(self, curr_dict: dict) -> None:
        """Append current step's data to results dictionary"""
        for k, v in curr_dict.items():
            self.results_dict[k].append(v)

    def create_top_down_map(self, height=None, meters_per_pixel=0.025):
        """Create top-down map visualization"""
        if height is None:
            scene_bb = self.sim.get_active_scene_graph().get_root_node().cumulative_bb
            height = scene_bb.y().min

        top_down_map = maps.get_topdown_map(
            self.sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        tdv = top_down_map
        tdv_dims = (tdv.shape[0], tdv.shape[1])

        return tdv, tdv_dims

    def save_topdown_map(self, step, history=None):
        """Save top-down map with trajectory"""
        if history is None:
            history = self.agent_state_history

        tdv, tdv_dims = self.create_top_down_map(
            height=self.start_position[1],
            meters_per_pixel=0.025,
        )

        if len(history) < 1:
            return

        path_xyz = np.array([s.position for s in history])
        grid_path = np.array([_sim_to_grid(self.sim, tdv_dims, p) for p in path_xyz])

        # Draw trajectory
        if len(grid_path) > 1:
            for i in range(len(grid_path)-1):
                p1 = (grid_path[i][1], grid_path[i][0])
                p2 = (grid_path[i+1][1], grid_path[i+1][0])
                cv2.line(tdv, p1, p2, (255, 0, 255), 2)

        # Start
        s = grid_path[0]
        cv2.circle(tdv, (s[1], s[0]), 6, (255,255,255), -1)
        cv2.circle(tdv, (s[1], s[0]), 4, (0,0,255), -1)

        # Goal
        g = _sim_to_grid(self.sim, tdv_dims, self.final_goal_position)
        cv2.drawMarker(tdv, (g[1], g[0]), (0,255,0), cv2.MARKER_TILTED_CROSS, 18, 2)

        # Agent heading
        curr = history[-1]
        p = grid_path[-1]
        heading = _quat_to_heading(curr.rotation)
        maps.draw_agent(tdv, p, heading, agent_radius_px=6)

        outdir = self.episode_results_path / "top_down_maps"
        outdir.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(outdir / f"step_{step:05d}.png"),
                    cv2.cvtColor(tdv, cv2.COLOR_RGB2BGR))

    def plot_heatmap_with_colorbar(self, image, values, coordinates=None, step=0,
                                   alpha=0.6, cmap='turbo',
                                   save_path=None, dpi=150):
        """
        Plot heatmap visualization of goal mask with both relative and absolute scales.
        Saves two versions: one with relative vmin/vmax and one with absolute vmax.
        """
        h, w = image.shape[:2]
        heatmap = np.full((h, w), fill_value=100, dtype=np.float32)

        if coordinates is None:
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            coordinates = list(zip(y_coords.flatten(), x_coords.flatten()))

        for (y, x), val in zip(coordinates, values):
            heatmap[y, x] = val

        # Compute scales
        vmin_rel = np.min(values)
        vmax_rel = np.max(values)

        # Track absolute maximum
        if not hasattr(self, 'goal_mask_absolute_max') or self.goal_mask_absolute_max is None:
            self.goal_mask_absolute_max = vmax_rel
        else:
            self.goal_mask_absolute_max = max(self.goal_mask_absolute_max, vmax_rel)
        vmax_abs = self.goal_mask_absolute_max

        # Save relative scale version
        display_heatmap_rel = np.clip(heatmap.copy(), vmin_rel, vmax_rel)
        fig_rel, ax_rel = plt.subplots(figsize=(10, 8))
        heat_rel = ax_rel.imshow(display_heatmap_rel, cmap=cmap, vmin=vmin_rel, vmax=vmax_rel)
        ax_rel.set_title(f"Step {step} | RELATIVE vmin={vmin_rel:.2f}, vmax={vmax_rel:.2f}", fontsize=12)
        cbar_rel = fig_rel.colorbar(heat_rel, ax=ax_rel, fraction=0.03, pad=0.01)
        cbar_rel.set_label('Distance to Goal', fontsize=12)
        ax_rel.axis('off')
        plt.tight_layout()
        
        if save_path:
            rel_path = str(save_path).replace('.png', '_rel.png')
            fig_rel.savefig(rel_path, dpi=dpi, facecolor='white')
            logger.info(f"Saved RELATIVE heatmap to {rel_path}")
        plt.close(fig_rel)

        # Save absolute scale version
        display_heatmap_abs = np.clip(heatmap.copy(), vmin_rel, vmax_abs)
        fig_abs, ax_abs = plt.subplots(figsize=(10, 8))
        heat_abs = ax_abs.imshow(display_heatmap_abs, cmap=cmap, vmin=vmin_rel, vmax=vmax_abs)
        ax_abs.set_title(f"Step {step} | ABSOLUTE vmax={vmax_abs:.2f}", fontsize=12)
        cbar_abs = fig_abs.colorbar(heat_abs, ax=ax_abs, fraction=0.03, pad=0.01)
        cbar_abs.set_label('Distance to Goal', fontsize=12)
        ax_abs.axis('off')
        plt.tight_layout()
        
        if save_path:
            abs_path = str(save_path).replace('.png', '_abs.png')
            fig_abs.savefig(abs_path, dpi=dpi, facecolor='white')
            logger.info(f"Saved ABSOLUTE heatmap to {abs_path}")
        plt.close(fig_abs)

def init_results_dir_and_save_cfg(cfg: DictConfig, default_logger=None):
    """Initialize results directory and save configuration"""
    # Build results path from config
    results_path = Path(cfg.results_dirpath) if cfg.results_dirpath.startswith('/') else Path.cwd() / cfg.results_dirpath

    # Create structured folder path
    task_str = cfg.task_type
    if cfg.get('reverse', False):
        task_str += '_reverse'
    
    results_dirpath = (results_path / task_str / cfg.exp_name /
    f'{datetime.now().strftime("%Y%m%d-%H-%M-%S")}_{cfg.controller.name}_{cfg.goal_source}')
    results_dirpath.mkdir(exist_ok=True, parents=True)

    # Update logger file handler
    if default_logger is not None:
        default_logger.update_file_handler_root(results_dirpath / 'output.log')
    
    logger.info(f'Logging to {results_dirpath}')

    # Save config
    OmegaConf.save(cfg, results_dirpath / 'config.yaml')
    return results_dirpath