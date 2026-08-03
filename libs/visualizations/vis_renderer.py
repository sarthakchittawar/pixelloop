"""
Visualization Rendering Module

Creates rendered visualizations from raw data.
Can be used online (during navigation) or offline (batch processing).
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Any
from omegaconf import DictConfig
import logging
import magnum as mn
from habitat.utils.visualizations import maps

logger = logging.getLogger("[VisualizationRenderer]")


class VisualizationRenderer:
    """
    Handles creation of rendered visualizations from raw data.
    
    Separate from VisualizationDataCollector to decouple:
    - Raw data storage (fast, always happens)
    - Visualization rendering (optional, can be slow)
    """
    
    def __init__(self, vis_cfg: DictConfig, episode_results_path: Path, output_subdir: str = "visualizations"):
        """
        Initialize renderer.
        
        Args:
            vis_cfg: Visualization config (cfg.visualization section only)
            episode_results_path: Path to episode results
            output_subdir: Subdirectory name for output (default: "visualizations")
        """
        self.vis_cfg = vis_cfg
        self.episode_results_path = Path(episode_results_path)
        self.vis_dir = self.episode_results_path / output_subdir
        
        # Check if online rendering is enabled
        self.enabled = vis_cfg.render_visualizations.enabled
        self.online_render = vis_cfg.render_visualizations.online
        
        # Topdown map state (cached per episode)
        self._topdown_base_map = None
        self._topdown_dims = None
        self._sim = None  # Will be set when rendering topdown
        self._start_position = None
        self._goal_position = None
        
        # Offline mode support
        self._offline_mode = False
        self._pathfinder_bounds = None  # (lower, upper) tuple for offline coordinate conversion
        
        logger.info(f"VisualizationRenderer initialized: enabled={self.enabled}, online_render={self.online_render}")
    
    def init_offline_mode(self, pathfinder_bounds: tuple):
        """
        Initialize offline rendering mode.
        
        Args:
            pathfinder_bounds: Tuple of (lower_bound, upper_bound) numpy arrays
        """
        self._offline_mode = True
        self._pathfinder_bounds = pathfinder_bounds
        logger.info("Offline mode enabled for visualization rendering")
    
    def render_step_visualizations(
        self,
        step: int,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        costmap: Optional[np.ndarray] = None,
        matches_data: Optional[dict] = None,
        ref_img_path: Optional[Path] = None,
        sim = None,
        trajectory_history: Optional[List[Any]] = None,
        start_position: Optional[np.ndarray] = None,
        goal_position: Optional[np.ndarray] = None,
        waypoints: Optional[np.ndarray] = None,
        waypoints_adjusted: Optional[np.ndarray] = None,
        care_data: Optional[dict] = None,
        ref_img: Optional[np.ndarray] = None,
        ref_img_idx: Optional[int] = None,
        agent_position: Optional[np.ndarray] = None,
        agent_rotation: Optional[np.ndarray] = None,
    ):
        """
        Render visualizations for a single step.
        
        Args:
            step: Step index
            rgb: RGB image (H, W, 3)
            depth: Depth map (H, W) in meters for topdown obstacle visualization
            costmap: Costmap array (H, W)
            matches_data: Dict with match data
            ref_img_path: Path to reference image for match visualization
            sim: Habitat simulator instance (for topdown map)
            trajectory_history: List of agent states (for topdown map)
            start_position: Start position [x, y, z] (for topdown map)
            goal_position: Goal position [x, y, z] (for topdown map)
            waypoints: Original predicted waypoints (N, 2) for combined viz
            waypoints_adjusted: CARE-adjusted waypoints (N, 2) if CARE is enabled
            care_data: CARE collision avoidance data dict (obstacles, theta_rot, etc.)
            ref_img: Reference image for combined viz
            ref_img_idx: Reference image index for combined viz
            agent_position: Current agent 3D position [x, y, z]
            agent_rotation: Current agent rotation quaternion
        """
        if not self.online_render:
            return
        
        # Ensure visualization directories exist
        self._ensure_vis_dirs()
        
        # Render each visualization type based on config
        if rgb is not None and self.vis_cfg.render_visualizations.rgb_images:
            self._save_rgb_visualization(step, rgb)
        
        if costmap is not None and self.vis_cfg.render_visualizations.costmap_heatmaps:
            self._render_costmap_heatmap(step, rgb, costmap)
        
        if matches_data is not None and ref_img_path is not None and self.vis_cfg.render_visualizations.match_vis:
            pass  # Match visualization removed - to be reimplemented later
        
        # Render topdown map if trajectory history is provided
        if trajectory_history is not None:
            self.render_topdown_map(step, trajectory_history, sim, start_position, goal_position)
        
        # Render combined step visualization
        if rgb is not None and getattr(self.vis_cfg.render_visualizations, 'combined_step_vis', False):
            self.render_combined_step_vis(
                step=step,
                rgb=rgb,
                costmap=costmap,
                waypoints=waypoints,
                waypoints_adjusted=waypoints_adjusted,
                care_data=care_data,
                matches_data=matches_data,
                ref_img=ref_img,
                ref_img_idx=ref_img_idx,
                agent_position=agent_position,
                agent_rotation=agent_rotation,
            )
        
        # Render waypoint + topdown visualization
        if waypoints is not None and getattr(self.vis_cfg.render_visualizations, 'waypoint_topdown_vis', False):
            self.render_waypoint_topdown_vis(
                step=step,
                waypoints=waypoints,
                waypoints_adjusted=waypoints_adjusted,
                care_data=care_data,
                depth=depth,
            )
    
    def _ensure_vis_dirs(self):
        """Create visualization subdirectories."""
        (self.vis_dir / "costmaps").mkdir(parents=True, exist_ok=True)
        (self.vis_dir / "matches").mkdir(parents=True, exist_ok=True)
        (self.vis_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (self.vis_dir / "topdown").mkdir(parents=True, exist_ok=True)
    
    def _save_rgb_visualization(self, step: int, rgb: np.ndarray):
        """
        Save RGB image to centralized visualization directory for easy browsing.
        
        Args:
            step: Step index
            rgb: RGB image (H, W, 3)
        """
        save_path = self.vis_dir / "rgb" / f"step_{step:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        logger.debug(f"Saved RGB visualization: {save_path}")
    
    def _render_costmap_heatmap(self, step: int, rgb: np.ndarray, costmap: np.ndarray):
        """
        Render costmap as heatmap with colorbar.
        
        Similar to sg_habitat's create_heatmaps.py but simplified.
        """
        save_path = self.vis_dir / "costmaps" / f"step_{step:04d}.png"
        
        # Create heatmap visualization
        h, w = costmap.shape
        
        # Mask out invalid values
        heatmap = np.where(costmap >= 99, np.nan, costmap)
        valid_costs = heatmap[~np.isnan(heatmap)]
        
        if valid_costs.size > 0:
            vmin, vmax = np.percentile(valid_costs, [5, 95])
        else:
            vmin, vmax = 0, 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap_obj = plt.get_cmap('turbo').copy()
        cmap_obj.set_bad(color='white')
        
        im = ax.imshow(heatmap, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance to Goal', fontsize=12)
        
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        logger.debug(f"Rendered costmap heatmap: {save_path}")
    
    # =============================================
    # Top-down Map Visualization Methods
    # =============================================
    
    def _quat_to_heading(self, q) -> float:
        """
        Convert quaternion to heading angle.
        
        Args:
            q: Quaternion (habitat quaternion with .scalar or numpy array [w, x, y, z])
        
        Returns:
            Heading angle in radians
        """
        if isinstance(q, np.ndarray):
            # numpy array [w, x, y, z]
            quat = mn.Quaternion(mn.Vector3(float(q[1]), float(q[2]), float(q[3])), float(q[0]))
            R = np.array(quat.to_matrix())
        else:
            # habitat quaternion object (has .imag and .real)
            R = np.array(mn.Quaternion(q.imag, q.real).to_matrix())
        R_bc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R = R @ R_bc
        return np.arctan2(R[0, 2], R[2, 2])
    
    def _sim_to_grid(self, tdv_dims, pos) -> tuple:
        """
        Convert simulation coordinates to grid coordinates.
        
        Args:
            tdv_dims: (height, width) of topdown view
            pos: 3D position [x, y, z]
        
        Returns:
            (row, col) grid coordinates
        """
        if self._offline_mode:
            # Use stored bounds directly (same math as maps.to_grid)
            lower, upper = self._pathfinder_bounds
            grid_size = (
                abs(upper[2] - lower[2]) / tdv_dims[0],
                abs(upper[0] - lower[0]) / tdv_dims[1],
            )
            grid_x = int((pos[2] - lower[2]) / grid_size[0])
            grid_y = int((pos[0] - lower[0]) / grid_size[1])
            return (grid_x, grid_y)
        else:
            return maps.to_grid(pos[2], pos[0], tdv_dims, pathfinder=self._sim.pathfinder)
    
    def _create_topdown_base_map(self, height: float, meters_per_pixel: float = 0.025):
        """
        Create base topdown map showing navigable areas.
        
        Args:
            height: Height level to slice the map at
            meters_per_pixel: Map resolution
            
        Returns:
            (tdv, tdv_dims): RGB topdown view and (height, width) dimensions
        """
        top_down_map = maps.get_topdown_map(
            self._sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        # Recolor: 0=navigable->white, 1=border->gray, 2=obstacle->black
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        tdv = recolor_map[top_down_map]
        tdv_dims = (tdv.shape[0], tdv.shape[1])
        return tdv, tdv_dims
    
    def render_topdown_map(
        self,
        step: int,
        trajectory_history: List[Any],
        sim=None,
        start_position: Optional[np.ndarray] = None,
        goal_position: Optional[np.ndarray] = None
    ):
        """
        Render topdown map with trajectory overlay.
        
        Args:
            step: Current step index
            trajectory_history: List of agent states (with .position, .rotation)
            sim: Habitat simulator instance (optional in offline mode)
            start_position: Start position [x, y, z] (optional in offline mode)
            goal_position: Goal position [x, y, z] (optional in offline mode)
        """
        if not self.vis_cfg.render_visualizations.topdown_trajectory:
            return
        
        self._ensure_vis_dirs()
        
        # Cache sim reference if provided
        if sim is not None:
            self._sim = sim
        
        # Create base map if not cached (only in online mode)
        if self._topdown_base_map is None and sim is not None:
            height = start_position[1]  # Use start height for map slice
            meters_per_pixel = getattr(
                self.vis_cfg.render_visualizations, 
                'topdown_meters_per_pixel', 0.025
            )
            self._topdown_base_map, self._topdown_dims = self._create_topdown_base_map(
                height, meters_per_pixel
            )
            self._start_position = start_position
            self._goal_position = goal_position
            logger.info(f"Created topdown base map: {self._topdown_dims}")
        
        if len(trajectory_history) < 1:
            return
        
        # Copy base map to draw on
        tdv = self._topdown_base_map.copy()
        
        # Get trajectory path in grid coordinates
        path_xyz = np.array([s.position for s in trajectory_history])
        grid_path = np.array([self._sim_to_grid(self._topdown_dims, p) for p in path_xyz])
        
        # Draw trajectory line (magenta)
        if len(grid_path) > 1:
            for i in range(len(grid_path) - 1):
                p1 = (int(grid_path[i][1]), int(grid_path[i][0]))
                p2 = (int(grid_path[i + 1][1]), int(grid_path[i + 1][0]))
                cv2.line(tdv, p1, p2, (255, 0, 255), 2)
        
        # Draw start position (red circle with white border)
        s = self._sim_to_grid(self._topdown_dims, self._start_position)
        cv2.circle(tdv, (int(s[1]), int(s[0])), 6, (255, 255, 255), -1)
        cv2.circle(tdv, (int(s[1]), int(s[0])), 4, (0, 0, 255), -1)
        
        # Draw goal position (green cross marker)
        g = self._sim_to_grid(self._topdown_dims, self._goal_position)
        cv2.drawMarker(tdv, (int(g[1]), int(g[0])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
        
        # Draw agent with heading
        curr = trajectory_history[-1]
        p = grid_path[-1]
        heading = self._quat_to_heading(curr.rotation)
        maps.draw_agent(tdv, p, heading, agent_radius_px=6)
        
        # Save
        save_path = self.vis_dir / "topdown" / f"step_{step:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(tdv, cv2.COLOR_RGB2BGR))
        logger.debug(f"Rendered topdown map: {save_path}")
    
    # =============================================
    # Combined Step Visualization Methods
    # =============================================
    
    def _create_match_panel(
        self, 
        rgb: np.ndarray, 
        matches_data: Optional[dict] = None,
        ref_img: Optional[np.ndarray] = None,
        ref_img_idx: Optional[int] = None,
        n_samples: int = 20
    ) -> np.ndarray:
        """
        Create match visualization panel with query and reference images.
        
        Args:
            rgb: Query RGB image
            matches_data: Dict with 'match_pairs', 'match_confidences'
            ref_img: Reference image (same size as rgb)
            ref_img_idx: Reference image index for title
            n_samples: Max matches to draw
            
        Returns:
            Panel image as numpy array
        """
        H, W = rgb.shape[:2]
        header_height = 40
        
        # If no matches or ref image, just show query
        if matches_data is None or ref_img is None:
            # Create header
            header = np.ones((header_height, W, 3), dtype=np.uint8) * 255
            cv2.putText(header, "Query Image (no matches)", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            return np.vstack([header, rgb])
        
        # Resize ref_img to match query
        ref_img = cv2.resize(ref_img, (W, H))
        
        # Create side-by-side
        combined_width = W * 2
        header = np.ones((header_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Get match data - handle explicit None values
        # Keys: qry_mkpts (N,2), ref_mkpts (N,2), confidences (N,)
        qry_mkpts = matches_data.get('qry_mkpts')
        ref_mkpts = matches_data.get('ref_mkpts')
        confidences = matches_data.get('confidences')
        
        if qry_mkpts is None:
            qry_mkpts = np.array([]).reshape(0, 2)
        if ref_mkpts is None:
            ref_mkpts = np.array([]).reshape(0, 2)
        if confidences is None:
            confidences = np.array([])
        
        # Ensure keypoints are 2D and confidences is 1D
        if qry_mkpts.ndim == 1 and len(qry_mkpts) > 0:
            qry_mkpts = qry_mkpts.reshape(-1, 2)
        if ref_mkpts.ndim == 1 and len(ref_mkpts) > 0:
            ref_mkpts = ref_mkpts.reshape(-1, 2)
        confidences = confidences.flatten() if hasattr(confidences, 'flatten') else np.array(confidences).flatten()
        
        n_total = len(confidences)
        n_disp = min(n_samples, n_total)
        
        # Title with stats (matching old codebase format)
        title = f"Query Img vs. Ref Img {ref_img_idx if ref_img_idx is not None else '?'} | Matches (Uniformly Sampled): {n_disp}/{n_total}"
        cv2.putText(header, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        if n_total > 0:
            stats = f"Conf Range: [{confidences.min():.2f}, {confidences.max():.2f}] | Mean: {confidences.mean():.2f}"
            cv2.putText(header, stats, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Side-by-side image
        side_by_side = np.hstack([rgb, ref_img])
        panel = np.vstack([header, side_by_side])
        
        # Sample and draw matches
        if n_total > 0 and len(qry_mkpts) > 0:
            if n_total > n_samples:
                indices = np.linspace(0, n_total - 1, n_samples, dtype=int)
            else:
                indices = np.arange(n_total)
            
            sampled_qry = qry_mkpts[indices]
            sampled_ref = ref_mkpts[indices]
            sampled_conf = confidences[indices]  # Already flattened earlier
            
            # Normalize for colormap
            cmap = plt.get_cmap('turbo')
            if sampled_conf.max() == sampled_conf.min():
                norm_conf = np.ones(len(sampled_conf)) * 0.5
            else:
                norm_conf = (sampled_conf - sampled_conf.min()) / (sampled_conf.max() - sampled_conf.min())
            
            # Draw match lines - keypoints are (x, y) pixel coordinates
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(sampled_qry)):
                qry_pt = sampled_qry[i]
                ref_pt = sampled_ref[i]
                conf_val = sampled_conf[i]
                
                # Get normalized confidence - handle numpy arrays properly
                nc = norm_conf[i]
                if hasattr(nc, 'item'):
                    nc = nc.item()
                else:
                    nc = float(nc)
                
                # Get actual confidence value for annotation
                if hasattr(conf_val, 'item'):
                    conf_val = conf_val.item()
                else:
                    conf_val = float(conf_val)
                
                # Query point on left image
                qry_x = int(qry_pt[0].item() if hasattr(qry_pt[0], 'item') else float(qry_pt[0]))
                qry_y = int(qry_pt[1].item() if hasattr(qry_pt[1], 'item') else float(qry_pt[1]))
                # Reference point on right image (offset by W)
                ref_x = int(ref_pt[0].item() if hasattr(ref_pt[0], 'item') else float(ref_pt[0])) + W
                ref_y = int(ref_pt[1].item() if hasattr(ref_pt[1], 'item') else float(ref_pt[1]))
                
                # Offset for header
                qry_viz = (qry_x, qry_y + header_height)
                ref_viz = (ref_x, ref_y + header_height)
                
                # Get color from colormap (BGR)
                color_rgba = cmap(nc)
                color_bgr = (int(color_rgba[2] * 255), int(color_rgba[1] * 255), int(color_rgba[0] * 255))
                
                cv2.circle(panel, qry_viz, 3, color_bgr, -1)
                cv2.circle(panel, ref_viz, 3, color_bgr, -1)
                cv2.line(panel, qry_viz, ref_viz, color_bgr, 1)
                
                # Add confidence text at midpoint
                mid_x = (qry_viz[0] + ref_viz[0]) // 2
                mid_y = (qry_viz[1] + ref_viz[1]) // 2
                text = f'{conf_val:.2f}'
                (tw, th), _ = cv2.getTextSize(text, font, 0.3, 1)
                cv2.rectangle(panel, (mid_x, mid_y - th), (mid_x + tw, mid_y), color_bgr, -1)
                cv2.putText(panel, text, (mid_x, mid_y), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        return panel
    
    def _create_costmap_panel(self, costmap: Optional[np.ndarray], target_height: int = 280, target_width: int = 300) -> np.ndarray:
        """
        Create costmap heatmap panel with colorbar and stats header.
        
        Args:
            costmap: Distance costmap (H, W)
            target_height: Target panel height
            target_width: Target panel width (for image portion)
            
        Returns:
            Panel image as numpy array
        """
        header_height = 40
        content_height = target_height - header_height
        
        if costmap is None:
            # Return placeholder
            placeholder = np.ones((target_height, target_width, 3), dtype=np.uint8) * 200
            cv2.putText(placeholder, "No costmap", (target_width // 4, target_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            return placeholder
        
        # Mask invalid values
        heatmap = np.where(costmap >= 99, np.nan, costmap)
        valid_costs = heatmap[~np.isnan(heatmap)]
        
        if valid_costs.size > 0:
            vmin, vmax = np.percentile(valid_costs, [5, 95])
            cost_min = float(valid_costs.min())
            cost_max = float(valid_costs.max())
            cost_median = float(np.median(valid_costs))
        else:
            vmin, vmax = 0, 1
            cost_min, cost_max, cost_median = 0, 0, 0
        
        # Create header with stats
        header = np.ones((header_height, target_width, 3), dtype=np.uint8) * 255
        cv2.putText(header, "Query Costmap", (10, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stats_text = f"Min:{cost_min:.1f} Max:{cost_max:.1f} Med:{cost_median:.1f}"
        cv2.putText(header, stats_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Create heatmap using matplotlib
        fig, ax = plt.subplots(figsize=(target_width / 100, content_height / 100), dpi=100)
        cmap_obj = plt.get_cmap('turbo').copy()
        cmap_obj.set_bad(color='white')
        
        im = ax.imshow(heatmap, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.axis('off')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        cbar.ax.tick_params(labelsize=7)
        
        plt.tight_layout(pad=0.1)
        
        # Convert to numpy
        fig.canvas.draw()
        heatmap_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        heatmap_img = heatmap_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Resize to fit content area
        heatmap_img = cv2.resize(heatmap_img, (target_width, content_height))
        
        # Combine header and heatmap
        panel = np.vstack([header, heatmap_img])
        
        return panel
    
    def _create_waypoints_panel(self, waypoints: Optional[np.ndarray], target_height: int = 280, target_width: int = 300) -> np.ndarray:
        """
        Create waypoints visualization panel.
        
        Args:
            waypoints: Predicted waypoints (N, 2) in robot frame (x_forward, y_left)
            target_height: Target panel height
            target_width: Target panel width
            
        Returns:
            Panel image as numpy array
        """
        header_height = 40
        content_height = target_height - header_height
        
        # Create header
        header = np.ones((header_height, target_width, 3), dtype=np.uint8) * 255
        cv2.putText(header, "Waypoints (robot frame)", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        if waypoints is None or len(waypoints) == 0:
            # Return placeholder
            content = np.ones((content_height, target_width, 3), dtype=np.uint8) * 200
            cv2.putText(content, "No waypoints", (target_width // 4, content_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            return np.vstack([header, content])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(target_width / 100, content_height / 100), dpi=100)
        fig.patch.set_facecolor('white')
        
        # Plot waypoints (x_forward, y_left) -> plot as (y, x)
        waypoints = np.asarray(waypoints)
        ax.plot(waypoints[:, 1], waypoints[:, 0], 'c-o', markersize=4, linewidth=1.5, alpha=0.8, label='waypoints')
        
        # Add arrows showing direction
        if len(waypoints) > 1:
            for i in range(len(waypoints) - 1):
                dx = waypoints[i + 1, 0] - waypoints[i, 0]
                dy = waypoints[i + 1, 1] - waypoints[i, 1]
                ax.arrow(waypoints[i, 1], waypoints[i, 0], dy * 0.3, dx * 0.3, 
                        head_width=0.1, head_length=0.05, fc='cyan', ec='cyan', alpha=0.6)
        
        # Mark start
        ax.plot(waypoints[0, 1], waypoints[0, 0], 'go', markersize=8, label='start')
        
        ax.set_xlabel("y_left [m]", fontsize=8)
        ax.set_ylabel("x_forward [m]", fontsize=8)
        ax.set_ylim(-0.5, max(3.0, waypoints[:, 0].max() + 0.5))
        ax.set_xlim(-2.0, 2.0)
        ax.invert_xaxis()
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=7, loc='upper right')
        
        plt.tight_layout(pad=0.1)
        
        # Convert to numpy
        fig.canvas.draw()
        content = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        content = content.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Resize to fit content area
        content = cv2.resize(content, (target_width, content_height))
        
        # Combine header and content
        panel = np.vstack([header, content])
        
        return panel
    
    def render_combined_step_vis(
        self,
        step: int,
        rgb: np.ndarray,
        costmap: Optional[np.ndarray] = None,
        waypoints: Optional[np.ndarray] = None,
        waypoints_adjusted: Optional[np.ndarray] = None,
        care_data: Optional[dict] = None,
        matches_data: Optional[dict] = None,
        ref_img: Optional[np.ndarray] = None,
        ref_img_idx: Optional[int] = None,
        agent_position: Optional[np.ndarray] = None,
        agent_rotation: Optional[np.ndarray] = None,
    ):
        """
        Create and save 3-panel combined visualization using matplotlib.
        
        Layout: [Matches Image | Costmap | Waypoints (with CARE if available)]
        
        Args:
            step: Step index
            rgb: Query RGB image
            costmap: Distance costmap
            waypoints: Original predicted waypoints (N, 2)
            waypoints_adjusted: CARE-adjusted waypoints (N, 2) - plotted in magenta if present
            care_data: CARE data dict with obstacles, theta_rot, k_star, etc.
            matches_data: Dict with match data
            ref_img: Reference image
            ref_img_idx: Reference image index
            agent_position: Agent 3D position [x, y, z]
            agent_rotation: Agent rotation quaternion
        """
        if not getattr(self.vis_cfg.render_visualizations, 'combined_step_vis', False):
            return
        
        self._ensure_vis_dirs()
        (self.vis_dir / "combined").mkdir(parents=True, exist_ok=True)
        
        # 1. Generate match visualization image (pixels with drawings)
        match_viz_img, match_title = self._create_match_image(rgb, matches_data, ref_img, ref_img_idx)
        
        # 2. Create matplotlib figure
        # Figure size: 20x5.5 inches (increased to accommodate title)
        fig = plt.figure(figsize=(20, 5.5), dpi=100)
        fig.patch.set_facecolor('white')
        
        # Add overall title with agent pose information
        if agent_position is not None:
            heading = self._compute_heading_from_rotation(agent_rotation) if agent_rotation is not None else None
            title_str = self._format_agent_pose_title(step, agent_position, heading)
            fig.suptitle(title_str, fontsize=12, fontweight='bold', y=0.98)
        
        # GridSpec: Matches gets 2.5 width units
        gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.08, top=0.93)
        
        # Panel 1: Matches Image
        ax_match = fig.add_subplot(gs[0, 0])
        ax_match.imshow(match_viz_img)
        ax_match.axis('off')
        if match_title:
            ax_match.set_title(match_title, fontsize=10)
        
        # Panel 2: Costmap
        ax_costmap = fig.add_subplot(gs[0, 1])
        self._render_costmap_subplot(ax_costmap, costmap)
        
        # Panel 3: Waypoints (with CARE data if available)
        ax_waypoints = fig.add_subplot(gs[0, 2])
        self._render_waypoints_subplot(
            ax_waypoints, 
            waypoints,
            waypoints_adjusted=waypoints_adjusted,
            care_data=care_data
        )
        
        plt.tight_layout(pad=1.0, w_pad=2.0, rect=[0, 0, 1, 0.96])
        
        # Save
        save_path = self.vis_dir / "combined" / f"step_{step:04d}.png"
        fig.savefig(str(save_path), dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.debug(f"Rendered combined visualization: {save_path}")
    
    def _create_match_image(self, rgb, matches_data, ref_img, ref_img_idx, n_samples=20):
        """
        Create concatenated match visualization image using OpenCV drawing commands.
        Returns:
            match_viz_img: RGB numpy array
            title_str: Metadata string for display
        """
        H, W = rgb.shape[:2]
        
        # Prepare combined image canvas
        if ref_img is not None:
            ref_img_resized = cv2.resize(ref_img, (W, H))
            # Ensure 3 channels
            if ref_img_resized.ndim == 2:
                ref_img_resized = cv2.cvtColor(ref_img_resized, cv2.COLOR_GRAY2RGB)
            elif ref_img_resized.shape[2] == 4:
                ref_img_resized = cv2.cvtColor(ref_img_resized, cv2.COLOR_RGBA2RGB)
                
            combined = np.hstack([rgb, ref_img_resized])
        else:
            combined = np.hstack([rgb, np.ones_like(rgb) * 200])
            
        # Ensure contiguous and writable
        combined = np.ascontiguousarray(combined)
        
        # Default empty return
        default_title = f"Query Img vs. Ref Img {ref_img_idx or '?'} | No matches"
        
        if matches_data is None:
            return combined, default_title
            
        qry_mkpts = matches_data.get('qry_mkpts')
        ref_mkpts = matches_data.get('ref_mkpts')
        confidences = matches_data.get('confidences')
        
        if qry_mkpts is None or ref_mkpts is None or confidences is None:
            return combined, default_title
            
        # Reshape/Flatten
        confidences = np.asanyarray(confidences).flatten()
        qry_mkpts = np.asanyarray(qry_mkpts)
        ref_mkpts = np.asanyarray(ref_mkpts)
        
        if qry_mkpts.ndim == 1:
            qry_mkpts = qry_mkpts.reshape(-1, 2)
        if ref_mkpts.ndim == 1:
            ref_mkpts = ref_mkpts.reshape(-1, 2)
            
        n_total = len(confidences)
        n_disp = min(n_samples, n_total)

        min_conf = confidences.min() if n_total > 0 else 0.0
        max_conf = confidences.max() if n_total > 0 else 0.0
        mean_conf = confidences.mean() if n_total > 0 else 0.0
        
        # Format Title
        title_str = (
            f"Query Img vs. Ref Img {ref_img_idx or '?'} | Matches (Uniformly Sampled): {n_disp}/{n_total}\n"
            f"Conf Range: [{min_conf:.2f}, {max_conf:.2f}] | Mean: {mean_conf:.2f}"
        )
        
        if n_total == 0:
            return combined, title_str
            
        # Sample
        if n_total > n_samples:
            indices = np.linspace(0, n_total - 1, n_samples, dtype=int)
        else:
            indices = np.arange(n_total)
            
        sampled_qry = qry_mkpts[indices]
        sampled_ref = ref_mkpts[indices]
        sampled_conf = confidences[indices]
        
        # Color Map
        if sampled_conf.max() == sampled_conf.min():
            norm_conf = np.ones(len(sampled_conf)) * 0.5
        else:
            norm_conf = (sampled_conf - sampled_conf.min()) / (sampled_conf.max() - sampled_conf.min())
            
        cmap = plt.get_cmap('turbo')
        
        # Draw Matches using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        
        for i in range(len(sampled_qry)):
            qry_pt = sampled_qry[i]
            ref_pt = sampled_ref[i]
            conf_val = sampled_conf[i]
            
            # Colors
            nc = norm_conf[i]
            color_rgba = cmap(float(nc))
            color_rgb = tuple(int(c * 255) for c in color_rgba[:3]) # matplotlib cmap returns 0-1
            # cv2 uses BGR? But here we are passing to matplotlib imshow which expects RGB if we don't convert.
            # Wait, previously I was converting RGB to BGR before imwrite. 
            # If I return RGB image here, ax.imshow handles it fine.
            # So let's use RGB color tuple.
            
            # Coords
            qx, qy = int(qry_pt[0]), int(qry_pt[1])
            rx, ry = int(ref_pt[0]) + W, int(ref_pt[1])
            
            # Draw Line
            cv2.line(combined, (qx, qy), (rx, ry), color_rgb, 1, cv2.LINE_AA)
            # Draw Circles
            cv2.circle(combined, (qx, qy), 4, color_rgb, -1)
            cv2.circle(combined, (rx, ry), 4, color_rgb, -1)
            
            # Text Annotation
            mx, my = (qx + rx) // 2, (qy + ry) // 2
            text = f"{conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Box background
            cv2.rectangle(combined, (mx, my - th - 2), (mx + tw, my + 2), color_rgb, -1)
            # White text
            cv2.putText(combined, text, (mx, my), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
        return combined, title_str
    
    def _render_costmap_subplot(self, ax, costmap):
        """Render costmap heatmap into a matplotlib axes."""
        if costmap is None:
            ax.text(0.5, 0.5, 'No costmap', ha='center', va='center', fontsize=12)
            ax.set_title("Query Costmap", fontsize=10)
            ax.axis('off')
            return
        
        # Mask invalid values
        heatmap = np.where(costmap >= 99, np.nan, costmap)
        valid_costs = heatmap[~np.isnan(heatmap)]
        
        if valid_costs.size > 0:
            vmin, vmax = np.percentile(valid_costs, [5, 95])
            cost_min = float(valid_costs.min())
            cost_max = float(valid_costs.max())
            cost_median = float(np.median(valid_costs))
        else:
            vmin, vmax = 0, 1
            cost_min, cost_max, cost_median = 0, 0, 0
        
        cmap_obj = plt.get_cmap('turbo').copy()
        cmap_obj.set_bad(color='white')
        
        im = ax.imshow(heatmap, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.set_title(f"Query Costmap\nMin:{cost_min:.1f} Max:{cost_max:.1f} Med:{cost_median:.1f}", fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    
    def _compute_heading_from_rotation(self, rotation) -> float:
        """
        Compute heading angle from rotation quaternion.
        
        Args:
            rotation: Quaternion as numpy array [x, y, z, w] or object with .imag/.real
        
        Returns:
            Heading angle in degrees (0-360)
        """
        if rotation is None:
            return None
        
        try:
            rotation = np.asarray(rotation)
            if len(rotation) == 4:
                # Handle different quaternion formats
                # Check if first element looks like w (typically close to ±1)
                if abs(rotation[0]) > 0.5 and abs(rotation[0]) >= abs(rotation[1:]).max():
                    # Likely [w, x, y, z]
                    quat = mn.Quaternion(mn.Vector3(float(rotation[1]), float(rotation[2]), float(rotation[3])), float(rotation[0]))
                else:
                    # Likely [x, y, z, w]
                    quat = mn.Quaternion(mn.Vector3(float(rotation[0]), float(rotation[1]), float(rotation[2])), float(rotation[3]))
                
                # Convert to rotation matrix
                R = np.array(quat.to_matrix())
                
                # Apply coordinate transformation (Habitat to standard)
                R_bc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                R = R @ R_bc
                
                # Extract yaw angle (heading)
                heading_rad = np.arctan2(R[0, 2], R[2, 2])
                heading_deg = np.degrees(heading_rad)
                
                # Normalize to 0-360
                heading_deg = heading_deg % 360
                
                return heading_deg
        except Exception as e:
            logger.warning(f"Could not compute heading from rotation: {e}")
            return None
    
    def _format_agent_pose_title(self, step: int, position: np.ndarray, heading: float) -> str:
        """
        Format agent pose information as title string.
        
        Args:
            step: Step index
            position: 3D position [x, y, z]
            heading: Heading angle in degrees (0-360) or None
        
        Returns:
            Formatted title string
        """
        position = np.asarray(position)
        
        # Format position with 2 decimal places
        pos_str = f"Pos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        
        # Add heading if available
        if heading is not None:
            heading_str = f"Heading: {heading:.1f}°"
            title = f"Step {step:04d} | {pos_str} | {heading_str}"
        else:
            title = f"Step {step:04d} | {pos_str}"
        
        return title
    
    def _render_waypoints_subplot(
        self, 
        ax, 
        waypoints,
        waypoints_adjusted: Optional[np.ndarray] = None,
        care_data: Optional[dict] = None
    ):
        """
        Render waypoints into a matplotlib axes.
        
        Shows original waypoints in cyan and CARE-adjusted waypoints in magenta
        (matching save_care_visualization style from care.py).
        
        Args:
            ax: matplotlib axes
            waypoints: Original waypoints (N, 2) with (x_forward, y_left)
            waypoints_adjusted: CARE-adjusted waypoints (N, 2) - optional
            care_data: CARE data dict with obstacles, theta_rot, k_star, etc. - optional
        """
        ax.set_facecolor('white')
        
        has_waypoints = waypoints is not None and len(waypoints) > 0
        has_adjusted = waypoints_adjusted is not None and len(waypoints_adjusted) > 0
        
        if not has_waypoints and not has_adjusted:
            ax.text(0.5, 0.5, 'No waypoints', ha='center', va='center', fontsize=12)
            ax.set_title("Waypoints (robot frame)", fontsize=9)
            return
        
        # Extract CARE data if available
        obstacles = None
        theta_rot = None
        k_star = None
        if care_data is not None:
            obstacles = care_data.get('obstacles')
            theta_rot = care_data.get('theta_rot')
            k_star = care_data.get('k_star')
            # Handle scalar values stored as 0-d numpy arrays
            if hasattr(theta_rot, 'item'):
                theta_rot = theta_rot.item()
            if hasattr(k_star, 'item'):
                k_star = int(k_star.item())
        
        # Calculate y-limits for the plot based on available data
        max_x_forward = 3.0
        if has_waypoints:
            waypoints = np.asarray(waypoints)
            max_x_forward = max(max_x_forward, waypoints[:, 0].max() + 0.5)
        if has_adjusted:
            waypoints_adjusted = np.asarray(waypoints_adjusted)
            max_x_forward = max(max_x_forward, waypoints_adjusted[:, 0].max() + 0.5)
        
        # Plot obstacles if available (like care.py visualization)
        if obstacles is not None and len(obstacles) > 0:
            obstacles = np.asarray(obstacles)
            # obstacles are in (x_forward, y_left) format
            ax.scatter(obstacles[:, 1], obstacles[:, 0], c='red', marker='x', 
                      s=20, alpha=0.7, label='obstacles', zorder=1)
        
        # Plot original waypoints (cyan) - labeled as "initial" like in care.py
        if has_waypoints:
            # Plot waypoints (x_forward, y_left) -> plot as (y, x) for x-axis=left, y-axis=forward
            ax.plot(waypoints[:, 1], waypoints[:, 0], 'c-o', markersize=4, 
                   linewidth=1.5, alpha=0.8, label='initial', zorder=2)
            
            # Mark k_star waypoint if available (point with max repulsive force)
            if k_star is not None and 0 <= k_star < len(waypoints):
                ax.plot(waypoints[k_star, 1], waypoints[k_star, 0], 'y*', 
                       markersize=12, label=f'k*={k_star}', zorder=4)
            
            # Add arrows showing direction for original waypoints
            if len(waypoints) > 1:
                for i in range(len(waypoints) - 1):
                    dx = waypoints[i + 1, 0] - waypoints[i, 0]
                    dy = waypoints[i + 1, 1] - waypoints[i, 1]
                    ax.arrow(waypoints[i, 1], waypoints[i, 0], dy * 0.3, dx * 0.3, 
                            head_width=0.1, head_length=0.05, fc='cyan', ec='cyan', alpha=0.6)
        
        # Plot adjusted waypoints (magenta) - labeled as "corrected" like in care.py
        if has_adjusted:
            ax.plot(waypoints_adjusted[:, 1], waypoints_adjusted[:, 0], 'm-s', 
                   markersize=4, linewidth=1.5, alpha=0.8, label='corrected', zorder=3)
            
            # Add arrows for adjusted waypoints
            if len(waypoints_adjusted) > 1:
                for i in range(len(waypoints_adjusted) - 1):
                    dx = waypoints_adjusted[i + 1, 0] - waypoints_adjusted[i, 0]
                    dy = waypoints_adjusted[i + 1, 1] - waypoints_adjusted[i, 1]
                    ax.arrow(waypoints_adjusted[i, 1], waypoints_adjusted[i, 0], dy * 0.3, dx * 0.3, 
                            head_width=0.1, head_length=0.05, fc='magenta', ec='magenta', alpha=0.6)
        
        # Mark origin/robot position
        ax.plot(0, 0, 'go', markersize=8, label='robot', zorder=5)
        
        # Build title with CARE info
        if theta_rot is not None:
            title = f"Waypoints (θ_rot={np.degrees(theta_rot):.1f}°)"
        elif has_adjusted:
            title = "Waypoints (initial + corrected)"
        else:
            title = "Waypoints (robot frame)"
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("y_left [m]", fontsize=7)
        ax.set_ylabel("x_forward [m]", fontsize=7)
        ax.set_ylim(-0.5, max_x_forward)
        ax.set_xlim(-2.0, 2.0)
        ax.invert_xaxis()
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(labelsize=7)
    
    # =============================================
    # Waypoint + Topdown Visualization Methods
    # =============================================
    
    def _render_topdown_depth_subplot(
        self, 
        ax, 
        depth: np.ndarray,
        intrinsics: Optional[dict] = None,
        params: Optional[dict] = None
    ):
        """
        Render top-down obstacle map from depth into a matplotlib axes.
        
        Adapted from care.py _plot_topdown_depth function.
        Projects depth to top-down view and plots obstacle scatter.
        
        Args:
            ax: matplotlib axes
            depth: Depth map (H, W) in meters
            intrinsics: Camera intrinsics dict with fx, fy, cx, cy
                       If None, computes from depth shape assuming 90 FOV
            params: CARE-style params dict with tau_z, sample_step, etc.
                   If None, uses defaults
        """
        import math
        
        if depth is None or depth.size == 0:
            ax.text(0.5, 0.5, 'No depth data', ha='center', va='center', fontsize=12)
            ax.set_title("Top-down Obstacles", fontsize=9)
            return
        
        H, W = depth.shape
        
        # Default intrinsics: assume 90 FOV if not provided
        if intrinsics is None:
            fov = 90.0
            fx = fy = (W / 2) / math.tan(math.radians(fov) / 2)
            cx = W / 2
            cy = H / 2
            intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        
        # Default params matching CARE
        if params is None:
            params = {
                "tau_z": 0.5,
                "depth_offset": 0.0,
                "cam_x_offset": 0.0,
                "cam_y_offset": 0.0,
                "sample_step": 10,
            }
        
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        tau_z = params.get("tau_z", 0.5)
        depth_offset = params.get("depth_offset", 0.0)
        cam_x_off = params.get("cam_x_offset", 0.0)
        cam_y_off = params.get("cam_y_offset", 0.0)
        step = params.get("sample_step", 10)
        
        xs = []
        ys = []
        for v in range(0, H, step):
            for u in range(0, W, step):
                z = float(depth[v, u]) - depth_offset
                if z <= 0 or z > tau_z:
                    continue
                x_cam = (u - cx) * z / fx
                z_cam = z
                x_forward = z_cam + cam_x_off
                y_left = -x_cam + cam_y_off
                xs.append(x_forward)
                ys.append(y_left)
        
        ax.set_facecolor('white')
        
        if len(xs) > 0:
            xs = np.array(xs)
            ys = np.array(ys)
            ax.scatter(
                ys,
                xs,
                s=2,
                c=xs,
                cmap="plasma",
                alpha=0.6,
                edgecolors="none",
            )
        
        ax.set_title("Top-down Obstacles", fontsize=9)
        ax.set_xlabel("y_left [m]", fontsize=7)
        ax.set_ylabel("x_forward [m]", fontsize=7)
        ax.set_ylim(-0.2, tau_z + 0.3)
        ax.set_xlim(-tau_z, tau_z)
        ax.invert_xaxis()
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=7)
    
    def render_waypoint_topdown_vis(
        self,
        step: int,
        waypoints: Optional[np.ndarray] = None,
        waypoints_adjusted: Optional[np.ndarray] = None,
        care_data: Optional[dict] = None,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[dict] = None,
        params: Optional[dict] = None,
    ):
        """
        Create and save dual-panel visualization: waypoints (left) + topdown depth (right).
        
        Similar to create_controller_vis.py reference code layout.
        
        Args:
            step: Step index
            waypoints: Original waypoints (N, 2) with (x_forward, y_left)
            waypoints_adjusted: CARE-adjusted waypoints (optional)
            care_data: CARE data dict with obstacles, theta_rot, k_star, etc.
            depth: Depth map (H, W) in meters
            intrinsics: Camera intrinsics dict (optional, auto-computed if None)
            params: CARE-style params dict (optional)
        """
        if not getattr(self.vis_cfg.render_visualizations, 'waypoint_topdown_vis', False):
            return
        
        self._ensure_vis_dirs()
        (self.vis_dir / "waypoint_topdown").mkdir(parents=True, exist_ok=True)
        
        # Create figure with two subplots side by side - fixed size like reference code
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(wspace=0.05)  # Tighter spacing like reference
        
        # Left panel: Waypoints
        ax_waypoints = axes[0]
        self._render_waypoints_subplot(
            ax_waypoints,
            waypoints,
            waypoints_adjusted=waypoints_adjusted,
            care_data=care_data
        )
        
        # Right panel: Top-down depth/obstacles
        ax_topdown = axes[1]
        self._render_topdown_depth_subplot(
            ax_topdown,
            depth,
            intrinsics=intrinsics,
            params=params
        )
        
        # Save with fixed size (no bbox_inches='tight' to maintain consistent dimensions)
        save_path = self.vis_dir / "waypoint_topdown" / f"step_{step:04d}.png"
        fig.savefig(str(save_path), dpi=200, pad_inches=0.0, facecolor='white')
        plt.close(fig)
        logger.debug(f"Rendered waypoint+topdown visualization: {save_path}")
    
    # =============================================
    # Video Creation Methods
    # =============================================
    
    def create_video_from_frames(
        self,
        frames_dir: str,
        output_filename: str = None,
        fps: int = 5
    ) -> bool:
        """
        Create video from frames in a subdirectory.
        
        Handles varying frame sizes by resizing all frames to first frame dimensions.
        
        Args:
            frames_dir: Subdirectory name (e.g., "rgb", "costmaps", "combined", "topdown")
            output_filename: Output video filename (default: "{frames_dir}_video.mp4")
            fps: Frames per second (default: 5)
            
        Returns:
            True if video created successfully, False otherwise
        """
        frames_path = self.vis_dir / frames_dir
        
        if not frames_path.exists():
            logger.debug(f"Frames directory does not exist: {frames_path}")
            return False
        
        # Find all frame files (sorted naturally)
        frame_files = sorted(frames_path.glob("step_*.png"))
        
        if len(frame_files) == 0:
            logger.debug(f"No frames found in: {frames_path}")
            return False
        
        # Set output filename
        if output_filename is None:
            output_filename = f"{frames_dir}_video.mp4"
        
        output_path = self.vis_dir / output_filename
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            logger.warning(f"Could not read first frame: {frame_files[0]}")
            return False
        
        H, W = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        
        if not video_writer.isOpened():
            logger.error(f"Failed to create video writer for: {output_path}")
            return False
        
        resize_count = 0
        
        try:
            for frame_path in frame_files:
                frame = cv2.imread(str(frame_path))
                
                if frame is None:
                    logger.warning(f"Could not read frame: {frame_path}")
                    continue
                
                # Resize if dimensions don't match
                if frame.shape[:2] != (H, W):
                    # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
                    interp = cv2.INTER_AREA if frame.shape[0] > H else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (W, H), interpolation=interp)
                    resize_count += 1
                
                video_writer.write(frame)
            
            video_writer.release()
            
            if resize_count > 0:
                logger.warning(f"Resized {resize_count}/{len(frame_files)} frames to ({H}, {W})")
            
            logger.info(f"Created video: {output_path} ({len(frame_files)} frames, {fps} fps)")
            return True
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            video_writer.release()
            return False
    
    def create_all_videos(self, fps: int = 5) -> dict:
        """
        Create videos for all visualization types that have frames.
        
        Saves videos directly to vis_dir (not in subfolders).
        
        Args:
            fps: Frames per second (default: 5)
            
        Returns:
            Dict mapping video type to success status
        """
        video_types = ["rgb", "costmaps", "topdown", "combined", "waypoint_topdown"]
        results = {}
        
        logger.info(f"Creating videos from frames (fps={fps})...")
        
        for vid_type in video_types:
            success = self.create_video_from_frames(vid_type, fps=fps)
            results[vid_type] = success
        
        # Summary
        created = sum(1 for v in results.values() if v)
        logger.info(f"Created {created}/{len(video_types)} videos")
        
        return results


# Offline batch processing functions

def render_all_visualizations_offline(
    episode_results_path: Path,
    vis_cfg: DictConfig,
    ref_images_dir: Path
):
    """
    Batch render all visualizations from saved raw data.
    
    Args:
        episode_results_path: Path to episode results
        vis_cfg: Visualization config (cfg.visualization section)
        ref_images_dir: Path to reference images directory
    """
    from .data_storage import load_depth_png, load_matches_npz
    
    renderer = VisualizationRenderer(vis_cfg, episode_results_path)
    renderer.online_render = True  # Force rendering in offline mode
    
    step_data_dir = episode_results_path / "step_data"
    
    # Find all step directories
    step_dirs = sorted(step_data_dir.glob("step_*"))
    
    logger.info(f"Rendering visualizations for {len(step_dirs)} steps...")
    
    for step_dir in step_dirs:
        step = int(step_dir.name.split("_")[1])
        
        # Load raw data
        rgb_path = step_dir / "rgb.png"
        costmap_path = step_dir / "costmap.npy"
        matches_path = step_dir / "matches.npz"
        
        rgb = cv2.imread(str(rgb_path))
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        costmap = np.load(costmap_path) if costmap_path.exists() else None
        
        matches_data = None
        ref_img_path = None
        if matches_path.exists():
            matches_data = load_matches_npz(matches_path)
            ref_img_idx = matches_data['closest_map_img_idx']
            ref_img_path = ref_images_dir / f"{ref_img_idx:05d}.jpg"
        
        # Render
        renderer.render_step_visualizations(
            step, rgb, costmap, matches_data, ref_img_path
        )
    
    logger.info(f"Finished rendering visualizations to {episode_results_path / 'visualizations'}")
