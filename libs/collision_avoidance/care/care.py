"""
CARE-style collision avoidance (simplified, standalone implementation).

Requirements:
- numpy
- optionally: cv2 for visualization (not required)

Key functions:
- ConstructTopDownObstacleMap(depth, intrinsics, params)
- EstimateRepulsiveDirection(waypoints, obstacles, params)
- RotateTrajectory(waypoints, theta)
- care_step(rgb, depth, waypoints, intrinsics, params)
"""

import numpy as np
import math

# ---------- Default parameters (matching paper) ----------
DEFAULT_PARAMS = {
    # CARE-specific
    "theta_clip": math.pi / 4.0,    # maximum rotation induced by repulsive force (θclip)
    "theta_thres": math.pi / 6.0,   # threshold to do rotate-in-place (θthres)
    "tau_z": 0.5,                   # max depth sensing range in meters (τz)
    "depth_offset": -0.0,           # depth offset to compensate for camera/robot geometry
    "vertical_offset": -0.05,       # ϵ in paper (filter ceiling), negative means exclude above camera
    "sample_step": 10,              # sample every N pixels from depth for top-down map
    "min_obstacle_dist": 0.05,      # minimum distance to consider for repulsion (avoid singularity)
    # robot motion limits (example)
    "v_forward": 0.15,   # m/s for MoveForwardAndTurn
    "v_max": 0.2,
    "omega_max": 0.8,
    # camera extrinsics relative to robot base (meters)
    # camera is at (x_cam, y_cam, z_cam) in robot frame. For most front-mounted cameras,
    # x_cam positive = forward, y_cam positive = left, z_cam = camera height above ground
    "cam_x_offset": 0.0,
    "cam_y_offset": 0.0,
    "cam_height": 0.0,
}


# ---------- Utilities ----------
def clamp(x, a, b):
    return max(a, min(b, x))


# ---------- Top-Down Projection ----------
def ConstructTopDownObstacleMap(depth,
                                intrinsics,
                                params=DEFAULT_PARAMS):
    """
    depth: H x W depth map in meters (float)
    intrinsics: dict with fx, fy, cx, cy (camera intrinsics)
    returns:
      obstacles: (M,2) array of obstacle points (x_forward, y_left) in robot local frame (meters)
    Assumptions:
      - depth is metric in meters and represents Z (forward) in camera frame
      - camera optical axes follow pinhole convention
      - camera coordinate frame: x right, y down, z forward (common). We'll convert to robot frame:
          robot-forward = cam_z, robot-left = -cam_x (so left is positive)
      - camera pose relative to robot provided via params["cam_x_offset"], cam_height etc.
    """
    fx = intrinsics['fx']; fy = intrinsics['fy']; cx = intrinsics['cx']; cy = intrinsics['cy']
    H, W = depth.shape
    step = params.get("sample_step", 10)
    tau_z = params.get("tau_z", 1.0)
    depth_offset = params.get("depth_offset", 0.0)
    vertical_offset = params.get("vertical_offset", 0.0)
    cam_x_off = params.get("cam_x_offset", 0.0)
    cam_y_off = params.get("cam_y_offset", 0.0)
    cam_h = params.get("cam_height", 0.0)

    pts = []  # will collect (x_forward, y_left) in robot frame

    # Sample pixels to reduce compute
    for v in range(0, H, step):
        for u in range(0, W, step):
            z = float(depth[v, u]) - depth_offset
            if z <= 0 or z > tau_z:
                continue
            # backproject to camera coordinates
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            z_cam = z
            # Convert camera coords to robot local frame:
            # robot-forward = z_cam
            # robot-left = -x_cam
            # robot-up = -y_cam
            # Filter out points that are above camera by vertical_offset (paper uses Y >= -eps)
            # Here y_cam is vertical relative to camera center: positive = down. so points with
            # (camera_height - point_z_in_world_y) can be used, but we keep it simple:
            # approximate Y (height above ground) = cam_height - y_cam.
            point_height = cam_h - (-y_cam)  # because y_cam positive down => -y_cam is camera-up
            # if point_height > -vertical_offset:
            #     # This filters ceiling points (paper uses Y >= -eps)
            #     continue

            x_forward = z_cam + cam_x_off
            y_left = -x_cam + cam_y_off
            pts.append((x_forward, y_left))

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=float)

    pts = np.array(pts, dtype=float)

    # For each x bin, pick closest z (paper discretizes x axis into bins and selects min z)
    # We can bin along forward axis into M bins:
    M = 64
    x_min, x_max = 0.0, params.get("tau_z", 1.0)
    bins = np.linspace(x_min, x_max, M + 1)
    obstacles = []
    for i in range(M):
        lo, hi = bins[i], bins[i + 1]
        mask = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if not np.any(mask):
            continue
        selected = pts[mask]
        # choose the point closest to the centerline (y_left = 0)
        idx = np.argmin(np.abs(selected[:, 1]))
        obstacles.append(selected[idx])

    if len(obstacles) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.array(obstacles, dtype=float)


# ---------- Repulsive Force Estimation ----------
def EstimateRepulsiveDirection(waypoints, obstacles, params=DEFAULT_PARAMS):
    """
    waypoints: K x D (D>=2). First two dims are (x_forward, y_left) in robot local frame.
    obstacles: M x 2 (x_forward, y_left)
    returns:
      theta_rep (float) - repulsive direction angle in radians (relative to robot forward)
      k_star (int) - index of waypoint with maximum repulsive magnitude
      Frep_all - repulsive vectors per waypoint (for debugging/visualization)
    Implementation detail:
      Use Frep(pk) = sum_m - (pk - om) / (||pk - om||^3 + eps)
      This yields a vector that points away from obstacles. We then compute angle via atan2.
    """

    if obstacles.shape[0] == 0:
        return 0.0, None, None  # no adjustment

    # Use only the positional components (x_forward, y_left)
    wp_xy = waypoints[:, :2]
    K = wp_xy.shape[0]
    M = obstacles.shape[0]
    eps = 1e-6
    Frep_all = np.zeros((K, 2), dtype=float)
    for k in range(K):
        pk = wp_xy[k]
        vec = np.zeros(2, dtype=float)
        for m in range(M):
            om = obstacles[m]
            diff = pk - om
            dist = np.linalg.norm(diff)
            if dist < params.get("min_obstacle_dist", 0.05):
                dist = params.get("min_obstacle_dist", 0.05)
            contrib = diff / (dist ** 3 + eps)
            vec += contrib
        Frep_all[k] = vec

    mags = np.linalg.norm(Frep_all, axis=1)
    k_star = int(np.argmax(mags))
    v = Frep_all[k_star]
    # angle of repulsive vector: arctan2(y, x) where x forward, y left
    theta_rep = math.atan2(v[1], v[0])

    # Clip angle to avoid excessive deviation
    theta_clip = params.get("theta_clip", math.pi / 4.0)
    theta_rot = clamp(theta_rep, -theta_clip, theta_clip)
    return float(theta_rot), k_star, Frep_all


# ---------- Rotate trajectory ----------
def RotateTrajectory(waypoints, theta):
    """
    Rotate each waypoint pk by rotation matrix R(theta) about the robot origin.
    Waypoints are K x D:
      - first two dims: (x_forward, y_left)
      - remaining dims (e.g., heading hx, hy) are kept as-is.
    Rotation is applied only to the first two dimensions.
    """
    if waypoints.size == 0:
        return waypoints

    c = math.cos(theta); s = math.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])

    # Ensure 2D array
    wp = np.asarray(waypoints)
    wp_shape = wp.shape
    if wp.ndim != 2 or wp_shape[1] < 2:
        raise ValueError(f"waypoints must be KxD with D>=2, got shape {wp_shape}")

    pos = wp[:, :2]                       # (K, 2)
    pos_rot = (R @ pos.T).T               # (K, 2)

    if wp_shape[1] == 2:
        return pos_rot

    # Preserve remaining dims (e.g., heading hx, hy)
    rest = wp[:, 2:]                      # (K, D-2)
    adjusted = np.concatenate([pos_rot, rest], axis=1)
    return adjusted


# ---------- Desired heading and motion command ----------
def ComputeDesiredHeading(adjusted_waypoints, k_star):
    """
    Compute desired heading angle to the selected waypoint p'_kstar.
    Waypoints may be KxD with D>=2; only first two dims (x_forward, y_left) are used.
    Returns theta_des in radians (atan2(y, x))
    """
    if (
        k_star is None
        or adjusted_waypoints is None
        or adjusted_waypoints.shape[0] == 0
    ):
        return 0.0
    p = adjusted_waypoints[k_star, :2]
    return math.atan2(p[1], p[0])


def MotionCommandFromHeading(theta_des, params=DEFAULT_PARAMS):
    """
    Apply Safe-FOV mechanism:
      if |theta_des| > theta_thres:
          v = 0, omega = sign(theta_des) * min(|theta_des|, omega_max)
      else:
          v = v_forward, omega = sign(theta_des) * min(|theta_des|, omega_max)
    Returns (v, omega)
    """
    theta_thres = params.get("theta_thres", math.pi / 6.0)
    v_forward = params.get("v_forward", 0.15)
    v_max = params.get("v_max", 0.2)
    omega_max = params.get("omega_max", 0.8)

    ang = float(theta_des)
    ang_clamped = clamp(ang, -omega_max, omega_max)  # limit to omega_max if you like (rad/s)
    if abs(ang) > theta_thres:
        return 0.0, math.copysign(min(abs(ang), omega_max), ang)
    else:
        return float(min(v_forward, v_max)), math.copysign(min(abs(ang), omega_max), ang)


# ---------- Full CARE step ----------
def care_step(rgb,
              depth,
              waypoints,
              intrinsics,
              params=DEFAULT_PARAMS):
    """
    Inputs:
      rgb: HxWx3 (not used in core algorithm; for completeness)
      depth: HxW depth in meters
      waypoints: KxD np.array (first two dims are x_forward, y_left) from controller (robot local frame)
      intrinsics: dict fx, fy, cx, cy
      params: CARE params

    Outputs:
      v, omega (control command)
      adjusted_waypoints (Kx2)
      theta_rot
      k_star
    """
    obstacles = ConstructTopDownObstacleMap(depth, intrinsics, params)
    if obstacles.shape[0] == 0:
        # no obstacles: follow original waypoints -> desired heading is waypoint 1 (or 2 in paper)
        k_idx = 1 if waypoints.shape[0] > 1 else 0
        theta_des = ComputeDesiredHeading(waypoints, k_idx)
        v, omega = MotionCommandFromHeading(theta_des, params)
        return {
            "v": v,
            "omega": omega,
            "adjusted_waypoints": waypoints,
            "theta_rot": 0.0,
            "k_star": None,
            "obstacles": obstacles,
        }

    theta_rot, k_star, frep_all = EstimateRepulsiveDirection(waypoints, obstacles, params)
    adjusted_waypoints = RotateTrajectory(waypoints, theta_rot)
    theta_des = ComputeDesiredHeading(adjusted_waypoints, k_star)
    v, omega = MotionCommandFromHeading(theta_des, params)
    return {
        "v": v,
        "omega": omega,
        "adjusted_waypoints": adjusted_waypoints,
        "theta_rot": theta_rot,
        "k_star": k_star,
        "obstacles": obstacles,
        "frep_all": frep_all,
    }


# ---------- Visualization helpers ----------
def _plot_traj(ax, traj, color="c", label=None, quiver_freq=1):
    """
    Plot trajectory in (x_forward, y_left) frame.

    If traj is KxD with D>=2, only the first two dims are plotted.
    """
    if traj is None or len(traj) == 0:
        return

    traj = np.asarray(traj)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"traj must be KxD with D>=2, got shape {traj.shape}")
    traj_xy = traj[:, :2]

    ax.plot(
        traj_xy[:, 1],
        traj_xy[:, 0],
        color=color,
        alpha=0.7,
        marker="o",
        markersize=3,
        linewidth=1.0,
        label=label,
    )
    # lazy import to avoid hard dependency in core
    import numpy as _np

    def _gen_bearings_from_waypoints(wp):
        # finite-diff bearings; fall back to last segment for final point
        if wp.shape[0] < 2:
            return _np.zeros_like(wp)
        dirs = _np.zeros_like(wp)
        dirs[:-1] = wp[1:] - wp[:-1]
        dirs[-1] = dirs[-2]
        # normalize
        norms = _np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        return dirs / norms

    bearings = _gen_bearings_from_waypoints(traj_xy)
    ax.quiver(
        traj_xy[::quiver_freq, 1],
        traj_xy[::quiver_freq, 0],
        -bearings[::quiver_freq, 1],
        bearings[::quiver_freq, 0],
        color=color,
        scale=10.0,
        width=0.003,
        alpha=0.7,
    )


def _plot_topdown_depth(ax, depth, intrinsics, params):
    """
    Plot a simple top-down point cloud from depth (same projection as ConstructTopDownObstacleMap).
    """
    import numpy as _np

    H, W = depth.shape
    fx = intrinsics["fx"]; fy = intrinsics["fy"]
    cx = intrinsics["cx"]; cy = intrinsics["cy"]
    tau_z = params.get("tau_z", 1.0)
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

    if len(xs) > 0:
        xs = _np.array(xs)
        ys = _np.array(ys)
        ax.scatter(
            ys,
            xs,
            s=2,
            c=xs,
            cmap="plasma",
            alpha=0.6,
            edgecolors="none",
        )

    ax.set_title("Top-down depth / obstacles", fontsize=8)
    ax.set_xlabel("y_left [m]", fontsize=6)
    ax.set_ylabel("x_forward [m]", fontsize=6)
    ax.set_ylim(-1, params.get("tau_z", 1.0) + 0.5)
    ax.set_xlim(-params.get("tau_z", 1.0), params.get("tau_z", 1.0))
    ax.invert_xaxis()
    ax.set_aspect("equal", "box")
    ax.grid(False)


def save_care_visualization(rgb,
                            depth,
                            waypoints,
                            intrinsics,
                            care_output,
                            save_path,
                            params=DEFAULT_PARAMS):
    """
    Save a 2x2 visualization:
      - top-left: RGB
      - top-right: top-down depth
      - bottom-left: initial & corrected waypoints (robot frame)
      - bottom-right: debug text (theta, v, omega)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    adjusted_waypoints = care_output.get("adjusted_waypoints", None)
    obstacles = care_output.get("obstacles", np.zeros((0, 2)))
    v = care_output.get("v", 0.0)
    omega = care_output.get("omega", 0.0)
    theta_rot = care_output.get("theta_rot", 0.0)
    k_star = care_output.get("k_star", None)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.tight_layout(pad=1.0)

    # 1) RGB
    ax = axes[0, 0]
    if rgb is not None:
        if rgb.dtype != np.uint8:
            rgb_vis = np.clip(rgb, 0, 1)
        else:
            rgb_vis = rgb
        ax.imshow(rgb_vis)
    ax.set_title("RGB", fontsize=8)
    ax.axis("off")

    # 2) Top-down depth / obstacles
    ax = axes[0, 1]
    _plot_topdown_depth(ax, depth, intrinsics, params)
    if obstacles is not None and obstacles.shape[0] > 0:
        ax.scatter(
            obstacles[:, 1],
            obstacles[:, 0],
            c="r",
            s=8,
            marker="x",
            label="obstacles (binned)",
        )
        ax.legend(fontsize=6, loc="upper right")

    # 3) Trajectories
    ax = axes[1, 0]
    _plot_traj(ax, waypoints, color="c", label="initial", quiver_freq=1)
    _plot_traj(ax, adjusted_waypoints, color="m", label="corrected", quiver_freq=1)
    ax.set_title("Waypoints (robot frame)", fontsize=8)
    ax.set_ylim(-1, params.get("tau_z", 1.0) + 2.0)
    ax.set_xlim(-params.get("tau_z", 1.0) - 1.0, params.get("tau_z", 1.0) + 1.0)
    ax.invert_xaxis()
    ax.set_aspect("equal", "box")
    ax.grid(False)
    ax.legend(fontsize=6, loc="upper right")

    # 4) Debug text
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"v: {v:.3f} m/s",
        f"omega: {omega:.3f} rad/s",
        f"theta_rot: {theta_rot:.3f} rad ({math.degrees(theta_rot):.1f} deg)",
        f"k_star: {k_star}",
        f"#obstacles: {0 if obstacles is None else obstacles.shape[0]}",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.0, dpi=200)
    plt.close(fig)


if __name__ == "__main__":

    import glob
    import os
    import imageio
    from tqdm import tqdm

    # Paths
    d = 'outppput/tmp/original/tag=Wo6kuutE9i7_oracle_LC_gt_depth_sim_fov120_min15_pl_goalImg87/val/full/'
    for ext_dir in os.listdir(d):
        # 20251125-15-56-41_learnt_topological_pixelwise/run2_learnt_topological_pixelwise'
        for d2 in os.listdir(os.path.join(d, ext_dir)):
            if os.path.isdir(os.path.join(d, ext_dir, d2)):
                base_dir = os.path.join(d, ext_dir, d2)
                # base_dir = os.path.join(d, base_dir) + '/Wo6kuutE9i7_learnt_topological_pixelwise'

                rgb_dir = os.path.join(base_dir, 'observed_rgb')
                depth_dir = os.path.join(base_dir, 'step_visualizations')
                waypoints_dir = os.path.join(base_dir, 'action_waypoints')
                vis_dir = os.path.join(base_dir, 'care_visualizations')
                os.makedirs(vis_dir, exist_ok=True)

                # Find all steps
                rgb_files = sorted(glob.glob(os.path.join(rgb_dir, 'step_*_obs.png')))
                for rgb_path in tqdm(rgb_files):
                    # Extract step number
                    step_num = rgb_path.split('step_')[1].split('_')[0]
                    depth_path = os.path.join(depth_dir, f'step_{step_num}', 'depth_observed.npy')
                    waypoints_path = os.path.join(waypoints_dir, f'action_waypoints_{step_num}.npy')

                    if not (os.path.exists(depth_path) and os.path.exists(waypoints_path)):
                        print(f"Skipping step {step_num}: missing files.")
                        continue

                    rgb = imageio.imread(rgb_path)
                    depth = np.load(depth_path)
                    waypoints = np.load(waypoints_path)

                    # print(f"Step {step_num}:")
                    # print("rgb shape:", rgb.shape, "dtype:", rgb.dtype)
                    # print(depth)
                    # print("depth shape:", depth.shape, "dtype:", depth.dtype, "min/max:", np.min(depth), np.max(depth))
                    # print("waypoints shape:", waypoints.shape, "dtype:", waypoints.dtype)

                    # intrinsics = {"fx": 256.0, "fy": 256.0, "cx": 160.0, "cy": 120.0}
                    # calculate intrinsics based on fov
                    fov = 90.0  # degrees
                    H, W = depth.shape
                    fx = fy = (W / 2) / math.tan(math.radians(fov) / 2)
                    cx = W / 2
                    cy = H / 2
                    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


                    out = care_step(rgb, depth, waypoints, intrinsics)
                    # print("v, omega:", out["v"], out["omega"])
                    # print("theta_rot (deg):", math.degrees(out["theta_rot"]))
                    # print("k_star:", out["k_star"])
                    # print("num obstacles:", out["obstacles"].shape[0])

                    save_path = os.path.join(vis_dir, f"step_{step_num}.png")
                    save_care_visualization(
                        rgb,
                        depth,
                        waypoints,
                        intrinsics,
                        out,
                        save_path=save_path,
                    )
                
                # save video
                vis_files = sorted(glob.glob(os.path.join(vis_dir, 'step_*.png')))
                video_path = os.path.join(base_dir, 'care_visualization.mp4')
                with imageio.get_writer(video_path, fps=5) as video_writer:
                    for vf in vis_files:
                        frame = imageio.imread(vf)
                        video_writer.append_data(frame)
                print(f"Saved CARE visualization video to {video_path}")
