import numpy as np
import torch
from typing import Optional

try:
    import habitat_sim
    from habitat_sim.utils import common as utils
except:
    print("habitat_sim not found")

def get_sim_settings(scene, default_agent=0, sensor_height=1.5, width=256, height=256, hfov=90):
    sim_settings = {
        "scene": scene,  # Scene path
        "default_agent": default_agent,  # Index of the default agent
        "sensor_height": sensor_height,  # Height of sensors in meters, relative to the agent
        "width": width,  # Spatial resolution of the observations
        "height": height,
        "hfov": hfov,
        "scene_dataset_config_file": "data/hm3d/hm3d_annotated_minival_basis.scene_dataset_config.json",  # Scene dataset config file
    }
    return sim_settings

def get_sim_agent(test_scene, updateNavMesh=False, agent_radius=0.75, width=320, height=240, hfov=90, sensor_height=1.5):
    sim_settings = get_sim_settings(scene=test_scene, width=width, height=height,  hfov=hfov, sensor_height=sensor_height)
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
    sim.pathfinder.seed(42)
    agent_state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    if updateNavMesh:
        # update navmesh to avoid tight spaces
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = agent_radius
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        # sim_topdown_map = sim.pathfinder.get_topdown_view(0.1, 0)

    return sim, agent, action_names

def depth_to_3d_points(depth_img, intrinsics):
    """
    Convert depth image to 3D points using camera intrinsics.
    
    Parameters
    ----------
    depth_img : np.ndarray
        Depth image of shape (H, W) or (H, W, 1)
    intrinsics : torch.Tensor or np.ndarray
        Camera intrinsic matrix of shape (3, 3)
        
    Returns
    -------
    points_3d : np.ndarray
        3D points of shape (H, W, 3) in camera coordinates
    """
    
    # Extract intrinsic parameters
    if isinstance(intrinsics, torch.Tensor):
        K = intrinsics.cpu().numpy()
    else:
        K = intrinsics
        
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal point
    
    # Ensure depth is 2D
    if depth_img.ndim == 3:
        depth_img = depth_img[:, :, 0]
    
    # Get image dimensions
    H, W = depth_img.shape
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert to 3D coordinates using pinhole camera model
    z = depth_img  # Depth values
    x = (u - cx) * z / fx  # X coordinate
    y = (v - cy) * z / fy  # Y coordinate
    
    # Stack coordinates to create 3D points
    points_3d = np.stack([x, y, z], axis=-1)  # Shape: (H, W, 3)
    
    return points_3d

def build_intrinsics(image_width: int,
                     image_height: int,
                     field_of_view_radians_u: float,
                     field_of_view_radians_v: Optional[float] = None,
                     device='cpu') -> torch.Tensor:
    if field_of_view_radians_v is None:
        field_of_view_radians_v = field_of_view_radians_u
    center_u = image_width / 2
    center_v = image_height / 2
    fov_u = (image_width / 2.) / np.tan(field_of_view_radians_u / 2.)
    fov_v = (image_height / 2.) / np.tan(field_of_view_radians_v / 2.)
    intrinsics = np.array([
        [fov_u, 0., center_u],
        [0., fov_v, center_v],
        [0., 0., 1]
    ])
    intrinsics = torch.from_numpy(intrinsics).to(device)
    return intrinsics

def apply_velocity(vel_control, agent, sim, velocity, steer, time_step):
    # Update position
    forward_vec = habitat_sim.utils.quat_rotate_vector(agent.state.rotation, np.array([0, 0, -1.0]))
    new_position = agent.state.position + forward_vec * velocity

    # Update rotation
    new_rotation = habitat_sim.utils.quat_from_angle_axis(steer, np.array([0, 1.0, 0]))
    new_rotation = new_rotation * agent.state.rotation

    # Step the physics simulation
    # Integrate the velocity and apply the transform.
    # Note: this can be done at a higher frequency for more accuracy
    agent_state = agent.state
    previous_rigid_state = habitat_sim.RigidState(
        utils.quat_to_magnum(agent_state.rotation), agent_state.position
    )

    target_rigid_state = habitat_sim.RigidState(
        utils.quat_to_magnum(new_rotation), new_position
    )

    # manually integrate the rigid state
    target_rigid_state = vel_control.integrate_transform(
        time_step, target_rigid_state
    )

    # snap rigid state to navmesh and set state to object/agent
    # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
    end_pos = sim.step_filter(
        previous_rigid_state.translation, target_rigid_state.translation
    )

    # set the computed state
    agent_state.position = end_pos
    agent_state.rotation = utils.quat_from_magnum(
        target_rigid_state.rotation
    )
    agent.set_state(agent_state)

    # Check if a collision occured
    dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
    ).dot()
    dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
    ).dot()

    # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
    # collision _didn't_ happen. One such case is going up stairs.  Instead,
    # we check to see if the the amount moved after the application of the filter
    # is _less_ than the amount moved before the application of the filter
    EPS = 1e-5
    collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
    # run any dynamics simulation
    sim.step_physics(dt=time_step)

    return agent, sim, collided