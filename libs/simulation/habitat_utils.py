import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import curses
import datetime
import os
import quaternion
import cv2

try:
    import habitat_sim
except:
    print("habitat_sim not found")

def get_sim_agent(scene_path, 
        update_nav_mesh=False, 
        agent_radius=0.75,
        width=320,
        height=240,
        hfov=90,
        sensor_height=1.5
    ):

    sim_settings = get_sim_settings(scene=scene_path, sensor_height=sensor_height, width=width, height=height, hfov=hfov)
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()

    sim.pathfinder.seed(42)
    agent_state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    if update_nav_mesh:
        # update navmesh to avoid tight spaces
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = agent_radius
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    return sim, agent, action_names

def get_sim_settings(scene, default_agent=0, sensor_height=1.5, width=256, height=256, hfov=90):
    sim_settings = {
        "scene": scene,  # Scene path
        "default_agent": default_agent,  # Index of the default agent
        "sensor_height": sensor_height,  # Height of sensors in meters, relative to the agent
        "width": width,  # Spatial resolution of the observations
        "height": height,
        "hfov": hfov,
        # scene_dataset_config_file is auto-detected by find_annotation_path()
    }
    return sim_settings

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    if "scene_dataset_config_file" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    else:
        anno_config_path = find_annotation_path(settings["scene"])
        if anno_config_path is not None:
            print(f"Annotation file found: {anno_config_path} | {os.path.exists(anno_config_path)=}")
            sim_cfg.scene_dataset_config_file = anno_config_path
        else:
            print(f"Annotation file not found for {settings['scene']}")

    # agent
    hardware_config = habitat_sim.agent.AgentConfiguration()
    # # Modify the attributes you need
    # hardware_config.height = 20  # Setting the height to 1.6 meters
    # hardware_config.radius = 10  # Setting the radius to 0.2 meters
    # discrete actions defined for objectnav task in habitat-lab/habitat/config/habitat/task/objectnav.yaml
    custom_action_dict = {'stop': habitat_sim.ActionSpec(name='move_forward', actuation=habitat_sim.ActuationSpec(amount=0))}
    for k in hardware_config.action_space.keys():
        custom_action_dict[k] = hardware_config.action_space[k]
    custom_action_dict['look_up'] = habitat_sim.ActionSpec(name='look_up',
                                                           actuation=habitat_sim.ActuationSpec(amount=30))
    custom_action_dict['look_down'] = habitat_sim.ActionSpec(name='look_down',
                                                             actuation=habitat_sim.ActuationSpec(amount=30))

    hardware_config.action_space = custom_action_dict
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    # add depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    # Semantic sensor
    # TODO : This can probably be removed for pixelReact
    # semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    # semantic_sensor_spec.uuid = "semantic_sensor"
    # semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    # semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    # semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    # semantic_sensor_spec.hfov = settings["hfov"]

    # hardware_config.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    hardware_config.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [hardware_config])

# Habitat Semantics
def find_annotation_path(scene_path):
    # find split name from among ['train', 'val', 'test', 'minival']
    split = None
    for s in ['train', 'minival', 'val', 'test']:  # TODO: 'val' inside 'minival'
        if s in scene_path:
            split = s
            path_till_split = scene_path.split(split)[0]
            break
    if split is None:
        return None
    else:
        return f"{path_till_split}{split}/hm3d_annotated_{split}_basis.scene_dataset_config.json"