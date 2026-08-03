import os
import json
import subprocess
import uuid
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hydra
from omegaconf import DictConfig, OmegaConf

AVAILABLE_GPUS = [1, 2]

child_processes = []


def get_scene_list(cfg):
    scene_list_file = cfg.scenes.scene_list_file
    return open(scene_list_file, 'r').read().split()


def get_anchor_episode_path(scene_dir: str):
    """Return an episode subfolder path to use as run_nav episode_path."""
    def is_episode_dir(path: str):
        return (
            os.path.exists(os.path.join(path, "agent_states.npy"))
            or os.path.exists(os.path.join(path, "poses_odom.txt"))
        ) and os.path.isdir(os.path.join(path, "images_fov90"))

    if is_episode_dir(scene_dir):
        return scene_dir

    scene_list_path = os.path.join(scene_dir, "scene_list.txt")
    if os.path.exists(scene_list_path):
        with open(scene_list_path, "r") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        for name in names:
            ep = os.path.join(scene_dir, name)
            if os.path.isdir(ep) and is_episode_dir(ep):
                return ep
        if names:
            ep = os.path.join(scene_dir, names[0])
            if os.path.isdir(ep):
                return ep

    # Fallback: first immediate episode-like subdirectory.
    subdirs = sorted([
        os.path.join(scene_dir, d)
        for d in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, d))
    ])
    episode_subdirs = [d for d in subdirs if is_episode_dir(d)]
    return episode_subdirs[0] if episode_subdirs else scene_dir


def resolve_scene_costmap_path(scene_dir: str, goal_img_idx: int, costmap_dirname: str):
    """Find the scene-level global costmap for the given goal index."""
    topo_dir = os.path.join(scene_dir, costmap_dirname)
    if not os.path.isdir(topo_dir):
        return None
    matches = sorted([
        os.path.join(topo_dir, f)
        for f in os.listdir(topo_dir)
        if f.endswith(".npz") and ("goalImg{}".format(goal_img_idx) in f)
    ])
    return matches[0] if matches else None

def get_common_filename_params(cfg) -> str:
    """Generate common filename parameters used across all graph/costmap filenames.
    
    Returns:
        String containing: {w}x{h}_EC_{ec_mode}_NC_{nc_mode}_NCF_{nc_factor}_{dist}_BS_{block_str}_{match_feature_str}
    """
    # Image and graph configuration
    ec_mode = cfg.graph.edge_culling_mode
    nc_mode = cfg.graph.node_culling_mode
    nc_factor = cfg.graph.node_culling_factor
    w, h = cfg.image.width, cfg.image.height
    mast3r_subsample = cfg.model.subsample_or_initxy1
    
    # MASt3R matcher params
    dist = cfg.model.dist
    if dist is None:
        dist = "NONE"
    else:
        dist = str.upper(cfg.model.dist)

    block_size = cfg.model.block_size
    match_feature_type = cfg.model.match_feature_type
    
    # Format match feature type
    if match_feature_type == "descriptor":
        match_feature_str = "DESC"
    elif match_feature_type == "pointmap":
        match_feature_str = "PTS3D"
    else:
        match_feature_str = match_feature_type  # fallback
    
    # Format block size
    if block_size is None:
        block_str = "NONE"
    else:
        block_str = str(block_size)
    
    if block_size is None and dist.lower() == 'dot' and mast3r_subsample == 8:
        # return the old naming when using the default values
        base = f"{w}x{h}_EC_{ec_mode}_NC_{nc_mode}_NCF_{nc_factor}"
    else:
        base = f"{w}x{h}_EC_{ec_mode}_NC_{nc_mode}_NCF_{nc_factor}_{dist}_BS_{block_str}_{match_feature_str}_SUB{mast3r_subsample}"

    # Append depth source tag if non-default (mirrors create_topomap._get_common_filename_params)
    depth_source = cfg.model.get("depth_source", "mast3r")
    if depth_source not in ("mast3r", None):
        base += f"_DEPTH_{depth_source}"

    return base

def get_costmap_filename(cfg, goal_img_idx: int = None):
    """Generate descriptive filename for costmaps"""
    common_params = get_common_filename_params(cfg)

    filename = f"costmaps_{common_params}"

    if cfg.graph.get("enable_loop_closure", False):
        lc_mode = cfg.graph.get("loop_closure_mode", "subsample")
        if lc_mode == "mapanything":
            filename += f"_DEPTH_mapanything_LC_{lc_mode}"
        else:
            filename += f"_LC_{lc_mode}"

    if goal_img_idx is not None:
        filename += f"_goalImg{goal_img_idx}"
    
    filename += ".npz"
    
    return filename

def run_for_goal(scene_name, scene_dir, idx, goal, cfg, config_path, run_nav_path, device_id=0, edge_culling_mode="EMST_SINGLE", node_culling_mode="NONE", node_culling_factor=10, suffix="noLC", start_indices=None):
    goal_img_idx = goal['goal_image_index']
    # costmap_filename = f"costmaps_320x240_EC_{edge_culling_mode}_NC_{node_culling_mode}_NCF_{node_culling_factor}"
    costmap_filename = get_costmap_filename(cfg, goal_img_idx)
    # if suffix.startswith("LC_"):
    #     costmap_filename += f"_{suffix}"
    # costmap_filename += f"_goalImg{goal_img_idx}.npz"
    episode_path = get_anchor_episode_path(scene_dir)
    costmap_path = resolve_scene_costmap_path(scene_dir, goal_img_idx, cfg.scenes.get('costmap_dirname', 'topo_map_outputs'))
    matcher = cfg.matcher.get("name", "mast3r")

    depth_source = cfg.model.get('depth_source', 'gt' if cfg.model.get('use_gt_depth', False) else 'mast3r')
    results_root = cfg.get("multi_scene_results_root", "outputs_local/multi_scene_runs")
    cmd = [
        sys.executable, run_nav_path,
        f"episode_path={episode_path}",
        f"+goal_image_index={goal_img_idx}",
        f"costmap_dirname={cfg.scenes.get('costmap_dirname', 'topo_map_outputs')}",
        f"device={'cuda' if device_id >=0 else 'cpu'}",
        f"results_dirpath={results_root}_{suffix}/{scene_name}/goalImg{goal_img_idx}",
        f"graph.enable_loop_closure={str(cfg.graph.get('enable_loop_closure', False)).lower()}",
        f"graph.loop_closure_mode={cfg.graph.get('loop_closure_mode', '')}",
        f"graph.node_culling_factor={node_culling_factor}",
        f"model.depth_source={depth_source}",
        f"task_type=EC_{edge_culling_mode}_NC_{node_culling_mode}_NCF_{node_culling_factor}_DEPTH_{depth_source}_MATCHES_{'gt' if cfg.model.get('use_gt_matches', False) else matcher}",
        f"matcher={matcher}",
        "+force_global_start_indices=true",
    ]
    if costmap_path is not None:
        cmd.append(f"+costmap_file_path={costmap_path}")
    else:
        cmd.append(f"costmap_filename={costmap_filename}")
        cmd.append(f"costmap_base_dir={os.path.dirname(scene_dir)}")
    if start_indices is not None and len(start_indices) > 0:
        starts_str = ",".join(str(int(s)) for s in start_indices)
        cmd.append("start_state_mode=fixed_idx")
        cmd.append(f"start_indices=[{starts_str}]")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"Executing: CUDA_VISIBLE_DEVICES={device_id} {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, env=env)
        child_processes.append(proc)
        proc.wait()
        if proc.returncode != 0:
            print(f"[ERROR] run_nav.py failed for scene {scene_name}, goal {idx}: Return code {proc.returncode}")
    except Exception as e:
        print(f"[ERROR] run_nav.py failed for scene {scene_name}, goal {idx}: {e}")


def terminate_all_child_processes():
    print("\n[INFO] Terminating all child processes...")
    for proc in child_processes:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception as e:
            print(f"[WARN] Could not terminate process: {e}")

def signal_handler(sig, frame):
    print(f"\n[INFO] Caught signal {sig}, shutting down...")
    terminate_all_child_processes()
    sys.exit(1)


def load_start_states(scene_dir: str):
    """Load start_states.json for a scene if available."""
    path = os.path.join(scene_dir, "start_states.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Main config values: cfg.<key>
    # Mapper config values: cfg.mapper.<key>
    base_dir = cfg.scenes.base_dir
    edge_culling_mode = cfg.graph.edge_culling_mode
    node_culling_mode = cfg.graph.node_culling_mode
    node_culling_factor = cfg.graph.node_culling_factor
    if cfg.graph.get('enable_loop_closure', False):
        loop_closure_mode = cfg.graph.get('loop_closure_mode', '')
        suffix = f"LC_{loop_closure_mode}"
    else:
        suffix = "noLC"

    suffix += "_final"

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    scene_list = get_scene_list(cfg)
    scene_list = [os.path.join(base_dir, i) for i in scene_list]
    # scene_list = ["1W61QJVDBqe"]
    tasks = []
    for scene_dir in scene_list:
        # Uncomment to filter scenes
        # if scene_name != "1W61QJVDBqe":
        #     continue
        scene_name = os.path.basename(scene_dir)
        scene_dir = os.path.join(base_dir, scene_name)
        goal_info_path = os.path.join(scene_dir, "goal_info.json")
        if not os.path.exists(goal_info_path):
            print(f"[WARN] goal_info.json not found for scene {scene_name}, skipping.")
            continue
        with open(goal_info_path, 'r') as f:
            try:
                goals = json.load(f)
            except Exception as e:
                print(f"[ERROR] Could not parse {goal_info_path}: {e}")
                continue
        starts_by_goal = load_start_states(scene_dir)
        for idx, goal in enumerate(goals):
            goal_img_idx = int(goal.get("goal_image_index", -1)) if isinstance(goal, dict) else int(goal)
            start_indices = starts_by_goal.get(goal_img_idx, None)
            if start_indices is None or len(start_indices) == 0:
                print(f"[INFO] No start_states for scene={scene_name}, goalImg={goal_img_idx}; skipping")
                continue
            tasks.append((scene_name, scene_dir, idx, goal, cfg, cfg.get("config_path", "configs/config.yaml"), cfg.get("run_nav_path", "run_nav.py"), 0, edge_culling_mode, node_culling_mode, node_culling_factor, suffix, start_indices))

    for i, task in enumerate(tasks):
        device_id = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
        task_list = list(task)
        task_list[7] = device_id
        tasks[i] = tuple(task_list)

    # Parallel execution with per-GPU concurrency based on duplicates in AVAILABLE_GPUS
    gpu_capacity = {gpu: AVAILABLE_GPUS.count(gpu) for gpu in set(AVAILABLE_GPUS)}
    gpu_semaphores = {gpu: threading.Semaphore(cap) for gpu, cap in gpu_capacity.items()}

    def guarded_run(task):
        device_id = task[7]  # device_id position in tuple
        sem = gpu_semaphores[device_id]
        sem.acquire()
        try:
            return run_for_goal(*task)
        finally:
            sem.release()

    max_workers = min(len(AVAILABLE_GPUS), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(guarded_run, task) for task in tasks]
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Exception in parallel execution: {e}")
        except (KeyboardInterrupt, SystemExit):
            print("\n[INFO] KeyboardInterrupt/SystemExit received in main, terminating...")
            terminate_all_child_processes()
            sys.exit(1)

if __name__ == "__main__":
    main()
