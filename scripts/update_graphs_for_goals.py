import os
import json
import subprocess
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hydra
from omegaconf import DictConfig, OmegaConf

AVAILABLE_GPUS = [1, 2]

child_processes = []

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

@hydra.main(version_base=None, config_path="../configs/mapper", config_name="mapper_config")
def main(cfg: DictConfig):
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Allow CLI overrides for culling modes, factors, etc.
    EDGE_CULLING_MODE = cfg.graph.get("edge_culling_mode", "EMST_SINGLE")
    NODE_CULLING_MODE = cfg.graph.get("node_culling_mode", "NONE")
    NODE_CULLING_FACTOR = cfg.graph.get("node_culling_factor", 10)
    loop_closure = cfg.graph.get('enable_loop_closure', False)
    loop_closure_mode = cfg.graph.get('loop_closure_mode', '')
    SUFFIX = f"_LC_{loop_closure_mode}" if loop_closure else ""


    base_graph_filename = cfg.goal.base_graph_path
    if base_graph_filename is None:
        base_graph_filename = f"graph_base_320x240_EC_{EDGE_CULLING_MODE}_NC_{NODE_CULLING_MODE}_NCF_{NODE_CULLING_FACTOR}{SUFFIX}.pkl.b2s"
    base_dir = cfg.scenes.base_dir
    
    def get_scene_list(cfg):
        scene_list_file = cfg.scenes.scene_list_file
        return open(scene_list_file, 'r').read().split()
    
    scene_list = get_scene_list(cfg)
    scene_list = [os.path.join(base_dir, i) for i in scene_list]
    
    def run_goal_task(scene_name, idx, goal, device_id):
        cmd = [
            sys.executable, "-m", "libs.mapper.create_topomap",
            f"scenes.multi_scene=false",
            f"scenes.scene_name={os.path.basename(scene_name)}",
            f"scenes.base_dir={base_dir}",
            f"scenes.base_out_dir={cfg.scenes.get('base_out_dir', base_dir)}",
            f"goal.mode=update_graph",
            f"goal.image_idx={goal['goal_image_index']}",
            f"goal.pixel_x={goal['centroid_pixel']['x']}",
            f"goal.pixel_y={goal['centroid_pixel']['y']}",
            f"goal.base_graph_path={base_graph_filename}",
            f"graph.enable_loop_closure={str(loop_closure).lower()}",
            f"graph.loop_closure_mode={loop_closure_mode}",
            f"graph.node_culling_factor={NODE_CULLING_FACTOR}",
            f"graph.edge_culling_mode={EDGE_CULLING_MODE}",
            f"graph.node_culling_mode={NODE_CULLING_MODE}",
            f"map_matcher.name={cfg.map_matcher.name}",
            f"model.depth_source={cfg.model.get('depth_source', 'mast3r')}",
            f"model.use_gt_depth={str(cfg.model.get('use_gt_depth', False)).lower()}",
            f"model.use_gt_matches={str(cfg.model.get('use_gt_matches', False)).lower()}",
            f"model.gt_matches_filename={cfg.model.get('gt_matches_filename', 'null')}",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(f"Executing: CUDA_VISIBLE_DEVICES={device_id} {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, env=env)
            child_processes.append(proc)
            proc.wait()
            if proc.returncode != 0:
                print(f"[ERROR] Command failed for scene {scene_name}, goal {idx}: Return code {proc.returncode}")
        except Exception as e:
            print(f"[ERROR] Command failed for scene {scene_name}, goal {idx}: {e}")

    # Collect all tasks
    tasks = []
    for scene_name in scene_list:
        scene_dir = scene_name  # already full path after join above
        goal_info_path = os.path.join(scene_dir, "goal_info.json")
        
        if not os.path.exists(goal_info_path):
            print(f"[WARN] goal_info.json not found for scene {scene_name} at {goal_info_path}, skipping.")
            continue
        
        with open(goal_info_path, 'r') as f:
            try:
                goals = json.load(f)
            except Exception as e:
                print(f"[ERROR] Could not parse {goal_info_path}: {e}")
                continue
        
        for idx, goal in enumerate(goals):
            device_id = AVAILABLE_GPUS[len(tasks) % len(AVAILABLE_GPUS)]
            tasks.append((scene_name, idx, goal, device_id))

    # Parallel execution with per-GPU semaphores
    gpu_capacity = {gpu: AVAILABLE_GPUS.count(gpu) for gpu in set(AVAILABLE_GPUS)}
    gpu_semaphores = {gpu: threading.Semaphore(cap) for gpu, cap in gpu_capacity.items()}

    def guarded_run(task):
        scene_name, idx, goal, device_id = task
        sem = gpu_semaphores[device_id]
        sem.acquire()
        try:
            return run_goal_task(scene_name, idx, goal, device_id)
        finally:
            sem.release()

    max_workers = len(AVAILABLE_GPUS)
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