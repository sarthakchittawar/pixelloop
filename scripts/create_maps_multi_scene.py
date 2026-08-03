"""
Parallel map creation script for multiple scenes.

Runs libs.mapper.create_topomap in parallel across multiple scenes,
distributing work across available GPUs.

Usage:
    python scripts/create_maps_multi_scene.py [hydra overrides]
    
Examples:
    # Basic usage (uses config defaults)
    python scripts/create_maps_multi_scene.py
    
    # With specific GPU list
    python scripts/create_maps_multi_scene.py gpus=[0,1,2,3]
    
    # With loop closure enabled
    python scripts/create_maps_multi_scene.py graph.enable_loop_closure=true graph.loop_closure_mode=oracle
    
    # Use GT depth
    python scripts/create_maps_multi_scene.py model.use_gt_depth=true
    
    # Use SuperPoint matcher
    python scripts/create_maps_multi_scene.py map_matcher.name=superpoint
"""

import os
import subprocess
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# Default GPUs to use (can be overridden via config)
DEFAULT_GPUS = [1, 2, 1, 2]  # Example: alternating between GPU 1 and 2

# Track child processes for cleanup
child_processes = []


def get_scene_list(cfg):
    """Get list of scenes to process from config."""
    base_dir = Path(cfg.scenes.base_dir)
    
    # Priority 1: scene_list_file
    scene_list_file = cfg.scenes.get("scene_list_file", None)
    if scene_list_file and Path(scene_list_file).exists():
        with open(scene_list_file, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(scene_names)} scenes from {scene_list_file}")
    else:
        # Priority 2: all subdirectories in base_dir
        scene_names = [d.name for d in sorted(base_dir.iterdir()) if d.is_dir()]
        print(f"Found {len(scene_names)} scene directories in {base_dir}")
    
    # Apply start/end/step filtering
    start_idx = cfg.scenes.get("start_idx", 0)
    end_idx = cfg.scenes.get("end_idx", -1)
    step = cfg.scenes.get("step", 1)
    
    if end_idx == -1:
        end_idx = len(scene_names)
    
    scene_names = scene_names[start_idx:end_idx:step]
    print(f"After filtering (start={start_idx}, end={end_idx}, step={step}): {len(scene_names)} scenes")
    
    return scene_names


def run_mapper_for_scene(scene_name: str, device_id: int, extra_overrides: list):
    """
    Run the mapper for a single scene as a subprocess.
    
    Args:
        scene_name: Name of the scene directory
        device_id: GPU device ID to use
        extra_overrides: Additional Hydra overrides to pass through
    """
    # Build command - only pass scene-specific overrides + any user overrides
    cmd = [
        "python", "-m", "libs.mapper.create_topomap",
        # Force single-scene mode
        "scenes.multi_scene=false",
        f"scenes.scene_name={scene_name}",
    ]
    
    # Add any user-specified overrides (forwarded from command line)
    cmd.extend(extra_overrides)
    
    # Set environment with GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    print(f"\n{'='*60}")
    print(f"[Scene: {scene_name}] GPU: {device_id}")
    print(f"Command: CUDA_VISIBLE_DEVICES={device_id} {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        proc = subprocess.Popen(cmd, env=env)
        child_processes.append(proc)
        proc.wait()
        
        if proc.returncode != 0:
            print(f"[ERROR] Mapper failed for scene {scene_name}: Return code {proc.returncode}")
            return (scene_name, False, proc.returncode)
        
        print(f"[SUCCESS] Scene {scene_name} completed")
        return (scene_name, True, 0)
        
    except Exception as e:
        print(f"[ERROR] Mapper failed for scene {scene_name}: {e}")
        return (scene_name, False, -1)


def terminate_all_child_processes():
    """Terminate all running child processes."""
    print("\n[INFO] Terminating all child processes...")
    for proc in child_processes:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception as e:
            print(f"[WARN] Could not terminate process: {e}")


def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n[INFO] Caught signal {sig}, shutting down...")
    terminate_all_child_processes()
    sys.exit(1)


@hydra.main(version_base=None, config_path="../configs/mapper", config_name="mapper_config")
def main(cfg: DictConfig):
    """
    Main entry point for parallel map creation.
    
    Uses mapper_config.yaml as the base config, with overrides for parallel execution.
    """
    print("\n" + "="*80)
    print("PARALLEL MAP CREATION")
    print("="*80)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get GPU list from config or use defaults
    available_gpus = cfg.get("gpus", DEFAULT_GPUS)
    if isinstance(available_gpus, str):
        available_gpus = [int(g) for g in available_gpus.strip("[]").split(",")]
    available_gpus = list(available_gpus)
    print(f"Using GPUs: {available_gpus}")
    
    # Extract user overrides from command line to forward to subprocesses
    # Filter out this script's specific args (gpus, scene filtering)
    skip_prefixes = ("gpus=", "scenes.start_idx=", "scenes.end_idx=", "scenes.step=", 
                     "scenes.multi_scene=", "scenes.scene_name=")
    extra_overrides = [
        arg for arg in sys.argv[1:] 
        if not arg.startswith(skip_prefixes) and "=" in arg
    ]
    if extra_overrides:
        print(f"Forwarding overrides: {extra_overrides}")
    
    # Get scene list
    scene_names = get_scene_list(cfg)
    
    if len(scene_names) == 0:
        print("[ERROR] No scenes to process")
        return
    
    print(f"\nProcessing {len(scene_names)} scenes across {len(available_gpus)} GPUs")
    print(f"Config summary:")
    print(f"  - Matcher: {cfg.get('map_matcher', {}).get('name', 'mast3r')}")
    print(f"  - Use GT depth: {cfg.get('model', {}).get('use_gt_depth', False)}")
    print(f"  - Loop closure: {cfg.get('graph', {}).get('enable_loop_closure', False)}")
    if cfg.get('graph', {}).get('enable_loop_closure', False):
        print(f"  - LC mode: {cfg.get('graph', {}).get('loop_closure_mode', 'N/A')}")
    print()
    
    # Build task list with GPU assignments
    tasks = []
    for i, scene_name in enumerate(scene_names):
        device_id = available_gpus[i % len(available_gpus)]
        tasks.append((scene_name, device_id, extra_overrides))
    
    # Process in parallel with per-GPU concurrency based on duplicates in available_gpus
    gpu_capacity = {gpu: available_gpus.count(gpu) for gpu in set(available_gpus)}
    gpu_semaphores = {gpu: threading.Semaphore(cap) for gpu, cap in gpu_capacity.items()}

    def guarded_run(task):
        device_id = task[1]
        sem = gpu_semaphores[device_id]
        sem.acquire()
        try:
            return run_mapper_for_scene(*task)
        finally:
            sem.release()

    max_workers = min(len(available_gpus), len(tasks))
    results = {}

    print(f"Starting parallel execution with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(guarded_run, task): task[0]
            for task in tasks
        }

        try:
            for future in as_completed(futures):
                scene_name = futures[future]
                try:
                    result = future.result()
                    results[result[0]] = result[1]
                except Exception as e:
                    print(f"[ERROR] Exception for scene {scene_name}: {e}")
                    results[scene_name] = False

        except (KeyboardInterrupt, SystemExit):
            print("\n[INFO] Interrupted, terminating...")
            terminate_all_child_processes()
            sys.exit(1)
    
    # Summary
    successful = sum(results.values())
    failed = len(results) - successful
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total scenes:      {len(results)}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"Success rate:      {100*successful/len(results):.1f}%")
    
    if failed > 0:
        print(f"\nFailed scenes:")
        for scene, success in results.items():
            if not success:
                print(f"  - {scene}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
