import json
import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import hydra
import networkx as nx
from natsort import natsorted
from omegaconf import DictConfig
from tqdm import tqdm

from libs.mapper.create_topomap import (
    MapTopological3DPoints,
    get_scene_list,
    read_oracle_lc_pairs,
    timing_manager,
)
from libs.common.graph_utils import load_compressed_graph_chunked


class MultiFolderTopologicalMapper(MapTopological3DPoints):
    """
    Build a single topological graph from multiple map folders.

    The graph uses one global frame index space across all folders:
    - folder 0 frames -> global [0, ..., n0-1]
    - folder 1 frames -> global [n0, ..., n0+n1-1]
    - ...

    Loop-closure edges should be provided in this global index space via:
    graph.loop_closure_file
    """

    def __init__(self, scene_dirs: List[Path], out_dir: Path, cfg: DictConfig):
        if len(scene_dirs) == 0:
            raise ValueError("No scene directories provided")

        self.scene_dirs = scene_dirs
        self.global_idx_to_scene_local: List[Tuple[str, int]] = []

        # Use first folder only for base initialization, then overwrite image list.
        first_img_dir = scene_dirs[0] / "images_fov90"
        if not first_img_dir.exists():
            raise FileNotFoundError(f"Missing images folder: {first_img_dir}")

        super().__init__(str(first_img_dir), str(out_dir), cfg)

        # For multi-folder mode, GT/mapanything depth and GT matches are ambiguous
        # because the original implementation expects a single scene_dir.
        if self.use_gt_matches:
            raise NotImplementedError(
                "Multi-folder graph currently supports matcher-based matching only; "
                "set model.use_gt_matches=false."
            )
        if self.depth_source in ("gt", "mapanything"):
            raise NotImplementedError(
                "Multi-folder graph currently supports model.depth_source='mast3r' only."
            )

        self._set_global_image_list_from_folders()
        self._write_global_index_map()

    def _set_global_image_list_from_folders(self) -> None:
        """Concatenate image paths from all scene folders into one global list."""
        all_paths: List[Path] = []
        all_pairs: List[Tuple[str, int]] = []

        for scene_dir in self.scene_dirs:
            img_dir = scene_dir / "images_fov90"
            if not img_dir.exists():
                print(f"[WARN] Skipping folder without images_fov90: {scene_dir}")
                continue

            img_names = natsorted(os.listdir(img_dir))
            full_paths = [img_dir / name for name in img_names]

            start_idx = self.cfg.processing.subsample_start_idx
            end_idx = self.cfg.processing.subsample_end_idx
            step = self.cfg.processing.subsample_step
            full_paths = full_paths[start_idx:end_idx:step]

            all_paths.extend(full_paths)
            all_pairs.extend((scene_dir.name, i) for i in range(len(full_paths)))

        if len(all_paths) == 0:
            raise ValueError("No images found across provided folders")

        self.img_paths = all_paths
        self.img_names = [p.name for p in all_paths]
        self.global_idx_to_scene_local = all_pairs

        print("\n" + "=" * 80)
        print("MULTI-FOLDER IMAGE SUMMARY")
        print("=" * 80)
        print(f"Folders: {len(self.scene_dirs)}")
        print(f"Total global frames: {len(self.img_paths)}")

    def _write_global_index_map(self) -> None:
        """Save a mapping from global frame index to (scene, local_idx)."""
        mapping_path = self.out_dir / "global_frame_index_map.json"
        payload = {
            "total_frames": len(self.global_idx_to_scene_local),
            "entries": [
                {
                    "global_idx": i,
                    "scene_name": scene_name,
                    "local_idx": local_idx,
                }
                for i, (scene_name, local_idx) in enumerate(self.global_idx_to_scene_local)
            ],
        }

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved global frame index map: {mapping_path}")

    def collect_loop_closure_match_pairs(self):
        """
        Collect loop-closure match pairs from an explicit global index pair file.

        Required config in multi-folder mode:
          graph.loop_closure_file: path to txt with pairs "i j" in GLOBAL frame indices.
        """
        print("Collecting loop-closure match pairs (multi-folder mode)...")
        loop_match_pairs = []

        if not self.cfg.graph.get("enable_loop_closure", False):
            return loop_match_pairs

        loop_file = self.cfg.graph.get("loop_closure_file", None)
        if not loop_file:
            raise ValueError(
                "Multi-folder mode requires graph.loop_closure_file with global frame-index pairs"
            )

        loop_file_path = Path(loop_file)
        if not loop_file_path.is_absolute():
            loop_file_path = Path.cwd() / loop_file_path

        if not loop_file_path.exists():
            raise FileNotFoundError(f"Loop closure file not found: {loop_file_path}")

        loop_pairs = read_oracle_lc_pairs(loop_file_path, len(self.img_paths))
        print(f"Loaded {len(loop_pairs)} global LC image pairs from {loop_file_path}")

        for img_i, img_j in tqdm(loop_pairs, desc="LC matching"):
            matches_i, matches_j = self.get_mast3r_matches(
                self.img_paths[img_i],
                self.img_paths[img_j],
            )

            matches_per_pair = []
            for k in range(len(matches_i)):
                pi = (img_i, int(matches_i[k][0]), int(matches_i[k][1]))
                pj = (img_j, int(matches_j[k][0]), int(matches_j[k][1]))
                matches_per_pair.append((pi, pj))

            matches_per_pair = matches_per_pair[:: self.cfg.graph.node_culling_factor]
            loop_match_pairs.extend(matches_per_pair)

        print(f"Collected {len(loop_match_pairs)} LC match pairs")
        return loop_match_pairs

    def load_base_graph_and_add_goal(self, base_graph_path: Path) -> nx.Graph:
        """Load an existing base graph, add goal node/distances, and return updated graph."""
        if not base_graph_path.exists():
            raise FileNotFoundError(f"Base graph not found: {base_graph_path}")

        print(f"Loading base graph from: {base_graph_path}")
        if str(base_graph_path).endswith(".b2s"):
            self.G = load_compressed_graph_chunked(str(base_graph_path))
        else:
            with open(base_graph_path, "rb") as f:
                self.G = pickle.load(f)

        goal_img_idx = self.cfg.goal.image_idx
        goal_px = self.cfg.goal.pixel_x
        goal_py = self.cfg.goal.pixel_y
        self.compute_distances_to_goal_node(goal_img_idx, goal_px, goal_py)
        return self.G



def make_topo_map_multi_folder(scene_dirs: List[Path], out_dir: Path, cfg: DictConfig) -> nx.Graph:
    """Create one topological map from multiple scene folders."""
    print(f"\n{'=' * 80}")
    print("PROCESSING MULTI-FOLDER TOPO MAP")
    print(f"Folders: {len(scene_dirs)}")
    print(f"Output: {out_dir}")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    timing_manager.enable()

    topo_map = MultiFolderTopologicalMapper(scene_dirs, out_dir, cfg)

    base_filename = topo_map._get_base_graph_filename()
    goal_mode = cfg.goal.get("mode", "none")
    compressed_base_path = out_dir / (base_filename + ".b2s")
    uncompressed_base_path = out_dir / base_filename
    base_graph_exists = compressed_base_path.exists() or uncompressed_base_path.exists()

    if (not topo_map.recreate_graphs) and base_graph_exists and goal_mode in ("config", "update_graph"):
        print("Base graph exists. Reusing it to compute goal distances (no graph rebuild).")
        base_graph_path = compressed_base_path if compressed_base_path.exists() else uncompressed_base_path
        topo_map.load_base_graph_and_add_goal(base_graph_path)

        print("\nSaving goal graph...")
        goal_filename = topo_map._get_goal_graph_filename(cfg.goal.image_idx)
        if cfg.compression.enabled:
            topo_map.save_compressed_graph_chunked(str(out_dir / goal_filename))
        else:
            with open(out_dir / goal_filename, "wb") as f:
                pickle.dump(topo_map.G, f)

        total_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"MULTI-FOLDER COMPLETE in {total_time:.2f}s")
        print(f"{'=' * 80}\n")
        return topo_map.G

    if (not topo_map.recreate_graphs) and base_graph_exists and goal_mode == "none":
        print(f"Base graph already exists at {compressed_base_path if compressed_base_path.exists() else uncompressed_base_path}, skipping.")
        return None

    print("\nCreating topological map from combined folders...")
    t1 = time.time()
    topo_map.create_map_topo()
    print(f"Created topological map in {time.time() - t1:.2f}s")
    print(f"Graph: {topo_map.G.number_of_nodes()} nodes, {topo_map.G.number_of_edges()} edges")

    print("\nSaving base graph...")
    if cfg.compression.enabled:
        topo_map.save_compressed_graph_chunked(str(out_dir / base_filename))
    else:
        with open(out_dir / base_filename, "wb") as f:
            pickle.dump(topo_map.G, f)

    # Keep same goal handling as existing pipeline.
    if goal_mode in ("config", "episode"):
        if goal_mode == "episode":
            raise NotImplementedError(
                "goal.mode='episode' is not supported in multi-folder mode. "
                "Use goal.mode='config' with global image_idx."
            )

        print("\nComputing distances to goal node...")
        t2 = time.time()
        topo_map.compute_distances_to_goal_node()
        print(f"Computed distances in {time.time() - t2:.2f}s")

        print("\nSaving goal graph...")
        goal_filename = topo_map._get_goal_graph_filename()
        if cfg.compression.enabled:
            topo_map.save_compressed_graph_chunked(str(out_dir / goal_filename))
        else:
            with open(out_dir / goal_filename, "wb") as f:
                pickle.dump(topo_map.G, f)

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"MULTI-FOLDER COMPLETE in {total_time:.2f}s")
    print(f"{'=' * 80}\n")

    timing_manager.print_all_function_summaries()
    csv_path = out_dir / "timing_analysis.csv"
    timing_manager.save_function_timings_csv(str(csv_path))
    timing_manager.clear_all_data()

    return topo_map.G


@hydra.main(version_base=None, config_path="../../configs/mapper", config_name="mapper_config")
def main(cfg: DictConfig):
    print("\n" + "=" * 80)
    print("MULTI-FOLDER TOPOLOGICAL MAP CREATION")
    print("=" * 80)
    print(f"\nUsing configuration from: {cfg}")
    print("=" * 80 + "\n")

    scenes = get_scene_list(cfg)
    if len(scenes) == 0:
        raise ValueError("No scene folders found")

    base_out_dir = cfg.scenes.get("base_out_dir", None)
    if base_out_dir is None:
        out_dir = Path(cfg.scenes.base_dir) / cfg.scenes.costmap_dirname
    else:
        out_dir = Path(base_out_dir) / cfg.scenes.costmap_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save which folders were used for reproducibility.
    manifest = {
        "scene_dirs": [str(s) for s in scenes],
        "count": len(scenes),
    }
    with open(out_dir / "multi_folder_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    make_topo_map_multi_folder(scenes, out_dir, cfg)


if __name__ == "__main__":
    main()
