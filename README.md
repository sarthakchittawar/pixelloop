<h1 align="center">PixelLoop: Shortcut Topological Navigation with Pixel-Level Loops</h1>
<h3 align="center">Accepted to IROS 2026</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2607.12811"><img src="https://img.shields.io/badge/arXiv-2607.12811-b31b1b" alt="arXiv"></a>
  <a href="https://pixelloop-nav.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
  <a href="https://pixelloop-nav.github.io/static/videos/pixelloop_video.mp4"><img src="https://img.shields.io/badge/Video-Demo-red" alt="Video"></a>
  <a href="https://huggingface.co/datasets/sarthakchittawar/pixelloop_hm3d"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="Dataset"></a>
</p>

<p align="center">
  <a href="https://www.iiit.ac.in/">
    <img src="figs/iiit_logo.jpg" alt="IIIT Hyderabad" height="70">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://robotics.iiit.ac.in/">
    <img src="figs/rrc_logo.png"
         alt="Robotics Research Center"
         height="70"
         style="background:white; padding:8px; border-radius:8px;">
  </a>
</p>

<p align="center">
  <img src="figs/teaser.jpg" alt="Teaser Figure" width="70%">
</p>

## Setup

### Clone Repository

Clone the repository with submodules:

```bash
# Clone with submodules
git clone --recursive https://github.com/sarthakchittawar/pixelloop.git
cd mast3r-nav/
```

If you already cloned without `--recursive`, initialize submodules:

```bash
# Initialize and update submodules
git submodule update --init --recursive
```

The repository includes five submodules:
- **libs/matcher/mast3r** - MASt3R for 3D scene reconstruction
- **libs/control/visualnav_transformer** - Visual Navigation Transformer for learned control
- **libs/habitat-lab** - Habitat-Lab v0.2.4
- **libs/habitat-sim** - Habitat-Sim v0.2.4
- **libs/loop_closure/TimeSformer** - TimeSformer backbone used by the SeqVLAD loop-closure pipeline

### Environment Setup

This project uses Pixi for the Python environment, Habitat-Sim build, and model
dependencies.

```bash
# Install Pixi if needed.
curl -fsSL https://pixi.sh/install.sh | bash

# Create the environment.
pixi install

# Build Habitat-Sim/Habitat-Lab and install extra Python deps.
pixi run init

# Optional sanity check.
pixi run verify
```

### Checkpoints

Download the MASt3R and controller checkpoints, then point the configs at them.
The SeqVLAD loop-closure step also needs
`checkpoints/msls_cct384_tr8fz1__seqvlad_seq5.pth`.

```bash
mkdir -p checkpoints/gnm_mast3r_nav

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -O libs/matcher/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

pixi run pip install gdown
pixi run gdown --id 16n6CL2t-asQ_tf8x4ZyJT_Y4UQQedsxh \
  -O checkpoints/msls_cct384_tr8fz1__seqvlad_seq5.pth

wget https://huggingface.co/vanshg1729/mast3r-nav/resolve/main/latest.pth \
  -O checkpoints/gnm_mast3r_nav/latest.pth
```

Set `mast3r_model_path` in `configs/config.yaml`, `model.path` in
`configs/mapper/mapper_config.yaml`, and `load_run` in `configs_corl/gnm_Obj.yaml`
to match these local paths. `load_run` should point to the controller checkpoint
directory, `checkpoints/gnm_mast3r_nav`, which should contain `latest.pth`. The
default MASt3R path is
`libs/matcher/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`.

### Data

The benchmark scenes used by the mapping and inference commands are available
from the Hugging Face dataset repository:

```text
https://huggingface.co/datasets/sarthakchittawar/pixelloop_hm3d
```

Download the dataset using the Hugging Face CLI:

```bash
huggingface-cli download \
    sarthakchittawar/pixelloop_hm3d \
    --repo-type dataset \
    --local-dir data/hm3d-0.2
```

This will create:

```text
data/hm3d-0.2/
├── benchmarking_single_run/
└── benchmarking_multi_run/
```

## SeqVLAD Mapping and Inference

For multi-run experiments, run these commands from the `multi_run` branch (to be updated) and
use `data/hm3d-0.2/benchmarking_multi_run` as `benchmark_root`.

Expected scene layout:

```text
benchmark_root/
└── scene_name/
    ├── images_fov90/
    ├── agent_states.npy
    ├── poses_odom.txt
    ├── goal_info.json
    └── start_states.json
```

Create a text file with one scene name per line:

```text
CETmJJqkhcK
1W61QJVDBqe
```

Then run the overall pipeline:

```bash
# 1. Detect SeqVLAD loop closures.
pixi run python libs/loop_closure/generate_loop_closures.py \
  --base-dir /path/to/benchmark_root \
  --save-file seqvlad_loops_ufm.txt \
  --checkpoint /path/to/checkpoints/msls_cct384_tr8fz1__seqvlad_seq5.pth

# 2. Build base topological graphs.
pixi run python scripts/create_maps_multi_scene.py \
  scenes.base_dir=/path/to/benchmark_root \
  scenes.base_out_dir=/path/to/benchmark_root \
  scenes.scene_list_file=/path/to/scene_list.txt \
  scenes.costmap_dirname=topo_map_outputs \
  graph.enable_loop_closure=true \
  graph.loop_closure_mode=seqvlad \
  graph.node_culling_factor=1

# 3. Add goals from goal_info.json and write per-goal costmaps.
pixi run python scripts/update_graphs_for_goals.py \
  scenes.base_dir=/path/to/benchmark_root \
  scenes.base_out_dir=/path/to/benchmark_root \
  scenes.scene_list_file=/path/to/scene_list.txt \
  scenes.costmap_dirname=topo_map_outputs \
  graph.enable_loop_closure=true \
  graph.loop_closure_mode=seqvlad \
  graph.node_culling_factor=1

# 4. Run navigation inference.
pixi run python scripts/run_nav_multi_scene.py \
  scenes.base_dir=/path/to/benchmark_root \
  scenes.scene_list_file=/path/to/scene_list.txt \
  scenes.costmap_dirname=topo_map_outputs \
  graph.enable_loop_closure=true \
  graph.loop_closure_mode=seqvlad \
  graph.node_culling_factor=1
```

The shortcut scripts use these SeqVLAD settings by default:

```bash
./scripts/create_base_graphs.sh
./scripts/update_all_graphs.sh
./scripts/infer_all.sh
```

### Aggregate Results

After inference finishes, aggregate the latest runs into Markdown and CSV tables:

```bash
pixi run python scripts/aggregate_results.py \
  --base-path /path/to/results_parent \
  --output benchmarking_results
```

`--base-path` should be the parent directory containing result folders such as
`multi_scene_runs_LC_seqvlad`. The script writes `summary.md` and CSV files under
the `--output` directory.

Mapping outputs are written under `scene_name/topo_map_outputs/`. The
important files are `graph_base_*.pkl.b2s`,
`graph_with_distances_to_goal_*.pkl.b2s`, and
`costmaps_*_LC_seqvlad_goalImg*.npz`.

For a single inference run, pass the scene root as `episode_path` and the
costmap file separately:

```bash
pixi run python run_nav.py \
  multi_episode=false \
  episode_path=/path/to/benchmark_root/scene_name \
  +goal_image_index=205 \
  +costmap_file_path=/path/to/benchmark_root/scene_name/topo_map_outputs/costmaps_320x240_EC_EMST_SINGLE_NC_NONE_NCF_1_LC_seqvlad_goalImg205.npz \
  results_dirpath=/path/to/results
```

## Citation

If you use this work or find it useful for your research, please cite our paper:

```bibtex
@inproceedings{chittawar2026pixelloop,
  title={PixelLoop: Shortcut Topological Navigation with Pixel-Level Loops},
  author={Chittawar, Sarthak and Garg, Vansh and Vadali, Aditya and Pandya, Krish and Jayanti, Rohit and Garg, Sourav and Krishna, K Madhava},
  booktitle={Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```

