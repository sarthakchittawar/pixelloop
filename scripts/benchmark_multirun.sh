#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BASE_ROOT="${1:-data/hm3d-0.2/benchmarking2}"
OUT_ROOT="${2:-outputs_local/benchmarking2_batch}"
PIXEL_X_DEFAULT="${3:-192}"
PIXEL_Y_DEFAULT="${4:-120}"
SKIP_MAP_BUILD="${SKIP_MAP_BUILD:-0}"
USE_LOOP_CLOSURE="${USE_LOOP_CLOSURE:-1}"

if [[ "$USE_LOOP_CLOSURE" == "1" ]]; then
  COSTMAP_DIRNAME="${COSTMAP_DIRNAME:-topo_map_outputs_multi}"
  OUT_ROOT="${2:-outputs_local/benchmarking2_batch}"
  RUN_TAG="withLC"
else
  COSTMAP_DIRNAME="${COSTMAP_DIRNAME:-topo_map_outputs_multi_noLC}"
  OUT_ROOT="${2:-outputs_local/benchmarking2_batch_noLC}"
  RUN_TAG="noLC"
fi

echo "Root: $ROOT_DIR"
echo "Scenes root: $BASE_ROOT"
echo "Results root: $OUT_ROOT"
echo "Default goal pixel: ($PIXEL_X_DEFAULT, $PIXEL_Y_DEFAULT)"
echo "Skip map build phase: $SKIP_MAP_BUILD"
echo "Use loop closure: $USE_LOOP_CLOSURE"
echo "Costmap dirname: $COSTMAP_DIRNAME"

mkdir -p "$OUT_ROOT"

SCENES_READY=()

echo ""
echo "============================================================"
echo "PHASE 1: BUILD / UPDATE TOPO MAPS FOR ALL SCENES"
echo "============================================================"

for scene_dir in "$BASE_ROOT"/*; do
  [[ -d "$scene_dir" ]] || continue
  scene_name="$(basename "$scene_dir")"
  echo ""
  echo "============================================================"
  echo "[PHASE 1] Scene: $scene_name"
  echo "============================================================"

  scene_list_file="$scene_dir/scene_list.txt"
  loops_file="$scene_dir/global_loops.txt"
  goal_info_file="$scene_dir/goal_info.json"
  start_states_file="$scene_dir/start_states.json"
  topo_dir="$scene_dir/$COSTMAP_DIRNAME"

  if [[ ! -f "$scene_list_file" ]]; then
    echo "[INFO] scene_list.txt missing, creating from subfolders"
    find "$scene_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort > "$scene_list_file"
  fi

  if [[ "$USE_LOOP_CLOSURE" == "1" && ! -f "$loops_file" ]]; then
    echo "[WARN] Missing $loops_file (required when USE_LOOP_CLOSURE=1), skipping scene"
    continue
  fi
  if [[ ! -f "$goal_info_file" ]]; then
    echo "[WARN] Missing $goal_info_file, skipping scene"
    continue
  fi
  if [[ ! -f "$start_states_file" ]]; then
    echo "[WARN] Missing $start_states_file, skipping scene"
    continue
  fi

  SCENES_READY+=("$scene_dir")

  if [[ "$SKIP_MAP_BUILD" == "1" ]]; then
    echo "[INFO] SKIP_MAP_BUILD=1, skipping map build/update for $scene_name"
    continue
  fi

  mkdir -p "$topo_dir"

  echo "[STEP] Building/ensuring base multi-folder graph"
  base_cmd=(
    pixi run python -m libs.mapper.create_topomap_multi_folder
    scenes.multi_scene=true
    scenes.base_dir="$scene_dir"
    scenes.scene_list_file="$scene_list_file"
    scenes.base_out_dir="$scene_dir"
    scenes.costmap_dirname="$COSTMAP_DIRNAME"
    goal.mode=none
    graph.enable_loop_closure="$USE_LOOP_CLOSURE"
    processing.recreate_graphs=false
    processing.force_recompute_graph=false
  )
  if [[ "$USE_LOOP_CLOSURE" == "1" ]]; then
    base_cmd+=(+graph.loop_closure_file="$loops_file")
  fi
  "${base_cmd[@]}"

  echo "[STEP] Updating graph for all goals in goal_info.json"
  while IFS='|' read -r goal_idx px py; do
    echo "  - goal_idx=$goal_idx pixel=($px,$py)"
    goal_cmd=(
      pixi run python -m libs.mapper.create_topomap_multi_folder
      scenes.multi_scene=true
      scenes.base_dir="$scene_dir"
      scenes.scene_list_file="$scene_list_file"
      scenes.base_out_dir="$scene_dir"
      scenes.costmap_dirname="$COSTMAP_DIRNAME"
      goal.mode=update_graph
      goal.image_idx="$goal_idx"
      goal.pixel_x="$px"
      goal.pixel_y="$py"
      graph.enable_loop_closure="$USE_LOOP_CLOSURE"
      processing.recreate_graphs=false
      processing.force_recompute_graph=false
    )
    if [[ "$USE_LOOP_CLOSURE" == "1" ]]; then
      goal_cmd+=(+graph.loop_closure_file="$loops_file")
    fi
    "${goal_cmd[@]}"
  done < <(
    python3 - "$goal_info_file" "$PIXEL_X_DEFAULT" "$PIXEL_Y_DEFAULT" <<'PY'
import json
import sys

goal_info_path, px_default, py_default = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(goal_info_path, "r") as f:
    data = json.load(f)

if isinstance(data, dict):
    data = [data]

for item in data:
    if isinstance(item, dict):
        gi = int(item.get("goal_image_index", item.get("goal_idx", -1)))
        if gi < 0:
            continue

        # Prefer centroid pixel from goal_info if present.
        cp = item.get("centroid_pixel", None)
        if isinstance(cp, dict) and ("x" in cp and "y" in cp):
            gx = int(cp["x"])
            gy = int(cp["y"])
        elif isinstance(cp, (list, tuple)) and len(cp) >= 2:
            gx = int(cp[0])
            gy = int(cp[1])
        elif "centroid_x" in item and "centroid_y" in item:
            gx = int(item["centroid_x"])
            gy = int(item["centroid_y"])
        else:
            gx = int(item.get("goal_pixel_x", item.get("pixel_x", px_default)))
            gy = int(item.get("goal_pixel_y", item.get("pixel_y", py_default)))

        print("{}|{}|{}".format(gi, gx, gy))
    else:
        gi = int(item)
        print("{}|{}|{}".format(gi, px_default, py_default))
PY
  )

done

echo ""
echo "============================================================"
echo "PHASE 2: RUN NAV FOR EACH START-GOAL PAIR"
echo "============================================================"

if [[ ${#SCENES_READY[@]} -eq 0 ]]; then
  echo "[WARN] No scenes passed phase-1 checks. Skipping phase-2 navigation."
  exit 0
fi

tmp_scene_list="$(mktemp)"
trap 'rm -f "$tmp_scene_list"' EXIT

for scene_dir in "${SCENES_READY[@]}"; do
  basename "$scene_dir" >> "$tmp_scene_list"
done

echo "[STEP] Running run_nav_multi_scene.py with start_states.json for all ready scenes"
pixi run python scripts/run_nav_multi_scene.py \
  scenes.base_dir="$BASE_ROOT" \
  scenes.scene_list_file="$tmp_scene_list" \
  scenes.costmap_dirname="$COSTMAP_DIRNAME" \
  +multi_scene_results_root="$OUT_ROOT" \
  graph.enable_loop_closure="$USE_LOOP_CLOSURE" \
  localizer.use_gt_localization=true \
  +localizer.global_pose_localization=true \
  +localizer.odom_rerank_with_costmap=false

echo ""
echo "Batch completed. Results in: $OUT_ROOT"
