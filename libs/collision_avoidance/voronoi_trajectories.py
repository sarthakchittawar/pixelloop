"""
voronoi_trajectories.py

Generates Voronoi-based trajectories for an HM3D scene (Habitat-Sim).
Primary functions:
 - sample_navigable_points(sim, n)
 - build_voronoi_graph(points_3d, is_navigable_fn, settings)
 - sample_trajectories_from_graph(G, n_trajs, traj_len)

Dependencies:
  numpy, scipy, shapely, networkx

Notes:
 - The code tries to use common Habitat-Sim pathfinder methods (sim.pathfinder.get_random_navigable_point,
   sim.pathfinder.is_navigable). If your Habitat version differs, supply your own `is_navigable_fn` and/or
   `navigable_points`.
 - It works in X-Z plane (y is ground height). Output trajectories are lists of (x,y,z) so you can use them
   directly with Habitat agents.
"""

import random
from typing import Callable, List, Tuple, Optional

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point, Polygon
import networkx as nx

import habitat_sim

# fixed camera / agent height used for all trajectories
CAMERA_HEIGHT: float = 0.9

# optional plotting dependency
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# -------------------------
# Helpers to sample navigable points from a Habitat sim (best-effort)
# -------------------------
def sample_navigable_points_from_sim(sim, n_samples: int = 1000, seed: Optional[int] = None) -> np.ndarray:
    """
    Try to sample random navigable points using common habitat-sim pathfinder helpers.
    Returns an (N,3) numpy array of (x,y,z) points.
    Uses only those points whose y is within 0.1m of CAMERA_HEIGHT.
    If your sim API differs, call sample_navigable_points_from_sim with your own wrapper or pass
    navigable_points directly to build_voronoi_graph.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    points = []
    # Common habitat-sim function names used across versions:
    pf = getattr(sim, "pathfinder", None) or getattr(sim, "get_pathfinder", lambda: None)()
    if pf is None:
        raise RuntimeError("Simulator does not expose a pathfinder as sim.pathfinder. Provide navigable_points manually.")

    # Many builds of habitat-sim expose get_random_navigable_point or sample_navigable_point
    get_point_fn = getattr(pf, "get_random_navigable_point", None) or getattr(pf, "sample_navigable_point", None)

    print("Using get_point_fn:", get_point_fn)
    if get_point_fn is None:
        # Fallback: try bounding-box grid sampling and pf.is_navigable
        # Try to get navmesh bounds (some versions have pf.get_navmesh_bounds or pf.get_bounds)
        bounds = None
        for nm in ("get_navmesh_bounds", "get_bounds", "bounds"):
            fn = getattr(pf, nm, None)
            if callable(fn):
                try:
                    bounds = fn()
                    break
                except Exception:
                    pass
        if bounds is None:
            raise RuntimeError("Cannot find a method to sample navmesh. Provide navigable_points directly.")
        # Expect bounds as ((minx, miny, minz), (maxx, maxy, maxz)) or similar
        try:
            minb, maxb = bounds
        except Exception:
            raise RuntimeError("Unexpected bounds format returned by pathfinder.")
        minx, miny, minz = minb
        maxx, maxy, maxz = maxb

        is_nav = getattr(pf, "is_navigable", None)
        if is_nav is None:
            raise RuntimeError("Pathfinder has no is_navigable; cannot fallback to grid sampling.")
        attempts = 0
        while len(points) < n_samples and attempts < n_samples * 20:
            rx = random.uniform(minx, maxx)
            rz = random.uniform(minz, maxz)
            # use mid y between miny and maxy as candidate height
            ry = (miny + maxy) / 2.0
            pos = np.array([rx, ry, rz], dtype=np.float32)
            if is_nav(pos):
                points.append(pos)
            attempts += 1
    else:
        # Use get_point_fn repeatedly
        for _ in range(n_samples * 3):  # oversample a bit to survive height filtering
            try:
                pt = get_point_fn()
            except TypeError:
                pt = get_point_fn()
            if pt is None:
                continue
            pt = np.array(pt, dtype=np.float32).reshape(3)
            points.append(pt)
            if len(points) >= n_samples * 3:
                break

    points = np.array(points, dtype=np.float32)
    if points.shape[0] == 0:
        raise RuntimeError("No navigable points sampled. Check your simulator or provide navigable_points.")

    # keep only points whose y is within ±0.1m of CAMERA_HEIGHT
    lower = CAMERA_HEIGHT - 1
    upper = CAMERA_HEIGHT + 1
    mask = (points[:, 1] >= lower) & (points[:, 1] <= upper)
    filtered = points[mask]
    print(f"Unique y values in sampled points: {np.unique(points[:,1])}")

    if filtered.shape[0] == 0:
        raise RuntimeError(
            f"Sampled {points.shape[0]} navigable points but none are within 0.1m of CAMERA_HEIGHT={CAMERA_HEIGHT}."
        )

    print(f"Filtered to {filtered.shape[0]} points out of {points.shape[0]} sampled within ±0.1m of CAMERA_HEIGHT={CAMERA_HEIGHT}.")

    return filtered


# -------------------------
# Build Voronoi graph from navigable points
# -------------------------
def build_voronoi_graph(
    points_3d: np.ndarray,
    is_navigable_fn: Callable[[np.ndarray], bool],
    sample_edge_resolution: float = 0.25,
    min_edge_len: float = 0.5,
    clip_polygon: Optional[Polygon] = None,
) -> nx.Graph:
    """
    points_3d: (N,3) array of navigable (x,y,z) points (sampled across the navmesh).
    Assumed to lie near plane y=CAMERA_HEIGHT (within about 0.1m).
    is_navigable_fn: function accepting 3D point (x,y,z) -> bool, to test whether a point is navigable.
    sample_edge_resolution: how far apart (meters) to sample along each Voronoi edge to check navigability.
    min_edge_len: ignore Voronoi edges shorter than this length.
    clip_polygon: optional shapely Polygon in XZ plane to clip voronoi vertices/edges to nav area.

    Returns: networkx Graph where node attributes 'pos' have 3D coordinates (x,y,z).
    """
    assert points_3d.shape[1] == 3
    # convert to XZ plane for 2D voronoi
    pts_xz = points_3d[:, [0, 2]]

    vor = Voronoi(pts_xz)
    vertices = vor.vertices  # (M,2) in XZ

    G = nx.Graph()

    def to_3d(xz_point: Tuple[float, float]) -> Tuple[float, float, float]:
        x, z = float(xz_point[0]), float(xz_point[1])
        # represent graph nodes on CAMERA_HEIGHT plane
        return (x, CAMERA_HEIGHT, z)

    # Build polygon to clip to (optional)
    if clip_polygon is None:
        # derive bounding polygon from sampled points convex hull
        from shapely.geometry import MultiPoint
        clip_polygon = MultiPoint(pts_xz).convex_hull.buffer(1.0)  # small buffer

    # Process each ridge (edge) in Voronoi
    for (v1_idx, v2_idx) in vor.ridge_vertices:
        # ridge vertices can be -1 for infinite; skip infinite edges
        if v1_idx == -1 or v2_idx == -1:
            continue
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        line = LineString([tuple(v1), tuple(v2)])
        if not line.length or line.length < min_edge_len:
            continue
        # clip to polygon
        if not clip_polygon.intersects(line):
            continue
        clipped = line.intersection(clip_polygon)
        # clipped can be MultiLineString or LineString or Point
        segs = []
        if clipped.is_empty:
            continue
        if isinstance(clipped, LineString):
            segs = [clipped]
        else:
            # MultiLineString or GeometryCollection
            for geom in getattr(clipped, "geoms", [clipped]):
                if isinstance(geom, LineString) and geom.length >= min_edge_len:
                    segs.append(geom)

        for seg in segs:
            # sample along seg and check navigability
            seg_len = seg.length
            n_samples = max(2, int(np.ceil(seg_len / sample_edge_resolution)) + 1)
            sample_pts = [seg.interpolate(float(t) / (n_samples - 1), normalized=True) for t in range(n_samples)]
            # map 2D (x,z) to 3D (x,CAMERA_HEIGHT,z) for nav check
            sample_3d = [np.array((pt.x, CAMERA_HEIGHT, pt.y), dtype=np.float32) for pt in sample_pts]
            # check navigability; use all samples (conservative) or majority (less conservative)
            nav_flags = [is_navigable_fn(s) for s in sample_3d]
            if not any(nav_flags):
                # no part of this segment is navigable
                continue

            # create nodes for segment endpoints (3D)
            a3 = to_3d((seg.coords[0][0], seg.coords[0][1]))
            b3 = to_3d((seg.coords[-1][0], seg.coords[-1][1]))
            # use tuple as node key
            a_key = (round(a3[0], 4), round(a3[1], 4), round(a3[2], 4))
            b_key = (round(b3[0], 4), round(b3[1], 4), round(b3[2], 4))
            if a_key not in G:
                G.add_node(a_key, pos=a3)
            if b_key not in G:
                G.add_node(b_key, pos=b3)
            G.add_edge(a_key, b_key, length=float(np.linalg.norm(np.array(a3) - np.array(b3))))
    return G


# -------------------------
# Longest / global path utilities
# -------------------------
def _find_farthest_pair(G: nx.Graph) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Approximate farthest pair of nodes by all-pairs shortest path on edge 'length'.
    Returns (u, v) node keys.
    """
    # use Dijkstra-all-pairs; for non-huge graphs this is fine
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    max_d = -1.0
    best_pair = None
    for u, dists in all_lengths.items():
        for v, d in dists.items():
            if d > max_d:
                max_d = d
                best_pair = (u, v)
    if best_pair is None:
        # graph empty or single node
        nodes = list(G.nodes)
        if len(nodes) == 1:
            return nodes[0], nodes[0]
        elif len(nodes) >= 2:
            return nodes[0], nodes[1]
        else:
            raise ValueError("Graph has no nodes.")
    return best_pair


def _smooth_polyline(
    pts: List[Tuple[float, float, float]],
    step: float = 0.1,
) -> List[Tuple[float, float, float]]:
    """
    Given a sequence of 3D points, re-sample along the polyline with roughly
    constant spacing `step` to get a smoother trajectory.
    """
    if len(pts) < 2:
        return pts

    seg_lengths = []
    for i in range(len(pts) - 1):
        a = np.array(pts[i], dtype=np.float32)
        b = np.array(pts[i + 1], dtype=np.float32)
        seg_lengths.append(float(np.linalg.norm(b - a)))
    total_len = sum(seg_lengths)
    if total_len == 0.0:
        return pts

    n_samples = max(2, int(np.ceil(total_len / step)) + 1)
    target_ds = np.linspace(0.0, total_len, n_samples)

    # cumulative distances
    cum = [0.0]
    for L in seg_lengths:
        cum.append(cum[-1] + L)

    smoothed: List[Tuple[float, float, float]] = []
    for td in target_ds:
        # find which segment contains td
        i = np.searchsorted(cum, td, side="right") - 1
        if i >= len(seg_lengths):
            smoothed.append(tuple(pts[-1]))
            continue
        seg_start_d = cum[i]
        seg_end_d = cum[i + 1]
        if seg_end_d == seg_start_d:
            alpha = 0.0
        else:
            alpha = (td - seg_start_d) / (seg_end_d - seg_start_d)
        a = np.array(pts[i], dtype=np.float32)
        b = np.array(pts[i + 1], dtype=np.float32)
        p = (1.0 - alpha) * a + alpha * b
        smoothed.append((float(p[0]), float(p[1]), float(p[2])))

    return smoothed


def build_global_longest_trajectory(
    G: nx.Graph,
    smooth_step: float = 0.1,
) -> List[Tuple[float, float, float]]:
    """
    Build one long, smooth trajectory trying to span the whole graph:
      1) find approximate farthest node pair in geodesic distance
      2) take shortest path between them
      3) resample densely along the path for smoothness
    """
    if G.number_of_nodes() == 0:
        return []

    if G.number_of_nodes() == 1:
        # single node graph
        only = next(iter(G.nodes))
        return [G.nodes[only]["pos"]]

    u, v = _find_farthest_pair(G)
    backbone_nodes = nx.shortest_path(G, source=u, target=v, weight="length")
    backbone_pts = [G.nodes[n]["pos"] for n in backbone_nodes]

    # smooth / densify
    smooth_traj = _smooth_polyline(backbone_pts, step=smooth_step)
    return smooth_traj


# -------------------------
# Trajectory sampling utilities
# -------------------------
def sample_trajectories_from_graph(
    G: nx.Graph,
    n_trajs: int = 50,
    traj_len: int = 20,
    method: str = "random_walk",
    seed: Optional[int] = None,
) -> List[List[Tuple[float, float, float]]]:
    """
    Sample trajectories from the Voronoi graph G.
    method: "random_walk", "shortest_path_between_junctions", or "global_longest"
    Returns list of trajectories, where each trajectory is list of (x,y,z).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # special case: one global smooth trajectory spanning the scene
    if method == "global_longest":
        # ignore n_trajs/traj_len and just return one long trajectory
        long_traj = build_global_longest_trajectory(G, smooth_step=0.1)
        return [long_traj]

    trajectories = []
    nodes = list(G.nodes)

    # precompute node degrees to find junctions
    deg = dict(G.degree())
    junctions = [n for n, d in deg.items() if d != 2]  # junction = endpoints or branching points
    if len(junctions) < 2:
        junctions = nodes

    for _ in range(n_trajs):
        if method == "random_walk":
            # pick a random start node (prefer junctions)
            start = random.choice(junctions if random.random() < 0.8 else nodes)
            current = start
            traj = [G.nodes[current]["pos"]]
            for _step in range(traj_len - 1):
                nbrs = list(G.neighbors(current))
                if not nbrs:
                    break
                # prefer not going back immediately (if previous exists)
                if len(traj) >= 2:
                    prev = tuple(map(lambda x: round(x, 4), traj[-2]))
                    # map prev to node key shape (rounded)
                    prev_key = prev
                    choices = [n for n in nbrs if n != prev_key]
                    if not choices:
                        choices = nbrs
                else:
                    choices = nbrs
                current = random.choice(choices)
                traj.append(G.nodes[current]["pos"])
            trajectories.append(traj)

        elif method == "shortest_path_between_junctions":
            # pick two distinct junctions and compute shortest path
            a, b = random.sample(junctions, 2) if len(junctions) >= 2 else random.sample(nodes, 2)
            try:
                sp = nx.shortest_path(G, source=a, target=b, weight="length")
                sp_coords = [G.nodes[n]["pos"] for n in sp]
                # if path longer than traj_len, subsample
                if len(sp_coords) > traj_len:
                    # evenly sample traj_len points
                    idxs = np.linspace(0, len(sp_coords) - 1, traj_len).astype(int)
                    sp_coords = [sp_coords[i] for i in idxs]
                trajectories.append(sp_coords)
            except nx.NetworkXNoPath:
                # fallback to random walk
                trajectories.extend(sample_trajectories_from_graph(G, 1, traj_len, "random_walk", seed))
        else:
            raise ValueError("Unknown sampling method: " + str(method))

    # convert from np arrays to tuples of floats
    cleaned = [[(float(p[0]), float(p[1]), float(p[2])) for p in t] for t in trajectories]
    return cleaned


# -------------------------
# Visualization utilities
# -------------------------
def plot_trajectories_2d(
    points_3d: np.ndarray,
    G: nx.Graph,
    trajectories: List[List[Tuple[float, float, float]]],
    out_path: str,
) -> None:
    """
    Save a top-down (X-Z) plot of the Voronoi graph and trajectories.
    """
    if plt is None:
        print("matplotlib not available; cannot save trajectory image.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # plot seed points in XZ
    if points_3d is not None and len(points_3d) > 0:
        ax.scatter(points_3d[:, 0], points_3d[:, 2], s=1, c="lightgray", label="nav samples")

    # plot Voronoi graph edges
    for u, v in G.edges:
        pu = G.nodes[u]["pos"]
        pv = G.nodes[v]["pos"]
        ax.plot([pu[0], pv[0]], [pu[2], pv[2]], color="blue", linewidth=0.5, alpha=0.5)

    # plot trajectories
    for i, traj in enumerate(trajectories):
        xs = [p[0] for p in traj]
        zs = [p[2] for p in traj]
        ax.plot(xs, zs, color="red", linewidth=1.0, alpha=0.8)
        if traj:
            ax.scatter(xs[0], zs[0], c="green", s=10)  # start
            ax.scatter(xs[-1], zs[-1], c="black", s=10)  # end

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", "box")
    ax.set_title("Voronoi trajectories (top-down XZ)")
    ax.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved trajectory image to {out_path}")


def plot_navigable_points_2d(
    points_3d: np.ndarray,
    out_path: str,
) -> None:
    """
    Save a top-down (X-Z) plot of the filtered navigable points only.
    """
    if plt is None:
        print("matplotlib not available; cannot save navigable-points image.")
        return

    if points_3d is None or len(points_3d) == 0:
        print("No navigable points to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points_3d[:, 0], points_3d[:, 2], s=2, c="black")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", "box")
    ax.set_title("Navigable points (top-down XZ)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved navigable-points image to {out_path}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    """
    Example:
      - If you have a habitat-sim Simulator object `sim`, call sample_navigable_points_from_sim(sim, n)
      - Otherwise build points_3d yourself (e.g., sampling the navmesh)
      - Define is_navigable_fn(pos3) that returns True/False for a 3D point pos3 = np.array([x,y,z])
    """
    import argparse
    import pickle
    import os

    # lazy import of habitat_sim configs/builders for the CLI demo
    from habitat_sim.utils.settings import default_sim_settings
    from habitat_sim.simulator import SimulatorConfiguration, AgentConfiguration, Configuration

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_out", type=str, default="voronoi_trajs.pkl", help="where to save trajectories")
    parser.add_argument("--n_points", type=int, default=1500, help="how many nav samples for Voronoi seeds")
    parser.add_argument("--n_trajs", type=int, default=1, help="how many trajectories to sample")
    parser.add_argument("--traj_len", type=int, default=25, help="waypoints per trajectory (ignored for global_longest)")
    parser.add_argument(
        "--method",
        type=str,
        default="global_longest",
        choices=["random_walk", "shortest_path_between_junctions", "global_longest"],
        help="trajectory sampling method",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_sim", action="store_true", help="try to sample points from sim.pathfinder (requires sim in scope)")
    parser.add_argument("--traj_img", type=str, default="voronoi_trajs.png", help="path to save trajectory image")
    parser.add_argument("--nav_img", type=str, default="navigable_points.png", help="path to save nav-points image")
    args = parser.parse_args()

    # --- FALLBACK: simple synthetic demo on a plane (for testing the script without Habitat) ---
    if not args.use_sim:
        print("No sim provided; running a synthetic demo (plane with obstacles). Replace this with real nav points.")
        # create a grid of floor points and carve out a rectangular obstacle
        xs = np.linspace(-10, 10, 120)
        zs = np.linspace(-10, 10, 120)
        grid = []
        for x in xs:
            for z in zs:
                # simple obstacle: box from -2..2 in x and -4..-1 in z
                if (-2.0 < x < 2.0) and (-4.0 < z < -1.0):
                    continue
                grid.append([x, 0.0, z])
        points_3d = np.array(grid, dtype=np.float32)
        # ensure y=CAMERA_HEIGHT for all synthetic points
        points_3d[:, 1] = CAMERA_HEIGHT

        def is_nav(p3: np.ndarray) -> bool:
            x, y, z = p3.tolist()
            if (-2.0 < x < 2.0) and (-4.0 < z < -1.0):
                return False
            return True
    else:
        # create a minimal Habitat-Sim simulator instance
        # adjust these paths and settings to match your local assets
        sim_settings = default_sim_settings.copy()
        # choose a default scene; user should override by exporting HABITAT_SCENE or editing this string
        sim_settings["scene"] = os.environ.get(
            "HABITAT_SCENE",
            "data/scene_datasets/habitat-test-scenes/apartment_0.glb",
        )
        sim_settings["enable_physics"] = False
        sim_settings["sensor_height"] = CAMERA_HEIGHT
        sim_settings["width"] = 640
        sim_settings["height"] = 480

        sim_cfg = SimulatorConfiguration()
        sim_cfg.scene_id = sim_settings["scene"]
        sim_cfg.enable_physics = sim_settings["enable_physics"]

        agent_cfg = AgentConfiguration()
        # you can further customize agent_cfg if needed

        cfg = Configuration(sim_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        # sample navigable points from this sim
        points_3d = sample_navigable_points_from_sim(sim, args.n_points, seed=args.seed)

        print(f"Sampled {len(points_3d)} navigable points from the simulator.")
        print(f"First 5 points:\n{points_3d[:5]}")

        def is_nav(p3: np.ndarray) -> bool:
            # always query pathfinder at y=CAMERA_HEIGHT
            q = np.array([p3[0], CAMERA_HEIGHT, p3[2]], dtype=np.float32)
            return sim.pathfinder.is_navigable(q)

    print("Building Voronoi graph from", len(points_3d), "seed points...")
    G = build_voronoi_graph(points_3d, is_nav, sample_edge_resolution=15, min_edge_len=0.1)
    print("Graph nodes:", len(G.nodes), "edges:", len(G.edges))

    print("Sampling trajectories...")
    trajs = sample_trajectories_from_graph(
        G,
        n_trajs=args.n_trajs,
        traj_len=args.traj_len,
        method=args.method,
        seed=args.seed,
    )
    print("Generated", len(trajs), "trajectories. Example trajectory length:", len(trajs[0]) if trajs else 0)

    # Save to disk
    out = {"graph": G, "trajectories": trajs}
    with open(args.save_out, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved trajectories and graph to {args.save_out}")
    print("Each trajectory is a list of (x,y,z) waypoints. Load and visualize in Habitat or your viewer.")

    # additionally save an image of the trajectories in XZ
    if args.traj_img:
        plot_trajectories_2d(points_3d, G, trajs, args.traj_img)

    # new: save an image of the raw navigable points only
    if args.nav_img:
        plot_navigable_points_2d(points_3d, args.nav_img)

    # clean up sim if it was created
    if "sim" in locals():
        sim.close()
