"""BEV (bird's-eye-view) trajectory planner for placing a moving character.

Renders a top-down map of the scene from LiDAR points + tracked object bounding
boxes + ego trajectory, lets you click waypoints, checks them against obstacles,
smooths with Catmull-Rom, and exports a trajectory JSON for render_runner_video.

No GPU / 3DGS needed — works purely from the processed Waymo data on disk.

Usage:
    python tools/bev_trajectory_planner.py \
        --scene_dir data/waymo/processed/training/552 \
        --out_bev outputs/bev_scene552.png \
        --out_traj outputs/trajectories/scene552_traj.json

Controls in the picker window:
    Left click   - add waypoint
    Right click  - undo last waypoint
    Close window - finalize & save trajectory

The output JSON is consumed by:
    python tools/render_runner_video.py --path_json outputs/trajectories/scene552_traj.json ...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bev_planner")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_ego_trajectory(scene_dir: Path) -> np.ndarray:
    """Load ego poses, normalized so frame 0 is at the origin. Returns (N,3) XYZ."""
    ego_dir = scene_dir / "ego_pose"
    paths = sorted(ego_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No ego_pose under {ego_dir}")
    start = np.loadtxt(paths[0])
    pts = []
    for p in paths:
        rel = np.linalg.inv(start) @ np.loadtxt(p)
        pts.append(rel[:3, 3])
    return np.array(pts)


def load_lidar_bev(scene_dir: Path, max_frames: int = 50) -> np.ndarray:
    """Accumulate LiDAR points (downsampled) across frames, in ego-normalized coords.

    Waymo lidar bins are (N,14): origin(3), point(3), flow(3), flow_class(1),
    ground(1), intensity(1), elongation(1), laser_id(1). We use the point column.
    Returns (M, 3) XYZ.
    """
    lidar_dir = scene_dir / "lidar"
    if not lidar_dir.exists():
        return np.zeros((0, 3))
    paths = sorted(lidar_dir.glob("*.bin"))[:max_frames]
    ego_paths = sorted((scene_dir / "ego_pose").glob("*.txt"))
    start = np.loadtxt(ego_paths[0])
    all_pts = []
    for i, lp in enumerate(paths):
        arr = np.memmap(lp, dtype=np.float32, mode="r").reshape(-1, 14)
        pts = np.array(arr[:, 3:6])  # point xyz in lidar frame
        ego = np.linalg.inv(start) @ np.loadtxt(ego_paths[i])
        hom = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
        world = (ego @ hom.T).T[:, :3]
        all_pts.append(world)
    pts = np.concatenate(all_pts, axis=0)
    if len(pts) > 200000:
        idx = np.random.default_rng(0).choice(len(pts), 200000, replace=False)
        pts = pts[idx]
    return pts


def load_obstacles(scene_dir: Path) -> List[dict]:
    """Load tracked object bounding boxes (vehicles, pedestrians, cyclists).

    Returns a list of {center: (x,y), size: (dx,dy), class: str} in ego-normalized
    coords. We sample each object at a representative frame and project its box
    corners to get an occupancy footprint.
    """
    ii_path = scene_dir / "instances" / "instances_info.json"
    ego_dir = scene_dir / "ego_pose"
    if not ii_path.exists():
        return []
    ii = json.load(open(ii_path))
    ego_paths = sorted(ego_dir.glob("*.txt"))
    start = np.loadtxt(ego_paths[0])
    inv_start = np.linalg.inv(start)
    obstacles = []
    for key, info in ii.items():
        cls = info.get("class_name", "?")
        fa = info.get("frame_annotations", {})
        oms = fa.get("obj_to_world", [])
        sizes = fa.get("box_size", [])
        if not oms or not sizes:
            continue
        # sample at middle frame for a stable footprint
        mid = len(oms) // 2
        om = np.array(oms[mid]).reshape(4, 4)
        sz = np.array(sizes[mid])
        # obj_to_world is in absolute Waymo world; normalize to ego frame 0
        rel = inv_start @ om
        cx, cy = float(rel[0, 3]), float(rel[1, 3])
        dx, dy = float(sz[0]), float(sz[1])
        obstacles.append({"center": (cx, cy), "size": (dx, dy), "class": cls})
    return obstacles


# --------------------------------------------------------------------------- #
# BEV rendering
# --------------------------------------------------------------------------- #
def render_bev_image(
    ego_traj: np.ndarray,
    lidar: np.ndarray,
    obstacles: List[dict],
    out_path: Path,
    padding: float = 5.0,
    dpi: int = 120,
) -> Tuple[np.ndarray, dict]:
    """Render a BEV PNG. Returns (image_array, meta) where meta maps pixel<->world."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # compute bounds from ego trajectory + obstacles + lidar
    xs = list(ego_traj[:, 0]) + [ob["center"][0] for ob in obstacles]
    ys = list(ego_traj[:, 1]) + [ob["center"][1] for ob in obstacles]
    if len(lidar):
        xs += lidar[:, 0].tolist()
        ys += lidar[:, 1].tolist()
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    xmin -= padding; xmax += padding; ymin -= padding; ymax += padding

    fig_w = (xmax - xmin) * 0.4
    fig_h = (ymax - ymin) * 0.4
    fig, ax = plt.subplots(figsize=(max(8, fig_w), max(6, fig_h)), dpi=dpi)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    ax.set_title("BEV — left click: add waypoint, right click: undo", fontsize=10)

    # lidar points (road/buildings) — color by height
    if len(lidar):
        z = lidar[:, 2]
        sc = ax.scatter(lidar[:, 0], lidar[:, 1], c=z, cmap="gray_r", s=0.3,
                        alpha=0.5, vmin=-1, vmax=3)

    # obstacles (vehicles=red, pedestrians=yellow, cyclists=orange)
    cls_color = {"Vehicle": "#e74c3c", "Pedestrian": "#f1c40f", "Cyclist": "#e67e22"}
    for ob in obstacles:
        cx, cy = ob["center"]
        dx, dy = ob["size"]
        color = cls_color.get(ob["class"], "#95a5a6")
        rect = Rectangle((cx - dx / 2, cy - dy / 2), dx, dy,
                         linewidth=1, edgecolor=color, facecolor=color, alpha=0.4)
        ax.add_patch(rect)

    # ego trajectory (cyan line)
    ax.plot(ego_traj[:, 0], ego_traj[:, 1], "c-", lw=2, alpha=0.8, label="ego path")
    ax.plot(ego_traj[0, 0], ego_traj[0, 1], "g^", ms=10, label="start")
    ax.plot(ego_traj[-1, 0], ego_traj[-1, 1], "rv", ms=10, label="end")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="#1a1a1a")
    plt.close(fig)

    # meta: world<->pixel transform. We'll use the figure's data coords directly
    # in the interactive picker (matplotlib works in data coords), so meta just
    # records bounds for reference.
    meta = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    return meta


# --------------------------------------------------------------------------- #
# Obstacle collision check
# --------------------------------------------------------------------------- #
def point_in_obstacle(x: float, y: float, obstacles: List[dict], margin: float = 0.5) -> Optional[dict]:
    """Return the obstacle containing (x,y) within margin, or None."""
    for ob in obstacles:
        cx, cy = ob["center"]
        dx, dy = ob["size"]
        if abs(x - cx) < dx / 2 + margin and abs(y - cy) < dy / 2 + margin:
            return ob
    return None


def trajectory_collides(traj: np.ndarray, obstacles: List[dict], margin: float = 0.5) -> List[int]:
    """Return indices of trajectory samples that fall inside an obstacle."""
    bad = []
    for i, (x, y) in enumerate(traj):
        if point_in_obstacle(x, y, obstacles, margin):
            bad.append(i)
    return bad


# --------------------------------------------------------------------------- #
# Trajectory smoothing
# --------------------------------------------------------------------------- #
def catmull_rom(waypts: np.ndarray, n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Catmull-Rom spline through waypoints. Returns (samples XY, arc-length)."""
    pts = np.asarray(waypts, dtype=np.float64)
    p = np.vstack([pts[0:1], pts, pts[-1:1]])
    samples = []
    segs = max(1, len(p) - 3)
    per = max(2, n_samples // segs)
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for t in np.linspace(0, 1, per, endpoint=(i == len(p) - 4)):
            t2, t3 = t * t, t * t * t
            x = 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            samples.append((x, y))
    samples = np.array(samples)
    diffs = np.diff(samples, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    arclen = np.concatenate([[0], np.cumsum(seglen)])
    return samples, arclen


# --------------------------------------------------------------------------- #
# Interactive picker
# --------------------------------------------------------------------------- #
def interactive_pick(
    ego_traj: np.ndarray,
    lidar: np.ndarray,
    obstacles: List[dict],
    meta: dict,
    out_traj: Path,
    n_samples: int = 300,
    margin: float = 0.5,
):
    """Open matplotlib window, let user click waypoints, validate & save trajectory."""
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(meta["xmin"], meta["xmax"])
    ax.set_ylim(meta["ymin"], meta["ymax"])
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    ax.set_title("BEV Trajectory Planner\nLeft=add  Right=undo  Close=save", fontsize=11)

    if len(lidar):
        ax.scatter(lidar[:, 0], lidar[:, 1], c=lidar[:, 2], cmap="gray_r",
                   s=0.3, alpha=0.5, vmin=-1, vmax=3)
    cls_color = {"Vehicle": "#e74c3c", "Pedestrian": "#f1c40f", "Cyclist": "#e67e22"}
    for ob in obstacles:
        cx, cy = ob["center"]; dx, dy = ob["size"]
        c = cls_color.get(ob["class"], "#95a5a6")
        ax.add_patch(Rectangle((cx - dx / 2, cy - dy / 2), dx, dy,
                               linewidth=1, edgecolor=c, facecolor=c, alpha=0.35))
    ax.plot(ego_traj[:, 0], ego_traj[:, 1], "c-", lw=2, alpha=0.8)
    ax.plot(ego_traj[0, 0], ego_traj[0, 1], "g^", ms=12)
    ax.plot(ego_traj[-1, 0], ego_traj[-1, 1], "rv", ms=12)

    clicks: List[Tuple[float, float]] = []
    (wp_plot,) = ax.plot([], [], "o-", color="#2ecc71", ms=10, mfc="white", lw=2)
    (traj_plot,) = ax.plot([], [], "-", color="#3498db", lw=3, alpha=0.8)
    warn_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                        color="#e74c3c", fontsize=10, fontweight="bold")

    def redraw():
        if clicks:
            xs, ys = zip(*clicks)
            wp_plot.set_data(xs, ys)
            samples, _ = catmull_rom(np.array(clicks), n_samples)
            traj_plot.set_data(samples[:, 0], samples[:, 1])
            bad = trajectory_collides(samples, obstacles, margin)
            if bad:
                frac = 100 * len(bad) / len(samples)
                warn_text.set_text(f"⚠ {len(bad)} samples ({frac:.0f}%) collide with obstacles")
            else:
                warn_text.set_text("✓ no collision")
        else:
            wp_plot.set_data([], []); traj_plot.set_data([], [])
            warn_text.set_text("")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            ob = point_in_obstacle(event.xdata, event.ydata, obstacles, margin)
            if ob:
                logger.warning("Waypoint inside %s — ignored", ob["class"])
                return
            clicks.append((event.xdata, event.ydata))
            logger.info("Added waypoint %d: (%.1f, %.1f)", len(clicks), event.xdata, event.ydata)
        elif event.button == 3 and clicks:
            clicks.pop()
            logger.info("Removed last waypoint (%d left)", len(clicks))
        redraw()

    def on_close(_event):
        if len(clicks) < 2:
            logger.warning("Need >= 2 waypoints; got %d. Not saving.", len(clicks))
            return
        samples, arclen = catmull_rom(np.array(clicks), n_samples)
        bad = trajectory_collides(samples, obstacles, margin)
        if bad:
            logger.warning("%d/%d samples collide — saving anyway (check warnings)", len(bad), len(samples))
        # uniform by arc length
        target = np.linspace(0, arclen[-1], n_samples)
        sx = np.interp(target, arclen, samples[:, 0])
        sy = np.interp(target, arclen, samples[:, 1])
        traj = np.stack([sx, sy], axis=1)
        out_traj.parent.mkdir(parents=True, exist_ok=True)
        with open(out_traj, "w") as f:
            json.dump({
                "waypoints": clicks,
                "trajectory": traj.tolist(),
                "total_length": float(arclen[-1]),
                "colliding_samples": len(bad),
            }, f, indent=2)
        logger.info("Trajectory saved: %d pts, %.1f m, %d collisions → %s",
                    len(traj), arclen[-1], len(bad), out_traj)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser("BEV trajectory planner")
    p.add_argument("--scene_dir", required=True, help="processed waymo scene dir, e.g. data/waymo/processed/training/552")
    p.add_argument("--out_bev", default="", help="BEV PNG output (empty=auto: outputs/waymo_omnire/scene<N>/bev/bev.png)")
    p.add_argument("--out_traj", default="", help="trajectory JSON (empty=auto: outputs/waymo_omnire/scene<N>/trajectories/traj.json)")
    p.add_argument("--no_pick", action="store_true", help="only render BEV PNG, skip picker")
    p.add_argument("--n_samples", type=int, default=300, help="smoothed trajectory points")
    p.add_argument("--margin", type=float, default=0.5, help="obstacle collision margin (m)")
    p.add_argument("--lidar_frames", type=int, default=50, help="lidar frames to accumulate")
    args = p.parse_args()

    scene_dir = Path(args.scene_dir)
    # derive scene number and auto-set output paths under the scene's own directory
    scene_num = scene_dir.name.lstrip("0") or "0"
    scene_out = Path(f"outputs/waymo_omnire/scene{scene_num}")
    out_bev = Path(args.out_bev) if args.out_bev else scene_out / "bev" / "bev.png"
    out_traj = Path(args.out_traj) if args.out_traj else scene_out / "trajectories" / "traj.json"
    out_bev.parent.mkdir(parents=True, exist_ok=True)
    out_traj.parent.mkdir(parents=True, exist_ok=True)
    args = p.parse_args()

    logger.info("Loading ego trajectory...")
    ego = load_ego_trajectory(scene_dir)
    logger.info("Ego: %d frames, X[%.1f,%.1f] Y[%.1f,%.1f]",
                len(ego), ego[:, 0].min(), ego[:, 0].max(), ego[:, 1].min(), ego[:, 1].max())

    logger.info("Loading LiDAR BEV (%d frames)...", args.lidar_frames)
    lidar = load_lidar_bev(scene_dir, args.lidar_frames)
    logger.info("LiDAR: %d points", len(lidar))

    logger.info("Loading obstacles...")
    obstacles = load_obstacles(scene_dir)
    logger.info("Obstacles: %d (%s)", len(obstacles),
                {c: sum(1 for o in obstacles if o["class"] == c) for c in set(o["class"] for o in obstacles)})

    meta = render_bev_image(ego, lidar, obstacles, out_bev)
    logger.info("BEV image saved to %s", out_bev)

    if args.no_pick:
        logger.info("--no_pick: done. Use the PNG to plan waypoints manually.")
        return

    interactive_pick(ego, lidar, obstacles, meta, out_traj,
                     n_samples=args.n_samples, margin=args.margin)


if __name__ == "__main__":
    main()
