"""Real-time 3D trajectory previewer with dynamic obstacles and 3D clicking.

Features:
  - 3DGS background rendered live via gsplat (nerfview)
  - Dynamic obstacles: vehicles/pedestrians/cyclists move with the time slider
  - 3D click-to-add waypoints: click in the scene to place trajectory points
  - Trajectory length vs video duration guidance
  - Collision detection (3D AABB, time-synchronized)
  - Playback controls (play/pause, speed, scrub, FPS selection)

Workflow:
  1. Launch previewer, click in 3D scene to add waypoints
  2. Scrub the time slider to check collisions at different moments
  3. Export trajectory JSON when satisfied
  4. Run render_runner_video.py for final video

Usage:
    conda activate drivestudio
    python tools/trajectory_previewer.py \\
        --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \\
        --scene_dir data/waymo/processed/training/023 \\
        --port 8080

    # Load existing trajectory for editing
    python tools/trajectory_previewer.py \\
        --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \\
        --scene_dir data/waymo/processed/training/023 \\
        --path_json outputs/waymo_omnire/scene23/trajectories/traj.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from tools.gait_utils import compute_gait_params, GaitParams


# ========================================================================= #
#  Data loading
# ========================================================================= #
def load_ego_trajectory(scene_dir: Path) -> np.ndarray:
    ego_dir = scene_dir / "ego_pose"
    paths = sorted(ego_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No ego_pose under {ego_dir}")
    start = np.loadtxt(paths[0])
    inv_start = np.linalg.inv(start)
    traj = []
    for p in paths:
        m = np.loadtxt(p)
        rel = inv_start @ m
        traj.append(rel[:3, 3])
    return np.array(traj)


def load_dynamic_obstacles(scene_dir: Path) -> List[dict]:
    """Load per-frame obstacle bounding boxes.

    Returns list of {class, frames:[(frame_idx, cx, cy, cz, dx, dy, dz, heading), ...]}.
    Each obstacle has a full per-frame trajectory.
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
        fis = fa.get("frame_idx", [])
        oms = fa.get("obj_to_world", [])
        sizes = fa.get("box_size", [])
        if not oms or not sizes:
            continue
        frames = []
        for fi, om, sz in zip(fis, oms, sizes):
            om = np.array(om).reshape(4, 4)
            sz = np.array(sz)
            rel = inv_start @ om
            cx, cy, cz = float(rel[0, 3]), float(rel[1, 3]), float(rel[2, 3])
            dx, dy, dz = float(sz[0]), float(sz[1]), float(sz[2])
            heading = float(np.arctan2(rel[1, 0], rel[0, 0]))
            frames.append((fi, cx, cy, cz, dx, dy, dz, heading))
        # Compute total displacement to flag moving vs static
        if len(frames) >= 2:
            f0, fn = frames[0], frames[-1]
            disp = math.hypot(fn[1] - f0[1], fn[2] - f0[2])
        else:
            disp = 0.0
        obstacles.append({"class": cls, "frames": frames, "displacement": disp})
    return obstacles


def sample_obstacle_at_frame(ob: dict, frame_idx: int) -> Optional[dict]:
    """Get obstacle state at a given frame index (nearest available frame)."""
    frames = ob["frames"]
    if not frames:
        return None
    # find nearest frame
    best = min(frames, key=lambda f: abs(f[0] - frame_idx))
    fi, cx, cy, cz, dx, dy, dz, heading = best
    return {"center": (cx, cy, cz), "size": (dx, dy, dz), "heading": heading,
            "class": ob["class"], "frame_idx": fi}


def catmull_rom_spline(points: np.ndarray, n_samples: int = 300) -> np.ndarray:
    """Smooth waypoints with Catmull-Rom spline, arc-length resampled."""
    if len(points) < 2:
        return points.copy()
    if len(points) == 2:
        return np.linspace(points[0], points[1], n_samples)
    # Add phantom endpoints
    pts = np.vstack([points[0], points, points[-1]])
    samples = []
    n_segments = len(pts) - 3
    for i in range(n_segments):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for t in np.linspace(0, 1, max(2, n_samples // n_segments), endpoint=False):
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            samples.append([x, y])
    samples.append(points[-1].tolist())
    samples = np.array(samples)
    # Arc-length resample
    seg_lens = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    target = np.linspace(0, total, n_samples)
    result = np.column_stack([
        np.interp(target, cum, samples[:, 0]),
        np.interp(target, cum, samples[:, 1]),
    ])
    return result


def trajectory_length(traj: np.ndarray) -> float:
    if len(traj) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def trajectory_yaw_at(traj: np.ndarray, idx: int) -> float:
    if len(traj) < 2:
        return 0.0
    i0 = max(0, idx - 1)
    i1 = min(len(traj) - 1, idx + 1)
    dx = traj[i1, 0] - traj[i0, 0]
    dy = traj[i1, 1] - traj[i0, 1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return float(np.arctan2(dy, dx))


def check_collision_3d(char_min: np.ndarray, char_max: np.ndarray,
                       obstacles: List[dict], margin: float = 0.3) -> List[dict]:
    hits = []
    for ob in obstacles:
        cx, cy, cz = ob["center"]
        dx, dy, dz = ob["size"]
        ob_min = np.array([cx - dx / 2, cy - dy / 2, cz - dz / 2])
        ob_max = np.array([cx + dx / 2, cy + dy / 2, cz + dz / 2])
        overlap = np.all(char_min - margin < ob_max) and np.all(char_max + margin > ob_min)
        if overlap:
            hits.append(ob)
    return hits


# ========================================================================= #
#  Main
# ========================================================================= #
def main():
    parser = argparse.ArgumentParser("Real-time 3D trajectory previewer (dynamic)")
    parser.add_argument("--resume_from", required=True)
    parser.add_argument("--scene_dir", required=True,
                        help="e.g. data/waymo/processed/training/023")
    parser.add_argument("--path_json", default=None,
                        help="load existing trajectory for editing")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--char_size", default="0.6,0.4,1.8",
                        help="character bbox W,D,H meters")
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--render_fps", type=float, default=10.0,
                        help="output video FPS (for length guidance)")
    parser.add_argument("--max_obstacles", type=int, default=80,
                        help="max dynamic obstacles to show (perf limit)")
    parser.add_argument("--cycle_stride", type=float, default=2.6,
                        help="meters per gait cycle (2 steps). 2.6m = 1.3m/step")
    args = parser.parse_args()

    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_size = tuple(float(x) for x in args.char_size.split(","))
    scene_dir = Path(args.scene_dir)

    # 1. Load data
    print("[1/3] Loading scene data...")
    n_frames = len(list((scene_dir / "ego_pose").glob("*.txt")))
    print(f"  {n_frames} frames in scene")
    obstacles_all = load_dynamic_obstacles(scene_dir)
    # Sort by displacement, keep the most relevant ones (moving + near road)
    obstacles_all.sort(key=lambda o: o["displacement"], reverse=True)
    obstacles_show = obstacles_all[:args.max_obstacles]
    n_moving = sum(1 for o in obstacles_show if o["displacement"] > 0.5)
    print(f"  {len(obstacles_all)} total obstacles, showing {len(obstacles_show)} ({n_moving} moving)")

    # Initial trajectory
    if args.path_json and os.path.exists(args.path_json):
        d = json.load(open(args.path_json))
        waypoints = [list(w) for w in d["waypoints"]]
        print(f"  loaded {len(waypoints)} waypoints from {args.path_json}")
    else:
        waypoints = []
    traj = catmull_rom_spline(np.array(waypoints), 300) if len(waypoints) >= 2 else np.zeros((0, 2))

    # 2. Build trainer
    print("[2/3] Loading DriveStudio checkpoint...")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    trainer.set_eval()

    # 3. Viser server
    print(f"[3/3] Starting viser on port {args.port}...")
    import viser
    import nerfview

    server = viser.ViserServer(port=args.port, verbose=False)
    server.scene.set_up_direction("+z")
    viewer = nerfview.Viewer(
        server=server, render_fn=trainer._viewer_render_fn, mode="rendering")

    # ---- State ----
    state = {
        "waypoints": waypoints,
        "traj": traj,
        "t": 0.0,          # trajectory progress [0,1]
        "frame": 0,        # scene frame index [0, n_frames)
        "playing": False,
        "speed": 0.3,
        "last_time": time.time(),
    }

    def get_obstacles_at_frame(frame_idx: int) -> List[dict]:
        """Sample all obstacles at a given frame."""
        result = []
        for ob in obstacles_show:
            s = sample_obstacle_at_frame(ob, frame_idx)
            if s is not None:
                result.append(s)
        return result

    # ---- Static trajectory line (rebuilt on waypoint change) ----
    def rebuild_trajectory():
        """Recompute spline + redraw trajectory line and waypoint markers."""
        server.scene.remove_by_name("trajectory")
        for i in range(100):  # clear old waypoint markers
            server.scene.remove_by_name(f"wp_{i}")
        if len(state["waypoints"]) >= 2:
            state["traj"] = catmull_rom_spline(np.array(state["waypoints"]), 300)
        else:
            state["traj"] = np.zeros((0, 2))
        traj = state["traj"]
        if len(traj) >= 2:
            pts = np.column_stack([traj, np.full(len(traj), args.ground_z + 0.05)])
            server.scene.add_line_segments(
                "trajectory", points=pts.reshape(-1, 2, 3),
                colors=np.array([[0.0, 1.0, 0.0]] * len(pts)).reshape(-1, 2, 3),
                line_width=3.0)
        for i, wp in enumerate(state["waypoints"]):
            server.scene.add_icosphere(
                f"wp_{i}", radius=0.3,
                position=(wp[0], wp[1], args.ground_z + 0.1),
                color=(255, 200, 0))
        update_length_label()

    # ---- Dynamic obstacles (rebuilt on frame change) ----
    obstacle_handles: List = []

    def rebuild_obstacles(frame_idx: int):
        """Remove old obstacle boxes and add new ones at frame_idx."""
        nonlocal obstacle_handles
        for h in obstacle_handles:
            try:
                h.remove()
            except Exception:
                pass
        obstacle_handles = []
        cls_colors = {
            "Vehicle": (220, 60, 60),
            "Pedestrian": (240, 200, 60),
            "Cyclist": (230, 140, 40),
        }
        obs = get_obstacles_at_frame(frame_idx)
        for i, ob in enumerate(obs):
            cx, cy, cz = ob["center"]
            dx, dy, dz = ob["size"]
            color = cls_colors.get(ob["class"], (150, 150, 150))
            h = server.scene.add_box(
                f"dyn_obs_{i}",
                position=(cx, cy, cz + dz / 2),
                dimensions=(dx, dy, dz),
                color=color, opacity=0.3)
            obstacle_handles.append(h)

    # ---- Character marker ----
    char_handle = server.scene.add_box(
        "character",
        position=(0, 0, args.ground_z + char_size[2] / 2),
        dimensions=char_size, color=(60, 120, 255), opacity=0.7)

    # ---- Labels (viser labels are immutable, so re-add on change) ----
    label_state = {"collision": None, "length": None}

    def make_label(name: str, text: str, pos: Tuple):
        if label_state.get(name) is not None:
            label_state[name].remove()
        h = server.scene.add_label(name, text=text, position=pos)
        label_state[name] = h

    def update_collision_label(x, y, z, n_hits):
        text = f"⚠ COLLISION x{n_hits}!" if n_hits > 0 else "✓ clear"
        make_label("collision_lbl", text, (x, y, z + 2.5))

    def update_length_label():
        """Show gait-matched parameters: length, steps, speed, anim_speed."""
        server.scene.remove_by_name("length_lbl")
        tl = trajectory_length(state["traj"])
        if tl < 0.1:
            server.scene.add_label(
                "length_lbl", text="(添加路径点以查看步频参数)",
                position=(0, 0, args.ground_z + 8))
            gui_gait_info.value = "(添加路径点后显示步频参数)"
            return
        gp = compute_gait_params(
            trajectory_length=tl,
            n_video_frames=n_frames,
            fps=args.render_fps,
            cycle_stride=args.cycle_stride,
        )
        text = (
            f"len={gp.trajectory_length:.1f}m  "
            f"steps={gp.n_steps:.0f}  "
            f"speed={gp.char_speed:.1f}m/s  "
            f"anim_speed={gp.anim_speed:.2f}  "
            f"{gp.speed_assessment()}"
        )
        server.scene.add_label(
            "length_lbl", text=text,
            position=(0, 0, args.ground_z + 8))
        # Update GUI text with detailed info
        gui_gait_info.value = (
            f"len={gp.trajectory_length:.1f}m | "
            f"{gp.n_steps:.0f}步 | "
            f"{gp.char_speed:.1f}m/s | "
            f"步频{gp.step_freq:.1f}Hz | "
            f"anim_speed={gp.anim_speed:.3f} | "
            f"{gp.speed_assessment()}"
        )

    # ---- Update character position ----
    def update_character():
        traj = state["traj"]
        if len(traj) == 0:
            return
        idx = int(state["t"] * (len(traj) - 1))
        idx = max(0, min(idx, len(traj) - 1))
        x, y = traj[idx]
        z = args.ground_z
        char_handle.position = (x, y, z + char_size[2] / 2)
        # Collision check at current frame
        obs = get_obstacles_at_frame(state["frame"])
        char_min = np.array([x - char_size[0] / 2, y - char_size[1] / 2, z])
        char_max = np.array([x + char_size[0] / 2, y + char_size[1] / 2, z + char_size[2]])
        hits = check_collision_3d(char_min, char_max, obs)
        update_collision_label(x, y, z, len(hits))

    # ---- 3D click to add waypoints ----
    @server.scene.on_click()
    def on_scene_click(event):
        """Ray-ground intersection to place a waypoint."""
        ro = np.array(event.ray_origin)
        rd = np.array(event.ray_direction)
        # Intersect with ground plane z = ground_z
        if abs(rd[2]) < 1e-6:
            return  # ray parallel to ground
        t_hit = (args.ground_z - ro[2]) / rd[2]
        if t_hit < 0:
            return  # ground behind camera
        pt = ro + t_hit * rd
        state["waypoints"].append([float(pt[0]), float(pt[1])])
        print(f"  + waypoint at ({pt[0]:.2f}, {pt[1]:.2f})  "
              f"[{len(state['waypoints'])} total]")
        rebuild_trajectory()
        update_character()

    # ---- GUI controls ----
    gui_folder = server.gui.add_folder("Controls")

    with gui_folder:
        gui_t = server.gui.add_slider(
            "traj progress", min=0.0, max=1.0, step=0.001, initial_value=0.0)
        gui_frame = server.gui.add_slider(
            "scene frame", min=0, max=n_frames - 1, step=1, initial_value=0)
        gui_play = server.gui.add_checkbox("play", initial_value=False)
        gui_speed = server.gui.add_slider(
            "speed", min=0.05, max=2.0, step=0.05, initial_value=0.3)
        gui_undo = server.gui.add_button("undo last waypoint")
        gui_clear = server.gui.add_button("clear all waypoints")
        gui_sync = server.gui.add_checkbox(
            "sync frame to traj", initial_value=True,
            hint="auto-set scene frame based on trajectory progress")
        gui_export = server.gui.add_button("export traj.json")

    # ---- Gait parameters folder ----
    gait_folder = server.gui.add_folder("步频参数 (Gait)")

    with gait_folder:
        gui_cycle_stride = server.gui.add_slider(
            "步态周期步幅(m)", min=1.0, max=4.0, step=0.1,
            initial_value=args.cycle_stride,
            hint="一个步态周期(左右各一步)覆盖的距离。2.6m=1.3m/步")
        gui_gait_info = server.gui.add_text(
            "gait info", initial_value="(添加路径点后显示步频参数)", disabled=True)
        gui_video_info = server.gui.add_text(
            "video info",
            initial_value=f"{n_frames}f @ {args.render_fps}fps = "
                          f"{n_frames/args.render_fps:.1f}s",
            disabled=True)

    @gui_t.on_update
    def _on_t(event):
        state["t"] = gui_t.value
        if gui_sync.value and len(state["traj"]) > 0:
            state["frame"] = int(state["t"] * (n_frames - 1))
            gui_frame.value = state["frame"]
            rebuild_obstacles(state["frame"])
        update_character()

    @gui_frame.on_update
    def _on_frame(event):
        state["frame"] = int(gui_frame.value)
        rebuild_obstacles(state["frame"])
        update_character()

    @gui_play.on_update
    def _on_play(event):
        state["playing"] = gui_play.value

    @gui_speed.on_update
    def _on_speed(event):
        state["speed"] = gui_speed.value

    @gui_sync.on_update
    def _on_sync(event):
        pass  # read live in callback

    @gui_cycle_stride.on_update
    def _on_stride(event):
        args.cycle_stride = gui_cycle_stride.value
        rebuild_trajectory()  # recalculates length label

    @gui_undo.on_click
    def _on_undo(event):
        if state["waypoints"]:
            state["waypoints"].pop()
            print(f"  - undo  [{len(state['waypoints'])} waypoints]")
            rebuild_trajectory()
            update_character()

    @gui_clear.on_click
    def _on_clear(event):
        state["waypoints"] = []
        print("  cleared all waypoints")
        rebuild_trajectory()

    @gui_export.on_click
    def _on_export(event):
        if len(state["waypoints"]) < 2:
            print("  need >= 2 waypoints to export")
            return
        traj = state["traj"]
        tl = trajectory_length(traj)
        gp = compute_gait_params(
            trajectory_length=tl, n_video_frames=n_frames,
            fps=args.render_fps, cycle_stride=args.cycle_stride)
        out = {
            "waypoints": state["waypoints"],
            "trajectory": traj.tolist(),
            "total_length": tl,
            "colliding_samples": 0,
            # Gait-matched parameters for render_runner_video.py
            "gait": {
                "cycle_stride": gp.cycle_stride,
                "step_length": gp.step_length,
                "n_steps": round(gp.n_steps, 1),
                "n_cycles": round(gp.n_cycles, 1),
                "anim_speed": round(gp.anim_speed, 4),
                "char_speed": round(gp.char_speed, 2),
                "step_freq": round(gp.step_freq, 2),
                "n_video_frames": gp.n_video_frames,
                "fps": gp.fps,
                "video_duration": round(gp.video_duration, 2),
            },
        }
        out_path = f"outputs/waymo_omnire/scene{cfg.data.scene_idx}/trajectories/traj_live.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"  exported to {out_path}")
        print(gp.detail())
        print(f"  → 渲染命令加: --anim_speed {gp.anim_speed:.3f}")

    # ---- Init ----
    rebuild_trajectory()
    rebuild_obstacles(0)
    update_character()

    print(f"\n{'='*60}")
    print(f"Trajectory previewer (dynamic) running!")
    print(f"  Open: http://localhost:{args.port}")
    print(f"  - CLICK in 3D scene to add waypoints")
    print(f"  - 'undo'/'clear' buttons to edit waypoints")
    print(f"  - 'scene frame' slider moves obstacles in time")
    print(f"  - 'sync frame to traj' links time to trajectory progress")
    print(f"  - 'export traj.json' saves for render_runner_video.py")
    print(f"  - Length label shows if trajectory is too long for video")
    print(f"{'='*60}\n")

    # Animation loop
    try:
        while True:
            now = time.time()
            dt = now - state["last_time"]
            state["last_time"] = now
            if state["playing"] and len(state["traj"]) > 0:
                state["t"] += state["speed"] * dt * 0.1
                if state["t"] > 1.0:
                    state["t"] = 1.0
                    state["playing"] = False
                    gui_play.value = False
                gui_t.value = state["t"]
                if gui_sync.value:
                    state["frame"] = int(state["t"] * (n_frames - 1))
                    gui_frame.value = state["frame"]
                    rebuild_obstacles(state["frame"])
                update_character()
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
