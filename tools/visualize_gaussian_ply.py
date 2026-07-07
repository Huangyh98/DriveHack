"""
Interactive Viser viewer for exported 3D Gaussian PLY files.

Features:
  - Loads GraphDECO/Inria-style Gaussian PLY files exported by
    tools/export_gaussians_ply.py.
  - Time slider for Waymo frame index.
  - Optional PLY sequence switching when gaussians_frameXYZ.ply files exist.
  - Camera follow mode using the original processed Waymo camera trajectory.
  - Free-view mode through the browser's standard Viser controls.

Example:
    /home/avm/miniconda3/envs/drivestudio/bin/python \
        tools/visualize_gaussian_ply.py \
        --ply outputs/waymo_omnire/scene23/ue_gaussians \
        --scene_dir data/waymo/processed/training/023
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import viser
from viser import transforms as tf

from datasets.waymo.waymo_sourceloader import OPENCV2DATASET


logger = logging.getLogger("visualize_gaussian_ply")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SH_C0 = 0.28209479177387814
CAMERA_NAMES = {
    0: "front_camera",
    1: "front_left_camera",
    2: "front_right_camera",
    3: "left_camera",
    4: "right_camera",
}


class SplatData(dict):
    centers: npt.NDArray[np.float32]
    rgbs: npt.NDArray[np.float32]
    opacities: npt.NDArray[np.float32]
    covariances: npt.NDArray[np.float32]


def sigmoid(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussian_ply(
    ply_path: Path,
    *,
    center: bool = False,
    max_gaussians: int = 0,
    seed: int = 0,
) -> SplatData:
    start = time.time()
    vertex = read_binary_little_endian_ply_vertices(ply_path)
    total = len(vertex)

    if max_gaussians > 0 and total > max_gaussians:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total, size=max_gaussians, replace=False))
        logger.info("Downsampling %s from %d to %d gaussians", ply_path.name, total, max_gaussians)
    else:
        indices = slice(None)

    positions = np.stack([vertex["x"][indices], vertex["y"][indices], vertex["z"][indices]], axis=-1).astype(np.float32)
    if center:
        positions -= positions.mean(axis=0, keepdims=True)

    scales = np.exp(
        np.stack(
            [vertex["scale_0"][indices], vertex["scale_1"][indices], vertex["scale_2"][indices]],
            axis=-1,
        )
    ).astype(np.float32)
    rotations = np.stack(
        [vertex["rot_0"][indices], vertex["rot_1"][indices], vertex["rot_2"][indices], vertex["rot_3"][indices]],
        axis=-1,
    ).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True).clip(1e-8)

    colors = (
        0.5
        + SH_C0
        * np.stack(
            [vertex["f_dc_0"][indices], vertex["f_dc_1"][indices], vertex["f_dc_2"][indices]],
            axis=-1,
        )
    ).astype(np.float32)
    colors = np.clip(colors, 0.0, 1.0)
    opacities = sigmoid(vertex["opacity"][indices].astype(np.float32))[:, None].astype(np.float32)

    rot_mats = tf.SO3(rotations).as_matrix().astype(np.float32)
    covariances = np.einsum(
        "nij,njk,nlk->nil",
        rot_mats,
        np.eye(3, dtype=np.float32)[None, :, :] * scales[:, None, :] ** 2,
        rot_mats,
    ).astype(np.float32)

    logger.info("Loaded %s: %d gaussians in %.2fs", ply_path, positions.shape[0], time.time() - start)
    return SplatData(
        centers=positions,
        rgbs=colors,
        opacities=opacities,
        covariances=covariances,
    )


def read_binary_little_endian_ply_vertices(ply_path: Path) -> np.memmap:
    with ply_path.open("rb") as f:
        header_bytes = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header in {ply_path}")
            header_bytes.extend(line)
            if line == b"end_header\n":
                break
        data_offset = f.tell()

    header = header_bytes.decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{ply_path} must be binary_little_endian PLY")

    vertex_count = None
    properties: list[str] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and len(parts) == 3 and parts[:2] == ["property", "float"]:
            properties.append(parts[2])

    if vertex_count is None:
        raise ValueError(f"No vertex element found in {ply_path}")

    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = required - set(properties)
    if missing:
        raise ValueError(f"{ply_path} is missing required 3DGS fields: {sorted(missing)}")

    dtype = np.dtype([(name, "<f4") for name in properties])
    return np.memmap(ply_path, mode="r", dtype=dtype, offset=data_offset, shape=(vertex_count,))


def discover_ply_frames(ply: Path) -> Dict[int, Path]:
    if ply.is_file():
        frame = parse_frame_id(ply) or 0
        return {frame: ply}

    files = sorted(ply.glob("gaussians_frame*.ply"))
    if not files:
        files = sorted(ply.glob("*.ply"))

    frames: Dict[int, Path] = {}
    for idx, path in enumerate(files):
        frames[parse_frame_id(path) if parse_frame_id(path) is not None else idx] = path
    if not frames:
        raise FileNotFoundError(f"No .ply files found under {ply}")
    return frames


def discover_layered_ply_frames(ply: Path) -> tuple[Optional[Path], Dict[int, Path]]:
    if not ply.is_dir():
        return None, discover_ply_frames(ply)

    backgrounds = sorted(ply.glob("Background_frame*.ply"))
    static_ply = backgrounds[0] if backgrounds else None
    dynamic_files = sorted(ply.glob("Dynamic_frame*.ply"))
    if not dynamic_files:
        return static_ply, discover_ply_frames(ply)

    dynamic_frames: Dict[int, Path] = {}
    for idx, path in enumerate(dynamic_files):
        dynamic_frames[parse_frame_id(path) if parse_frame_id(path) is not None else idx] = path
    return static_ply, dynamic_frames


def parse_frame_id(path: Path) -> Optional[int]:
    match = re.search(r"frame(\d+)", path.stem)
    return int(match.group(1)) if match else None


def infer_scene_dir(ply: Path, data_root: Path) -> Optional[Path]:
    for part in [ply.name, *[p.name for p in ply.parents]]:
        match = re.fullmatch(r"scene(\d+)", part)
        if match:
            scene_dir = data_root / f"{int(match.group(1)):03d}"
            if scene_dir.exists():
                return scene_dir
    return None


def load_camera_trajectory(scene_dir: Path, cam_id: int) -> tuple[npt.NDArray[np.float32], Optional[npt.NDArray[np.float32]]]:
    ego_dir = scene_dir / "ego_pose"
    extrinsics_path = scene_dir / "extrinsics" / f"{cam_id}.txt"
    intrinsics_path = scene_dir / "intrinsics" / f"{cam_id}.txt"
    if not ego_dir.exists() or not extrinsics_path.exists():
        raise FileNotFoundError(f"Missing ego_pose or extrinsics for camera {cam_id} under {scene_dir}")

    frame_paths = sorted(ego_dir.glob("*.txt"))
    start_pose = np.loadtxt(frame_paths[0])
    cam_to_ego = np.loadtxt(extrinsics_path) @ OPENCV2DATASET

    c2ws = []
    for path in frame_paths:
        ego_to_world = np.linalg.inv(start_pose) @ np.loadtxt(path)
        c2ws.append(ego_to_world @ cam_to_ego)

    intrinsics = np.loadtxt(intrinsics_path) if intrinsics_path.exists() else None
    return np.stack(c2ws, axis=0).astype(np.float32), intrinsics.astype(np.float32) if intrinsics is not None else None


def set_client_to_camera(
    client: viser.ClientHandle,
    c2w: npt.NDArray[np.float32],
    *,
    look_distance: float = 8.0,
) -> None:
    position = c2w[:3, 3]
    forward = c2w[:3, 2]
    up = -c2w[:3, 1]
    forward = forward / np.linalg.norm(forward).clip(1e-8)
    up = up / np.linalg.norm(up).clip(1e-8)

    with client.atomic():
        client.camera.position = position
        client.camera.up_direction = up
        client.camera.look_at = position + forward * look_distance


def add_or_update_current_frustum(
    server: viser.ViserServer,
    c2w: npt.NDArray[np.float32],
    intrinsics: Optional[npt.NDArray[np.float32]],
    handle,
):
    if intrinsics is not None and intrinsics.shape[0] >= 4:
        fx, fy, cx, cy = intrinsics[:4]
        fov = float(2.0 * np.arctan2(cy, fy))
        aspect = float(cx / cy) if cy > 0 else 1.5
    else:
        fov = np.deg2rad(60.0)
        aspect = 1.5

    wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
    if handle is None:
        return server.scene.add_camera_frustum(
            "/trajectory/current_camera",
            fov=fov,
            aspect=aspect,
            scale=1.0,
            line_width=3.0,
            color=(255, 80, 20),
            wxyz=wxyz,
            position=c2w[:3, 3],
        )
    handle.wxyz = wxyz
    handle.position = c2w[:3, 3]
    return handle


def launch_app_window(url: str, browser: Optional[str], user_data_dir: Path) -> Optional[subprocess.Popen]:
    browser_path = browser or shutil.which("google-chrome") or shutil.which("google-chrome-stable") or shutil.which("chromium") or shutil.which("chromium-browser")
    if browser_path is None:
        logger.warning("No Chrome/Chromium executable found; open %s manually.", url)
        return None

    user_data_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        browser_path,
        f"--app={url}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--disable-extensions",
        "--disable-background-networking",
        "--window-size=1800,1000",
    ]
    logger.info("Launching local app window: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@dataclass
class AssetSpec:
    """Placement of an animated GLB asset in the scene.

    The GLB's embedded animation is auto-played by viser's client
    (see viser/src/viser/client/src/mesh/GlbLoaderUtils.ts:87), so we only need
    to fix its world placement here. If ``path_end`` is given, the asset moves
    from ``position`` to ``path_end`` along a straight line over one playback
    cycle, facing its direction of travel.
    """

    glb_path: Path
    position: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    scale: float
    path_end: Optional[tuple[float, float, float]] = None


def parse_float_csv(values: str, count: int, field: str) -> tuple[float, ...]:
    parts = [v.strip() for v in values.split(",") if v.strip() != ""]
    if len(parts) != count:
        raise argparse.ArgumentTypeError(
            f"--asset {field} expects {count} comma-separated numbers, got {values!r}"
        )
    try:
        return tuple(float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--asset {field} has non-numeric value: {values!r}") from e


def parse_asset_spec(spec: str) -> AssetSpec:
    """Parse an --asset spec.

    Formats (fields are separated by ``;``):
        path/to/asset.glb
        path/to/asset.glb;x,y,z
        path/to/asset.glb;x,y,z;roll,pitch,yaw_deg
        path/to/asset.glb;x,y,z;roll,pitch,yaw_deg;scale
        path/to/asset.glb;x,y,z;roll,pitch,yaw_deg;scale;ex,ey,ez

    ``roll,pitch,yaw`` are degrees around the world X, Y, Z axes. The optional
    last field ``ex,ey,ez`` is a path end point: when set, the asset walks from
    the start position to this end over each playback cycle and its heading
    auto-follows the travel direction (yaw is then ignored).

    ``x|y|z`` may itself be a ``start~end`` pair (e.g. ``0~10,-1,0``) as a
    shortcut for setting both the start coordinate and the path end.
    """
    fields = [f.strip() for f in spec.split(";") if f.strip() != ""]
    if not fields:
        raise argparse.ArgumentTypeError("--asset is empty")

    glb_path = Path(fields[0])
    if not glb_path.exists():
        raise argparse.ArgumentTypeError(f"--asset GLB not found: {glb_path}")
    if glb_path.suffix.lower() != ".glb":
        raise argparse.ArgumentTypeError(
            f"--asset expects a .glb (convert FBX with tools/fbx_to_glb.py): {glb_path}"
        )

    position = (0.0, 0.0, 0.0)
    rpy_deg = (0.0, 0.0, 0.0)
    scale = 1.0
    path_end: Optional[tuple[float, float, float]] = None

    if len(fields) > 1:
        # support per-axis "start~end" shorthand
        raw = [p.strip() for p in fields[1].split(",")]
        if len(raw) != 3:
            raise argparse.ArgumentTypeError(
                f"--asset position expects 3 comma-separated values, got {fields[1]!r}"
            )
        pos_vals: list[float] = []
        end_vals: list[float] = []
        has_pair = False
        for p in raw:
            if "~" in p:
                a, b = p.split("~")
                pos_vals.append(float(a.strip()))
                end_vals.append(float(b.strip()))
                has_pair = True
            else:
                v = float(p)
                pos_vals.append(v)
                end_vals.append(v)
        position = tuple(pos_vals)  # type: ignore[assignment]
        if has_pair:
            path_end = tuple(end_vals)  # type: ignore[assignment]
    if len(fields) > 2:
        vals = parse_float_csv(fields[2], 3, "rotation")
        rpy_deg = tuple(vals)  # type: ignore[assignment]
    if len(fields) > 3:
        scale = float(fields[3])
    if len(fields) > 4:
        path_end = parse_float_csv(fields[4], 3, "path end")  # type: ignore[assignment]

    return AssetSpec(glb_path=glb_path, position=position, rpy_deg=rpy_deg, scale=scale, path_end=path_end)


def rpy_deg_to_wxyz(rpy_deg: tuple[float, float, float]) -> npt.NDArray[np.float32]:
    """Convert roll/pitch/yaw (deg, around world X/Y/Z) to a viser wxyz quaternion."""
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cx, sx = np.cos(roll / 2), np.sin(roll / 2)
    cy, sy = np.cos(pitch / 2), np.sin(pitch / 2)
    cz, sz = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return np.array([w, x, y, z], dtype=np.float32)


def _qmul(a: np.ndarray, b: np.ndarray) -> npt.NDArray[np.float32]:
    """Hamilton product of two wxyz quaternions (a then b = world R = b * a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def asset_base_wxyz() -> npt.NDArray[np.float32]:
    """Quaternion standing the GLB (Y-up glTF) upright in viser's Z-up world.

    The exported runner is a standard Y-up glTF (head at +Y, local forward +Z).
    Rotating +90deg about world X maps local +Y -> world +Z (head up) and local
    +Z -> world -Y (so the character faces -Y after standing).
    """
    return rpy_deg_to_wxyz((90.0, 0.0, 0.0))


def heading_wxyz_to(from_xyz: np.ndarray, to_xyz: np.ndarray) -> npt.NDArray[np.float32]:
    """Z-axis spin (world wxyz) that points the character's world forward (-Y
    after the base roll) at the horizontal travel direction (to - from).

    After `asset_base_wxyz`, forward is world -Y. A yaw of atan2(dx, -dy) about
    world Z turns that -Y onto the travel direction d=(dx,dy).
    """
    d = np.asarray(to_xyz, dtype=np.float64) - np.asarray(from_xyz, dtype=np.float64)
    yaw = np.arctan2(d[0], -d[1])
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def asset_pose_at(asset: "AssetSpec", t: float) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Return (wxyz, position) for an asset at normalized progress t in [0,1].

    The GLB is Y-up and is stood upright via asset_base_wxyz (+90deg X). A
    moving asset (path_end set) is then yawed to face its travel direction. A
    fixed asset just keeps the upright base orientation (ignores rpy_deg, which
    only applied to the pre-upright convention).
    """
    start = np.asarray(asset.position, dtype=np.float64)
    base = asset_base_wxyz()
    if asset.path_end is None:
        return _qmul(heading_wxyz_to(start, start), base), start.astype(np.float32)
    end = np.asarray(asset.path_end, dtype=np.float64)
    wxyz = _qmul(heading_wxyz_to(start, end), base)
    pos = start + (end - start) * t
    return wxyz, pos.astype(np.float32)


def load_animated_assets(
    server: viser.ViserServer,
    assets: List[AssetSpec],
) -> tuple[List[viser.GlbHandle], List[viser.GuiInputHandle[bool]], List]:
    """Add each GLB to the scene at its world placement.

    Each asset is parented under a draggable transform-control gizmo
    (/asset_node/asset_XX), so the user can grab and move it in 3D. The GLB's
    run-in-place animation keeps playing on its own; moving the gizmo just
    relocates the whole character.

    Returns (glb handles, visibility toggles, gizmo handles).
    """
    handles: List[viser.GlbHandle] = []
    visibility_toggles: List[viser.GuiInputHandle[bool]] = []
    gizmos: List = []
    for i, asset in enumerate(assets):
        glb_bytes = Path(asset.glb_path).read_bytes()
        wxyz, pos = asset_pose_at(asset, 0.0)
        # The gizmo is the parent and carries the FULL orientation
        # (heading * base upright roll). The GLB child must use the identity
        # orientation so the base roll is not applied twice.
        gizmo = server.scene.add_transform_controls(
            f"/asset_node/asset_{i:02d}",
            scale=1.5,
            wxyz=wxyz,
            position=tuple(pos.tolist()),
            depth_test=False,
        )
        handle = server.scene.add_glb(
            f"/asset_node/asset_{i:02d}/glb",
            glb_data=glb_bytes,
            scale=asset.scale,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            visible=True,
        )
        handles.append(handle)
        gizmos.append(gizmo)
        visibility_toggles.append(
            server.gui.add_checkbox(
                f"{asset.glb_path.stem} visible", initial_value=True
            )
        )
        if asset.path_end is not None:
            logger.info(
                "Loaded MOVING asset %d: %s from %s to %s scale=%s (drag gizmo + enable manual)",
                i, asset.glb_path, asset.position, asset.path_end, asset.scale,
            )
        else:
            logger.info(
                "Loaded asset %d: %s at pos=%s scale=%s",
                i, asset.glb_path, asset.position, asset.scale,
            )
    return handles, visibility_toggles, gizmos


def main() -> None:
    parser = argparse.ArgumentParser("Visualize exported Gaussian PLY files with trajectory controls")
    parser.add_argument(
        "--ply",
        type=Path,
        default=Path("outputs/waymo_omnire/scene23/ue_gaussians"),
        help="A .ply file or a directory containing gaussians_frame*.ply files.",
    )
    parser.add_argument("--scene_dir", type=Path, default=None, help="Processed Waymo scene directory, e.g. data/waymo/processed/training/023")
    parser.add_argument("--data_root", type=Path, default=Path("data/waymo/processed/training"))
    parser.add_argument("--camera", type=int, default=0, choices=sorted(CAMERA_NAMES))
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max_gaussians", type=int, default=0, help="Optional random downsample limit; 0 keeps all gaussians.")
    parser.add_argument("--max_background_gaussians", type=int, default=None, help="Optional limit for static background only. Defaults to --max_gaussians.")
    parser.add_argument("--center", action="store_true", help="Center the PLY around its mean. Leave off for Waymo trajectory alignment.")
    parser.add_argument("--play_fps", type=float, default=10.0)
    parser.add_argument(
        "--preload_frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preload all dynamic frames and switch visibility during playback. This avoids repeated WebGL splat destroy/recreate operations.",
    )
    parser.add_argument("--window", action=argparse.BooleanOptionalAction, default=True, help="Open a local Chrome app window.")
    parser.add_argument("--browser", default=None, help="Chrome/Chromium executable for --window.")
    parser.add_argument(
        "--asset",
        type=parse_asset_spec,
        action="append",
        default=[],
        metavar="GLB[;x,y,z[;roll,pitch,yaw_deg[;scale[;ex,ey,ez]]]",
        help=(
            "Animated GLB asset to drop into the scene. May be given multiple times. "
            "Convert FBX first via tools/fbx_to_glb.py. A trailing end point "
            "(ex,ey,ez) or per-axis start~end pairs make the asset walk that path. "
            "Example: --asset outputs/assets/runner.glb;1,0,0;0,0,0;1.0;10,0,0"
        ),
    )
    parser.add_argument(
        "--asset_cycle_sec",
        type=float,
        default=8.0,
        help="Seconds for a moving asset to traverse its path (loops). Default 8.",
    )
    parser.add_argument(
        "--ground_z",
        type=float,
        default=0.0,
        help="World Z of the ground plane. Asset Z is raised so feet rest on it.",
    )
    args = parser.parse_args()

    # Anchor asset feet on the ground: the character's feet sit at local Z=0
    # after the upright base roll, so we add ground_z to every asset Z.
    if args.asset and args.ground_z != 0.0:
        for spec in args.asset:
            spec.position = (spec.position[0], spec.position[1], spec.position[2] + args.ground_z)
            if spec.path_end is not None:
                spec.path_end = (spec.path_end[0], spec.path_end[1], spec.path_end[2] + args.ground_z)

    static_ply, ply_frames = discover_layered_ply_frames(args.ply)
    ply_frame_ids = sorted(ply_frames)
    scene_dir = args.scene_dir or infer_scene_dir(args.ply, args.data_root)

    c2ws = None
    intrinsics = None
    if scene_dir is not None:
        c2ws, intrinsics = load_camera_trajectory(scene_dir, args.camera)
        logger.info("Loaded camera trajectory: %s, camera=%d (%s), frames=%d", scene_dir, args.camera, CAMERA_NAMES[args.camera], len(c2ws))
    else:
        logger.warning("No scene_dir found. Trajectory controls will still exist, but camera follow is disabled.")

    max_frame = len(ply_frame_ids) - 1
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True
    if hasattr(server, "initial_camera"):
        server.initial_camera.position = (0.0, -20.0, 8.0)
        server.initial_camera.look_at = (10.0, 0.0, 3.0)

    # Animated GLB assets (e.g. a walking character from FBX). Their embedded
    # animation clips are auto-played by viser's client via AnimationMixer, so
    # they loop on their own, independent of the Waymo timeline.
    asset_handles: List[viser.GlbHandle] = []
    asset_visibility_toggles: List[viser.GuiInputHandle[bool]] = []
    asset_gizmos: List = []
    manual_ctrl: Optional[viser.GuiInputHandle[bool]] = None
    show_gizmos: Optional[viser.GuiInputHandle[bool]] = None
    if args.asset:
        with server.gui.add_folder("Animated assets"):
            asset_handles, asset_visibility_toggles, asset_gizmos = load_animated_assets(server, args.asset)
            manual_ctrl = server.gui.add_checkbox("Manual control (drag gizmo)", initial_value=False)
            show_gizmos = server.gui.add_checkbox("Show gizmos", initial_value=True)
            with server.gui.add_folder("Height nudge"):
                up_btn = server.gui.add_button("Up (+0.2)")
                down_btn = server.gui.add_button("Down (-0.2)")
                nudge_step = server.gui.add_slider("Step", min=0.05, max=1.0, step=0.05, initial_value=0.2)
        for handle, toggle in zip(asset_handles, asset_visibility_toggles):
            def _make_cb(h: viser.GlbHandle):
                def _(_cb):  # noqa: ANN001
                    h.visible = toggle.value
                return _
            toggle.on_update(_make_cb(handle))

        @show_gizmos.on_update
        def _(_) -> None:
            for g in asset_gizmos:
                g.visible = show_gizmos.value

        def _nudge(delta: float) -> None:
            step = float(nudge_step.value)
            for gizmo, spec in zip(asset_gizmos, args.asset):
                p = list(gizmo.position)
                p[2] += delta * step
                gizmo.position = tuple(p)
                # remember new ground height so auto mode keeps it
                spec.position = tuple(p)
                if spec.path_end is not None:
                    pe = list(spec.path_end)
                    pe[2] += delta * step
                    spec.path_end = tuple(pe)

        @up_btn.on_click
        def _(_) -> None:
            _nudge(+1.0)

        @down_btn.on_click
        def _(_) -> None:
            _nudge(-1.0)

    state_lock = threading.Lock()
    loaded_frame: Optional[int] = None
    static_splat_handle = None
    dynamic_splat_handles = {}
    frustum_handle = None

    if static_ply is not None:
        bg_limit = args.max_background_gaussians if args.max_background_gaussians is not None else args.max_gaussians
        static_splat = load_gaussian_ply(
            static_ply,
            center=args.center,
            max_gaussians=bg_limit,
            seed=0,
        )
        static_splat_handle = server.scene.add_gaussian_splats(
            "/background_splats",
            centers=static_splat["centers"],
            rgbs=static_splat["rgbs"],
            opacities=static_splat["opacities"],
            covariances=static_splat["covariances"],
        )

    def source_frame(playback_index: int) -> int:
        playback_index = int(np.clip(playback_index, 0, len(ply_frame_ids) - 1))
        return ply_frame_ids[playback_index]

    def add_dynamic_splat(ply_frame: int, *, visible: bool):
        splat_data = load_gaussian_ply(
            ply_frames[ply_frame],
            center=args.center,
            max_gaussians=args.max_gaussians,
            seed=ply_frame,
        )
        handle = server.scene.add_gaussian_splats(
            f"/dynamic_splats/frame_{ply_frame:06d}" if static_splat_handle is not None else f"/full_splats/frame_{ply_frame:06d}",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
            visible=visible,
        )
        return handle

    if args.preload_frames:
        logger.info("Preloading %d dynamic playback frames.", len(ply_frame_ids))
        for preload_frame in ply_frame_ids:
            dynamic_splat_handles[preload_frame] = add_dynamic_splat(preload_frame, visible=False)

    def load_scene_for_frame(frame: int) -> None:
        nonlocal loaded_frame
        ply_frame = source_frame(frame)
        if loaded_frame == ply_frame:
            return
        with server.atomic():
            if loaded_frame in dynamic_splat_handles:
                dynamic_splat_handles[loaded_frame].visible = False
            if ply_frame not in dynamic_splat_handles:
                dynamic_splat_handles[ply_frame] = add_dynamic_splat(ply_frame, visible=False)
            dynamic_splat_handles[ply_frame].visible = True
        loaded_frame = ply_frame

    if c2ws is not None and len(c2ws) > 1:
        server.scene.add_spline_catmull_rom(
            "/trajectory/path",
            positions=c2ws[:, :3, 3],
            color=(40, 180, 255),
            line_width=3.0,
        )

    with server.gui.add_folder("Timeline"):
        frame_slider = server.gui.add_slider("Playback frame", min=0, max=max_frame, step=1, initial_value=0)
        source_frame_number = server.gui.add_number("Source Waymo frame", initial_value=source_frame(0), disabled=True)
        if hasattr(server.gui, "add_progress_bar"):
            progress = server.gui.add_progress_bar(0.0, animated=False)
        else:
            progress = server.gui.add_number("Progress %", initial_value=0.0, disabled=True)
        play = server.gui.add_checkbox("Play", initial_value=False)
        follow_camera = server.gui.add_checkbox("Follow original camera", initial_value=False)
        active_camera = server.gui.add_dropdown(
            "Camera",
            options=[f"{idx}: {name}" for idx, name in CAMERA_NAMES.items()],
            initial_value=f"{args.camera}: {CAMERA_NAMES[args.camera]}",
        )

    with server.gui.add_folder("View"):
        show_trajectory = server.gui.add_checkbox("Show trajectory", initial_value=True)
        jump_button = server.gui.add_button("Jump to current frame")

    def current_frame() -> int:
        return int(frame_slider.value)

    def apply_frame(frame: int, *, move_clients: bool) -> None:
        nonlocal c2ws, intrinsics, frustum_handle
        with state_lock:
            load_scene_for_frame(frame)
            progress.value = float(100.0 * frame / max(1, max_frame))
            actual_frame = source_frame(frame)
            source_frame_number.value = actual_frame
            if c2ws is not None and 0 <= actual_frame < len(c2ws):
                frustum_handle = add_or_update_current_frustum(server, c2ws[actual_frame], intrinsics, frustum_handle)
                if move_clients:
                    for client in server.get_clients().values():
                        set_client_to_camera(client, c2ws[actual_frame])

    def update_moving_assets(elapsed: float) -> None:
        """Advance each moving asset along its path.

        ``elapsed`` is wall-clock seconds; one path traversal takes
        ``args.asset_cycle_sec``. The GLB's run-in-place animation keeps playing
        on its own in the browser, independent of this. When "Manual control"
        is on, we stop auto-driving and let the user drag the gizmo instead;
        the dragged pose is remembered so turning manual back off resumes from
        there instead of snapping back.
        """
        if not asset_handles or not args.asset:
            return
        cycle = max(args.asset_cycle_sec, 1e-3)
        for gizmo, spec in zip(asset_gizmos, args.asset):
            if manual_ctrl is not None and manual_ctrl.value:
                # Remember the user's drag as the new starting pose.
                spec.position = tuple(np.asarray(gizmo.position, dtype=np.float64).tolist())
                continue
            if spec.path_end is None:
                continue
            t = (elapsed % cycle) / cycle
            wxyz, pos = asset_pose_at(spec, t)
            gizmo.wxyz = tuple(wxyz.tolist())
            gizmo.position = tuple(pos.tolist())

    def reload_camera() -> None:
        nonlocal c2ws, intrinsics, frustum_handle
        if scene_dir is None:
            return
        cam_id = int(str(active_camera.value).split(":", 1)[0])
        c2ws, intrinsics = load_camera_trajectory(scene_dir, cam_id)
        frustum_handle = None
        apply_frame(current_frame(), move_clients=follow_camera.value)

    @frame_slider.on_update
    def _(_) -> None:
        apply_frame(current_frame(), move_clients=False)

    @active_camera.on_update
    def _(_) -> None:
        reload_camera()

    @jump_button.on_click
    def _(_) -> None:
        apply_frame(current_frame(), move_clients=True)

    @show_trajectory.on_update
    def _(_) -> None:
        try:
            server.scene.set_global_visibility("/trajectory", show_trajectory.value)
        except AttributeError:
            # Older viser versions do not expose group visibility; current frustum
            # and path remain visible in that case.
            pass

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.far = 1000.0
        actual_frame = source_frame(current_frame())
        if c2ws is not None and actual_frame < len(c2ws):
            set_client_to_camera(client, c2ws[actual_frame])

    apply_frame(0, move_clients=False)
    url = f"http://localhost:{args.port}"
    logger.info("Viewer running at %s", url)
    logger.info("Use 'Follow original camera' for trajectory playback; turn it off for free-view navigation.")
    if args.window:
        launch_app_window(url, args.browser, PROJECT_ROOT / f".viser_chrome_profile_{args.port}_{os.getpid()}")

    last_tick = time.time()
    asset_start = time.time()
    while True:
        # Moving assets advance on wall-clock time, independent of the Waymo
        # playback timeline, so they keep walking whether Play is on or off.
        update_moving_assets(time.time() - asset_start)
        if play.value:
            now = time.time()
            if now - last_tick >= 1.0 / max(args.play_fps, 1e-3):
                next_frame = current_frame() + 1
                if next_frame > max_frame:
                    next_frame = 0
                frame_slider.value = next_frame
                apply_frame(next_frame, move_clients=follow_camera.value)
                last_tick = now
        else:
            last_tick = time.time()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
