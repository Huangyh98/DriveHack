"""Offline adversarial-sample compositor: insert a textured mesh character into
a 3DGS-reconstructed Waymo scene with correct depth occlusion, multi-view
consistency, swappable clothing textures, and batch generation.

Pipeline per output frame:
  1. Render the 3DGS scene (rgb + depth) for each requested Waymo camera.
  2. Place the baked runner mesh at a world position; the run-in-place
     animation frame is advanced over time.
  3. Rasterize the character with nvdiffrast using the SAME camera K/c2w.
  4. Depth-occlude: only write character pixels where char_z < scene_depth.
  5. (optional) apply a coarse directional light estimated from the background.

Modes
-----
  --mode video      : single front-camera video (legacy, for quick checks)
  --mode multiview  : per-frame PNGs across cameras, batched over positions &
                      clothing textures (for adversarial training data)

Example (multiview batch):
    PYTHONPATH=$(pwd) python tools/render_runner_video.py \
        --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
        --seq outputs/assets/runner_seq.npz \
        --mode multiview --cameras 0,1,2 \
        --positions "5,-1,0;8,1,0" \
        --clothes_textures "" \
        --frames 0,40,80,120 \
        --out_dir outputs/adversarial/scene23/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

logger = logging.getLogger("render_runner_video")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CAMERA_NAMES = {0: "front", 1: "front_left", 2: "front_right", 3: "left", 4: "right"}
MESH_NAMES = ["man", "clothes_1", "pants_1"]


# --------------------------------------------------------------------------- #
# Character placement
# --------------------------------------------------------------------------- #
def build_world_transform(position, yaw_deg: float, scale: float, feet_offset: float = 0.0) -> np.ndarray:
    """Rigid transform placing the baked character into the Waymo scene.

    The baked mesh is already Z-up (head at +Z, feet at Z~0) and faces -Y in its
    local frame, matching the Waymo world (Z-up, X forward). So NO base roll is
    needed — only a yaw about world Z to point the character's -Y forward onto
    the heading direction. yaw_deg=0 means facing +X (scene forward).

    feet_offset lifts the character so its lowest point (feet during run cycle,
    which can dip slightly below Z=0) sits exactly on the ground, preventing
    the feet from clipping through the road.
    """
    # local forward = -Y. To face a heading angle measured from +X axis:
    # rotate about Z so that -Y maps onto heading. Rz(theta) @ (0,-1,0) =
    # (sin theta, -cos theta). For heading +X (theta=90deg): (1, 0). Good.
    theta = np.deg2rad(yaw_deg + 90.0)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64) * scale
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Rz
    pos = np.asarray(position, dtype=np.float64).copy()
    pos[2] += feet_offset
    M[:3, 3] = pos
    return M


def ground_z_at(scene_depth: np.ndarray, c2w: np.ndarray, K: np.ndarray, H: int, W: int,
                world_xy, near_limit: float = 1.0) -> float:
    """Estimate ground Z at the character's world (x,y).

    Method: scan a vertical line at the character's (x,y) from Z=0 upward.
    For each test height, project to image and compare expected depth vs scene
    depth. The FIRST height where scene depth ≈ expected depth is the ground
    surface (road). We stop at the first match to get the LOWEST surface, not
    the median (which would be biased upward by cars/walls).

    We also sample a small patch (±2 pixels) and use the minimum depth (nearest
    surface) to be robust against one-pixel noise.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w2c = np.linalg.inv(c2w)
    wx, wy = world_xy

    # fine scan from 0 to 1.5m (road surface is always near 0)
    prev_depth = None
    for test_z in np.arange(0.0, 1.5, 0.05):
        wp = np.array([wx, wy, test_z, 1.0])
        cp = w2c @ wp
        if cp[2] < near_limit:
            continue
        px = int(fx * cp[0] / cp[2] + cx)
        py = int(fy * cp[1] / cp[2] + cy)
        if not (0 <= px < W and 0 <= py < H):
            continue

        # sample a 5x5 patch, take the MINIMUM depth (nearest surface)
        d_min = 999.0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                yy, xx = py + dy, px + dx
                if 0 <= yy < H and 0 <= xx < W:
                    dd = scene_depth[yy, xx]
                    if near_limit < dd < 200 and dd < d_min:
                        d_min = dd
        if d_min > 900:
            continue

        expected_depth = cp[2]
        # When scene depth matches expected depth, we found the surface
        if abs(d_min - expected_depth) < 0.5:
            # back-project to get precise world Z
            x_c = (px - cx) / fx * d_min
            y_c = (py - cy) / fy * d_min
            world_pt = c2w @ np.array([x_c, y_c, d_min, 1.0])
            return float(world_pt[2])

    # fallback: scan was inconclusive, try the old median approach as last resort
    return 0.0


def apply_transform(verts: np.ndarray, M: np.ndarray) -> np.ndarray:
    hom = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=verts.dtype)], axis=1)
    return (hom @ M.T)[:, :3]


def precompute_gz_table(get_scene, frame_ids, char_traj_xy, cam_id,
                         fallback_z: float = 0.0, sample_step: int = 5) -> np.ndarray:
    """Pre-compute ground Z for every output frame along the trajectory.

    Samples the 3DGS depth at every `sample_step`-th frame, finds the road Z
    at the character's XY position, then linearly interpolates to all frames.
    Each sampled Z is clamped to be >= the previous sample (monotonic floor)
    to prevent sudden dips; and all values are median-smoothed.

    Returns: (len(frame_ids),) array of ground Z values, one per frame.
    """
    n = len(frame_ids)
    raw_z = np.full(n, fallback_z)

    # sample ground Z every N frames
    sample_indices = list(range(0, n, sample_step))
    if sample_indices[-1] != n - 1:
        sample_indices.append(n - 1)

    prev_valid_z = fallback_z
    for si in sample_indices:
        frame_idx = frame_ids[si]
        s_rgb, s_depth, s_bg_depth, s_c2w, s_K, s_H, s_W, _ = get_scene(frame_idx, cam_id)
        tx = char_traj_xy[min(si, len(char_traj_xy) - 1)]
        gz = ground_z_at(s_bg_depth, s_c2w, s_K, s_H, s_W,
                         (tx[0], tx[1]))
        # sanity: ground shouldn't jump more than 0.5m from previous sample
        if abs(gz) > 5.0:
            gz = prev_valid_z  # bad read, use previous
        gz = max(gz, prev_valid_z - 0.3)  # monotonic: don't drop more than 0.3m
        raw_z[si] = gz
        prev_valid_z = gz
        logger.info("  gz_table[%d/%d] frame %d at (%.1f,%.1f): Z=%.3f",
                    si, n, frame_idx, tx[0], tx[1], gz)

    # linear interpolate between samples
    for i in range(n):
        if raw_z[i] == fallback_z and i > 0:
            # find surrounding samples
            left = i
            while left > 0 and raw_z[left] == fallback_z:
                left -= 1
            right = i
            while right < n - 1 and raw_z[right] == fallback_z:
                right += 1
            if raw_z[left] != fallback_z or raw_z[right] != fallback_z:
                frac = (i - left) / max(1, right - left)
                raw_z[i] = raw_z[left] * (1 - frac) + raw_z[right] * frac

    # smooth (moving average, window 5)
    kernel = np.ones(5) / 5
    gz_smooth = np.convolve(raw_z, kernel, mode='same')
    # pad edges
    gz_smooth[:2] = raw_z[:2]
    gz_smooth[-2:] = raw_z[-2:]

    logger.info("Ground Z table: min=%.3f max=%.3f mean=%.3f (samples=%d)",
                gz_smooth.min(), gz_smooth.max(), gz_smooth.mean(), len(sample_indices))
    return gz_smooth


def parse_waypoints(spec: str):
    """Parse 'x1,y1;x2,y2;...' into a list of (x,y).

    Default for scene552: person starts at front-LEFT of the car (high X, high Y),
    runs toward the car (decreasing X), then passes on the car's RIGHT side
    (Y goes negative but stays near the road, not into buildings). The trajectory
    stays on the road corridor (Y ~ -1..8) to avoid colliding with background
    buildings. Speed is controlled separately via --anim_speed / --max_output_frames.
    """
    if not spec or not spec.strip():
        # front-left -> toward car -> past car's right side
        return [(38.0, 9.0), (28.0, 4.0), (18.0, 0.0), (8.0, -1.0)]
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(",")
        out.append((float(a), float(b)))
    return out if out else [(10.0, 4.0), (14.0, -4.0), (18.0, -1.0)]


def sample_polyline(waypoints, t: float):
    """Sample position at normalized progress t in [0,1] along the polyline."""
    pts = np.asarray(waypoints, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    total = seglen.sum()
    if total < 1e-6:
        return float(pts[0, 0]), float(pts[0, 1])
    target = t * total
    acc = 0.0
    for i, L in enumerate(seglen):
        if acc + L >= target or i == len(seglen) - 1:
            u = (target - acc) / max(L, 1e-6)
            p = pts[i] + u * (pts[i + 1] - pts[i])
            return float(p[0]), float(p[1])
        acc += L
    return float(pts[-1, 0]), float(pts[-1, 1])


def polyline_yaw_at(waypoints, t: float) -> float:
    """Heading (deg from +X) of the polyline tangent at progress t."""
    pts = np.asarray(waypoints, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    total = seglen.sum()
    if total < 1e-6:
        return 0.0
    target = t * total
    acc = 0.0
    for i, L in enumerate(seglen):
        if acc + L >= target or i == len(seglen) - 1:
            d = seg[i]
            return float(np.degrees(np.arctan2(d[1], d[0])))
        acc += L
    d = seg[-1]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _load_traj_json(path: str):
    """Load a trajectory JSON from plan_trajectory_bev.py. Cached on the function."""
    import json
    if not hasattr(_load_traj_json, "_cache"):
        _load_traj_json._cache = {}
    if path not in _load_traj_json._cache:
        with open(path) as f:
            _load_traj_json._cache[path] = json.load(f)
    return _load_traj_json._cache[path]


def sample_traj_json(path: str, t: float):
    """Sample world (x,y) at normalized progress t from a JSON trajectory."""
    data = _load_traj_json(path)
    traj = np.asarray(data["trajectory"], dtype=np.float64)
    n = len(traj)
    idx = int(np.clip(t * (n - 1), 0, n - 1))
    return float(traj[idx, 0]), float(traj[idx, 1])


def traj_json_yaw(path: str, t: float) -> float:
    """Heading (deg from +X) along the JSON trajectory at progress t."""
    data = _load_traj_json(path)
    traj = np.asarray(data["trajectory"], dtype=np.float64)
    n = len(traj)
    idx = int(np.clip(t * (n - 1), 0, n - 1))
    j = min(idx + 1, n - 1)
    d = traj[j] - traj[idx]
    if np.linalg.norm(d) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(d[1], d[0])))


# --------------------------------------------------------------------------- #
# Rasterization (returns RGBA + per-pixel char depth)
# --------------------------------------------------------------------------- #
def rasterize_mesh(rast, verts_world, faces, uvs, tex, c2w, K, H, W, device):
    """Rasterize a triangle mesh. Returns (rgba HxWx4 numpy, char_depth HxW numpy).

    char_depth is the camera-space z (forward distance, meters) of the closest
    character triangle at each pixel, or +inf where uncovered. This is directly
    comparable to the gsplat scene depth for occlusion.

    To respect UV seams we build an UNWOUND mesh: each triangle's 3 corners
    become 3 unique vertices (positions + uv), avoiding the per-vertex uv
    averaging that corrupts seams (missing feet/back-of-head).
    """
    import nvdiffrast.torch as dr
    verts = torch.as_tensor(verts_world, dtype=torch.float32, device=device) if not torch.is_tensor(verts_world) else verts_world
    faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    uvs_np = uvs.detach().cpu().numpy() if torch.is_tensor(uvs) else np.asarray(uvs)

    # Build unwound geometry: for each triangle, 3 unique verts.
    F = faces_np.shape[0]
    # gather corner vertex positions: (F,3,3)
    corner_pos = verts_world[faces_np] if not torch.is_tensor(verts_world) else verts[faces_np].cpu().numpy()
    unw_pos = corner_pos.reshape(F * 3, 3)            # (F*3, 3)
    unw_uv = uvs_np.reshape(F * 3, 2)                 # (F*3, 2)
    unw_tri = (np.arange(F * 3).reshape(F, 3)).astype(np.int32)  # (F,3) -> [0,1,2],[3,4,5],...

    w2c = np.linalg.inv(c2w)
    verts_cam = unw_pos @ w2c[:3, :3].T + w2c[:3, 3]
    x, y, z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]
    z_safe = np.maximum(z, 1e-3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    px = fx * x / z_safe + cx
    py = fy * y / z_safe + cy
    ndcx = (2.0 * px - W) / W
    ndcy = (2.0 * py - H) / H
    wclip = z_safe
    near, far = 0.1, 1000.0
    z_ndc = (z_safe - near) / (far - near)
    zclip = z_ndc * wclip
    clip = np.stack([ndcx * wclip, ndcy * wclip, zclip, wclip], axis=1).astype(np.float32)
    clip_t = torch.from_numpy(clip).to(device)
    tri = torch.from_numpy(unw_tri).int().to(device).contiguous()

    rast_out, _ = dr.rasterize(rast, clip_t.unsqueeze(0).contiguous(), tri, resolution=(H, W))

    # uv attribute per unwound vertex (no averaging -> seams preserved)
    vuv = torch.from_numpy(unw_uv.astype(np.float32)).to(device)
    interp, _ = dr.interpolate(vuv.contiguous(), rast_out, tri)
    mask = (rast_out[0, ..., 3:] > 0).float()
    th, tw = tex.shape[:2]
    sx = torch.clamp(interp[0, ..., 0] * tw, 0, tw - 1).long()
    sy = torch.clamp(interp[0, ..., 1] * th, 0, th - 1).long()
    sampled = tex[sy, sx]
    rgb = sampled[..., :3]
    a = sampled[..., 3:4] * mask

    # per-pixel character depth from the unwound vertex z
    z_t = torch.from_numpy(z.astype(np.float32)).to(device).view(1, -1, 1)
    z_interp, _ = dr.interpolate(z_t.contiguous(), rast_out, tri)
    char_depth = z_interp[0, ..., 0]
    char_depth = torch.where(mask[..., 0] > 0, char_depth, torch.full_like(char_depth, float("inf")))

    rgba = torch.cat([rgb, a], dim=-1).cpu().numpy()
    char_depth_np = char_depth.cpu().numpy()
    return rgba, char_depth_np


# --------------------------------------------------------------------------- #
# 3D bounding box drawing
# --------------------------------------------------------------------------- #
def draw_3d_bbox(img_pil, center_world, size, c2w, K, color=(0, 255, 0), lw=2):
    """Draw a 3D bounding box (8 corners, 12 edges) projected onto the image.

    center_world: (x, y, z) of box CENTER in world coords.
    size: (dx, dy, dz) full dimensions.
    """
    from PIL import ImageDraw
    cx, cy, cz = center_world
    dx, dy, dz = size
    x0, x1 = cx - dx / 2, cx + dx / 2
    y0, y1 = cy - dy / 2, cy + dy / 2
    z0, z1 = cz, cz + dz  # bottom at feet, top at head
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    _draw_box_edges(img_pil, corners, c2w, K, color, lw)


def draw_tight_3d_bbox(img_pil, verts_world, c2w, K, color=(0, 255, 0), lw=2, margin=0.03):
    """Draw a 3D bbox that tightly fits the character's actual vertex extent.

    The box is a world-axis-aligned AABB of the transformed vertices (with a
    small margin), so it follows the character's pose — e.g. narrower when
    arms are down, wider when arms swing out.
    """
    mn = verts_world.min(axis=0) - margin
    mx = verts_world.max(axis=0) + margin
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    _draw_box_edges(img_pil, corners, c2w, K, color, lw)


def _draw_box_edges(img_pil, corners, c2w, K, color, lw):
    """Project 8 corners and draw the 12 edges of the box."""
    from PIL import ImageDraw
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    w2c = np.linalg.inv(c2w)
    pts2d = []
    for c in corners:
        pc = w2c @ np.array([c[0], c[1], c[2], 1.0])
        if pc[2] < 0.1:
            pts2d.append(None)
            continue
        px = K[0, 0] * pc[0] / pc[2] + K[0, 2]
        py = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        pts2d.append((int(px), int(py)))
    draw = ImageDraw.Draw(img_pil)
    for a, b in edges:
        if pts2d[a] is not None and pts2d[b] is not None:
            draw.line([pts2d[a], pts2d[b]], fill=color, width=lw)


# --------------------------------------------------------------------------- #
# BEV mini-map rendering (per-frame)
# --------------------------------------------------------------------------- #
class BevMiniMap:
    """Renders a small BEV map for each video frame, showing the ego trajectory,
    obstacles, and the character's current position as a highlighted marker."""

    def __init__(self, scene_dir, char_traj_xy, map_size=480):
        """Precompute static BEV elements once.

        char_traj_xy: (N,2) array of the character's full trajectory XY (for drawing).
        """
        from pathlib import Path
        import json
        scene_dir = Path(scene_dir)
        # ego trajectory
        ego_paths = sorted((scene_dir / "ego_pose").glob("*.txt"))
        start = np.loadtxt(ego_paths[0])
        ego = []
        for p in ego_paths:
            rel = np.linalg.inv(start) @ np.loadtxt(p)
            ego.append(rel[:3, 3])
        self.ego = np.array(ego)
        self.char_traj = np.asarray(char_traj_xy)
        # obstacles
        ii_path = scene_dir / "instances" / "instances_info.json"
        self.obstacles = []
        if ii_path.exists():
            ii = json.load(open(ii_path))
            inv_start = np.linalg.inv(start)
            for info in ii.values():
                fa = info.get("frame_annotations", {})
                oms = fa.get("obj_to_world", [])
                sizes = fa.get("box_size", [])
                if not oms or not sizes:
                    continue
                mid = len(oms) // 2
                om = np.array(oms[mid]).reshape(4, 4)
                sz = np.array(sizes[mid])
                rel = inv_start @ om
                self.obstacles.append({
                    "center": (float(rel[0, 3]), float(rel[1, 3])),
                    "size": (float(sz[0]), float(sz[1])),
                    "cls": info.get("class_name", "?"),
                })
        # bounds
        xs = list(self.ego[:, 0]) + list(self.char_traj[:, 0]) + \
             [o["center"][0] for o in self.obstacles]
        ys = list(self.ego[:, 1]) + list(self.char_traj[:, 1]) + \
             [o["center"][1] for o in self.obstacles]
        pad = 5.0
        self.xmin, self.xmax = min(xs) - pad, max(xs) + pad
        self.ymin, self.ymax = min(ys) - pad, max(ys) + pad
        self.map_size = map_size
        # pre-render static layer (background, obstacles, ego path, char trajectory)
        self._static = self._render_static()

    def _world_to_map(self, x, y):
        sx = self.map_size / (self.xmax - self.xmin)
        sy = self.map_size / (self.ymax - self.ymin)
        s = min(sx, sy)
        mx = int((x - self.xmin) * s)
        my = int((self.ymax - y) * s)  # flip Y so +Y is up
        return mx, my, s

    def _render_static(self):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (self.map_size, self.map_size), (26, 26, 26))
        draw = ImageDraw.Draw(img)
        # obstacles
        cls_color = {"Vehicle": (231, 76, 60), "Pedestrian": (241, 196, 15),
                     "Cyclist": (230, 126, 34)}
        for ob in self.obstacles:
            mx, my, s = self._world_to_map(*ob["center"])
            dx, dy = ob["size"]
            w, h = int(dx * s), int(dy * s)
            c = cls_color.get(ob["cls"], (149, 165, 166))
            draw.rectangle([mx - w // 2, my - h // 2, mx + w // 2, my + h // 2],
                           outline=c, fill=(c[0] // 3, c[1] // 3, c[2] // 3))
        # ego trajectory
        pts = [self._world_to_map(x, y)[:2] for x, y, _ in self.ego]
        draw.line(pts, fill=(0, 200, 200), width=2)
        # start/end markers as small circles
        for pt, c in [(pts[0], (0, 255, 0)), (pts[-1], (255, 0, 0))]:
            draw.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], fill=c)
        # character full trajectory (dashed green)
        if len(self.char_traj) > 1:
            cpts = [self._world_to_map(x, y)[:2] for x, y in self.char_traj]
            draw.line(cpts, fill=(46, 204, 113), width=2)
        # title
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            draw.text((8, 8), "BEV", fill=(255, 255, 0), font=font)
        except Exception:
            draw.text((8, 8), "BEV", fill=(255, 255, 0))
        return img

    def render_frame(self, char_x, char_y, ego_idx=0):
        """Return a PIL image of the BEV with the character's current position marked."""
        from PIL import ImageDraw
        img = self._static.copy()
        draw = ImageDraw.Draw(img)
        mx, my, s = self._world_to_map(char_x, char_y)
        # pulsing circle + crosshair for the character
        r = 8
        draw.ellipse([mx - r, my - r, mx + r, my + r], outline=(46, 204, 113), width=3)
        draw.ellipse([mx - r - 4, my - r - 4, mx + r + 4, my + r + 4], outline=(46, 204, 113), width=1)
        draw.line([mx - r - 8, my, mx + r + 8, my], fill=(46, 204, 113), width=1)
        draw.line([mx, my - r - 8, mx, my + r + 8], fill=(46, 204, 113), width=1)
        # ego position (current camera)
        if ego_idx < len(self.ego):
            ex, ey = self.ego[ego_idx, 0], self.ego[ego_idx, 1]
            emx, emy, _ = self._world_to_map(ex, ey)
            draw.ellipse([emx - 5, emy - 5, emx + 5, emy + 5], fill=(0, 200, 200))
        return img


# --------------------------------------------------------------------------- #
# Coarse lighting from background (simple directional + ambient)
# --------------------------------------------------------------------------- #
def estimate_light_direction(scene_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate a coarse light direction & ambient from the scene image.

    Uses the brightness gradient across the image as a proxy for sun azimuth.
    Returns (light_dir_xy_unit, ambient) where light_dir is in image space
    (pointing toward the brighter side).
    """
    g = scene_rgb.mean(axis=-1)  # HxW
    H, W = g.shape
    # horizontal/vertical brightness gradient
    gx = np.mean(g[:, W // 2:], axis=None) - np.mean(g[:, : W // 2], axis=None)
    gy = np.mean(g[H // 2:, :], axis=None) - np.mean(g[: H // 2, :], axis=None)
    norm = np.hypot(gx, gy) + 1e-6
    ambient = float(np.clip(0.35 + 0.2 * (1.0 - min(g.mean(), 1.0)), 0.3, 0.7))
    return np.array([gx / norm, gy / norm], dtype=np.float32), ambient


def shade(rgb: np.ndarray, light_dir_img: np.ndarray, ambient: float) -> np.ndarray:
    """Cheap shading: brighten the side of the character facing the light.

    light_dir_img is a 2-vector in image space (x=right, y=down). We build a
    soft lateral gradient across the character's bounding region and multiply.
    This is a rough approximation, not physically-based.
    """
    H, W = rgb.shape[:2]
    if np.linalg.norm(light_dir_img) < 1e-3:
        return rgb
    xs = np.linspace(-1, 1, W)
    ys = np.linspace(-1, 1, H)
    XX, YY = np.meshgrid(xs, ys)
    grad = 0.5 + 0.5 * (XX * light_dir_img[0] + YY * light_dir_img[1]) * 0.6
    grad = grad[..., None]
    lit = rgb * ambient + rgb * (1.0 - ambient) * grad
    return np.clip(lit, 0, 1)


def add_contact_shadow(composite: np.ndarray, foot_pixel, radius: float = 25.0,
                       darkness: float = 0.45) -> np.ndarray:
    """Stamp a soft elliptical shadow at the character's feet to ground it.

    foot_pixel: (x, y) image coordinate of the foot contact point.
    radius: shadow radius in pixels (squared for ellipse).
    darkness: 0=no shadow, 1=black.
    """
    H, W = composite.shape[:2]
    fx, fy = foot_pixel
    ys, xs = np.mgrid[0:H, 0:W]
    # elliptical falloff, wider than tall (ground projection)
    d2 = ((xs - fx) / radius) ** 2 + ((ys - fy) / (radius * 0.4)) ** 2
    falloff = np.clip(1.0 - d2, 0, 1) ** 1.5
    shadow = falloff[..., None] * darkness
    return np.clip(composite * (1.0 - shadow), 0, 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    logger.info("Loading dataset + trainer ...")
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
    logger.info("Checkpoint loaded.")

    # ---- baked character sequence ----
    seq = np.load(args.seq)
    n_anim = int(seq["n_frames"])
    base_meshes = {}
    for name in MESH_NAMES:
        base_meshes[name] = {
            "verts": torch.from_numpy(seq[f"{name}/verts"]).float().to(device),
            "faces": torch.from_numpy(seq[f"{name}/faces"]).int().to(device),
            "uvs": torch.from_numpy(seq[f"{name}/uvs"]).float().to(device),
            "tex": torch.from_numpy(seq[f"{name}/tex"]).float().to(device) / 255.0,
        }
    logger.info("Loaded %d animation frames, %d meshes.", n_anim, len(base_meshes))

    import nvdiffrast.torch as dr
    rast = dr.RasterizeCudaContext(device=device)

    num_cams = dataset.pixel_source.num_cams
    full = dataset.full_image_set
    camera_downscale = trainer._get_downscale_factor()

    # ---- resolve positions / textures / frames / cameras ----
    positions = parse_positions(args.positions)
    clothes_tex_list = parse_texture_list(args.clothes_textures, base_meshes["clothes_1"]["tex"])
    cam_ids = [int(c) for c in args.cameras.split(",")] if args.cameras else list(range(num_cams))
    cam_ids = [c for c in cam_ids if c < num_cams]

    num_frames = dataset.num_img_timesteps
    if args.frames:
        frame_ids = [int(f) for f in args.frames.split(",")]
    else:
        # Render every Nth frame; default step=1 (real-time playback at scene fps).
        step = max(1, args.frame_step)
        frame_ids = list(range(0, num_frames, step))
        if args.max_output_frames > 0:
            frame_ids = frame_ids[: args.max_output_frames]
    frame_ids = [f for f in frame_ids if 0 <= f < num_frames]
    logger.info("cameras=%s positions=%d textures=%d frames=%d", cam_ids, len(positions), len(clothes_tex_list), len(frame_ids))

    # ---- precompute scene rgb+depth per (frame, cam) once (shared across pos/tex) ----
    scene_cache = {}

    def get_scene(frame_idx: int, cam_idx: int):
        key = (frame_idx, cam_idx)
        if key in scene_cache:
            return scene_cache[key]
        img_idx = frame_idx * num_cams + cam_idx
        img_infos, cam_infos = full.get_image(img_idx, camera_downscale)
        for k, v in img_infos.items():
            if isinstance(v, torch.Tensor):
                img_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)
        results = trainer(img_infos, cam_infos)
        rgb = results["rgb"].clamp(0, 1).detach().cpu().numpy()
        depth = results["depth"].detach().cpu().numpy().squeeze(-1)  # HxW meters
        # Background-only depth (static scene, no cars/pedestrians) for ground Z
        if "Background_depth" in results:
            bg_depth = results["Background_depth"].detach().cpu().numpy().squeeze(-1)
        else:
            bg_depth = depth  # fallback: use full depth
        H = int(cam_infos["height"].item())
        W = int(cam_infos["width"].item())
        c2w = cam_infos["camera_to_world"].detach().cpu().numpy()
        K = cam_infos["intrinsics"].detach().cpu().numpy()
        cam_name = str(cam_infos["cam_name"])
        # free
        del results, img_infos, cam_infos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scene_cache[key] = (rgb, depth, bg_depth, c2w, K, H, W, cam_name)
        return scene_cache[key]

    # ---- compute stride-matched animation speed ----
    # Priority: --stride flag > JSON gait params > --cycle_stride > fixed anim_speed
    loop_len = max(1, n_anim // 2)  # one gait cycle = 20 frames

    # Try reading gait params from trajectory JSON (written by trajectory_previewer)
    gait_from_json = None
    _traj_json_full = None  # full JSON dict (for top-level total_length)
    if args.path_json:
        try:
            import json as _json
            _traj_json_full = _json.load(open(args.path_json))
            if "gait" in _traj_json_full:
                gait_from_json = _traj_json_full["gait"]
        except Exception:
            pass

    if args.stride > 0 and len(frame_ids) > 1:
        # Legacy: explicit --stride flag overrides everything
        traj_pts = []
        for oi in range(len(frame_ids)):
            tt = oi / max(1, len(frame_ids) - 1)
            if args.path_json:
                tx, ty = sample_traj_json(args.path_json, tt)
            else:
                wp = parse_waypoints(args.path)
                tx, ty = sample_polyline(wp, tt)
            traj_pts.append((tx, ty))
        traj_pts = np.array(traj_pts)
        path_len = float(np.sum(np.linalg.norm(np.diff(traj_pts, axis=0), axis=1)))
        per_frame_dist = path_len / max(1, len(frame_ids) - 1)
        args.anim_speed = (per_frame_dist / args.stride) * loop_len
        logger.info("Stride-matched (--stride): path=%.1fm, %d frames → anim_speed=%.3f",
                    path_len, len(frame_ids), args.anim_speed)

    elif gait_from_json is not None:
        # Use gait params from trajectory_previewer (ensures slide-free + complete)
        # total_length is at JSON top level, not inside gait dict
        path_len = 0.0
        if _traj_json_full is not None:
            path_len = _traj_json_full.get("total_length", 0.0)
        if path_len < 1e-6:
            # fallback: recalculate from trajectory array
            traj_pts = np.array(_traj_json_full.get("trajectory", [])) if _traj_json_full else np.array([])
            if len(traj_pts) > 1:
                path_len = float(np.sum(np.linalg.norm(np.diff(traj_pts, axis=0), axis=1)))

        cycle_stride = gait_from_json.get("cycle_stride", 2.6)
        n_cycles = path_len / cycle_stride if cycle_stride > 0 else 0
        args.anim_speed = (n_cycles * loop_len) / max(1, len(frame_ids))

        logger.info("Gait-matched (from JSON): path=%.1fm, stride=%.1fm/cycle, "
                    "%d steps, %.1f cycles → anim_speed=%.3f (slide-free)",
                    path_len, cycle_stride, int(path_len / (cycle_stride / 2)),
                    n_cycles, args.anim_speed)
        logger.info("  speed=%.1fm/s, step_freq=%.1fHz",
                    gait_from_json.get("char_speed", 0),
                    gait_from_json.get("step_freq", 0))

    elif args.cycle_stride > 0 and len(frame_ids) > 1:
        # Compute from --cycle_stride flag
        traj_pts = []
        for oi in range(len(frame_ids)):
            tt = oi / max(1, len(frame_ids) - 1)
            if args.path_json:
                tx, ty = sample_traj_json(args.path_json, tt)
            else:
                wp = parse_waypoints(args.path)
                tx, ty = sample_polyline(wp, tt)
            traj_pts.append((tx, ty))
        traj_pts = np.array(traj_pts)
        path_len = float(np.sum(np.linalg.norm(np.diff(traj_pts, axis=0), axis=1)))
        n_cycles = path_len / args.cycle_stride
        args.anim_speed = (n_cycles * loop_len) / max(1, len(frame_ids))
        logger.info("Gait-matched (--cycle_stride): path=%.1fm, stride=%.1fm/cycle → "
                    "anim_speed=%.3f (%.0f steps, slide-free)",
                    path_len, args.cycle_stride, args.anim_speed,
                    path_len / (args.cycle_stride / 2))

    else:
        logger.info("Using fixed anim_speed=%.2f (no gait matching)", args.anim_speed)

    if args.mode == "video":
        render_video(args, get_scene, base_meshes, rast, n_anim, cam_ids, frame_ids, device)
    elif args.mode == "multicam_grid":
        render_multicam_grid(args, get_scene, base_meshes, rast, n_anim, cam_ids, frame_ids, device)
    else:
        render_multiview(args, get_scene, base_meshes, rast, n_anim, cam_ids, frame_ids, positions, clothes_tex_list, device, num_cams)


def render_video(args, get_scene, base_meshes, rast, n_anim, cam_ids, frame_ids, device):
    """Single-camera video with depth occlusion, ground anchoring, contact shadow."""
    cam_idx = cam_ids[0]
    writer = imageio.get_writer(args.out, mode="I", fps=args.fps)

    # Precompute character XY trajectory for ground Z table
    char_traj_xy = []
    for oi in range(len(frame_ids)):
        tt = oi / max(1, len(frame_ids) - 1)
        if args.path_json:
            tx, ty = sample_traj_json(args.path_json, tt)
        else:
            wp = parse_waypoints(args.path)
            tx, ty = sample_polyline(wp, tt)
        char_traj_xy.append((tx, ty))
    char_traj_xy = np.array(char_traj_xy)

    # Precompute ground Z table (per-frame, depth-sampled + interpolated)
    if args.adaptive_ground_z:
        gz_table = precompute_gz_table(get_scene, frame_ids, char_traj_xy, cam_idx, args.ground_z)
    else:
        gz_table = np.full(len(frame_ids), args.ground_z)

    for oi, frame_idx in enumerate(frame_ids):
        scene_rgb, scene_depth, scene_bg_depth, c2w, K, H, W, _ = get_scene(frame_idx, cam_idx)
        t = oi / max(1, len(frame_ids) - 1)
        if args.path_json:
            x, y = sample_traj_json(args.path_json, t)
            yaw = traj_json_yaw(args.path_json, t)
        else:
            waypoints = parse_waypoints(args.path)
            x, y = sample_polyline(waypoints, t)
            yaw = polyline_yaw_at(waypoints, t)

        gz = gz_table[oi]
        position = (x, y, gz)
        # animation: advance ~1 anim cycle per ~1m of travel so stride matches
        # displacement. anim_speed here is anim-frames per output-frame.
        anim_idx = int((oi * args.anim_speed) % n_anim)
        M = build_world_transform(position, yaw, args.scale, feet_offset=args.feet_offset)

        composite = scene_rgb.copy()
        light_dir, ambient = estimate_light_direction(scene_rgb)

        # collect foot pixel for contact shadow (project character origin)
        foot_world = np.array([position[0], position[1], gz, 1.0])
        foot_cam = np.linalg.inv(c2w) @ foot_world
        foot_px = (K[0, 0] * foot_cam[0] / foot_cam[2] + K[0, 2],
                   K[1, 1] * foot_cam[1] / foot_cam[2] + K[1, 2])

        char_mask = np.zeros((H, W), dtype=bool)
        for name in MESH_NAMES:
            m = base_meshes[name]
            verts_w = apply_transform(m["verts"][anim_idx].cpu().numpy(), M)
            rgba, char_depth = rasterize_mesh(rast, torch.from_numpy(verts_w).float().to(device),
                                              m["faces"], m["uvs"], m["tex"], c2w, K, H, W, device)
            # depth occlusion: keep char pixel only if char is closer
            closer = (char_depth < scene_depth) & (rgba[..., 3] > 0.5)
            alpha = closer[..., None].astype(np.float32)
            rgb_layer = shade(rgba[..., :3], light_dir, ambient)
            composite = composite * (1 - alpha) + rgb_layer * alpha
            char_mask |= closer

        # contact shadow at the feet, only on background pixels (not over body)
        if char_mask.any():
            composite = add_contact_shadow(composite, foot_px, radius=max(15.0, 30.0 * 8.0 / max(position[0], 1.0)))

        frame_u8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
        writer.append_data(frame_u8)
        if oi % 10 == 0:
            logger.info("  video frame %d/%d pos=%s", oi, len(frame_ids), position)
    writer.close()
    logger.info("Saved video to %s", args.out)


def render_multicam_grid(args, get_scene, base_meshes, rast, n_anim, cam_ids, frame_ids, device):
    """Multi-camera grid video: displays all cameras in a single frame grid layout.

    Creates a video where each frame shows all requested camera views arranged
    in a grid, with the character composited into each view using the same
    world position (so multi-view consistency is preserved).
    """
    # Layout: for 5 cameras, use 2x3 grid (front, front_left, front_right, left, right)
    # Order: front (0), front_left (1), front_right (2), left (3), right (4)
    n_cams = len(cam_ids)
    if n_cams <= 1:
        logger.warning("multicam_grid mode requires multiple cameras; falling back to video mode")
        return render_video(args, get_scene, base_meshes, rast, n_anim, cam_ids, frame_ids, device)

    # Determine grid layout — fixed 3x2 grid matching Waymo camera topology:
    #   [front_left] [front] [front_right]
    #   [left      ] [   ]  [right      ]
    # cam_id -> grid cell (row, col). Missing cells stay black.
    GRID_POS = {1: (0, 0), 0: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 2)}
    grid_cols, grid_rows = 3, 2

    writer = imageio.get_writer(args.out, mode="I", fps=args.fps)

    # Precompute character's full trajectory XY for the BEV mini-map
    char_traj_xy = []
    for oi in range(len(frame_ids)):
        tt = oi / max(1, len(frame_ids) - 1)
        if args.path_json:
            tx, ty = sample_traj_json(args.path_json, tt)
        else:
            wp = parse_waypoints(args.path)
            tx, ty = sample_polyline(wp, tt)
        char_traj_xy.append((tx, ty))
    char_traj_xy = np.array(char_traj_xy)

    # Build BEV mini-map (needs scene_dir from config)
    bev_map = None
    config_path = os.path.join(os.path.dirname(args.resume_from), "config.yaml")
    if os.path.exists(config_path):
        cfg_data = OmegaConf.load(config_path).data
        scene_dir = os.path.join(cfg_data.data_root, f"{int(cfg_data.scene_idx):03d}")
        if os.path.isdir(scene_dir):
            bev_map = BevMiniMap(scene_dir, char_traj_xy, map_size=480)
            logger.info("BEV mini-map enabled for grid center cell")

    # Get first frame to determine resolution
    first_rgb, first_depth, first_bg_depth, first_c2w, first_K, H, W, _ = get_scene(frame_ids[0], cam_ids[0])

    # Pre-compute ground Z along the entire trajectory (a Z lookup table).
    # For each sampled point, we query the 3DGS depth to find the actual road
    # surface Z at that (x,y). During rendering, we interpolate this table —
    # stable (no jitter), accurate (follows terrain), no per-frame depth reads.
    if args.adaptive_ground_z:
        gz_table = precompute_gz_table(
            get_scene, frame_ids, char_traj_xy, cam_ids[0], args.ground_z)
    else:
        # flat ground: every point has the same Z
        gz_table = np.full(len(frame_ids), args.ground_z)

    for oi, frame_idx in enumerate(frame_ids):
        t = oi / max(1, len(frame_ids) - 1)
        if args.path_json:
            x, y = sample_traj_json(args.path_json, t)
            yaw = traj_json_yaw(args.path_json, t)
        else:
            waypoints = parse_waypoints(args.path)
            x, y = sample_polyline(waypoints, t)
            yaw = polyline_yaw_at(waypoints, t)

        # Ground Z from the precomputed table (interpolated, stable)
        gz = gz_table[oi]
        position = (x, y, gz)

        anim_idx = int((oi * args.anim_speed) % n_anim)
        M = build_world_transform(position, yaw, args.scale, feet_offset=args.feet_offset)

        # Render each camera view
        cam_frames = []
        for ci, cam_idx in enumerate(cam_ids):
            scene_rgb, scene_depth, scene_bg_depth, c2w, K, H, W, cam_name = get_scene(frame_idx, cam_idx)
            composite = scene_rgb.copy()
            light_dir, ambient = estimate_light_direction(scene_rgb)

            foot_world = np.array([position[0], position[1], gz, 1.0])
            foot_cam = np.linalg.inv(c2w) @ foot_world
            foot_px = (K[0, 0] * foot_cam[0] / foot_cam[2] + K[0, 2],
                       K[1, 1] * foot_cam[1] / foot_cam[2] + K[1, 2])

            char_mask = np.zeros((H, W), dtype=bool)
            all_verts_w = []  # collect transformed verts for tight bbox
            for name in MESH_NAMES:
                m = base_meshes[name]
                verts_w = apply_transform(m["verts"][anim_idx].cpu().numpy(), M)
                all_verts_w.append(verts_w)
                rgba, char_depth = rasterize_mesh(rast, torch.from_numpy(verts_w).float().to(device),
                                                  m["faces"], m["uvs"], m["tex"], c2w, K, H, W, device)
                closer = (char_depth < scene_depth) & (rgba[..., 3] > 0.5)
                alpha = closer[..., None].astype(np.float32)
                rgb_layer = shade(rgba[..., :3], light_dir, ambient)
                composite = composite * (1 - alpha) + rgb_layer * alpha
                char_mask |= closer

            if char_mask.any():
                composite = add_contact_shadow(composite, foot_px, radius=max(15.0, 30.0 * 8.0 / max(position[0], 1.0)))

            # Add camera name label + tight 3D bounding box around character
            from PIL import Image, ImageDraw, ImageFont
            cam_pil = Image.fromarray((np.clip(composite, 0, 1) * 255).astype(np.uint8))
            draw = ImageDraw.Draw(cam_pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
            label = CAMERA_NAMES.get(cam_idx, f"cam{cam_idx}")
            draw.text((10, 10), label, fill=(255, 255, 0), font=font)
            cam_frames.append(np.array(cam_pil))

        # Arrange in grid — cameras have different resolutions (e.g. 1280x1920
        # vs 866x1920 for Waymo side cams), so resize all to a uniform cell size.
        # Placement follows the physical camera topology (GRID_POS), not order.
        # The center cell (1,1) shows a live BEV mini-map.
        cell_h, cell_w = 480, 640  # uniform cell resolution for the grid
        from PIL import Image as PImage
        grid_H, grid_W = cell_h * grid_rows, cell_w * grid_cols
        grid_frame = np.zeros((grid_H, grid_W, 3), dtype=np.uint8)
        for ci, cam_idx in enumerate(cam_ids):
            if ci >= len(cam_frames):
                break
            pos = GRID_POS.get(cam_idx)
            if pos is None:
                continue
            row, col = pos
            pil = PImage.fromarray(cam_frames[ci]).resize((cell_w, cell_h), PImage.BILINEAR)
            y_start, y_end = row * cell_h, (row + 1) * cell_h
            x_start, x_end = col * cell_w, (col + 1) * cell_w
            grid_frame[y_start:y_end, x_start:x_end] = np.array(pil)

        # Place BEV mini-map in the center cell (row=1, col=1)
        if bev_map is not None:
            ego_idx = min(frame_idx, len(bev_map.ego) - 1) if hasattr(bev_map, 'ego') else 0
            bev_img = bev_map.render_frame(position[0], position[1], ego_idx=ego_idx)
            bev_img = bev_img.resize((cell_w, cell_h), PImage.BILINEAR)
            y_start, y_end = 1 * cell_h, 2 * cell_h
            x_start, x_end = 1 * cell_w, 2 * cell_w
            grid_frame[y_start:y_end, x_start:x_end] = np.array(bev_img)

        writer.append_data(np.ascontiguousarray(grid_frame))
        if oi % 10 == 0:
            logger.info("  multicam frame %d/%d pos=%s", oi, len(frame_ids), position)

    writer.close()
    logger.info("Saved multicam grid video to %s", args.out)


def render_multiview(args, get_scene, base_meshes, rast, n_anim, cam_ids, frame_ids, positions, clothes_tex_list, device, num_cams):
    """Batch multi-view PNGs across positions x textures x frames x cameras."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(positions) * len(clothes_tex_list) * len(frame_ids) * len(cam_ids)
    logger.info("Generating %d images into %s", total, out_dir)
    done = 0

    for pi, position in enumerate(positions):
        yaw = args.yaw
        M = build_world_transform(position, yaw, args.scale, feet_offset=args.feet_offset)
        for ti, clothes_tex in enumerate(clothes_tex_list):
            sub = out_dir / f"pos{pi:02d}_tex{ti:02d}"
            sub.mkdir(parents=True, exist_ok=True)
            for fi, frame_idx in enumerate(frame_ids):
                anim_idx = int((frame_idx * args.anim_speed) % n_anim)
                for cam_idx in cam_ids:
                    scene_rgb, scene_depth, scene_bg_depth, c2w, K, H, W, cam_name = get_scene(frame_idx, cam_idx)
                    composite = scene_rgb.copy()
                    light_dir, ambient = estimate_light_direction(scene_rgb)
                    for name in MESH_NAMES:
                        m = base_meshes[name]
                        # swap clothes texture for this batch
                        tex = clothes_tex if name == "clothes_1" else m["tex"]
                        verts_w = apply_transform(m["verts"][anim_idx].cpu().numpy(), M)
                        rgba, char_depth = rasterize_mesh(rast, torch.from_numpy(verts_w).float().to(device),
                                                          m["faces"], m["uvs"], tex, c2w, K, H, W, device)
                        closer = (char_depth < scene_depth) & (rgba[..., 3] > 0.5)
                        alpha = closer[..., None].astype(np.float32)
                        rgb_layer = shade(rgba[..., :3], light_dir, ambient)
                        composite = composite * (1 - alpha) + rgb_layer * alpha

                    frame_u8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
                    fname = sub / f"frame{frame_idx:03d}_{cam_name}.png"
                    imageio.imwrite(str(fname), frame_u8)
                    done += 1
                    if done % 20 == 0 or done == total:
                        logger.info("  %d/%d  %s", done, total, fname.name)

    logger.info("Done. %d images in %s", done, out_dir)


# --------------------------------------------------------------------------- #
# CLI parsing helpers
# --------------------------------------------------------------------------- #
def parse_positions(spec: str) -> List[Tuple[float, float, float]]:
    """Parse 'x,y,z;x,y,z' into a list of positions. Empty -> single default."""
    if not spec or not spec.strip():
        return [(6.0, -1.0, 0.0)]
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        vals = [float(v) for v in part.split(",")]
        if len(vals) != 3:
            raise ValueError(f"position needs x,y,z; got {part!r}")
        out.append((vals[0], vals[1], vals[2]))
    return out


def parse_texture_list(spec: str, default_tex: torch.Tensor) -> List[torch.Tensor]:
    """Parse ';'-separated PNG paths into a list of texture tensors.

    Empty entries use the original clothes texture. Returns at least one entry.
    """
    out = []
    if not spec or not spec.strip():
        return [default_tex]
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            out.append(default_tex)
            continue
        img = imageio.imread(part)  # HxWx(3 or 4)
        if img.ndim == 2:
            img = np.stack([img] * 3 + [255 * np.ones_like(img)], axis=-1)
        elif img.shape[-1] == 3:
            img = np.concatenate([img, 255 * np.ones((*img.shape[:2], 1), dtype=img.dtype)], axis=-1)
        # resize to match the original clothes tex resolution
        th, tw = default_tex.shape[:2]
        if img.shape[0] != th or img.shape[1] != tw:
            from PIL import Image
            img_pil = Image.fromarray(img.astype(np.uint8)).resize((tw, th), Image.BILINEAR)
            img = np.asarray(img_pil)
        out.append(torch.from_numpy(img.astype(np.float32) / 255.0).to(default_tex.device))
    return out if out else [default_tex]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Adversarial-sample compositor (3DGS + mesh)")
    parser.add_argument("--resume_from", required=True)
    parser.add_argument("--seq", default="outputs/assets/runner_seq.npz")
    parser.add_argument("--mode", choices=["video", "multiview", "multicam_grid"], default="multicam_grid",
                        help="video=single camera; multiview=PNG batch; multicam_grid=grid video with all cameras + BEV")
    # video mode
    parser.add_argument("--out", default="outputs/runner_composite.mp4", help="output video")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame_step", type=int, default=1, help="render every Nth scene frame (1=real-time)")
    parser.add_argument("--path", default="", help="polyline waypoints 'x1,y1;x2,y2;...' for character (empty=default)")
    parser.add_argument("--path_json", default="", help="trajectory JSON from tools/plan_trajectory_bev.py (overrides --path)")
    parser.add_argument("--lateral", type=float, default=-1.0, help="Y offset (legacy, unused when --path set)")
    # shared
    parser.add_argument("--scale", type=float, default=0.90, help="character uniform scale")
    parser.add_argument("--yaw", type=float, default=0.0, help="heading degrees (0 = +X forward)")
    parser.add_argument("--ground_z", type=float, default=0.0, help="world Z of ground plane (feet anchor)")
    parser.add_argument("--adaptive_ground_z", action="store_true", default=True,
                        help="auto-estimate ground Z from scene depth (default True)")
    parser.add_argument("--no-adaptive_ground_z", dest="adaptive_ground_z", action="store_false",
                        help="disable adaptive ground Z estimation")
    parser.add_argument("--feet_offset", type=float, default=0.01, help="lift character so feet don't clip ground")
    parser.add_argument("--anim_speed", type=float, default=1, help="anim frames per output frame (overridden by --stride/--cycle_stride/JSON gait)")
    parser.add_argument("--stride", type=float, default=0,
                        help="[legacy] meters per running cycle (2 steps). Prefer --cycle_stride.")
    parser.add_argument("--cycle_stride", type=float, default=0,
                        help="meters per gait cycle (2 steps) for slide-free animation. "
                             "2.6=1.3m/step. 0=auto from JSON gait or fixed anim_speed. "
                             "Priority: --stride > JSON gait > --cycle_stride > --anim_speed")
    parser.add_argument("--max_output_frames", type=int, default=0, help="video mode: cap frames; 0=all")
    # multiview mode
    parser.add_argument("--out_dir", default="outputs/adversarial/", help="output dir (multiview mode)")
    parser.add_argument("--cameras", default="", help="comma-sep cam ids, e.g. 0,1,2 (empty=all)")
    parser.add_argument("--positions", default="", help="'x,y,z;x,y,z' placements (empty=default)")
    parser.add_argument("--clothes_textures", default="", help="';'-separated PNG paths (empty=original)")
    parser.add_argument("--frames", default="", help="comma-sep frame ids (empty=auto)")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    main(args)
