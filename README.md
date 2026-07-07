<div align="center">

# 🏃 DriveHack

### Drop a ghost runner into any 3DGS driving scene, along a custom trajectory, slide-free.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p>
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="docs/trajectory_pipeline.md">Docs</a> •
  <a href="#how-it-works">How It Works</a>
</p>

<img src="assets/banner.png" width="100%" alt="DriveHack: 5-camera grid with ghost runner">

</div>

## Overview

DriveHack lets you **inject a fully-textured running character** into a [DriveStudio](https://github.com/ziyc/drivestudio) / OmniRe reconstructed autonomous driving scene, moving along a **custom trajectory** you draw in a browser-based 3D previewer — with **gait-matched, slide-free** animation and correct **depth occlusion** against the 3DGS background.

Built for **autonomous driving robustness testing**: generate adversarial pedestrian scenarios at scale.

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 🎮 Browser-based 3D Trajectory Editor

Draw trajectories directly in the 3DGS scene. Real-time gsplat rendering via viser/nerfview.

- Click to add waypoints
- Dynamic obstacles (vehicles/pedestrians) move with time slider
- Real-time 3D collision detection

</td>
<td width="50%" valign="top">

### 👟 Gait-Matched Animation

Never slide. Never overshoot. The step frequency is auto-computed from trajectory length and video duration.

```
length → steps → cycles → anim_speed
```

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🎬 5-Camera + BEV Grid Rendering

Waymo's 5 cameras tiled in a 3×2 grid with a BEV mini-map center cell. Depth-occluded compositing against the 3DGS background.

</td>
<td width="50%" valign="top">

### 📦 Export & Visualization

Export trained 3DGS checkpoints to standard PLY. Interactive viewers for both 3DGS scenes and PLY files.

</td>
</tr>
</table>

## Quick Start

### Prerequisites

1. [DriveStudio](https://github.com/ziyc/drivestudio) installed and configured
2. A trained scene checkpoint (e.g., `outputs/waymo_omnire/scene23/checkpoint_final.pth`)
3. Waymo processed data (e.g., `data/waymo/processed/training/023/`)

### Install

```bash
# Inside your DriveStudio root directory
cd /path/to/drivestudio

git clone https://github.com/<your-username>/DriveHack.git
cd DriveHack && bash setup.sh
```

That's it. `setup.sh` copies scripts, configs, and downloads character assets.

### Usage (2 steps)

```bash
cd /path/to/drivestudio
conda activate drivestudio

# 1️⃣ Draw trajectory in browser (open localhost:8080)
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --scene_dir data/waymo/processed/training/023 \
    --port 8080

# 2️⃣ Render the video
python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --path_json outputs/waymo_omnire/scene23/trajectories/traj_live.json \
    --out outputs/waymo_omnire/scene23/videos_eval/scene23_v3.mp4
```

> 📖 Full parameter reference: [docs/trajectory_pipeline.md](docs/trajectory_pipeline.md)

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  BEV Planner     │     │  3D Previewer    │     │  Video Renderer  │
│  (matplotlib)    │ ──▶ │  (viser+gsplat)  │ ──▶ │  (nvdiffrast)    │
│                  │     │                  │     │                  │
│ • LiDAR + obs    │     │ • 3DGS live      │     │ • 5-cam + BEV    │
│ • Click waypoints│     │ • Click in 3D    │     │ • Depth occlusion│
│ • 2D collision   │     │ • Gait calc      │     │ • Gait-matched   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        ↓                        ↓                        ↓
     traj.json             traj_live.json          scene_v3.mp4
```

### Gait Matching (No Foot Sliding)

The character's step frequency is computed from physical parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `cycle_stride` | 2.6m | Distance per gait cycle (2 steps) |
| `step_length` | 1.3m | Distance per single step |
| `cycle_frames` | 20 | Animation frames per gait cycle |

Given trajectory length `L` and `N` video frames:
```
anim_speed = (L / 2.6) × 20 / N
```

This guarantees the character takes exactly `L/1.3` steps, covering the entire trajectory with consistent stride — no sliding, no leftover, no overtime.

## Project Structure

```
DriveHack/
├── setup.sh                         # One-click installer
├── tools/
│   ├── trajectory_previewer.py      # 3D real-time preview (viser + nerfview)
│   ├── render_runner_video.py       # Final video renderer (nvdiffrast)
│   ├── bev_trajectory_planner.py    # BEV trajectory planner (matplotlib)
│   ├── gait_utils.py                # Gait-matched speed calculator
│   ├── bake_runner_frames.py        # Character animation baker (Blender)
│   ├── export_gaussians_ply.py      # 3DGS checkpoint → PLY exporter
│   ├── visualize_gaussian_ply.py    # Interactive PLY viewer
│   └── viewer.py                    # 3DGS scene viewer
├── configs/
│   └── omnire_extended_cam.yaml     # 5-camera training config
├── docs/
│   ├── trajectory_pipeline.md       # Complete pipeline documentation
│   └── adversarial_composition.md   # Technical details
└── assets/
    └── download_assets.sh           # Character asset downloader
```

## Scene Index Reference

| Scene | Checkpoint | Data Directory |
|-------|-----------|---------------|
| scene23 | `outputs/waymo_omnire/scene23/` | `data/waymo/processed/training/023/` |
| scene114 | `outputs/waymo_omnire/scene114/` | `data/waymo/processed/training/114/` |
| scene552 | `outputs/waymo_omnire/scene552/` | `data/waymo/processed/training/552/` |

## Custom Character (Blender Baking)

Want to use your own character? Bake a Mixamo animation into the npz format:

```bash
~/blender/blender-4.4.3-linux-x64/blender --background \
    --python tools/bake_runner_frames.py -- \
    --blend your_character.blend \
    --out outputs/assets/runner_seq.npz \
    --frames 40
```

> 📖 Full guide: [docs/baking_guide.md](docs/baking_guide.md)

## Acknowledgements

- [DriveStudio / OmniRe](https://github.com/ziyc/drivestudio) — 3DGS urban scene reconstruction
- [Mixamo](https://www.mixamo.com/) — Character animations
- [viser](https://github.com/nerfstudio-project/viser) / [nerfview](https://github.com/nerfstudio-project/nerfview) — 3D visualization
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast) — Mesh rasterization

## Citation

If you find this useful for your research:

```bibtex
@misc{drivehack2026,
  title  = {DriveHack: Injecting Adversarial Characters into 3DGS Driving Scenes},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/<your-username>/DriveHack}
}
```

## License

MIT
