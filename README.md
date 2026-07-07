<div align="center">

# 🏃 DriveHack

### Drop a ghost runner into any 3DGS driving scene, along a custom trajectory, slide-free.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p>
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="docs/trajectory_pipeline.md">Docs</a>
</p>

<table>
<tr>
<td width="50%" align="center"><b>🎮 Trajectory Editor Demo</b></td>
<td width="50%" align="center"><b>🎬 Rendered Output (5-cam + BEV)</b></td>
</tr>
<tr>
<td width="50%"><video src="assets/trac_demo.mp4" controls width="100%"></video></td>
<td width="50%"><video src="assets/scene23.mp4" controls width="100%"></video></td>
</tr>
</table>

</div>

---

## 📖 Overview

**DriveHack** lets you inject a fully-textured running character into a [DriveStudio](https://github.com/ziyc/drivestudio) / OmniRe reconstructed autonomous driving scene. Draw a trajectory in a browser-based 3D editor, and the character runs along it with **gait-matched, slide-free** animation and correct **depth occlusion** against the 3DGS background.

Built for **autonomous driving robustness testing**: generate adversarial pedestrian scenarios at scale on photoreal 3DGS scenes.

> 💡 DriveHack is an **add-on** to [DriveStudio](https://github.com/ziyc/drivestudio). You need a working DriveStudio installation (with at least one trained scene checkpoint) before using DriveHack.

## ✨ Features

| | Feature | Description |
|---|---------|-------------|
| 🎮 | **Browser-based 3D Trajectory Editor** | Draw trajectories directly in the 3DGS scene. Real-time gsplat rendering via viser/nerfview. |
| 🚗 | **Dynamic Obstacles** | Vehicles/pedestrians/cyclists move with the time slider. Time-synchronized 3D collision detection. |
| 👟 | **Gait-Matched Animation** | Never slide. Never overshoot. Step frequency auto-computed from trajectory length & video duration. |
| 🎬 | **5-Camera + BEV Grid Rendering** | Waymo's 5 cameras tiled 3×2 with a BEV mini-map. Depth-occluded compositing. |
| 🧊 | **Custom Characters** | Bake any Mixamo character with Blender. |
| 📦 | **Export & Visualization** | Export 3DGS checkpoints to PLY. Interactive viewers for scenes and PLY files. |

## 🔰 Full Installation Guide

DriveHack runs **on top of** DriveStudio. Follow these steps in order.

### Step 0 — System Requirements

- **GPU**: NVIDIA RTX 30xx/40xx (16GB+ VRAM recommended)
- **OS**: Linux (Ubuntu 20.04+ tested)
- **CUDA**: 11.8+ (matches PyTorch)
- **Disk**: ~50GB for one Waymo scene (raw data + training output)

### Step 1 — Install DriveStudio

DriveStudio is the base 3DGS reconstruction framework. Install it first:

```bash
git clone https://github.com/ziyc/drivestudio.git
cd drivestudio
```

Follow the [DriveStudio installation guide](https://github.com/ziyc/drivestudio#installation) to set up the conda environment:

```bash
conda create -n drivestudio python=3.9
conda activate drivestudio

# Install PyTorch (match your CUDA version)
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install DriveStudio dependencies
pip install -r requirements.txt

# Install gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

Verify the installation:
```bash
python -c "import gsplat, nvdiffrast, viser, nerfview; print('All OK')"
```

> 📖 For DriveStudio issues (data download, training crashes, etc.), see the [DriveStudio README](https://github.com/ziyc/drivestudio) and [Issues](https://github.com/ziyc/drivestudio/issues).

### Step 2 — Prepare Waymo Data

Download and process Waymo data following the [DriveStudio data docs](https://github.com/ziyc/drivestudio#data-preparation):

```bash
# Your processed data should look like:
data/waymo/processed/training/023/
├── ego_pose/          # 000.txt, 001.txt, ... (4×4 matrices)
├── lidar/             # 000.bin, 001.bin, ...
├── images/            # camera images
├── instances/         # instances_info.json, frame_instances.json
├── extrinsics/
├── intrinsics/
└── ...
```

### Step 3 — Train a Scene

Train a 3DGS scene using DriveStudio's config:

```bash
# Use the included 5-camera config (better coverage than default 3-cam)
python -m tools.train \
    --config_file configs/omnire_extended_cam.yaml \
    --output_root ./outputs \
    --project waymo_omnire \
    --run_name scene23 \
    dataset=waymo/5cams data.scene_idx=23
```

This takes ~2-5 hours per scene. When done, you'll have:
```
outputs/waymo_omnire/scene23/checkpoint_final.pth
```

> 💡 DriveHack includes `configs/omnire_extended_cam.yaml` which uses all 5 Waymo cameras (vs. DriveStudio's default 3). Run `bash setup.sh` to install it.

### Step 4 — Install DriveHack

Now install DriveHack on top:

```bash
# Inside your DriveStudio root directory
cd /path/to/drivestudio

git clone https://github.com/<your-username>/DriveHack.git
cd DriveHack && bash setup.sh
```

`setup.sh` will:
1. ✅ Copy 8 scripts to `tools/`
2. ✅ Copy the 5-camera config to `configs/`
3. ✅ Download `runner_seq.npz` (character animation, 118MB) — **[configure the download link first](assets/download_assets.sh)**

If auto-download isn't configured, see [Custom Character Baking](#-custom-character-baking) to generate it yourself, or manually place `runner_seq.npz` at `outputs/assets/runner_seq.npz`.

## 🚀 Quick Start

After [installation](#-full-installation-guide), generate an adversarial video in **2 commands**:

```bash
cd /path/to/drivestudio
conda activate drivestudio
```

**1️⃣ Draw a trajectory** (browser opens at `localhost:8080`):

```bash
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --scene_dir data/waymo/processed/training/023 \
    --port 8080
```

In the browser:
- 🖱️ **Click** in the 3D scene to add waypoints
- 🟢 Green spline appears (your trajectory)
- 📊 Right panel shows gait params: `len=27.7m | 21 steps | 1.4m/s | ✓ walking`
- 🚗 Drag `scene frame` slider → obstacles move in time
- ▶️ Check `play` → character animates along the path
- 💾 Click `export traj.json` → saves with gait params

**2️⃣ Render the video**:

```bash
python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --path_json outputs/waymo_omnire/scene23/trajectories/traj_live.json \
    --out outputs/waymo_omnire/scene23/videos_eval/scene23_v3.mp4
```

Done! Output is a 3×2 grid: 5 camera views + BEV mini-map, with the character depth-occluded by the scene.

> 📖 Full parameter reference: [docs/trajectory_pipeline.md](docs/trajectory_pipeline.md)

## 🧊 Custom Character Baking

Want your own character instead of the default runner? Bake a [Mixamo](https://www.mixamo.com/) animation:

```bash
# Requires Blender 4.x
~/blender/blender-4.4.3-linux-x64/blender --background \
    --python tools/bake_runner_frames.py -- \
    --blend your_character.blend \
    --out outputs/assets/runner_seq.npz \
    --frames 40
```

> 📖 Full guide: [docs/baking_guide.md](docs/baking_guide.md)

## 🔧 How It Works

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

The character's step frequency is physically computed:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `cycle_stride` | 2.6m | Distance per gait cycle (2 steps) |
| `step_length` | 1.3m | Distance per single step |
| `cycle_frames` | 20 | Animation frames per gait cycle |

Given trajectory length `L` and `N` video frames:
```
anim_speed = (L / 2.6) × 20 / N
```

This guarantees the character takes exactly `L/1.3` steps — **no sliding, no overshoot, no leftover**.

## 📂 Project Structure

```
DriveHack/
├── setup.sh                         # One-click installer
├── tools/
│   ├── trajectory_previewer.py      # 🎮 3D real-time preview (viser + nerfview)
│   ├── render_runner_video.py       # 🎬 Final video renderer (nvdiffrast)
│   ├── bev_trajectory_planner.py    # 🗺️ BEV trajectory planner (matplotlib)
│   ├── gait_utils.py                # 👟 Gait-matched speed calculator
│   ├── bake_runner_frames.py        # 🧊 Character animation baker (Blender)
│   ├── export_gaussians_ply.py      # 📦 3DGS checkpoint → PLY exporter
│   ├── visualize_gaussian_ply.py    # 👁️ Interactive PLY viewer
│   └── viewer.py                    # 🖥️ 3DGS scene viewer
├── configs/
│   └── omnire_extended_cam.yaml     # 5-camera training config
├── docs/
│   ├── trajectory_pipeline.md       # Complete pipeline docs
│   ├── baking_guide.md              # Blender baking guide
│   └── adversarial_composition.md   # Technical deep-dive
└── assets/
    ├── trac_demo.mp4                # Trajectory editor demo
    ├── scene23.mp4                  # Rendered output demo
    └── download_assets.sh           # Character asset downloader
```

## 📋 Scene Index Reference

DriveStudio output dirs vs Waymo data dirs (note the zero-padding):

| Scene | Checkpoint | Data Directory |
|-------|-----------|---------------|
| scene23 | `outputs/waymo_omnire/scene23/` | `data/waymo/processed/training/023/` |
| scene114 | `outputs/waymo_omnire/scene114/` | `data/waymo/processed/training/114/` |
| scene552 | `outputs/waymo_omnire/scene552/` | `data/waymo/processed/training/552/` |

## 🙏 Acknowledgements

DriveHack is built on top of these amazing open-source projects:

- **[DriveStudio / OmniRe](https://github.com/ziyc/drivestudio)** (ICLR 2025 Spotlight) — The foundational 3DGS urban scene reconstruction framework that DriveHack extends. Without Ziyc's excellent work, this project would not exist. If you use DriveHack, please also cite [OmniRe](https://arxiv.org/abs/2408.16760).
- **[gsplat](https://github.com/nerfstudio-project/gsplat)** — GPU-accelerated Gaussian Splatting rendering.
- **[viser](https://github.com/nerfstudio-project/viser) / [nerfview](https://github.com/nerfstudio-project/nerfview)** — 3D visualization and the viewer infrastructure.
- **[nvdiffrast](https://github.com/NVlabs/nvdiffrast)** — High-performance mesh rasterization (NVIDIA).
- **[Mixamo](https://www.mixamo.com/)** — Character rigging and animations (Adobe).
- **[Blender](https://www.blender.org/)** — 3D modeling and animation baking.

## 📄 Citation

If DriveHack is useful for your research:

```bibtex
@misc{drivehack2026,
  title  = {DriveHack: Injecting Adversarial Characters into 3DGS Driving Scenes},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/<your-username>/DriveHack}
}
```

And please cite the foundational work:

```bibtex
@inproceedings{omnire2025,
  title     = {OmniRe: Omni Urban Scene Reconstruction},
  author    = {Ziyu Chen and Jiawei Yang and Jiahui Huang and Kai Zhang and Simon Green and Evangelos Kalogerakis and Leonidas Guibas and Andreas Geiger},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

## 📜 License

MIT
