# Adversarial Drive Studio

在 [DriveStudio](https://github.com/ziyc/drivestudio) (OmniRe) 重建的自动驾驶 3DGS 场景中，沿自定义轨迹放置一个带纹理的跑步角色，生成对抗样本视频。

## 功能

- 🗺️ **BEV 轨迹规划**：基于 LiDAR 点云 + 障碍物包围盒的鸟瞰图，点击画轨迹
- 🎮 **3D 实时预览**：浏览器中以 3DGS 实时渲染场景，点击添加路径点，检查碰撞、地面贴合
- 👟 **步频匹配**：根据轨迹长度和视频帧数自动计算步频，确保**不滑步、不超时、不残留**
- 🎬 **五视角渲染**：五相机 + BEV 小地图的 3×2 网格视频，深度遮挡合成
- 📦 **导出/查看工具**：导出 3DGS checkpoint 为 PLY，交互式查看器

## 快速开始

### 前提条件

1. 已安装并配置好 [DriveStudio](https://github.com/ziyc/drivestudio)
2. 已有训练好的场景 checkpoint（如 `outputs/waymo_omnire/scene23/checkpoint_final.pth`）
3. 已下载 Waymo 处理数据（如 `data/waymo/processed/training/023/`）
4. GPU 环境（conda 环境 `drivestudio`）

### 一键安装

```bash
# 在 DriveStudio 根目录下执行
cd /path/to/drivestudio

# 克隆本项目
git clone https://github.com/<your-username>/adversarial_drive_studio.git

# 运行安装脚本（自动复制脚本到 DriveStudio、下载角色资产）
cd adversarial_drive_studio
bash setup.sh
```

安装脚本会：
1. 将 `tools/` 下的脚本复制到 DriveStudio 的 `tools/` 目录
2. 将 `configs/omnire_extended_cam.yaml` 复制到 DriveStudio 的 `configs/`
3. 下载角色动画资产 `runner_seq.npz`（约 118MB）到 `outputs/assets/`

### 三步使用

```bash
cd /path/to/drivestudio
conda activate drivestudio

# 步骤 1：3D 预览器画轨迹（浏览器打开 localhost:8080）
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --scene_dir data/waymo/processed/training/023 \
    --port 8080

# 步骤 2：在浏览器中画轨迹 → 点击 export → 生成 traj_live.json

# 步骤 3：渲染视频
python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --path_json outputs/waymo_omnire/scene23/trajectories/traj_live.json \
    --out outputs/waymo_omnire/scene23/videos_eval/scene23_v3.mp4
```

> 详细参数说明见 [docs/trajectory_pipeline.md](docs/trajectory_pipeline.md)

## 项目结构

```
adversarial_drive_studio/
├── setup.sh                         # 一键安装脚本
├── tools/
│   ├── trajectory_previewer.py      # 3D 实时预览器（viser + nerfview + gsplat）
│   ├── render_runner_video.py       # 最终视频渲染（nvdiffrast + gsplat）
│   ├── bev_trajectory_planner.py    # BEV 轨迹规划（matplotlib）
│   ├── gait_utils.py                # 步频计算模块
│   ├── bake_runner_frames.py        # 角色动画烘焙（Blender，一次性）
│   ├── export_gaussians_ply.py      # 导出 3DGS checkpoint 为 PLY
│   ├── visualize_gaussian_ply.py    # PLY 交互式查看器
│   └── viewer.py                    # 3DGS 场景查看器
├── configs/
│   └── omnire_extended_cam.yaml     # 五相机训练配置
├── docs/
│   ├── trajectory_pipeline.md       # 完整 pipeline 文档
│   └── adversarial_composition.md   # 技术细节文档
└── assets/
    └── download_assets.sh           # 下载角色动画资产
```

## 各场景使用

场景编号对照（DriveStudio 输出目录 vs Waymo 数据目录）：

| 场景 | checkpoint | 数据目录 |
|------|-----------|----------|
| scene23 | `outputs/waymo_omnire/scene23/` | `data/waymo/processed/training/023/` |
| scene114 | `outputs/waymo_omnire/scene114/` | `data/waymo/processed/training/114/` |
| scene552 | `outputs/waymo_omnire/scene552/` | `data/waymo/processed/training/552/` |

## 技术要点

### 步频匹配（不滑步）

```
轨迹长度 L → 总步数 = L / 1.3m
           → 步态周期数 = L / 2.6m
           → 动画总帧 = 周期数 × 20帧
           → anim_speed = 动画总帧 / 视频帧数
```

角色恰好走完整个轨迹，每步 1.3m，动画周期与位移严格匹配。

### 深度遮挡合成

角色 mesh 用 nvdiffrast 光栅化，与 3DGS 背景的深度图做逐像素比较：
- `mesh_depth < bg_depth` → 角色可见（被背景遮挡的部分自动隐藏）
- 这样角色能被场景中的车辆、建筑正确遮挡

### 坐标系

- 场景 3DGS：ego-normalized 世界坐标系（Z-up，原点为第 0 帧 ego 位姿）
- 角色动画：Z-up（脚 Z≈0，头 Z≈1.82）
- 两者一致，无需额外转换

## 依赖

本项目依赖 DriveStudio 已安装的环境，额外需要：

```
nvdiffrast    # mesh 光栅化
viser         # 网页查看器
nerfview      # 3DGS 查看器
```

这些已在 DriveStudio 的 conda 环境中预装。

## 致谢

- [DriveStudio / OmniRe](https://github.com/ziyc/drivestudio) — 3DGS 自动驾驶场景重建
- [Mixamo](https://www.mixamo.com/) — 角色动画
