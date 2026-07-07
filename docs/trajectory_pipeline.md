# 对抗样本轨迹生成与渲染 Pipeline

## 概述

本 pipeline 用于在 DriveStudio 3DGS 重建的驾驶场景中，沿自定义轨迹放置一个跑步角色，生成对抗样本视频。

**核心工作流**：画轨迹 → 实时预览验证 → 渲染最终视频

```
bev_trajectory_planner.py  →  trajectory_previewer.py  →  render_runner_video.py
     (画轨迹)                  (3D实时预览+步频)            (五视角+BEV 渲染)
         ↓                            ↓                            ↓
    traj.json (BEV)            traj_live.json (3D)          scene_v3.mp4
```

---

## 环境准备

```bash
cd ~/4drivestudio
conda activate drivestudio
```

**必需文件**：
- checkpoint: `outputs/waymo_omnire/scene<N>/checkpoint_final.pth`
- 场景数据: `data/waymo/processed/training/<NNN>/`
- 角色动画: `outputs/assets/runner_seq.npz`（40帧跑步循环，已含纹理）

---

## 步骤 1：画轨迹（可选，也可直接在预览器里画）

用 BEV 鸟瞰图工具画轨迹，基于 LiDAR 点云 + 障碍物包围盒：

```bash
python tools/bev_trajectory_planner.py \
    --scene_dir data/waymo/processed/training/023
```

- **左键**：添加路径点（落在障碍物内会拒绝）
- **右键**：撤销上一个点
- **关闭窗口**：自动保存到 `outputs/waymo_omnire/scene23/trajectories/traj.json`
- BEV 图保存到 `outputs/waymo_omnire/scene23/bev/bev.png`

> 此步可跳过——直接在步骤 2 的 3D 预览器里画轨迹更直观。

---

## 步骤 2：3D 实时预览（画轨迹 + 验证 + 步频计算）

在浏览器中以 3DGS 实时渲染场景，点击添加路径点，检查碰撞、地面贴合、步频是否合理。

```bash
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --scene_dir data/waymo/processed/training/023 \
    --port 8080
```

> **注意**：`--scene_dir` 的路径。scene23 对应 `023`（3位数），scene552 对应 `552`。

浏览器打开 **http://localhost:8080**

### 操作方式

| 操作 | 说明 |
|------|------|
| **鼠标拖拽** | 旋转 3D 场景视角 |
| **滚轮** | 缩放 |
| **点击场景** | 添加路径点（射线-地面交点） |
| `undo last waypoint` | 撤销上一个点 |
| `clear all waypoints` | 清空所有点 |
| `traj progress` 滑块 | 拖动查看角色在轨迹上的位置 |
| `scene frame` 滑块 | 拖动改变时间（障碍物随时间运动） |
| `sync frame to traj` | 勾选后，轨迹进度与场景时间同步 |
| `play` | 自动播放动画 |
| `export traj.json` | **导出轨迹（含步频参数）** |

### 步频参数面板（右侧 "步频参数"）

实时显示：
```
len=27.7m | 21步 | 1.4m/s | 步频1.1Hz | anim_speed=1.071 | ✓ 正常步行
```

- **步态周期步幅** 滑块：一个步态周期（左右各一步）覆盖的距离，默认 2.6m（=1.3m/步）
- 速度自动评估：正常步行(≤1.4m/s) / 快走慢跑 / 跑步(≤4m/s) / ⚠ 非人类速度

### 步频不滑步的原理

```
轨迹长度 L → 总步数 = L / 1.3m
           → 步态周期数 = L / 2.6m
           → 动画总帧 = 周期数 × 20帧
           → anim_speed = 动画总帧 / 视频帧数
```

这样角色恰好走完整个轨迹，每步 1.3m，**不滑步、不超时、不残留**。

### 导出

点击 `export traj.json` 后保存到：
```
outputs/waymo_omnire/scene23/trajectories/traj_live.json
```

JSON 含完整步频参数（`gait` 字段），渲染时自动读取。

### 加载已有轨迹编辑

```bash
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --scene_dir data/waymo/processed/training/023 \
    --path_json outputs/waymo_omnire/scene23/trajectories/traj_live.json \
    --port 8080
```

### 停止预览器

```bash
pkill -f trajectory_previewer
```

---

## 步骤 3：渲染最终视频

```bash
python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --path_json outputs/waymo_omnire/scene23/trajectories/traj_live.json \
    --out outputs/waymo_omnire/scene23/videos_eval/scene23_v3.mp4
```

> 以上是最简命令。`--mode multicam_grid`（五视角+BEV）、`--adaptive_ground_z`（地面贴合）、`--scale 0.90` 已是默认值，无需显式指定。

### 渲染日志确认

启动后日志应显示：
```
Gait-matched (from JSON): path=27.7m, stride=2.6/cycle, 21 steps, 10.7 cycles → anim_speed=1.071 (slide-free)
  speed=1.4m/s, step_freq=1.1Hz
```

如果看到 `anim_speed=0.000`，说明 JSON 缺少 gait 字段或 total_length，回到步骤 2 重新导出。

### 输出

视频为五视角 + BEV 小地图的 3×2 网格布局：
```
┌──────────┬──────────┬──────────┐
│ cam 0    │ cam 1    │ cam 2    │  (前 / 左前 / 右前)
├──────────┼──────────┼──────────┤
│ cam 3    │ cam 4    │  BEV     │  (左 / 右 / 鸟瞰小地图)
└──────────┴──────────┴──────────┘
```

---

## 完整示例：从零渲染 scene552

```bash
# 0. 环境
cd ~/4drivestudio && conda activate drivestudio

# 1. 3D 预览器画轨迹
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene552/checkpoint_final.pth \
    --scene_dir data/waymo/processed/training/552 \
    --port 8080
# → 浏览器画轨迹 → 点 export → traj_live.json 生成

# 2. 渲染
python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene552/checkpoint_final.pth \
    --path_json outputs/waymo_omnire/scene552/trajectories/traj_live.json \
    --out outputs/waymo_omnire/scene552/videos_eval/scene552_v3.mp4
```

---

## 参数参考

### render_runner_video.py（常用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resume_from` | **必填** | checkpoint 路径 |
| `--path_json` | **必填** | 轨迹 JSON（含 gait 参数） |
| `--out` | `outputs/runner_composite.mp4` | 输出视频路径 |
| `--mode` | `multicam_grid` | 渲染模式（`video`=单视角，`multicam_grid`=五视角+BEV） |
| `--scale` | `0.90` | 角色缩放 |
| `--adaptive_ground_z` | `True` | 自适应地面高度（用 `--no-adaptive_ground_z` 关闭） |
| `--fps` | `10` | 输出视频帧率 |
| `--seq` | `outputs/assets/runner_seq.npz` | 角色动画文件 |
| `--cameras` | 全部 | 指定相机，如 `--cameras 0,1` |

### render_runner_video.py（步频控制，通常不用手动指定）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--anim_speed` | `1` | 动画帧/视频帧（被 JSON gait 自动覆盖） |
| `--stride` | `0` | [旧] 每周期步幅米数，优先级最高 |
| `--cycle_stride` | `0` | 每周期步幅，优先级低于 JSON gait |
| `--frame_step` | `1` | 每 N 帧渲染 1 帧（1=实时） |
| `--max_output_frames` | `0` | 限制输出帧数（0=全部，**不要设小否则角色跑不完**） |

**步频优先级**：`--stride` > JSON `gait` > `--cycle_stride` > `--anim_speed`

> 正常使用只需传 `--path_json`，步频自动从 JSON 读取。无需手动指定 `--stride` 或 `--anim_speed`。

### trajectory_previewer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resume_from` | **必填** | checkpoint 路径 |
| `--scene_dir` | **必填** | Waymo 数据目录（如 `data/waymo/processed/training/023`） |
| `--path_json` | `None` | 加载已有轨迹编辑 |
| `--port` | `8080` | 网页端口 |
| `--render_fps` | `10.0` | 视频帧率（用于步频计算） |
| `--cycle_stride` | `2.6` | 步态周期步幅（米） |
| `--char_size` | `0.6,0.4,1.8` | 角色碰撞盒 W,D,H（米） |
| `--max_obstacles` | `80` | 显示的动态障碍物数量上限 |

### bev_trajectory_planner.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scene_dir` | **必填** | Waymo 数据目录 |
| `--out_bev` | 自动 | BEV 图输出路径 |
| `--out_traj` | 自动 | 轨迹 JSON 输出路径 |
| `--n_samples` | `300` | 平滑后轨迹点数 |
| `--margin` | `0.5` | 碰撞检测余量（米） |

---

## 场景编号对照

DriveStudio 输出目录和 Waymo 数据目录的编号可能不同位数：

| 场景 | checkpoint 路径 | 数据目录 |
|------|----------------|----------|
| scene23 | `outputs/waymo_omnire/scene23/` | `data/waymo/processed/training/023/` |
| scene114 | `outputs/waymo_omnire/scene114/` | `data/waymo/processed/training/114/` |
| scene552 | `outputs/waymo_omnire/scene552/` | `data/waymo/processed/training/552/` |

> 预览器的 `--scene_dir` 必须指向数据目录（含 `ego_pose/`, `instances/`, `lidar/`）。

---

## 常见问题

### Q: 角色不动 / 静止
**A**: JSON 的 gait 字段未被正确读取。检查 `traj_live.json` 是否含 `gait` 字段和顶层 `total_length`。重新在预览器中 export。

### Q: 角色没跑完轨迹就结束了
**A**: 去掉 `--max_output_frames`（或设为 0）。步频匹配已确保角色在全部帧内走完。

### Q: 角色滑步（脚步和位移不匹配）
**A**: 确认没有手动传 `--anim_speed` 或 `--stride`，让 JSON 的 gait 参数自动生效。

### Q: 角色穿地 / 悬空
**A**: 确认 `--adaptive_ground_z` 开启（默认已开启）。如仍有问题，试 `--feet_offset 0.05`。

### Q: 角色太大/太小
**A**: 调整 `--scale`（默认 0.90）。0.8=较小，1.0=原始大小。

### Q: scene_dir 路径报错
**A**: scene23 的数据目录是 `023`（补零到 3 位），scene552 是 `552`。

### Q: 渲染时报 `FileNotFoundError: runner_seq.npz`
**A**: 确认 `outputs/assets/runner_seq.npz` 存在。
```bash
ls outputs/assets/runner_seq.npz
```

---

## 文件结构

```
tools/
├── trajectory_previewer.py    # 3D 实时预览器（viser + nerfview）
├── render_runner_video.py     # 最终视频渲染（nvdiffrast + gsplat）
├── bev_trajectory_planner.py  # BEV 轨迹规划（matplotlib）
├── gait_utils.py              # 步频计算模块
└── bake_runner_frames.py      # Blender 烘焙角色动画（一次性）

outputs/assets/
└── runner_seq.npz             # 角色动画（40帧跑步循环 + 纹理）

outputs/waymo_omnire/scene<N>/
├── checkpoint_final.pth       # 3DGS 训练结果
├── trajectories/
│   ├── traj.json              # BEV 规划器输出
│   └── traj_live.json         # 3D 预览器输出（含 gait 参数）
├── bev/
│   └── bev.png                # BEV 鸟瞰图
└── videos_eval/
    └── scene<N>_v3.mp4        # 最终渲染视频
```
