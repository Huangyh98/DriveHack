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

## 进阶：动画模式（walk / jog / run / stand）

预览器右侧"步频参数"面板的**动画模式**下拉框可切换角色运动方式：

| 模式 | 周期步幅 | 每步 | 说明 |
|------|---------|------|------|
| `run`（默认） | 2.6m | 1.3m | 快跑 |
| `jog` | 2.0m | 1.0m | 慢跑 |
| `walk` | 1.2m | 0.6m | 步行（步频更快，步幅更小） |
| `stand` | — | — | 原地站立（冻结在单帧） |

切换模式会自动设定推荐步幅，并实时更新步频评估。模式会写入导出 JSON 的 `gait.anim_mode` 字段，渲染器自动读取——无需手动传参。

> **注意**：所有模式复用同一个 `runner_seq.npz` 跑步动画；模式仅改变播放速度（步频/步幅），而非动画本身。`stand` 模式冻结在中性帧（非第 0 帧的触地瞬间）。

```bash
# 在预览器里选 walk，导出后渲染即为步行效果
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --scene_dir data/waymo_processed/training/023 --anim_mode walk
```

---

## 进阶：多角色注入

可同时在场景里注入多个角色（各自独立轨迹、步频、缩放、纹理）。

**方式 1：预览器里画**

```bash
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --scene_dir data/waymo/processed/training/023 \
    --extra_traj outputs/waymo_omnire/scene23/trajectories/traj_jaywalk.json
```

主轨迹（绿色，可点击编辑）+ 额外轨迹（蓝/橙色，只读）。画好后点 `export multi_config.json` 生成多角色配置。

**方式 2：手写配置**

```json
{
  "characters": [
    {"path_json": ".../traj_a.json", "scale": 0.90, "anim_mode": "walk", "offset_t": 0.0},
    {"path_json": ".../traj_b.json", "scale": 1.1, "anim_mode": "run", "offset_t": 0.15}
  ]
}
```

`offset_t` 错开角色出发时机（0.15 = 第二个角色延迟 15% 进度出发）。见 `configs/trajectories/multi_two_pedestrians_example.json`。

**渲染：**

```bash
python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --multi_traj outputs/waymo_omnire/scene23/trajectories/multi_config.json \
    --out outputs/waymo_omnire/scene23/videos_eval/scene23_multi.mp4
```

每个角色独立计算地面高度和步频，背景只渲染一次后多角色依次叠加。

---

## 进阶：批量渲染

`tools/batch_render_scenes.py` 对多个场景批量渲染：

```bash
# 3 个场景，用轨迹库里的 jaywalk，干跑预览命令
python tools/batch_render_scenes.py --scenes 23,114,552 --traj jaywalk --dry_run

# 实际渲染（含中断恢复）
python tools/batch_render_scenes.py --scenes 23,114,552 --traj jaywalk --resume
```

| 参数 | 说明 |
|------|------|
| `--scenes` | 逗号分隔场景号 |
| `--scenes_file` | 每行一个场景号的文件 |
| `--traj` | 轨迹库名（`configs/trajectories/`）或路径 |
| `--traj_path` | 直接指定轨迹 JSON（覆盖 `--traj`） |
| `--mode` | 渲染模式（默认 `multicam_grid`） |
| `--resume` | 中断恢复 |
| `--dry_run` | 只打印命令不执行 |

脚本自动处理 scene23↔023 的零填充问题，缺 checkpoint/数据目录的场景会被跳过并告警。

---

## 进阶：渲染中断恢复（--resume）

长视频渲染中途崩溃时，加 `--resume` 可从断点续渲：

```bash
python tools/render_runner_video.py \
    --resume_from .../checkpoint_final.pth \
    --path_json .../traj_live.json \
    --out .../scene23.mp4 --resume
```

- 启用后每帧落盘为 `<out>_frames/frameNNNNNN.png`
- 重跑时自动跳过已存在帧
- 全部完成后自动 mux 成目标 mp4
- 默认关闭（不加 `--resume` 时直接写 mp4，行为不变）

---

## 进阶：轨迹库

`configs/trajectories/` 存放可复用的轨迹 JSON（含 `gait` 参数）。轨迹库里的轨迹是**模式**（相对坐标），不是绝对路径——加载到预览器里按自己场景调整路点后再导出。

| 文件 | 模式 | 长度 |
|------|------|------|
| `jaywalk_cross_scene23.json` | 横穿马路 | 8.0m |
| `multi_two_pedestrians_example.json` | 双角色配置示例 | — |

详见 `configs/trajectories/README.md`。

---

## 常见问题

### Q: 角色不动 / 静止
**A**: JSON 的 gait 字段未被正确读取。检查 `traj_live.json` 是否含 `gait` 字段和顶层 `total_length`。重新在预览器中 export。若 `anim_mode=stand` 则是有意冻结。

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
