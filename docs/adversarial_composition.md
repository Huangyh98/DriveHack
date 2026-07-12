# 动态人物插入 3DGS 场景 — 技术文档

将带骨骼动画的人物模型插入 Waymo 3DGS（4D 高斯泼溅）重建的驾驶场景中，生成合成视频 / 多视角图像，用于对抗样本训练。

---

## 目录

- [整体流程](#整体流程)
- [第一步：资产准备](#第一步资产准备)
- [第二步：训练 3DGS 场景](#第二步训练-3dgs-场景)
- [第三步：BEV 轨迹规划](#第三步bev-轨迹规划)
- [第四步：生成合成视频](#第四步生成合成视频)
- [3D 标注框与 BEV 小地图](#3d-标注框与-bev-小地图)
- [步幅匹配](#步幅匹配)
- [自适应地面高度](#自适应地面高度)
- [衣服纹理替换（对抗核心）](#衣服纹理替换对抗核心)
- [速度调节](#速度调节)
- [输出目录结构](#输出目录结构)
- [坐标系约定](#坐标系约定)
- [故障排查](#故障排查)

---

## 整体流程

```
.blend 人物 ──bake_runner_frames──▶ runner_seq.npz (逐帧网格)
                                          │
Waymo 数据 ──train.py──▶ 3DGS checkpoint  │
                             │             ├──(可选) BEV选点 ──▶ traj.json
                             ▼             │                      │
                    render_runner_video ◀──┘                      │
                             │                                      │
                    gsplat渲染场景RGB+深度                           │
                    nvdiffrast光栅化人物 ◀──────────────────────────┘
                    深度遮挡 + 合成
                             │
                             ▼
                    合成视频 / 多视角PNG
```

**四个工具，四步走：**

| 步骤 | 工具 | 作用 |
|---|---|---|
| 1. 烘焙人物 | `tools/bake_runner_frames.py` | .blend → 逐帧网格 npz（一次性） |
| 2. 训练场景 | `tools/train.py` | Waymo 数据 → 3DGS checkpoint |
| 3. 规划轨迹 | `tools/bev_trajectory_planner.py` | BEV 俯视图选点 → 轨迹 JSON |
| 4. 生成视频 | `tools/render_runner_video.py` | 合成人物 + 场景 → mp4 / PNG |

> 人物是三角网格，3DGS 是高斯点云，两者在同一 3D 空间、同一套 Waymo 相机参数下分别渲染，再按深度做遮挡合成。人物不会被"贴"上去——它在场景坐标系里有真实的 3D 位置，多视角天然一致。

---

## 第一步：资产准备

### 安装 Blender 4.4+

项目的 `.blend` 文件由 Blender 4.4+ 保存，系统自带的 3.0.1 无法打开。

```bash
cd /tmp
wget https://download.blender.org/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz
mkdir -p ~/blender && tar -xf blender-4.4.3-linux-x64.tar.xz -C ~/blender
```

### 烘焙人物网格序列

把 .blend 里的 Mixamo 跑步动画逐帧烘焙成三角网格（顶点 + 面 + UV + 纹理）。**只需做一次。**

```bash
~/blender/blender-4.4.3-linux-x64/blender --background \
    --python tools/bake_runner_frames.py -- \
    --blend man/AdvSerial_v2_runing_rd.blend \
    --out outputs/assets/runner_seq.npz \
    --frames 40
```

输出 `outputs/assets/runner_seq.npz`，包含：
- `man/verts` `(40, V, 3)` — 身体逐帧坐标（**Z-up**，面朝 -Y）
- `clothes_1/*`、`pants_1/*` — 衣服、裤子（结构相同）
- 每个网格：`faces` `(F,3)`、`uvs` `(F,3,2)`、`tex` `(H,W,4)` RGBA 纹理

---

## 第二步：训练 3DGS 场景

```bash
PYTHONPATH=$(pwd) python tools/train.py \
    --config_file outputs/waymo_omnire/scene552/config.yaml \
    --output_root outputs/waymo_omnire/scene552 \
    logging.saveckpt_freq=10000
```

> **关键**：必须指定 `--output_root`，否则 checkpoint 存到 `work_dirs/`。

### 防 OOM（RTX 4080 16GB）

```bash
    trainer.gaussian_ctrl_general_cfg.densify_grad_thresh=0.0008 \
    trainer.gaussian_ctrl_general_cfg.cull_alpha_thresh=0.01
```

### 场景选择

| 场景 | 行驶距离 | 特点 |
|---|---|---|
| **scene552** | 29m | 最干净的直行，已验证 |
| scene788 | 87m | 最长直行 |
| scene023 | 3.8m | 静止路口 |

---

## 第三步：BEV 轨迹规划

`tools/bev_trajectory_planner.py` — 从 Waymo 数据生成俯视图，交互选点，碰撞检测，输出平滑轨迹。**不依赖 GPU。**

```bash
python tools/bev_trajectory_planner.py \
    --scene_dir data/waymo/processed/training/552
```

输出自动归到对应场景目录：
- BEV 图 → `outputs/waymo_omnire/scene552/bev/bev.png`
- 轨迹 JSON → `outputs/waymo_omnire/scene552/trajectories/traj.json`

交互窗口操作：**左键**添加路点（落在障碍物内被拒绝）/ **右键**撤销 / **关窗**保存

| 参数 | 默认 | 说明 |
|---|---|---|
| `--scene_dir` | 必填 | processed waymo 场景目录 |
| `--out_bev` | 自动 | BEV PNG 输出路径 |
| `--out_traj` | 自动 | 轨迹 JSON 输出路径 |
| `--no_pick` | False | 只生成 BEV 图不弹窗口 |
| `--n_samples` | 300 | 平滑后轨迹点数 |
| `--margin` | 0.5 | 障碍物碰撞余量（米） |
| `--lidar_frames` | 50 | 累积 LiDAR 帧数 |

> 需要图形界面（TkAgg），在本地终端运行。

---

## 第四步：生成合成视频

`tools/render_runner_video.py` — 主力渲染工具。

### 渲染流程（每帧）

1. **gsplat 渲染 3DGS 场景** → 场景 RGB + 深度 + Background-only 深度
2. **计算人物位姿** → 沿轨迹取位置，朝向 = 行进方向
3. **nvdiffrast 光栅化人物** → 人物 RGBA + 人物深度
4. **深度遮挡** → 人物深度 < 场景深度的像素才画人物
5. **alpha 合成 + 接触阴影**

### 命令示例

**多相机网格视频（推荐，5 视角 + BEV 小地图）：**
```bash
PYTHONPATH=$(pwd) python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene552/checkpoint_final.pth \
    --seq outputs/assets/runner_seq.npz \
    --mode multicam_grid \
    --cameras 0,1,2,3,4 \
    --path_json outputs/waymo_omnire/scene552/trajectories/traj.json \
    --adaptive_ground_z \
    --stride 2.6 --scale 0.90 \
    --max_output_frames 198 --fps 10 \
    --out outputs/waymo_omnire/scene552/videos_eval/scene552.mp4
```

**单相机视频：**
```bash
PYTHONPATH=$(pwd) python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene552/checkpoint_final.pth \
    --seq outputs/assets/runner_seq.npz \
    --mode video \
    --path "12,4;25,7;38,1" \
    --adaptive_ground_z \
    --stride 2.6 --scale 0.90 \
    --max_output_frames 198 --fps 10 \
    --out outputs/waymo_omnire/scene552/videos_eval/scene552_front.mp4
```

**批量多视角 PNG（对抗数据生产）：**
```bash
PYTHONPATH=$(pwd) python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene552/checkpoint_final.pth \
    --seq outputs/assets/runner_seq.npz \
    --mode multiview \
    --cameras 0,1,2,3,4 \
    --positions "12,0,0;15,2,0" \
    --clothes_textures "tex0.png;tex1.png;" \
    --frames 0,50,100,150 \
    --out_dir outputs/waymo_omnire/scene552/adversarial/
```

### 参数详解

| 参数 | 默认 | 说明 |
|---|---|---|
| `--resume_from` | 必填 | 3DGS checkpoint 路径 |
| `--seq` | `outputs/assets/runner_seq.npz` | 烘焙的人物网格序列 |
| `--mode` | `video` | `video`（单相机）/ `multicam_grid`（5相机网格）/ `multiview`（批量PNG） |
| `--out` | `runner_composite.mp4` | 视频输出路径 |
| `--cameras` | 全部 | 相机 ID：`0,1,2,3,4`（0=front 1=front_left 2=front_right 3=left 4=right） |
| `--path_json` | 空 | BEV 工具生成的轨迹 JSON（优先级高于 `--path`） |
| `--path` | 默认轨迹 | 手写 polyline 路点 `x1,y1;x2,y2;...` |
| `--fps` | 10 | 输出帧率（Waymo 原始约 10fps） |
| `--frame_step` | 1 | 每隔 N 个场景帧渲染一次；1=真实速度 |
| `--max_output_frames` | 0 | 最多输出帧数；0=全部 |
| `--stride` | 0 | 步幅匹配：每跑步周期(两步)移动的米数（推荐 2.6；0=禁用用固定 anim_speed） |
| `--anim_speed` | 1 | 跑步动画频率（仅 `--stride 0` 时生效） |
| `--ground_z` | 0 | 地面 Z 高度（`--adaptive_ground_z` 关闭时用） |
| `--adaptive_ground_z` | False | 自适应地面高度：用静态背景深度沿轨迹预采样，**推荐开启** |
| `--feet_offset` | 0.01 | 人物上抬量（米），防脚底穿模 |
| `--scale` | 1.0 | 人物缩放（0.90 ≈ 1.64m） |
| `--positions` | 默认 | multiview 模式放置位置 `x,y,z;x,y,z` |
| `--clothes_textures` | 原纹理 | 衣服纹理 PNG 路径，`;` 分隔 |
| `--bg_only` | False | 只渲染静态背景（移除动态车辆/行人）。用 `Background_depth` 做遮挡，避免角色被旧动态物轮廓误挡 |
| `--bev_to_black` | False | 把 multicam_grid 中心格的 BEV 小地图替换为黑色 `CAM_BACK` 占位（凑 6 视角） |
| `--multi_traj` | 空 | 多角色配置 JSON：`[{path_json,scale,yaw,offset_t,clothes_textures,anim_mode}]`。覆盖单角色 `--path`/`--path_json` |
| `--resume` | False | 中断恢复：每帧落盘 PNG 到 `<out>_frames/`，重跑跳过已存在帧，完成后 mux 成 mp4 |
| `--frames` | 自动 | 指定场景帧 ID |

### 深度遮挡

每个像素比较人物深度与场景深度：人物更近 → 画人物；否则 → 保留场景（人物被前景挡住）。

---

## 3D 标注框与 BEV 小地图

`multicam_grid` 模式网格中心放**实时 BEV 小地图**：

```
┌──────────────┬──────────────┬──────────────┐
│  front_left  │    front     │ front_right  │
├──────────────┼──────────────┼──────────────┤
│    left      │   BEV地图    │    right     │
└──────────────┴──────────────┴──────────────┘
```

BEV 小地图内容：灰色点云（LiDAR路面）、红/黄/橙框（车/行人/骑车人）、青色线（ego轨迹）、绿色线（人物轨迹）、绿色十字（人物当前位置）。

不同相机分辨率（1280×1920 vs 866×1920）自动统一缩放到 480×640。

---

## 步幅匹配

```bash
--stride 2.6   # 每个跑步周期(两步)移动2.6m，每步1.3m
```

脚本根据轨迹总长度和帧数自动计算 `anim_speed`，使步幅恒定。轨迹长→步频快，短→慢。设 `--stride 0` 则用固定 `--anim_speed`。

---

## 自适应地面高度

### 工作原理

`--adaptive_ground_z` 开启后，脚本在**渲染前**沿轨迹每隔 5 帧采样一次：

1. 渲染**只含静态背景（Background）的深度图**——不含动态车辆/行人，避免读到车顶/人头导致人物飞到天上
2. 在人物 (x,y) 位置从 Z=0 向上扫描，投影到背景深度图
3. 第一个深度匹配的位置 = 真实路面 Z
4. 对所有采样点**线性插值 + 平滑**，生成逐帧 Z 查找表
5. 渲染时直接查表——**稳定不抖，跟随地形，不被动态物体干扰**

```bash
# 推荐：自适应（沿轨迹预采样静态背景深度）
--adaptive_ground_z

# 固定高度（已知平地 Z）
--ground_z 0.5
```

### 调优

| 问题 | 解决 |
|---|---|
| 脚穿模 | 调大 `--feet_offset`（如 0.05） |
| 人物悬空 | 减小 `--feet_offset`，或检查 `--ground_z` |
| 遇到行人飞上天 | 已修复（用 Background-only 深度，不含动态物体） |

---

## 衣服纹理替换（对抗核心）

运行时动态加载本地 PNG 替换衣服纹理，**不需重新烘焙**。身体/裤子/头部不变。

```bash
--clothes_textures "tex0.png;tex1.png;tex2.png;tex3.png"  # 4张纹理
```

---

## 速度调节

| 参数 | 控制什么 | 调大 | 调小 |
|---|---|---|---|
| `--stride` | 步幅（每周期移动距离） | 步幅大 | 步幅小（步频快） |
| `--anim_speed` | 动画频率（仅 stride=0 时） | 步频快 | 步频慢 |
| `--max_output_frames` | 视频总帧数 | 停留更久 | 很快跑完 |
| `--frame_step` | 场景播放速度 | 场景快放 | 场景慢放 |

```bash
# 推荐：步幅匹配
--stride 2.6 --max_output_frames 198

# 慢跑
--stride 0 --anim_speed 0.6 --max_output_frames 200
```

---

## 输出目录结构

```
outputs/waymo_omnire/scene552/
├── checkpoint_final.pth        # 3DGS 权重
├── config.yaml                 # 训练配置
├── bev/
│   └── bev.png                 # BEV 俯视图
├── trajectories/
│   └── traj.json               # 平滑轨迹
├── videos_eval/                # 合成视频
├── adversarial/                # 对抗样本PNG（multiview）
└── composite_test/             # 调试图
```

---

## 坐标系约定

| 空间 | 约定 |
|---|---|
| **Waymo 世界** | X 前 / Y 左 / Z 上，米，原点在首帧 ego |
| **相机（OpenCV）** | X 右 / Y 下 / Z 前 |
| **人物网格** | Z 上（头 +Z，脚 Z≈0），面朝 -Y |

---

## 故障排查

### checkpoint 找不到
训练没指定 `--output_root`。手动复制：
```bash
cp work_dirs/drivestudio/omnire/checkpoint_final.pth outputs/waymo_omnire/scene552/
cp work_dirs/drivestudio/omnire/config.yaml outputs/waymo_omnire/scene552/
```

### 训练 CUDA OOM
```bash
trainer.gaussian_ctrl_general_cfg.densify_grad_thresh=0.0008
trainer.gaussian_ctrl_general_cfg.cull_alpha_thresh=0.01
```

### 脚底穿模
`--adaptive_ground_z`（用静态背景深度预采样路面）。仍穿模则调大 `--feet_offset`。

### 人物飞到天上（遇到行人/车辆）
已修复。`--adaptive_ground_z` 现在用 Background-only 深度（不含动态车辆/行人）。

### multicam_grid 报 shape 不匹配
相机分辨率不同已自动缩放。确认 grid 拼接用 `cell_h/cell_w`。

### BrokenPipeError
终端粘贴命令多条粘在一起。确保每条命令单独一行。

### BEV 选点窗口不弹出
需图形界面（TkAgg），在本地终端运行。
