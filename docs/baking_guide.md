# 角色动画烘焙指南

本文档说明如何使用 Blender 将 Mixamo 角色动画烘焙为 `runner_seq.npz` 格式。

## 前提条件

### 1. 安装 Blender 4.x

```bash
# 下载 Blender 4.4.3（或更高版本）
wget https://download.blender.org/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz
tar xf blender-4.4.3-linux-x64.tar.xz -C ~/
# Blender 安装在 ~/blender/blender-4.4.3-linux-x64/blender
```

### 2. 准备角色文件

从 [Mixamo](https://www.mixamo.com/) 下载角色 + 跑步动画：
1. 选一个角色，下载 `.fbx`（含骨架）
2. 选一个 "Run" 动画，下载 `.fbx`（含动画）
3. 在 Blender 中合并角色和动画，导出为 `.blend` 文件

`.blend` 文件需满足以下结构：
- 一个名为 `Armature` 的骨架对象
- 三个网格对象：`man`（身体）、`clothes_1`（上衣）、`pants_1`（裤子）
- 每个网格有 UV 贴图和带纹理的材质
- `Armature` 上有一个跑步循环动画（in-place，不含位移）

## 烘焙命令

```bash
cd /path/to/drivestudio

~/blender/blender-4.4.3-linux-x64/blender --background \
    --python tools/bake_runner_frames.py -- \
    --blend man/AdvSerial_v2_runing_rd.blend \
    --out outputs/assets/runner_seq.npz \
    --frames 40
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--blend` | `.blend` 文件路径 |
| `--out` | 输出 `.npz` 路径 |
| `--frames` | 烘焙帧数（默认 40，即 2 个完整步态周期，每周期 20 帧） |

> **为什么是 40 帧？** 跑步动画的一个步态周期（左脚+右脚各迈一次）通常为 20 帧。烘焙 40 帧可确保循环平滑。`gait_utils.py` 中 `CYCLE_FRAMES = 20` 对应此设置。

## 输出格式

`runner_seq.npz` 包含：

| Key | 形状 | 说明 |
|-----|------|------|
| `n_frames` | `()` | 帧数（40） |
| `man/verts` | `(40, V, 3)` | 身体每帧顶点坐标 |
| `man/faces` | `(F, 3)` | 三角面索引 |
| `man/uvs` | `(F, 3, 2)` | 每面顶点的 UV 坐标 |
| `man/tex` | `(H, W, 4)` | RGBA 纹理图像 |
| `clothes_1/*` | 同上 | 上衣 |
| `pants_1/*` | 同上 | 裤子 |

### 坐标系

烘焙输出的顶点为 **Z-up**（脚部 Z≈0，头部 Z≈1.82），与 DriveStudio 场景坐标系一致。

## 验证

烘焙完成后，检查输出：

```bash
python -c "
import numpy as np
d = np.load('outputs/assets/runner_seq.npz', allow_pickle=True)
print(f'frames: {d[\"n_frames\"]}')
v = d['man/verts']
print(f'man verts: {v.shape}')
print(f'frame 0 bbox: X[{v[0,:,0].min():.2f},{v[0,:,0].max():.2f}] '
      f'Y[{v[0,:,1].min():.2f},{v[0,:,1].max():.2f}] '
      f'Z[{v[0,:,2].min():.2f},{v[0,:,2].max():.2f}]')
"
```

预期输出：
```
frames: 40
man verts: (40, 122696, 3)
frame 0 bbox: X[-0.24,0.36] Y[-0.71,0.46] Z[-0.01,1.82]
```

## 常见问题

### Q: Blender 报 "Cannot open .blend file"
**A**: 检查 `.blend` 文件路径是否正确。Blender 4.x 无法打开 Blender 3.x 保存的部分文件（exit code 139），请用同版本或更高版本 Blender。

### Q: 报 "KeyError: 'Armature'" 或找不到网格
**A**: `.blend` 文件中的对象名称必须是 `Armature`、`man`、`clothes_1`、`pants_1`。在 Blender 中重命名对象。

### Q: 纹理缺失（全是灰色）
**A**: 检查 Blender 中每个网格的材质是否有 `TEX_IMAGE` 节点指向纹理图像。烘焙脚本只提取第一个找到的 `TEX_IMAGE` 节点。

### Q: 想用不同的角色
**A**: 只需准备新的 `.blend` 文件（保持相同的对象命名结构），重新运行烘焙命令即可。`runner_seq.npz` 会被覆盖。
