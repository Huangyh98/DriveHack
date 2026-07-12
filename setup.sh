#!/bin/bash
# setup.sh — One-click installer: integrate DriveHack into DriveStudio
#
# Usage:
#   cd /path/to/drivestudio
#   git clone https://github.com/<user>/DriveHack.git
#   cd DriveHack && bash setup.sh
#
# What it does:
#   1. Copies scripts to DriveStudio's tools/ directory
#   2. Copies config to configs/
#   3. Downloads character animation assets

set -e

# 找到 DriveStudio 根目录（本仓库的上级目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVESTUDIO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo " DriveHack Installer"
echo "=========================================="
echo "DriveStudio root: $DRIVESTUDIO_ROOT"
echo ""

# 1. 检查 DriveStudio 目录
if [ ! -f "$DRIVESTUDIO_ROOT/tools/train.py" ]; then
    echo "❌ 错误：未找到 DriveStudio（$DRIVESTUDIO_ROOT/tools/train.py 不存在）"
    echo "   请确保本仓库克隆在 DriveStudio 根目录下"
    exit 1
fi
echo "✓ 检测到 DriveStudio"

# 2. 复制脚本
echo ""
echo "[1/3] 复制自研脚本到 tools/..."
for f in tools/*.py; do
    fname=$(basename "$f")
    # 跳过 DriveStudio 已有的 train.py, eval.py, __init__.py
    if [ "$fname" = "train.py" ] || [ "$fname" = "eval.py" ] || [ "$fname" = "__init__.py" ]; then
        echo "  跳过 $fname（DriveStudio 已有）"
        continue
    fi
    cp -v "$f" "$DRIVESTUDIO_ROOT/tools/$fname"
done

# 3. 复制配置
echo ""
echo "[2/3] 复制配置文件到 configs/..."
mkdir -p "$DRIVESTUDIO_ROOT/configs"
for f in configs/*.yaml; do
    cp -v "$f" "$DRIVESTUDIO_ROOT/configs/"
done
# 复制轨迹库（configs/trajectories/*.json + README）
if [ -d "configs/trajectories" ]; then
    mkdir -p "$DRIVESTUDIO_ROOT/configs/trajectories"
    cp -v configs/trajectories/* "$DRIVESTUDIO_ROOT/configs/trajectories/" 2>/dev/null || true
fi

# 4. 下载角色资产
echo ""
echo "[3/3] 下载角色动画资产..."
mkdir -p "$DRIVESTUDIO_ROOT/outputs/assets"
ASSET_PATH="$DRIVESTUDIO_ROOT/outputs/assets/runner_seq.npz"

if [ -f "$ASSET_PATH" ]; then
    echo "  runner_seq.npz 已存在，跳过下载"
else
    if [ -f "assets/download_assets.sh" ]; then
        bash assets/download_assets.sh "$ASSET_PATH"
    else
        echo "  ⚠ 未找到 assets/download_assets.sh"
        echo "  请手动下载 runner_seq.npz 到 $ASSET_PATH"
        echo "  下载链接见 README.md"
    fi
fi

echo ""
echo "=========================================="
echo " ✅ 安装完成！"
echo "=========================================="
echo ""
echo "使用方法："
echo "  cd $DRIVESTUDIO_ROOT"
echo "  conda activate drivestudio"
echo ""
echo "  # 3D 预览器画轨迹"
echo "  python tools/trajectory_previewer.py \\"
echo "      --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \\"
echo "      --scene_dir data/waymo/processed/training/023 --port 8080"
echo ""
echo "  # 渲染视频"
echo "  python tools/render_runner_video.py \\"
echo "      --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \\"
echo "      --path_json outputs/waymo_omnire/scene23/trajectories/traj_live.json \\"
echo "      --out outputs/waymo_omnire/scene23/videos_eval/scene23_v3.mp4"
echo ""
echo "详细文档: docs/trajectory_pipeline.md"
