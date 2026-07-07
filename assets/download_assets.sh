#!/bin/bash
# download_assets.sh — 下载角色动画资产 runner_seq.npz
#
# 用法：bash assets/download_assets.sh [输出路径]
#
# runner_seq.npz 包含 40 帧跑步循环动画（含 man/clothes/pants 三个子网格的
# 顶点、面、UV、纹理），由 Blender 从 Mixamo 角色烘焙生成。
#
# 如果你想用自己的角色，运行 tools/bake_runner_frames.py 自行烘焙。

set -e

OUTPUT="${1:-outputs/assets/runner_seq.npz}"
DIR="$(dirname "$OUTPUT")"
mkdir -p "$DIR"

echo "下载 runner_seq.npz 到 $OUTPUT ..."

# ====== 下载链接配置 ======
# 方式 1：Google Drive（推荐）
GDRIVE_ID="YOUR_FILE_ID_HERE"

# 方式 2：直接 URL（如有自己的服务器）
# DIRECT_URL="https://your-server.com/runner_seq.npz"

# 尝试方式 2（直接下载）— 如果配置了 DIRECT_URL
if [ -n "${DIRECT_URL:-}" ]; then
    echo "从 $DIRECT_URL 下载..."
    if wget -q --show-progress -O "$OUTPUT" "$DIRECT_URL"; then
        echo "✓ 下载完成: $OUTPUT"
        exit 0
    fi
fi

# 尝试方式 1（Google Drive）
if [ "$GDRIVE_ID" != "YOUR_FILE_ID_HERE" ]; then
    echo "从 Google Drive (ID: $GDRIVE_ID) 下载..."
    if command -v gdown &> /dev/null; then
        gdown "$GDRIVE_ID" -O "$OUTPUT"
        echo "✓ 下载完成: $OUTPUT"
        exit 0
    else
        echo "gdown 未安装，尝试安装..."
        pip install gdown
        gdown "$GDRIVE_ID" -O "$OUTPUT"
        echo "✓ 下载完成: $OUTPUT"
        exit 0
    fi
fi

# 都没配置，提示手动操作
echo "⚠ 自动下载未配置。请选择以下方式之一："
echo ""
echo "方式 A：手动下载"
echo "  1. 从你的云盘下载 runner_seq.npz"
echo "  2. 放到 $OUTPUT"
echo ""
echo "方式 B：自行烘焙（需要 Blender）"
echo "  ~/blender/blender-4.4.3-linux-x64/blender --background \\"
echo "      --python tools/bake_runner_frames.py -- \\"
echo "      --blend man/AdvSerial_v2_runing_rd.blend \\"
echo "      --out $OUTPUT --frames 40"
echo ""
echo "方式 C：配置本脚本"
echo "  编辑 assets/download_assets.sh，设置 GDRIVE_ID 或 DIRECT_URL"
exit 1
