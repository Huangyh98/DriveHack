#!/bin/bash
# download_assets.sh — Download character animation asset runner_seq.npz
#
# Usage: bash assets/download_assets.sh [output_path]
#
# runner_seq.npz contains a 40-frame run-loop animation (man/clothes/pants
# meshes + vertices, faces, UVs, textures), baked from a Mixamo character
# via Blender. See docs/baking_guide.md to bake your own.

set -e

OUTPUT="${1:-outputs/assets/runner_seq.npz}"
DIR="$(dirname "$OUTPUT")"
mkdir -p "$DIR"

# ====== Download URL ======
# GitHub Release v1.0
URL="https://github.com/Huangyh98/DriveHack/releases/download/v1.0/runner_seq.npz"

if [ -f "$OUTPUT" ]; then
    echo "runner_seq.npz already exists, skipping: $OUTPUT"
    exit 0
fi

echo "Downloading runner_seq.npz ..."
echo "  URL: $URL"
echo "  Output: $OUTPUT"

# Try wget first (follows redirects)
if wget -q --show-progress -O "$OUTPUT" "$URL"; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo "✓ Downloaded: $OUTPUT ($SIZE)"
else
    rm -f "$OUTPUT"
    echo "wget failed, trying curl..."
    if curl -fL -o "$OUTPUT" "$URL"; then
        SIZE=$(du -h "$OUTPUT" | cut -f1)
        echo "✓ Downloaded: $OUTPUT ($SIZE)"
    else
        rm -f "$OUTPUT"
        echo "❌ Download failed!"
        echo ""
        echo "Manual options:"
        echo "  1. Browser: $URL"
        echo "  2. Save to: $OUTPUT"
        echo ""
        echo "Or bake your own (requires Blender):"
        echo "  blender --background --python tools/bake_runner_frames.py --"
        echo "      --blend your_character.blend --out $OUTPUT --frames 40"
        echo "  See docs/baking_guide.md"
        exit 1
    fi
fi
