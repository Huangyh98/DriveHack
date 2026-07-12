"""Batch-render adversarial pedestrian videos across multiple Waymo scenes.

Where ``run_remaining_scenes.sh`` batch-*trains* scenes, this script
batch-*renders* them: given a list of scene indices and a trajectory library
selection, it invokes ``render_runner_video.py`` once per scene, routing each
scene's checkpoint and data dir correctly (handling the scene23↔023
zero-padding gotcha via ``scene_utils``).

Supports ``--dry_run`` to only print the commands without executing — useful
for sanity-checking a batch before committing GPU time.

Usage:
    python tools/batch_render_scenes.py --scenes 23,114,552 \\
        --traj jaywalk --mode multicam_grid --dry_run

    # render all scenes listed in a file (one scene id per line)
    python tools/batch_render_scenes.py --scenes_file scenes.txt --traj run

Trajectory library names map to files under configs/trajectories/. For a custom
trajectory, pass its path directly via --traj_path instead.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("batch_render_scenes")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.scene_utils import resolve_scene_dir, resolve_output_dir


TRAJ_LIBRARY = PROJECT_ROOT / "configs" / "trajectories"


def resolve_trajectory(traj_name: str, traj_path: str) -> str:
    """Resolve a trajectory library name or an explicit path to a JSON file."""
    if traj_path:
        return traj_path
    if not traj_name:
        return ""
    # library lookup: exact <name>.json
    cand = TRAJ_LIBRARY / f"{traj_name}.json"
    if cand.is_file():
        return str(cand)
    # fuzzy: any library file whose stem starts with <name>
    if TRAJ_LIBRARY.is_dir():
        matches = sorted(p for p in TRAJ_LIBRARY.glob("*.json") if p.stem.startswith(traj_name))
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Ambiguous trajectory '{traj_name}' in {TRAJ_LIBRARY}, matches: "
                f"{[p.stem for p in matches]}. Use the full stem.")
    # maybe they passed a path directly as --traj
    if Path(traj_name).is_file():
        return traj_name
    available = sorted(p.stem for p in TRAJ_LIBRARY.glob("*.json")) if TRAJ_LIBRARY.is_dir() else []
    raise FileNotFoundError(
        f"Trajectory '{traj_name}' not found in library {TRAJ_LIBRARY}. "
        f"Available: {available}")


def parse_scenes(scenes_arg: str, scenes_file: str) -> list[int]:
    """Parse '23,114,552' or a file (one id per line) into a list of ints."""
    if scenes_file:
        with open(scenes_file) as f:
            return [int(line.strip()) for line in f if line.strip() and not line.startswith("#")]
    if not scenes_arg:
        return []
    out = []
    for part in scenes_arg.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def build_render_command(
    scene_idx: int, traj_json: str, args, output_root: str, data_root: str,
) -> list[str]:
    """Construct the render_runner_video.py command for one scene."""
    ckpt = resolve_output_dir(scene_idx, output_root) / "checkpoint_final.pth"
    python = args.python
    cmd = [python, str(PROJECT_ROOT / "tools" / "render_runner_video.py"),
           "--resume_from", str(ckpt)]
    if traj_json:
        cmd += ["--path_json", traj_json]
    cmd += ["--mode", args.mode, "--fps", str(args.fps),
            "--scale", str(args.scale)]
    if args.anim_mode:
        cmd += ["--cameras", args.cameras]
    if args.resume:
        cmd.append("--resume")
    out_dir = resolve_output_dir(scene_idx, output_root)
    out_video = out_dir / "videos_eval" / f"scene{scene_idx}_{args.suffix}.mp4"
    cmd += ["--out", str(out_video)]
    return cmd


def main():
    p = argparse.ArgumentParser("Batch-render pedestrian videos across scenes")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scenes", default="", help="comma-sep scene ids, e.g. 23,114,552")
    g.add_argument("--scenes_file", default="", help="file with one scene id per line")
    p.add_argument("--traj", default="jaywalk",
                   help="trajectory library name (under configs/trajectories/) or a path")
    p.add_argument("--traj_path", default="",
                   help="explicit trajectory JSON path (overrides --traj)")
    p.add_argument("--mode", default="multicam_grid", choices=["video", "multicam_grid", "multiview"])
    p.add_argument("--cameras", default="0,1,2,3,4")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--scale", type=float, default=0.90)
    p.add_argument("--anim_mode", default="", help="reserved; mode comes from JSON gait")
    p.add_argument("--resume", action="store_true", help="crash-safe per-frame PNG + skip existing")
    p.add_argument("--suffix", default="adv", help="output filename suffix")
    p.add_argument("--output_root", default="outputs/waymo_omnire")
    p.add_argument("--data_root", default="data/waymo/processed/training")
    p.add_argument("--python", default=sys.executable, help="python interpreter to use")
    p.add_argument("--dry_run", action="store_true", help="print commands without running")
    args = p.parse_args()

    scenes = parse_scenes(args.scenes, args.scenes_file)
    if not scenes:
        p.error("provide --scenes or --scenes_file")
    traj_json = resolve_trajectory(args.traj, args.traj_path)

    logger.info("Batch render: %d scenes, traj=%s, mode=%s, dry_run=%s",
                len(scenes), traj_json or "(none)", args.mode, args.dry_run)

    n_ok, n_fail = 0, 0
    for scene_idx in scenes:
        # validate checkpoint + data dir exist
        ckpt = resolve_output_dir(scene_idx, args.output_root) / "checkpoint_final.pth"
        if not ckpt.is_file():
            logger.warning("scene %d: checkpoint missing (%s) — skipping", scene_idx, ckpt)
            n_fail += 1
            continue
        try:
            data_dir = resolve_scene_dir(scene_idx, args.data_root)
        except FileNotFoundError as e:
            logger.warning("scene %d: %s — skipping", scene_idx, str(e).splitlines()[0])
            n_fail += 1
            continue
        # ensure output dir exists
        out_dir = resolve_output_dir(scene_idx, args.output_root) / "videos_eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_render_command(scene_idx, traj_json, args, args.output_root, args.data_root)
        logger.info("scene %d → %s", scene_idx, " ".join(cmd))
        if args.dry_run:
            continue
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if ret.returncode == 0:
            n_ok += 1
            logger.info("scene %d: done ✓", scene_idx)
        else:
            n_fail += 1
            logger.error("scene %d: render failed (exit %d)", scene_idx, ret.returncode)

    if not args.dry_run:
        logger.info("Batch complete: %d ok, %d failed", n_ok, n_fail)


if __name__ == "__main__":
    main()
