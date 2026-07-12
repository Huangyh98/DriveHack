# Trajectory Library

Reusable, gait-annotated trajectory JSONs for injecting characters into driving
scenes. Each file is consumed by `render_runner_video.py --path_json <file>` or
referenced from a multi-character config (`--multi_traj`).

## Conventions

- **Coordinates** are in the ego-normalized Waymo world frame (X forward / Y
  left / Z up, meters, origin = first ego frame). These are *scene-relative*:
  the same numbers mean different physical locations across scenes, so a
  trajectory here is a *pattern* you adapt to a scene, not an absolute path.
- **`gait`** carries stride-matched animation parameters written by
  `trajectory_previewer.py`. The renderer reads `anim_mode` and `cycle_stride`
  from here automatically.
- **`total_length`** is the arc-length of the smoothed trajectory (meters).

## Using a library trajectory

```bash
# 1. Preview & adapt it to your scene in the 3D editor
python tools/trajectory_previewer.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --scene_dir data/waymo/processed/training/023 \
    --path_json configs/trajectories/jaywalk_cross_scene23.json

# 2. Render
python tools/render_runner_video.py \
    --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
    --path_json outputs/waymo_omnire/scene23/trajectories/traj_live.json \
    --out outputs/waymo_omnire/scene23/videos_eval/scene23_jaywalk.mp4
```

> **Tip**: load a library trajectory into the previewer as a *starting point*,
> then click in your own scene to adjust the waypoints before exporting.

## Bundled examples

| File | Pattern | Length | Mode |
|------|---------|--------|------|
| `jaywalk_cross_scene23.json` | Pedestrian crosses the road left→right in front of ego | 8.0 m | walk |

These were derived from scene023. To add your own, draw a trajectory in the
previewer and export it, then copy the resulting `traj_live.json` here with a
descriptive name.
