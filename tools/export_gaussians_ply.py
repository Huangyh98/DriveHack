"""
Export a trained driving Gaussian checkpoint to GraphDECO/Inria-style PLY.

Usage:
    export PYTHONPATH=$(pwd)
    python tools/export_gaussians_ply.py \
        --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth \
        --output outputs/waymo_omnire/scene23/gaussians_frame000.ply \
        --frame 0
"""

import argparse
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.misc import import_str


logger = logging.getLogger("export_gaussians_ply")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


GAUSSIAN_CLASSES = ("Background", "RigidNodes", "DeformableNodes", "SMPLNodes")


def _as_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu().contiguous()


def _logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def _normalize_quats(quats: torch.Tensor) -> torch.Tensor:
    return quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _pad_scales(scales: torch.Tensor) -> torch.Tensor:
    if scales.shape[-1] == 3:
        return scales
    if scales.shape[-1] == 1:
        return scales.repeat(1, 3)
    if scales.shape[-1] == 2:
        return torch.cat([scales, torch.full_like(scales[..., :1], 1e-6)], dim=-1)
    raise ValueError(f"Unsupported scale shape: {tuple(scales.shape)}")


def _feature_rest_flat(model, mask: torch.Tensor) -> torch.Tensor:
    rest = model._features_rest[mask]
    if rest.numel() == 0:
        return rest.reshape(rest.shape[0], -1)
    # GraphDECO PLY stores all SH coefficients flattened after the DC terms.
    return rest.reshape(rest.shape[0], -1)


def _collect_background(model, alpha_thresh: float) -> Optional[Dict[str, torch.Tensor]]:
    opacities = model.get_opacity.squeeze(-1)
    mask = opacities > alpha_thresh
    if not mask.any():
        return None

    scales = _pad_scales(model.get_scaling[mask]).clamp_min(1e-8)
    return {
        "positions": model._means[mask],
        "f_dc": model._features_dc[mask],
        "f_rest": _feature_rest_flat(model, mask),
        "opacities": _logit(opacities[mask]).unsqueeze(-1),
        "scales": torch.log(scales),
        "rotations": _normalize_quats(model.get_quats[mask]),
    }


def _collect_dynamic(model, frame: int, alpha_thresh: float) -> Optional[Dict[str, torch.Tensor]]:
    if hasattr(model, "set_cur_frame"):
        model.set_cur_frame(frame)
    else:
        model.cur_frame = frame

    valid_mask = model.get_pts_valid_mask()
    if not valid_mask.any():
        return None

    means = model._means
    scales = model.get_scaling
    quats = model._quats

    if hasattr(model, "deform_network"):
        if model.ctrl_cfg.use_deformgs_for_nonrigid and model.step > model.ctrl_cfg.use_deformgs_after:
            delta_xyz, delta_quat, delta_scale = model.get_deformation(local_means=model._means)
            means = model._means + delta_xyz
            quats = model._quats + delta_quat if delta_quat is not None else model._quats
            scales = model.get_scaling + delta_scale if delta_scale is not None else model.get_scaling
        world_means = model.transform_means(means)
        world_quats = model.transform_quats(quats)
    elif hasattr(model, "smpl_qauts"):
        instance_mask = model.instances_fv[frame]
        if not instance_mask.any():
            return None
        if model.ball_gaussians:
            world_means = model.transform_means(means)
            world_quats = quats
            scales = torch.exp(model._scales.repeat(1, 3))
        else:
            world_means, world_quats = model.transform_means_and_quats(means, quats)
            scales = torch.exp(model._scales)
    else:
        world_means = model.transform_means(means)
        world_quats = model.transform_quats(quats)

    opacities = (model.get_opacity * valid_mask.float().unsqueeze(-1)).squeeze(-1)
    mask = opacities > alpha_thresh
    if not mask.any():
        return None

    scales = _pad_scales(scales[mask]).clamp_min(1e-8)
    return {
        "positions": world_means[mask],
        "f_dc": model._features_dc[mask],
        "f_rest": _feature_rest_flat(model, mask),
        "opacities": _logit(opacities[mask]).unsqueeze(-1),
        "scales": torch.log(scales),
        "rotations": _normalize_quats(world_quats[mask]),
    }


def _merge_gaussians(chunks: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    chunks = list(chunks)
    if not chunks:
        raise ValueError("No gaussians passed the opacity threshold.")

    max_rest = max(chunk["f_rest"].shape[1] for chunk in chunks)
    padded_chunks: List[Dict[str, torch.Tensor]] = []
    for chunk in chunks:
        chunk = dict(chunk)
        rest = chunk["f_rest"]
        if rest.shape[1] < max_rest:
            pad = torch.zeros(rest.shape[0], max_rest - rest.shape[1], device=rest.device, dtype=rest.dtype)
            chunk["f_rest"] = torch.cat([rest, pad], dim=1)
        padded_chunks.append(chunk)

    keys = ("positions", "f_dc", "f_rest", "opacities", "scales", "rotations")
    return {key: torch.cat([chunk[key] for chunk in padded_chunks], dim=0) for key in keys}


def _apply_position_scale(chunk: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    if scale == 1.0:
        return chunk
    chunk = dict(chunk)
    chunk["positions"] = chunk["positions"] * scale
    chunk["scales"] = chunk["scales"] + torch.log(torch.tensor(scale, device=chunk["scales"].device, dtype=chunk["scales"].dtype))
    return chunk


def _ply_properties(num_rest: int) -> List[str]:
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props.extend(f"f_rest_{idx}" for idx in range(num_rest))
    props.extend(["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    return props


def write_gaussian_ply(path: Path, chunk: Dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    positions = _as_cpu_float(chunk["positions"])
    f_dc = _as_cpu_float(chunk["f_dc"])
    f_rest = _as_cpu_float(chunk["f_rest"])
    opacities = _as_cpu_float(chunk["opacities"])
    scales = _as_cpu_float(chunk["scales"])
    rotations = _as_cpu_float(chunk["rotations"])

    if positions.shape[0] == 0:
        raise ValueError(f"No gaussians to write for {path}")

    properties = _ply_properties(f_rest.shape[1])
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {positions.shape[0]}",
    ]
    header_lines.extend(f"property float {prop}" for prop in properties)
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    zero_normals = torch.zeros(positions.shape[0], 3)
    rows = torch.cat([positions, zero_normals, f_dc, f_rest, opacities, scales, rotations], dim=1)
    packer = struct.Struct("<" + "f" * rows.shape[1])

    with path.open("wb") as f:
        f.write(header)
        for row in rows.tolist():
            f.write(packer.pack(*row))

    logger.info("Wrote %s (%d gaussians)", path, positions.shape[0])


def _infer_num_timesteps(checkpoint: Dict) -> int:
    for class_name in ("RigidNodes", "DeformableNodes", "SMPLNodes"):
        model_state = checkpoint.get("models", {}).get(class_name)
        if model_state is not None and "instances_fv" in model_state:
            return int(model_state["instances_fv"].shape[0])
    return 1


def build_trainer(resume_from: str, device: torch.device):
    log_dir = os.path.dirname(resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    checkpoint = torch.load(resume_from, map_location=device)
    num_timesteps = _infer_num_timesteps(checkpoint)

    gaussian_model_config = OmegaConf.create({
        class_name: cfg.model[class_name]
        for class_name in GAUSSIAN_CLASSES
        if class_name in cfg.model
    })

    logger.info("Building Gaussian-only trainer...")
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=num_timesteps,
        model_config=gaussian_model_config,
        num_train_images=1,
        num_full_images=1,
        test_set_indices=[],
        scene_aabb=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device),
        device=device,
    )
    trainer.load_state_dict(checkpoint, load_only_model=True, strict=True)
    trainer.set_eval()
    return trainer, num_timesteps


def _parse_frames(args: argparse.Namespace, num_timesteps: int) -> List[int]:
    if args.frames:
        frames: List[int] = []
        for part in args.frames.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                pieces = [int(x) for x in part.split(":")]
                if len(pieces) == 2:
                    start, end = pieces
                    step = args.frame_stride
                elif len(pieces) == 3:
                    start, end, step = pieces
                else:
                    raise ValueError(f"Bad --frames range: {part}")
                frames.extend(range(start, end + 1, step))
            else:
                frames.append(int(part))
    else:
        frames = list(range(args.frame, args.frame_end + 1, args.frame_stride)) if args.frame_end is not None else [args.frame]

    frames = sorted(dict.fromkeys(frames))
    for frame in frames:
        if frame < 0 or frame >= num_timesteps:
            raise ValueError(f"Frame {frame} must be in [0, {num_timesteps - 1}]")
    return frames


def _collect_frame_chunks(trainer, frame: int, alpha_thresh: float, position_scale: float, include_background: bool) -> Dict[str, Dict[str, torch.Tensor]]:
    chunks: Dict[str, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for class_name in GAUSSIAN_CLASSES:
            if class_name not in trainer.models:
                continue
            if class_name == "Background" and not include_background:
                continue
            model = trainer.models[class_name]
            if class_name == "Background":
                chunk = _collect_background(model, alpha_thresh)
            else:
                chunk = _collect_dynamic(model, frame, alpha_thresh)
            if chunk is None:
                logger.info("Skipped %s frame %03d: no visible gaussians at alpha_thresh=%s", class_name, frame, alpha_thresh)
                continue
            chunks[class_name] = _apply_position_scale(chunk, position_scale)
    return chunks


def export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    trainer, num_timesteps = build_trainer(args.resume_from, device)
    frames = _parse_frames(args, num_timesteps)
    output = Path(args.output)

    if len(frames) > 1 and output.suffix.lower() == ".ply":
        raise ValueError("--output must be a directory when exporting multiple frames")

    if args.reuse_background and len(frames) > 1:
        bg_chunks = _collect_frame_chunks(trainer, frames[0], args.alpha_thresh, args.position_scale, include_background=True)
        if "Background" in bg_chunks:
            out_dir = output
            write_gaussian_ply(out_dir / f"Background_frame{frames[0]:03d}.ply", bg_chunks["Background"])
        else:
            logger.warning("No background chunk was exported.")

        for frame in frames:
            chunks = _collect_frame_chunks(trainer, frame, args.alpha_thresh, args.position_scale, include_background=False)
            if not chunks:
                continue
            dynamic = _merge_gaussians(chunks.values())
            write_gaussian_ply(output / f"Dynamic_frame{frame:03d}.ply", dynamic)
        return

    for frame in frames:
        chunks = _collect_frame_chunks(
            trainer,
            frame,
            args.alpha_thresh,
            args.position_scale,
            include_background=not args.dynamic_only,
        )
        if args.separate:
            out_dir = output if output.suffix == "" else output.with_suffix("")
            for class_name, chunk in chunks.items():
                write_gaussian_ply(out_dir / f"{class_name}_frame{frame:03d}.ply", chunk)

        merged = _merge_gaussians(chunks.values())
        if args.dynamic_only:
            merged_name = f"Dynamic_frame{frame:03d}.ply"
        else:
            merged_name = f"gaussians_frame{frame:03d}.ply"
        merged_path = output if output.suffix.lower() == ".ply" else output / merged_name
        write_gaussian_ply(merged_path, merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export trained driving Gaussians to UE-compatible 3DGS PLY")
    parser.add_argument("--resume_from", required=True, help="Path to checkpoint_final.pth")
    parser.add_argument("--output", required=True, help="Output .ply path, or output directory when no .ply suffix is given")
    parser.add_argument("--frame", type=int, default=0, help="Waymo timestep used to place dynamic objects")
    parser.add_argument("--frame_end", type=int, default=None, help="Inclusive end frame for sequence export")
    parser.add_argument("--frame_stride", type=int, default=1, help="Frame stride for sequence export")
    parser.add_argument("--frames", default=None, help="Comma/range list, e.g. '0,10,20' or '0:60:5'")
    parser.add_argument("--alpha_thresh", type=float, default=0.01, help="Drop gaussians below this activated opacity")
    parser.add_argument(
        "--position_scale",
        type=float,
        default=1.0,
        help="Scale positions and Gaussian radii. Use 100.0 if your UE importer expects centimeters.",
    )
    parser.add_argument("--separate", action="store_true", help="Also write one PLY per Gaussian class")
    parser.add_argument("--dynamic_only", action="store_true", help="Export only dynamic Gaussian classes")
    parser.add_argument(
        "--reuse_background",
        action="store_true",
        help="For multi-frame export, write Background_frameXXX once and Dynamic_frameXXX per frame.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
