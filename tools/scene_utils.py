"""Scene directory resolution helpers.

DriveStudio outputs live under ``outputs/waymo_omnire/scene<N>/`` while the
processed Waymo data lives under ``data/waymo/processed/training/<NNN>/``.
The scene index is zero-padded to 3 digits for the data dir but *not* for the
output dir (e.g. ``scene23`` ↔ ``023``), which is a frequent source of
``FileNotFoundError``. This module resolves either form to the correct
directory and reports the available scenes when the lookup fails.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def normalize_scene_idx(scene_idx_or_name: Union[str, int]) -> int:
    """Parse ``"23"`` / ``"023"`` / ``"scene23"`` / ``23`` into the int ``23``.

    Raises ``ValueError`` with a helpful message if the value can't be parsed.
    """
    if isinstance(scene_idx_or_name, int):
        return scene_idx_or_name
    s = str(scene_idx_or_name).strip()
    if s.lower().startswith("scene"):
        s = s[len("scene"):]
    s = s.lstrip("0") or "0"
    try:
        return int(s)
    except ValueError:
        raise ValueError(
            f"Cannot parse scene index from {scene_idx_or_name!r}; "
            f"expected '23', '023', or 'scene23'."
        )


def _list_available(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        return []
    return sorted(
        p.name for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()
    )


def resolve_scene_dir(
    scene_idx_or_name: Union[str, int],
    data_root: PathLike = "data/waymo/processed/training",
    *,
    zero_pad: int = 3,
) -> Path:
    """Resolve a processed-Waymo scene data directory from a flexible id.

    Accepts ``23`` / ``023`` / ``"scene23"`` and returns
    ``<data_root>/023``. On failure it raises ``FileNotFoundError`` listing the
    scenes actually present under ``data_root``.
    """
    idx = normalize_scene_idx(scene_idx_or_name)
    data_root = Path(data_root)
    padded = f"{idx:0{zero_pad}d}"
    candidate = data_root / padded
    if candidate.is_dir():
        return candidate
    available = _list_available(data_root)
    raise FileNotFoundError(
        f"Scene data directory not found: {candidate}\n"
        f"  resolved index={idx} -> expected padded='{padded}'\n"
        f"  data_root={data_root}\n"
        f"  available scenes: {available if available else '(none / dir missing)'}"
    )


def resolve_output_dir(
    scene_idx_or_name: Union[str, int],
    output_root: PathLike = "outputs/waymo_omnire",
) -> Path:
    """Resolve the DriveStudio output directory for a scene.

    Returns ``<output_root>/scene<N>`` (no zero-padding, matching DriveStudio).
    """
    idx = normalize_scene_idx(scene_idx_or_name)
    return Path(output_root) / f"scene{idx}"


def default_out_json(resume_from: PathLike, name: str = "traj_live.json") -> Path:
    """Default trajectory-export path next to a checkpoint.

    ``resume_from`` is ``.../scene<N>/checkpoint_final.pth``; we place the
    exported trajectory at ``.../scene<N>/trajectories/<name>``. This matches
    the rest of the pipeline (BEV planner / renderer all read from there).
    """
    ckpt = Path(resume_from).resolve()
    scene_dir = ckpt.parent  # .../scene<N>
    return scene_dir / "trajectories" / name
