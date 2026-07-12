"""Cross-platform font loading for PIL image annotations.

The renderers and BEV mini-map annotate frames with text. The original code
hard-coded ``/usr/share/fonts/truetype/dejavu/DejaVuSans*.ttf`` inside
``try/except`` blocks that silently fell back to the tiny default bitmap font
on systems without DejaVu. This centralizes the lookup so every tool gets a
consistent TrueType font when one is available, and a clear fallback otherwise.

Usage::

    from tools.fonts import get_font
    font = get_font(24)          # regular
    font_b = get_font(16, bold=True)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

try:
    from PIL import ImageFont
except ImportError:  # pragma: no cover - PIL is a hard dep of the renderers
    ImageFont = None  # type: ignore[assignment]


# Ordered by likelihood of being present on a Linux system. DejaVu ships with
# most distros; Liberation/Noto are common alternatives; the bare family names
# let PIL's freetype search the configured font paths as a last resort.
_REGULAR_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)
_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


def _first_existing(paths) -> Optional[str]:
    import os

    for p in paths:
        if os.path.isfile(p):
            return p
    return None


@lru_cache(maxsize=32)
def get_font(size: int = 16, bold: bool = False):
    """Return a PIL ImageFont at ``size`` px.

    Tries the TrueType candidates above, then PIL's family-name search, then
    finally the built-in default (which is tiny but always present). The result
    is cached by (size, bold).
    """
    if ImageFont is None:
        raise RuntimeError("Pillow is required for get_font() but is not installed.")

    candidates = _BOLD_CANDIDATES if bold else _REGULAR_CANDIDATES
    path = _first_existing(candidates)
    try:
        if path is not None:
            return ImageFont.truetype(path, size)
        # freetype default-path search by family name
        family = "DejaVu Sans" if not bold else "DejaVu Sans"
        try:
            return ImageFont.truetype(family, size)
        except Exception:
            return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()
