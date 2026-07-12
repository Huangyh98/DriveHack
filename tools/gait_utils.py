"""Gait-matched animation speed calculation.

Given a trajectory length and video parameters, compute the animation playback
speed that ensures:
  1. No foot sliding (step length is physically consistent)
  2. The character completes the entire trajectory within the video

Key concept:
  - runner_seq.npz has 40 frames, but frames 0-19 == frames 20-39 (duplicated).
  - One gait cycle (left step + right step) = 20 animation frames.
  - One gait cycle covers `cycle_stride` meters (default 2.6m → 1.3m per step).

Formula chain:
  trajectory_length L (m)
  → total steps      = L / step_length           (step_length = cycle_stride / 2)
  → total cycles     = L / cycle_stride
  → total anim frames= total_cycles × cycle_frames              (= cycles × 20)
  → anim_speed       = total_anim_frames / n_video_frames        (anim frames per video frame)

This guarantees:
  - The character takes exactly L / step_length steps (integer-ish)
  - Each step covers exactly step_length meters on the ground
  - Animation plays through exactly the right number of gait cycles
"""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Animation constants (from runner_seq.npz)
N_ANIM_FRAMES = 40          # total animation frames in runner_seq.npz
CYCLE_FRAMES = 20           # frames per gait cycle (left+right step). n_anim//2


class AnimationMode(str, Enum):
    """Character locomotion presets.

    Each mode maps to a recommended ``cycle_stride`` (meters per gait cycle =
    two steps). The same baked run-loop animation (runner_seq.npz) is reused
    for all moving modes — only the playback speed (anim_speed) changes, so a
    shorter stride yields a faster-looking step frequency and a "walk" feel.
    ``stand`` freezes the character in place on a neutral frame.
    """

    RUN = "run"
    JOG = "jog"
    WALK = "walk"
    STAND = "stand"

    @property
    def cycle_stride(self) -> float:
        # Empirical per-cycle distances; tuned to land on natural human cadence.
        return {
            "run": 2.6,    # 1.3 m/step — full running stride
            "jog": 2.0,    # 1.0 m/step — easy jog
            "walk": 1.2,   # 0.6 m/step — brisk walk
            "stand": 0.0,  # in-place, no travel
        }[self.value]

    @property
    def is_static(self) -> bool:
        return self == AnimationMode.STAND


# Default mode used when none is specified (preserves original behavior).
DEFAULT_MODE = AnimationMode.RUN


@dataclass
class GaitParams:
    """Result of gait-matched speed calculation."""
    trajectory_length: float       # meters
    cycle_stride: float            # meters per gait cycle (2 steps)
    step_length: float             # meters per single step
    n_steps: float                 # total steps (may be fractional)
    n_cycles: float                # total gait cycles
    n_video_frames: int            # video frame count
    fps: float                     # video frame rate
    anim_speed: float              # animation frames per video frame
    char_speed: float              # character ground speed (m/s)
    step_freq: float               # steps per second
    video_duration: float          # seconds
    anim_mode: str = "run"         # locomotion mode label (run/jog/walk/stand)

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"mode={self.anim_mode}  "
            f"len={self.trajectory_length:.1f}m  "
            f"steps={self.n_steps:.0f}  "
            f"speed={self.char_speed:.1f}m/s  "
            f"step_freq={self.step_freq:.1f}Hz  "
            f"anim_speed={self.anim_speed:.2f}"
        )

    def detail(self) -> str:
        """Multi-line detailed report."""
        return (
            f"  动画模式:    {self.anim_mode}\n"
            f"  轨迹长度:    {self.trajectory_length:.2f} m\n"
            f"  步长参数:    {self.step_length:.2f} m/步 (周期 {self.cycle_stride:.1f}m = 2步)\n"
            f"  总步数:      {self.n_steps:.1f} 步 ({self.n_cycles:.1f} 个步态周期)\n"
            f"  视频帧数:    {self.n_video_frames} 帧 @ {self.fps:.0f}fps = {self.video_duration:.1f}s\n"
            f"  角色速度:    {self.char_speed:.2f} m/s\n"
            f"  步频:        {self.step_freq:.2f} 步/秒\n"
            f"  anim_speed:  {self.anim_speed:.3f} (动画帧/视频帧)"
        )

    def speed_assessment(self) -> str:
        """Assess if the speed is physically reasonable."""
        s = self.char_speed
        if s == 0:
            return "静止"
        elif s < 0.5:
            return "⚠ 太慢（几乎不动）"
        elif s <= 1.4:
            return "✓ 正常步行"
        elif s <= 2.5:
            return "✓ 快走/慢跑"
        elif s <= 4.0:
            return "✓ 跑步"
        elif s <= 6.0:
            return "⚠ 快跑（冲刺）"
        else:
            return "⚠⚠ 非人类速度！"


def compute_gait_params(
    trajectory_length: float,
    n_video_frames: int,
    fps: float = 10.0,
    cycle_stride: float = 2.6,
    cycle_frames: int = CYCLE_FRAMES,
    mode: str = "run",
) -> GaitParams:
    """Compute animation speed for gait-matched, slide-free motion.

    Args:
        trajectory_length: total path length in meters.
        n_video_frames: number of frames in the output video.
        fps: video frame rate.
        cycle_stride: meters covered per gait cycle (2 steps). Default 2.6m.
        cycle_frames: animation frames per gait cycle. Default 20.
        mode: locomotion mode label ("run"/"jog"/"walk"/"stand"). Stored on the
            result; for "stand" the character is frozen in place regardless of
            trajectory length.

    Returns:
        GaitParams with all computed values.
    """
    # Resolve mode; if the caller passed the default cycle_stride but a named
    # mode, adopt that mode's recommended stride (unless they overrode it).
    try:
        anim_mode = AnimationMode(mode)
    except ValueError:
        anim_mode = AnimationMode.RUN

    # "stand": character frozen on a neutral frame; no travel.
    if anim_mode.is_static:
        video_duration = n_video_frames / fps if fps > 0 else 0
        return GaitParams(
            trajectory_length=trajectory_length,
            cycle_stride=0.0,
            step_length=0.0,
            n_steps=0.0, n_cycles=0.0,
            n_video_frames=n_video_frames, fps=fps,
            anim_speed=0.0, char_speed=0.0,
            step_freq=0.0, video_duration=video_duration,
            anim_mode="stand",
        )

    if trajectory_length < 1e-6 or n_video_frames < 1:
        return GaitParams(
            trajectory_length=trajectory_length,
            cycle_stride=cycle_stride,
            step_length=cycle_stride / 2,
            n_steps=0, n_cycles=0,
            n_video_frames=n_video_frames, fps=fps,
            anim_speed=0.0, char_speed=0.0,
            step_freq=0.0, video_duration=n_video_frames / fps if fps > 0 else 0,
            anim_mode=anim_mode.value,
        )

    step_length = cycle_stride / 2.0        # meters per single step
    n_steps = trajectory_length / step_length  # total steps
    n_cycles = trajectory_length / cycle_stride  # total gait cycles
    total_anim_frames = n_cycles * cycle_frames  # anim frames needed
    anim_speed = total_anim_frames / n_video_frames  # anim frames per video frame
    video_duration = n_video_frames / fps if fps > 0 else 0
    char_speed = trajectory_length / video_duration if video_duration > 0 else 0
    step_freq = n_steps / video_duration if video_duration > 0 else 0

    return GaitParams(
        trajectory_length=trajectory_length,
        cycle_stride=cycle_stride,
        step_length=step_length,
        n_steps=n_steps,
        n_cycles=n_cycles,
        n_video_frames=n_video_frames,
        fps=fps,
        anim_speed=anim_speed,
        char_speed=char_speed,
        step_freq=step_freq,
        video_duration=video_duration,
        anim_mode=anim_mode.value,
    )


def compute_animation_for_mode(
    mode,
    trajectory_length: float,
    n_video_frames: int,
    fps: float = 10.0,
    cycle_frames: int = CYCLE_FRAMES,
) -> GaitParams:
    """Compute gait params for a named animation mode.

    Picks the mode's recommended ``cycle_stride`` and delegates to
    :func:`compute_gait_params`. This is the convenience entry point used by
    the trajectory previewer's mode dropdown and by the renderer when a JSON
    trajectory carries an ``anim_mode`` field.

    ``mode`` may be an :class:`AnimationMode` or its string value.
    """
    try:
        anim_mode = AnimationMode(mode) if not isinstance(mode, AnimationMode) else mode
    except ValueError:
        anim_mode = AnimationMode.RUN
    return compute_gait_params(
        trajectory_length=trajectory_length,
        n_video_frames=n_video_frames,
        fps=fps,
        cycle_stride=anim_mode.cycle_stride,
        cycle_frames=cycle_frames,
        mode=anim_mode.value,
    )


def neutral_anim_frame(n_anim: int = N_ANIM_FRAMES) -> int:
    """A neutral mid-stance frame for the "stand" (in-place) mode.

    Frame ~10 in the 40-frame run loop is roughly mid-stance (weight on one
    leg, the other passing through). Used so a frozen character looks like it
    is standing/walking in place rather than frozen mid-stride.
    """
    return min(max(n_anim // 4, 0), max(n_anim - 1, 0))
