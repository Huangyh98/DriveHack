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
from dataclasses import dataclass
from typing import Optional


# Animation constants (from runner_seq.npz)
N_ANIM_FRAMES = 40          # total animation frames in runner_seq.npz
CYCLE_FRAMES = 20           # frames per gait cycle (left+right step). n_anim//2


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

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"len={self.trajectory_length:.1f}m  "
            f"steps={self.n_steps:.0f}  "
            f"speed={self.char_speed:.1f}m/s  "
            f"step_freq={self.step_freq:.1f}Hz  "
            f"anim_speed={self.anim_speed:.2f}"
        )

    def detail(self) -> str:
        """Multi-line detailed report."""
        return (
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
) -> GaitParams:
    """Compute animation speed for gait-matched, slide-free motion.

    Args:
        trajectory_length: total path length in meters.
        n_video_frames: number of frames in the output video.
        fps: video frame rate.
        cycle_stride: meters covered per gait cycle (2 steps). Default 2.6m.
        cycle_frames: animation frames per gait cycle. Default 20.

    Returns:
        GaitParams with all computed values.
    """
    if trajectory_length < 1e-6 or n_video_frames < 1:
        return GaitParams(
            trajectory_length=trajectory_length,
            cycle_stride=cycle_stride,
            step_length=cycle_stride / 2,
            n_steps=0, n_cycles=0,
            n_video_frames=n_video_frames, fps=fps,
            anim_speed=0.0, char_speed=0.0,
            step_freq=0.0, video_duration=n_video_frames / fps if fps > 0 else 0,
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
    )
