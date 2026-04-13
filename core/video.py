"""
Project Senxe — Video Utilities
================================
Shared video saving and side-by-side comparison generation.

Provides two main utilities for creating benchmark demonstration videos:

- ``save_video()``: Encodes a frame list to MP4 with automatic temporal
  downsampling to a target duration.
- ``make_side_by_side()``: Generates a synchronized dual-panel comparison
  video (e.g., CL1 Bio vs PPO) with labeled title bars.

Both functions use imageio + ffmpeg for headless-compatible encoding
(no display server required), making them suitable for CI pipelines
and remote server execution.
"""

from __future__ import annotations

import numpy as np
import imageio
import cv2
from tqdm import tqdm
from typing import List, Optional


def save_video(
    frames: List[np.ndarray],
    path: str,
    fps: int = 30,
    target_seconds: int = 20,
) -> None:
    """Save a list of frames as an MP4 video with automatic downsampling.

    If the total frame count exceeds ``fps × target_seconds``, frames are
    uniformly subsampled to fit the target duration while preserving
    temporal coverage across the full recording.

    Args:
        frames: List of (H, W, 3) uint8 RGB frames.
        path: Output file path (e.g., ``"output.mp4"``).
        fps: Video frame rate (default: 30).
        target_seconds: Approximate maximum video duration in seconds.
    """
    if not frames:
        print(f"  [Warning] No frames to save for {path}")
        return

    target = fps * target_seconds
    if len(frames) > target:
        idx = np.linspace(0, len(frames) - 1, target, dtype=int)
        frames = [frames[i] for i in idx]

    print(f"\n  Saving: {path} ({len(frames)} frames, ~{len(frames)/fps:.0f}s)")
    writer = imageio.get_writer(path, fps=fps, quality=8)
    for f in tqdm(frames, desc="  Encode", ncols=90, leave=False):
        writer.append_data(f)
    writer.close()
    print(f"  Saved: {path}")


def make_side_by_side(
    cl1_frames: List[np.ndarray],
    ppo_frames: List[np.ndarray],
    path: str,
    fps: int = 30,
    left_label: str = "CL1 Bio-Computer",
    right_label: str = "PPO Traditional RL",
    center_label: str = "",
) -> None:
    """Generate a synchronized side-by-side comparison video.

    Creates a dual-panel video with a colored title bar:
    - Left panel (red tint): CL1 biological neural controller
    - Right panel (blue tint): PPO traditional RL baseline

    Frames are synchronized by index and uniformly subsampled if the
    shorter sequence exceeds 25 seconds at the specified FPS.

    Args:
        cl1_frames: Left-side (CL1) frames, list of (H, W, 3) uint8.
        ppo_frames: Right-side (PPO) frames, list of (H, W, 3) uint8.
        path: Output file path.
        fps: Video frame rate (default: 30).
        left_label: Title text for the left panel.
        right_label: Title text for the right panel.
        center_label: Optional small center label (e.g., task name).
    """
    if not cl1_frames or not ppo_frames:
        print(f"  [Warning] Missing frames for side-by-side video")
        return

    n = min(len(cl1_frames), len(ppo_frames))
    target = fps * 25
    idx = np.linspace(0, n - 1, min(n, target), dtype=int)

    print(f"\n  Side-by-side: {path} ({len(idx)} frames)")
    writer = imageio.get_writer(path, fps=fps, quality=8)

    for i in tqdm(idx, desc="  SBS", ncols=90, leave=False):
        f1 = cl1_frames[i]
        f2 = ppo_frames[i]

        h = min(f1.shape[0], f2.shape[0])
        f1 = f1[:h]
        f2 = f2[:h]

        # Concatenate with 4px separator line
        sep = np.ones((h, 4, 3), dtype=np.uint8) * 200
        combined = np.concatenate([f1, sep, f2], axis=1)
        cw = combined.shape[1]
        mid = cw // 2

        # Title bar (35px height, split-colored)
        title_h = 35
        tb = np.zeros((title_h, cw, 3), dtype=np.uint8)
        tb[:, :mid, 0] = 90; tb[:, :mid, 1] = 25; tb[:, :mid, 2] = 25
        tb[:, mid:, 0] = 20; tb[:, mid:, 1] = 40; tb[:, mid:, 2] = 90
        combined = np.concatenate([tb, combined], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Left title (white text with black shadow for readability)
        cv2.putText(combined, left_label, (10, 24), font, 0.45,
                     (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(combined, left_label, (10, 24), font, 0.45,
                     (255, 255, 255), 1, cv2.LINE_AA)

        # Right title
        cv2.putText(combined, right_label, (mid + 10, 24), font, 0.45,
                     (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(combined, right_label, (mid + 10, 24), font, 0.45,
                     (255, 255, 255), 1, cv2.LINE_AA)

        # Center label (small, positioned above main titles)
        if center_label:
            (tw, _), _ = cv2.getTextSize(center_label, font, 0.28, 1)
            cx = mid - tw // 2
            cv2.putText(combined, center_label, (cx + 1, 11), font, 0.28,
                         (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(combined, center_label, (cx, 10), font, 0.28,
                         (140, 140, 140), 1, cv2.LINE_AA)

        writer.append_data(combined)

    writer.close()
    print(f"  Saved: {path}")
