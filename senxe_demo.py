#!/usr/bin/env python3
"""
Project Senxe v3.0 — FetchPickAndPlace Benchmark (Legacy)
==========================================================
CL1 Bio-Computer vs PPO vs Random — Pick-and-Place Sample Efficiency Benchmark

Demonstrates key biological control components:
  - VIE  (Virtual Interference Encoding)   — Doom-style visual + tactile encoding
  - Antagonistic Decoding                  — Flexor/extensor motor output
  - PDI  (Physical Disturbance Index)      — FEP-inspired explore/exploit gate
  - Dopamine-like Reward Injection         — Structured burst reinforcement
  - Channel Warm-up Calibration            — 10-second responsiveness probing
  - Metabolic Guardrail                    — Per-channel health monitoring

Usage:  python senxe_demo.py
Output: cl1_pickandplace.mp4, side_by_side_demo.mp4, learning_curve.png

To use real CL1 hardware: pip install cl-sdk  (auto-detected, zero code changes)
"""

import os
import sys
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # Register Fetch environments
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import imageio
import cv2
from tqdm import tqdm
from collections import deque
from contextlib import contextmanager

# ═══ Core Modules (shared with v4.0) ═══
from core.neurons import (
    cl_open, warmup_calibration,
    ChannelSet, StimDesign, BurstDesign,
    MockNeurons, CL_AVAILABLE,
)
from core.decoder import AntagonisticDecoder
from core.pdi import PDI
from core.video import save_video, make_side_by_side

# ═══ Configuration ═══
# Senxe v3.0 — FetchPickAndPlace + Doom-style Visual Object Encoding

ENV_NAME          = "FetchPickAndPlace-v4"   # Gymnasium Robotics pick-and-place
REWARD_TYPE       = "dense"                  # "dense" or "sparse"
CL1_EPISODES      = 300                      # CL1 training episodes
CL1_MAX_STEPS     = 100                      # Max steps per episode
PPO_TIMESTEPS     = 10_000                   # PPO total training steps
PPO_EVAL_EPS      = 300                      # PPO evaluation episodes (aligned with CL1)
RANDOM_EPISODES   = 300                      # Random baseline episodes
VIDEO_FPS         = 30                       # Video frame rate
VIDEO_CL1         = "cl1_pickandplace.mp4"   # CL1 standalone video
VIDEO_SIDE        = "side_by_side_demo.mp4"  # Side-by-side comparison video
PLOT_FILE         = "learning_curve.png"     # Learning curve plot
RECORD_LAST_N     = 100                      # Only record last N mature episodes
WARMUP_SECONDS    = 10                       # Channel warm-up calibration duration (seconds)
ACTION_SCALE      = 0.35                     # Action scaling factor

# Dopamine reward injection parameters
DOPAMINE_TOP_K    = 8                        # Stimulate top-K responsive channels on reward
DOPAMINE_BURST_N  = 15                       # Dopamine burst pulse count
DOPAMINE_BURST_HZ = 300                      # Dopamine burst frequency (Hz)

# ═══ CL1 Neural Interface — see core/neurons.py ═══

# ═══ VIE: Virtual Interference Encoding ═══
#
# Rate coding:    distance → burst frequency (farther = more urgent)
# Temporal coding: velocity → traveling wave patterns
# v3.0: + Doom-style visual brightness encoding + object direction
#
# 64-channel layout (shared with AntagonisticDecoder):
#   CH 0-15:  Distance/pressure (rate coding)
#   CH 16-31: Velocity (temporal coding, traveling waves)
#   CH 32-47: Position (absolute grip position)
#   CH 48-63: Goal/object direction (delta vector guidance)

class VIE:
    """Virtual Interference Encoding for v3.0 FetchPickAndPlace.

    Maps distance, velocity, position, goal direction, and visual brightness
    onto a 64-channel MEA using rate coding and temporal coding schemes.
    """

    CH_PRESSURE = list(range(0,  16))
    CH_VELOCITY = list(range(16, 32))
    CH_POSITION = list(range(32, 48))
    CH_GOALDIR  = list(range(48, 64))

    def __init__(self, neurons, env=None):
        self.neurons = neurons
        self.env = env  # For Doom-style visual brightness encoding

    def encode(self, obs, goal):
        """Encode observation into neural stimulation patterns on the 64-ch MEA.

        Converts arm state (position, velocity, distance) and visual brightness
        into charge-balanced biphasic pulse trains across channel groups.

        Args:
            obs: Observation array or dict from Gymnasium environment.
            goal: Desired goal position (3D).
        """

        # ── Safely extract observation data ──
        if isinstance(obs, dict):
            grip_pos = obs.get("observation", np.zeros(25))[:3]
            grip_vel = obs.get("observation", np.zeros(25))[3:6] if len(obs.get("observation", np.zeros(25))) >= 6 else np.zeros(3)
            # Safely extract object position
            object_pos = obs.get("object", np.zeros(3))[:3]
            # Safely extract gripper state
            gripper_state = obs.get("observation", np.zeros(25))[9:11] if len(obs.get("observation", np.zeros(25))) >= 11 else np.zeros(2)
        else:
            grip_pos = obs[:3]
            grip_vel = obs[3:6] if len(obs) >= 6 else np.zeros(3)
            # Object position (FetchPickAndPlace obs layout)
            object_pos = obs[3:6] if len(obs) >= 15 else grip_pos.copy()
            gripper_state = obs[9:11] if len(obs) >= 11 else np.zeros(2)

        delta     = goal - grip_pos
        distance  = np.linalg.norm(delta)
        direction = delta / (distance + 1e-8)

        stim = StimDesign(160, -1.0, 160, 1.0)  # Standard biphasic pulse

        # ── Rate Coding: distance → burst frequency ──
        # FEP interpretation: distance = surprise, needs minimization
        burst_hz = int(np.clip(50 + 400 * distance, 50, 400))
        burst_n  = max(1, min(10, int(distance * 15)))
        self.neurons.stim(
            ChannelSet(*self.CH_PRESSURE[:8]), stim, BurstDesign(burst_n, burst_hz)
        )

        # ── Doom-style visual brightness encoding ──
        # Average rendered frame brightness → rate coding
        visual_brightness = 0.5  # Default medium brightness (fallback)
        if self.env is not None:
            try:
                vis_frame = self.env.render()
                if vis_frame is not None and vis_frame.size > 0:
                    visual_brightness = np.mean(vis_frame.astype(np.float32)) / 255.0
            except Exception:
                pass  # Use default on render failure
        vis_hz = int(np.clip(50 + 300 * visual_brightness, 50, 350))
        vis_n  = max(1, int(visual_brightness * 5))
        vis_stim = StimDesign(160, -0.8, 160, 0.8)
        self.neurons.stim(
            ChannelSet(*self.CH_PRESSURE[8:16]), vis_stim, BurstDesign(vis_n, vis_hz)
        )

        # ── Object direction encoding (CH 56-63 region) ──
        obj_delta = object_pos - grip_pos
        obj_dist  = np.linalg.norm(obj_delta)
        obj_dir   = obj_delta / (obj_dist + 1e-8)
        for axis in range(3):
            ch_base = self.CH_GOALDIR[8] + axis * 2  # CH 56-63 region
            chs = ChannelSet(*[ch_base, min(ch_base + 1, 63)])
            d = obj_dir[axis]
            intensity = np.clip(abs(d) * 1.5 + 0.1, 0.1, 2.0)
            obj_stim = StimDesign(160, -intensity * np.sign(d),
                                  160,  intensity * np.sign(d))
            obj_burst = BurstDesign(max(1, int(abs(d) * 5)),
                                    int(50 + abs(d) * 100))
            self.neurons.stim(chs, obj_stim, obj_burst)

        # ── Temporal Coding: velocity → traveling waves ──
        vel_mag = np.linalg.norm(grip_vel)
        if vel_mag > 0.003:
            for axis in range(3):
                v = grip_vel[axis]
                if abs(v) > 0.003:
                    ch_base = self.CH_VELOCITY[0] + axis * 5
                    chs = ChannelSet(*range(ch_base, min(ch_base + 5, 32)))
                    intensity = np.clip(abs(v) * 5, 0.1, 2.0)
                    wave_stim = StimDesign(160, -intensity, 160, intensity)
                    wave_hz = int(np.clip(60 * abs(v), 20, 200))
                    self.neurons.stim(chs, wave_stim, BurstDesign(2, wave_hz))

        # ── Goal direction encoding (CH 48-55 region) ──
        for axis in range(3):
            ch_base = self.CH_GOALDIR[0] + axis * 2  # CH 48-55 region
            chs = ChannelSet(*[ch_base, min(ch_base + 1, 55)])
            d = direction[axis]
            intensity = np.clip(abs(d) * 1.5 + 0.1, 0.1, 2.0)
            dir_stim = StimDesign(160, -intensity * np.sign(d),
                                  160,  intensity * np.sign(d))
            dir_burst = BurstDesign(max(1, int(abs(d) * 5)),
                                    int(50 + abs(d) * 100))
            self.neurons.stim(chs, dir_stim, dir_burst)

# ═══ AntagonisticDecoder, PDI — see core/decoder.py, core/pdi.py ═══

# ═══ Real-time HUD Overlay (OpenCV) ═══
#
# Three overlays drawn onto the MuJoCo rgb_array frame:
#   1. Top-left text:   Episode / Reward / PDI / Min Health
#   2. Top-right bars:  Top-8 channel firing rates
#   3. Bottom curve:    Rolling reward (last 50 steps, red line)

def _overlay_text(frame, ep, reward, pdi, min_health, distance=0.0,
                   success_rate=0.0, gripper_status=""):
    """Top-left HUD text overlay with semi-transparent background."""
    lines = [
        f"Ep:{ep} R:{reward:.1f}",
        f"PDI:{pdi:.2f} H:{min_health:.2f}",
        f"Dist:{distance:.3f}",
    ]
    # Show success rate and gripper status
    if gripper_status:
        lines.append(f"SR:{success_rate:.0f}% Grip:{gripper_status}")
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.38
    thickness = 1
    line_h    = 16
    x0, y0    = 6, 6
    box_w     = 160
    box_h     = len(lines) * line_h + 8

    # ── Light semi-transparent background (alpha 0.25) ──
    roi = frame[y0:y0+box_h, x0:x0+box_w].astype(np.float32)
    roi *= 0.25
    frame[y0:y0+box_h, x0:x0+box_w] = roi.astype(np.uint8)

    for i, line in enumerate(lines):
        y = y0 + 13 + i * line_h
        cv2.putText(frame, line, (x0 + 4, y), font, scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (x0 + 4, y), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def _overlay_bar_chart(frame, firing_rates, top_k=8):
    """Top-right bar chart: top-8 most active channel firing rates (real-time).

    Uses blended relative/absolute normalization so bar heights reflect
    both per-frame dynamics and absolute firing rate magnitude. Top-K
    channels are re-sorted each frame.

    Args:
        frame: (H, W, 3) uint8 image.
        firing_rates: (64,) per-channel mean firing rate from spike detection.
    """
    h, w = frame.shape[:2]

    # Re-select top-k channels each frame (sorted by current firing rate)
    top_idx = np.argsort(firing_rates)[-top_k:][::-1]
    top_vals = firing_rates[top_idx]

    # Chart region parameters (compact layout)
    chart_w    = 110                   # Total chart width
    chart_h    = 55                    # Total chart height
    margin_r   = 6                     # Right margin
    margin_t   = 6                     # Top margin
    x0         = w - chart_w - margin_r
    y0         = margin_t
    bar_w      = chart_w // top_k      # Per-bar width

    # ── Light semi-transparent background ──
    overlay_region = frame[y0:y0+chart_h, x0:x0+chart_w].astype(np.float32)
    overlay_region *= 0.45
    frame[y0:y0+chart_h, x0:x0+chart_w] = overlay_region.astype(np.uint8)

    # Dynamic normalization: blend relative (per-frame spread) with absolute
    # (global reference) to amplify frame-to-frame dynamics while staying stable
    val_min   = top_vals.min()
    val_max   = top_vals.max()
    val_range = max(val_max - val_min, 1.0)   # Avoid division by zero
    usable_h  = chart_h - 16                  # Leave room for channel label

    for i, val in enumerate(top_vals):
        # Relative + absolute blended normalization
        relative = (val - val_min) / val_range               # 0-1 per-frame spread
        absolute = np.clip((val - 50.0) / 350.0, 0.0, 1.0)  # Global reference scale
        normed   = 0.6 * relative + 0.4 * absolute           # Weight relative higher
        normed   = max(0.08, normed)                          # Minimum visible height
        bar_h    = max(2, int(normed * usable_h))
        bx  = x0 + i * bar_w + 2
        by  = y0 + chart_h - bar_h - 12
        bx2 = bx + bar_w - 4

        # Bar color: green → yellow → red gradient (by absolute rate)
        color = (
            int(255 * min(normed * 2, 1.0)),          # R: low→0, high→255
            int(255 * min((1.0 - normed) * 2, 1.0)),  # G: low→255, high→0
            30,                                         # B: fixed
        )
        cv2.rectangle(frame, (bx, by), (bx2, y0 + chart_h - 12), color, -1)

        # Channel number label (white, below bar)
        label = str(top_idx[i])
        cv2.putText(frame, label, (bx, y0 + chart_h - 1),
                    cv2.FONT_HERSHEY_PLAIN, 0.65, (220, 220, 220), 1, cv2.LINE_AA)


def _overlay_reward_curve(frame, reward_history):
    """Bottom rolling reward curve (red polyline).

    Draws a line plot of the last 50 step rewards at the bottom of the frame.

    Args:
        frame: (H, W, 3) uint8 image.
        reward_history: deque of recent per-step rewards (max 50).
    """
    if len(reward_history) < 2:
        return

    h, w = frame.shape[:2]

    # Chart region parameters
    chart_h   = 50                     # Curve region height
    margin_b  = 6                      # Bottom margin
    margin_lr = 30                     # Left-right margin
    y_top     = h - chart_h - margin_b
    y_bot     = h - margin_b
    x_left    = margin_lr
    x_right   = w - margin_lr

    # ── Semi-transparent background ──
    roi = frame[y_top:y_bot, x_left:x_right].astype(np.float32)
    roi *= 0.35
    frame[y_top:y_bot, x_left:x_right] = roi.astype(np.uint8)

    # ── Map reward values to pixel coordinates ──
    data = np.array(reward_history, dtype=np.float32)
    r_min, r_max = data.min(), data.max()
    if r_max - r_min < 1e-6:
        r_max = r_min + 1.0           # Avoid flat-line division by zero
    chart_pixel_w = x_right - x_left
    chart_pixel_h = y_bot - y_top - 4  # 2px padding top/bottom

    n = len(data)
    xs = np.linspace(0, chart_pixel_w - 1, n).astype(int) + x_left
    ys = y_bot - 2 - ((data - r_min) / (r_max - r_min) * chart_pixel_h).astype(int)

    # ── Draw red polyline ──
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=False,
                  color=(255, 80, 80), thickness=2, lineType=cv2.LINE_AA)

    # ── Zero-line reference (simulated dashed grey line) ──
    if r_min < 0 < r_max:
        zero_y = int(y_bot - 2 - ((0 - r_min) / (r_max - r_min) * chart_pixel_h))
        for x in range(x_left, x_right, 8):
            cv2.line(frame, (x, zero_y), (min(x + 4, x_right), zero_y),
                     (120, 120, 120), 1)


# ── Combined overlay entry point ──

def draw_overlay(frame, ep, reward, pdi, min_health, firing_rates, reward_history,
                 distance=0.0, success_rate=0.0, gripper_status=""):
    """Draw all real-time HUD overlays on a MuJoCo-rendered frame.

    Modifies frame in-place for minimal memory overhead.

    Args:
        frame: (H, W, 3) uint8 MuJoCo rgb_array rendered frame.
        ep: Current episode number.
        reward: Current episode cumulative reward.
        pdi: Current Physical Disturbance Index value.
        min_health: Minimum channel health across all 64 channels.
        firing_rates: (64,) per-channel firing rates.
        reward_history: deque of last 50 per-step rewards.
        distance: Current distance to goal.
        success_rate: Current episode success rate (%).
        gripper_status: Gripper state string ("OPEN" / "CLOSED").
    """
    # Ensure frame is writable and C-contiguous (MuJoCo frames may not be)
    frame = np.ascontiguousarray(frame)
    _overlay_text(frame, ep, reward, pdi, min_health, distance=distance,
                  success_rate=success_rate, gripper_status=gripper_status)
    _overlay_bar_chart(frame, firing_rates)
    _overlay_reward_curve(frame, reward_history)
    return frame


# ═══ CL1 Biological Agent ═══
#
# Per-step pipeline:
#   VIE encode → read spikes → PDI → FEP boost → antagonistic decode → execute
#
# Key features:
#   - Dopamine reward injection: structured burst on top-K calibrated channels
#   - Channel calibration: 10s warm-up to rank channel responsiveness

class CL1Agent:
    """CL1 biological neural agent for FetchPickAndPlace control."""

    def __init__(self, env, neurons, channel_ranking=None):
        self.env     = env
        self.neurons = neurons
        self.action_dim = env.action_space.shape[0]  # 4 for FetchPickAndPlace

        self.vie     = VIE(neurons, env=env)  # Pass env for Doom-style visual encoding
        self.decoder = AntagonisticDecoder(self.action_dim, action_scale=ACTION_SCALE)
        self.pdi     = PDI()

        self.episode_rewards = []
        self.lr          = 0.015
        self.action_bias = np.zeros(self.action_dim)
        self.best_reward = -np.inf
        self.best_bias   = np.zeros(self.action_dim)

        # Channel calibration results — used for dopamine injection targeting
        if channel_ranking is not None:
            self.top_channels = channel_ranking[:DOPAMINE_TOP_K].tolist()
        else:
            self.top_channels = list(range(DOPAMINE_TOP_K))

    def _detect_spikes(self):
        """Read 10ms of neural data and detect spiking channels.

        Returns:
            Tuple of:
                - spike_channels: List of channel indices that exceeded the
                  99.5th percentile threshold.
                - firing_rates: (64,) per-channel mean absolute amplitude,
                  used for HUD overlay visualization.
        """
        frames = self.neurons.read(250, None)  # 250 frames ~ 10ms @ 25kHz
        threshold = np.percentile(frames, 99.5)
        spike_mask = frames > threshold
        spike_channels = list(set(np.where(spike_mask)[1]))
        # Per-channel mean firing rate (lightweight computation for HUD)
        firing_rates = np.mean(np.abs(frames.astype(np.float32)), axis=0)
        return spike_channels, firing_rates

    def _dopamine_inject(self, reward):
        """Dopamine-like reward injection — positive reinforcement pathway.

        When reward > 0, delivers structured burst stimulation to the top-K
        calibrated channels. Under the FEP framework, positive reward indicates
        correct prediction — reinforce the current neural circuit.

        Args:
            reward: Step reward from the environment.
        """
        if reward <= 0:
            return
        # Stronger reward → stronger stimulation intensity
        amp = np.clip(reward * 2.0, 0.5, 3.0)
        stim = StimDesign(200, -amp, 200, amp)  # Wider pulse for dopamine burst
        burst = BurstDesign(DOPAMINE_BURST_N, DOPAMINE_BURST_HZ)
        self.neurons.stim(ChannelSet(*self.top_channels), stim, burst)

    def run_episode(self, max_steps=CL1_MAX_STEPS, record=False, ep_num=0):
        """Run a single episode of CL1 biological control."""
        obs_dict, _ = self.env.reset()
        obs  = obs_dict["observation"]
        goal = obs_dict["desired_goal"]

        self.pdi.reset()
        self.decoder.reset()
        total_reward = 0.0
        frames = []
        # Success tracking
        episode_successes = []
        # Rolling reward history (last 50 steps, for overlay curve)
        step_rewards = deque(maxlen=50)
        # Current firing rates cache for HUD
        cur_firing_rates = np.zeros(64)

        for step in range(max_steps):
            # 1. VIE encode observation → stimulate neurons
            self.vie.encode(obs, goal)

            # 2. Read neural response (also get firing rates for HUD overlay)
            spikes, cur_firing_rates = self._detect_spikes()

            # 3. PDI: compute motion stability
            vel = obs[3:6] if len(obs) >= 6 else np.zeros(3)
            self.pdi.update(vel)
            pdi_val = self.pdi.compute()

            # 4. FEP: PDI → exploration boost
            fep_boost = pdi_val * 0.4

            # 5. Antagonistic decode → action
            raw = self.decoder.decode(spikes, pdi_boost=fep_boost)
            action = np.clip(raw + self.action_bias, -1.0, 1.0)

            # 6. Execute action in environment
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            obs  = obs_dict["observation"]
            goal = obs_dict["desired_goal"]
            total_reward += reward

            # Real-time distance computation (for HUD overlay)
            cur_distance = float(np.linalg.norm(goal - obs[:3]))

            # Success check: object-to-goal distance < 0.05m and gripper closed
            object_pos = obs[3:6] if len(obs) >= 6 else obs[:3]
            obj_goal_dist = float(np.linalg.norm(goal - object_pos))
            gripper_open = obs[9:11] if len(obs) >= 11 else np.array([0.0, 0.0])
            gripper_is_closed = float(np.mean(gripper_open)) < 0.035
            success = 1 if (obj_goal_dist < 0.05 and gripper_is_closed) else 0
            episode_successes.append(success)

            # Record per-step reward (for HUD rolling curve)
            step_rewards.append(reward)

            # 7. Dopamine-like reward injection (reward > 0 → reinforce active pathway)
            self._dopamine_inject(reward)

            # 8. Record video frame with HUD overlay
            if record:
                frame = self.env.render()
                if frame is not None:
                    # Get minimum channel health for display
                    min_health = 1.0
                    if hasattr(self.neurons, 'get_health'):
                        min_health = float(self.neurons.get_health().min())
                    # Compute current success rate and gripper status for HUD
                    cur_success_rate = np.mean(episode_successes) * 100 if episode_successes else 0.0
                    grip_status = "CLOSED" if gripper_is_closed else "OPEN"
                    # Draw overlays on frame
                    frame = draw_overlay(
                        frame, ep_num, total_reward, pdi_val,
                        min_health, cur_firing_rates, step_rewards,
                        distance=cur_distance,
                        success_rate=cur_success_rate,
                        gripper_status=grip_status
                    )
                    frames.append(frame)

            # 9. Reward-modulated learning (Hebbian + dopamine synergy)
            if reward > -0.5:
                self.action_bias += self.lr * action * (reward + 1.0)
                self.action_bias = np.clip(self.action_bias, -0.5, 0.5)

            if terminated or truncated:
                break

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_bias   = self.action_bias.copy()

        # Compute episode success rate
        success_rate = np.mean(episode_successes) * 100 if episode_successes else 0.0
        return total_reward, self.pdi.compute(), frames, success_rate

    def train(self, num_episodes=CL1_EPISODES, record_last_n=RECORD_LAST_N):
        """Train the CL1 biological agent, recording only the last N mature episodes."""
        print("\n" + "=" * 60)
        print("  CL1 Bio-Computer Training")
        print("=" * 60)
        print(f"  Episodes: {num_episodes} | Env: {ENV_NAME}")
        backend = "Cortical Labs cl-sdk" if CL_AVAILABLE else "Built-in mock"
        print(f"  Backend:  {backend}")
        print(f"  Modules:  VIE + Antagonistic Decoder + PDI/FEP + Dopamine Injection")
        print(f"  Record:   last {record_last_n} mature episodes\n")

        all_frames = []
        # Success rate tracking
        all_success_rates = []
        # Record only last N mature episodes (skip early random movements)
        record_start = max(0, num_episodes - record_last_n)

        pbar = tqdm(range(num_episodes), desc="CL1", ncols=90)
        for ep in pbar:
            rec = (ep >= record_start)
            reward, pdi_val, frames, success_rate = self.run_episode(record=rec, ep_num=ep)
            self.episode_rewards.append(reward)
            all_success_rates.append(success_rate)
            if rec:
                all_frames.extend(frames)

            avg = np.mean(self.episode_rewards[-20:])
            pbar.set_postfix(R=f"{reward:.1f}", avg20=f"{avg:.1f}",
                             PDI=f"{pdi_val:.2f}", SR=f"{success_rate:.0f}%")

            # Metabolic guardrail health check
            if hasattr(self.neurons, 'get_health') and (ep + 1) % 100 == 0:
                health = self.neurons.get_health()
                min_h = health.min()
                tqdm.write(
                    f"  [Health] Episode {ep+1}: min_channel_health={min_h:.3f} "
                    f"{'OK' if min_h > 0.5 else 'WARNING: low health!'}"
                )

            if (ep + 1) % 50 == 0:
                avg_sr = np.mean(all_success_rates[-50:])
                tqdm.write(
                    f"  CL1 Episode {ep+1}: Reward {reward:.1f} "
                    f"(avg20={avg:.1f}, PDI={pdi_val:.2f}, SuccessRate={avg_sr:.1f}%)"
                )

        final = np.mean(self.episode_rewards[-20:])
        final_sr = np.mean(all_success_rates[-20:])
        print(f"\n  CL1 Training Done | Final avg20: {final:.2f} | Final SuccessRate: {final_sr:.1f}%")
        self.all_success_rates = all_success_rates
        return all_frames

# ═══ PPO Baseline ═══

def train_ppo_baseline(record_last_n=RECORD_LAST_N):
    """Train and evaluate PPO baseline, recording last N episode videos."""
    from stable_baselines3 import PPO

    print("\n" + "=" * 60)
    print("  PPO Baseline Training")
    print("=" * 60)
    print(f"  Training: {PPO_TIMESTEPS} steps | Eval: {PPO_EVAL_EPS} episodes\n")

    train_env = gym.make(ENV_NAME, reward_type=REWARD_TYPE)
    train_env = gym.wrappers.FlattenObservation(train_env)

    eval_env = gym.make(ENV_NAME, reward_type=REWARD_TYPE, render_mode="rgb_array")
    eval_env = gym.wrappers.FlattenObservation(eval_env)

    model = PPO("MlpPolicy", train_env, verbose=0,
                n_steps=256, batch_size=64, learning_rate=3e-4)

    all_rewards = []
    all_frames  = []
    n_chunks = 30
    chunk_steps    = PPO_TIMESTEPS // n_chunks
    evals_per_chunk = PPO_EVAL_EPS // n_chunks   # 10

    record_start = PPO_EVAL_EPS - record_last_n  # Record only last N episodes
    ep_count = 0

    pbar = tqdm(range(n_chunks), desc="PPO ", ncols=90)
    for chunk in pbar:
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)

        for _ in range(evals_per_chunk):
            obs, _ = eval_env.reset()
            ep_r, done = 0.0, False
            ep_frames = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = eval_env.step(action)
                ep_r += r
                done = term or trunc
                if ep_count >= record_start:
                    frame = eval_env.render()
                    if frame is not None:
                        ep_frames.append(frame)
            all_rewards.append(ep_r)
            if ep_count >= record_start:
                all_frames.extend(ep_frames)
            ep_count += 1

        avg = np.mean(all_rewards[-evals_per_chunk:])
        pbar.set_postfix(avg_R=f"{avg:.1f}", eps=len(all_rewards))

    train_env.close()
    eval_env.close()

    final = np.mean(all_rewards[-20:])
    print(f"\n  PPO Training Done | Final avg20: {final:.2f}")
    return all_rewards, all_frames

# ═══ Random Baseline ═══

def run_random_baseline(num_episodes=RANDOM_EPISODES):
    """Pure random actions baseline — verifies CL1 and PPO are actually learning."""
    print("\n" + "=" * 60)
    print("  Random Agent Baseline")
    print("=" * 60)
    print(f"  Episodes: {num_episodes}\n")

    env = gym.make(ENV_NAME, reward_type=REWARD_TYPE)
    all_rewards = []

    pbar = tqdm(range(num_episodes), desc="RNG ", ncols=90)
    for ep in pbar:
        obs_dict, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_dict, r, term, trunc, _ = env.step(action)
            total_reward += r
            done = term or trunc
        all_rewards.append(total_reward)

        avg = np.mean(all_rewards[-20:])
        pbar.set_postfix(R=f"{total_reward:.1f}", avg20=f"{avg:.1f}")

    env.close()
    final = np.mean(all_rewards[-20:])
    print(f"\n  Random Baseline Done | Final avg20: {final:.2f}")
    return all_rewards

# ═══ Video Utilities — now from core/video.py ═══
# save_video() and make_side_by_side() imported at top from core.video

# ═══ Learning Curve Plot ═══

def plot_learning_curves(cl1_rewards, ppo_rewards, random_rewards, path=PLOT_FILE,
                         cl1_success_rates=None):
    """Plot three learning curves: CL1 (bio) + PPO (RL) + Random, with optional success rate."""
    # Font configuration for cross-platform compatibility
    for font in ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']:
        try:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(13, 6.5))

    def rolling(data, w=20):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w) / w, mode='valid')

    cl1_s = rolling(cl1_rewards)
    ppo_s = rolling(ppo_rewards)
    rnd_s = rolling(random_rewards)
    off = 19  # rolling window offset

    # Raw data (transparent background traces)
    ax.plot(cl1_rewards, alpha=0.12, color='#e74c3c')
    ax.plot(ppo_rewards, alpha=0.12, color='#3498db')
    ax.plot(random_rewards, alpha=0.12, color='#95a5a6')

    # Smoothed curves (rolling average)
    ax.plot(range(off, off + len(cl1_s)), cl1_s,
            color='#e74c3c', lw=2.5,
            label='CL1 Bio-Computer (VIE + Antagonistic + Dopamine)')
    ax.plot(range(off, off + len(ppo_s)), ppo_s,
            color='#3498db', lw=2.5,
            label='PPO Traditional RL')
    ax.plot(range(off, off + len(rnd_s)), rnd_s,
            color='#95a5a6', lw=2.0, linestyle='--',
            label='Random Agent (baseline)')

    ax.set_title(
        "Project Senxe v3.0 — Ready for Real CL1\n"
        "CL1 Bio-Computer vs PPO vs Random: PickAndPlace Sample Efficiency",
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(len(cl1_rewards), len(ppo_rewards), len(random_rewards)))

    # Second Y-axis for success rate
    if cl1_success_rates is not None and len(cl1_success_rates) > 0:
        ax2 = ax.twinx()
        sr_smooth = rolling(cl1_success_rates)
        ax2.plot(range(off, off + len(sr_smooth)), sr_smooth,
                 color='#2ecc71', lw=2.0, linestyle='-.',
                 label='CL1 Success Rate (%)')
        ax2.set_ylabel("Success Rate (%)", fontsize=12, color='#2ecc71')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.legend(fontsize=9, loc='center right')

    # Annotation box with experiment details
    ax.text(0.02, 0.95,
            f"Senxe v3.0 — Doom-style visual + pick-and-place task\n"
            f"CL1: {CL1_EPISODES} eps (online, bio-inspired)\n"
            f"PPO: {PPO_TIMESTEPS} pre-train -> {PPO_EVAL_EPS} eval\n"
            f"Random: {RANDOM_EPISODES} eps (pure random)\n"
            f"---\n"
            f"FetchPickAndPlace-v4 (visual object encoding)\n"
            f"Real CL1 expected to show superior sample efficiency",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Learning curve saved: {path}")

# ═══ Main Entry Point ═══

def main():
    print("+" + "=" * 58 + "+")
    print("|  Project Senxe v3.0 -- Ready for Real CL1               |")
    print("|  CL1 Bio-Computer x PickAndPlace Control                 |")
    print("+" + "=" * 58 + "+\n")

    if CL_AVAILABLE:
        print("  CL SDK: Cortical Labs cl-sdk (real/simulator)")
    else:
        print("  CL SDK unavailable -- using built-in mock")
        print("     (pip install cl-sdk to use real CL1 backend)")
    print(f"  Env: {ENV_NAME} | reward_type={REWARD_TYPE}\n")

    # Phase 0: Channel Warm-up Calibration (10 sec)
    print("-" * 60)
    print("  Phase 0: Channel Warm-up Calibration")
    print("-" * 60)

    with cl_open() as neurons:
        channel_ranking, responsiveness = warmup_calibration(neurons, WARMUP_SECONDS)

        # Phase 1: CL1 Training
        env = gym.make(ENV_NAME, render_mode="rgb_array", reward_type=REWARD_TYPE)
        agent = CL1Agent(env, neurons, channel_ranking=channel_ranking)
        cl1_frames = agent.train(
            num_episodes=CL1_EPISODES, record_last_n=RECORD_LAST_N
        )
        cl1_rewards = agent.episode_rewards.copy()
        cl1_success_rates = agent.all_success_rates.copy()
        env.close()

    # Phase 2: PPO Baseline (with video recording)
    ppo_rewards, ppo_frames = train_ppo_baseline(record_last_n=RECORD_LAST_N)

    # Phase 3: Random Baseline
    random_rewards = run_random_baseline(num_episodes=RANDOM_EPISODES)

    # Phase 4: Save Videos
    print("\n" + "-" * 60)
    print("  Phase 4: Generating Videos")
    print("-" * 60)

    # Video 1: CL1 standalone
    save_video(cl1_frames, VIDEO_CL1, target_seconds=20)

    # Video 2: Side-by-side comparison (left: CL1, right: PPO)
    make_side_by_side(cl1_frames, ppo_frames, VIDEO_SIDE, fps=VIDEO_FPS,
                      left_label="CL1 Bio-Computer",
                      right_label="PPO Traditional RL",
                      center_label="PickAndPlace")

    # Phase 5: Plot Learning Curves
    print()
    plot_learning_curves(cl1_rewards, ppo_rewards, random_rewards,
                         cl1_success_rates=cl1_success_rates)

    # Done
    print("\n" + "=" * 60)
    print("  Project Senxe v3.0 Demo Complete!")
    print("=" * 60)
    print(f"  Video (CL1):       {VIDEO_CL1}")
    print(f"  Video (side-by-side): {VIDEO_SIDE}")
    print(f"  Plot:              {PLOT_FILE}")
    print(f"  CL1 final avg20:   {np.mean(cl1_rewards[-20:]):.2f}")
    print(f"  PPO final avg20:   {np.mean(ppo_rewards[-20:]):.2f}")
    print(f"  Random final avg20: {np.mean(random_rewards[-20:]):.2f}")
    print(f"  CL1 final SR:      {np.mean(cl1_success_rates[-20:]):.1f}%")
    print()
    print("  Switch to real CL1 hardware:")
    print("    pip install cl-sdk   # Auto-detected, zero code changes!")
    print()


if __name__ == "__main__":
    main()
