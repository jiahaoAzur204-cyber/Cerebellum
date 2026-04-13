#!/usr/bin/env python3
"""
Project Senxe v4.0 — RoboSuite NutAssembly (Native Force/Torque Sensors)
=========================================================================
CL1 Bio-Computer vs PPO vs Random — Industrial Assembly Sample Efficiency

Benchmark comparing biological neural control (Cortical Labs CL1) against
PPO reinforcement learning and random baselines on the RoboSuite NutAssembly
task with a Franka Panda robot arm and native force/torque sensors.

Usage:  python senxe_demo_robosuite.py
Output: cl1_nutassembly.mp4, side_by_side_nutassembly.mp4, learning_curve_nutassembly.png

To use real CL1 hardware: pip install cl-sdk  (auto-detected, zero code changes)
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, imageio, cv2
from tqdm import tqdm
from collections import deque
from contextlib import contextmanager

# ═══ Core Modules (shared with v3.0) ═══
from core.neurons import (
    cl_open, warmup_calibration,
    ChannelSet, StimDesign, BurstDesign,
    MockNeurons, CL_AVAILABLE,
)
from core.decoder import AntagonisticDecoder
from core.pdi import PDI
from core.curiosity import NeuralCuriosity
from core.video import save_video, make_side_by_side

# ═══ Configuration ═══
ENV_NAME        = "NutAssembly"
ROBOT           = "Panda"
CL1_EPISODES    = 200;  CL1_MAX_STEPS = 200
PPO_TIMESTEPS   = 20_000;  PPO_EVAL_EPS = 200
RANDOM_EPISODES = 200
VIDEO_FPS       = 30
VIDEO_CL1       = "cl1_nutassembly.mp4"
VIDEO_SIDE      = "side_by_side_nutassembly.mp4"
PLOT_FILE       = "learning_curve_nutassembly.png"
RECORD_LAST_N   = 80;  WARMUP_SECONDS = 10;  ACTION_SCALE = 0.25
RENDER_W        = 720;  RENDER_H = 720        # Render resolution
INSERTION_DEPTH_THRESHOLD = 0.02
FORCE_SAFETY_THRESHOLD    = 20.0
TORQUE_SAFETY_THRESHOLD   = 5.0
DOPAMINE_TOP_K  = 8;  DOPAMINE_BURST_N = 15;  DOPAMINE_BURST_HZ = 300

# ═══ RoboSuite Environment ═══
def make_robosuite_env(render=False):
    """Create RoboSuite NutAssembly environment with native force/torque sensors."""
    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    raw = suite.make(ENV_NAME, robots=ROBOT, has_renderer=False,
                     has_offscreen_renderer=render, use_camera_obs=False,
                     render_camera="frontview", horizon=CL1_MAX_STEPS, reward_shaping=True)
    return GymWrapper(raw), raw

def extract_obs(obs):
    """Extract force, torque, position, and goal vectors from the observation."""
    if isinstance(obs, dict):
        eef = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64).flatten()[:3]
        vel = np.array(obs.get("robot0_eef_vel_lin", obs.get("robot0_eef_vel", np.zeros(3))), dtype=np.float64).flatten()[:3]
        frc = np.array(obs.get("robot0_eef_force", obs.get("ft_force", obs.get("robot0_force", np.zeros(3)))), dtype=np.float64).flatten()[:3]
        trq = np.array(obs.get("robot0_eef_torque", obs.get("ft_torque", obs.get("robot0_torque", np.zeros(3)))), dtype=np.float64).flatten()[:3]
        p2h = np.array(obs.get("peg_to_hole", obs.get("hole_pos", np.zeros(3)) - eef), dtype=np.float64).flatten()[:3]
        jnt = np.array(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float64).flatten()
        return dict(eef_pos=eef, eef_vel=vel, force=frc, torque=trq, peg_to_hole=p2h, joint_pos=jnt)
    o = np.array(obs, dtype=np.float64).flatten(); n = len(o)
    return dict(eef_pos=o[:3] if n>=3 else np.zeros(3), eef_vel=o[3:6] if n>=6 else np.zeros(3),
                force=o[6:9] if n>=9 else np.zeros(3), torque=o[9:12] if n>=12 else np.zeros(3),
                peg_to_hole=o[12:15] if n>=15 else np.zeros(3), joint_pos=o[15:22] if n>=22 else np.zeros(7))

def compute_insertion_depth(info):
    d = np.linalg.norm(info["peg_to_hole"]); return max(0.0, 0.1 - d), d

# ═══ CL1 Neural Interface — now from core/ (see core/neurons.py) ═══

# ═══ VIE: Virtual Interference Encoding ═══
# v4.0: Native force/torque sensor encoding (industrial tactile feedback)
# CH 0-15: Force (rate coding) | CH 16-31: Torque/friction (traveling waves)
# CH 32-47: Position           | CH 48-63: Goal direction + insertion depth

class VIE:
    """Virtual Interference Encoding — maps physical sensor data to neural stimulation.

    Encodes force, torque, position, goal direction, and insertion depth onto
    a 64-channel MEA using biologically realistic coding schemes:

    - CH 0-15:  Force magnitude and per-axis force → rate coding
                (higher force → higher burst frequency, mimicking mechanoreceptors)
    - CH 16-31: Torque and friction → traveling wave temporal coding
                (phase-delayed pulses simulate proprioceptive spindle fibers)
    - CH 32-47: End-effector absolute position → position encoding
    - CH 48-55: Goal direction (peg-to-hole delta vector)
    - CH 56-63: Insertion depth progress encoding

    Includes online adaptive gain adjustment: channels with weak responses
    get amplified, over-responsive channels get attenuated.
    """

    CH_FORCE = list(range(0, 16)); CH_TORQUE = list(range(16, 32))
    CH_POSITION = list(range(32, 48)); CH_GOALDIR = list(range(48, 64))

    def __init__(self, neurons, raw_env=None):
        self.neurons = neurons; self.raw_env = raw_env
        self.channel_gain = np.ones(64)    # Per-channel encoding gain (online-adjusted)
        self.stim_history = np.zeros(64)   # Cumulative stimulation tracker
        self.response_history = np.zeros(64)  # Cumulative response tracker
        self.adaptation_rate = 0.005

    def encode(self, obs_info):
        """Encode observation into neural stimulation patterns on the 64-ch MEA.

        Converts force/torque/position/goal sensor readings into charge-balanced
        biphasic pulse trains delivered across the channel groups. Force uses
        rate coding (burst frequency proportional to magnitude), while torque
        uses traveling-wave temporal coding with phase delays.

        Args:
            obs_info: Dictionary with keys 'eef_pos', 'eef_vel', 'force',
                      'torque', 'peg_to_hole' from extract_obs().
        """
        eef_pos = obs_info["eef_pos"]; eef_vel = obs_info["eef_vel"]
        force = obs_info["force"]; torque = obs_info["torque"]
        peg_to_hole = obs_info["peg_to_hole"]
        distance = np.linalg.norm(peg_to_hole)
        direction = peg_to_hole / (distance + 1e-8)
        stim = StimDesign(160, -1.0, 160, 1.0)

        # ══ Native force/torque feedback (rate + temporal coding) ══

        # Force → Rate Coding (CH 0-15)
        force_mag = np.linalg.norm(force)
        fnorm = np.clip(force_mag / FORCE_SAFETY_THRESHOLD, 0.0, 1.5)
        fhz = int(np.clip(50 + 350 * fnorm, 50, 400))
        fn = max(1, min(10, int(fnorm * 8 * self.channel_gain[self.CH_FORCE[0]])))
        self.neurons.stim(ChannelSet(*self.CH_FORCE[:8]), stim, BurstDesign(fn, fhz))
        for ax in range(3):
            cb = self.CH_FORCE[8] + ax * 2
            chs = ChannelSet(*[cb, min(cb + 1, 15)])
            f = force[ax]; inten = np.clip(abs(f) / (FORCE_SAFETY_THRESHOLD / 3), 0.1, 2.0)
            fs = StimDesign(160, -inten * np.sign(f), 160, inten * np.sign(f))
            fb = BurstDesign(max(1, int(abs(f) / 3)), int(50 + abs(f) * 15))
            self.neurons.stim(chs, fs, fb)

        # Light temporal coding (traveling waves) for dynamic force feedback
        # Phase delays (1-5ms) between CH 8-15 simulate sliding/dynamic force transients
        if force_mag > 0.05:
            for wave_i in range(8):
                ch_idx = self.CH_FORCE[8 + wave_i]
                phase_delay_ms = 1.0 + 4.0 * (wave_i / 7.0)          # 1-5ms gradient
                wave_freq = int(np.clip(40 + 160 * fnorm, 40, 200))   # scale with force
                wave_amp = np.clip(fnorm * 0.6, 0.05, 1.2)
                wave_stim = StimDesign(
                    int(160 + phase_delay_ms * 10), -wave_amp,
                    int(160 + phase_delay_ms * 10),  wave_amp
                )
                self.neurons.stim(ChannelSet(ch_idx), wave_stim, BurstDesign(1, wave_freq))

        # Torque/Friction → Traveling Waves (CH 16-31)
        tmag = np.linalg.norm(torque)
        if tmag > 0.01:
            for ax in range(3):
                t = torque[ax]
                if abs(t) > 0.01:
                    cb = self.CH_TORQUE[0] + ax * 5
                    chs = ChannelSet(*range(cb, min(cb + 5, 32)))
                    inten = np.clip(abs(t) * 3.0 * self.channel_gain[cb], 0.1, 2.0)
                    ws = StimDesign(160, -inten, 160, inten)
                    whz = int(np.clip(60 * abs(t), 20, 200))
                    self.neurons.stim(chs, ws, BurstDesign(2, whz))

        # Position Encoding (CH 32-47)
        for ax in range(3):
            cb = self.CH_POSITION[0] + ax * 5
            chs = ChannelSet(*range(cb, min(cb + 5, 48)))
            p = eef_pos[ax]; g = self.channel_gain[cb]
            phz = int(np.clip((100 + 200 * abs(p)) * g, 50, 350))
            ps = StimDesign(160, -abs(p) * 0.8 * g, 160, abs(p) * 0.8 * g)
            self.neurons.stim(chs, ps, BurstDesign(2, phz))

        # Goal Direction (CH 48-55)
        for ax in range(3):
            cb = self.CH_GOALDIR[0] + ax * 2
            chs = ChannelSet(*[cb, min(cb + 1, 55)])
            d = direction[ax]; inten = np.clip((abs(d) * 1.5 + 0.1) * self.channel_gain[cb], 0.1, 2.0)
            ds = StimDesign(160, -inten * np.sign(d), 160, inten * np.sign(d))
            db = BurstDesign(max(1, int(abs(d) * 5)), int(50 + abs(d) * 100))
            self.neurons.stim(chs, ds, db)

        # Insertion Depth (CH 56-63)
        depth, _ = compute_insertion_depth(obs_info)
        dn = np.clip(depth / INSERTION_DEPTH_THRESHOLD, 0.0, 2.0)
        dg = np.mean(self.channel_gain[self.CH_GOALDIR[8:16]])
        dhz = int(np.clip((50 + 300 * dn) * dg, 50, 400)); dnn = max(1, int(dn * 6 * dg))
        dstim = StimDesign(160, -0.8, 160, 0.8)
        self.neurons.stim(ChannelSet(*self.CH_GOALDIR[8:16]), dstim, BurstDesign(dnn, dhz))

        # Velocity supplement (reuse CH_TORQUE tail)
        vmag = np.linalg.norm(eef_vel)
        if vmag > 0.003:
            for ax in range(3):
                v = eef_vel[ax]
                if abs(v) > 0.003:
                    cb = min(self.CH_TORQUE[0] + 15 + ax, 31)
                    vi = np.clip(abs(v) * 5, 0.1, 2.0)
                    vs = StimDesign(160, -vi, 160, vi)
                    vhz = int(np.clip(60 * abs(v), 20, 200))
                    self.neurons.stim(ChannelSet(cb), vs, BurstDesign(2, vhz))

    def adapt(self, firing_rates):
        """Online adaptation of channel encoding gains.

        Implements homeostatic gain control: under-responsive channels get
        amplified, over-responsive channels get attenuated. Target is uniform
        response (~0.5 normalized) across all channels.

        Args:
            firing_rates: Per-channel firing rates, shape (64,).
        """
        fr_norm = firing_rates / (firing_rates.max() + 1e-6)
        # Target: uniform response across channels (~0.5 normalized)
        error = 0.5 - fr_norm
        self.channel_gain += self.adaptation_rate * error
        self.channel_gain = np.clip(self.channel_gain, 0.3, 3.0)

# ═══ AntagonisticDecoder, PDI, NeuralCuriosity — now from core/ ═══
# (see core/decoder.py, core/pdi.py, core/curiosity.py)

# ═══ HUD Overlay System ═══
# v5.0 Cold Cyberpunk HUD — Bloom + EMA + Breathing + Zero Matplotlib
# Inspired by native macOS AppKit fluid aesthetics (Sense_CL1_Integrated.py)
# ═══════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE:
#   1. All high-intensity elements are drawn onto a separate black `glow_layer`.
#   2. The glow_layer is Gaussian-blurred, then additively blended back
#      onto the main frame — producing a hardware-accelerated-looking bloom.
#   3. The force gauge cursor uses EMA smoothing for fluid "inertia" motion.
#   4. Critical UI labels pulse via `np.sin(time * freq)` for a "breathing" feel.
#   5. The evolution heatmap is a pure cv2 scrolling sparkline — zero Matplotlib.
# ═══════════════════════════════════════════════════════════════════════

import time as _time  # for breathing pulse, safe re-import

# ── Module-level persistent state ──
_overlay_frame_counter = [0]                # Global frame tick
_particle_pool = []                         # Particle system pool
_last_spike_time = np.zeros(64)             # Per-channel last-spike timestamp
_force_ema = [0.0]                          # EMA-smoothed force magnitude for gauge
_FORCE_EMA_ALPHA = 0.18                     # EMA coefficient: lower = smoother glide

# ── Channel Evolution — pure cv2 sparkline (replaces Matplotlib heatmap) ──
# Accumulator: list of (64,) firing rate arrays, one per completed episode.
# Referenced externally by CL1Agent.run_episode — DO NOT RENAME.
_episode_firing_history = []
_evolution_cache = [None, 0]                # [cached_image, last_update_ep_count]

# ── Cold Cyberpunk 4-color semantic palette (RGB order) ──
# Force=Ice Blue | Torque=Neon Cyan | Position=Magenta | Goal=Muted Amber
_GROUP_COLORS = {
    'force':    {'echo': (255, 255, 255), 'active': (180, 210, 255), 'inactive': (20, 30, 50)},
    'torque':   {'echo': (255, 255, 255), 'active': (0,   255, 240), 'inactive': (5,  40, 38)},
    'position': {'echo': (255, 255, 255), 'active': (220, 80,  220), 'inactive': (38, 12, 38)},
    'goal':     {'echo': (255, 255, 255), 'active': (220, 185, 90),  'inactive': (40, 32, 12)},
}
_CH_GROUP_MAP = ['force'] * 16 + ['torque'] * 16 + ['position'] * 16 + ['goal'] * 16

# Bloom kernel size (must be odd). Larger = softer glow, more GPU-like feel.
_BLOOM_KSIZE = 31
_BLOOM_SIGMA = 12


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper: Drop-shadow text (military HUD typography)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _hud_text(target, text, x, y, color=(255, 255, 255),
              scale=0.35, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Render pixel-perfect HUD text with a 2px black drop shadow for depth."""
    # Shadow pass (1px down-right offset, thick black outline for contrast)
    cv2.putText(target, text, (x + 1, y + 1), font, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Foreground pass
    cv2.putText(target, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Particle system — spawn + update + draw onto glow_layer for bloom
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _spawn_particles(cx, cy, count, color_base, speed=1.8, life_range=(10, 22)):
    """Spawn radial particles around (cx, cy). Drawn onto glow_layer for bloom."""
    for _ in range(count):
        angle = np.random.uniform(0, 2 * np.pi)
        spd = np.random.uniform(0.4, speed)
        lf = np.random.randint(life_range[0], life_range[1])
        _particle_pool.append(dict(
            x=float(cx) + np.random.uniform(-3, 3),
            y=float(cy) + np.random.uniform(-3, 3),
            vx=np.cos(angle) * spd, vy=np.sin(angle) * spd,
            life=lf, max_life=lf, color_base=color_base
        ))


def _update_and_draw_particles(frame, glow_layer):
    """Advance physics & render particles onto glow_layer for additive bloom."""
    h, w = frame.shape[:2]
    alive = []
    for p in _particle_pool:
        p['x'] += p['vx']; p['y'] += p['vy']; p['life'] -= 1
        p['vx'] *= 0.93; p['vy'] *= 0.93          # drag
        if p['life'] <= 0:
            continue
        px, py = int(p['x']), int(p['y'])
        if px < 4 or py < 4 or px >= w - 4 or py >= h - 4:
            continue
        t = p['life'] / p['max_life']               # 1.0→0.0 fade
        cb = p['color_base']
        brightness = t * 0.85
        r = max(2, int(3 * t))
        color = (int(cb[0] * brightness),
                 int(cb[1] * brightness),
                 int(cb[2] * brightness))
        cv2.circle(glow_layer, (px, py), r + 1, color, -1, cv2.LINE_AA)
        alive.append(p)
    _particle_pool.clear()
    _particle_pool.extend(alive)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feathered darken (unchanged utility, used for subtle panel backdrops)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _feathered_darken(frame, y0, y1, x0, x1, darkness=0.08, feather=12):
    """Darken a rectangular ROI with feathered edges for a frosted-glass effect."""
    h, w = frame.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    rh, rw = y1 - y0, x1 - x0
    if rh <= 0 or rw <= 0:
        return
    alpha = np.ones((rh, rw), dtype=np.float32)
    f = min(feather, rh // 2, rw // 2)
    for i in range(f):
        t = (i + 1) / (f + 1)
        alpha[i, :] = np.minimum(alpha[i, :], t)
        alpha[rh - 1 - i, :] = np.minimum(alpha[rh - 1 - i, :], t)
        alpha[:, i] = np.minimum(alpha[:, i], t)
        alpha[:, rw - 1 - i] = np.minimum(alpha[:, rw - 1 - i], t)
    factor = darkness + (1.0 - darkness) * (1.0 - alpha)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    roi *= factor[:, :, np.newaxis]
    frame[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HUD Text Overlay — Military-grade precision typography
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _overlay_text(frame, ep, reward, pdi, min_health, glow_layer,
                  distance=0.0, force_mag=0.0, torque_mag=0.0,
                  depth=0.0, success_rate=0.0, force_safe_rate=0.0):
    """Minimalist military HUD — small scale, pure white, drop shadows.
    WARNING / DANGER text pulses with a sine-wave breathing effect
    and is drawn onto glow_layer for bloom."""
    h, w = frame.shape[:2]
    fc = _overlay_frame_counter[0]
    curr_time = fc / 30.0

    # ── Safety classification ──
    fn = force_mag / FORCE_SAFETY_THRESHOLD if FORCE_SAFETY_THRESHOLD > 0 else 0
    if fn < 0.5:
        status, s_color = "NOMINAL", (160, 220, 160)
    elif fn < 1.0:
        status, s_color = "CAUTION", (220, 200, 80)
    else:
        status, s_color = "DANGER", (255, 80, 80)

    # ── Top-left HUD block (3 lines, larger + brighter for 720p legibility) ──
    lx, ly = 12, 22
    line_h = 20  # increased vertical spacing for breathing room

    _hud_text(frame, f"EP {ep:03d}   R {reward:+.1f}", lx, ly, (255, 255, 255), 0.50)
    _hud_text(frame, f"F {force_mag:5.1f}N  T {torque_mag:4.2f}Nm  D {depth:.3f}m",
              lx, ly + line_h, (140, 240, 255), 0.42)
    _hud_text(frame, f"SR {success_rate:3.0f}%  FSR {force_safe_rate:3.0f}%  PDI {pdi:.2f}",
              lx, ly + line_h * 2, (100, 210, 230), 0.42)

    # ── Status badge — breathing pulse on WARNING/DANGER ──
    badge_x = lx + 300
    badge_y = ly
    if fn >= 0.5:
        # Breathing: sinusoidal alpha modulation (0.5–1.0 range)
        breath = 0.5 + 0.5 * np.sin(curr_time * 5.0)      # ~0.8 Hz pulse
        pulse_color = tuple(int(c * breath) for c in s_color)
        # Draw on glow_layer for bloom halo around warning text
        _hud_text(glow_layer, status, badge_x, badge_y, pulse_color, 0.40, 1)
    _hud_text(frame, status, badge_x, badge_y, s_color, 0.40, 1)

    # ── Bottom-left: minimal health indicator ──
    _hud_text(frame, f"HEALTH {min_health:.2f}", lx, h - 12, (100, 110, 120), 0.30)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8×8 Neuron Grid — Circles + Cold Cyberpunk + Bloom spikes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _overlay_neuron_grid(frame, firing_rates, min_health, glow_layer,
                         health_arr=None, force_mag=0.0):
    """8×8 neuron grid rendered as sleek circles with Cold Cyberpunk palette.
    Force=Ice Blue | Torque=Neon Cyan | Position=Magenta | Goal=Muted Amber.
    Spiking neurons flash WHITE with heavy colored bloom, then smoothly decay.
    Right-side force safety gauge with EMA-smoothed cursor."""
    h, w = frame.shape[:2]
    fc = _overlay_frame_counter[0]
    curr_time = fc / 30.0

    # ── Grid geometry ──
    radius = 6                              # circle radius (px)
    spacing = 16                            # center-to-center distance
    grid_n = 8
    total = grid_n * spacing                # 128px
    gauge_w = 6; gauge_gap = 10
    gx0, gy0 = 14, 86                      # origin (pushed down for breathing room below HUD)

    if gy0 + total + 5 > h or gx0 + total + gauge_gap + gauge_w + 5 > w:
        return

    # ── Spike detection + echo decay ──
    spike_thresh = np.percentile(firing_rates, 75)
    spiking = firing_rates > spike_thresh
    for ch in range(64):
        if spiking[ch]:
            _last_spike_time[ch] = curr_time

    echo_decay = np.zeros(64, dtype=np.float32)
    for ch in range(64):
        dt = curr_time - _last_spike_time[ch]
        # Smooth exponential decay over 0.45s — longer tail than before
        echo_decay[ch] = max(0.0, 1.0 - dt / 0.70)

    fr_median = np.median(firing_rates)
    fr_range = max(firing_rates.max() - firing_rates.min(), 1.0)

    # ── Pure black background behind the entire grid area ──
    pad = 4  # padding around the grid
    bg_x0 = max(0, gx0 - pad)
    bg_y0 = max(0, gy0 - pad)
    bg_x1 = min(w, gx0 + total + pad)
    bg_y1 = min(h, gy0 + total + pad)
    frame[bg_y0:bg_y1, bg_x0:bg_x1] = 0

    # ── Draw 8×8 circle grid ──
    for row in range(grid_n):
        for col in range(grid_n):
            ch = row * grid_n + col
            cx = gx0 + col * spacing + radius
            cy = gy0 + row * spacing + radius

            if cx + radius >= w or cy + radius >= h:
                continue

            group = _CH_GROUP_MAP[ch]
            colors = _GROUP_COLORS[group]
            ed = echo_decay[ch]
            fr = firing_rates[ch]

            if ed > 0.05:
                # ── STATE A: Spike echo — flash white → decay to active color ──
                ec = colors['echo']   # pure white
                ac = colors['active']
                t = ed                # 1.0 at spike, decays to 0
                cr = int(ec[0] * t + ac[0] * (1 - t))
                cg = int(ec[1] * t + ac[1] * (1 - t))
                cb_c = int(ec[2] * t + ac[2] * (1 - t))
                # Draw filled circle on frame
                cv2.circle(frame, (cx, cy), radius, (cr, cg, cb_c), -1, cv2.LINE_AA)
                # ── BLOOM: Draw high-intensity version onto glow_layer ──
                bloom_brightness = ed * 1.2
                bloom_color = (int(min(255, ac[0] * bloom_brightness)),
                               int(min(255, ac[1] * bloom_brightness)),
                               int(min(255, ac[2] * bloom_brightness)))
                cv2.circle(glow_layer, (cx, cy), radius + 5, bloom_color, -1, cv2.LINE_AA)
                # Spawn particles on recent spikes (ed > 0.65 for wider emission window)
                if ed > 0.65:
                    _spawn_particles(cx, cy, 3, ac, speed=2.0, life_range=(8, 16))

            elif fr > fr_median:
                # ── STATE B: Active — saturated group color, thin stroke ──
                ac = colors['active']
                intensity = np.clip((fr - fr_median) / (fr_range * 0.5 + 1e-6), 0, 1)
                alpha = 0.55 + intensity * 0.45
                cr = int(ac[0] * alpha)
                cg = int(ac[1] * alpha)
                cb_c = int(ac[2] * alpha)
                cv2.circle(frame, (cx, cy), radius, (cr, cg, cb_c), -1, cv2.LINE_AA)
                # Ultra-thin bright stroke for definition
                cv2.circle(frame, (cx, cy), radius, ac, 1, cv2.LINE_AA)
            else:
                # ── STATE C: Inactive — ghost circle, barely visible stroke ──
                ic = colors['inactive']
                cv2.circle(frame, (cx, cy), radius, ic, 1, cv2.LINE_AA)

    # ── Force Safety Gauge (right of grid) — EMA-smoothed cursor ──
    #
    # EMA LOGIC: Instead of directly mapping force_mag to the bar fill,
    # we smoothly interpolate using an exponential moving average.
    # _force_ema[0] = alpha * new_value + (1 - alpha) * old_value
    # Lower alpha → smoother/slower response (more "inertia").
    raw_fn = np.clip(force_mag / FORCE_SAFETY_THRESHOLD, 0, 1.5)
    _force_ema[0] = _FORCE_EMA_ALPHA * raw_fn + (1.0 - _FORCE_EMA_ALPHA) * _force_ema[0]
    smoothed_fn = _force_ema[0]

    bar_x = gx0 + total + gauge_gap
    bar_y0 = gy0
    bar_h = total
    fill_h = int(bar_h * min(smoothed_fn, 1.0))

    if bar_x + gauge_w <= w and bar_y0 + bar_h <= h:
        # Subtle darkened background track
        _feathered_darken(frame, bar_y0, bar_y0 + bar_h, bar_x - 1, bar_x + gauge_w + 1,
                          darkness=0.12, feather=4)

        # Draw gradient fill from bottom upward
        if fill_h > 0:
            fill_top = bar_y0 + bar_h - fill_h
            for py in range(fill_top, bar_y0 + bar_h):
                t = (bar_y0 + bar_h - py) / bar_h  # 0=bottom, 1=top
                # Cold gradient: Ice-blue(0) → Cyan(0.5) → Magenta-red(1.0)
                if t < 0.5:
                    t2 = t * 2.0
                    cr = int(80 + 100 * t2)
                    cg = int(200 + 55 * (1 - t2))
                    cb_c = int(255 - 30 * t2)
                else:
                    t2 = (t - 0.5) * 2.0
                    cr = int(180 + 75 * t2)
                    cg = int(140 * (1 - t2))
                    cb_c = int(225 * (1 - t2) + 60 * t2)
                for px in range(bar_x, min(bar_x + gauge_w, w)):
                    frame[py, px] = [cr, cg, cb_c]

        # Threshold line at 100% — crisp white dash
        thresh_y = bar_y0
        if 0 <= thresh_y < h:
            cv2.line(frame, (bar_x - 2, thresh_y), (bar_x + gauge_w + 2, thresh_y),
                     (200, 200, 200), 1, cv2.LINE_AA)

        # Floating cursor for current smoothed force (horizontal tick mark)
        cursor_y = int(bar_y0 + bar_h - bar_h * min(smoothed_fn, 1.3))
        cursor_y = max(bar_y0, min(bar_y0 + bar_h - 1, cursor_y))
        cursor_color = (255, 255, 255)
        if smoothed_fn > 1.0:
            # Danger: pulsing red cursor on glow_layer
            breath = 0.6 + 0.4 * np.sin(curr_time * 8.0)
            cursor_color = (int(255 * breath), int(60 * breath), int(60 * breath))
            cv2.line(glow_layer, (bar_x - 4, cursor_y), (bar_x + gauge_w + 4, cursor_y),
                     (255, 80, 80), 2, cv2.LINE_AA)
        cv2.line(frame, (bar_x - 3, cursor_y), (bar_x + gauge_w + 3, cursor_y),
                 cursor_color, 2, cv2.LINE_AA)

        # Tiny force label next to gauge
        _hud_text(frame, f"{force_mag:.0f}N", bar_x - 2, bar_y0 + bar_h + 14,
                  (160, 170, 180), 0.28)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Channel Evolution — Pure cv2 scrolling sparkline (ZERO Matplotlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_EVOLUTION_WINDOW = 50   # Show last N episodes as scrolling sparkline columns

# Sparkline group colors (BGR for cv2) — matches the Cold Cyberpunk palette
_SPARK_COLORS = [
    (255, 210, 180),   # Force  — Ice Blue (BGR)
    (240, 255, 0),     # Torque — Neon Cyan (BGR)
    (220, 80, 220),    # Position — Magenta (BGR)
    (90, 185, 220),    # Goal   — Muted Amber (BGR)
]
_SPARK_LABELS = ['F', 'T', 'P', 'G']


def _overlay_evolution_heatmap(frame, glow_layer):
    """Channel Evolution — Pure cv2 scrolling dot-matrix sparkline.
    4 rows (Force/Torque/Position/Goal), each a horizontal sparkline
    of per-episode average firing rate. Cached per-episode for speed."""
    h, w = frame.shape[:2]
    n_eps = len(_episode_firing_history)
    if n_eps < 2:
        return

    # ── Panel geometry (top-right) ──
    panel_w, panel_h = 200, 100
    margin_r, margin_t = 10, 10
    px0 = w - panel_w - margin_r
    py0 = margin_t

    if px0 < w // 3 or py0 + panel_h > h:
        return

    # Only re-render when new episode data arrives (cached)
    need_update = (_evolution_cache[0] is None or _evolution_cache[1] != n_eps)

    if need_update:
        _evolution_cache[1] = n_eps

        window = min(n_eps, _EVOLUTION_WINDOW)
        recent = np.array(_episode_firing_history[-window:])  # (window, 64)

        # Group into 4 channel banks: mean firing rate per bank per episode
        # Force(0-15), Torque(16-31), Position(32-47), Goal(48-63)
        grouped = np.zeros((4, window), dtype=np.float32)
        for gi, (lo, hi) in enumerate([(0, 16), (16, 32), (32, 48), (48, 64)]):
            grouped[gi] = recent[:, lo:hi].mean(axis=1)

        # Per-row normalize to [0, 1] for sparkline height
        for gi in range(4):
            mn, mx = grouped[gi].min(), grouped[gi].max()
            rng = mx - mn if (mx - mn) > 0.5 else 0.5
            grouped[gi] = (grouped[gi] - mn) / rng

        # Render onto a small black canvas
        canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        row_h = panel_h // 4  # 25px per sparkline row

        for gi in range(4):
            base_y = gi * row_h
            color = _SPARK_COLORS[gi]
            vals = grouped[gi]

            # Draw sparkline as connected dots
            x_step = max(1.0, (panel_w - 24) / max(window - 1, 1))
            pts = []
            for si in range(window):
                sx = int(20 + si * x_step)
                # Map normalized value to vertical position within row
                sy = int(base_y + row_h - 3 - vals[si] * (row_h - 6))
                pts.append((sx, sy))

                # Dot-matrix trail: small filled circles
                brightness = 0.3 + 0.7 * (si / max(window - 1, 1))  # fade-in
                dot_color = tuple(int(c * brightness) for c in color)
                cv2.circle(canvas, (sx, sy), 2, dot_color, -1, cv2.LINE_AA)

            # Connect with thin line
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    t = (i + 1) / len(pts)
                    line_color = tuple(int(c * t * 0.5) for c in color)
                    cv2.line(canvas, pts[i], pts[i + 1], line_color, 1, cv2.LINE_AA)

            # Latest point gets a bright glow dot
            if pts:
                lx, ly = pts[-1]
                cv2.circle(canvas, (lx, ly), 3, color, -1, cv2.LINE_AA)

            # Row label (left side)
            label_y = base_y + row_h // 2 + 4
            cv2.putText(canvas, _SPARK_LABELS[gi], (3, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

            # Subtle separator line
            if gi < 3:
                sep_y = base_y + row_h
                cv2.line(canvas, (18, sep_y), (panel_w - 4, sep_y),
                         (30, 35, 40), 1, cv2.LINE_AA)

        # Episode range label at bottom
        start_ep = max(1, n_eps - window + 1)
        ep_label = f"ep {start_ep}-{n_eps}"
        cv2.putText(canvas, ep_label, (panel_w - 70, panel_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 90, 100), 1, cv2.LINE_AA)

        _evolution_cache[0] = canvas

    # ── Alpha blend cached sparkline panel onto frame ──
    cached = _evolution_cache[0]
    if cached is not None:
        ch_h, ch_w = cached.shape[:2]
        y_end = min(py0 + ch_h, h)
        x_end = min(px0 + ch_w, w)
        ah = y_end - py0; aw = x_end - px0
        if ah > 0 and aw > 0:
            # Darken backdrop for contrast
            _feathered_darken(frame, py0, y_end, px0, x_end, darkness=0.06, feather=8)
            # Blend sparkline canvas (only non-black pixels to preserve transparency)
            roi = frame[py0:y_end, px0:x_end].astype(np.float32)
            overlay = cached[:ah, :aw].astype(np.float32)
            # Additive-style blend: wherever overlay is bright, add it
            mask = (overlay.max(axis=2, keepdims=True) > 10).astype(np.float32)
            blended = roi * (1.0 - mask * 0.7) + overlay * 0.9
            frame[py0:y_end, px0:x_end] = np.clip(blended, 0, 255).astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN ENTRY: draw_overlay — Global Bloom Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_overlay(frame, ep, reward, pdi, min_health, firing_rates, reward_history,
                 distance=0.0, force_mag=0.0, torque_mag=0.0, depth=0.0,
                 success_rate=0.0, force_safe_rate=0.0,
                 force_vec=None, health_arr=None):
    """v5.0 Cold Cyberpunk HUD — Global Bloom + EMA + Breathing + Zero Matplotlib.

    BLOOM PIPELINE:
      1. Create a black glow_layer (same size as frame).
      2. Draw high-intensity elements (spike flashes, warning text, danger cursor)
         onto this glow_layer.
      3. Apply fast cv2.GaussianBlur to glow_layer → produces soft glow halos.
      4. Additively blend glow_layer back onto the main frame using cv2.add().
    This emulates the hardware-accelerated bloom seen in native macOS AppKit UIs.

    EMA SMOOTHING:
      The force gauge cursor uses exponential moving average so the indicator
      glides with physical "inertia" instead of jittering per-frame.
      Formula: ema = alpha * new + (1 - alpha) * old   (alpha = 0.18)
    """
    frame = np.ascontiguousarray(frame)
    _overlay_frame_counter[0] += 1

    # ── Step 1: Allocate glow layer (black canvas, same dims) ──
    glow_layer = np.zeros_like(frame)

    # ── Step 2: Draw all UI components ──
    # Neuron grid draws spiking neurons + force gauge onto glow_layer
    _overlay_neuron_grid(frame, firing_rates, min_health, glow_layer,
                         health_arr=health_arr, force_mag=force_mag)

    # Particles are drawn onto glow_layer for bloom effect
    _update_and_draw_particles(frame, glow_layer)

    # HUD text (warning/danger text drawn onto glow_layer for bloom)
    _overlay_text(frame, ep, reward, pdi, min_health, glow_layer,
                  distance=distance, force_mag=force_mag, torque_mag=torque_mag,
                  depth=depth, success_rate=success_rate,
                  force_safe_rate=force_safe_rate)

    # ── Step 3: Blur the glow layer → soft bloom halos ──
    bloom = cv2.GaussianBlur(glow_layer, (_BLOOM_KSIZE, _BLOOM_KSIZE), _BLOOM_SIGMA)

    # ── Step 4: Additive blend → premium glowing edges ──
    # cv2.add() clamps at 255 automatically, which is exactly what we want
    # for additive light blending (like real optical bloom).
    frame = cv2.add(frame, bloom)

    return frame

# ═══ CL1 Biological Agent ═══
class CL1Agent:
    def __init__(self, env, raw_env, neurons, channel_ranking=None, responsiveness=None):
        self.env = env; self.raw_env = raw_env; self.neurons = neurons
        self.action_dim = env.action_space.shape[0]
        self.vie = VIE(neurons, raw_env=raw_env)
        # P1: Population Vector — pass calibration responsiveness as decode weights
        resp_weights = responsiveness if channel_ranking is not None else None
        self.decoder = AntagonisticDecoder(self.action_dim, action_scale=ACTION_SCALE,
                                            channel_weights=resp_weights)
        self.pdi = PDI()
        self.episode_rewards = []
        self.best_reward = -np.inf
        self.top_channels = (channel_ranking[:DOPAMINE_TOP_K].tolist()
                             if channel_ranking is not None else list(range(DOPAMINE_TOP_K)))

    def _detect_spikes(self):
        frames = self.neurons.read(250, None)
        threshold = np.percentile(frames, 99.5)
        spike_channels = list(set(np.where(frames > threshold)[1]))
        firing_rates = np.mean(np.abs(frames.astype(np.float32)), axis=0)
        return spike_channels, firing_rates

    def _dopamine_inject(self, reward):
        """Dopamine-like reward injection — positive reinforcement pathway.

        When reward > 0, delivers structured burst stimulation to the top-K
        calibrated channels. Under the Free Energy Principle, predictable
        structured stimulation reinforces the current behavioral policy
        (the neural culture learns to reproduce the rewarded activity pattern).

        Args:
            reward: Step reward from the environment.
        """
        if reward <= 0: return
        amp = np.clip(reward * 2.0, 0.5, 3.0)
        s = StimDesign(200, -amp, 200, amp)
        self.neurons.stim(ChannelSet(*self.top_channels), s,
                          BurstDesign(DOPAMINE_BURST_N, DOPAMINE_BURST_HZ))

    def _punishment_inject(self, penalty):
        """Punishment noise injection — negative reinforcement pathway.

        When a negative event occurs (force violation, increasing distance,
        large negative reward), delivers unpredictable random-frequency
        stimulation to non-top channels. Under the DishBrain/FEP framework,
        unpredictable noise signals an undesirable state, driving the neural
        culture to modify its activity patterns to avoid this condition.

        Triggers: force > safety threshold, distance increasing, reward < -1.0.

        Args:
            penalty: Negative penalty magnitude (should be <= 0).
        """
        if penalty >= 0: return
        amp = np.clip(abs(penalty) * 1.5, 0.3, 2.0)
        # Select 8 random channels (excluding top_channels — noise, not signal)
        available = [ch for ch in range(64) if ch not in self.top_channels]
        random_chs = np.random.choice(available, size=min(8, len(available)), replace=False).tolist()
        # Irregular pulses: random frequency and burst count
        stim = StimDesign(160, -amp, 160, amp)
        burst = BurstDesign(np.random.randint(3, 10), np.random.randint(50, 300))
        self.neurons.stim(ChannelSet(*random_chs), stim, burst)

    def run_episode(self, max_steps=CL1_MAX_STEPS, record=False, ep_num=0):
        obs, _ = self.env.reset()
        obs_info = extract_obs(obs)
        self.pdi.reset(); self.decoder.reset()
        total_reward = 0.0; frames_list = []
        ep_successes = []; ep_force_safe = []
        step_rewards = deque(maxlen=50); cur_fr = np.zeros(64)
        ep_firing_acc = []  # Accumulate per-step firing rates for evolution heatmap
        prev_dist = None  # Track distance change for punishment

        for step in range(max_steps):
            self.vie.encode(obs_info)
            spikes, cur_fr = self._detect_spikes()
            ep_firing_acc.append(cur_fr.copy())
            # P2: Adaptive VIE encoding — online gain adjustment
            self.vie.adapt(cur_fr)
            vel = obs_info["eef_vel"]
            self.pdi.update(vel); pdi_val = self.pdi.compute()
            # Pure PDI exploration boost (no software curiosity — let neurons handle novelty)
            fep_boost = pdi_val * 0.4
            raw = self.decoder.decode(spikes, pdi_boost=fep_boost)
            # Pure antagonistic decode — let neurons do the learning via STDP
            action = np.clip(raw * ACTION_SCALE, -1.0, 1.0)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs_info = extract_obs(obs)
            total_reward += reward

            force_mag = np.linalg.norm(obs_info["force"])
            torque_mag = np.linalg.norm(obs_info["torque"])
            depth, cur_dist = compute_insertion_depth(obs_info)
            inserted = depth > INSERTION_DEPTH_THRESHOLD
            force_safe = force_mag < FORCE_SAFETY_THRESHOLD
            success = 1 if (inserted and force_safe) else 0
            ep_successes.append(success)
            ep_force_safe.append(1 if force_safe else 0)
            step_rewards.append(reward)

            # === Dual feedback: Dopamine (positive) + Punishment (negative) ===
            # Positive: reward > 0 → structured burst on top-K channels
            self._dopamine_inject(reward)
            # Negative: force exceeds safety OR distance increasing → random noise
            punishment = 0.0
            if force_mag > FORCE_SAFETY_THRESHOLD:
                punishment -= (force_mag - FORCE_SAFETY_THRESHOLD) * 0.5
            if prev_dist is not None and cur_dist > prev_dist + 0.005:
                punishment -= (cur_dist - prev_dist) * 10.0
            if reward < -1.0:
                punishment += reward * 0.3  # already negative
            self._punishment_inject(punishment)
            prev_dist = cur_dist

            if record:
                try:
                    frame = self.raw_env.sim.render(width=RENDER_W, height=RENDER_H, camera_name="frontview")
                    if frame is not None:
                        frame = frame[::-1]
                    else:
                        frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                except Exception:
                    frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                if frame is None or frame.size == 0:
                    continue
                health_full = self.neurons.get_health() if hasattr(self.neurons, 'get_health') else None
                min_h = float(health_full.min()) if health_full is not None else 1.0
                sr = np.mean(ep_successes) * 100
                fsr = np.mean(ep_force_safe) * 100
                frame = draw_overlay(frame, ep_num, total_reward, pdi_val, min_h, cur_fr,
                                     step_rewards, distance=cur_dist, force_mag=force_mag,
                                     torque_mag=torque_mag, depth=depth,
                                     success_rate=sr, force_safe_rate=fsr,
                                     force_vec=obs_info["force"], health_arr=health_full)
                frames_list.append(frame)

            if terminated or truncated: break

        if total_reward > self.best_reward:
            self.best_reward = total_reward

        # Record episode average firing rates for evolution heatmap
        if ep_firing_acc:
            ep_avg_fr = np.mean(ep_firing_acc, axis=0)
            _episode_firing_history.append(ep_avg_fr)

        success_rate = np.mean(ep_successes) * 100 if ep_successes else 0.0
        force_safe_rate = np.mean(ep_force_safe) * 100 if ep_force_safe else 100.0
        return total_reward, self.pdi.compute(), frames_list, success_rate, force_safe_rate

    def train(self, num_episodes=CL1_EPISODES, record_last_n=RECORD_LAST_N):
        """Train the CL1 biological agent over multiple episodes."""
        print("\n" + "=" * 60)
        print("  CL1 Bio-Computer Training (RoboSuite NutAssembly)")
        print("=" * 60)
        print(f"  Episodes: {num_episodes} | Env: {ENV_NAME} ({ROBOT})")
        backend = "Cortical Labs cl-sdk" if CL_AVAILABLE else "Built-in mock"
        print(f"  Backend:  {backend}")
        print(f"  Modules:  VIE(Adaptive) + PopVector + PDI + Dopamine/Punishment (FEP dual-feedback)")
        print(f"  Record:   last {record_last_n} mature episodes\n")

        all_frames = []; all_sr = []; all_fsr = []
        record_start = max(0, num_episodes - record_last_n)

        pbar = tqdm(range(num_episodes), desc="CL1", ncols=90)
        for ep in pbar:
            rec = (ep >= record_start)
            reward, pdi_val, frames, sr, fsr = self.run_episode(record=rec, ep_num=ep)
            self.episode_rewards.append(reward)
            all_sr.append(sr); all_fsr.append(fsr)
            if rec: all_frames.extend(frames)

            avg = np.mean(self.episode_rewards[-20:])
            pbar.set_postfix(R=f"{reward:.1f}", avg20=f"{avg:.1f}",
                             PDI=f"{pdi_val:.2f}", SR=f"{sr:.0f}%", FSR=f"{fsr:.0f}%")

            if hasattr(self.neurons, 'get_health') and (ep + 1) % 100 == 0:
                health = self.neurons.get_health()
                min_h = health.min()
                tqdm.write(f"  [Health] Ep {ep+1}: min_health={min_h:.3f} "
                           f"{'OK' if min_h > 0.5 else 'WARNING!'}")

            if (ep + 1) % 50 == 0:
                avg_sr = np.mean(all_sr[-50:]); avg_fsr = np.mean(all_fsr[-50:])
                tqdm.write(f"  CL1 Ep {ep+1}: R={reward:.1f} avg20={avg:.1f} "
                           f"SR={avg_sr:.1f}% FSR={avg_fsr:.1f}%")

        final = np.mean(self.episode_rewards[-20:])
        final_sr = np.mean(all_sr[-20:]); final_fsr = np.mean(all_fsr[-20:])
        print(f"\n  CL1 Done | avg20={final:.2f} SR={final_sr:.1f}% FSR={final_fsr:.1f}%")
        self.all_success_rates = all_sr; self.all_force_safe_rates = all_fsr
        return all_frames

# ═══ PPO Baseline ═══
def train_ppo_baseline(record_last_n=RECORD_LAST_N):
    from stable_baselines3 import PPO
    print("\n" + "=" * 60)
    print("  PPO Baseline (RoboSuite NutAssembly)")
    print("=" * 60)
    print(f"  Training: {PPO_TIMESTEPS} steps | Eval: {PPO_EVAL_EPS} episodes\n")

    train_env, _ = make_robosuite_env(render=False)
    eval_env, eval_raw = make_robosuite_env(render=True)

    model = PPO("MlpPolicy", train_env, verbose=0,
                n_steps=256, batch_size=64, learning_rate=3e-4)

    all_rewards = []; all_frames = []; all_sr = []; all_fsr = []
    n_chunks = 20; chunk_steps = PPO_TIMESTEPS // n_chunks
    evals_per_chunk = PPO_EVAL_EPS // n_chunks
    record_start = PPO_EVAL_EPS - record_last_n; ep_count = 0

    pbar = tqdm(range(n_chunks), desc="PPO ", ncols=90)
    for chunk in pbar:
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
        for _ in range(evals_per_chunk):
            obs, _ = eval_env.reset()
            ep_r = 0.0; done = False; ep_frames = []
            ep_forces = []; ep_depths = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = eval_env.step(action)
                ep_r += r; done = term or trunc
                oi = extract_obs(obs)
                ep_forces.append(np.linalg.norm(oi["force"]))
                d, _ = compute_insertion_depth(oi)
                ep_depths.append(d)
                if ep_count >= record_start:
                    try:
                        frame = eval_raw.sim.render(width=RENDER_W, height=RENDER_H, camera_name="frontview")
                        if frame is not None:
                            frame = frame[::-1]
                        else:
                            frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                    except Exception:
                        frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                    if frame is not None and frame.size > 0:
                        ep_frames.append(frame)
            all_rewards.append(ep_r)
            max_d = max(ep_depths) if ep_depths else 0
            max_f = max(ep_forces) if ep_forces else 0
            # HER-like hindsight bonus: partial credit for progress
            if max_d > 0 and max_d <= INSERTION_DEPTH_THRESHOLD:
                her_bonus = np.clip(max_d / INSERTION_DEPTH_THRESHOLD, 0.0, 0.8) * 0.3
                all_rewards[-1] += her_bonus
            sr = 100.0 if (max_d > INSERTION_DEPTH_THRESHOLD and max_f < FORCE_SAFETY_THRESHOLD) else 0.0
            fsr = 100.0 if max_f < FORCE_SAFETY_THRESHOLD else 0.0
            all_sr.append(sr); all_fsr.append(fsr)
            if ep_count >= record_start: all_frames.extend(ep_frames)
            ep_count += 1
        avg = np.mean(all_rewards[-evals_per_chunk:])
        pbar.set_postfix(avg_R=f"{avg:.1f}", eps=len(all_rewards))

    train_env.close(); eval_env.close()
    final = np.mean(all_rewards[-20:])
    print(f"\n  PPO Done | avg20={final:.2f}")
    return all_rewards, all_frames, all_sr, all_fsr

# ═══ Random Baseline ═══
def run_random_baseline(num_episodes=RANDOM_EPISODES):
    print("\n" + "=" * 60)
    print("  Random Agent Baseline (RoboSuite NutAssembly)")
    print("=" * 60)
    print(f"  Episodes: {num_episodes}\n")

    env, _ = make_robosuite_env(render=False)
    all_rewards = []; all_sr = []; all_fsr = []

    pbar = tqdm(range(num_episodes), desc="RNG ", ncols=90)
    for ep in pbar:
        obs, _ = env.reset()
        total_reward = 0.0; done = False
        ep_forces = []; ep_depths = []
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            total_reward += r; done = term or trunc
            oi = extract_obs(obs)
            ep_forces.append(np.linalg.norm(oi["force"]))
            d, _ = compute_insertion_depth(oi)
            ep_depths.append(d)
        all_rewards.append(total_reward)
        max_d = max(ep_depths) if ep_depths else 0
        max_f = max(ep_forces) if ep_forces else 0
        sr = 100.0 if (max_d > INSERTION_DEPTH_THRESHOLD and max_f < FORCE_SAFETY_THRESHOLD) else 0.0
        fsr = 100.0 if max_f < FORCE_SAFETY_THRESHOLD else 0.0
        all_sr.append(sr); all_fsr.append(fsr)
        avg = np.mean(all_rewards[-20:])
        pbar.set_postfix(R=f"{total_reward:.1f}", avg20=f"{avg:.1f}")

    env.close()
    print(f"\n  Random Done | avg20={np.mean(all_rewards[-20:]):.2f}")
    return all_rewards, all_sr, all_fsr

# ═══ Video Utilities — now from core/video.py ═══
# save_video() and make_side_by_side() imported at top from core.video

# ═══ Learning Curve ═══
def plot_learning_curves(cl1_r, ppo_r, rnd_r, path=PLOT_FILE,
                         cl1_sr=None, cl1_fsr=None, ppo_sr=None, rnd_sr=None):
    for font in ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']:
        try: plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']; break
        except Exception: continue
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [3, 2]})
    ax = axes[0]

    def rolling(data, w=20):
        if len(data) < w: return data
        return np.convolve(data, np.ones(w)/w, mode='valid')

    cl1_s = rolling(cl1_r); ppo_s = rolling(ppo_r); rnd_s = rolling(rnd_r)
    off = 19

    ax.plot(cl1_r, alpha=0.12, color='#e74c3c')
    ax.plot(ppo_r, alpha=0.12, color='#3498db')
    ax.plot(rnd_r, alpha=0.12, color='#95a5a6')
    ax.plot(range(off, off+len(cl1_s)), cl1_s, color='#e74c3c', lw=2.5,
            label='CL1 Bio (VIE+Force/Torque+Dopamine)')
    ax.plot(range(off, off+len(ppo_s)), ppo_s, color='#3498db', lw=2.5,
            label='PPO Traditional RL')
    ax.plot(range(off, off+len(rnd_s)), rnd_s, color='#95a5a6', lw=2.0, ls='--',
            label='Random (baseline)')

    ax.set_title("Project Senxe v4.0 — RoboSuite NutAssembly (Native Force/Torque)\n"
                 "CL1 Bio vs PPO vs Random: Industrial Assembly Sample Efficiency",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.legend(fontsize=10, loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(len(cl1_r), len(ppo_r), len(rnd_r)))

    ax.text(0.02, 0.95,
            f"Senxe v4.0 — Native Force/Torque Sensors\n"
            f"CL1: {CL1_EPISODES} eps | PPO: {PPO_TIMESTEPS} steps\n"
            f"Random: {RANDOM_EPISODES} eps\n"
            f"NutAssembly ({ROBOT}) — ready for real robotic arm\n"
            f"Force safety: <{FORCE_SAFETY_THRESHOLD}N | Depth: >{INSERTION_DEPTH_THRESHOLD}m",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.7))

    # Success Rate + Force Safety Rate subplot
    ax2 = axes[1]
    if cl1_sr and len(cl1_sr) > 0:
        sr_s = rolling(cl1_sr); ax2.plot(range(off, off+len(sr_s)), sr_s,
            color='#2ecc71', lw=2.0, label='CL1 Success Rate (%)')
    if cl1_fsr and len(cl1_fsr) > 0:
        fsr_s = rolling(cl1_fsr); ax2.plot(range(off, off+len(fsr_s)), fsr_s,
            color='#e67e22', lw=2.0, ls='-.', label='CL1 Force Safety Rate (%)')
    if ppo_sr and len(ppo_sr) > 0:
        psr_s = rolling(ppo_sr); ax2.plot(range(off, off+len(psr_s)), psr_s,
            color='#3498db', lw=1.5, ls=':', label='PPO Success Rate (%)')
    if rnd_sr and len(rnd_sr) > 0:
        rsr_s = rolling(rnd_sr); ax2.plot(range(off, off+len(rsr_s)), rsr_s,
            color='#95a5a6', lw=1.5, ls=':', label='Random Success Rate (%)')

    ax2.set_xlabel("Episode"); ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(0, 105); ax2.legend(fontsize=9, loc='upper left'); ax2.grid(True, alpha=0.3)
    ax2.set_title("Success Rate + Force Safety Rate", fontsize=11)

    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Learning curve saved: {path}")

# ═══ Main Entry Point ═══
def main():
    print("+" + "=" * 58 + "+")
    print("|  Project Senxe v4.0 — RoboSuite NutAssembly            |")
    print("|  Native Force/Torque — Ready for Real Robotic Arm       |")
    print("+" + "=" * 58 + "+\n")

    if CL_AVAILABLE:
        print("  CL SDK: Cortical Labs cl-sdk (real/simulator)")
    else:
        print("  CL SDK unavailable -- using built-in mock")
        print("     (pip install cl-sdk to use real CL1 backend)")
    print(f"  Env: {ENV_NAME} | Robot: {ROBOT}")
    print(f"  Force safety: <{FORCE_SAFETY_THRESHOLD}N | Depth: >{INSERTION_DEPTH_THRESHOLD}m\n")

    # Phase 0: Calibration
    print("-" * 60); print("  Phase 0: Channel Warm-up Calibration"); print("-" * 60)
    with cl_open() as neurons:
        ranking, resp = warmup_calibration(neurons, WARMUP_SECONDS)

        # Phase 1: CL1 Training
        env, raw_env = make_robosuite_env(render=True)
        agent = CL1Agent(env, raw_env, neurons, channel_ranking=ranking, responsiveness=resp)
        cl1_frames = agent.train(num_episodes=CL1_EPISODES, record_last_n=RECORD_LAST_N)
        cl1_rewards = agent.episode_rewards.copy()
        cl1_sr = agent.all_success_rates.copy()
        cl1_fsr = agent.all_force_safe_rates.copy()
        env.close()

    # Phase 2: PPO
    ppo_rewards, ppo_frames, ppo_sr, ppo_fsr = train_ppo_baseline(record_last_n=RECORD_LAST_N)

    # Phase 3: Random
    rnd_rewards, rnd_sr, rnd_fsr = run_random_baseline(num_episodes=RANDOM_EPISODES)

    # Phase 4: Videos
    print("\n" + "-" * 60); print("  Phase 4: Generating Videos"); print("-" * 60)
    save_video(cl1_frames, VIDEO_CL1, fps=VIDEO_FPS, target_seconds=20)
    make_side_by_side(cl1_frames, ppo_frames, VIDEO_SIDE, fps=VIDEO_FPS,
                      left_label="CL1 Bio (Force/Torque)",
                      right_label="PPO Traditional RL",
                      center_label="NutAssembly")

    # Phase 5: Plot
    print()
    plot_learning_curves(cl1_rewards, ppo_rewards, rnd_rewards,
                         cl1_sr=cl1_sr, cl1_fsr=cl1_fsr, ppo_sr=ppo_sr, rnd_sr=rnd_sr)

    # Done
    print("\n" + "=" * 60)
    print("  Project Senxe v4.0 Demo Complete!")
    print("=" * 60)
    print(f"  Video (CL1):          {VIDEO_CL1}")
    print(f"  Video (side-by-side): {VIDEO_SIDE}")
    print(f"  Plot:                 {PLOT_FILE}")
    print(f"  CL1 avg20:  {np.mean(cl1_rewards[-20:]):.2f}")
    print(f"  PPO avg20:  {np.mean(ppo_rewards[-20:]):.2f}")
    print(f"  RNG avg20:  {np.mean(rnd_rewards[-20:]):.2f}")
    print(f"  CL1 SR:     {np.mean(cl1_sr[-20:]):.1f}%")
    print(f"  CL1 FSR:    {np.mean(cl1_fsr[-20:]):.1f}%")
    print()
    print("  Switch to real CL1 hardware:")
    print("    pip install cl-sdk   # Auto-detected, zero code changes!")
    print()

if __name__ == "__main__":
    main()
