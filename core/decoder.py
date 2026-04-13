"""
Project Senxe — Antagonistic Decoding Module
=============================================
Biological flexor/extensor antagonistic motor control decoder.

This module implements the core motor output pathway of the Senxe framework.
Inspired by the antagonistic muscle pairs in vertebrate motor systems
(e.g., biceps/triceps), it converts raw neural spike channels into smooth,
continuous action vectors suitable for robotic control.

64 channels are split into two antagonistic populations (32 each):

    Flexor   (CH 0–31)  → positive force per action dimension
    Extensor (CH 32–63) → negative force per action dimension

    Action[i] = (flexor_count − extensor_count) / (total + ε)

The differential signal naturally produces:
    - Balanced activity → near-zero output (co-contraction / stability)
    - Flexor dominance  → positive action (extension movement)
    - Extensor dominance → negative action (flexion movement)

An EMA (Exponential Moving Average) low-pass filter smooths the output,
mimicking the mechanical inertia of biological muscle tissue and producing
jerk-free trajectories suitable for force-sensitive industrial tasks.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


class AntagonisticDecoder:
    """Antagonistic Decoder — converts spike channels to smoothed action vectors.

    Maps spiking activity from a 64-channel MEA onto an N-dimensional action
    space using the biological flexor/extensor antagonistic principle. Each
    action dimension is controlled by a pair of channel sub-populations whose
    differential activity determines the output magnitude and direction.

    The decoder supports optional per-channel weighting from calibration data,
    allowing more responsive channels to contribute proportionally more to
    the population vector — analogous to how motor cortex neurons with
    stronger corticospinal projections dominate movement commands.

    Args:
        action_dim: Number of action dimensions (4 for Fetch, 7 for Panda).
        ema_alpha: EMA smoothing coefficient in [0, 1].
                   Higher values → more responsive (less smoothing).
                   Lower values → smoother trajectories (more inertia).
        action_scale: Multiplicative scaling factor applied after decoding.
        channel_weights: Optional (64,) array of per-channel weights from
                         warm-up calibration. If provided, each spike
                         contributes its channel's weight instead of 1.0.
    """

    def __init__(
        self,
        action_dim: int = 4,
        ema_alpha: float = 0.35,
        action_scale: float = 0.25,
        channel_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.action_dim: int = action_dim
        self.group_size: int = max(1, 32 // action_dim)
        self.ema_alpha: float = ema_alpha
        self.action_scale: float = action_scale
        self.prev_action: np.ndarray = np.zeros(action_dim)

        # Population vector weights from calibration responsiveness
        if channel_weights is not None:
            self.ch_weights = np.array(channel_weights, dtype=np.float64)
            self.ch_weights = self.ch_weights / (self.ch_weights.max() + 1e-6)
        else:
            self.ch_weights = np.ones(64)

    def decode(
        self,
        spike_channels: List[int],
        pdi_boost: float = 0.0,
    ) -> np.ndarray:
        """Decode active spike channels into a smoothed action vector.

        For each action dimension i, the decoder computes:

            flexor_sum  = Σ weight[ch] for ch in [i*G, (i+1)*G)      (CH 0–31)
            extensor_sum = Σ weight[ch] for ch in [32+i*G, 32+(i+1)*G) (CH 32–63)
            raw_action[i] = (flexor_sum − extensor_sum) / (flexor_sum + extensor_sum + ε)

        where G = 32 // action_dim is the group size per dimension.

        The raw action is then:
            1. Perturbed by FEP exploration noise (if PDI is high).
            2. Smoothed via EMA: action = α·raw + (1−α)·prev_action.
            3. Scaled and clipped to [-1, 1].

        Args:
            spike_channels: List of active (spiking) channel indices [0–63].
            pdi_boost: FEP exploration noise magnitude (typically PDI × 0.4).
                       When PDI is high (unstable state), additional Gaussian
                       noise is injected to promote exploration.

        Returns:
            np.ndarray: Action vector of shape (action_dim,), clipped to [-1, 1].
        """
        action = np.zeros(self.action_dim)

        for i in range(self.action_dim):
            f_lo = i * self.group_size
            f_hi = min((i + 1) * self.group_size, 32)
            e_lo = 32 + f_lo
            e_hi = min(32 + f_hi, 64)

            flex = sum(self.ch_weights[ch] for ch in spike_channels if f_lo <= ch < f_hi)
            ext = sum(self.ch_weights[ch] for ch in spike_channels if e_lo <= ch < e_hi)
            action[i] = (flex - ext) / (flex + ext + 1e-6)

        # FEP: high PDI (unstable state) → inject exploration noise
        if pdi_boost > 0.1:
            action += np.random.randn(self.action_dim) * pdi_boost * 0.25

        # EMA smoothing → bio-muscle-like smooth movement trajectory
        action = self.ema_alpha * action + (1 - self.ema_alpha) * self.prev_action
        self.prev_action = action.copy()

        return np.clip(action * self.action_scale, -1.0, 1.0)

    def reset(self) -> None:
        """Reset EMA state for a new episode."""
        self.prev_action = np.zeros(self.action_dim)
