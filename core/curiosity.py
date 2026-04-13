"""
Project Senxe — Neural Intrinsic Curiosity
============================================
Firing-pattern novelty detection for exploration drive.

This module implements an intrinsic curiosity mechanism based on the
information-theoretic principle that novel neural activity patterns
should drive exploration. It is inspired by the observation that
biological neural circuits exhibit increased activity and plasticity
when encountering unfamiliar sensory patterns.

    Familiar firing pattern → Low novelty  → Reduce exploration noise
    Novel firing pattern    → High novelty → Increase exploration

The novelty score is computed as the Euclidean distance between the
current (normalized) firing pattern and the mean of recent patterns
stored in a rolling memory buffer.
"""

from __future__ import annotations

import numpy as np
from collections import deque


class NeuralCuriosity:
    """Neural intrinsic curiosity — novelty-driven exploration modulator.

    Maintains a rolling memory of recent firing-rate patterns and computes
    a novelty score for each new observation. The score quantifies how
    different the current pattern is from the recent history, providing
    an intrinsic motivation signal that complements the extrinsic reward.

    This is conceptually related to prediction-error-based curiosity
    (Schmidhuber, 2010) but operates directly on raw neural firing
    patterns rather than learned feature representations.

    Args:
        n_channels: Number of neural channels (default: 64).
        memory_size: Maximum number of patterns retained in the
                     rolling memory buffer.
    """

    def __init__(self, n_channels: int = 64, memory_size: int = 100) -> None:
        self.memory: deque = deque(maxlen=memory_size)
        self.n_channels: int = n_channels

    def compute_novelty(self, firing_rates: np.ndarray) -> float:
        """Compute novelty of the current firing pattern relative to history.

        The firing pattern is normalized to [0, 1] per channel, then
        compared against the mean of the last K patterns (K ≤ 20) via
        Euclidean distance. The raw distance is scaled by 3× and clipped
        to [0, 2] to produce a usable novelty signal.

        During the initial phase (< 5 patterns in memory), returns a
        fixed high-curiosity value of 1.0 to encourage early exploration.

        Args:
            firing_rates: Per-channel firing rates, shape (n_channels,).

        Returns:
            float: Novelty score in [0.0, 2.0].
                   High values indicate novel patterns (drive exploration).
                   Low values indicate familiar patterns (drive exploitation).
        """
        fr_norm = firing_rates / (firing_rates.max() + 1e-6)
        self.memory.append(fr_norm.copy())

        if len(self.memory) < 5:
            return 1.0  # High curiosity during initial exploration phase

        # Euclidean distance to mean of recent K patterns
        recent = np.array(list(self.memory)[-min(20, len(self.memory)):])
        mean_pattern = recent.mean(axis=0)
        dist = np.linalg.norm(fr_norm - mean_pattern)
        # Normalize: typical distance ~0.1–0.5 for 64-dim unit vectors
        novelty = np.clip(dist * 3.0, 0.0, 2.0)
        return float(novelty)

    def reset(self) -> None:
        """Clear pattern memory for a new episode or session."""
        self.memory.clear()
