"""
Project Senxe — Physical Disturbance Index (PDI)
=================================================
FEP-inspired explore/exploit gate based on kinematic stability.

The Physical Disturbance Index quantifies the motion stability of the
end-effector by combining velocity variance and acceleration variance
over a rolling window:

    PDI = std(velocity) + std(acceleration)

This metric operationalizes the Free Energy Principle (FEP) as a
practical control signal:

    High PDI → Unstable kinematics → Increase exploration (surprise minimization)
    Low  PDI → Stable kinematics   → Exploit current policy (precision control)

Unlike hand-tuned epsilon-greedy schedules, PDI provides a principled,
physics-grounded mechanism for balancing exploration and exploitation
that naturally adapts to the current task phase.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional


class PDI:
    """Physical Disturbance Index — kinematic stability tracker.

    Maintains rolling histories of velocity and acceleration vectors,
    computing their standard deviations to produce a scalar stability
    metric. The PDI value directly modulates the exploration noise
    injected into the antagonistic decoder output.

    The relationship to the Free Energy Principle:
    - High PDI ≈ high sensory surprise → the agent's predictions about
      its own kinematics are inaccurate → explore to gather information.
    - Low PDI ≈ low surprise → the agent's motor model is accurate →
      exploit the current policy for precise task execution.

    Args:
        window: Rolling window size for velocity/acceleration history.
                Larger windows produce smoother PDI transitions.
    """

    def __init__(self, window: int = 20) -> None:
        self.velocities: deque = deque(maxlen=window)
        self.accelerations: deque = deque(maxlen=window)
        self.prev_vel: Optional[np.ndarray] = None

    def update(self, velocity: np.ndarray) -> None:
        """Record a new velocity observation and compute acceleration.

        Acceleration is derived as the first-order finite difference
        of consecutive velocity observations.

        Args:
            velocity: Current end-effector velocity vector (typically 3D).
        """
        self.velocities.append(velocity.copy())
        if self.prev_vel is not None:
            self.accelerations.append(velocity - self.prev_vel)
        self.prev_vel = velocity.copy()

    def compute(self) -> float:
        """Compute the current Physical Disturbance Index.

        PDI = mean(std(velocity per axis)) + mean(std(acceleration per axis))

        Returns:
            float: PDI value clipped to [0.0, 2.0].
                   Returns 0.5 (moderate exploration) if insufficient
                   data has been collected (< 3 velocity samples).
        """
        if len(self.velocities) < 3:
            return 0.5
        vel_std = np.mean(np.std(np.array(self.velocities), axis=0))
        acc_std = 0.0
        if len(self.accelerations) > 1:
            acc_std = np.mean(np.std(np.array(self.accelerations), axis=0))
        return float(np.clip(vel_std + acc_std, 0.0, 2.0))

    def reset(self) -> None:
        """Clear all history for a new episode."""
        self.velocities.clear()
        self.accelerations.clear()
        self.prev_vel = None
