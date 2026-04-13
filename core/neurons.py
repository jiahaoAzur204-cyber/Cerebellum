"""
Project Senxe — Neural Interface Module
========================================
CL1 biological neural interface: built-in simulator + real CL1 auto-detection.

This module provides the hardware abstraction layer for interfacing with
Cortical Labs CL1 biological neural organoids on a 64-channel MEA
(Micro-Electrode Array). When the ``cl-sdk`` package is installed, all
operations transparently route to real biological hardware. Otherwise, a
built-in mock simulator reproduces key electrophysiological behaviors
(Poisson spiking, STDP-like plasticity, metabolic homeostasis).

Exports:
    MockChannelSet / MockStimDesign / MockBurstDesign — Mock CL1 SDK types
    MockNeurons    — Built-in 64-channel MEA simulator with metabolic guardrail
    cl_open()      — Unified context manager: prefer real cl-sdk, fallback to mock
    warmup_calibration() — 10-second channel warm-up and responsiveness ranking
    ChannelSet / StimDesign / BurstDesign — Unified type aliases (real or mock)
    CL_AVAILABLE   — Boolean flag: True if real cl-sdk is installed

Usage::

    from core.neurons import cl_open, warmup_calibration, ChannelSet, StimDesign, BurstDesign

    with cl_open() as neurons:
        ranking, resp = warmup_calibration(neurons)
        neurons.stim(ChannelSet(0, 1, 2), StimDesign(160, -1.0, 160, 1.0), BurstDesign(3, 100))
        frames = neurons.read(250, None)
"""

from __future__ import annotations

import os
import numpy as np
from contextlib import contextmanager
from typing import Optional, Set, Tuple

# ═══ CL1 SDK Auto-Detection ═══
os.environ.setdefault("CL_MOCK_ACCELERATED_TIME", "1")
os.environ.setdefault("CL_MOCK_RANDOM_SEED", "42")

try:
    import cl as _cl_sdk
    from cl import ChannelSet as _RealChannelSet
    from cl import StimDesign as _RealStimDesign
    from cl import BurstDesign as _RealBurstDesign
    CL_AVAILABLE: bool = True
except ImportError:
    CL_AVAILABLE: bool = False


# ═══ Mock CL1 SDK Types ═══

class MockChannelSet:
    """Mock channel set — mirrors the ``cl.ChannelSet`` interface.

    Represents a selection of electrode channels on the MEA for targeted
    stimulation or recording.

    Args:
        *channels: Variable number of integer channel indices (0–63).
    """

    def __init__(self, *channels: int) -> None:
        self.channels: Set[int] = set(channels)

    def __repr__(self) -> str:
        return f"MockChannelSet({sorted(self.channels)})"


class MockStimDesign:
    """Mock biphasic pulse waveform — mirrors the ``cl.StimDesign`` interface.

    Defines a charge-balanced biphasic stimulation waveform. Each phase is
    specified as (duration_us, amplitude_uA) pairs. Charge balance is
    critical for neural safety in real hardware.

    Args:
        *phases: Alternating (duration_us, amplitude_uA) values.
                 Example: (160, -1.0, 160, 1.0) → 160 µs cathodic at -1 µA,
                 then 160 µs anodic at +1 µA.
    """

    def __init__(self, *phases: float) -> None:
        self.phases: Tuple[float, ...] = phases

    def __repr__(self) -> str:
        return f"MockStimDesign({self.phases})"


class MockBurstDesign:
    """Mock burst parameters — mirrors the ``cl.BurstDesign`` interface.

    Controls the temporal pattern of pulse delivery. Higher frequency
    and pulse count increase the total charge delivered to the neural
    tissue, increasing the probability of evoking an action potential.

    Args:
        count: Number of pulses in the burst.
        frequency_hz: Pulse repetition frequency in Hz.
    """

    def __init__(self, count: int, frequency_hz: int) -> None:
        self.count: int = count
        self.frequency_hz: int = frequency_hz

    def __repr__(self) -> str:
        return f"MockBurstDesign(count={self.count}, freq={self.frequency_hz})"


# ═══ Mock Neurons (64-ch MEA simulator) ═══

class MockNeurons:
    """Built-in CL1 simulator — 64-channel MEA with metabolic guardrail.

    Simulates neural activity on a 64-channel micro-electrode array with
    biologically-motivated dynamics:

    - **Baseline activity**: Poisson process (~170 Hz per channel), mimicking
      spontaneous cortical firing rates observed in organoid cultures.
    - **Stimulation response**: Stimulated channels show enhanced firing rates
      proportional to pulse amplitude × burst count × frequency.
    - **STDP-like plasticity**: Cumulative stimulation gradually increases
      channel sensitivity, modeling long-term potentiation (LTP).
    - **Metabolic guardrail**: Per-channel health (1.0 = healthy, 0.3 = floor)
      decays with over-stimulation and slowly recovers during read cycles,
      preventing excitotoxicity — analogous to homeostatic synaptic scaling.

    This simulator is designed to be a drop-in replacement for the real
    ``cl-sdk`` neurons object, enabling full pipeline development and
    testing without access to biological hardware.
    """

    def __init__(self) -> None:
        self.n_channels: int = 64
        self.stim_buffer: np.ndarray = np.zeros(64)
        self.sensitivity: np.ndarray = np.ones(64) * 170.0
        self.timestamp: int = 0
        # Metabolic health tracking
        self.cumulative_stim: np.ndarray = np.zeros(64)
        self.health: np.ndarray = np.ones(64)  # 1.0 = healthy, <0.5 = needs rest

    def __enter__(self) -> "MockNeurons":
        return self

    def __exit__(self, *args) -> None:
        pass

    def take_control(self) -> None:
        """Acquire exclusive control of the MEA (no-op in mock)."""
        pass

    def start(self) -> None:
        """Begin recording/stimulation session (no-op in mock)."""
        pass

    def wait_until_readable(self, **kwargs) -> None:
        """Block until read buffer has sufficient data (no-op in mock)."""
        pass

    def wait_until_recordable(self, **kwargs) -> None:
        """Block until recording subsystem is ready (no-op in mock)."""
        pass

    def release_control(self) -> None:
        """Release exclusive MEA control (no-op in mock)."""
        pass

    def close(self) -> None:
        """Terminate session and release all resources (no-op in mock)."""
        pass

    def stim(
        self,
        channel_set: MockChannelSet,
        stim_design: MockStimDesign,
        burst_design: Optional[MockBurstDesign] = None,
    ) -> None:
        """Deliver electrical stimulation to the specified channels.

        The stimulation intensity is computed as:
            intensity = |amplitude| × burst_count × (burst_freq / 100)

        Each stimulated channel's response is modulated by its current
        metabolic health (unhealthy channels have reduced response),
        and a small plasticity term is applied (sensitivity += 0.01 × intensity).

        Metabolic health decays with cumulative stimulation to prevent
        excitotoxicity, with a hard floor at 0.3.

        Args:
            channel_set: Target channels for stimulation.
            stim_design: Biphasic waveform specification.
            burst_design: Optional burst parameters (default: single pulse at 100 Hz).
        """
        amp = abs(stim_design.phases[1]) if len(stim_design.phases) > 1 else 0.5
        n = burst_design.count if burst_design else 1
        freq = burst_design.frequency_hz if burst_design else 100
        intensity = amp * n * (freq / 100.0)

        for ch in channel_set.channels:
            # Metabolic guardrail: unhealthy channels have reduced response
            effective = intensity * self.health[ch]
            self.stim_buffer[ch] += effective
            self.sensitivity[ch] += 0.01 * effective  # LTP-like plasticity
            self.cumulative_stim[ch] += intensity
            # Health decays with cumulative stim (excitotoxicity model)
            self.health[ch] = max(0.3, 1.0 - self.cumulative_stim[ch] * 0.0001)

    def read(self, frame_count: int, from_timestamp: Optional[int]) -> np.ndarray:
        """Read neural activity data from the MEA.

        Generates Poisson-distributed spike data influenced by prior
        stimulations. The firing rate per channel is:
            rate = clip(sensitivity + stim_buffer × 30, 50, 600) Hz

        After each read, the stimulus buffer decays by 0.7× (exponential
        washout) and health recovers by +0.001 per channel (slow homeostatic
        recovery).

        Args:
            frame_count: Number of temporal frames to read.
            from_timestamp: Starting timestamp (ignored in mock; included
                            for API compatibility with real cl-sdk).

        Returns:
            np.ndarray: Shape (frame_count, 64), dtype int16.
                        Each value represents the instantaneous firing
                        rate sample for that channel at that time frame.
        """
        rates = np.clip(self.sensitivity + self.stim_buffer * 30.0, 50, 600)
        frames = np.zeros((frame_count, self.n_channels), dtype=np.int16)
        for ch in range(self.n_channels):
            frames[:, ch] = np.random.poisson(rates[ch], frame_count).astype(np.int16)
        self.stim_buffer *= 0.3  # Stimulus buffer exponential decay
        self.health = np.minimum(1.0, self.health + 0.001)  # Slow homeostatic recovery
        self.timestamp += frame_count
        return frames

    def get_health(self) -> np.ndarray:
        """Return a copy of per-channel metabolic health values.

        Returns:
            np.ndarray: Shape (64,), values in [0.3, 1.0].
                        1.0 = fully healthy, 0.3 = minimum floor.
        """
        return self.health.copy()


# ═══ Unified CL1 Entry Point ═══

@contextmanager
def _mock_open():
    """Context manager for the built-in mock neuron simulator."""
    neurons = MockNeurons()
    try:
        yield neurons
    finally:
        pass


def cl_open():
    """Unified CL1 entry point: prefer real cl-sdk, fallback to built-in mock.

    Returns a context manager that yields a neurons object with the
    standard CL1 interface (.stim(), .read(), .get_health(), etc.).

    When ``cl-sdk`` is installed, this delegates to ``cl.open()`` which
    connects to real Cortical Labs biological hardware. Otherwise, it
    returns a ``MockNeurons`` instance that simulates the full 64-channel
    MEA electrophysiology pipeline.

    Usage::

        with cl_open() as neurons:
            neurons.stim(ChannelSet(0, 1), StimDesign(160, -1.0, 160, 1.0))
            frames = neurons.read(250, None)
    """
    if CL_AVAILABLE:
        return _cl_sdk.open()
    return _mock_open()


# ═══ Unified Type Names ═══
# These resolve to real CL SDK types when available, else mock types.

if CL_AVAILABLE:
    ChannelSet = _RealChannelSet
    StimDesign = _RealStimDesign
    BurstDesign = _RealBurstDesign
else:
    ChannelSet = MockChannelSet
    StimDesign = MockStimDesign
    BurstDesign = MockBurstDesign


# ═══ Channel Warm-up Calibration ═══

def warmup_calibration(
    neurons,
    duration_sec: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Channel warm-up calibration — probe all 64 channels for responsiveness.

    This two-phase calibration protocol establishes a per-channel
    responsiveness profile, analogous to impedance testing in real MEA
    setups. The resulting ranking is used downstream for:

    - **Population vector decoding**: Weighting spike contributions by
      channel quality (more responsive channels contribute more).
    - **Dopamine injection targeting**: Reward signals are delivered to
      the top-K most responsive channels for maximum effect.

    Phase 1 (Baseline): Read spontaneous activity without stimulation
    to establish each channel's resting firing rate.

    Phase 2 (Probing): Deliver a standard biphasic pulse to every channel,
    then measure the evoked response. The responsiveness delta
    (post-stim − baseline) ranks channel quality.

    Args:
        neurons: CL1 neurons instance (real hardware or mock simulator).
        duration_sec: Total calibration duration in seconds (default: 10).
                      Higher values yield more stable rankings.

    Returns:
        Tuple of:
            - channel_ranking (np.ndarray): Shape (64,). Channel indices
              sorted by responsiveness in descending order (best first).
            - responsiveness (np.ndarray): Shape (64,). Per-channel
              response delta (evoked − baseline firing rate).
    """
    print("  [Calibration] Channel warm-up calibrating...")
    n_rounds = int(duration_sec * 2.5)
    baseline_responses = np.zeros(64)
    stim_responses = np.zeros(64)

    # Phase 1: Measure baseline activity (no stimulation)
    for _ in range(n_rounds // 2):
        frames = neurons.read(100, None)
        baseline_responses += np.mean(np.abs(frames.astype(float)), axis=0)
    baseline_responses /= (n_rounds // 2)

    # Phase 2: Stimulate each channel and measure evoked response
    stim = StimDesign(160, -1.0, 160, 1.0)
    burst = BurstDesign(3, 100)
    for ch in range(64):
        neurons.stim(ChannelSet(ch), stim, burst)
    for _ in range(n_rounds // 2):
        frames = neurons.read(100, None)
        stim_responses += np.mean(np.abs(frames.astype(float)), axis=0)
    stim_responses /= (n_rounds // 2)

    # Response delta = evoked - baseline
    responsiveness = stim_responses - baseline_responses
    channel_ranking = np.argsort(responsiveness)[::-1]

    top8 = channel_ranking[:8]
    print(f"  [Calibration] Done! Top-8: {top8.tolist()} "
          f"range: {responsiveness[top8[0]]:.1f}~{responsiveness[top8[-1]]:.1f}")
    return channel_ranking, responsiveness
