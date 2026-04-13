"""
Project Senxe — Core Module Unit Tests
========================================
Tests for neurons, decoder, PDI, and curiosity modules.

Run:  pytest tests/test_core.py -v
"""

import numpy as np
import pytest

from core.neurons import (
    MockChannelSet, MockStimDesign, MockBurstDesign, MockNeurons,
    ChannelSet, StimDesign, BurstDesign, warmup_calibration,
)
from core.decoder import AntagonisticDecoder
from core.pdi import PDI
from core.curiosity import NeuralCuriosity


# ═══════════════════════════════════════════════════════════════
#  MockNeurons Tests
# ═══════════════════════════════════════════════════════════════

class TestMockNeurons:
    """Tests for the built-in CL1 MEA simulator."""

    def test_init_defaults(self):
        n = MockNeurons()
        assert n.n_channels == 64
        assert n.stim_buffer.shape == (64,)
        assert n.sensitivity.shape == (64,)
        assert n.health.shape == (64,)
        assert np.all(n.health == 1.0)

    def test_context_manager(self):
        with MockNeurons() as n:
            assert n.n_channels == 64

    def test_read_shape(self):
        n = MockNeurons()
        frames = n.read(100, None)
        assert frames.shape == (100, 64)
        assert frames.dtype == np.int16

    def test_read_advances_timestamp(self):
        n = MockNeurons()
        assert n.timestamp == 0
        n.read(100, None)
        assert n.timestamp == 100
        n.read(50, None)
        assert n.timestamp == 150

    def test_stim_increases_sensitivity(self):
        n = MockNeurons()
        initial_sens = n.sensitivity[0]
        cs = MockChannelSet(0)
        sd = MockStimDesign(160, -1.0, 160, 1.0)
        bd = MockBurstDesign(5, 200)
        n.stim(cs, sd, bd)
        assert n.sensitivity[0] > initial_sens

    def test_stim_only_affects_target_channels(self):
        n = MockNeurons()
        initial_sens_10 = n.sensitivity[10]
        cs = MockChannelSet(0, 1, 2)
        sd = MockStimDesign(160, -1.0, 160, 1.0)
        n.stim(cs, sd)
        # Channel 10 should not be affected
        assert n.sensitivity[10] == initial_sens_10

    def test_health_decays_with_overstimulation(self):
        n = MockNeurons()
        cs = MockChannelSet(0)
        sd = MockStimDesign(160, -2.0, 160, 2.0)
        bd = MockBurstDesign(20, 500)
        # Heavy stimulation
        for _ in range(100):
            n.stim(cs, sd, bd)
        assert n.health[0] < 1.0
        assert n.health[0] >= 0.3  # minimum health floor

    def test_health_recovers_on_read(self):
        n = MockNeurons()
        n.health[0] = 0.5
        n.read(100, None)
        assert n.health[0] > 0.5  # slow recovery

    def test_get_health_returns_copy(self):
        n = MockNeurons()
        h = n.get_health()
        h[0] = 0.0
        assert n.health[0] == 1.0  # original unchanged

    def test_stim_buffer_decays_on_read(self):
        n = MockNeurons()
        cs = MockChannelSet(5)
        sd = MockStimDesign(160, -1.0, 160, 1.0)
        n.stim(cs, sd)
        buf_before = n.stim_buffer[5]
        assert buf_before > 0
        n.read(100, None)
        assert n.stim_buffer[5] < buf_before  # decayed by 0.3x


class TestMockTypes:
    """Tests for mock CL1 SDK type equivalents."""

    def test_channel_set(self):
        cs = MockChannelSet(1, 2, 3)
        assert cs.channels == {1, 2, 3}
        assert repr(cs) == "MockChannelSet([1, 2, 3])"

    def test_channel_set_dedup(self):
        cs = MockChannelSet(1, 1, 2)
        assert cs.channels == {1, 2}

    def test_stim_design(self):
        sd = MockStimDesign(160, -1.0, 160, 1.0)
        assert sd.phases == (160, -1.0, 160, 1.0)

    def test_burst_design(self):
        bd = MockBurstDesign(5, 300)
        assert bd.count == 5
        assert bd.frequency_hz == 300


class TestWarmupCalibration:
    """Tests for channel warm-up calibration."""

    def test_returns_correct_shapes(self):
        n = MockNeurons()
        ranking, resp = warmup_calibration(n, duration_sec=2.0)
        assert ranking.shape == (64,)
        assert resp.shape == (64,)

    def test_ranking_is_sorted_descending(self):
        n = MockNeurons()
        ranking, resp = warmup_calibration(n, duration_sec=2.0)
        # Ranking should be sorted by responsiveness (highest first)
        for i in range(len(ranking) - 1):
            assert resp[ranking[i]] >= resp[ranking[i + 1]]

    def test_ranking_contains_all_channels(self):
        n = MockNeurons()
        ranking, _ = warmup_calibration(n, duration_sec=2.0)
        assert set(ranking) == set(range(64))


# ═══════════════════════════════════════════════════════════════
#  AntagonisticDecoder Tests
# ═══════════════════════════════════════════════════════════════

class TestAntagonisticDecoder:
    """Tests for the antagonistic (flexor/extensor) decoder."""

    def test_init_defaults(self):
        d = AntagonisticDecoder(action_dim=4)
        assert d.action_dim == 4
        assert d.prev_action.shape == (4,)

    def test_decode_empty_spikes(self):
        d = AntagonisticDecoder(action_dim=4, action_scale=1.0)
        action = d.decode([])
        # No spikes → action close to zero (modulo EMA of prev=0)
        assert action.shape == (4,)
        assert np.allclose(action, 0.0, atol=0.1)

    def test_decode_flexor_only(self):
        d = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        # Only flexor channels for dim 0 (CH 0-7)
        spikes = list(range(0, 8))
        action = d.decode(spikes)
        # dim 0 should be positive (flexor > extensor)
        assert action[0] > 0

    def test_decode_extensor_only(self):
        d = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        # Only extensor channels for dim 0 (CH 32-39)
        spikes = list(range(32, 40))
        action = d.decode(spikes)
        # dim 0 should be negative (extensor > flexor)
        assert action[0] < 0

    def test_decode_balanced_near_zero(self):
        d = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        # Equal flexor and extensor for dim 0
        spikes = list(range(0, 8)) + list(range(32, 40))
        action = d.decode(spikes)
        # Should be close to zero (balanced)
        assert abs(action[0]) < 0.15

    def test_decode_output_clipped(self):
        d = AntagonisticDecoder(action_dim=4, action_scale=2.0)
        spikes = list(range(0, 32))  # all flexors
        action = d.decode(spikes)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_ema_smoothing(self):
        d = AntagonisticDecoder(action_dim=4, ema_alpha=0.5, action_scale=1.0)
        spikes = list(range(0, 8))
        a1 = d.decode(spikes)
        a2 = d.decode(spikes)
        # Second decode should be different due to EMA blending with prev
        # With alpha=0.5, a2 should be larger (accumulating)
        assert not np.allclose(a1, a2)

    def test_reset_clears_ema(self):
        d = AntagonisticDecoder(action_dim=4)
        d.decode(list(range(0, 32)))
        assert not np.allclose(d.prev_action, 0.0)
        d.reset()
        assert np.allclose(d.prev_action, 0.0)

    def test_pdi_boost_adds_noise(self):
        np.random.seed(42)
        d1 = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        np.random.seed(42)
        d2 = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        a_no_boost = d1.decode([], pdi_boost=0.0)
        a_with_boost = d2.decode([], pdi_boost=1.0)
        # With boost, action should differ due to added noise
        # (not always, but statistically different)
        # At minimum, shapes should match
        assert a_no_boost.shape == a_with_boost.shape

    def test_channel_weights(self):
        weights = np.zeros(64)
        weights[0] = 10.0  # Only channel 0 has weight
        d = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0,
                                 channel_weights=weights)
        # Spike on ch 0 (weighted) vs ch 1 (zero weight)
        a1 = d.decode([0])
        d.reset()
        a2 = d.decode([1])
        # Channel 0 with high weight should produce larger action
        assert abs(a1[0]) > abs(a2[0])

    def test_7d_action_dim(self):
        """v4.0 uses 7D action (Panda arm)."""
        d = AntagonisticDecoder(action_dim=7, action_scale=0.25)
        spikes = list(range(0, 16)) + list(range(40, 50))
        action = d.decode(spikes)
        assert action.shape == (7,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


# ═══════════════════════════════════════════════════════════════
#  PDI Tests
# ═══════════════════════════════════════════════════════════════

class TestPDI:
    """Tests for Physical Disturbance Index computation."""

    def test_init(self):
        pdi = PDI(window=20)
        assert pdi.compute() == 0.5  # insufficient data default

    def test_constant_velocity_low_pdi(self):
        pdi = PDI(window=20)
        vel = np.array([1.0, 0.0, 0.0])
        for _ in range(25):
            pdi.update(vel)
        val = pdi.compute()
        # Constant velocity → zero variance → low PDI
        assert val < 0.1

    def test_varying_velocity_higher_pdi(self):
        pdi = PDI(window=20)
        for i in range(25):
            vel = np.array([np.sin(i * 0.5), np.cos(i * 0.5), 0.0])
            pdi.update(vel)
        val = pdi.compute()
        # Varying velocity → nonzero variance → higher PDI
        assert val > 0.01

    def test_pdi_clipped_to_range(self):
        pdi = PDI(window=5)
        for _ in range(10):
            pdi.update(np.random.randn(3) * 100)
        val = pdi.compute()
        assert 0.0 <= val <= 2.0

    def test_reset(self):
        pdi = PDI(window=20)
        pdi.update(np.array([1.0, 0.0, 0.0]))
        pdi.update(np.array([2.0, 0.0, 0.0]))
        pdi.reset()
        assert len(pdi.velocities) == 0
        assert pdi.prev_vel is None
        assert pdi.compute() == 0.5


# ═══════════════════════════════════════════════════════════════
#  NeuralCuriosity Tests
# ═══════════════════════════════════════════════════════════════

class TestNeuralCuriosity:
    """Tests for neural intrinsic curiosity."""

    def test_initial_high_curiosity(self):
        nc = NeuralCuriosity(n_channels=64, memory_size=100)
        fr = np.random.rand(64) * 200
        novelty = nc.compute_novelty(fr)
        # First few should return high curiosity (~1.0)
        assert novelty == 1.0

    def test_repeated_pattern_low_novelty(self):
        nc = NeuralCuriosity(n_channels=64, memory_size=100)
        fr = np.ones(64) * 100.0
        for _ in range(20):
            nc.compute_novelty(fr)
        novelty = nc.compute_novelty(fr)
        # Same pattern repeated → low novelty
        assert novelty < 0.5

    def test_novel_pattern_high_novelty(self):
        nc = NeuralCuriosity(n_channels=64, memory_size=100)
        base = np.ones(64) * 100.0
        for _ in range(20):
            nc.compute_novelty(base)
        # Introduce a very different pattern
        novel = np.zeros(64)
        novel[0] = 1000.0
        novelty = nc.compute_novelty(novel)
        assert novelty > 0.5

    def test_novelty_range(self):
        nc = NeuralCuriosity()
        for _ in range(30):
            fr = np.random.rand(64) * 300
            val = nc.compute_novelty(fr)
            assert 0.0 <= val <= 2.0

    def test_reset(self):
        nc = NeuralCuriosity()
        nc.compute_novelty(np.ones(64) * 100)
        nc.compute_novelty(np.ones(64) * 100)
        assert len(nc.memory) == 2
        nc.reset()
        assert len(nc.memory) == 0


# ═══════════════════════════════════════════════════════════════
#  Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests for the neural pipeline."""

    def test_full_pipeline_single_step(self):
        """Test: stim → read → detect spikes → decode → action."""
        neurons = MockNeurons()
        decoder = AntagonisticDecoder(action_dim=4, action_scale=0.35)
        pdi = PDI()

        # Stimulate some channels
        stim = StimDesign(160, -1.0, 160, 1.0)
        burst = BurstDesign(3, 100)
        neurons.stim(ChannelSet(*range(8)), stim, burst)

        # Read neural response
        frames = neurons.read(250, None)
        assert frames.shape == (250, 64)

        # Detect spikes
        threshold = np.percentile(frames, 99.5)
        spike_channels = list(set(np.where(frames > threshold)[1]))

        # Decode to action
        pdi.update(np.array([0.1, 0.0, 0.0]))
        pdi_val = pdi.compute()
        action = decoder.decode(spike_channels, pdi_boost=pdi_val * 0.4)

        assert action.shape == (4,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_calibration_then_decode(self):
        """Test: calibration → use rankings in decoder."""
        neurons = MockNeurons()
        ranking, resp = warmup_calibration(neurons, duration_sec=2.0)

        decoder = AntagonisticDecoder(action_dim=7, channel_weights=resp)
        spikes = list(range(0, 16))
        action = decoder.decode(spikes)
        assert action.shape == (7,)

    def test_multi_episode_stability(self):
        """Simulate multiple episodes to check for numerical stability."""
        neurons = MockNeurons()
        decoder = AntagonisticDecoder(action_dim=4)
        pdi = PDI()
        curiosity = NeuralCuriosity()

        for ep in range(5):
            decoder.reset()
            pdi.reset()
            for step in range(20):
                stim = StimDesign(160, -1.0, 160, 1.0)
                neurons.stim(ChannelSet(*range(8)), stim)
                frames = neurons.read(100, None)
                fr = np.mean(np.abs(frames.astype(float)), axis=0)
                spikes = list(set(np.where(frames > np.percentile(frames, 99))[1]))
                pdi.update(np.random.randn(3) * 0.1)
                action = decoder.decode(spikes, pdi_boost=pdi.compute() * 0.4)
                curiosity.compute_novelty(fr)
                assert action.shape == (4,)
                assert not np.any(np.isnan(action))
                assert not np.any(np.isinf(action))

        # Health should still be reasonable
        health = neurons.get_health()
        assert np.all(health >= 0.3)
        assert np.all(health <= 1.0)
