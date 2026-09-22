"""Tests for synchronisation primitives."""

import numpy as np
import pytest

from src.dsp import sync
from tests.signals.generators import generate_bpsk, generate_qpsk


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(1234)


class TestPulseShaping:
    def test_rrc_unit_energy_and_symmetric(self):
        taps = sync.rrc_taps(8, 0.35, 8)
        assert len(taps) % 2 == 1
        assert np.isclose(np.sum(taps**2), 1.0)
        assert np.allclose(taps, taps[::-1])

    def test_rrc_cascade_is_nyquist(self):
        # RRC * RRC = RC, which has zero ISI at symbol spacing
        sps = 8
        taps = sync.rrc_taps(sps, 0.35, 10)
        rc = np.convolve(taps, taps)
        centre = len(rc) // 2
        peak = rc[centre]
        for k in range(1, 4):
            assert abs(rc[centre + k * sps]) < 0.02 * peak


class TestResample:
    def test_resample_to_target_sps(self):
        sig, _ = generate_bpsk(500, 10e3, 200e3)
        out, sps = sync.resample_to_sps(sig, 200e3, 10e3, target_sps=8)
        assert np.isclose(sps, 8.0)
        assert abs(len(out) - len(sig) * 8 / 20) <= 2

    def test_resample_noop_when_already_target(self):
        sig, _ = generate_bpsk(200, 10e3, 80e3)
        out, sps = sync.resample_to_sps(sig, 80e3, 10e3, target_sps=8)
        assert sps == 8.0
        assert len(out) == len(sig)


class TestGardner:
    @pytest.mark.parametrize("offset", [0, 3, 5, 11])
    def test_converges_on_bpsk(self, offset):
        sig, _ = generate_bpsk(3000, 10e3, 200e3)
        x, sps = sync.resample_to_sps(sig[offset:], 200e3, 10e3, 8)
        x = sync.matched_filter(x, sps)
        res = sync.gardner_timing_recovery(x, sps)
        assert len(res.symbols) > 2500
        tail = res.symbols[-800:]
        # Well-timed BPSK has near-constant |real| and tiny imaginary part
        spread = np.std(np.abs(tail.real)) / np.mean(np.abs(tail.real))
        assert spread < 0.2
        assert np.isclose(res.period_estimate, 8.0, atol=0.2)

    def test_too_short_returns_empty(self):
        res = sync.gardner_timing_recovery(np.ones(10, dtype=complex), 8)
        assert len(res.symbols) == 0

    def test_maximum_energy_timing(self):
        sig, _ = generate_bpsk(500, 10e3, 200e3)
        syms, phase = sync.maximum_energy_timing(sig[3:], 20)
        assert len(syms) >= 499
        assert phase == 17  # 20 - 3


class TestCarrier:
    @pytest.mark.parametrize("order,cfo", [(2, 0.01), (4, 0.005), (4, -0.02)])
    def test_mpower_cfo_estimate(self, order, cfo):
        n = 4000
        k = np.random.randint(0, order, n)
        syms = np.exp(1j * 2 * np.pi * k / order) * np.exp(1j * 2 * np.pi * cfo * np.arange(n))
        est = sync.estimate_cfo_mpower(syms, order)
        assert abs(est - cfo) < 2e-4

    def test_costas_locks_qpsk(self):
        sig, _ = generate_qpsk(3000, 12.5e3, 250e3, snr_db=15.0)
        x, sps = sync.resample_to_sps(sig, 250e3, 12.5e3, 8)
        x = sync.matched_filter(x, sps)
        syms = sync.gardner_timing_recovery(x, sps).symbols
        # Add a static phase + small CFO
        n = np.arange(len(syms))
        syms = syms * np.exp(1j * (0.7 + 2 * np.pi * 0.002 * n))
        res = sync.costas_loop(syms, 4)
        assert abs(res.cfo_cycles_per_symbol - 0.002) < 5e-4
        assert res.lock_metric < 0.25
        # Corrected symbols cluster near the 4 QPSK points
        tail = res.symbols[-500:]
        ang = np.angle(tail) % (np.pi / 2)
        dev = np.minimum(np.abs(ang - np.pi / 4), np.pi / 4 - np.abs(ang - np.pi / 4))
        assert np.mean(dev) < 0.3

    def test_evm_zero_for_perfect(self):
        s = np.array([1, -1, 1j, -1j])
        assert sync.evm_percent(s, s) == 0.0
