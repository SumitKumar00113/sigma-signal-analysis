"""Tests for the feature-based modulation classifier."""

import numpy as np
import pytest

from src.core.enums import ModulationType
from src.dsp.classification import (
    amplitude_level_count,
    classify_modulation,
    cumulant_features,
    envelope_features,
)
from tests.signals.channels import add_awgn
from tests.signals.generators import generate_2fsk, generate_bpsk, generate_qam16, generate_qpsk


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(7)


def _gen_4fsk(nsym: int, rs: float, fs: float, dev: float, snr: float) -> np.ndarray:
    syms = np.random.randint(0, 4, nsym)
    lv = np.array([-3, -1, 1, 3])[syms]
    sps = int(fs / rs)
    up = np.repeat(lv, sps)
    ph = np.cumsum(2 * np.pi * up * dev / fs)
    return add_awgn(np.exp(1j * ph).astype(np.complex64), snr)


class TestFeatures:
    def test_cumulants_theoretical_values(self):
        n = 20000
        bpsk = (2 * np.random.randint(0, 2, n) - 1).astype(complex)
        qpsk = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.random.randint(0, 4, n)))
        c_b = cumulant_features(bpsk)
        c_q = cumulant_features(qpsk)
        assert np.isclose(c_b["c20"], 1.0, atol=0.05)
        assert np.isclose(c_b["c40"], 2.0, atol=0.1)
        assert np.isclose(c_b["c42"], -2.0, atol=0.1)
        assert c_q["c20"] < 0.05
        assert np.isclose(c_q["c40"], 1.0, atol=0.05)
        assert np.isclose(c_q["c42"], -1.0, atol=0.05)

    def test_envelope_cv_constant_vs_shaped(self):
        fsk, _ = generate_2fsk(1000, 8e3, 240e3, 20e3)
        bpsk, _ = generate_bpsk(1000, 10e3, 200e3)
        assert envelope_features(fsk)["env_cv"] < 0.05
        assert envelope_features(bpsk)["env_cv"] > 0.3

    def test_amplitude_rings(self):
        levels = np.array([-3, -1, 1, 3])
        pts = np.array([complex(i, q) for i in levels for q in levels])
        syms = np.repeat(pts, 200)
        assert amplitude_level_count(syms) == 3
        assert amplitude_level_count(np.exp(1j * np.random.rand(2000) * 6.28)) == 1


class TestClassify:
    @pytest.mark.parametrize("snr,cfo", [(15.0, 1200.0), (3.0, 2000.0)])
    def test_bpsk(self, snr, cfo):
        sig, _ = generate_bpsk(4000, 10e3, 200e3, snr, cfo)
        r = classify_modulation(sig, 200e3, 10e3)
        assert r.modulation == ModulationType.BPSK
        assert r.confidence > 0.8

    @pytest.mark.parametrize("snr", [12.0, 6.0])
    def test_qpsk(self, snr):
        sig, _ = generate_qpsk(4000, 12.5e3, 250e3, snr)
        r = classify_modulation(sig, 250e3, 12.5e3)
        assert r.modulation == ModulationType.QPSK
        assert r.confidence > 0.6

    @pytest.mark.parametrize("snr", [20.0, 14.0])
    def test_qam16(self, snr):
        sig, _ = generate_qam16(4000, 10e3, 200e3, snr)
        r = classify_modulation(sig, 200e3, 10e3)
        assert r.modulation == ModulationType.QAM16

    @pytest.mark.parametrize("snr", [14.0, 6.0])
    def test_fsk2(self, snr):
        sig, _ = generate_2fsk(4000, 8e3, 2.4e6, 20e3, snr)
        r = classify_modulation(sig, 2.4e6, 8e3)
        assert r.modulation == ModulationType.FSK2

    def test_fsk4(self):
        sig = _gen_4fsk(4000, 8e3, 2.4e6, 10e3, 14.0)
        r = classify_modulation(sig, 2.4e6, 8e3)
        assert r.modulation == ModulationType.FSK4

    def test_candidates_ranked_and_evidence_present(self):
        sig, _ = generate_qpsk(4000, 12.5e3, 250e3, 12.0)
        r = classify_modulation(sig, 250e3, 12.5e3)
        probs = [p for _, p in r.candidates]
        assert probs == sorted(probs, reverse=True)
        assert r.evidence
        assert "c40" in r.features

    def test_insufficient_input(self):
        r = classify_modulation(np.zeros(100, dtype=np.complex64), 1e5, 1e3)
        assert r.modulation == ModulationType.UNKNOWN
