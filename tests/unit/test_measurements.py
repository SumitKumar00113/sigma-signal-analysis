"""Unit tests for the signal measurement module."""

import numpy as np

from src.dsp.measurements import (
    compute_instantaneous_amplitude,
    compute_instantaneous_frequency,
    compute_instantaneous_phase,
    estimate_frequency_offset,
    estimate_snr,
    estimate_symbol_rate_candidates,
)


def _tone(freq: float, sr: float = 10000.0, n: int = 16384) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return (np.exp(1j * 2 * np.pi * freq * t)).astype(np.complex64)


class TestMeasurements:
    def test_estimate_snr(self):
        # Pure tone should have high SNR (capped at 100)
        sig = _tone(1000.0)
        snr = estimate_snr(sig)
        assert snr > 80.0

        # Pure noise should have 0 SNR
        noise = (np.random.randn(10000) + 1j * np.random.randn(10000)).astype(np.complex64)
        snr = estimate_snr(noise)
        assert snr < 1.0

    def test_estimate_snr_empty(self):
        assert estimate_snr(np.array([], dtype=np.complex64)) == 0.0

    def test_estimate_frequency_offset(self):
        sr = 10000.0
        sig = _tone(1234.0, sr=sr)

        offset = estimate_frequency_offset(sig, sr, fft_size=4096)
        assert abs(offset - 1234.0) < 10.0

        sig_neg = _tone(-3456.0, sr=sr)
        offset_neg = estimate_frequency_offset(sig_neg, sr, fft_size=4096)
        assert abs(offset_neg - (-3456.0)) < 10.0

    def test_estimate_symbol_rate(self):
        # Generate a BPSK-like signal with amplitude dips at symbol transitions
        sr = 48000.0
        sym_rate = 2400.0
        sps = int(sr / sym_rate)
        n_syms = 1000

        # Random symbols +1, -1
        syms = np.sign(np.random.randn(n_syms))

        # Upsample
        upsampled = np.zeros(n_syms * sps)
        upsampled[::sps] = syms

        # Pulse shaping (simple RC filter approximation)
        t = np.arange(-sps, sps+1)
        pulse = (np.sinc(t / sps) * np.cos(np.pi * 0.5 * t / sps)
                 / (1 - (2 * 0.5 * t / sps)**2 + 1e-10))

        sig = np.convolve(upsampled, pulse, mode='same').astype(np.complex64)

        candidates = estimate_symbol_rate_candidates(sig, sr)

        # Ensure it returns something
        assert len(candidates) > 0

        # The true rate (2400) should be near the top candidate
        rates = [c[0] for c in candidates]
        found = any(abs(r - sym_rate) < 100.0 for r in rates)
        assert found

    def test_instantaneous_metrics(self):
        sr = 10000.0
        t = np.arange(10000) / sr

        # Signal with AM and FM
        # AM: 1 + 0.5*sin(2*pi*10*t)
        # FM: 100 Hz center + 20 Hz deviation at 5 Hz rate
        phase = 2 * np.pi * 100 * t - (20 / 5) * np.cos(2 * np.pi * 5 * t)
        amp = 1.0 + 0.5 * np.sin(2 * np.pi * 10 * t)

        sig = (amp * np.exp(1j * phase)).astype(np.complex64)

        inst_amp = compute_instantaneous_amplitude(sig)
        inst_freq = compute_instantaneous_frequency(sig, sr)
        inst_phase = compute_instantaneous_phase(sig)

        assert len(inst_amp) == len(sig)
        assert len(inst_freq) == len(sig)
        assert len(inst_phase) == len(sig)

        # Amp should be around 1.0 mean
        assert abs(np.mean(inst_amp) - 1.0) < 0.1

        # Freq should be around 100 mean
        assert abs(np.mean(inst_freq) - 100.0) < 2.0


class TestNewEstimators:
    """PSD SNR and FSK-aware symbol-rate estimation."""

    def test_snr_psd_matches_awgn_definition(self):
        from src.dsp.measurements import estimate_snr_psd
        from tests.signals.generators import generate_bpsk
        np.random.seed(0)
        for snr in (3.0, 12.0):
            sig, _ = generate_bpsk(4000, 10e3, 200e3, snr)
            assert abs(estimate_snr_psd(sig, 200e3) - snr) < 1.5

    def test_symbol_rate_fsk(self):
        from src.dsp.measurements import estimate_symbol_rate_candidates
        from tests.signals.generators import generate_2fsk
        np.random.seed(0)
        sig, _ = generate_2fsk(4000, 8e3, 2.4e6, 20e3, 14.0)
        cands = estimate_symbol_rate_candidates(sig, 2.4e6)
        assert abs(cands[0][0] - 8e3) < 50

    def test_symbol_rate_psk(self):
        from src.dsp.measurements import estimate_symbol_rate_candidates
        from tests.signals.generators import generate_qpsk
        np.random.seed(0)
        sig, _ = generate_qpsk(4000, 12.5e3, 250e3, 10.0)
        cands = estimate_symbol_rate_candidates(sig, 250e3)
        assert abs(cands[0][0] - 12.5e3) < 50

    def test_frequency_offset_fsk_symmetric(self):
        from src.dsp.measurements import estimate_frequency_offset
        from tests.signals.generators import generate_2fsk
        np.random.seed(0)
        sig, _ = generate_2fsk(4000, 8e3, 2.4e6, 20e3, 14.0)
        assert abs(estimate_frequency_offset(sig, 2.4e6)) < 1500


def test_rate_candidates_merge_does_not_crash(monkeypatch):
    """Regression: the agreement bonus looked up a merged key with a bare
    next() and raised StopIteration when the stronger rate had drifted."""
    import src.dsp.measurements as m

    monkeypatch.setattr(m, "estimate_symbol_rate_envelope", lambda *a: [(1000.0, 0.3)])
    monkeypatch.setattr(m, "estimate_symbol_rate_instfreq", lambda *a: [(1009.0, 0.5)])
    monkeypatch.setattr(m, "estimate_symbol_rate_transitions", lambda *a: [(1018.0, 0.9)])
    monkeypatch.setattr(m, "estimate_symbol_rate_quadrature", lambda *a: [])
    out = m.estimate_symbol_rate_candidates(np.zeros(4096, dtype=np.complex64), 48000.0)
    assert out and 990 < out[0][0] < 1030
