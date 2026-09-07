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
        pulse = np.sinc(t / sps) * np.cos(np.pi * 0.5 * t / sps) / (1 - (2 * 0.5 * t / sps)**2 + 1e-10)
        
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
