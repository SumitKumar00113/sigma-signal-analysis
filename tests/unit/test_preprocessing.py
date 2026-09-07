"""Unit tests for DSP preprocessing utilities."""

import numpy as np

from src.dsp.preprocessing import (
    apply_bandpass,
    apply_lowpass,
    normalize_peak,
    normalize_percentile,
    normalize_rms,
    remove_dc_highpass,
    remove_dc_mean,
    translate_frequency,
)


def _tone(freq: float, sr: float = 10000.0, n: int = 4000) -> np.ndarray:
    """Generate a complex sinusoid test signal."""
    t = np.arange(n, dtype=np.float64) / sr
    return (0.8 * np.exp(1j * 2 * np.pi * freq * t)).astype(np.complex64)


# ---------------------------------------------------------------------------
# DC offset removal
# ---------------------------------------------------------------------------


class TestRemoveDCMean:
    def test_removes_dc(self):
        samples = _tone(100.0) + (0.5 + 0.3j)  # inject DC
        cleaned = remove_dc_mean(samples.astype(np.complex64))
        assert abs(np.mean(cleaned.real)) < 0.01
        assert abs(np.mean(cleaned.imag)) < 0.01

    def test_preserves_dtype(self):
        out = remove_dc_mean(_tone(100.0))
        assert out.dtype == np.complex64

    def test_preserves_signal_energy(self):
        sig = _tone(200.0)
        cleaned = remove_dc_mean(sig)
        # Energy should be roughly preserved (only DC removed)
        assert abs(np.mean(np.abs(cleaned) ** 2) - np.mean(np.abs(sig) ** 2)) < 0.01


class TestRemoveDCHighpass:
    def test_removes_dc(self):
        sig = _tone(500.0, sr=10000.0) + (1.0 + 0.5j)
        cleaned = remove_dc_highpass(sig.astype(np.complex64), 10000.0, cutoff_hz=50.0)
        # After settling transient, DC should be removed
        tail = cleaned[1000:]
        assert abs(np.mean(tail.real)) < 0.15
        assert abs(np.mean(tail.imag)) < 0.15

    def test_preserves_signal(self):
        sig = _tone(1000.0, sr=10000.0)
        cleaned = remove_dc_highpass(sig, 10000.0, cutoff_hz=10.0)
        # 1000 Hz signal should mostly pass through
        tail = cleaned[500:]
        power_ratio = np.mean(np.abs(tail) ** 2) / np.mean(np.abs(sig[500:]) ** 2)
        assert power_ratio > 0.9

    def test_cutoff_clamp(self):
        """Cutoff >= Nyquist should be clamped safely."""
        sig = _tone(100.0, sr=1000.0)
        # cutoff_hz = 600 > Nyquist = 500, should not crash
        cleaned = remove_dc_highpass(sig, 1000.0, cutoff_hz=600.0)
        assert len(cleaned) == len(sig)


# ---------------------------------------------------------------------------
# Gain normalization
# ---------------------------------------------------------------------------


class TestNormalizeRMS:
    def test_target_rms(self):
        sig = _tone(100.0)
        normed = normalize_rms(sig, target_rms=0.5)
        rms = np.sqrt(np.mean(np.abs(normed) ** 2))
        assert abs(rms - 0.5) < 0.01

    def test_zero_input(self):
        zeros = np.zeros(100, dtype=np.complex64)
        out = normalize_rms(zeros, target_rms=1.0)
        assert np.allclose(out, 0)

    def test_preserves_dtype(self):
        assert normalize_rms(_tone(100.0)).dtype == np.complex64


class TestNormalizePeak:
    def test_target_peak(self):
        sig = _tone(100.0)
        normed = normalize_peak(sig, target_peak=0.5)
        assert abs(np.max(np.abs(normed)) - 0.5) < 0.01

    def test_zero_input(self):
        zeros = np.zeros(100, dtype=np.complex64)
        out = normalize_peak(zeros, target_peak=1.0)
        assert np.allclose(out, 0)


class TestNormalizePercentile:
    def test_percentile_scaling(self):
        sig = _tone(100.0)
        normed = normalize_percentile(sig, percentile=99.0, target=0.7)
        p99 = np.percentile(np.abs(normed), 99)
        assert abs(p99 - 0.7) < 0.05

    def test_zero_input(self):
        zeros = np.zeros(100, dtype=np.complex64)
        out = normalize_percentile(zeros)
        assert np.allclose(out, 0)


# ---------------------------------------------------------------------------
# Frequency translation
# ---------------------------------------------------------------------------


class TestTranslateFrequency:
    def test_shift_up(self):
        sr = 10000.0
        sig = _tone(100.0, sr=sr, n=10000)
        shifted = translate_frequency(sig, sr, shift_hz=500.0)
        # PSD peak should be near 600 Hz now
        from src.dsp.spectral import compute_psd

        freqs, psd = compute_psd(shifted, sr, fft_size=4096, window="hann")
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 600.0) < 10.0

    def test_shift_down(self):
        sr = 10000.0
        sig = _tone(500.0, sr=sr, n=10000)
        shifted = translate_frequency(sig, sr, shift_hz=-200.0)
        from src.dsp.spectral import compute_psd

        freqs, psd = compute_psd(shifted, sr, fft_size=4096, window="hann")
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 300.0) < 10.0

    def test_zero_shift(self):
        sig = _tone(100.0)
        shifted = translate_frequency(sig, 10000.0, shift_hz=0.0)
        assert np.allclose(sig, shifted, atol=1e-5)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestApplyLowpass:
    def test_passband(self):
        sr = 10000.0
        sig = _tone(100.0, sr=sr, n=10000)
        filtered = apply_lowpass(sig, sr, cutoff_hz=2000.0, order=64)
        # 100 Hz signal should pass through LPF with 2 kHz cutoff
        tail = filtered[200:]  # skip transient
        power_ratio = np.mean(np.abs(tail) ** 2) / np.mean(np.abs(sig[200:]) ** 2)
        assert power_ratio > 0.8

    def test_stopband(self):
        sr = 10000.0
        sig = _tone(4000.0, sr=sr, n=10000)
        filtered = apply_lowpass(sig, sr, cutoff_hz=500.0, order=64)
        tail = filtered[200:]
        # 4000 Hz signal should be heavily attenuated
        power = np.mean(np.abs(tail) ** 2)
        assert power < 0.1  # significant attenuation


class TestApplyBandpass:
    def test_passband(self):
        sr = 10000.0
        sig = _tone(1000.0, sr=sr, n=10000)
        filtered = apply_bandpass(sig, sr, low_hz=500.0, high_hz=2000.0, order=128)
        tail = filtered[500:]
        power_ratio = np.mean(np.abs(tail) ** 2) / np.mean(np.abs(sig[500:]) ** 2)
        assert power_ratio > 0.5  # within passband

    def test_stopband(self):
        sr = 10000.0
        sig = _tone(100.0, sr=sr, n=10000)
        filtered = apply_bandpass(sig, sr, low_hz=2000.0, high_hz=4000.0, order=128)
        tail = filtered[500:]
        power = np.mean(np.abs(tail) ** 2)
        assert power < 0.1  # outside band, should be attenuated
