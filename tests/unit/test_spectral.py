"""Unit tests for DSP spectral analysis math."""

import numpy as np

from src.dsp.spectral import (
    analyze_spectrum,
    compute_psd,
    compute_spectrogram,
    detect_peaks,
    estimate_noise_floor,
)


class TestSpectralMath:
    def test_compute_psd(self):
        sr = 1000.0
        t = np.arange(1000) / sr
        # Complex sinusoid at 100 Hz
        samples = np.exp(1j * 2 * np.pi * 100 * t).astype(np.complex64)

        freqs, psd = compute_psd(samples, sr, fft_size=512, window="hann")
        assert len(freqs) == 512
        assert len(psd) == 512

        # Peak should be at 100 Hz
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - 100) < (sr / 512) * 2  # within a bin or two

    def test_estimate_noise_floor(self):
        # Create fake PSD with a noise floor of -80 dB and some peaks
        psd = np.full(1000, -80.0)
        psd[500] = -10.0
        psd[600] = -20.0

        noise = estimate_noise_floor(psd)
        assert abs(noise - (-80.0)) < 1.0

    def test_detect_peaks(self):
        freqs = np.linspace(-500, 500, 1000)
        psd = np.full(1000, -80.0)
        # Put a 50 dB peak at index 500 (freq=0)
        psd[500] = -30.0
        # Put a 20 dB peak at index 750 (freq=250)
        psd[750] = -60.0

        peaks = detect_peaks(freqs, psd, noise_floor_db=-80.0, threshold_db=10.0)
        assert len(peaks) == 2
        assert peaks[0].power_db == -30.0
        assert peaks[0].snr_db == 50.0

    def test_compute_spectrogram(self):
        sr = 1000.0
        t = np.arange(4000) / sr
        samples = np.exp(1j * 2 * np.pi * 100 * t).astype(np.complex64)

        times, freqs, sxx = compute_spectrogram(
            samples, sr, fft_size=256, overlap_frac=0.5, window="hann"
        )
        assert len(freqs) == 256
        assert len(times) > 10
        assert sxx.shape == (len(freqs), len(times))

    def test_analyze_spectrum(self):
        sr = 1000.0
        t = np.arange(10000) / sr
        samples = np.exp(1j * 2 * np.pi * 100 * t).astype(np.complex64)

        result = analyze_spectrum(samples, sr)
        assert len(result.peaks) >= 1
        # Highest peak should be the 100 Hz one
        best_peak = max(result.peaks, key=lambda p: p.power_db)
        assert abs(best_peak.frequency_hz - 100) < 5.0
        assert result.occupied_bandwidth_hz > 0
