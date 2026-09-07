"""Unit tests for the signal region detection module."""

import numpy as np

from src.core.enums import ConfidenceLevel, DetectionMethod
from src.dsp.detection import DetectionConfig, detect_signal_regions


def _tone(freq: float, sr: float = 10000.0, n: int = 16384) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return (np.exp(1j * 2 * np.pi * freq * t)).astype(np.complex64)


class TestSignalDetection:
    def test_detects_single_tone(self):
        sr = 10000.0
        sig = _tone(1500.0, sr=sr)
        # Add a bit of noise
        noise = (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))) * 0.05
        sig += noise.astype(np.complex64)

        config = DetectionConfig(threshold_db=15.0, min_bandwidth_hz=10.0, fft_size=1024)
        regions = detect_signal_regions(sig, sr, config)
        
        assert len(regions) == 1
        r = regions[0]
        assert r.snr_db > 15.0

    def test_detects_multiple_tones(self):
        sr = 10000.0
        sig1 = _tone(1500.0, sr=sr)
        sig2 = _tone(-2500.0, sr=sr)
        sig = sig1 + sig2 * 0.5
        noise = (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))) * 0.01
        sig += noise.astype(np.complex64)

        # Small merge gap so they don't merge
        config = DetectionConfig(threshold_db=10.0, min_bandwidth_hz=5.0, merge_gap_hz=500.0, fft_size=1024)
        regions = detect_signal_regions(sig, sr, config)
        
        assert len(regions) >= 2

    def test_merges_close_peaks(self):
        sr = 10000.0
        sig1 = _tone(1000.0, sr=sr)
        sig2 = _tone(1200.0, sr=sr)  # Very close
        sig = sig1 + sig2
        
        # Merge gap is 500 Hz, so 1000 and 1200 should merge
        config = DetectionConfig(threshold_db=10.0, min_bandwidth_hz=10.0, merge_gap_hz=500.0, fft_size=1024)
        regions = detect_signal_regions(sig, sr, config)
        
        assert len(regions) == 1

    def test_noise_only(self):
        sr = 10000.0
        sig = (np.random.randn(16384) + 1j * np.random.randn(16384)).astype(np.complex64)
        
        config = DetectionConfig(threshold_db=15.0)
        regions = detect_signal_regions(sig, sr, config)
        
        assert len(regions) == 0

    def test_too_short(self):
        sr = 10000.0
        sig = _tone(1000.0, sr=sr, n=100) # shorter than fft_size
        
        config = DetectionConfig(fft_size=1024)
        regions = detect_signal_regions(sig, sr, config)
        
        assert len(regions) == 0
