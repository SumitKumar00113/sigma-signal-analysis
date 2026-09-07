"""Unit tests for the validator engine."""

import numpy as np

from src.ingestion.validator import validate_samples
from src.core.models import FileValidationReport


class TestValidator:
    def test_quality_perfect(self):
        # Sine wave with amplitude envelope to avoid 100% clipping
        t = np.arange(1000)
        envelope = np.linspace(0, 1, 1000)
        samples = (np.exp(1j * 2 * np.pi * 0.1 * t) * envelope).astype(np.complex64)
        report = validate_samples(samples, FileValidationReport())

        assert report.nan_count == 0
        assert report.inf_count == 0
        assert report.clipping_percentage < 2.0  # Only the very end is > 0.99
        assert abs(report.dc_offset_i) < 0.05
        assert abs(report.dc_offset_q) < 0.05
        assert abs(report.iq_imbalance_db) < 0.5

    def test_nans_and_infs(self):
        samples = np.array([1+1j, complex(np.nan, 0), complex(0, np.inf), 0+0j], dtype=np.complex64)
        report = validate_samples(samples, FileValidationReport())
        assert report.nan_count == 1
        assert report.inf_count == 1
        assert len(report.warnings) >= 1

    def test_clipping(self):
        # Generate some clipped samples (absolute value >= 0.99)
        samples = np.array([1.0+0j, -1.0+0j, 0.5+0j, 0.1+0.1j], dtype=np.complex64)
        report = validate_samples(samples, FileValidationReport())
        assert report.clipping_percentage == 50.0  # 2 out of 4

    def test_dc_offset(self):
        t = np.arange(1000)
        samples = np.exp(1j * 2 * np.pi * 0.1 * t).astype(np.complex64)
        samples.real += 0.5  # Add massive I DC offset
        report = validate_samples(samples, FileValidationReport())
        assert abs(report.dc_offset_i - 0.5) < 0.05
        assert "DC offset detected" in str(report.warnings)

    def test_iq_imbalance(self):
        t = np.arange(1000)
        # Q is half the amplitude of I
        samples = np.cos(2 * np.pi * 0.1 * t) + 1j * 0.5 * np.sin(2 * np.pi * 0.1 * t)
        samples = samples.astype(np.complex64)
        report = validate_samples(samples, FileValidationReport())
        # Power of I should be 4x power of Q (6 dB difference)
        assert report.iq_imbalance_db > 5.0
        assert "IQ gain imbalance" in str(report.warnings)
