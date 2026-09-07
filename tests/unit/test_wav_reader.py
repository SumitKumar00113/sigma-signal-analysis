"""Unit tests for the WAV file reader."""

import numpy as np
import pytest
from pathlib import Path

from tests.fixtures.generate_fixtures import generate_tone_wav, generate_stereo_iq_wav
from src.core.enums import FileFormat, WavInterpretation
from src.ingestion.wav_reader import WavReader


@pytest.fixture
def mono_wav(tmp_path: Path) -> Path:
    return generate_tone_wav(tmp_path / "tone.wav", sample_rate=48000, duration=0.1)


@pytest.fixture
def stereo_iq_wav(tmp_path: Path) -> Path:
    return generate_stereo_iq_wav(tmp_path / "iq.wav", sample_rate=48000, duration=0.1)


class TestWavReaderMono:
    def test_validate(self, mono_wav: Path):
        reader = WavReader(mono_wav)
        report = reader.validate()
        assert report.is_valid
        assert report.exists
        assert report.header_valid
        assert report.sha256 != ""

    def test_read_metadata(self, mono_wav: Path):
        reader = WavReader(mono_wav)
        meta = reader.read_metadata()
        assert meta.source_format == FileFormat.WAV
        assert meta.sample_rate_hz == 48000
        assert meta.num_channels == 1
        assert meta.sample_count > 0
        assert meta.duration_seconds > 0

    def test_read_samples(self, mono_wav: Path):
        reader = WavReader(mono_wav)
        samples = reader.read_samples(0, 1000)
        assert samples.dtype == np.complex64
        assert len(samples) == 1000
        # Mono → imaginary part should be zero
        assert np.allclose(samples.imag, 0, atol=1e-6)

    def test_total_samples(self, mono_wav: Path):
        reader = WavReader(mono_wav)
        total = reader.total_samples()
        assert total == int(48000 * 0.1)

    def test_read_all(self, mono_wav: Path):
        reader = WavReader(mono_wav)
        samples = reader.read_samples()
        assert len(samples) == reader.total_samples()


class TestWavReaderStereoIQ:
    def test_stereo_iq_interpretation(self, stereo_iq_wav: Path):
        reader = WavReader(stereo_iq_wav, interpretation=WavInterpretation.STEREO_IQ)
        samples = reader.read_samples(0, 1000)
        assert samples.dtype == np.complex64
        assert len(samples) == 1000
        # Complex sinusoid should have both real and imag parts
        assert not np.allclose(samples.imag, 0, atol=0.01)

    def test_metadata_channels(self, stereo_iq_wav: Path):
        reader = WavReader(stereo_iq_wav, interpretation=WavInterpretation.STEREO_IQ)
        meta = reader.read_metadata()
        assert meta.num_channels == 2


class TestWavReaderInvalidFile:
    def test_nonexistent_file(self, tmp_path: Path):
        reader = WavReader(tmp_path / "nonexistent.wav")
        report = reader.validate()
        assert not report.is_valid
        assert not report.exists

    def test_invalid_wav(self, tmp_path: Path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"NOT A WAV FILE AT ALL")
        reader = WavReader(bad)
        report = reader.validate()
        assert not report.header_valid
