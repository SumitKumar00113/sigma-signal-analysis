"""Unit tests for the SigMF file reader."""

from pathlib import Path

import numpy as np
import pytest

from src.core.enums import FileFormat
from src.ingestion.sigmf_reader import SigMFReader
from tests.fixtures.generate_fixtures import generate_sigmf_pair


@pytest.fixture
def sigmf_pair(tmp_path: Path) -> tuple[Path, Path]:
    return generate_sigmf_pair(tmp_path / "test", sample_rate=2.4e6, duration=0.01)


class TestSigMFReader:
    def test_validate(self, sigmf_pair: tuple[Path, Path]):
        meta_path, data_path = sigmf_pair
        reader = SigMFReader(meta_path)
        report = reader.validate()
        assert report.is_valid
        assert report.exists
        assert report.sample_aligned
        assert report.sha256 != ""

    def test_read_metadata(self, sigmf_pair: tuple[Path, Path]):
        meta_path, _ = sigmf_pair
        reader = SigMFReader(meta_path)
        meta = reader.read_metadata()
        assert meta.source_format == FileFormat.SIGMF
        assert meta.sample_rate_hz == 2.4e6
        assert meta.center_frequency_hz == 145.5e6
        assert meta.sample_count > 0
        assert meta.duration_seconds > 0
        assert "Test signal" in meta.notes

    def test_read_samples(self, sigmf_pair: tuple[Path, Path]):
        meta_path, _ = sigmf_pair
        reader = SigMFReader(meta_path)
        samples = reader.read_samples(0, 1000)
        assert samples.dtype == np.complex64
        assert len(samples) == 1000
        # Check magnitude > 0 (it's a 0.8 mag complex sinusoid)
        assert np.mean(np.abs(samples)) > 0.5

    def test_total_samples(self, sigmf_pair: tuple[Path, Path]):
        meta_path, _ = sigmf_pair
        reader = SigMFReader(meta_path)
        total = reader.total_samples()
        assert total == int(2.4e6 * 0.01)

    def test_invalid_meta(self, tmp_path: Path):
        meta_path = tmp_path / "bad.sigmf-meta"
        meta_path.write_text("{invalid json]")
        reader = SigMFReader(meta_path)
        report = reader.validate()
        assert not report.is_valid
        assert not report.exists

    def test_missing_data(self, sigmf_pair: tuple[Path, Path]):
        meta_path, data_path = sigmf_pair
        data_path.unlink()  # delete the data file
        reader = SigMFReader(meta_path)
        report = reader.validate()
        assert not report.is_valid
        assert not report.exists
