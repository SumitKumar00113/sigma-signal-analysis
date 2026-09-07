"""Unit tests for the raw IQ file reader."""

import numpy as np
import pytest
from pathlib import Path

from tests.fixtures.generate_fixtures import generate_raw_iq_cf32, generate_raw_iq_cu8
from src.core.enums import FileFormat, IQOrder, SampleDatatype
from src.ingestion.raw_iq_reader import RawIQReader


@pytest.fixture
def cf32_file(tmp_path: Path) -> Path:
    return generate_raw_iq_cf32(tmp_path / "test.cf32", sample_rate=2.4e6, duration=0.01)


@pytest.fixture
def cu8_file(tmp_path: Path) -> Path:
    return generate_raw_iq_cu8(tmp_path / "test.cu8", sample_rate=2.4e6, duration=0.01)


class TestRawIQReaderCF32:
    def test_validate(self, cf32_file: Path):
        reader = RawIQReader(cf32_file, SampleDatatype.CF32_LE, sample_rate_hz=2.4e6)
        report = reader.validate()
        assert report.is_valid
        assert report.sample_aligned
        assert report.sha256 != ""

    def test_read_metadata(self, cf32_file: Path):
        reader = RawIQReader(cf32_file, SampleDatatype.CF32_LE, sample_rate_hz=2.4e6)
        meta = reader.read_metadata()
        assert meta.source_format == FileFormat.RAW_IQ
        assert meta.sample_rate_hz == 2.4e6
        assert meta.sample_count > 0

    def test_read_samples(self, cf32_file: Path):
        reader = RawIQReader(cf32_file, SampleDatatype.CF32_LE, sample_rate_hz=2.4e6)
        samples = reader.read_samples(0, 1000)
        assert samples.dtype == np.complex64
        assert len(samples) == 1000
        # Should be a complex sinusoid — magnitude ≈ 0.8
        magnitudes = np.abs(samples)
        assert np.mean(magnitudes) > 0.5

    def test_total_samples(self, cf32_file: Path):
        reader = RawIQReader(cf32_file, SampleDatatype.CF32_LE, sample_rate_hz=2.4e6)
        total = reader.total_samples()
        expected = int(2.4e6 * 0.01)
        assert total == expected

    def test_chunked_read(self, cf32_file: Path):
        reader = RawIQReader(cf32_file, SampleDatatype.CF32_LE, sample_rate_hz=2.4e6)
        chunk = reader.read_chunk(0, 500)
        assert len(chunk) == 500


class TestRawIQReaderCU8:
    def test_read_samples(self, cu8_file: Path):
        reader = RawIQReader(cu8_file, SampleDatatype.CU8, sample_rate_hz=2.4e6)
        samples = reader.read_samples(0, 1000)
        assert samples.dtype == np.complex64
        assert len(samples) == 1000
        # After unsigned centering, mean should be near zero
        assert abs(np.mean(samples.real)) < 5.0

    def test_qi_ordering(self, cu8_file: Path):
        reader_iq = RawIQReader(cu8_file, SampleDatatype.CU8, iq_order=IQOrder.IQ)
        reader_qi = RawIQReader(cu8_file, SampleDatatype.CU8, iq_order=IQOrder.QI)
        s_iq = reader_iq.read_samples(0, 100)
        s_qi = reader_qi.read_samples(0, 100)
        # QI should swap I and Q — not identical to IQ
        assert not np.allclose(s_iq, s_qi)


class TestRawIQReaderEdgeCases:
    def test_nonexistent(self, tmp_path: Path):
        reader = RawIQReader(tmp_path / "nope.iq", SampleDatatype.CF32_LE)
        report = reader.validate()
        assert not report.is_valid

    def test_byte_offset(self, cf32_file: Path):
        reader = RawIQReader(cf32_file, SampleDatatype.CF32_LE, byte_offset=8)
        total = reader.total_samples()
        assert total > 0
        # Should be 1 sample less than no-offset
        reader2 = RawIQReader(cf32_file, SampleDatatype.CF32_LE, byte_offset=0)
        assert reader.total_samples() == reader2.total_samples() - 1

    def test_empty_read(self, cf32_file: Path):
        reader = RawIQReader(cf32_file, SampleDatatype.CF32_LE)
        total = reader.total_samples()
        samples = reader.read_samples(total, 100)
        assert len(samples) == 0
