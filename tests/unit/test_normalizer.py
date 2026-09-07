"""Unit tests for the raw bytes normalizer."""

import numpy as np

from src.core.enums import IQOrder, SampleDatatype
from src.ingestion.normalizer import normalize_samples


class TestNormalizer:
    def test_cf32_le(self):
        # 1.0 + 2.0j
        raw = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        res = normalize_samples(raw, SampleDatatype.CF32_LE)
        assert res.dtype == np.complex64
        assert len(res) == 1
        assert res[0] == 1.0 + 2.0j

    def test_ci16_le(self):
        # 32767 + 0j
        raw = np.array([32767, 0], dtype=np.int16).tobytes()
        res = normalize_samples(raw, SampleDatatype.CI16_LE)
        assert res.dtype == np.complex64
        assert len(res) == 1
        assert res[0] == 32767.0 + 0.0j

    def test_cu8(self):
        # 255 + 0j (which is 127.5 max, mapped to approx +1.0)
        # 127 + 127j (which is approx 0)
        raw = np.array([255, 0, 127, 127], dtype=np.uint8).tobytes()
        res = normalize_samples(raw, SampleDatatype.CU8)
        assert res.dtype == np.complex64
        assert len(res) == 2
        assert res[0].real == 127.0
        assert res[0].imag == -128.0
        assert res[1].real == -1.0
        assert res[1].imag == -1.0

    def test_ci8(self):
        # 127 + -128j
        raw = np.array([127, -128], dtype=np.int8).tobytes()
        res = normalize_samples(raw, SampleDatatype.CI8)
        assert res.dtype == np.complex64
        assert len(res) == 1
        assert res[0].real > 0.9
        assert res[0].imag < -0.9

    def test_iq_order_qi(self):
        raw = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        res = normalize_samples(raw, SampleDatatype.CF32_LE, iq_order=IQOrder.QI)
        assert res[0] == 2.0 + 1.0j

    def test_invalid_length(self):
        # 3 floats (not a multiple of 2)
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        # Should drop the last float safely
        res = normalize_samples(raw, SampleDatatype.CF32_LE)
        assert len(res) == 1
