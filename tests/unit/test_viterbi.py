"""Tests for convolutional encoding and Viterbi decoding."""

import numpy as np
import pytest

from src.decoding.viterbi import STANDARD_CODES, ConvCode, conv_encode, viterbi_decode


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(3)


class TestEncoder:
    def test_k3_known_vector(self):
        # K=3 (7,5): input 1 0 1 1 → textbook output 11 10 00 01 (then flush)
        code = ConvCode(3, (0o7, 0o5))
        out = conv_encode(np.array([1, 0, 1, 1]), code, terminate=False)
        assert out.tolist() == [1, 1, 1, 0, 0, 0, 0, 1]

    def test_rate_and_lengths(self):
        code = ConvCode.standard("K7 r1/2 (NASA/CCSDS 171,133)")
        bits = np.random.randint(0, 2, 100)
        out = conv_encode(bits, code)
        assert code.rate == 0.5
        assert len(out) == 2 * (100 + 6)

    def test_puncturing_rate(self):
        code = ConvCode(7, (0o171, 0o133), puncture=(1, 1, 0, 1, 1, 0))
        assert np.isclose(code.rate, 0.75)
        out = conv_encode(np.random.randint(0, 2, 300), code, terminate=False)
        assert len(out) == 400


class TestDecoder:
    @pytest.mark.parametrize("name", list(STANDARD_CODES))
    def test_clean_roundtrip(self, name):
        code = ConvCode.standard(name)
        bits = np.random.randint(0, 2, 400).astype(np.uint8)
        r = viterbi_decode(conv_encode(bits, code), code)
        assert np.array_equal(r.bits, bits)
        assert r.estimated_errors == 0

    def test_corrects_errors_k7(self):
        code = ConvCode.standard("K7 r1/2 (NASA/CCSDS 171,133)")
        bits = np.random.randint(0, 2, 2000).astype(np.uint8)
        coded = conv_encode(bits, code)
        flip = np.random.rand(len(coded)) < 0.04
        rx = coded ^ flip.astype(np.uint8)
        r = viterbi_decode(rx, code)
        # K=7 hard-decision at 4 % channel BER: theory gives ~1e-3 output BER
        assert np.mean(r.bits != bits) < 0.006
        assert abs(r.estimated_errors - flip.sum()) <= 5

    def test_soft_decision_beats_hard(self):
        code = ConvCode.standard("K3 r1/2 (7,5)")
        bits = np.random.randint(0, 2, 3000).astype(np.uint8)
        coded = conv_encode(bits, code)
        soft = (1 - 2 * coded.astype(float)) + 0.9 * np.random.randn(len(coded))
        hard = (soft < 0).astype(np.uint8)
        r_soft = viterbi_decode(soft, code, soft=True)
        r_hard = viterbi_decode(hard, code)
        assert np.mean(r_soft.bits != bits) <= np.mean(r_hard.bits != bits)

    def test_punctured_roundtrip(self):
        code = ConvCode(7, (0o171, 0o133), puncture=(1, 1, 0, 1, 1, 0))
        bits = np.random.randint(0, 2, 600).astype(np.uint8)
        r = viterbi_decode(conv_encode(bits, code), code)
        assert np.array_equal(r.bits, bits)

    def test_unterminated(self):
        code = ConvCode.standard("K5 r1/2 (23,35)")
        bits = np.random.randint(0, 2, 500).astype(np.uint8)
        r = viterbi_decode(conv_encode(bits, code, terminate=False), code, terminated=False)
        # All but the last few bits (no flush) must match
        assert np.array_equal(r.bits[:-6], bits[:-6])


class TestErrorCount:
    def test_punctured_erasures_not_counted(self):
        # Erased (punctured) positions used to be counted as channel errors
        code = ConvCode(7, (0o171, 0o133), puncture=(1, 1, 0, 1, 1, 0))
        coded = conv_encode(np.random.randint(0, 2, 600), code, terminate=True)
        assert viterbi_decode(coded, code).estimated_errors == 0

    def test_punctured_counts_real_errors(self):
        code = ConvCode(7, (0o171, 0o133), puncture=(1, 1, 0, 1))
        coded = conv_encode(np.random.randint(0, 2, 600), code, terminate=True)
        coded[[50, 300, 600]] ^= 1
        assert viterbi_decode(coded, code).estimated_errors == 3
