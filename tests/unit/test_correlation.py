"""Tests for bit-stream correlation and framing."""

import numpy as np
import pytest

from src.decoding.correlation import (
    bit_autocorrelation,
    bits_to_hex,
    bits_to_string,
    detect_frame_period,
    find_sync_word,
    parse_pattern,
    split_frames,
)

SYNC = "0x1ACFFC1D"


@pytest.fixture
def framed():
    np.random.seed(8)
    sync = parse_pattern(SYNC)
    parts = [np.random.randint(0, 2, 37).astype(np.uint8)]
    payloads = []
    for _ in range(20):
        hdr = np.random.randint(0, 2, 16).astype(np.uint8)
        pl = np.random.randint(0, 2, 200).astype(np.uint8)
        payloads.append((hdr, pl))
        parts += [sync, hdr, pl]
    return np.concatenate(parts), payloads


class TestPattern:
    def test_hex(self):
        assert parse_pattern("0xA5").tolist() == [1, 0, 1, 0, 0, 1, 0, 1]

    def test_binary(self):
        assert parse_pattern("1011 0").tolist() == [1, 0, 1, 1, 0]

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_pattern("hello")

    def test_render(self):
        b = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1])
        assert bits_to_hex(b) == "A5"
        assert bits_to_string(b) == "101001011"


class TestSyncSearch:
    def test_finds_all(self, framed):
        stream, _ = framed
        res = find_sync_word(stream, SYNC)
        assert len(res.matches) == 20
        assert res.matches[0].position == 37
        assert res.best_period == 248

    def test_tolerates_errors(self, framed):
        stream, _ = framed
        noisy = stream.copy()
        noisy[37 + 3] ^= 1  # corrupt first sync word by one bit
        assert len(find_sync_word(noisy, SYNC, max_errors=0).matches) == 19
        assert len(find_sync_word(noisy, SYNC, max_errors=1).matches) == 20

    def test_inverted_stream(self, framed):
        stream, _ = framed
        res = find_sync_word(stream ^ 1, SYNC)
        assert len(res.matches) == 20 and all(m.inverted for m in res.matches)
        assert not find_sync_word(stream ^ 1, SYNC, allow_inverted=False).matches

    def test_split_frames(self, framed):
        stream, payloads = framed
        res = find_sync_word(stream, SYNC)
        frames = split_frames(stream, res, header_bits=16, frame_bits=248)
        assert len(frames) == 20
        assert len(frames[0].header) == 48
        assert np.array_equal(frames[0].header[32:], payloads[0][0])
        assert np.array_equal(frames[0].payload, payloads[0][1])

    def test_split_frames_undoes_inversion(self, framed):
        stream, payloads = framed
        res = find_sync_word(stream ^ 1, SYNC)
        frames = split_frames(stream ^ 1, res, header_bits=16, frame_bits=248)
        assert np.array_equal(frames[0].payload, payloads[0][1])


class TestPeriod:
    def test_autocorrelation_normalised(self):
        b = np.random.randint(0, 2, 1000)
        ac = bit_autocorrelation(b, 50)
        assert np.isclose(ac[0], 1.0) and len(ac) == 51

    def test_detects_frame_period(self, framed):
        stream, _ = framed
        cands = detect_frame_period(stream)
        assert cands and cands[0].period == 248

    def test_random_has_no_period(self):
        np.random.seed(1)
        cands = detect_frame_period(np.random.randint(0, 2, 5000))
        assert not cands or cands[0].strength < 0.1
