"""Tests for automatic frame-sync and header discovery."""

import numpy as np
import pytest

from src.decoding.correlation import parse_pattern
from src.decoding.framing import analyse_framing


def _stream(rng, sync, n_frames, period, fixed="", counter_bits=16, ber=0.0, inv_frac=0.0,
            lead=37, first=5):
    s = parse_pattern(sync)
    f = parse_pattern(fixed) if fixed else np.zeros(0, np.uint8)
    out = [rng.integers(0, 2, lead, dtype=np.uint8)]
    for i in range(n_frames):
        c = np.array([((i + first) >> (counter_bits - 1 - j)) & 1 for j in range(counter_bits)],
                     dtype=np.uint8)
        body = np.concatenate([s, f, c])
        frame = np.concatenate([body, rng.integers(0, 2, period - len(body), dtype=np.uint8)])
        if rng.random() < inv_frac:
            frame ^= 1
        out.append(frame)
    x = np.concatenate(out)
    return x ^ (rng.random(len(x)) < ber).astype(np.uint8)


@pytest.mark.parametrize("ber,inv", [(0.0, 0.0), (0.02, 0.3)])
def test_known_sync_with_header_fields(ber, inv):
    rng = np.random.default_rng(1)
    x = _stream(rng, "0x1ACFFC1D", 40, 1024, fixed="0xA5", ber=ber, inv_frac=inv)
    r = analyse_framing(x)
    assert r.found and r.method == "known" and "CCSDS" in r.sync_name
    assert r.sync_hex == "0x1ACFFC1D"
    assert r.period == 1024 and r.regular and len(r.frames) == 40
    assert r.frames[0].start == 37
    kinds = [(f.kind, f.start, f.length) for f in r.fields]
    assert kinds == [("sync", 0, 32), ("fixed", 32, 8), ("counter", 40, 16)]
    assert "first value 5" in r.fields[2].description
    assert r.header_bits == 56 and len(r.frames[1].header) == 56
    if inv:
        assert 0 < r.inverted_frames < 40
        # inversion undone per frame: every header starts with the sync word
        sync = parse_pattern("0x1ACFFC1D")
        assert all(np.sum(fr.header[:32] != sync) <= 3 for fr in r.frames)


def test_blind_discovery_of_unknown_sync():
    rng = np.random.default_rng(2)
    x = _stream(rng, "0xB5E3", 60, 800, counter_bits=8, ber=0.005)
    r = analyse_framing(x)
    assert r.found and r.method == "blind"
    assert r.sync_hex == "0xB5E3" and r.period == 800 and len(r.frames) == 60
    assert [(f.kind, f.start, f.length) for f in r.fields] == [("sync", 0, 16),
                                                               ("counter", 16, 8)]


def test_blind_long_sync_with_fixed_bits():
    rng = np.random.default_rng(3)
    x = _stream(rng, "0xDEADBEEFCAFE", 30, 2000, fixed="0x3C", counter_bits=0) ^ 1
    r = analyse_framing(x)
    assert r.found and r.inverted_frames == 0      # majority polarity is "normal"
    # blindly, the sync word and the constant field that follows are one
    # block; the polarity of a wholly inverted stream cannot be known
    assert r.sync_hex in ("0xDEADBEEFCAFE3C", "0x215241103501C3")
    assert r.period == 2000 and len(r.frames) == 30


def test_short_sync_needs_regular_grid():
    rng = np.random.default_rng(4)
    x = _stream(rng, "0x47", 100, 1504, counter_bits=0)
    r = analyse_framing(x)
    assert r.found and r.sync_name == "MPEG-TS sync byte"
    assert len(r.frames) == 100 and r.regular


@pytest.mark.parametrize("n", [5_000, 50_000, 500_000])
def test_random_data_has_no_framing(n):
    false = 0
    for seed in range(5 if n < 500_000 else 1):
        x = np.random.default_rng(100 + seed).integers(0, 2, n, dtype=np.uint8)
        false += analyse_framing(x).found
    assert false == 0


def test_idle_fill_is_not_a_sync():
    rng = np.random.default_rng(5)
    parts = []
    for _ in range(40):          # fill runs of random length between random data
        fill = np.zeros(rng.integers(50, 300), np.uint8) if rng.random() < 0.5 else \
            np.tile([0, 1], rng.integers(25, 150)).astype(np.uint8)
        parts += [fill, rng.integers(0, 2, rng.integers(100, 600), dtype=np.uint8)]
    assert not analyse_framing(np.concatenate(parts)).found


def test_summary_text():
    rng = np.random.default_rng(6)
    r = analyse_framing(_stream(rng, "0x1ACFFC1D", 10, 512))
    text = r.summary()
    assert "0x1ACFFC1D" in text and "Frame length 512" in text and "frame counter" in text
    assert "No frame sync" in analyse_framing(np.zeros(100, np.uint8)).summary()
