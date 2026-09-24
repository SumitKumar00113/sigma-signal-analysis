"""Tests for the one-click decoding chain."""

import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.enums import InterleaverType
from src.core.enums import ModulationType as M
from src.core.models import RecordingMetadata
from src.decoding.auto_decode import auto_decode, mapping_transforms
from src.decoding.correlation import parse_pattern
from src.decoding.interleaving import InterleaverSpec, interleave
from src.decoding.reed_solomon import ReedSolomon, bytes_to_bits
from src.decoding.viterbi import ConvCode, conv_encode

K7 = ConvCode(7, (0o171, 0o133))
ASM = parse_pattern("0x1ACFFC1D")


def _frames(rng, n=30, period=1024):
    out = []
    for i in range(n):
        c = np.array([(i >> (15 - j)) & 1 for j in range(16)], dtype=np.uint8)
        out.append(np.concatenate([ASM, c, rng.integers(0, 2, period - 48, dtype=np.uint8)]))
    return np.concatenate(out)


def _noisy(x, ber, rng):
    return x ^ (rng.random(len(x)) < ber).astype(np.uint8)


def _contains(decoded, data, max_ber=1e-3):
    """*decoded* equals a stretch of *data* (the chain may drop a few bits)."""
    probe = decoded[200:1200]
    corr = np.correlate(1.0 - 2.0 * data.astype(float), 1.0 - 2.0 * probe.astype(float), "valid")
    off = int(np.argmax(corr)) - 200
    n = min(len(decoded), len(data) - off) - 200
    return np.mean(decoded[200:n] != data[off + 200: off + n]) <= max_ber


def test_interleaved_convolutional_framed_stream():
    rng = np.random.default_rng(0)
    data = _frames(rng)
    enc = conv_encode(data, K7, terminate=False)
    spec = InterleaverSpec(kind=InterleaverType.BLOCK, rows=16, cols=64)
    rx = _noisy(interleave(enc[: len(enc) // 1024 * 1024], spec), 0.01, rng)[300:]
    progress = []
    res = auto_decode(rx, 1, M.BPSK, ldpc_codes=[],
                      progress_cb=lambda f, m: progress.append(f))
    assert res.step("Interleaver").status == "found"
    assert res.interleaver.best.spec.rows * res.interleaver.best.spec.cols == 1024
    assert res.step("FEC").status == "found" and res.decode_ok
    assert set(res.stages) == {"raw", "deinterleaved", "decoded"}
    assert _contains(res.final_bits, data)
    f = res.framing
    assert res.framing_stage == "decoded" and f.found and "CCSDS" in f.sync_name
    assert f.period == 1024 and len(f.frames) >= 25
    assert any(h.kind == "counter" and h.start == 32 and h.length == 16 for h in f.fields)
    assert progress[-1] == 1.0 and progress == sorted(progress)
    text = res.summary()
    assert "✓ Interleaver" in text and "✓ Framing" in text


@pytest.mark.parametrize("k,mod,name", [(2, M.QPSK, "swap bits, invert bit 2"),
                                        (4, M.QAM16, "swap I/Q, negate I")])
def test_phase_ambiguity_mapping(k, mod, name):
    rng = np.random.default_rng(1)
    data = _frames(rng, 20)
    enc = _noisy(conv_encode(data, K7, terminate=False), 0.005, rng)
    rotated = dict(mapping_transforms(k))[name](enc)
    res = auto_decode(rotated, k, mod, ldpc_codes=[], search_interleaver=False)
    assert res.mapping is not None and res.mapping.name != "identity"
    assert res.step("Bit mapping").status == "found"
    assert res.decode_ok and res.framing.found and len(res.framing.frames) >= 18


def test_concatenated_rs_and_convolutional():
    rng = np.random.default_rng(2)
    rs = ReedSolomon(204, 188)
    outer = bytes_to_bits(np.concatenate(
        [rs.encode(rng.integers(0, 256, 188).astype(np.uint8)) for _ in range(12)]))
    outer = np.concatenate([rng.integers(0, 2, 333).astype(np.uint8), outer])
    rx = _noisy(conv_encode(outer, K7, terminate=False)[5:], 0.02, rng)
    res = auto_decode(rx, 1, M.BPSK, ldpc_codes=[], search_interleaver=False)
    assert res.fec.rs is not None and (res.fec.rs.n, res.fec.rs.k) == (204, 188)
    assert res.decode_ok and "RS(204,188)" in res.step("Decode").detail
    assert len(res.stages["decoded"]) >= 10 * 188 * 8


def test_ldpc_stream():
    sys.path.insert(0, str(Path(__file__).parent))
    from test_ldpc import ira_code

    from src.decoding.ldpc import ldpc_encode

    rng = np.random.default_rng(3)
    code = ira_code()
    infos = [rng.integers(0, 2, code.k).astype(np.uint8) for _ in range(6)]
    rx = _noisy(np.concatenate([ldpc_encode(i, code) for i in infos])[1234:], 0.01, rng)
    res = auto_decode(rx, 1, M.BPSK, ldpc_codes=[code], search_interleaver=False)
    assert res.fec.ldpc is not None and res.decode_ok
    assert np.array_equal(res.stages["decoded"], np.concatenate(infos[1:]))


def test_uncoded_framing_found_on_raw_stream():
    rng = np.random.default_rng(4)
    data = _noisy(_frames(rng, 20), 0.002, rng)
    res = auto_decode(data, 1, M.BPSK, ldpc_codes=[], search_interleaver=False)
    assert res.step("FEC").status == "none" and "decoded" not in res.stages
    assert res.framing.found and res.framing_stage == "raw"


def test_short_input_and_cancel():
    res = auto_decode(np.zeros(100, np.uint8))
    assert res.steps[0].status == "failed" and "256" in res.summary()
    rng = np.random.default_rng(5)
    res = auto_decode(rng.integers(0, 2, 20000, dtype=np.uint8), ldpc_codes=[],
                      cancel_check=lambda: True)
    assert res.cancelled and "Cancelled" in res.summary()


def test_end_to_end_from_samples():
    """Framed, K7-coded QPSK at a carrier offset: samples → analysis → chain."""
    from src.dsp import synth
    from src.dsp.pipeline import AnalysisPipeline

    rng = np.random.default_rng(6)
    np.random.seed(6)
    data = _frames(rng, 12)
    sig = synth.modulate_bits(conv_encode(data, K7, terminate=False), "qpsk", 4800, 48000,
                              snr_db=9, freq_offset_hz=700)
    r = AnalysisPipeline().run(sig, RecordingMetadata(sample_rate_hz=48000))
    assert r.analysis.modulation == M.QPSK
    res = auto_decode(r.demod.bits, r.demod.bits_per_symbol, r.demod.modulation,
                      ldpc_codes=[], search_interleaver=False)
    assert res.decode_ok and res.framing.found and len(res.framing.frames) >= 10
    assert _contains(res.stages["decoded"], data)
