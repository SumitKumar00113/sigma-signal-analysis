"""Tests for the command-line one-click analysis."""

import json

import numpy as np

from src.auto_analyse import main
from src.decoding.correlation import parse_pattern
from src.decoding.viterbi import ConvCode, conv_encode
from src.dsp import synth


def test_cli_decodes_framed_raw_iq(tmp_path, capsys):
    rng = np.random.default_rng(0)
    np.random.seed(0)
    asm = parse_pattern("0x1ACFFC1D")
    data = np.concatenate([np.concatenate([asm, rng.integers(0, 2, 992, dtype=np.uint8)])
                           for _ in range(10)])
    coded = conv_encode(data, ConvCode(7, (0o171, 0o133)), terminate=False)
    x = synth.modulate_bits(coded, "bpsk", 4800, 48000, snr_db=10, freq_offset_hz=500)
    iq = tmp_path / "capture.iq"
    x.astype(np.complex64).tofile(iq)
    report, bits = tmp_path / "r.json", tmp_path / "bits.bin"
    assert main([str(iq), "--fs", "48000", "--no-interleaver", "--json", str(report),
                 "--save-bits", str(bits)]) == 0
    out = capsys.readouterr().out
    assert "Modulation   BPSK" in out and "✓ Framing" in out and "CCSDS" in out
    r = json.loads(report.read_text())
    assert r["modulation"] == "BPSK" and abs(r["symbol_rate_hz"] - 4800) < 5
    dec = r["decoding"]
    assert dec["final_stage"] == "decoded"
    assert dec["framing"]["found"] and dec["framing"]["sync"] == "0x1ACFFC1D"
    assert dec["framing"]["frame_bits"] == 1024
    assert bits.stat().st_size == dec["final_bits"] // 8
