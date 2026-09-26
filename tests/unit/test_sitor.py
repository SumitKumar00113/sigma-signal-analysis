"""Tests for SITOR-B / NAVTEX decoding."""

import numpy as np
import pytest

from src.core.enums import ModulationType as M
from src.core.models import RecordingMetadata
from src.decoding.auto_decode import auto_decode
from src.decoding.baudot import detect_baudot
from src.decoding.sitor import (
    _FIG_OF,
    _LETTERS,
    ALPHA,
    CR,
    FIGS,
    LF,
    LTRS,
    SPACE,
    VALID,
    detect_sitor_b,
)
from src.dsp import synth
from src.dsp.pipeline import AnalysisPipeline

MSG = ("ZCZC EA33\nWZ 133\nENGLAND, SOUTH COAST. EDDYSTONE ROCKS SOUTH-WESTWARDS.\n"
       "1. E1 BUOY, 50-02.6N 004-22.5W, DAMAGED, UNLIT AND AIS INOPERATIVE.\nNNNN\n")
RQ = 0x33                          # phasing 2 (RX slots while idle)
_CODE = {v: k for k, v in _LETTERS.items()}
_FIG_CODE = {f: _CODE[ch] for ch, f in _FIG_OF.items() if f not in "'$"}


def encode(text: str) -> list[int]:
    codes, figs = [LTRS], False
    for ch in text:
        if ch == " ":
            codes.append(SPACE)
        elif ch == "\n":
            codes += [CR, LF]
        else:
            want = ch not in _CODE
            if want != figs:
                codes.append(FIGS if want else LTRS)
                figs = want
            codes.append(_FIG_CODE[ch] if figs else _CODE[ch])
    return codes


def fec_b(chars: list[int], idle: int = 20) -> np.ndarray:
    """FEC mode B: DX slot 2k = char k, RX slot 2k+1 = char k−2 (5 slots later)."""
    chars = [ALPHA] * idle + chars + [ALPHA] * idle
    slots = []
    for k, c in enumerate(chars):
        slots.append(c)
        slots.append(chars[k - 2] if k >= 2 else RQ)
    return np.array([(c >> b) & 1 for c in slots for b in range(7)], dtype=np.uint8)


class TestDecoder:
    def test_table_is_constant_ratio(self):
        assert all(c in VALID for c in list(_LETTERS) + [SPACE, LTRS, FIGS, CR, LF, ALPHA, RQ])
        assert len(set(_LETTERS)) == 26

    @pytest.mark.parametrize("inverted", [False, True])
    @pytest.mark.parametrize("lead", [0, 3])
    def test_round_trip(self, inverted, lead):
        rng = np.random.default_rng(0)
        bits = np.concatenate([rng.integers(0, 2, lead, np.uint8), fec_b(encode(MSG))])
        r = detect_sitor_b(bits ^ inverted)
        assert r.found and r.inverted == inverted
        assert "EDDYSTONE ROCKS SOUTH-WESTWARDS." in r.text
        assert "50-02.6N 004-22.5W" in r.text
        assert [m.header for m in r.messages] == ["EA33"]
        m = r.messages[0]
        assert m.station == "E" and m.subject == "navigational warning" and m.number == "33"

    def test_time_diversity_repairs_errors(self):
        bits = fec_b(encode(MSG))
        rng = np.random.default_rng(1)
        # corrupt one bit in 15 % of the DX slots: the RX copy restores them
        n_slots = len(bits) // 7
        for s in rng.choice(np.arange(0, n_slots, 2), int(0.15 * n_slots / 2), replace=False):
            bits[7 * s + rng.integers(0, 7)] ^= 1
        r = detect_sitor_b(bits)
        assert r.recovered > 10 and r.lost == 0
        assert "1. E1 BUOY, 50-02.6N 004-22.5W, DAMAGED" in r.text

    def test_bit_slip_is_followed(self):
        bits = fec_b(encode(MSG * 3))
        slipped = np.delete(bits, len(bits) // 2)              # one bit lost
        r = detect_sitor_b(slipped)
        assert len(r.messages) >= 2 and r.messages[-1].header == "EA33"

    @pytest.mark.parametrize("seed", range(3))
    def test_random_bits_are_not_sitor(self, seed):
        x = np.random.default_rng(seed).integers(0, 2, 30000, dtype=np.uint8)
        assert not detect_sitor_b(x).found

    def test_sitor_is_not_baudot_and_vice_versa(self):
        assert not detect_baudot(fec_b(encode(MSG * 2))).found


def test_navtex_from_samples_to_messages():
    """100 Bd, 170 Hz shift FSK through the whole chain."""
    np.random.seed(2)
    fs = 2000.0
    x = synth.modulate_bits(fec_b(encode(MSG * 2)), "2fsk", 100.0, fs, snr_db=18,
                            deviation_hz=85, freq_offset_hz=30)
    res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=fs))
    assert res.analysis.modulation == M.FSK2
    assert abs(res.analysis.symbol_rate_hz - 100) < 1
    chain = auto_decode(res.demod.bits, 1, M.FSK2, ldpc_codes=[], search_interleaver=False)
    assert chain.step("Text").status == "found" and "ZCZC EA33" in chain.summary()
    assert chain.step("FEC").status == "skipped"
    assert [m.header for m in chain.text.messages][-1] == "EA33"
    assert "EDDYSTONE ROCKS" in chain.text.text
