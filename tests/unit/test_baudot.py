"""Tests for asynchronous Baudot (RTTY) decoding."""

import os

import numpy as np
import pytest

from src.core.enums import ModulationType as M
from src.core.models import RecordingMetadata
from src.decoding.auto_decode import auto_decode
from src.decoding.baudot import _FIGURES, _LETTERS, FIGS, LTRS, codes_to_text, detect_baudot
from src.dsp import synth
from src.dsp.pipeline import AnalysisPipeline

TEXT = ("CQ CQ DE DDH47\nSEEWETTERBERICHT FUER DIE OSTSEE 28.02.23, 00 UTC:\n"
        "SKAGERRAK (58.3N 9.8E) WT: 6 C\nRYRYRYRYRY THE QUICK BROWN FOX 0123456789\n")


def encode(text: str) -> list[int]:
    """Text → ITA2 codes, inserting LTRS/FIGS shifts where needed."""
    codes, figs = [LTRS], False
    for ch in text:
        if ch in " \n":                          # same code in both tables
            codes.append(_LETTERS.index(ch))
            continue
        want_figs = ch not in _LETTERS
        if want_figs != figs:
            codes.append(FIGS if want_figs else LTRS)
            figs = want_figs
        codes.append((_FIGURES if figs else _LETTERS).index(ch))
    return codes


def elements(codes: list[int], stop: float = 1.5, idle: int = 20) -> np.ndarray:
    """Half-element samples (2 per element): idle mark, start, data LSB first, stop."""
    out = [1] * idle
    for c in codes:
        out += [0, 0]
        for b in range(5):
            out += [(c >> b) & 1] * 2
        out += [1] * int(round(stop * 2))
    return np.array(out + [1] * idle, dtype=np.uint8)


class TestDecoder:
    def test_codes_round_trip(self):
        assert codes_to_text(encode(TEXT)) == TEXT

    @pytest.mark.parametrize("stop", [1.0, 1.5, 2.0])
    @pytest.mark.parametrize("inverted", [False, True])
    def test_detects_polarity_and_stop(self, stop, inverted):
        x = elements(encode(TEXT * 3), stop)
        r = detect_baudot(x ^ 1 if inverted else x)
        assert r.found and r.inverted == inverted and r.stop_elements == stop
        assert r.text == TEXT * 3

    def test_element_rate_grid(self):
        """Bits at one sample per element (1- or 2-element stops)."""
        x = elements(encode(TEXT * 3), 2.0)[::2]
        r = detect_baudot(x)
        assert r.found and r.samples_per_element == 1 and r.text == TEXT * 3

    def test_bit_errors_are_local(self):
        rng = np.random.default_rng(0)
        x = elements(encode(TEXT * 5))
        x ^= (rng.random(len(x)) < 0.003).astype(np.uint8)
        r = detect_baudot(x)
        assert r.found and "SEEWETTERBERICHT" in r.text

    @pytest.mark.parametrize("seed", range(4))
    def test_random_bits_are_not_text(self, seed):
        x = np.random.default_rng(seed).integers(0, 2, 30000, dtype=np.uint8)
        assert not detect_baudot(x).found


def test_rtty_from_samples_to_text():
    """50 Bd, 450 Hz shift, 1.5 stop – like DWD – through the whole chain."""
    np.random.seed(1)
    fs = 2000.0
    half = elements(encode(TEXT * 4))
    x = synth.modulate_bits(half, "2fsk", 100.0, fs, snr_db=20, deviation_hz=225,
                            freq_offset_hz=40)
    res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=fs))
    assert res.analysis.modulation == M.FSK2
    assert abs(res.analysis.symbol_rate_hz - 100) < 1
    assert any("50.00 Bd" in w for w in res.analysis.warnings)
    chain = auto_decode(res.demod.bits, 1, M.FSK2, ldpc_codes=[], search_interleaver=False)
    assert chain.text is not None and chain.step("Text").status == "found"
    assert "SEEWETTERBERICHT FUER DIE OSTSEE" in chain.text.text
    # the demodulator locks after the first characters; nothing else is claimed
    assert chain.step("FEC").status == "skipped" and "decoded" not in chain.stages
    assert "SKAGERRAK (58.3N 9.8E) WT: 6 C" in chain.text.text


def test_panel_shows_text(qtbot):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from src.gui.decoding_panel import DecodingPanel

    panel = DecodingPanel()
    qtbot.addWidget(panel)
    panel.set_bits(elements(encode(TEXT * 3)), "rtty", 1, M.FSK2)
    with qtbot.waitSignal(panel.auto_decode_finished, timeout=120_000):
        panel.run_auto_decode()
    assert panel._view_combo.currentText() == "text"
    assert "SKAGERRAK" in panel._bits_view.toPlainText()
    panel._view_combo.setCurrentText("raw")
    assert "SKAGERRAK" not in panel._bits_view.toPlainText()
