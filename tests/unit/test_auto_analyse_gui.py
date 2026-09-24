"""GUI tests for one-click auto-decode (panel) and Auto-Analyse (main window)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from src.core.enums import FECType, InterleaverType  # noqa: E402
from src.core.enums import ModulationType as M  # noqa: E402
from src.core.models import RecordingMetadata  # noqa: E402
from src.decoding.correlation import parse_pattern  # noqa: E402
from src.decoding.interleaving import InterleaverSpec, interleave  # noqa: E402
from src.decoding.viterbi import ConvCode, conv_encode  # noqa: E402
from src.dsp import synth  # noqa: E402
from src.gui.decoding_panel import DecodingPanel  # noqa: E402
from src.gui.main_window import MainWindow  # noqa: E402

K7 = ConvCode(7, (0o171, 0o133))


def _frames(rng, n):
    asm = parse_pattern("0x1ACFFC1D")
    return np.concatenate([np.concatenate([asm, rng.integers(0, 2, 992, dtype=np.uint8)])
                           for _ in range(n)])


def test_panel_auto_decode_fills_every_step(qtbot):
    rng = np.random.default_rng(0)
    enc = conv_encode(_frames(rng, 24), K7, terminate=False)
    spec = InterleaverSpec(kind=InterleaverType.BLOCK, rows=16, cols=64)
    rx = interleave(enc[: len(enc) // 1024 * 1024], spec)
    rx = (rx ^ (rng.random(len(rx)) < 0.01))[200:].astype(np.uint8)

    panel = DecodingPanel()
    qtbot.addWidget(panel)
    panel.set_bits(rx, "test", 1, M.BPSK)
    with qtbot.waitSignal(panel.auto_decode_finished, timeout=300_000):
        panel.run_auto_decode()
    assert not panel._busy and not panel.auto_decode_running
    assert panel._il_type.currentData() == InterleaverType.BLOCK
    assert panel._il_rows.value() * panel._il_cols.value() == 1024
    assert panel._fec_type.currentData() == FECType.CONVOLUTIONAL
    assert panel._sync_edit.text() == "0x1ACFFC1D"
    assert panel._frame_bits.value() == 1024
    assert panel._frames_table.rowCount() >= 20
    assert panel._view_combo.currentText() == "decoded"
    assert "⚡ Auto-decode" in panel._auto_result.text()
    # the filled-in controls reproduce the result by hand
    decoded = panel._stages["decoded"].copy()
    panel._apply_deinterleave()
    panel._apply_fec()
    assert np.array_equal(panel._stages["decoded"], decoded)


def test_auto_decode_without_bits(qtbot):
    panel = DecodingPanel()
    qtbot.addWidget(panel)
    panel.run_auto_decode()
    assert "No bits" in panel._auto_result.text() and not panel.auto_decode_running


def _window_with(qtbot, samples, fs):
    win = MainWindow()
    qtbot.addWidget(win)
    win._current_samples = samples
    win._current_metadata = RecordingMetadata(sample_rate_hz=fs)
    win._show_viewers()
    return win


def test_one_click_auto_analyse(qtbot):
    rng = np.random.default_rng(1)
    np.random.seed(1)
    coded = conv_encode(_frames(rng, 10), K7, terminate=False)
    x = synth.modulate_bits(coded, "qpsk", 4800, 48000, snr_db=10, freq_offset_hz=-900)
    win = _window_with(qtbot, x, 48000.0)
    panel = win._decoding_panel
    with qtbot.waitSignal(panel.auto_decode_finished, timeout=300_000):
        win._on_auto_analyse()
    res = panel.last_auto_result
    assert res.decode_ok and res.framing.found and "CCSDS" in res.framing.sync_name
    assert win._central_stack.currentWidget() is panel
    log = win._console.toPlainText()
    assert "Auto-Analyse" in log and "Auto-decode result" in log and "✓ Framing" in log
    assert not win._auto_after_analysis


def test_auto_analyse_analog_signal(qtbot):
    np.random.seed(2)
    x, _ = synth.generate_fm(48000, 48000.0, 3000.0, None, 15, 1500.0)
    win = _window_with(qtbot, x, 48000.0)
    win._on_auto_analyse()
    qtbot.waitUntil(lambda: not win._analysis_running, timeout=120_000)
    assert "is analog" in win._console.toPlainText()
    assert win._decoding_panel.last_auto_result is None


def test_auto_analyse_without_recording(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(a))
    win = MainWindow()
    qtbot.addWidget(win)
    win._on_auto_analyse()
    assert shown and not win._auto_after_analysis
