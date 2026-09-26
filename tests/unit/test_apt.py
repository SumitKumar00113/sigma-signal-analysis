"""Tests for NOAA APT image decoding."""

import os
import zlib

import numpy as np
import pytest

from src.core.enums import ModulationType as M
from src.core.models import RecordingMetadata
from src.decoding.apt import IMAGE_A, IMAGE_B, decode_apt, write_png
from src.dsp import synth
from src.dsp.pipeline import AnalysisPipeline

FS = 50_000.0


def _scene(lines: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:lines, 0:909]
    a = 0.5 + 0.4 * np.sin(x / 60.0) * np.cos(y / 15.0)          # "clouds"
    b = np.clip(x / 909.0 * 0.8 + 0.1 * (y % 20 < 10), 0, 1)       # gradient + bands
    return a, b


def _corr(u, v):
    return float(np.corrcoef(u.ravel().astype(float), v.ravel().astype(float))[0, 1])


@pytest.mark.parametrize("clock_error", [0.0, 2e-4])
def test_image_is_recovered(clock_error):
    np.random.seed(0)
    a, b = _scene(40)
    x = synth.generate_apt(a, b, FS, snr_db=15, clock_error=clock_error)
    r = decode_apt(x, FS)
    assert r.found and r.sync_quality > 0.95
    assert 38 <= r.lines <= 40
    assert abs(r.line_rate_hz - 2.0 / (1 + clock_error)) < 2e-3
    n = min(r.lines, 38)
    assert _corr(r.channel_a[1:n], a[1:n]) > 0.9
    assert _corr(r.channel_b[1:n], b[1:n]) > 0.9
    assert 15_000 < r.fm_deviation_hz < 20_000


def test_plain_fm_is_not_apt():
    np.random.seed(1)
    x, _ = synth.generate_fm(int(10 * FS), FS, 5000, None, 15)
    assert not decode_apt(x, FS).found


def test_pipeline_decodes_apt():
    np.random.seed(2)
    a, b = _scene(30)
    x = synth.generate_apt(a, b, FS, snr_db=12)
    res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=FS))
    assert res.analysis.modulation == M.FM
    assert res.apt is not None and res.apt.lines >= 28
    assert any("NOAA APT" in w for w in res.analysis.warnings)


def test_png_writer(tmp_path):
    img = (np.arange(60 * 80) % 256).astype(np.uint8).reshape(60, 80)
    p = write_png(tmp_path / "x.png", img)
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    i = data.find(b"IDAT")
    n = int.from_bytes(data[i - 4:i], "big")
    raw = np.frombuffer(zlib.decompress(data[i + 4:i + 4 + n]), np.uint8).reshape(60, 81)
    assert np.array_equal(raw[:, 1:], img)


def test_gui_image_tab(qtbot):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from src.gui.main_window import MainWindow

    np.random.seed(3)
    a, b = _scene(20)
    res = AnalysisPipeline().run(synth.generate_apt(a, b, FS, snr_db=15),
                                 RecordingMetadata(sample_rate_hz=FS))
    win = MainWindow()
    qtbot.addWidget(win)
    win._show_viewers()
    win._on_analysis_done(res)
    assert win._central_stack.indexOf(win._image_viewer) >= 0
    assert win._save_image_action.isEnabled()
    assert win._image_viewer.image.shape[1] == 2080
    assert IMAGE_A.stop <= IMAGE_B.start
