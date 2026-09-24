"""GUI test: an analog result shows audio and enables audio export."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from src.core.models import RecordingMetadata  # noqa: E402
from src.dsp.pipeline import AnalysisPipeline  # noqa: E402
from src.gui.main_window import MainWindow  # noqa: E402
from tests.signals import generators as sg  # noqa: E402


def test_analog_result_in_gui(qtbot):
    np.random.seed(5)
    x, _ = sg.generate_fm(48000, 48000.0, 3000.0, None, 15, 1500.0)
    result = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=48000.0))
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win._save_audio_action.isEnabled()
    win._on_analysis_done(result)
    assert win._save_audio_action.isEnabled()
    assert "s @ 8,000 Hz" in win._results._rows["Audio"]._val.text()


def test_digital_result_disables_audio(qtbot):
    np.random.seed(5)
    x, _ = sg.generate_msk(2000, 2400.0, 48000.0, None, 15, 1500.0)
    result = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=48000.0))
    win = MainWindow()
    qtbot.addWidget(win)
    win._on_analysis_done(result)
    assert not win._save_audio_action.isEnabled()
    assert len(win._decoding_panel._stages["raw"]) > 1000


def test_bursts_overlay_and_selection(qtbot):
    from src.dsp import synth as s

    np.random.seed(7)
    fs = 48000.0
    x = ((np.random.randn(int(fs)) + 1j * np.random.randn(int(fs))) * 0.1).astype(np.complex64)
    b1, _ = s.generate_bpsk(500, 2400, fs, None, 5000)
    b2, _ = s.generate_mfsk(200, 600, fs, 2, 1.0, None, -8000)
    x[2000:2000 + len(b1)] += b1
    x[25000:25000 + len(b2)] += b2
    meta = RecordingMetadata(sample_rate_hz=fs)
    result = AnalysisPipeline().run(x, meta)
    assert len(result.bursts) == 2

    win = MainWindow()
    qtbot.addWidget(win)
    win._waterfall_viewer.set_data(x, fs)
    win._on_analysis_done(result)
    assert win._waterfall_viewer.region_count == 2
    regions_node = win._nav_tree.topLevelItem(1)
    assert regions_node.childCount() == 2
    target = next(i for i in range(2) if result.bursts[i] is not
                  result.bursts[result.primary_burst])
    win._on_nav_item_clicked(regions_node.child(target))
    assert result.bursts[result.primary_burst].index == target
    assert result.analysis.modulation == result.bursts[target].modulation
