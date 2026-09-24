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
