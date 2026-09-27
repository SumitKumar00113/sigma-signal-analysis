"""GUI tests: dashboard layout, view switching, projects, settings, recent
recordings, and analysing a selected span."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QFileDialog, QMessageBox  # noqa: E402

from src.core.enums import FileFormat, SampleDatatype  # noqa: E402
from src.core.models import RecordingMetadata  # noqa: E402
from src.dsp import synth  # noqa: E402
from src.gui.main_window import MainWindow  # noqa: E402
from src.gui.settings import AppSettings, Preferences, Project, SettingsDialog  # noqa: E402
from src.gui.widgets import MeterBar  # noqa: E402

FS = 48_000.0


@pytest.fixture
def settings(tmp_path):
    return AppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat))


@pytest.fixture
def recording(tmp_path):
    np.random.seed(11)
    x, _ = synth.generate_bpsk(3000, 2400, FS, 15)
    path = tmp_path / "bpsk.cf32"
    x.astype(np.complex64).tofile(path)
    meta = RecordingMetadata(source_path=str(path), source_format=FileFormat.RAW_IQ,
                             sample_datatype=SampleDatatype.CF32_LE, sample_rate_hz=FS)
    return str(path), meta, x


def _open(qtbot, win, path, meta):
    win._load_recording(path, meta)
    qtbot.waitUntil(lambda: win._current_samples is not None, timeout=30_000)


def _analyse(qtbot, win):
    win._on_run_analysis()
    qtbot.waitUntil(lambda: not win._analysis_running, timeout=120_000)


def test_welcome_then_workspace(qtbot, settings, recording):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    assert win._body.currentWidget() is win._welcome
    assert not win._run_btn.isEnabled() and not win._save_btn.isEnabled()

    path, meta, x = recording
    _open(qtbot, win, path, meta)
    assert win._body.currentIndex() == 1
    assert win._run_btn.isEnabled() and win._save_btn.isEnabled()
    assert win._view_nav.labels() == ["Waveform", "Spectrum", "Waterfall",
                                      "Constellation", "Decoding"]
    assert win._hero_title.text() == "bpsk.cf32"
    rec = win._nav_tree.topLevelItem(0)
    assert rec.childCount() == 1 and rec.child(0).text(1) == "Active"
    # the recording went into the recent list
    assert [p for p, _ in settings.recent()] == [path]
    assert "bpsk.cf32" in win._welcome._recent_list.item(0).text()


def test_segmented_nav_switches_views(qtbot, settings):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    win._show_viewers()
    win._view_nav.index_selected.emit(2)
    assert win._central_stack.currentWidget() is win._waterfall_viewer
    assert win._view_card.title_label.text() == "Waterfall"
    win._central_stack.setCurrentWidget(win._decoding_panel)
    assert win._view_nav._group.checkedId() == 4
    win._view_actions[1].trigger()                       # Ctrl+2
    assert win._central_stack.currentWidget() is win._spectrum_viewer


def test_analysis_fills_dashboard(qtbot, settings, recording):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    path, meta, _ = recording
    _open(qtbot, win, path, meta)
    _analyse(qtbot, win)
    assert win._last_result is not None
    mod = win._last_result.analysis.modulation.value
    assert win._dash_mod.text() == mod
    assert win._dash_snr.value_label.text().endswith("dB")
    assert win._dash_rows["Output"].value().endswith("bits")
    assert win._inspector_tabs.currentWidget() is win._results
    assert "Analysis complete" in win._console.toPlainText()


def test_analyse_selection(qtbot, settings, recording, monkeypatch):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    path, meta, x = recording
    _open(qtbot, win, path, meta)
    n = len(x)
    win._time_viewer._region.setRegion([0.5 * n / FS, 0.9 * n / FS])
    win._on_analyse_selection()
    qtbot.waitUntil(lambda: not win._analysis_running, timeout=120_000)
    assert win._last_result is not None
    assert win._analysis_offset_s == pytest.approx(0.5 * n / FS, rel=1e-3)
    assert "on " in win._console.toPlainText() and " s…" in win._console.toPlainText()

    # an empty selection explains what to do instead of analysing
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(a))
    win._time_viewer._region.setRegion([0.1, 0.1])
    win._on_analyse_selection()
    assert shown and not win._analysis_running


def test_project_round_trip(qtbot, settings, recording, tmp_path, monkeypatch):
    path, meta, _ = recording
    win = MainWindow(settings)
    qtbot.addWidget(win)
    _open(qtbot, win, path, meta)
    win._results._rs_override.setValue(2400)
    project_file = tmp_path / "session"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(project_file), ""))
    win._on_save()
    saved = tmp_path / "session.sigma"
    assert saved.exists() and win._project_path == saved
    assert "session.sigma" in win.windowTitle()

    project = Project.load(saved)
    assert project.recordings[0][0] == path and project.symbol_rate_override == 2400

    # New project resets everything
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    win._on_new_project()
    assert win._current_samples is None and not win._recordings
    assert win._body.currentWidget() is win._welcome
    assert win._results.symbol_rate_override() is None

    # Opening the project restores the recording and the override
    assert win.open_project(saved)
    qtbot.waitUntil(lambda: win._current_samples is not None, timeout=30_000)
    assert win._current_path == path
    assert win._results.symbol_rate_override() == 2400


def test_project_with_missing_recording(qtbot, settings, tmp_path):
    missing = str(tmp_path / "gone.cf32")
    Project(recordings=[(missing, RecordingMetadata(sample_rate_hz=FS))]).save(
        tmp_path / "p.sigma")
    win = MainWindow(settings)
    qtbot.addWidget(win)
    assert win.open_project(tmp_path / "p.sigma")
    assert win._current_samples is None
    assert win._nav_tree.topLevelItem(0).child(0).text(1) == "Missing"
    assert "not found" in win._console.toPlainText()


def test_settings_apply_and_persist(qtbot, settings):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    prefs = Preferences(classifier_mode="rules", max_samples_millions=3, worker_threads=2,
                        spectrum_fft=1024, waterfall_fft=512, waterfall_colormap="magma")
    win._apply_preferences(prefs)
    assert win._classifier_mode == "rules"
    assert win._classifier_actions["rules"].isChecked()
    assert win._spectrum_viewer._fft_spin.value() == 1024
    assert win._waterfall_viewer._fft_spin.value() == 512
    assert win._waterfall_viewer._cmap_combo.currentText() == "magma"
    assert settings.preferences() == prefs

    dlg = SettingsDialog(prefs)
    qtbot.addWidget(dlg)
    assert dlg.preferences() == prefs


def test_recent_skips_missing_files(settings, tmp_path):
    real = tmp_path / "a.cf32"
    real.write_bytes(b"\0" * 64)
    settings.add_recent(str(tmp_path / "missing.cf32"), RecordingMetadata())
    settings.add_recent(str(real), RecordingMetadata(sample_rate_hz=FS))
    assert [p for p, _ in settings.recent()] == [str(real)]
    settings.clear_recent()
    assert settings.recent() == []


def test_meter_bar_clamps(qtbot):
    bar = MeterBar()
    qtbot.addWidget(bar)
    bar.set_fraction(1.56)
    assert bar.fraction() == 1.0
    bar.set_fraction(-3)
    assert bar.fraction() == 0.0
