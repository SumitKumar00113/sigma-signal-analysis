"""Main application window.

Assembles the menu bar, toolbar, project navigator, visualisation workspace,
parameter inspector, and console panel following the layout in PRD §8.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.enums import FileFormat, WavInterpretation
from src.core.models import RecordingMetadata
from src.gui.input_wizard import InputWizard
from src.gui.recording_overview import RecordingOverview
from src.gui.viewers.constellation_viewer import ConstellationViewer
from src.gui.viewers.spectrum_viewer import SpectrumViewer
from src.gui.viewers.time_viewer import TimeViewer
from src.gui.viewers.waterfall_viewer import WaterfallViewer
from src.gui.welcome_screen import WelcomeScreen
from src.gui.workers import Worker, WorkerPool


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sigma Signal Analysis — v0.1.0")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 960)

        self._current_metadata: RecordingMetadata | None = None
        self._current_samples: np.ndarray | None = None

        self._build_menu_bar()
        self._build_toolbar()
        self._build_central()
        self._build_docks()
        self._build_status_bar()

        # Start with welcome screen
        self._show_welcome()

    # ==================================================================
    # Menu bar
    # ==================================================================

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")

        open_action = QAction("&Open Recording…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Results…", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = mb.addMenu("&View")
        self._toggle_nav_action = QAction("Project &Navigator", self)
        self._toggle_nav_action.setCheckable(True)
        self._toggle_nav_action.setChecked(True)
        view_menu.addAction(self._toggle_nav_action)

        self._toggle_console_action = QAction("&Console", self)
        self._toggle_console_action.setCheckable(True)
        self._toggle_console_action.setChecked(True)
        view_menu.addAction(self._toggle_console_action)

        # Analysis menu
        analysis_menu = mb.addMenu("&Analysis")
        run_action = QAction("&Run Analysis", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self._on_run_analysis)
        analysis_menu.addAction(run_action)

        # Help menu
        help_menu = mb.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ==================================================================
    # Toolbar
    # ==================================================================

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        self.addToolBar(tb)

        tb.addAction("📂 Open", self._on_open)
        tb.addAction("💾 Save", self._on_save)
        tb.addSeparator()
        tb.addAction("▶ Run Analysis", self._on_run_analysis)
        tb.addAction("📤 Export", self._on_export)
        tb.addSeparator()
        tb.addAction("⚙ Settings", self._on_settings)

    # ==================================================================
    # Central widget (tabbed viewers)
    # ==================================================================

    def _build_central(self) -> None:
        self._central_stack = QTabWidget()
        self._central_stack.setDocumentMode(True)
        self._central_stack.setTabsClosable(False)

        # Welcome tab
        self._welcome = WelcomeScreen()
        self._welcome.open_file_requested.connect(self._on_open)
        self._welcome.new_project_requested.connect(self._on_new_project)

        # Viewers
        self._time_viewer = TimeViewer()
        self._spectrum_viewer = SpectrumViewer()
        self._waterfall_viewer = WaterfallViewer()
        self._constellation_viewer = ConstellationViewer()

        self.setCentralWidget(self._central_stack)

    def _show_welcome(self) -> None:
        self._central_stack.clear()
        self._central_stack.addTab(self._welcome, "🏠 Welcome")

    def _show_viewers(self) -> None:
        self._central_stack.clear()
        self._central_stack.addTab(self._time_viewer, "📈 Waveform")
        self._central_stack.addTab(self._spectrum_viewer, "📊 Spectrum")
        self._central_stack.addTab(self._waterfall_viewer, "🌊 Waterfall")
        self._central_stack.addTab(self._constellation_viewer, "⭐ Constellation")

    # ==================================================================
    # Dock widgets
    # ==================================================================

    def _build_docks(self) -> None:
        # --- Project Navigator (left) ---
        self._nav_dock = QDockWidget("Project Navigator", self)
        self._nav_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._nav_tree = QTreeWidget()
        self._nav_tree.setHeaderLabels(["Name", "Status"])
        self._nav_tree.setAlternatingRowColors(True)

        recordings_item = QTreeWidgetItem(["📁 Recordings", ""])
        self._nav_tree.addTopLevelItem(recordings_item)
        regions_item = QTreeWidgetItem(["🎯 Regions of Interest", ""])
        self._nav_tree.addTopLevelItem(regions_item)
        jobs_item = QTreeWidgetItem(["⚙ Processing Jobs", ""])
        self._nav_tree.addTopLevelItem(jobs_item)
        results_item = QTreeWidgetItem(["📋 Results", ""])
        self._nav_tree.addTopLevelItem(results_item)

        self._nav_dock.setWidget(self._nav_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._nav_dock)
        self._toggle_nav_action.toggled.connect(self._nav_dock.setVisible)

        # --- Recording Overview (right) ---
        self._overview_dock = QDockWidget("Recording Overview", self)
        self._overview_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._overview = RecordingOverview()
        self._overview_dock.setWidget(self._overview)
        self.addDockWidget(Qt.RightDockWidgetArea, self._overview_dock)

        # --- Console (bottom) ---
        self._console_dock = QDockWidget("Console", self)
        self._console_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self._console = QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(5000)
        self._console_dock.setWidget(self._console)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._console_dock)
        self._toggle_console_action.toggled.connect(self._console_dock.setVisible)

    # ==================================================================
    # Status bar
    # ==================================================================

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_label = QLabel("Ready")
        sb.addWidget(self._status_label)
        sb.addPermanentWidget(QLabel("Sigma v0.1.0"))

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setVisible(False)
        sb.addPermanentWidget(self._progress_bar)

    # ==================================================================
    # Console logging
    # ==================================================================

    def _log(self, message: str) -> None:
        self._console.appendPlainText(message)

    # ==================================================================
    # Actions / slots
    # ==================================================================

    @Slot()
    def _on_open(self) -> None:
        wizard = InputWizard(self)
        wizard.recording_ready.connect(self._load_recording)
        wizard.exec()

    @Slot()
    def _on_new_project(self) -> None:
        self._log("ℹ New Project — not yet implemented (Phase 2)")

    @Slot()
    def _on_save(self) -> None:
        self._log("ℹ Save — not yet implemented")

    @Slot()
    def _on_export(self) -> None:
        if self._current_metadata is None:
            QMessageBox.information(self, "Export", "No recording loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON (*.json);;HTML (*.html);;All (*)"
        )
        if path:
            from src.reporting.exporter import export_metadata_json
            export_metadata_json(self._current_metadata, path)
            self._log(f"✓ Exported metadata to {path}")

    @Slot()
    def _on_run_analysis(self) -> None:
        if self._current_samples is None:
            QMessageBox.information(self, "Analysis", "No recording loaded.")
            return
        self._log("▶ Running spectral analysis…")
        self._status_label.setText("Analyzing…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)

        # Run in background worker
        samples = self._current_samples
        sr = self._current_metadata.sample_rate_hz if self._current_metadata else 1.0

        def _analyze(progress_cb, cancel_check):
            from src.dsp.spectral import analyze_spectrum
            result = analyze_spectrum(samples, sr)
            return result

        worker = Worker(_analyze)
        worker.signals.finished.connect(self._on_analysis_done)
        worker.signals.error.connect(self._on_analysis_error)
        WorkerPool.instance().start(worker)

    @Slot(object)
    def _on_analysis_done(self, result: object) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText("Ready")
        from src.dsp.spectral import SpectralAnalysis
        if isinstance(result, SpectralAnalysis):
            self._log(f"✓ Analysis complete:")
            self._log(f"  Noise floor: {result.noise_floor_db:.1f} dB")
            self._log(f"  Occupied BW: {result.occupied_bandwidth_hz:,.0f} Hz")
            self._log(f"  Peaks found: {len(result.peaks)}")
            for i, pk in enumerate(result.peaks[:5]):
                self._log(
                    f"  Peak {i+1}: {pk.frequency_hz:,.0f} Hz, "
                    f"{pk.power_db:.1f} dB, SNR {pk.snr_db:.1f} dB"
                )

    @Slot(str)
    def _on_analysis_error(self, error: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText("Error")
        self._log(f"✗ Analysis error: {error}")

    @Slot()
    def _on_settings(self) -> None:
        self._log("ℹ Settings — not yet implemented")

    @Slot()
    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Sigma",
            "Sigma Signal Analysis Platform\n"
            "Version 0.1.0 — Phase 1 MVP\n\n"
            "Automated RF signal analysis, classification,\n"
            "demodulation, and decoding workstation.",
        )

    # ==================================================================
    # Recording loading
    # ==================================================================

    @Slot(object, object)
    def _load_recording(self, path: str, meta: RecordingMetadata) -> None:
        """Load a recording from the wizard result."""
        self._log(f"📂 Loading: {path}")
        self._status_label.setText("Loading…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)

        def _do_load(progress_cb, cancel_check):
            from src.ingestion.wav_reader import WavReader
            from src.ingestion.raw_iq_reader import RawIQReader
            from src.ingestion.sigmf_reader import SigMFReader

            if meta.source_format == FileFormat.WAV:
                reader = WavReader(path, interpretation=meta.wav_interpretation)
            elif meta.source_format == FileFormat.SIGMF:
                reader = SigMFReader(path)
            else:
                reader = RawIQReader(
                    path,
                    datatype=meta.sample_datatype,
                    sample_rate_hz=meta.sample_rate_hz,
                    iq_order=meta.iq_order,
                    byte_offset=meta.byte_offset,
                )

            progress_cb(0.1, "Validating file…")
            report = reader.validate()

            progress_cb(0.3, "Reading metadata…")
            loaded_meta = reader.read_metadata()

            # Override with user-provided values
            if meta.sample_rate_hz > 0:
                loaded_meta.sample_rate_hz = meta.sample_rate_hz
            if meta.center_frequency_hz > 0:
                loaded_meta.center_frequency_hz = meta.center_frequency_hz

            progress_cb(0.5, "Reading samples…")
            # Read up to 10M samples for initial display
            max_preview = min(10_000_000, reader.total_samples())
            samples = reader.read_samples(0, max_preview)

            progress_cb(1.0, "Done")
            return loaded_meta, samples, report

        worker = Worker(_do_load)
        worker.signals.finished.connect(self._on_recording_loaded)
        worker.signals.error.connect(self._on_load_error)
        WorkerPool.instance().start(worker)

    @Slot(object)
    def _on_recording_loaded(self, result: object) -> None:
        meta, samples, report = result
        self._current_metadata = meta
        self._current_samples = samples

        self._progress_bar.setVisible(False)
        self._status_label.setText("Ready")

        # Update overview
        self._overview.update_metadata(meta)
        self._overview.update_validation(report)

        # Add to navigator
        rec_item = self._nav_tree.topLevelItem(0)
        name = Path(meta.source_path).name if meta.source_path else "Unknown"
        child = QTreeWidgetItem([name, "✓ Loaded"])
        rec_item.addChild(child)
        rec_item.setExpanded(True)

        # Switch to viewers and load data
        self._show_viewers()

        sr = meta.sample_rate_hz if meta.sample_rate_hz > 0 else 1.0
        self._time_viewer.set_data(samples, sr)
        self._spectrum_viewer.set_data(samples, sr)
        self._waterfall_viewer.set_data(samples, sr)
        self._time_viewer.enable_region_selection(True)

        self._log(f"✓ Loaded {len(samples):,} samples at {sr:,.0f} Hz")
        self._log(f"  Format: {meta.source_format.value} | "
                   f"Type: {meta.sample_datatype.value} | "
                   f"Duration: {meta.duration_seconds:.3f}s")

        if report.warnings:
            for w in report.warnings:
                self._log(f"  ⚠ {w}")

    @Slot(str)
    def _on_load_error(self, error: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText("Error")
        self._log(f"✗ Load error: {error}")
        QMessageBox.critical(self, "Load Error", f"Failed to load recording:\n{error}")
