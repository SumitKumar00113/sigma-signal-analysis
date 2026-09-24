"""Main application window.

Assembles the menu bar, toolbar, project navigator, visualisation workspace,
parameter inspector, and console panel following the layout in PRD §8.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
)

from src.core.enums import FileFormat
from src.core.models import RecordingMetadata
from src.dsp.pipeline import AnalysisPipeline, PipelineConfig, PipelineResult
from src.gui.decoding_panel import DecodingPanel
from src.gui.input_wizard import InputWizard
from src.gui.recording_overview import RecordingOverview
from src.gui.results_panel import ResultsPanel
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
        self.setWindowTitle("Sigma Signal Analysis — v0.2.0")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 960)

        self._current_metadata: RecordingMetadata | None = None
        self._current_samples: np.ndarray | None = None
        self._last_result: PipelineResult | None = None
        self._analysis_running = False
        self._classifier_mode = "hybrid"
        self._model = None                      # explicitly loaded model (else default)
        self._training = False

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

        self._save_audio_action = QAction("Save Demodulated &Audio…", self)
        self._save_audio_action.triggered.connect(self._on_save_audio)
        self._save_audio_action.setEnabled(False)
        file_menu.addAction(self._save_audio_action)

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
        analysis_menu.addSeparator()

        from src.ml.model import ml_available

        has_ml = ml_available()
        clf_menu = analysis_menu.addMenu("&Classifier")
        group = QActionGroup(self)
        group.setExclusive(True)
        self._classifier_actions: dict[str, QAction] = {}
        for mode, label, tip in (
            ("hybrid", "&Hybrid (rules + learned model)",
             "Rule-based decision, corrected by the learned model where it is much surer"),
            ("rules", "&Rules only", "Explainable feature/decision-tree classifier"),
            ("ml", "&Learned model only", "Gradient-boosted classifier trained on data"),
        ):
            act = QAction(label, self, checkable=True)
            act.setToolTip(tip)
            act.setData(mode)
            act.setChecked(mode == self._classifier_mode)
            act.setEnabled(has_ml or mode == "rules")
            act.triggered.connect(lambda _c=False, m=mode: self._set_classifier_mode(m))
            group.addAction(act)
            clf_menu.addAction(act)
            self._classifier_actions[mode] = act
        if not has_ml:
            self._classifier_mode = "rules"
            self._classifier_actions["rules"].setChecked(True)
        clf_menu.addSeparator()
        load_model = QAction("Load Classifier &Model…", self)
        load_model.triggered.connect(self._on_load_model)
        load_model.setEnabled(has_ml)
        clf_menu.addAction(load_model)
        self._train_action = QAction("&Train Classifier…", self)
        self._train_action.triggered.connect(self._on_train_classifier)
        self._train_action.setEnabled(has_ml)
        clf_menu.addAction(self._train_action)
        if not has_ml:
            for a in (load_model, self._train_action):
                a.setToolTip("Install the ML extras: pip install -e '.[ml]'")

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
        self._decoding_panel = DecodingPanel()

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
        self._central_stack.addTab(self._decoding_panel, "🔣 Bitstream & Decoding")

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
        self._nav_tree.itemClicked.connect(self._on_nav_item_clicked)

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

        # --- Analysis Results (right, tabbed with overview) ---
        self._results_dock = QDockWidget("Analysis Results", self)
        self._results_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._results = ResultsPanel()
        self._results.rerun_requested.connect(self._on_run_analysis)
        self._results_dock.setWidget(self._results)
        self.addDockWidget(Qt.RightDockWidgetArea, self._results_dock)
        self.tabifyDockWidget(self._overview_dock, self._results_dock)
        self._overview_dock.raise_()

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
        sb.addPermanentWidget(QLabel("Sigma v0.2.0"))

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
    # ---- classifier ------------------------------------------------------

    def _set_classifier_mode(self, mode: str) -> None:
        self._classifier_mode = mode
        if mode != "rules" and self._model is None:
            from src.ml.model import default_model_path

            path = default_model_path()
            if path is None:
                self._log("ℹ No learned model found — Analysis → Classifier → Train "
                          "Classifier… to create one; using rules until then.")
            else:
                self._log(f"ℹ Classifier: {mode} (model {path})")
        else:
            self._log(f"ℹ Classifier: {mode}")

    def _on_load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load classifier model", "",
                                              "Sigma model (*.joblib);;All (*)")
        if not path:
            return
        from src.ml.model import ModulationModel
        try:
            self._model = ModulationModel.load(path)
        except Exception as exc:  # noqa: BLE001
            self._log(f"✗ Could not load model: {exc}")
            return
        acc = self._model.info.get("holdout_accuracy")
        extra = f", held-out accuracy {acc:.1%}" if isinstance(acc, float) else ""
        self._log(f"✓ Loaded classifier model {path} ({len(self._model.classes)} classes{extra})")

    def _on_train_classifier(self) -> None:
        if self._training:
            self._log("ℹ Training already running.")
            return
        from src.gui.train_dialog import TrainClassifierDialog

        dlg = TrainClassifierDialog(self)
        if dlg.exec() != TrainClassifierDialog.Accepted:
            return
        opts = dlg.options()

        def _train(progress_cb, cancel_check):
            from src.ml.train import run_training

            return run_training(opts["per_class"], opts["snr_range"], opts["manifest"],
                                workers=opts["workers"], progress_cb=progress_cb)

        self._training = True
        self._train_action.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._log(f"▶ Training classifier ({opts['per_class']} examples/class"
                  + (f" + {opts['manifest'].name}" if opts["manifest"] else "") + ")…")
        worker = Worker(_train)
        worker.signals.progress.connect(self._on_analysis_progress)
        worker.signals.finished.connect(self._on_training_done)
        worker.signals.error.connect(self._on_training_error)
        WorkerPool.instance().start(worker)

    @Slot(object)
    def _on_training_done(self, res: object) -> None:
        self._training = False
        self._train_action.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText("Ready")
        self._model = getattr(res, "model", None)
        rec = getattr(res, "recording_accuracy", None)
        self._log(f"✓ Classifier trained on {res.n_examples:,} examples in {res.seconds:.0f} s: "
                  f"held-out accuracy {res.holdout_accuracy:.1%}"
                  + (f", recordings {rec:.1%}" if rec is not None else "")
                  + f". Saved to {res.path}")

    @Slot(str)
    def _on_training_error(self, error: str) -> None:
        self._training = False
        self._train_action.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._log(f"✗ Training failed: {error}")

    def _on_save_audio(self) -> None:
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Demodulated Audio", "",
                                              "WAV audio (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        from src.reporting.exporter import export_audio_wav
        try:
            seconds = export_audio_wav(self._last_result, path)
            self._log(f"✓ Saved {seconds:.1f} s of demodulated audio to {path}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"✗ Audio export failed: {exc}")

    def _on_export(self) -> None:
        if self._current_metadata is None:
            QMessageBox.information(self, "Export", "No recording loaded.")
            return
        path, flt = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON (*.json);;HTML report (*.html);;All (*)"
        )
        if not path:
            return
        from src.reporting.exporter import (
            export_html_report,
            export_metadata_json,
            export_pipeline_json,
        )
        try:
            if flt.startswith("HTML") or path.lower().endswith(".html"):
                export_html_report(self._current_metadata, path=path, pipeline=self._last_result)
                self._log(f"✓ Exported HTML report to {path}")
            elif self._last_result is not None:
                export_pipeline_json(self._last_result, path)
                self._log(f"✓ Exported analysis + bits (JSON) to {path}")
            else:
                export_metadata_json(self._current_metadata, path)
                self._log(f"✓ Exported metadata to {path} (run analysis to include results)")
        except Exception as exc:  # noqa: BLE001
            self._log(f"✗ Export failed: {exc}")
            QMessageBox.critical(self, "Export", f"Export failed:\n{exc}")

    @Slot()
    def _on_run_analysis(self) -> None:
        if self._current_samples is None or self._current_metadata is None:
            QMessageBox.information(self, "Analysis", "No recording loaded.")
            return
        if self._analysis_running:
            self._log("ℹ Analysis already running.")
            return

        cfg = PipelineConfig(
            modulation_override=self._results.modulation_override(),
            symbol_rate_override=self._results.symbol_rate_override(),
            classifier_mode=self._classifier_mode,
            model=self._model,
        )
        ov = []
        if cfg.modulation_override:
            ov.append(f"modulation={cfg.modulation_override.value}")
        if cfg.symbol_rate_override:
            ov.append(f"symbol rate={cfg.symbol_rate_override:,.0f} baud")
        suffix = f" (override: {', '.join(ov)})" if ov else ""
        self._log(f"▶ Running analysis pipeline{suffix}…")
        self._status_label.setText("Analyzing…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._analysis_running = True
        self._last_result = None
        self._results.set_busy(True)

        samples = self._current_samples
        meta = self._current_metadata

        def _analyze(progress_cb, cancel_check):
            pipeline = AnalysisPipeline(cfg)
            pipeline.on_progress = progress_cb
            return pipeline.run(samples, meta)

        worker = Worker(_analyze)
        worker.signals.progress.connect(self._on_analysis_progress)
        worker.signals.finished.connect(self._on_analysis_done)
        worker.signals.error.connect(self._on_analysis_error)
        WorkerPool.instance().start(worker)

    @Slot(float, str)
    def _on_analysis_progress(self, fraction: float, message: str) -> None:
        self._progress_bar.setValue(int(fraction * 100))
        self._status_label.setText(message)

    @Slot(object)
    def _on_analysis_done(self, result: object) -> None:
        self._analysis_running = False
        self._results.set_busy(False)
        self._progress_bar.setVisible(False)
        self._status_label.setText("Ready")
        if not isinstance(result, PipelineResult):
            return
        self._last_result = result
        a = result.analysis
        self._show_result(result)
        self._results_dock.raise_()
        has_audio = result.demod is not None and result.demod.audio is not None

        # Navigator tree
        self._populate_tree(result)

        # Console summary
        self._log("✓ Analysis complete:")
        self._log(f"  Modulation: {a.modulation.value} ({a.modulation_confidence:.0%}) "
                  f"— {a.overall_confidence.value}; decided by {result.classifier_source}")
        if result.model_prediction is not None and result.classifier_source != "model":
            p = result.model_prediction
            self._log(f"  Learned model: {p.modulation.value} ({p.probability:.0%})")
        if a.symbol_rate_hz > 0:
            self._log(f"  Symbol rate: {a.symbol_rate_hz:,.1f} baud "
                      f"({a.symbol_rate_confidence:.0%})")
        self._log(f"  SNR: {result.snr_db:.1f} dB (in-band {result.snr_inband_db:.1f} dB) | "
                  f"Carrier offset: {result.frequency_offset_hz:+,.0f} Hz | "
                  f"Occupied BW: {result.occupied_bandwidth_hz:,.0f} Hz")
        if has_audio:
            d = result.demod
            self._log(f"  Demodulated {d.modulation.value} to "
                      f"{len(d.audio) / d.audio_rate_hz:.1f} s of audio — "
                      "File → Save Demodulated Audio… to listen")
        elif result.demod is not None:
            self._log(f"  Demodulated {result.demod.num_symbols:,} symbols → "
                      f"{result.demod.num_bits:,} bits, EVM {result.demod.evm_percent:.1f}%")
        for stage, err in result.stage_errors.items():
            self._log(f"  ⚠ {stage}: {err}")
        self._log(f"  ({result.processing_time_ms:,.0f} ms)")

    def _show_result(self, result: PipelineResult) -> None:
        """Results panel, constellation, bits and burst overlay for the
        current main result (after an analysis or a burst selection)."""
        self._results.update_result(result)
        has_audio = result.demod is not None and result.demod.audio is not None
        self._save_audio_action.setEnabled(has_audio)
        if result.demod is not None and not has_audio:
            d = result.demod
            self._constellation_viewer.set_symbols(
                d.symbols, f"{d.modulation.value}  ·  EVM {d.evm_percent:.1f}%"
            )
            src = f"{d.modulation.value} @ {d.symbol_rate_hz:,.0f} baud"
            if result.primary_burst is not None and result.bursts:
                src += f", burst {result.bursts[result.primary_burst].index + 1}"
            self._decoding_panel.set_bits(d.bits, src)
        else:
            self._constellation_viewer.clear()
            self._decoding_panel.clear()
        centre = result.metadata.center_frequency_hz
        primary = (result.bursts[result.primary_burst].index
                   if result.primary_burst is not None and result.bursts else None)
        self._waterfall_viewer.set_regions([
            (r.start_time_sec, r.end_time_sec,
             r.center_frequency_hz - centre - r.bandwidth_hz / 2,
             r.center_frequency_hz - centre + r.bandwidth_hz / 2,
             f"{i + 1}: {r.label}" if r.label else str(i + 1), i == primary)
            for i, r in enumerate(result.regions)
        ] if len(result.regions) > 1 or result.bursts else [])

    def _on_nav_item_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        idx = item.data(0, Qt.UserRole)
        result = self._last_result
        if not isinstance(idx, int) or result is None:
            return
        burst = next((b for b in result.bursts if b.index == idx), None)
        if burst is None:
            return
        AnalysisPipeline.adopt_burst(result, burst)
        self._show_result(result)
        a = result.analysis
        self._log(f"▶ Burst {idx + 1} ({burst.region.start_time_sec:.3f}–"
                  f"{burst.region.end_time_sec:.3f} s, {burst.offset_hz:+,.0f} Hz): "
                  f"{a.modulation.value} ({a.modulation_confidence:.0%})"
                  + (f", {a.symbol_rate_hz:,.1f} baud" if a.symbol_rate_hz > 0 else ""))

    def _populate_tree(self, result: PipelineResult) -> None:
        """Fill the Regions / Jobs / Results nodes of the project navigator."""
        regions_item = self._nav_tree.topLevelItem(1)
        jobs_item = self._nav_tree.topLevelItem(2)
        results_item = self._nav_tree.topLevelItem(3)
        for item in (regions_item, results_item):
            item.takeChildren()

        centre = result.metadata.center_frequency_hz
        for i, r in enumerate(result.regions):
            label = f" · {r.label}" if r.label else ""
            child = QTreeWidgetItem([
                f"Burst {i + 1}{label}: {r.start_time_sec:.3f}–{r.end_time_sec:.3f} s, "
                f"{r.center_frequency_hz - centre:+,.0f} Hz, BW {r.bandwidth_hz / 1e3:,.1f} kHz",
                f"SNR {r.snr_db:.1f} dB",
            ])
            child.setData(0, Qt.UserRole, i)
            if any(b.index == i for b in result.bursts):
                child.setToolTip(0, "Click to show this burst's analysis")
            regions_item.addChild(child)
        regions_item.setExpanded(True)

        a = result.analysis
        job = QTreeWidgetItem([
            f"Analysis {jobs_item.childCount() + 1}",
            f"✓ {result.processing_time_ms:,.0f} ms",
        ])
        jobs_item.addChild(job)
        jobs_item.setExpanded(True)

        results_item.addChild(QTreeWidgetItem(
            [f"Modulation: {a.modulation.value}", f"{a.modulation_confidence:.0%}"]))
        if a.symbol_rate_hz > 0:
            results_item.addChild(QTreeWidgetItem(
                [f"Symbol rate: {a.symbol_rate_hz:,.0f} baud", f"{a.symbol_rate_confidence:.0%}"]))
        results_item.addChild(QTreeWidgetItem([f"SNR: {result.snr_db:.1f} dB", ""]))
        results_item.addChild(QTreeWidgetItem(
            [f"Carrier offset: {result.frequency_offset_hz:+,.0f} Hz", ""]))
        d = result.demod
        if d is not None and d.audio is not None and d.audio_rate_hz > 0:
            results_item.addChild(QTreeWidgetItem(
                [f"Audio: {len(d.audio) / d.audio_rate_hz:.1f} s", f"{d.audio_rate_hz:,.0f} Hz"]))
        elif d is not None:
            results_item.addChild(QTreeWidgetItem(
                [f"Bits: {d.num_bits:,}", f"EVM {d.evm_percent:.1f}%"]))
        results_item.addChild(QTreeWidgetItem(
            ["Overall", a.overall_confidence.value]))
        results_item.setExpanded(True)

    @Slot(str)
    def _on_analysis_error(self, error: str) -> None:
        self._analysis_running = False
        self._results.set_busy(False)
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
            "Version 0.2.0 — Phase 2\n\n"
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
        # Drop the previous recording so a stale analysis cannot run on it
        self._current_samples = None
        self._current_metadata = None
        self._last_result = None
        self._status_label.setText("Loading…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)

        def _do_load(progress_cb, cancel_check):
            from src.ingestion.raw_iq_reader import RawIQReader
            from src.ingestion.sigmf_reader import SigMFReader
            from src.ingestion.wav_reader import WavReader

            if meta.source_format == FileFormat.WAV:
                reader = WavReader(path, interpretation=meta.wav_interpretation,
                                   channel=meta.wav_channel, swap_iq=meta.wav_swap_iq)
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
        self._last_result = None
        self._results.clear()
        self._constellation_viewer.clear()
        self._decoding_panel.clear()

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
        self._overview_dock.raise_()

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
