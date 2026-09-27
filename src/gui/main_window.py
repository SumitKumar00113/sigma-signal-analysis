"""Main application window.

Layout (mission-control dashboard):

* **Header** — brand, a segmented switcher for the workspace views, the
  primary actions (Run Analysis, Auto-Analyse) and icon buttons for open,
  save, export and settings.
* **Hero line** — the active recording's name with its key properties.
* **Workspace** — project navigator (left), the active viewer (centre),
  and the inspector with recording overview / analysis results (right).
* **Dashboard** — classification, signal quality and parameter cards plus
  the console, all updated after every analysis.

The menu bar still offers every command, with keyboard shortcuts.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.enums import FileFormat
from src.core.models import RecordingMetadata
from src.dsp.pipeline import AnalysisPipeline, PipelineConfig, PipelineResult
from src.gui.decoding_panel import DecodingPanel
from src.gui.icons import icon
from src.gui.input_wizard import InputWizard
from src.gui.recording_overview import RecordingOverview, _format_duration, _format_hz
from src.gui.results_panel import ResultsPanel
from src.gui.settings import (
    CLASSIFIER_MODES,
    PROJECT_SUFFIX,
    AppSettings,
    Preferences,
    Project,
    SettingsDialog,
)
from src.gui.theme import (
    ACCENT_DANGER,
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    ACCENT_SUCCESS,
    ACCENT_WARNING,
    APP_NAME,
    APP_VERSION,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from src.gui.viewers.constellation_viewer import ConstellationViewer
from src.gui.viewers.image_viewer import ImageViewer
from src.gui.viewers.spectrum_viewer import SpectrumViewer
from src.gui.viewers.time_viewer import TimeViewer
from src.gui.viewers.waterfall_viewer import WaterfallViewer
from src.gui.welcome_screen import WelcomeScreen
from src.gui.widgets import (
    Card,
    KeyValueRow,
    MeterRow,
    SegmentedNav,
    SpaceBackdrop,
    icon_button,
    primary_button,
)
from src.gui.workers import Worker, WorkerPool

_PATH_ROLE = Qt.UserRole + 1          # navigator: recording path of a "Recordings" child

_VIEW_HINTS = {
    "Waveform": "Drag the shaded band to choose a span for Analysis → Analyse Selection",
    "Spectrum": "Triangles mark peaks · dashed line is the noise floor",
    "Waterfall": "Boxes outline detected bursts · click a burst in the navigator",
    "Constellation": "Symbols after timing and carrier recovery",
    "Decoding": "Auto-decode, or apply each stage by hand",
    "Image": "Decoded NOAA APT image · File → Save Decoded Image…",
}


class _ViewTabs(QTabWidget):
    """Tab widget without a visible tab bar that announces tab changes, so
    the header's segmented switcher can mirror it."""

    tabs_changed = Signal()

    def tabInserted(self, index: int) -> None:  # noqa: N802 – Qt API
        super().tabInserted(index)
        self.tabs_changed.emit()

    def tabRemoved(self, index: int) -> None:  # noqa: N802 – Qt API
        super().tabRemoved(index)
        self.tabs_changed.emit()


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — v{APP_VERSION}")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 960)

        self._settings = settings or AppSettings()
        self._prefs: Preferences = self._settings.preferences()

        self._current_metadata: RecordingMetadata | None = None
        self._current_samples: np.ndarray | None = None
        self._current_path: str | None = None
        self._recordings: dict[str, RecordingMetadata] = {}   # path → wizard metadata
        self._loading_input: tuple[str, RecordingMetadata] | None = None
        self._project_path: Path | None = None
        self._last_result: PipelineResult | None = None
        self._analysis_running = False
        self._analysis_offset_s = 0.0           # start of the analysed span in the recording
        self._pending_offset_s = 0.0
        self._auto_after_analysis = False       # one-click: decode once analysis is done
        self._classifier_mode = self._prefs.classifier_mode
        self._model = None                      # explicitly loaded model (else default)
        self._training = False

        from src.ml.model import ml_available

        self._has_ml = ml_available()
        if not self._has_ml:
            self._classifier_mode = "rules"

        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()
        self._apply_preferences(self._prefs, persist=False)

        self._refresh_recent()
        self._show_welcome()
        self._update_actions()

    # ==================================================================
    # Menu bar
    # ==================================================================

    def _action(self, text: str, slot, shortcut=None, tip: str = "",
                icon_name: str | None = None) -> QAction:
        act = QAction(text, self)
        if shortcut is not None:
            act.setShortcut(QKeySequence(shortcut))
        if tip:
            act.setStatusTip(tip)
            act.setToolTip(tip)
        if icon_name:
            act.setIcon(icon(icon_name, TEXT_SECONDARY, 16))
        act.triggered.connect(lambda _checked=False: slot())
        return act

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")
        file_menu.addAction(self._action("&New Project", self._on_new_project,
                                         QKeySequence.New, "Close everything and start over"))
        file_menu.addAction(self._action("&Open Recording…", self._on_open, QKeySequence.Open,
                                         icon_name="open"))
        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._recent_menu.aboutToShow.connect(self._fill_recent_menu)
        file_menu.addAction(self._action("Open &Project…", self._on_open_project,
                                         "Ctrl+Shift+O", icon_name="layers"))
        file_menu.addSeparator()
        self._save_project_action = self._action("&Save Project", self._on_save,
                                                 QKeySequence.Save, icon_name="save")
        file_menu.addAction(self._save_project_action)
        self._save_project_as_action = self._action("Save Project &As…", self._on_save_as,
                                                    "Ctrl+Shift+S")
        file_menu.addAction(self._save_project_as_action)
        file_menu.addSeparator()
        self._export_action = self._action("&Export Results…", self._on_export, "Ctrl+E",
                                           icon_name="export")
        file_menu.addAction(self._export_action)
        self._save_audio_action = self._action("Save Demodulated &Audio…", self._on_save_audio)
        self._save_audio_action.setEnabled(False)
        file_menu.addAction(self._save_audio_action)
        self._save_image_action = self._action("Save Decoded &Image…", self._on_save_image)
        self._save_image_action.setEnabled(False)
        file_menu.addAction(self._save_image_action)
        file_menu.addSeparator()
        settings_action = self._action("&Settings…", self._on_settings, QKeySequence.Preferences,
                                       icon_name="settings")
        settings_action.setMenuRole(QAction.PreferencesRole)
        file_menu.addAction(settings_action)
        quit_action = self._action("&Quit", self.close, QKeySequence.Quit)
        quit_action.setMenuRole(QAction.QuitRole)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = mb.addMenu("&View")
        self._toggle_nav_action = QAction("Project &Navigator", self, checkable=True)
        self._toggle_nav_action.setChecked(True)
        view_menu.addAction(self._toggle_nav_action)
        self._toggle_inspector_action = QAction("&Inspector", self, checkable=True)
        self._toggle_inspector_action.setChecked(True)
        view_menu.addAction(self._toggle_inspector_action)
        self._toggle_dash_action = QAction("&Dashboard Cards", self, checkable=True)
        self._toggle_dash_action.setChecked(True)
        view_menu.addAction(self._toggle_dash_action)
        self._toggle_console_action = QAction("&Console", self, checkable=True)
        self._toggle_console_action.setChecked(True)
        view_menu.addAction(self._toggle_console_action)
        view_menu.addSeparator()
        self._views_menu = view_menu.addMenu("&Go to View")
        self._view_actions: list[QAction] = []
        for i in range(9):
            act = self._action(f"View {i + 1}", lambda i=i: self._select_view(i), f"Ctrl+{i + 1}")
            act.setVisible(False)
            self._views_menu.addAction(act)
            self.addAction(act)            # shortcut works while the menu is closed
            self._view_actions.append(act)

        # Analysis menu
        analysis_menu = mb.addMenu("&Analysis")
        self._run_action = self._action("&Run Analysis", self._on_run_analysis, "Ctrl+R",
                                        icon_name="play")
        analysis_menu.addAction(self._run_action)
        self._selection_action = self._action(
            "Analyse &Selection", self._on_analyse_selection, "Ctrl+Alt+R",
            "Analyse only the span selected in the Waveform view", "selection")
        analysis_menu.addAction(self._selection_action)
        self._auto_action = self._action("&Auto-Analyse (analysis + decoding)",
                                         self._on_auto_analyse, "Ctrl+Shift+R",
                                         icon_name="bolt")
        analysis_menu.addAction(self._auto_action)
        analysis_menu.addSeparator()

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
            act.setEnabled(self._has_ml or mode == "rules")
            act.triggered.connect(lambda _c=False, m=mode: self._set_classifier_mode(m))
            group.addAction(act)
            clf_menu.addAction(act)
            self._classifier_actions[mode] = act
        clf_menu.addSeparator()
        load_model = QAction("Load Classifier &Model…", self)
        load_model.triggered.connect(self._on_load_model)
        load_model.setEnabled(self._has_ml)
        clf_menu.addAction(load_model)
        self._train_action = QAction("&Train Classifier…", self)
        self._train_action.triggered.connect(self._on_train_classifier)
        self._train_action.setEnabled(self._has_ml)
        clf_menu.addAction(self._train_action)
        if not self._has_ml:
            for a in (load_model, self._train_action):
                a.setToolTip("Install the ML extras: pip install -e '.[ml]'")

        # Help menu
        help_menu = mb.addMenu("&Help")
        about_action = self._action("&About Sigma", self._on_about)
        about_action.setMenuRole(QAction.AboutRole)
        help_menu.addAction(about_action)

    # ==================================================================
    # Central area
    # ==================================================================

    def _build_central(self) -> None:
        root = SpaceBackdrop()
        v = QVBoxLayout(root)
        v.setContentsMargins(20, 14, 20, 12)
        v.setSpacing(12)
        v.addLayout(self._build_header())

        self._hero = self._build_hero()
        v.addWidget(self._hero)

        # Welcome page
        self._welcome = WelcomeScreen()
        self._welcome.open_file_requested.connect(self._on_open)
        self._welcome.new_project_requested.connect(self._on_new_project)
        self._welcome.open_project_requested.connect(self._on_open_project)
        self._welcome.train_requested.connect(self._on_train_classifier)
        self._welcome.recent_requested.connect(self._load_recording)
        self._welcome.clear_recent_requested.connect(self._on_clear_recent)
        self._welcome.set_train_enabled(self._has_ml)

        # Viewers
        self._time_viewer = TimeViewer()
        self._spectrum_viewer = SpectrumViewer()
        self._waterfall_viewer = WaterfallViewer()
        self._constellation_viewer = ConstellationViewer()
        self._image_viewer = ImageViewer()
        self._decoding_panel = DecodingPanel()
        self._decoding_panel.auto_decode_finished.connect(self._on_auto_decode_finished)

        self._central_stack = _ViewTabs()
        self._central_stack.setDocumentMode(True)
        self._central_stack.tabBar().hide()
        self._central_stack.tabs_changed.connect(self._sync_view_nav)
        self._central_stack.currentChanged.connect(self._on_view_changed)

        self._body = QStackedWidget()
        self._body.addWidget(self._welcome)
        self._body.addWidget(self._build_workspace())
        v.addWidget(self._body, stretch=1)

        self.setCentralWidget(root)

    def _build_header(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        logo = QLabel("SIGMA")
        logo.setObjectName("brand")
        brand.addWidget(logo)
        tag = QLabel("SIGNAL ANALYSIS")
        tag.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
                          "letter-spacing: 2px; padding-top: 3px;")
        brand.addWidget(tag)
        brand.addStretch()
        grid.addLayout(brand, 0, 0)

        self._view_nav = SegmentedNav()
        self._view_nav.index_selected.connect(self._select_view)
        grid.addWidget(self._view_nav, 0, 1, alignment=Qt.AlignCenter)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addStretch()
        self._run_btn = primary_button("Run Analysis", "play", self._on_run_analysis,
                                       primary=False)
        self._run_btn.setToolTip("Analyse the recording (Ctrl+R)")
        actions.addWidget(self._run_btn)
        self._auto_btn = primary_button("Auto-Analyse", "bolt", self._on_auto_analyse)
        self._auto_btn.setToolTip("Analysis, then bit mapping → interleaver → FEC → framing "
                                  "(Ctrl+Shift+R)")
        actions.addWidget(self._auto_btn)
        actions.addSpacing(6)
        self._open_btn = icon_button("open", "Open recording (Ctrl+O)", self._on_open)
        self._save_btn = icon_button("save", "Save project (Ctrl+S)", self._on_save)
        self._export_btn = icon_button("export", "Export results (Ctrl+E)", self._on_export)
        self._settings_btn = icon_button("settings", "Settings", self._on_settings)
        for b in (self._open_btn, self._save_btn, self._export_btn, self._settings_btn):
            actions.addWidget(b)
        grid.addLayout(actions, 0, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)
        return grid

    def _build_hero(self) -> QWidget:
        hero = QWidget()
        h = QHBoxLayout(hero)
        h.setContentsMargins(2, 0, 2, 0)
        text = QVBoxLayout()
        text.setSpacing(2)
        self._hero_title = QLabel("Signal Analysis Hub")
        self._hero_title.setObjectName("heroTitle")
        text.addWidget(self._hero_title)
        self._hero_subtitle = QLabel("")
        self._hero_subtitle.setObjectName("heroSubtitle")
        text.addWidget(self._hero_subtitle)
        h.addLayout(text)
        h.addStretch()
        self._chips_layout = QHBoxLayout()
        self._chips_layout.setSpacing(6)
        h.addLayout(self._chips_layout)
        return hero

    def _build_workspace(self) -> QWidget:
        self._vsplit = QSplitter(Qt.Vertical)
        self._vsplit.setChildrenCollapsible(False)

        hsplit = QSplitter(Qt.Horizontal)
        hsplit.setChildrenCollapsible(False)

        # --- Project navigator (left) ---
        self._nav_card = Card("Project", "layers")
        self._nav_card.header.addWidget(
            icon_button("plus", "Add recording (Ctrl+O)", self._on_open))
        self._nav_tree = QTreeWidget()
        self._nav_tree.setHeaderHidden(True)
        self._nav_tree.setColumnCount(2)
        self._nav_tree.setIndentation(14)
        self._nav_tree.setUniformRowHeights(False)
        self._nav_tree.itemClicked.connect(self._on_nav_item_clicked)
        for label, icon_name in (("Recordings", "folder"), ("Regions of Interest", "target"),
                                 ("Processing Jobs", "cpu"), ("Results", "list")):
            item = QTreeWidgetItem([label, ""])
            item.setIcon(0, icon(icon_name, ACCENT_PRIMARY, 14))
            self._nav_tree.addTopLevelItem(item)
        header = self._nav_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._nav_card.body.addWidget(self._nav_tree)
        self._nav_card.setMinimumWidth(230)
        hsplit.addWidget(self._nav_card)
        self._toggle_nav_action.toggled.connect(self._nav_card.setVisible)

        # --- Viewer (centre) ---
        self._view_card = Card("Waveform", "wave")
        self._view_hint = QLabel("")
        self._view_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self._view_card.header.addWidget(self._view_hint)
        self._view_card.body.addWidget(self._central_stack)
        self._view_card.setMinimumWidth(520)
        hsplit.addWidget(self._view_card)

        # --- Inspector (right) ---
        self._inspector_card = Card("Inspector", "info")
        self._inspector_tabs = QTabWidget()
        self._overview = RecordingOverview()
        self._results = ResultsPanel()
        self._results.rerun_requested.connect(self._on_run_analysis)
        self._inspector_tabs.addTab(self._overview, "Recording")
        self._inspector_tabs.addTab(self._results, "Analysis")
        self._inspector_card.body.addWidget(self._inspector_tabs)
        self._inspector_card.setMinimumWidth(300)
        hsplit.addWidget(self._inspector_card)
        self._toggle_inspector_action.toggled.connect(self._inspector_card.setVisible)

        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)
        hsplit.setStretchFactor(2, 0)
        hsplit.setSizes([270, 960, 360])
        self._vsplit.addWidget(hsplit)

        self._dashboard = self._build_dashboard()
        self._vsplit.addWidget(self._dashboard)
        self._vsplit.setStretchFactor(0, 1)
        self._vsplit.setStretchFactor(1, 0)
        self._vsplit.setSizes([640, 230])
        return self._vsplit

    def _build_dashboard(self) -> QWidget:
        dash = QWidget()
        row = QHBoxLayout(dash)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        # Classification
        self._dash_clf = Card("Classification", "target")
        self._dash_mod = QLabel("—")
        self._dash_mod.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {TEXT_PRIMARY};")
        self._dash_clf.body.addWidget(self._dash_mod)
        self._dash_mod_sub = QLabel("Run an analysis to classify the signal")
        self._dash_mod_sub.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        self._dash_clf.body.addWidget(self._dash_mod_sub)
        self._dash_candidates = [MeterRow("—", c)
                                 for c in (ACCENT_PRIMARY, ACCENT_WARNING, ACCENT_SECONDARY)]
        for m in self._dash_candidates:
            self._dash_clf.body.addWidget(m)
        self._dash_clf.body.addStretch()

        # Signal quality
        self._dash_quality = Card("Signal Quality", "wave")
        self._dash_snr = MeterRow("SNR (full band)")
        self._dash_snr_in = MeterRow("SNR (in-band)", ACCENT_WARNING)
        self._dash_evm = MeterRow("EVM", ACCENT_SECONDARY)
        for m in (self._dash_snr, self._dash_snr_in, self._dash_evm):
            self._dash_quality.body.addWidget(m)
        self._dash_quality.body.addStretch()

        # Parameters
        self._dash_params = Card("Parameters", "list")
        self._dash_rows: dict[str, KeyValueRow] = {}
        for key, icon_name in (("Symbol rate", "clock"), ("Carrier offset", "target"),
                               ("Occupied BW", "wave"), ("Output", "cpu")):
            r = KeyValueRow(key, "—", icon_name)
            self._dash_rows[key] = r
            self._dash_params.body.addWidget(r)
        self._dash_params.body.addStretch()

        # Console
        self._console_card = Card("Console", "list")
        clear = QPushButton("Clear")
        clear.setCursor(Qt.PointingHandCursor)
        clear.setStyleSheet("padding: 3px 10px; font-size: 11px;")
        clear.clicked.connect(lambda: self._console.clear())
        self._console_card.header.addWidget(clear)
        self._console = QPlainTextEdit()
        self._console.setObjectName("console")
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(5000)
        self._console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._console_card.body.addWidget(self._console)

        for card in (self._dash_clf, self._dash_quality, self._dash_params):
            card.setMinimumWidth(230)
            row.addWidget(card, stretch=2)
        row.addWidget(self._console_card, stretch=3)

        def _toggle_cards(on: bool) -> None:
            for c in (self._dash_clf, self._dash_quality, self._dash_params):
                c.setVisible(on)
            self._update_dashboard_visibility()

        def _toggle_console(on: bool) -> None:
            self._console_card.setVisible(on)
            self._update_dashboard_visibility()

        self._toggle_dash_action.toggled.connect(_toggle_cards)
        self._toggle_console_action.toggled.connect(_toggle_console)
        return dash

    def _update_dashboard_visibility(self) -> None:
        self._dashboard.setVisible(self._toggle_dash_action.isChecked()
                                   or self._toggle_console_action.isChecked())

    # ---- view switching ----------------------------------------------

    def _show_welcome(self) -> None:
        self._central_stack.clear()
        self._body.setCurrentIndex(0)
        self._hero.setVisible(False)
        self._view_nav.setVisible(False)

    def _show_viewers(self) -> None:
        self._central_stack.clear()
        self._central_stack.addTab(self._time_viewer, "Waveform")
        self._central_stack.addTab(self._spectrum_viewer, "Spectrum")
        self._central_stack.addTab(self._waterfall_viewer, "Waterfall")
        self._central_stack.addTab(self._constellation_viewer, "Constellation")
        self._central_stack.addTab(self._decoding_panel, "Decoding")
        self._body.setCurrentIndex(1)
        self._hero.setVisible(True)
        self._sync_view_nav()

    def _select_view(self, index: int) -> None:
        if 0 <= index < self._central_stack.count():
            self._central_stack.setCurrentIndex(index)

    def _sync_view_nav(self) -> None:
        labels = [self._central_stack.tabText(i) for i in range(self._central_stack.count())]
        self._view_nav.set_items(labels, max(0, self._central_stack.currentIndex()))
        for i, act in enumerate(self._view_actions):
            act.setVisible(i < len(labels))
            if i < len(labels):
                act.setText(labels[i])
        self._on_view_changed(self._central_stack.currentIndex())

    def _on_view_changed(self, index: int) -> None:
        if index < 0:
            return
        self._view_nav.set_current(index)
        name = self._central_stack.tabText(index)
        self._view_card.set_title(name)
        self._view_hint.setText(_VIEW_HINTS.get(name, ""))

    # ==================================================================
    # Status bar
    # ==================================================================

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)
        self._status_label = QLabel("Ready")
        sb.addWidget(self._status_label, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setVisible(False)
        sb.addPermanentWidget(self._progress_bar)
        self._mode_label = QLabel("")
        sb.addPermanentWidget(self._mode_label)
        sb.addPermanentWidget(QLabel(f"Sigma v{APP_VERSION}"))

    # ==================================================================
    # Console logging
    # ==================================================================

    def _log(self, message: str) -> None:
        stripped = message.lstrip()
        if stripped.startswith("✓"):
            color = ACCENT_SUCCESS
        elif stripped.startswith("✗"):
            color = ACCENT_DANGER
        elif stripped.startswith("⚠"):
            color = ACCENT_WARNING
        elif stripped.startswith(("▶", "⚡")):
            color = ACCENT_PRIMARY
        elif message.startswith(" "):
            color = TEXT_SECONDARY
        else:
            color = TEXT_PRIMARY
        ts = datetime.now().strftime("%H:%M:%S")
        self._console.appendHtml(
            f'<span style="color:{TEXT_MUTED}">{ts}</span>&nbsp;&nbsp;'
            f'<span style="color:{color}; white-space:pre">{html.escape(message)}</span>')

    # ==================================================================
    # Enable / disable
    # ==================================================================

    def _update_actions(self) -> None:
        loaded = self._current_samples is not None and self._current_metadata is not None
        for w in (self._run_btn, self._auto_btn, self._export_btn,
                  self._run_action, self._auto_action, self._selection_action,
                  self._export_action):
            w.setEnabled(loaded)
        has_project = bool(self._recordings)
        for w in (self._save_btn, self._save_project_action, self._save_project_as_action):
            w.setEnabled(has_project)

    # ==================================================================
    # Actions / slots
    # ==================================================================

    @Slot()
    def _on_open(self) -> None:
        wizard = InputWizard(self)
        wizard.recording_ready.connect(self._load_recording)
        wizard.exec()

    # ---- project ----------------------------------------------------------

    def _reset_session(self) -> None:
        """Forget every recording and result; back to the welcome page."""
        self._current_samples = None
        self._current_metadata = None
        self._current_path = None
        self._last_result = None
        self._recordings.clear()
        self._project_path = None
        self._analysis_offset_s = 0.0
        for i in range(self._nav_tree.topLevelItemCount()):
            self._nav_tree.topLevelItem(i).takeChildren()
        self._overview.clear()
        self._results.clear()
        self._results.reset_overrides()
        self._constellation_viewer.clear()
        self._decoding_panel.clear()
        self._image_viewer.clear()
        self._save_audio_action.setEnabled(False)
        self._save_image_action.setEnabled(False)
        self._clear_dashboard()
        self._update_title()
        self._show_welcome()
        self._update_actions()

    def _busy(self) -> bool:
        return (self._analysis_running or self._training
                or self._decoding_panel.auto_decode_running)

    @Slot()
    def _on_new_project(self) -> None:
        if self._busy():
            self._log("ℹ Wait for the running job to finish first.")
            return
        if self._recordings and QMessageBox.question(
                self, "New Project",
                "Close the current recordings and start a new project?\n"
                "Unsaved analysis results are discarded.") != QMessageBox.Yes:
            return
        self._reset_session()
        self._log("ℹ New project")

    @Slot()
    def _on_save(self) -> None:
        if self._project_path is None:
            self._on_save_as()
        else:
            self._write_project(self._project_path)

    @Slot()
    def _on_save_as(self) -> None:
        if not self._recordings:
            QMessageBox.information(self, "Save Project", "Open a recording first.")
            return
        start = str(self._project_path or Path(self._current_path or "").with_suffix(""))
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", start,
                                              f"Sigma project (*{PROJECT_SUFFIX})")
        if path:
            self._write_project(Path(path))

    def _write_project(self, path: Path) -> None:
        paths = list(self._recordings)
        mod = self._results.modulation_override()
        project = Project(
            recordings=list(self._recordings.items()),
            active=paths.index(self._current_path) if self._current_path in paths else 0,
            classifier_mode=self._classifier_mode,
            modulation_override=mod.value if mod else None,
            symbol_rate_override=self._results.symbol_rate_override(),
        )
        try:
            self._project_path = project.save(path)
        except OSError as exc:
            self._log(f"✗ Could not save project: {exc}")
            QMessageBox.critical(self, "Save Project", f"Could not save the project:\n{exc}")
            return
        self._update_title()
        self._log(f"✓ Saved project to {self._project_path}")

    @Slot()
    def _on_open_project(self) -> None:
        if self._busy():
            self._log("ℹ Wait for the running job to finish first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "",
                                              f"Sigma project (*{PROJECT_SUFFIX});;All (*)")
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> bool:
        try:
            project = Project.load(path)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.critical(self, "Open Project", f"Could not open the project:\n{exc}")
            return False
        missing = [p for p, _ in project.recordings if not Path(p).exists()]
        self._reset_session()
        self._project_path = Path(path)
        mode = project.classifier_mode if self._has_ml else "rules"
        self._classifier_actions[mode].setChecked(True)
        self._set_classifier_mode(mode)
        self._results.set_overrides(project.modulation_override, project.symbol_rate_override)
        for p, meta in project.recordings:
            self._recordings[p] = meta
            self._set_recording_item(p, "Missing" if p in missing else "Not loaded")
        self._update_title()
        self._update_actions()
        self._log(f"✓ Opened project {path} ({len(project.recordings)} recording(s))")
        if missing:
            self._log(f"⚠ {len(missing)} recording(s) not found: " + ", ".join(missing))
        if project.recordings:
            active_path, active_meta = project.recordings[project.active]
            if active_path not in missing:
                self._load_recording(active_path, active_meta)
        return True

    def _update_title(self) -> None:
        name = f"{self._project_path.name} — " if self._project_path else ""
        self.setWindowTitle(f"{name}{APP_NAME} — v{APP_VERSION}")

    # ---- recent -----------------------------------------------------------

    def _refresh_recent(self) -> None:
        self._welcome.set_recent(self._settings.recent())

    def _fill_recent_menu(self) -> None:
        self._recent_menu.clear()
        entries = self._settings.recent()
        for path, meta in entries:
            act = self._recent_menu.addAction(Path(path).name)
            act.setToolTip(path)
            act.setStatusTip(path)
            act.triggered.connect(lambda _c=False, p=path, m=meta: self._load_recording(p, m))
        if not entries:
            none = self._recent_menu.addAction("No recent recordings")
            none.setEnabled(False)
        self._recent_menu.addSeparator()
        clear = self._recent_menu.addAction("Clear Recent")
        clear.setEnabled(bool(entries))
        clear.triggered.connect(self._on_clear_recent)

    def _on_clear_recent(self) -> None:
        self._settings.clear_recent()
        self._refresh_recent()

    # ---- settings -----------------------------------------------------------

    @Slot()
    def _on_settings(self) -> None:
        prefs = Preferences(**{**self._prefs.__dict__, "classifier_mode": self._classifier_mode})
        dlg = SettingsDialog(prefs, self._has_ml, self)
        if dlg.exec() != SettingsDialog.Accepted:
            return
        self._apply_preferences(dlg.preferences())
        self._log("✓ Settings saved")

    def _apply_preferences(self, prefs: Preferences, persist: bool = True) -> None:
        if not self._has_ml:
            prefs.classifier_mode = "rules"
        self._prefs = prefs
        WorkerPool.instance().set_max_threads(prefs.worker_threads)
        self._spectrum_viewer.set_fft_size(prefs.spectrum_fft)
        self._waterfall_viewer.set_defaults(prefs.waterfall_fft, prefs.waterfall_colormap)
        if prefs.classifier_mode != self._classifier_mode or not persist:
            self._classifier_actions[prefs.classifier_mode].setChecked(True)
            self._set_classifier_mode(prefs.classifier_mode, log=persist)
        if persist:
            self._settings.save_preferences(prefs)

    # ---- classifier ------------------------------------------------------

    def _set_classifier_mode(self, mode: str, log: bool = True) -> None:
        self._classifier_mode = mode
        self._mode_label.setText(f"Classifier: {CLASSIFIER_MODES[mode]}")
        if not log:
            return
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
        if not self._has_ml:
            QMessageBox.information(self, "Train Classifier",
                                    "Install the ML extras first:\n  pip install -e '.[ml]'")
            return
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
        self._status_label.setText("Error")
        self._log(f"✗ Training failed: {error}")

    # ---- exports -----------------------------------------------------------

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

    def _on_save_image(self) -> None:
        img = self._image_viewer.image
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Decoded Image", "", "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        from src.decoding.apt import write_png

        try:
            write_png(path, img)
        except OSError as exc:
            self._log(f"✗ Image export failed: {exc}")
            return
        self._log(f"✓ Saved {img.shape[1]} × {img.shape[0]} image to {path}")

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

    # ---- analysis ------------------------------------------------------------

    @Slot()
    def _on_run_analysis(self) -> None:
        self._start_analysis(None)

    @Slot()
    def _on_analyse_selection(self) -> None:
        if self._current_samples is None or self._current_metadata is None:
            QMessageBox.information(self, "Analysis", "No recording loaded.")
            return
        sel = self._time_viewer.selection()
        if sel is None or sel[1] - sel[0] < 256:
            QMessageBox.information(
                self, "Analyse Selection",
                "Select a span first: in the Waveform view, drag the shaded band "
                "(or its edges) over the part of the recording to analyse.")
            self._central_stack.setCurrentWidget(self._time_viewer)
            return
        self._start_analysis(sel)

    def _start_analysis(self, selection: tuple[int, int] | None) -> None:
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
        samples = self._current_samples
        meta = self._current_metadata
        fs = meta.sample_rate_hz if meta.sample_rate_hz > 0 else 1.0
        ov = []
        if cfg.modulation_override:
            ov.append(f"modulation={cfg.modulation_override.value}")
        if cfg.symbol_rate_override:
            ov.append(f"symbol rate={cfg.symbol_rate_override:,.0f} baud")
        suffix = f" (override: {', '.join(ov)})" if ov else ""
        if selection is not None:
            start, end = selection
            samples = samples[start:end]
            self._pending_offset_s = start / fs
            suffix += f" on {start / fs:.3f}–{end / fs:.3f} s"
        else:
            self._pending_offset_s = 0.0
        self._log(f"▶ Running analysis pipeline{suffix}…")
        self._status_label.setText("Analyzing…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._analysis_running = True
        self._last_result = None
        self._results.set_busy(True)
        self._run_btn.setEnabled(False)
        self._auto_btn.setEnabled(False)

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

    def _analysis_finished(self) -> None:
        self._analysis_running = False
        self._results.set_busy(False)
        self._progress_bar.setVisible(False)
        self._update_actions()

    @Slot(object)
    def _on_analysis_done(self, result: object) -> None:
        self._analysis_finished()
        self._status_label.setText("Ready")
        if not isinstance(result, PipelineResult):
            return
        self._analysis_offset_s = self._pending_offset_s
        self._pending_offset_s = 0.0
        self._last_result = result
        a = result.analysis
        self._show_result(result)
        self._inspector_tabs.setCurrentWidget(self._results)
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
                      f"({min(a.symbol_rate_confidence, 1.0):.0%})")
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

        if self._auto_after_analysis:
            self._auto_after_analysis = False
            self._start_auto_decode(result)

    def _show_image(self, result: PipelineResult) -> None:
        """Decoded NOAA APT image: its own view while there is one."""
        idx = self._central_stack.indexOf(self._image_viewer)
        apt = result.apt
        if apt is None:
            self._image_viewer.clear()
            if idx >= 0:
                self._central_stack.removeTab(idx)
            self._save_image_action.setEnabled(False)
            return
        self._image_viewer.set_image(apt.image, apt.describe() + "  ·  channel A | channel B")
        if idx < 0 and self._central_stack.indexOf(self._waterfall_viewer) >= 0:
            self._central_stack.addTab(self._image_viewer, "Image")
        self._save_image_action.setEnabled(True)
        self._log(f"  ✓ {apt.describe()} — see the Image view")

    def _show_result(self, result: PipelineResult) -> None:
        """Results panel, dashboard, constellation, bits and burst overlay for
        the current main result (after an analysis or a burst selection)."""
        self._results.update_result(result)
        self._update_dashboard(result)
        has_audio = result.demod is not None and result.demod.audio is not None
        self._save_audio_action.setEnabled(has_audio)
        self._show_image(result)
        if result.demod is not None and not has_audio:
            d = result.demod
            self._constellation_viewer.set_symbols(
                d.symbols, f"{d.modulation.value}  ·  EVM {d.evm_percent:.1f}%"
            )
            src = f"{d.modulation.value} @ {d.symbol_rate_hz:,.0f} baud"
            bits, n_bursts = AnalysisPipeline.train_bits(result)
            if n_bursts > 1:
                src += f", {n_bursts} bursts of the same signal joined"
            elif result.primary_burst is not None and result.bursts:
                src += f", burst {result.bursts[result.primary_burst].index + 1}"
            self._decoding_panel.set_bits(bits, src, d.bits_per_symbol, d.modulation,
                                         d.evm_percent)
        else:
            self._constellation_viewer.clear()
            self._decoding_panel.clear()
        centre = result.metadata.center_frequency_hz
        t_off = self._analysis_offset_s
        primary = (result.bursts[result.primary_burst].index
                   if result.primary_burst is not None and result.bursts else None)
        self._waterfall_viewer.set_regions([
            (r.start_time_sec + t_off, r.end_time_sec + t_off,
             r.center_frequency_hz - centre - r.bandwidth_hz / 2,
             r.center_frequency_hz - centre + r.bandwidth_hz / 2,
             f"{i + 1}: {r.label}" if r.label else str(i + 1), i == primary)
            for i, r in enumerate(result.regions)
        ] if len(result.regions) > 1 or result.bursts else [])

    # ---- dashboard ------------------------------------------------------------

    def _clear_dashboard(self) -> None:
        self._dash_mod.setText("—")
        self._dash_mod.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {TEXT_PRIMARY};")
        self._dash_mod_sub.setText("Run an analysis to classify the signal")
        for m in self._dash_candidates:
            m.reset("—")
        self._dash_snr.reset()
        self._dash_snr_in.reset()
        self._dash_evm.reset("EVM")
        for r in self._dash_rows.values():
            r.set_value("—")

    def _update_dashboard(self, result: PipelineResult) -> None:
        a = result.analysis
        self._dash_mod.setText(a.modulation.value)
        self._dash_mod.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {ACCENT_PRIMARY};")
        self._dash_mod_sub.setText(f"{a.overall_confidence.value}  ·  {a.modulation_confidence:.0%}"
                                   f"  ·  {result.classifier_source}")
        cands = list(a.modulation_candidates[:3])
        for i, m in enumerate(self._dash_candidates):
            if i < len(cands):
                prob = float(cands[i]["probability"])
                m.set(str(cands[i]["modulation"]), f"{prob:.0%}", prob)
                m.setVisible(True)
            else:
                m.reset("—")
                m.setVisible(i == 0 and not cands)

        def _snr(meter: MeterRow, db: float) -> None:
            color = ACCENT_SUCCESS if db >= 12 else ACCENT_WARNING if db >= 6 else ACCENT_DANGER
            meter.set(None, f"{db:.1f} dB", db / 30.0, color)

        _snr(self._dash_snr, result.snr_db)
        _snr(self._dash_snr_in, result.snr_inband_db)

        d = result.demod
        if d is not None and d.audio is None and d.num_symbols > 0:
            evm = d.evm_percent
            color = ACCENT_SUCCESS if evm < 20 else ACCENT_WARNING if evm < 35 else ACCENT_DANGER
            self._dash_evm.set("EVM", f"{evm:.1f} %", evm / 50.0, color)
        else:
            self._dash_evm.set("EVM", "n/a", 0.0)

        rs = a.symbol_rate_hz
        self._dash_rows["Symbol rate"].set_value(f"{rs:,.1f} baud" if rs > 0 else "—")
        self._dash_rows["Carrier offset"].set_value(f"{result.frequency_offset_hz:+,.0f} Hz")
        self._dash_rows["Occupied BW"].set_value(_format_hz(result.occupied_bandwidth_hz))
        if d is not None and d.audio is not None and d.audio_rate_hz > 0:
            out = f"{len(d.audio) / d.audio_rate_hz:.1f} s audio"
        elif d is not None:
            out = f"{d.num_bits:,} bits"
        else:
            out = "—"
        self._dash_rows["Output"].set_value(out)

    # ---- navigator ---------------------------------------------------------

    def _set_recording_item(self, path: str, status: str) -> None:
        """Add (or update) the navigator entry for recording *path*."""
        rec_item = self._nav_tree.topLevelItem(0)
        for i in range(rec_item.childCount()):
            child = rec_item.child(i)
            if child.data(0, _PATH_ROLE) == path:
                break
        else:
            child = QTreeWidgetItem([Path(path).name, ""])
            child.setData(0, _PATH_ROLE, path)
            child.setToolTip(0, path)
            rec_item.addChild(child)
        child.setText(1, status)
        rec_item.setExpanded(True)
        if status == "Active":
            for i in range(rec_item.childCount()):
                other = rec_item.child(i)
                if other is not child and other.text(1) == "Active":
                    other.setText(1, "Loaded")

    def _on_nav_item_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        path = item.data(0, _PATH_ROLE)
        if isinstance(path, str):
            if path == self._current_path or path not in self._recordings:
                return
            if self._busy():
                self._log("ℹ Wait for the running job to finish first.")
                return
            if not Path(path).exists():
                self._log(f"✗ Recording not found: {path}")
                return
            self._load_recording(path, self._recordings[path])
            return
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
        t0 = burst.region.start_time_sec + self._analysis_offset_s
        t1 = burst.region.end_time_sec + self._analysis_offset_s
        self._log(f"▶ Burst {idx + 1} ({t0:.3f}–{t1:.3f} s, {burst.offset_hz:+,.0f} Hz): "
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
        t_off = self._analysis_offset_s
        for i, r in enumerate(result.regions):
            label = f" · {r.label}" if r.label else ""
            t0, t1 = r.start_time_sec + t_off, r.end_time_sec + t_off
            child = QTreeWidgetItem([f"Burst {i + 1}{label}", f"{r.snr_db:.1f} dB"])
            child.setToolTip(0, f"{t0:.3f}–{t1:.3f} s, {r.center_frequency_hz - centre:+,.0f} Hz, "
                                f"BW {r.bandwidth_hz / 1e3:,.1f} kHz, SNR {r.snr_db:.1f} dB"
                             + ("\nClick to show this burst's analysis"
                                if any(b.index == i for b in result.bursts) else ""))
            child.setData(0, Qt.UserRole, i)
            regions_item.addChild(child)
        regions_item.setExpanded(True)

        a = result.analysis
        job = QTreeWidgetItem([f"Analysis {jobs_item.childCount() + 1}",
                               f"{result.processing_time_ms:,.0f} ms"])
        job.setToolTip(0, f"{a.modulation.value}, finished {datetime.now():%H:%M:%S}")
        jobs_item.addChild(job)
        jobs_item.setExpanded(True)

        def add(key: str, value: str) -> None:
            results_item.addChild(QTreeWidgetItem([key, value]))

        add("Modulation", f"{a.modulation.value} {a.modulation_confidence:.0%}")
        if a.symbol_rate_hz > 0:
            add("Symbol rate", f"{a.symbol_rate_hz:,.0f} Bd")
        add("SNR", f"{result.snr_db:.1f} dB")
        add("Carrier offset", f"{result.frequency_offset_hz:+,.0f} Hz")
        d = result.demod
        if d is not None and d.audio is not None and d.audio_rate_hz > 0:
            add("Audio", f"{len(d.audio) / d.audio_rate_hz:.1f} s")
        elif d is not None:
            add("Bits", f"{d.num_bits:,}")
            add("EVM", f"{d.evm_percent:.1f}%")
        add("Overall", a.overall_confidence.value)
        results_item.setExpanded(True)

    # ---- One-click auto-analyse ------------------------------------------

    @Slot()
    def _on_auto_analyse(self) -> None:
        """Analysis, then the full decoding chain on the demodulated bits."""
        if self._current_samples is None or self._current_metadata is None:
            QMessageBox.information(self, "Auto-Analyse", "No recording loaded.")
            return
        if self._analysis_running or self._decoding_panel.auto_decode_running:
            self._log("ℹ Analysis already running.")
            return
        self._log("⚡ Auto-Analyse: analysis → bit mapping → interleaver → FEC → framing")
        self._auto_after_analysis = True
        self._on_run_analysis()
        if not self._analysis_running:           # analysis did not start
            self._auto_after_analysis = False

    def _start_auto_decode(self, result: PipelineResult) -> None:
        d = result.demod
        if d is None:
            self._log("⚡ Auto-Analyse: nothing was demodulated, so there is no bit stream.")
            return
        if d.audio is not None:
            if result.apt is not None:
                self._central_stack.setCurrentWidget(self._image_viewer)
                self._log("⚡ Auto-Analyse: NOAA APT weather image decoded — Image view "
                          "(File → Save Decoded Image…).")
                return
            self._log(f"⚡ Auto-Analyse: {d.modulation.value} is analog — the demodulated "
                      "audio is ready (File → Save Demodulated Audio…); no bits to decode.")
            return
        if d.num_bits < 256:
            self._log(f"⚡ Auto-Analyse: only {d.num_bits} bits demodulated; need ≥ 256.")
            return
        self._central_stack.setCurrentWidget(self._decoding_panel)
        self._status_label.setText("Auto-decoding…")
        self._log(f"⚡ Decoding {d.num_bits:,} bits…")
        self._decoding_panel.run_auto_decode()

    @Slot(object)
    def _on_auto_decode_finished(self, res: object) -> None:
        self._status_label.setText("Ready")
        self._log("⚡ Auto-decode result:")
        for line in res.summary().splitlines():
            self._log(f"  {line}")
        for f in res.framing.fields:
            self._log(f"    bits {f.start}–{f.start + f.length - 1}: {f.description}")

    @Slot(str)
    def _on_analysis_error(self, error: str) -> None:
        self._analysis_finished()
        self._status_label.setText("Error")
        self._auto_after_analysis = False
        self._pending_offset_s = 0.0
        self._log(f"✗ Analysis error: {error}")

    @Slot()
    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Sigma",
            f"<b>{APP_NAME}</b><br>Version {APP_VERSION}<br><br>"
            "Automated RF signal analysis, classification,<br>"
            "demodulation, and decoding workstation.",
        )

    # ==================================================================
    # Recording loading
    # ==================================================================

    @Slot(object, object)
    def _load_recording(self, path: str, meta: RecordingMetadata) -> None:
        """Load a recording from the wizard result (or a recent/project entry)."""
        if self._busy():
            self._log("ℹ Wait for the running job to finish first.")
            return
        if not Path(path).exists():
            self._log(f"✗ Recording not found: {path}")
            QMessageBox.warning(self, "Open Recording", f"File not found:\n{path}")
            self._refresh_recent()
            return
        self._log(f"▶ Loading: {path}")
        # Drop the previous recording so a stale analysis cannot run on it
        self._current_samples = None
        self._current_metadata = None
        self._last_result = None
        self._loading_input = (path, meta)
        self._update_actions()
        self._status_label.setText("Loading…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        max_samples = self._prefs.max_samples_millions * 1_000_000

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
            samples = reader.read_samples(0, min(max_samples, reader.total_samples()))

            progress_cb(1.0, "Done")
            return loaded_meta, samples, report

        worker = Worker(_do_load)
        worker.signals.finished.connect(self._on_recording_loaded)
        worker.signals.error.connect(self._on_load_error)
        WorkerPool.instance().start(worker)

    @Slot(object)
    def _on_recording_loaded(self, result: object) -> None:
        meta, samples, report = result
        path, input_meta = self._loading_input or (meta.source_path, meta)
        self._loading_input = None
        self._current_metadata = meta
        self._current_samples = samples
        self._current_path = path
        self._recordings[path] = input_meta
        self._last_result = None
        self._analysis_offset_s = 0.0
        self._results.clear()
        self._constellation_viewer.clear()
        self._decoding_panel.clear()
        self._image_viewer.clear()
        self._save_audio_action.setEnabled(False)
        self._save_image_action.setEnabled(False)
        self._clear_dashboard()
        for i in (1, 3):                      # regions and results belong to the old recording
            self._nav_tree.topLevelItem(i).takeChildren()

        self._progress_bar.setVisible(False)
        self._status_label.setText("Ready")

        # Update overview
        self._overview.update_metadata(meta)
        self._overview.update_validation(report)
        self._set_recording_item(path, "Active")
        self._settings.add_recent(path, input_meta)
        self._refresh_recent()

        # Switch to viewers and load data
        self._show_viewers()
        self._update_hero(path, meta, len(samples))
        self._update_actions()

        sr = meta.sample_rate_hz if meta.sample_rate_hz > 0 else 1.0
        self._time_viewer.set_data(samples, sr)
        self._spectrum_viewer.set_data(samples, sr)
        self._waterfall_viewer.set_data(samples, sr)
        self._time_viewer.enable_region_selection(True)
        self._inspector_tabs.setCurrentWidget(self._overview)

        self._log(f"✓ Loaded {len(samples):,} samples at {sr:,.0f} Hz")
        self._log(f"  Format: {meta.source_format.value} | "
                  f"Type: {meta.sample_datatype.value} | "
                  f"Duration: {meta.duration_seconds:.3f}s")
        if meta.sample_count > len(samples):
            self._log(f"  ⚠ Showing the first {len(samples):,} of {meta.sample_count:,} samples "
                      "(raise the limit in Settings)")

        if report.warnings:
            for w in report.warnings:
                self._log(f"  ⚠ {w}")

    def _update_hero(self, path: str, meta: RecordingMetadata, n_loaded: int) -> None:
        p = Path(path)
        self._hero_title.setText(p.name)
        self._hero_subtitle.setText(str(p.parent))
        while self._chips_layout.count():
            w = self._chips_layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        chips = [meta.source_format.value, meta.sample_datatype.value]
        if meta.sample_rate_hz > 0:
            chips.append(f"fs {_format_hz(meta.sample_rate_hz)}")
        if meta.center_frequency_hz > 0:
            chips.append(f"fc {_format_hz(meta.center_frequency_hz)}")
        chips.append(_format_duration(meta.duration_seconds))
        chips.append(f"{n_loaded:,} samples")
        for text in chips:
            chip = QLabel(text)
            chip.setObjectName("chip")
            chip.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self._chips_layout.addWidget(chip, alignment=Qt.AlignVCenter)

    @Slot(str)
    def _on_load_error(self, error: str) -> None:
        self._loading_input = None
        self._progress_bar.setVisible(False)
        self._status_label.setText("Error")
        self._update_actions()
        self._log(f"✗ Load error: {error}")
        QMessageBox.critical(self, "Load Error", f"Failed to load recording:\n{error}")
