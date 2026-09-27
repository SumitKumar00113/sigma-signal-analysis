"""Welcome / landing screen.

Quick actions (open a recording, open a project, train the classifier),
the recently opened recordings, and a summary of what the platform does.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.models import RecordingMetadata
from src.gui.icons import icon, pixmap
from src.gui.theme import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    ACCENT_SUCCESS,
    APP_VERSION,
    CARD_BG,
    CARD_BORDER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from src.gui.widgets import Card, KeyValueRow


class _ActionCard(QFrame):
    """A clickable card on the welcome screen."""

    clicked = Signal()

    def __init__(
        self,
        icon_name: str,
        title: str,
        subtitle: str,
        accent: str = ACCENT_PRIMARY,
        shortcut: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("actionCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(150)
        self.setStyleSheet(f"""
            QFrame#actionCard {{
                background-color: {CARD_BG};
                border: 1px solid {CARD_BORDER};
                border-radius: 14px;
            }}
            QFrame#actionCard:hover {{
                border-color: {accent};
                background-color: rgba(255, 255, 255, 0.05);
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        top = QHBoxLayout()
        badge = QLabel()
        badge.setFixedSize(40, 40)
        badge.setAlignment(Qt.AlignCenter)
        badge.setPixmap(pixmap(icon_name, accent, 22))
        badge.setStyleSheet(f"background-color: rgba(255,255,255,0.05); "
                            f"border: 1px solid {CARD_BORDER}; border-radius: 10px;")
        top.addWidget(badge)
        top.addStretch()
        if shortcut:
            sc = QLabel(shortcut)
            sc.setObjectName("chip")
            top.addWidget(sc, alignment=Qt.AlignTop)
        layout.addLayout(top)
        layout.addSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {TEXT_PRIMARY};")
        layout.addWidget(title_label)

        sub_label = QLabel(subtitle)
        sub_label.setStyleSheet(f"font-size: 12px; color: {TEXT_SECONDARY};")
        sub_label.setWordWrap(True)
        layout.addWidget(sub_label)
        layout.addStretch()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class WelcomeScreen(QWidget):
    """Landing screen shown when no recording is loaded."""

    open_file_requested = Signal()
    new_project_requested = Signal()
    open_project_requested = Signal()
    train_requested = Signal()
    recent_requested = Signal(str, object)       # (path, RecordingMetadata)
    clear_recent_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(1)

        container = QWidget()
        container.setMaximumWidth(1080)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(18)

        title = QLabel("Signal Analysis Hub")
        title.setStyleSheet(f"font-size: 34px; font-weight: 700; color: {TEXT_PRIMARY};")
        layout.addWidget(title)
        subtitle = QLabel("Inspect, classify, demodulate and decode RF recordings — "
                          "from raw IQ to decoded frames in one click.")
        subtitle.setStyleSheet(f"font-size: 14px; color: {TEXT_SECONDARY};")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        layout.addSpacing(6)

        # Action cards row
        cards = QHBoxLayout()
        cards.setSpacing(14)
        open_card = _ActionCard(
            "open", "Open Recording",
            "WAV, raw IQ or SigMF. The input wizard detects the format and helps "
            "with sample rate and IQ layout.",
            ACCENT_PRIMARY, "Ctrl+O",
        )
        open_card.clicked.connect(self.open_file_requested.emit)
        cards.addWidget(open_card)

        project_card = _ActionCard(
            "layers", "Open Project",
            "Restore a saved session: its recordings, reader settings, classifier "
            "mode and overrides.",
            ACCENT_SECONDARY, "Ctrl+Shift+O",
        )
        project_card.clicked.connect(self.open_project_requested.emit)
        cards.addWidget(project_card)

        self._train_card = _ActionCard(
            "cpu", "Train Classifier",
            "Train the learned modulation model on synthetic signals and your own "
            "labelled recordings.",
            ACCENT_SUCCESS,
        )
        self._train_card.clicked.connect(self.train_requested.emit)
        cards.addWidget(self._train_card)
        layout.addLayout(cards)

        # Recent + capabilities
        lower = QHBoxLayout()
        lower.setSpacing(14)

        self._recent_card = Card("Recent recordings", "clock")
        clear_btn = QPushButton("Clear")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setStyleSheet("padding: 3px 10px; font-size: 11px;")
        clear_btn.clicked.connect(self.clear_recent_requested.emit)
        self._recent_card.header.addWidget(clear_btn)
        self._recent_list = QListWidget()
        self._recent_list.setMinimumHeight(170)
        self._recent_list.setCursor(Qt.PointingHandCursor)
        self._recent_list.itemClicked.connect(self._on_recent_activated)
        self._recent_card.body.addWidget(self._recent_list)
        self._recent_empty = QLabel("Recordings you open will appear here.")
        self._recent_empty.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        self._recent_empty.setAlignment(Qt.AlignCenter)
        self._recent_card.body.addWidget(self._recent_empty)
        lower.addWidget(self._recent_card, stretch=3)

        caps = Card("Capabilities", "info")
        for key, value in (
            ("Formats", "WAV · Raw IQ · SigMF"),
            ("Classification", "18 modulation types"),
            ("Demodulation", "PSK · QAM · FSK · ASK · AM/FM/SSB"),
            ("FEC", "Viterbi · Reed-Solomon · LDPC"),
            ("Decoders", "RTTY · NAVTEX · NOAA APT"),
        ):
            caps.body.addWidget(KeyValueRow(key, value))
        caps.body.addStretch()
        lower.addWidget(caps, stretch=2)
        layout.addLayout(lower)

        ver = QLabel(f"Sigma v{APP_VERSION}")
        ver.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        layout.addWidget(ver)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(container, stretch=10)
        row.addStretch()
        outer.addLayout(row)
        outer.addStretch(2)
        self.set_recent([])

    # ------------------------------------------------------------------

    def set_train_enabled(self, enabled: bool) -> None:
        self._train_card.setEnabled(enabled)
        self._train_card.setToolTip("" if enabled else
                                    "Install the ML extras: pip install -e '.[ml]'")

    def set_recent(self, entries: list[tuple[str, RecordingMetadata]]) -> None:
        self._recent_list.clear()
        for path, meta in entries:
            p = Path(path)
            rate = (f"{meta.sample_rate_hz / 1e3:,.1f} kHz" if meta.sample_rate_hz > 0
                    else "rate from file")
            item = QListWidgetItem(icon("wave", ACCENT_PRIMARY, 16),
                                   f"{p.name}\n{meta.source_format.value}  ·  {rate}  ·  "
                                   f"{p.parent}")
            item.setData(Qt.UserRole, (path, meta))
            item.setToolTip(path)
            self._recent_list.addItem(item)
        has = bool(entries)
        self._recent_list.setVisible(has)
        self._recent_empty.setVisible(not has)

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data:
            self.recent_requested.emit(data[0], data[1])
