"""Welcome / landing screen.

Provides quick access to new project creation, opening recent recordings,
and system capability summary.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.gui.theme import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    BG_LIGHT,
    BG_MID,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class _ActionCard(QFrame):
    """A clickable card on the welcome screen."""

    clicked = Signal()

    def __init__(
        self,
        icon_text: str,
        title: str,
        subtitle: str,
        accent: str = ACCENT_PRIMARY,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            _ActionCard {{
                background-color: {BG_MID};
                border: 1px solid {BG_LIGHT};
                border-radius: 12px;
                padding: 24px;
            }}
            _ActionCard:hover {{
                border-color: {accent};
                background-color: {BG_LIGHT};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        icon_label = QLabel(icon_text)
        icon_label.setStyleSheet(f"font-size: 32px; color: {accent}; background: transparent;")
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {TEXT_PRIMARY}; background: transparent;"
        )
        layout.addWidget(title_label)

        sub_label = QLabel(subtitle)
        sub_label.setStyleSheet(
            f"font-size: 12px; color: {TEXT_SECONDARY};"
            " background: transparent;"
        )
        sub_label.setWordWrap(True)
        layout.addWidget(sub_label)

    def mousePressEvent(self, event):  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)


class WelcomeScreen(QWidget):
    """Landing screen shown when no recording is loaded."""

    open_file_requested = Signal()
    new_project_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setMaximumWidth(720)
        layout = QVBoxLayout(container)
        layout.setSpacing(16)

        # Title
        title = QLabel("⚡ Sigma Signal Analysis")
        title.setStyleSheet(
            f"font-size: 28px; font-weight: bold; color: {TEXT_PRIMARY}; background: transparent;"
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Automated RF Signal Analysis & Demodulation Platform")
        subtitle.setStyleSheet(
            f"font-size: 14px; color: {TEXT_SECONDARY}; background: transparent;"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        # Action cards row
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        open_card = _ActionCard(
            "📂", "Open Recording",
            "Load a .wav, raw IQ, or SigMF file for analysis",
            ACCENT_PRIMARY,
        )
        open_card.clicked.connect(self.open_file_requested.emit)
        cards_layout.addWidget(open_card)

        new_card = _ActionCard(
            "📁", "New Project",
            "Create a new analysis project with multiple recordings",
            ACCENT_SECONDARY,
        )
        new_card.clicked.connect(self.new_project_requested.emit)
        cards_layout.addWidget(new_card)

        layout.addLayout(cards_layout)

        layout.addSpacing(16)

        # Supported formats
        formats_label = QLabel(
            "Supported formats: WAV (PCM 8/16/24/32, float32) · "
            "Raw IQ (cf32, ci16, cu8, …) · SigMF (.sigmf-meta/.sigmf-data)"
        )
        formats_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED}; background: transparent;"
        )
        formats_label.setAlignment(Qt.AlignCenter)
        formats_label.setWordWrap(True)
        layout.addWidget(formats_label)

        # Version
        ver = QLabel("v0.1.0 — Phase 1 MVP")
        ver.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED}; background: transparent;")
        ver.setAlignment(Qt.AlignCenter)
        layout.addWidget(ver)

        outer.addWidget(container)
