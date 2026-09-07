"""Application dark theme and colour palettes.

Provides a premium dark-mode stylesheet for PySide6 widgets and a set of
scientifically-appropriate colour maps for signal visualisation.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

# Core palette — deep-space inspired dark theme
BG_DARKEST = "#0a0e17"
BG_DARK = "#0f1420"
BG_MID = "#151c2c"
BG_LIGHT = "#1c2538"
BG_HOVER = "#232e45"
BG_SELECTED = "#2a3a5c"

ACCENT_PRIMARY = "#4ea8f5"       # electric blue
ACCENT_SECONDARY = "#7c5cfc"     # violet
ACCENT_SUCCESS = "#34d399"       # emerald
ACCENT_WARNING = "#fbbf24"       # amber
ACCENT_DANGER = "#f87171"        # red-400

TEXT_PRIMARY = "#e2e8f0"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED = "#64748b"
TEXT_DISABLED = "#475569"

BORDER = "#1e293b"
BORDER_FOCUS = ACCENT_PRIMARY

# ---------------------------------------------------------------------------
# Plot colour sequences
# ---------------------------------------------------------------------------

PLOT_COLORS = [
    "#4ea8f5",  # electric blue
    "#f97316",  # orange
    "#34d399",  # emerald
    "#f87171",  # red
    "#a78bfa",  # purple
    "#fbbf24",  # amber
    "#38bdf8",  # sky
    "#fb7185",  # rose
]

# High-contrast colours for accessibility
HIGH_CONTRAST_COLORS = [
    "#ffffff",
    "#ffff00",
    "#00ffff",
    "#ff00ff",
    "#ff8800",
    "#00ff00",
]


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

DARK_STYLESHEET = f"""
/* ---- Global ---- */
QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: "Inter", "Segoe UI", "Roboto", sans-serif;
    font-size: 13px;
    selection-background-color: {BG_SELECTED};
    selection-color: {TEXT_PRIMARY};
}}

QMainWindow {{
    background-color: {BG_DARKEST};
}}

/* ---- Menu bar ---- */
QMenuBar {{
    background-color: {BG_DARKEST};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER};
    padding: 2px 4px;
}}
QMenuBar::item:selected {{
    background-color: {BG_HOVER};
    border-radius: 4px;
}}
QMenu {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item:selected {{
    background-color: {ACCENT_PRIMARY};
    color: {BG_DARKEST};
    border-radius: 4px;
}}

/* ---- Tool bar ---- */
QToolBar {{
    background-color: {BG_DARKEST};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 4px;
}}
QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 12px;
    color: {TEXT_SECONDARY};
}}
QToolButton:hover {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}
QToolButton:pressed {{
    background-color: {BG_SELECTED};
}}

/* ---- Tab widget ---- */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {BG_DARK};
}}
QTabBar::tab {{
    background-color: {BG_MID};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 18px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {BG_DARK};
    color: {ACCENT_PRIMARY};
    border-bottom: 2px solid {ACCENT_PRIMARY};
}}
QTabBar::tab:hover:!selected {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}

/* ---- Dock widget ---- */
QDockWidget {{
    titlebar-close-icon: none;
    color: {TEXT_PRIMARY};
}}
QDockWidget::title {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    padding: 6px;
    text-align: left;
}}

/* ---- Scroll bars ---- */
QScrollBar:vertical {{
    background: {BG_DARKEST};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {BG_LIGHT};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}
QScrollBar:horizontal {{
    background: {BG_DARKEST};
    height: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: {BG_LIGHT};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0;
}}

/* ---- Inputs ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT_PRIMARY};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_PRIMARY};
    selection-color: {BG_DARKEST};
}}

/* ---- Push buttons ---- */
QPushButton {{
    background-color: {ACCENT_PRIMARY};
    color: {BG_DARKEST};
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: #5bb5ff;
}}
QPushButton:pressed {{
    background-color: #3a8fd4;
}}
QPushButton:disabled {{
    background-color: {BG_LIGHT};
    color: {TEXT_DISABLED};
}}
QPushButton[flat="true"] {{
    background-color: transparent;
    color: {ACCENT_PRIMARY};
    border: 1px solid {ACCENT_PRIMARY};
}}

/* ---- Group box ---- */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 18px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
}}

/* ---- Labels ---- */
QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QLabel[role="heading"] {{
    font-size: 18px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
}}
QLabel[role="subtitle"] {{
    font-size: 14px;
    color: {TEXT_SECONDARY};
}}

/* ---- Progress bar ---- */
QProgressBar {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    color: {TEXT_PRIMARY};
    height: 20px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_PRIMARY}, stop:1 {ACCENT_SECONDARY});
    border-radius: 5px;
}}

/* ---- Status bar ---- */
QStatusBar {{
    background-color: {BG_DARKEST};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}

/* ---- Splitter ---- */
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}

/* ---- Tree / List views ---- */
QTreeView, QListView, QTableView {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 6px;
    alternate-background-color: {BG_DARK};
}}
QTreeView::item:hover, QListView::item:hover {{
    background-color: {BG_HOVER};
}}
QTreeView::item:selected, QListView::item:selected {{
    background-color: {BG_SELECTED};
    color: {TEXT_PRIMARY};
}}
QHeaderView::section {{
    background-color: {BG_MID};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    padding: 6px;
}}

/* ---- Text edit (console) ---- */
QTextEdit, QPlainTextEdit {{
    background-color: {BG_DARKEST};
    color: {ACCENT_SUCCESS};
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 6px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the dark theme to the entire application."""
    app.setStyleSheet(DARK_STYLESHEET)

    # Set application font
    font = QFont("Inter", 13)
    app.setFont(font)
