"""Application dark theme and colour palettes.

A "mission control" dark theme: near-black space backdrop, translucent
glass cards with hairline borders, and a single electric-blue accent for
the active state and primary actions.  Signal plots use the blue / cyan
pair so the two I/Q channels stay distinguishable.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from src import __version__

APP_NAME = "Sigma Signal Analysis"
APP_VERSION = __version__
# 1024 px master of the app icon (the macOS bundle uses packaging/macos/Sigma.icns)
APP_ICON = Path(__file__).resolve().parent / "assets" / "app_icon.png"

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

BG_DARKEST = "#05070c"          # window backdrop
BG_DARK = "#0b0f17"             # dialogs, inputs' surroundings
BG_MID = "#121824"              # inputs, rows
BG_LIGHT = "#1a2231"            # raised rows, disabled buttons
BG_HOVER = "#222c3d"
BG_SELECTED = "#1e3450"         # blue-tinted selection

CARD_BG = "rgba(13, 18, 27, 0.82)"          # glass card
CARD_BORDER = "rgba(255, 255, 255, 0.08)"
ROW_BG = "rgba(255, 255, 255, 0.035)"       # key/value row inside a card
ROW_BORDER = "rgba(255, 255, 255, 0.06)"
PLOT_BG = "#080b12"

ACCENT_PRIMARY = "#4ea8f5"       # signal blue — active state, primary action
ACCENT_PRIMARY_HOVER = "#5bb5ff"
ACCENT_PRIMARY_PRESSED = "#3a8fd4"
ACCENT_SECONDARY = "#4cc9f0"     # cyan — second data series
ACCENT_SUCCESS = "#3ddc97"
ACCENT_WARNING = "#f5b642"
ACCENT_DANGER = "#ff5a5f"

TEXT_PRIMARY = "#e8ebf2"
TEXT_SECONDARY = "#9aa3b5"
TEXT_MUTED = "#6b7385"
TEXT_DISABLED = "#474e5e"

BORDER = "#1e2533"
BORDER_FOCUS = ACCENT_PRIMARY

# ---------------------------------------------------------------------------
# Plot colour sequences
# ---------------------------------------------------------------------------

PLOT_COLORS = [
    "#4ea8f5",  # blue
    "#4cc9f0",  # cyan
    "#f5b642",  # amber
    "#3ddc97",  # emerald
    "#b794f6",  # violet
    "#ff5a5f",  # red
    "#8fd3ff",  # sky
    "#ffa38a",  # salmon
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

# Replaced by the platform's own UI / monospace families in apply_theme()
_FONT_STACK = "@UI_FONT@"
_MONO_STACK = "@MONO_FONT@"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

DARK_STYLESHEET = f"""
/* ---- Global ---- */
QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-family: {_FONT_STACK};
    font-size: 13px;
    selection-background-color: {BG_SELECTED};
    selection-color: {TEXT_PRIMARY};
}}
QMainWindow, QDialog, QMessageBox, QWizard {{
    background-color: {BG_DARK};
}}
QToolTip {{
    background-color: {BG_MID};
    color: {TEXT_PRIMARY};
    border: 1px solid {CARD_BORDER};
    border-radius: 6px;
    padding: 6px 8px;
}}

/* ---- Cards (glass panels) ---- */
QFrame#card {{
    background-color: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
}}
QLabel#cardTitle {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
QFrame#cardRule {{
    background-color: {CARD_BORDER};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}
QFrame#kvRow {{
    background-color: {ROW_BG};
    border: 1px solid {ROW_BORDER};
    border-radius: 7px;
}}
QLabel#kvKey {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
QLabel#kvValue {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
    font-weight: 600;
}}

/* ---- Header ---- */
QLabel#brand {{
    font-size: 19px;
    font-weight: 800;
    letter-spacing: 3px;
    color: {TEXT_PRIMARY};
}}
QLabel#heroTitle {{
    font-size: 24px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
QLabel#heroSubtitle {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}
QLabel#chip {{
    background-color: rgba(18, 24, 36, 0.95);
    border: 1px solid {ROW_BORDER};
    border-radius: 10px;
    padding: 3px 10px;
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}
QFrame#segmented {{
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid {CARD_BORDER};
    border-radius: 9px;
}}
QPushButton#segment {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 6px 16px;
    font-weight: 500;
}}
QPushButton#segment:hover {{
    background-color: rgba(255, 255, 255, 0.06);
    color: {TEXT_PRIMARY};
}}
QPushButton#segment:checked {{
    background-color: {ACCENT_PRIMARY};
    color: #ffffff;
    font-weight: 600;
}}
QToolButton#iconButton {{
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid {CARD_BORDER};
    border-radius: 9px;
    padding: 7px;
}}
QToolButton#iconButton:hover {{
    background-color: rgba(255, 255, 255, 0.09);
    border-color: rgba(255, 255, 255, 0.16);
}}
QToolButton#iconButton:pressed {{
    background-color: {BG_SELECTED};
}}
QToolButton#iconButton:disabled {{
    background-color: transparent;
}}

/* ---- Menu bar ---- */
QMenuBar {{
    background-color: {BG_DARKEST};
    color: {TEXT_SECONDARY};
    border-bottom: 1px solid {BORDER};
    padding: 2px 6px;
}}
QMenuBar::item {{
    padding: 4px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
    border-radius: 5px;
}}
QMenu {{
    background-color: {BG_MID};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{
    padding: 6px 22px 6px 14px;
    border-radius: 5px;
    background: transparent;
}}
QMenu::item:selected {{
    background-color: {ACCENT_PRIMARY};
    color: #ffffff;
}}
QMenu::item:disabled {{
    color: {TEXT_DISABLED};
}}
QMenu::separator {{
    height: 1px;
    background: {CARD_BORDER};
    margin: 5px 8px;
}}

/* ---- Tab widget (inspector tabs inside cards) ---- */
QTabWidget::pane {{
    border: none;
    background: transparent;
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 14px;
    margin-right: 4px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT_PRIMARY};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
}}

/* ---- Scroll areas & bars ---- */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 0.12);
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: rgba(255, 255, 255, 0.24);
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: rgba(255, 255, 255, 0.12);
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    height: 0; width: 0; background: none;
}}

/* ---- Inputs ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 5px 9px;
    color: {TEXT_PRIMARY};
    min-height: 18px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_PRIMARY};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {TEXT_DISABLED};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_MID};
    border: 1px solid {CARD_BORDER};
    border-radius: 6px;
    selection-background-color: {ACCENT_PRIMARY};
    selection-color: #ffffff;
    outline: none;
}}
QCheckBox, QRadioButton {{
    spacing: 8px;
    background: transparent;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_LIGHT};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_PRIMARY};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {TEXT_PRIMARY};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}

/* ---- Push buttons ---- */
QPushButton {{
    background-color: rgba(255, 255, 255, 0.05);
    color: {TEXT_PRIMARY};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    padding: 7px 16px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: rgba(255, 255, 255, 0.10);
    border-color: rgba(255, 255, 255, 0.18);
}}
QPushButton:pressed {{
    background-color: {BG_SELECTED};
}}
QPushButton:disabled {{
    background-color: transparent;
    color: {TEXT_DISABLED};
    border-color: {BORDER};
}}
QPushButton:default, QPushButton[primary="true"] {{
    background-color: {ACCENT_PRIMARY};
    color: #ffffff;
    border: 1px solid {ACCENT_PRIMARY};
    font-weight: 600;
}}
QPushButton:default:hover, QPushButton[primary="true"]:hover {{
    background-color: {ACCENT_PRIMARY_HOVER};
    border-color: {ACCENT_PRIMARY_HOVER};
}}
QPushButton:default:pressed, QPushButton[primary="true"]:pressed {{
    background-color: {ACCENT_PRIMARY_PRESSED};
}}
QPushButton[primary="true"]:disabled {{
    background-color: rgba(78, 168, 245, 0.18);
    border-color: rgba(78, 168, 245, 0.10);
    color: rgba(255, 255, 255, 0.35);
}}
QPushButton[flat="true"] {{
    background-color: transparent;
    color: {ACCENT_PRIMARY};
    border: 1px solid {ACCENT_PRIMARY};
}}

/* ---- Group box ---- */
QGroupBox {{
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding: 14px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
}}

/* ---- Labels ---- */
QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QLabel[role="heading"] {{
    font-size: 17px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
QLabel[role="subtitle"] {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}

/* ---- Progress bar ---- */
QProgressBar {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-size: 11px;
    min-height: 14px;
    max-height: 16px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_PRIMARY};
    border-radius: 5px;
}}

/* ---- Status bar ---- */
QStatusBar {{
    background-color: {BG_DARKEST};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}
QStatusBar QLabel {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    padding: 0 6px;
}}
QStatusBar::item {{
    border: none;
}}

/* ---- Splitter ---- */
QSplitter::handle {{
    background-color: transparent;
}}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}
QSplitter::handle:hover {{
    background-color: rgba(78, 168, 245, 0.25);
    border-radius: 3px;
}}

/* ---- Tree / List / Table views ---- */
QTreeView, QListView, QTableView {{
    background-color: transparent;
    border: none;
    alternate-background-color: rgba(255, 255, 255, 0.02);
    outline: none;
}}
QTableView, QListView#bordered {{
    background-color: rgba(0, 0, 0, 0.18);
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
}}
QTreeView::item, QListView::item {{
    padding: 5px 4px;
    border-radius: 6px;
}}
QTreeView::item:hover, QListView::item:hover {{
    background-color: rgba(255, 255, 255, 0.05);
}}
QTreeView::item:selected, QListView::item:selected {{
    background-color: {BG_SELECTED};
    color: {TEXT_PRIMARY};
}}
QHeaderView {{
    background: transparent;
}}
QHeaderView::section {{
    background-color: transparent;
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {CARD_BORDER};
    padding: 6px;
    font-size: 11px;
    font-weight: 600;
}}
QTableCornerButton::section {{
    background: transparent;
    border: none;
}}

/* ---- Text edit (console, bit views) ---- */
QTextEdit, QPlainTextEdit {{
    background-color: rgba(0, 0, 0, 0.25);
    color: {TEXT_PRIMARY};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    font-family: {_MONO_STACK};
    font-size: 12px;
    padding: 6px;
}}
QPlainTextEdit#console {{
    background-color: transparent;
    border: none;
    color: {TEXT_SECONDARY};
    padding: 0;
}}
"""


def plot_background() -> str:
    """Background colour for pyqtgraph plots inside cards."""
    return PLOT_BG


def apply_theme(app: QApplication) -> None:
    """Apply the dark theme to the entire application."""
    # Palette fallback for widgets that paint their own background
    pal = app.palette()
    for role, colour in (
        (QPalette.Window, BG_DARK),
        (QPalette.Base, BG_MID),
        (QPalette.AlternateBase, BG_DARK),
        (QPalette.Button, BG_MID),
        (QPalette.Text, TEXT_PRIMARY),
        (QPalette.WindowText, TEXT_PRIMARY),
        (QPalette.ButtonText, TEXT_PRIMARY),
        (QPalette.Highlight, ACCENT_PRIMARY),
        (QPalette.HighlightedText, "#ffffff"),
        (QPalette.ToolTipBase, BG_MID),
        (QPalette.ToolTipText, TEXT_PRIMARY),
        (QPalette.PlaceholderText, TEXT_MUTED),
    ):
        pal.setColor(role, QColor(colour))
    app.setPalette(pal)

    ui = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    app.setStyleSheet(DARK_STYLESHEET
                      .replace(_FONT_STACK, f'"{ui.family()}"')
                      .replace(_MONO_STACK, f'"{mono.family()}", monospace'))
    font = QFont(ui)
    font.setPointSize(13)
    app.setFont(font)
