"""Constellation diagram viewer.

Displays I/Q scatter plot of demodulated symbols with optional
colour-coding by time or confidence.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.gui.theme import ACCENT_PRIMARY, BG_DARKEST, TEXT_PRIMARY, TEXT_SECONDARY


class ConstellationViewer(QWidget):
    """I/Q constellation scatter plot of demodulated symbols.

    For FSK the "symbols" are the sampled instantaneous frequency laid
    along the real axis, normalised so the outer tones sit at ±1.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._status_label = QLabel("Load a recording and demodulate to view constellation")
        self._status_label.setProperty("role", "subtitle")
        layout.addWidget(self._status_label)

        pg.setConfigOptions(background=BG_DARKEST, foreground=TEXT_PRIMARY, antialias=True)
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("bottom", "In-phase (I)")
        self._plot_widget.setLabel("left", "Quadrature (Q)")

        # Unit circle reference
        theta = np.linspace(0, 2 * np.pi, 200)
        self._plot_widget.plot(
            np.cos(theta), np.sin(theta),
            pen=pg.mkPen(TEXT_SECONDARY, width=1, style=pg.QtCore.Qt.DashLine),
        )

        self._scatter = pg.ScatterPlotItem(
            size=3, pen=pg.mkPen(None),
            brush=pg.mkBrush(ACCENT_PRIMARY),
        )
        self._plot_widget.addItem(self._scatter)

        layout.addWidget(self._plot_widget)

    def set_symbols(self, symbols: np.ndarray, label: str = "", max_points: int = 20_000) -> None:
        """Plot complex symbols on the constellation.

        Parameters
        ----------
        symbols : 1-D complex array of demodulated symbols.
        label : modulation / quality text shown above the plot.
        max_points : cap on plotted points for responsiveness.
        """
        s = np.asarray(symbols)
        if len(s) > max_points:
            s = s[np.linspace(0, len(s) - 1, max_points).astype(int)]
        self._scatter.setData(s.real, s.imag)
        self._plot_widget.setXRange(-1.6, 1.6)
        self._plot_widget.setYRange(-1.6, 1.6)
        text = f"{len(symbols):,} symbols"
        if label:
            text = f"{label}  ·  {text}"
        self._status_label.setText(text)

    def clear(self) -> None:
        self._scatter.setData([], [])
        self._status_label.setText("No symbols loaded")
