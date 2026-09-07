"""Time-domain waveform viewer.

Interactive plot showing real/imaginary channels, magnitude, and phase
with zoom, pan, region selection, and clipping markers.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.gui.theme import ACCENT_PRIMARY, ACCENT_SECONDARY, BG_DARKEST, TEXT_PRIMARY


class TimeViewer(QWidget):
    """Interactive time-domain waveform viewer using PyQtGraph."""

    region_selected = Signal(int, int)  # (start_sample, end_sample)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._samples: np.ndarray | None = None
        self._sample_rate: float = 1.0
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Controls bar
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Display:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "I & Q", "Magnitude", "Phase", "Real", "Imaginary",
        ])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_combo)
        controls.addStretch()

        self._cursor_label = QLabel("Sample: — | Time: —")
        self._cursor_label.setProperty("role", "subtitle")
        controls.addWidget(self._cursor_label)
        layout.addLayout(controls)

        # Plot widget
        pg.setConfigOptions(
            background=BG_DARKEST,
            foreground=TEXT_PRIMARY,
            antialias=True,
        )
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("bottom", "Time", units="s")
        self._plot_widget.setLabel("left", "Amplitude")

        # Region selection
        self._region = pg.LinearRegionItem(
            brush=pg.mkBrush(78, 168, 245, 30),
            pen=pg.mkPen(ACCENT_PRIMARY, width=1),
        )
        self._region.setVisible(False)
        self._plot_widget.addItem(self._region)
        self._region.sigRegionChangeFinished.connect(self._on_region_changed)

        # Crosshair
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen(TEXT_PRIMARY, width=1, style=Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
                                       pen=pg.mkPen(TEXT_PRIMARY, width=1, style=Qt.DashLine))
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._plot_widget.addItem(self._hline, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        layout.addWidget(self._plot_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, samples: np.ndarray, sample_rate: float) -> None:
        """Load new sample data into the viewer."""
        self._samples = samples
        self._sample_rate = sample_rate
        self._update_plot()

    def enable_region_selection(self, enable: bool = True) -> None:
        """Toggle the region selection overlay."""
        self._region.setVisible(enable)
        if enable and self._samples is not None:
            n = len(self._samples)
            self._region.setRegion([n * 0.25 / self._sample_rate,
                                     n * 0.75 / self._sample_rate])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_plot(self) -> None:
        if self._samples is None:
            return

        self._plot_widget.clear()
        self._plot_widget.addItem(self._region)
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._plot_widget.addItem(self._hline, ignoreBounds=True)

        n = len(self._samples)
        t = np.arange(n, dtype=np.float64) / self._sample_rate
        mode = self._mode_combo.currentText()

        if mode == "I & Q":
            self._plot_widget.plot(t, self._samples.real,
                                   pen=pg.mkPen(ACCENT_PRIMARY, width=1),
                                   name="I")
            self._plot_widget.plot(t, self._samples.imag,
                                   pen=pg.mkPen(ACCENT_SECONDARY, width=1),
                                   name="Q")
            self._plot_widget.setLabel("left", "Amplitude")
        elif mode == "Magnitude":
            self._plot_widget.plot(t, np.abs(self._samples),
                                   pen=pg.mkPen(ACCENT_PRIMARY, width=1))
            self._plot_widget.setLabel("left", "Magnitude")
        elif mode == "Phase":
            self._plot_widget.plot(t, np.angle(self._samples),
                                   pen=pg.mkPen(ACCENT_SECONDARY, width=1))
            self._plot_widget.setLabel("left", "Phase (rad)")
        elif mode == "Real":
            self._plot_widget.plot(t, self._samples.real,
                                   pen=pg.mkPen(ACCENT_PRIMARY, width=1))
            self._plot_widget.setLabel("left", "Real")
        elif mode == "Imaginary":
            self._plot_widget.plot(t, self._samples.imag,
                                   pen=pg.mkPen(ACCENT_SECONDARY, width=1))
            self._plot_widget.setLabel("left", "Imaginary")

    def _on_mode_changed(self, _index: int) -> None:
        self._update_plot()

    def _on_region_changed(self) -> None:
        lo, hi = self._region.getRegion()
        start = max(0, int(lo * self._sample_rate))
        end = int(hi * self._sample_rate)
        self.region_selected.emit(start, end)

    def _on_mouse_moved(self, evt: list) -> None:
        pos = evt[0]
        if self._plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            self._vline.setPos(x)
            self._hline.setPos(y)
            sample_idx = int(x * self._sample_rate) if self._sample_rate > 0 else 0
            self._cursor_label.setText(
                f"Sample: {sample_idx:,} | Time: {x:.6f} s | Amp: {y:.4f}"
            )
