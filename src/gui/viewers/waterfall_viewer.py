"""Waterfall / spectrogram viewer.

Displays a time-frequency heatmap with configurable dynamic range, colour
map, FFT resolution, and zoom.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.dsp.spectral import compute_spectrogram
from src.gui.theme import BG_DARKEST, TEXT_PRIMARY

# Colour map presets (viridis-like LUT)
_COLORMAPS = {
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "cividis": "cividis",
    "turbo": "turbo",
}


class WaterfallViewer(QWidget):
    """Interactive waterfall / spectrogram viewer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._samples: np.ndarray | None = None
        self._sample_rate: float = 1.0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("FFT:"))
        self._fft_spin = QSpinBox()
        self._fft_spin.setRange(128, 8192)
        self._fft_spin.setValue(1024)
        self._fft_spin.setSingleStep(256)
        self._fft_spin.valueChanged.connect(self._on_params_changed)
        controls.addWidget(self._fft_spin)

        controls.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(_COLORMAPS.keys()))
        self._cmap_combo.currentIndexChanged.connect(self._on_params_changed)
        controls.addWidget(self._cmap_combo)

        controls.addWidget(QLabel("Dynamic Range (dB):"))
        self._dr_slider = QSlider(Qt.Horizontal)
        self._dr_slider.setRange(20, 120)
        self._dr_slider.setValue(80)
        self._dr_slider.setFixedWidth(120)
        self._dr_slider.valueChanged.connect(self._on_params_changed)
        controls.addWidget(self._dr_slider)
        self._dr_label = QLabel("80")
        controls.addWidget(self._dr_label)

        controls.addStretch()
        self._info_label = QLabel("")
        self._info_label.setProperty("role", "subtitle")
        controls.addWidget(self._info_label)

        layout.addLayout(controls)

        # Image view
        pg.setConfigOptions(background=BG_DARKEST, foreground=TEXT_PRIMARY)
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("bottom", "Time", units="s")
        self._plot_widget.setLabel("left", "Frequency", units="Hz")

        self._img_item = pg.ImageItem()
        self._plot_widget.addItem(self._img_item)

        # Colour bar
        self._color_bar = pg.ColorBarItem(
            values=(-80, 0),
            colorMap=pg.colormap.get("viridis"),
            label="Power (dB)",
        )
        self._color_bar.setImageItem(self._img_item)

        layout.addWidget(self._plot_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, samples: np.ndarray, sample_rate: float) -> None:
        """Load new sample data."""
        self._samples = samples
        self._sample_rate = sample_rate
        self._update_plot()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_plot(self) -> None:
        if self._samples is None:
            return

        fft_size = self._fft_spin.value()
        dynamic_range = self._dr_slider.value()
        self._dr_label.setText(str(dynamic_range))

        times, freqs, sxx_db = compute_spectrogram(
            self._samples,
            self._sample_rate,
            fft_size=fft_size,
        )

        # Clamp to dynamic range
        vmax = float(np.max(sxx_db))
        vmin = vmax - dynamic_range
        display = np.clip(sxx_db, vmin, vmax)

        # Set image data — transposed so x=time, y=freq
        self._img_item.setImage(display.T, autoLevels=False)
        self._img_item.setLevels([vmin, vmax])

        # Scale to real units
        if len(times) > 1 and len(freqs) > 1:
            dt = float(times[1] - times[0])
            df = float(freqs[1] - freqs[0])
            t0 = float(times[0])
            f0 = float(freqs[0])
            from pyqtgraph import QtGui
            tr = QtGui.QTransform()
            tr.translate(t0, f0)
            tr.scale(dt, df)
            self._img_item.setTransform(tr)

        # Apply colormap
        cmap_name = self._cmap_combo.currentText()
        try:
            cm = pg.colormap.get(cmap_name)
        except Exception:
            cm = pg.colormap.get("viridis")
        self._img_item.setColorMap(cm)
        self._color_bar.setColorMap(cm)
        self._color_bar.setLevels(values=(vmin, vmax))

        self._info_label.setText(
            f"Duration: {times[-1]:.3f}s | BW: {abs(freqs[-1]-freqs[0]):,.0f} Hz"
        )

    def _on_params_changed(self, _: object = None) -> None:
        self._update_plot()
