"""Spectrum / FFT viewer.

Displays PSD with configurable FFT size, windowing, averaging, peak
markers, and noise-floor estimation.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.dsp.spectral import compute_psd, detect_peaks, estimate_noise_floor
from src.gui.theme import (
    ACCENT_DANGER,
    ACCENT_PRIMARY,
    ACCENT_WARNING,
    BG_DARKEST,
    TEXT_PRIMARY,
)


class SpectrumViewer(QWidget):
    """Interactive spectrum / PSD viewer."""

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

        controls.addWidget(QLabel("FFT Size:"))
        self._fft_spin = QSpinBox()
        self._fft_spin.setRange(128, 65536)
        self._fft_spin.setValue(4096)
        self._fft_spin.setSingleStep(1024)
        self._fft_spin.valueChanged.connect(self._on_params_changed)
        controls.addWidget(self._fft_spin)

        controls.addWidget(QLabel("Window:"))
        self._window_combo = QComboBox()
        self._window_combo.addItems([
            "hann", "hamming", "blackman", "blackmanharris",
            "flattop", "kaiser", "rectangular",
        ])
        self._window_combo.currentIndexChanged.connect(self._on_params_changed)
        controls.addWidget(self._window_combo)

        controls.addStretch()
        self._info_label = QLabel("Freq: — | Power: —")
        self._info_label.setProperty("role", "subtitle")
        controls.addWidget(self._info_label)

        layout.addLayout(controls)

        # Plot
        pg.setConfigOptions(background=BG_DARKEST, foreground=TEXT_PRIMARY, antialias=True)
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self._plot_widget.setLabel("left", "Power", units="dB/Hz")

        # Noise floor line
        self._noise_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(ACCENT_WARNING, width=1, style=Qt.DashLine),
            label="Noise Floor",
            labelOpts={"color": ACCENT_WARNING, "position": 0.05},
        )
        self._plot_widget.addItem(self._noise_line)

        # Crosshair
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen(TEXT_PRIMARY, width=1, style=Qt.DashLine))
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
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
        window = self._window_combo.currentText()
        if window == "rectangular":
            window = "boxcar"

        freqs, psd_db = compute_psd(
            self._samples, self._sample_rate,
            fft_size=fft_size, window=window,
        )

        self._plot_widget.clear()

        # PSD curve
        self._plot_widget.plot(
            freqs, psd_db,
            pen=pg.mkPen(ACCENT_PRIMARY, width=1.5),
        )

        # Noise floor
        noise = estimate_noise_floor(psd_db)
        self._noise_line = pg.InfiniteLine(
            pos=noise, angle=0, movable=False,
            pen=pg.mkPen(ACCENT_WARNING, width=1, style=Qt.DashLine),
        )
        self._plot_widget.addItem(self._noise_line)

        # Peak markers
        peaks = detect_peaks(freqs, psd_db, noise, threshold_db=6.0)
        for pk in peaks[:20]:  # limit markers
            marker = pg.ScatterPlotItem(
                [pk.frequency_hz], [pk.power_db],
                symbol="t", size=12,
                pen=pg.mkPen(ACCENT_DANGER),
                brush=pg.mkBrush(ACCENT_DANGER),
            )
            self._plot_widget.addItem(marker)

        # Re-add crosshair
        self._vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(TEXT_PRIMARY, width=1, style=Qt.DashLine),
        )
        self._plot_widget.addItem(self._vline, ignoreBounds=True)

    def _on_params_changed(self, _: object = None) -> None:
        self._update_plot()

    def _on_mouse_moved(self, evt: list) -> None:
        pos = evt[0]
        if self._plot_widget.sceneBoundingRect().contains(pos):
            pt = self._plot_widget.plotItem.vb.mapSceneToView(pos)
            self._vline.setPos(pt.x())
            self._info_label.setText(
                f"Freq: {pt.x():,.0f} Hz | Power: {pt.y():.1f} dB"
            )
