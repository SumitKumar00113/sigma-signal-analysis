"""Analysis results panel.

Shows the outcome of an :class:`AnalysisPipeline` run — modulation
verdict with ranked alternatives and evidence, estimated parameters,
demodulation quality — and offers analyst overrides that trigger a
re-run.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.enums import ConfidenceLevel, ModulationType
from src.dsp.pipeline import PipelineResult
from src.gui.recording_overview import _format_hz, _InfoRow
from src.gui.theme import (
    ACCENT_DANGER,
    ACCENT_SUCCESS,
    ACCENT_WARNING,
    TEXT_MUTED,
    TEXT_SECONDARY,
)

SUPPORTED_OVERRIDES: list[ModulationType] = [
    ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8,
    ModulationType.QAM16, ModulationType.QAM64,
    ModulationType.FSK2, ModulationType.FSK4,
]

_CONF_COLOR = {
    ConfidenceLevel.CONFIRMED: ACCENT_SUCCESS,
    ConfidenceLevel.PROBABLE: ACCENT_SUCCESS,
    ConfidenceLevel.POSSIBLE: ACCENT_WARNING,
    ConfidenceLevel.UNSUPPORTED: ACCENT_DANGER,
    ConfidenceLevel.UNKNOWN: TEXT_MUTED,
}


class ResultsPanel(QWidget):
    """Dock content summarising a pipeline result."""

    rerun_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, _InfoRow] = {}
        self._setup_ui()

    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(8)

        title = QLabel("Analysis Results")
        title.setProperty("role", "heading")
        layout.addWidget(title)

        # --- Verdict ---
        verdict_group = QGroupBox("Modulation")
        vg = QVBoxLayout(verdict_group)
        self._mod_label = QLabel("—")
        self._mod_label.setStyleSheet("font-size: 22px; font-weight: 700; background: transparent;")
        vg.addWidget(self._mod_label)
        self._overall_label = QLabel("No analysis run")
        self._overall_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        vg.addWidget(self._overall_label)
        self._candidates_label = QLabel("")
        self._candidates_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;"
        )
        self._candidates_label.setWordWrap(True)
        vg.addWidget(self._candidates_label)
        self._evidence_label = QLabel("")
        self._evidence_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; background: transparent;"
        )
        self._evidence_label.setWordWrap(True)
        self._evidence_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vg.addWidget(self._evidence_label)
        layout.addWidget(verdict_group)

        # --- Parameters ---
        param_group = QGroupBox("Estimated Parameters")
        pg_layout = QVBoxLayout(param_group)
        for key in ("Symbol Rate", "SNR (full band)", "SNR (in-band)", "Carrier Offset",
                    "Occupied BW", "Samples / Symbol", "Regions"):
            row = _InfoRow(key)
            self._rows[key] = row
            pg_layout.addWidget(row)
        layout.addWidget(param_group)

        # --- Demodulation ---
        demod_group = QGroupBox("Demodulation")
        dg = QVBoxLayout(demod_group)
        for key in ("Symbols", "Bits", "EVM", "Residual CFO", "FSK Tones", "Audio",
                    "Processing Time"):
            row = _InfoRow(key)
            self._rows[key] = row
            dg.addWidget(row)
        layout.addWidget(demod_group)

        # --- Overrides ---
        ov_group = QGroupBox("Analyst Overrides")
        form = QFormLayout(ov_group)
        form.setSpacing(6)
        self._mod_override = QComboBox()
        self._mod_override.addItem("Auto-detect", None)
        for m in SUPPORTED_OVERRIDES:
            self._mod_override.addItem(m.value, m)
        form.addRow("Modulation:", self._mod_override)
        self._rs_override = QDoubleSpinBox()
        self._rs_override.setRange(0, 1e9)
        self._rs_override.setDecimals(1)
        self._rs_override.setSuffix(" baud")
        self._rs_override.setSpecialValueText("Auto-detect")
        form.addRow("Symbol Rate:", self._rs_override)
        btn_row = QHBoxLayout()
        self._rerun_btn = QPushButton("↻ Re-run Analysis")
        self._rerun_btn.clicked.connect(self.rerun_requested.emit)
        btn_row.addWidget(self._rerun_btn)
        form.addRow(btn_row)
        layout.addWidget(ov_group)

        # --- Warnings ---
        self._warn_group = QGroupBox("Warnings")
        wg = QVBoxLayout(self._warn_group)
        self._warn_label = QLabel("")
        self._warn_label.setWordWrap(True)
        self._warn_label.setStyleSheet(
            f"color: {ACCENT_WARNING}; font-size: 12px; background: transparent;"
        )
        self._warn_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        wg.addWidget(self._warn_label)
        self._warn_group.setVisible(False)
        layout.addWidget(self._warn_group)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def modulation_override(self) -> ModulationType | None:
        # Qt hands StrEnum item data back as a plain str
        v = self._mod_override.currentData()
        return ModulationType(v) if v else None

    def symbol_rate_override(self) -> float | None:
        v = self._rs_override.value()
        return v if v > 0 else None

    def set_busy(self, busy: bool) -> None:
        self._rerun_btn.setEnabled(not busy)

    def clear(self) -> None:
        self._mod_label.setText("—")
        self._overall_label.setText("No analysis run")
        self._candidates_label.setText("")
        self._evidence_label.setText("")
        for row in self._rows.values():
            row.set_value("—")
        self._warn_group.setVisible(False)

    def update_result(self, res: PipelineResult) -> None:
        a = res.analysis
        color = _CONF_COLOR.get(a.overall_confidence, TEXT_MUTED)

        self._mod_label.setText(a.modulation.value)
        self._mod_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; background: transparent; color: {color};"
        )
        self._overall_label.setText(
            f"{a.overall_confidence.value}  ·  classifier confidence {a.modulation_confidence:.0%}"
        )

        if a.modulation_candidates:
            parts = [f"{c['modulation']} {c['probability']:.0%}"
                     for c in a.modulation_candidates[:4]]
            self._candidates_label.setText("Candidates: " + "  ·  ".join(parts))
        else:
            self._candidates_label.setText("")

        if res.classification and res.classification.evidence:
            self._evidence_label.setText(" ".join(res.classification.evidence))
        else:
            self._evidence_label.setText("")

        rs = a.symbol_rate_hz
        self._rows["Symbol Rate"].set_value(
            f"{rs:,.1f} baud  ({a.symbol_rate_confidence:.0%})" if rs > 0 else "—"
        )
        self._rows["SNR (full band)"].set_value(
            f"{res.snr_db:.1f} dB",
            ACCENT_WARNING if res.snr_db < 6 else None,
        )
        self._rows["SNR (in-band)"].set_value(f"{res.snr_inband_db:.1f} dB")
        self._rows["Carrier Offset"].set_value(f"{res.frequency_offset_hz:+,.1f} Hz")
        self._rows["Occupied BW"].set_value(_format_hz(res.occupied_bandwidth_hz))
        fs = res.metadata.sample_rate_hz
        self._rows["Samples / Symbol"].set_value(f"{fs / rs:.2f}" if rs > 0 and fs > 0 else "—")
        self._rows["Regions"].set_value(str(len(res.regions)))

        d = res.demod
        if d is not None:
            self._rows["Symbols"].set_value(f"{d.num_symbols:,}")
            self._rows["Bits"].set_value(f"{d.num_bits:,}  ({d.bits_per_symbol} / symbol)")
            evm_color = ACCENT_SUCCESS if d.evm_percent < 20 else (
                ACCENT_WARNING if d.evm_percent < 35 else ACCENT_DANGER)
            self._rows["EVM"].set_value(f"{d.evm_percent:.1f} %", evm_color)
            self._rows["Residual CFO"].set_value(f"{d.residual_cfo_hz:+.1f} Hz")
            self._rows["FSK Tones"].set_value(
                ", ".join(f"{t:+,.0f}" for t in d.fsk_levels_hz) + " Hz"
                if d.fsk_levels_hz else "—"
            )
            if d.audio is not None and d.audio_rate_hz > 0:
                for k in ("Symbols", "Bits", "EVM", "Residual CFO"):
                    self._rows[k].set_value("—")
                self._rows["Audio"].set_value(
                    f"{len(d.audio) / d.audio_rate_hz:.1f} s @ {d.audio_rate_hz:,.0f} Hz "
                    "(File → Save Demodulated Audio…)", ACCENT_SUCCESS)
            else:
                self._rows["Audio"].set_value("—")
        else:
            for k in ("Symbols", "Bits", "EVM", "Residual CFO", "FSK Tones", "Audio"):
                self._rows[k].set_value("—")
        self._rows["Processing Time"].set_value(f"{res.processing_time_ms:,.0f} ms")

        warnings = list(a.warnings)
        for stage, err in res.stage_errors.items():
            warnings.append(f"[{stage}] {err}")
        if warnings:
            self._warn_label.setText("\n".join(f"• {w}" for w in warnings))
            self._warn_group.setVisible(True)
        else:
            self._warn_group.setVisible(False)
