"""Recording overview panel.

Displays summary statistics for a loaded recording: duration, sample rate,
center frequency, format, channels, noise floor, clipping, DC offset,
file hash, and metadata completeness (PRD §8.2C).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.models import FileValidationReport, RecordingMetadata
from src.gui.theme import (
    ACCENT_DANGER,
    ACCENT_SUCCESS,
    ACCENT_WARNING,
    TEXT_MUTED,
)
from src.gui.widgets import KeyValueRow, MeterRow

# Kept under its historical name: other panels build their rows with it
_InfoRow = KeyValueRow


def _format_hz(hz: float) -> str:
    if hz >= 1e9:
        return f"{hz / 1e9:.3f} GHz"
    if hz >= 1e6:
        return f"{hz / 1e6:.3f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"


def _format_duration(s: float) -> str:
    if s >= 3600:
        return f"{s / 3600:.2f} h"
    if s >= 60:
        return f"{s / 60:.2f} min"
    return f"{s:.3f} s"


def _format_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.0f} KB"
    return f"{b} B"


def section_label(text: str) -> QLabel:
    """Small upper-case caption separating groups of rows."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
                      "letter-spacing: 1px; padding-top: 6px;")
    return lbl


class RecordingOverview(QWidget):
    """Summary panel for a loaded recording."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, KeyValueRow] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 4, 6, 4)
        layout.setSpacing(5)

        sections = (
            ("File", ("File", "Format", "Size", "SHA-256")),
            ("Signal", ("Sample Rate", "Center Frequency", "Duration",
                        "Samples", "Channels", "Data Type", "IQ Order")),
            ("Quality", ("DC Offset (I)", "DC Offset (Q)", "IQ Imbalance",
                         "Clipping", "NaN Count", "Inf Count")),
        )
        for title, keys in sections:
            layout.addWidget(section_label(title))
            for key in keys:
                row = KeyValueRow(key)
                self._rows[key] = row
                layout.addWidget(row)

        layout.addWidget(section_label("Metadata"))
        self._completeness = MeterRow("Completeness", ACCENT_SUCCESS)
        layout.addWidget(self._completeness)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        for row in self._rows.values():
            row.set_value("—")
        self._completeness.reset()

    def update_metadata(self, meta: RecordingMetadata) -> None:
        """Populate rows from recording metadata."""
        self._rows["File"].set_value(meta.source_path.split("/")[-1] if meta.source_path else "—")
        self._rows["Format"].set_value(meta.source_format.value)
        self._rows["Sample Rate"].set_value(_format_hz(meta.sample_rate_hz))
        self._rows["Center Frequency"].set_value(
            _format_hz(meta.center_frequency_hz) if meta.center_frequency_hz > 0 else "Unknown"
        )
        self._rows["Duration"].set_value(_format_duration(meta.duration_seconds))
        self._rows["Samples"].set_value(f"{meta.sample_count:,}")
        self._rows["Channels"].set_value(str(meta.num_channels))
        self._rows["Data Type"].set_value(meta.sample_datatype.value)
        self._rows["IQ Order"].set_value(meta.iq_order.value)

        # Completeness
        filled = sum(1 for v in [
            meta.sample_rate_hz > 0,
            meta.center_frequency_hz > 0,
            meta.sample_count > 0,
            meta.source_path,
            meta.num_channels > 0,
        ] if v)
        pct = filled / 5
        self._completeness.set(None, f"{pct:.0%}", pct,
                               ACCENT_SUCCESS if pct >= 0.8 else ACCENT_WARNING)

    def update_validation(self, report: FileValidationReport) -> None:
        """Populate quality rows from validation report."""
        self._rows["Size"].set_value(_format_bytes(report.file_size_bytes))
        self._rows["SHA-256"].set_value(report.sha256[:16] + "…" if report.sha256 else "—")
        if report.sha256:
            self._rows["SHA-256"].setToolTip(report.sha256)

        self._rows["DC Offset (I)"].set_value(f"{report.dc_offset_i:.6f}")
        self._rows["DC Offset (Q)"].set_value(f"{report.dc_offset_q:.6f}")
        self._rows["IQ Imbalance"].set_value(
            f"{report.iq_imbalance_db:.2f} dB",
            ACCENT_WARNING if abs(report.iq_imbalance_db) > 1.0 else None,
        )

        clip_color = ACCENT_DANGER if report.clipping_percentage > 1.0 else None
        self._rows["Clipping"].set_value(f"{report.clipping_percentage:.2f}%", clip_color)
        self._rows["NaN Count"].set_value(
            str(report.nan_count),
            ACCENT_DANGER if report.nan_count > 0 else None,
        )
        self._rows["Inf Count"].set_value(
            str(report.inf_count),
            ACCENT_DANGER if report.inf_count > 0 else None,
        )
