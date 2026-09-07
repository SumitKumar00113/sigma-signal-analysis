"""Input Characterization Wizard.

Multi-step wizard that collects or infers recording metadata:
file path, sample type, sample rate, center frequency, IQ order,
endianness, byte offset, and channel mapping (PRD §4.1).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.enums import FileFormat, IQOrder, SampleDatatype, WavInterpretation
from src.core.models import RecordingMetadata


class InputWizard(QDialog):
    """Multi-step wizard for recording metadata characterisation."""

    recording_ready = Signal(object, object)  # (path: str, metadata: RecordingMetadata)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Input Characterization Wizard")
        self.setMinimumSize(600, 500)
        self._file_path: str = ""
        self._detected_format: FileFormat | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("📡  Input Characterization")
        title.setProperty("role", "heading")
        layout.addWidget(title)

        # Stacked pages
        self._pages = QStackedWidget()

        # --- Page 1: File selection ---
        page1 = QWidget()
        p1_layout = QVBoxLayout(page1)
        p1_layout.setSpacing(12)

        file_group = QGroupBox("Select Recording File")
        file_form = QHBoxLayout(file_group)
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Path to .wav, .iq, .cf32, .sigmf-meta, …")
        file_form.addWidget(self._file_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self._browse_file)
        file_form.addWidget(browse_btn)
        p1_layout.addWidget(file_group)

        # Auto-detected info
        self._detect_label = QLabel("")
        self._detect_label.setProperty("role", "subtitle")
        p1_layout.addWidget(self._detect_label)
        p1_layout.addStretch()

        self._pages.addWidget(page1)

        # --- Page 2: Metadata entry ---
        page2 = QWidget()
        p2_layout = QVBoxLayout(page2)

        meta_group = QGroupBox("Recording Metadata")
        form = QFormLayout(meta_group)
        form.setSpacing(8)

        self._format_combo = QComboBox()
        self._format_combo.addItems([f.value for f in FileFormat])
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("File Format:", self._format_combo)

        self._dtype_combo = QComboBox()
        self._dtype_combo.addItems([d.value for d in SampleDatatype])
        form.addRow("Sample Type:", self._dtype_combo)

        self._sr_spin = QDoubleSpinBox()
        self._sr_spin.setRange(0, 100e9)
        self._sr_spin.setDecimals(0)
        self._sr_spin.setSuffix(" Hz")
        self._sr_spin.setValue(2_400_000)
        form.addRow("Sample Rate:", self._sr_spin)

        self._cf_spin = QDoubleSpinBox()
        self._cf_spin.setRange(0, 300e9)
        self._cf_spin.setDecimals(0)
        self._cf_spin.setSuffix(" Hz")
        self._cf_spin.setValue(0)
        form.addRow("Center Frequency:", self._cf_spin)

        self._iq_combo = QComboBox()
        self._iq_combo.addItems([o.value for o in IQOrder])
        form.addRow("IQ Order:", self._iq_combo)

        self._wav_interp_combo = QComboBox()
        self._wav_interp_combo.addItems([w.value for w in WavInterpretation])
        form.addRow("WAV Interpretation:", self._wav_interp_combo)

        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(0, 1_000_000)
        self._offset_spin.setSuffix(" bytes")
        form.addRow("Byte Offset:", self._offset_spin)

        self._channels_spin = QSpinBox()
        self._channels_spin.setRange(1, 16)
        self._channels_spin.setValue(1)
        form.addRow("Channels:", self._channels_spin)

        self._notes_edit = QLineEdit()
        self._notes_edit.setPlaceholderText("Optional notes…")
        form.addRow("Notes:", self._notes_edit)

        p2_layout.addWidget(meta_group)
        p2_layout.addStretch()

        self._pages.addWidget(page2)

        layout.addWidget(self._pages)

        # Navigation buttons
        nav = QHBoxLayout()
        nav.addStretch()
        self._back_btn = QPushButton("← Back")
        self._back_btn.setProperty("flat", "true")
        self._back_btn.clicked.connect(self._go_back)
        self._back_btn.setVisible(False)
        nav.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)

        self._finish_btn = QPushButton("✓ Open Recording")
        self._finish_btn.clicked.connect(self._finish)
        self._finish_btn.setVisible(False)
        nav.addWidget(self._finish_btn)

        layout.addLayout(nav)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_next(self) -> None:
        if self._pages.currentIndex() == 0:
            path = self._file_edit.text().strip()
            if not path or not Path(path).exists():
                self._detect_label.setText("⚠ File not found")
                return
            self._file_path = path
            self._auto_detect()
            self._pages.setCurrentIndex(1)
            self._back_btn.setVisible(True)
            self._next_btn.setVisible(False)
            self._finish_btn.setVisible(True)

    def _go_back(self) -> None:
        self._pages.setCurrentIndex(0)
        self._back_btn.setVisible(False)
        self._next_btn.setVisible(True)
        self._finish_btn.setVisible(False)

    def _finish(self) -> None:
        meta = self._build_metadata()
        self.recording_ready.emit(self._file_path, meta)
        self.accept()

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def _auto_detect(self) -> None:
        """Try to detect file format from extension."""
        p = Path(self._file_path)
        ext = p.suffix.lower()

        if ext == ".wav":
            self._detected_format = FileFormat.WAV
            self._format_combo.setCurrentText(FileFormat.WAV.value)
            self._dtype_combo.setEnabled(False)
            self._iq_combo.setEnabled(False)
            self._detect_label.setText("✓ Detected WAV format")
        elif ext in (".sigmf-meta", ".sigmf-data", ".sigmf-archive"):
            self._detected_format = FileFormat.SIGMF
            self._format_combo.setCurrentText(FileFormat.SIGMF.value)
            self._detect_label.setText("✓ Detected SigMF format")
        else:
            self._detected_format = FileFormat.RAW_IQ
            self._format_combo.setCurrentText(FileFormat.RAW_IQ.value)
            # Guess dtype from extension
            ext_map = {
                ".cf32": SampleDatatype.CF32_LE,
                ".cs16": SampleDatatype.CI16_LE,
                ".cu8": SampleDatatype.CU8,
                ".iq": SampleDatatype.CF32_LE,
            }
            if ext in ext_map:
                self._dtype_combo.setCurrentText(ext_map[ext].value)
                self._detect_label.setText(f"✓ Detected raw IQ ({ext})")
            else:
                self._detect_label.setText("ℹ Raw IQ — please specify metadata")

    def _on_format_changed(self, _index: int) -> None:
        fmt = self._format_combo.currentText()
        is_raw = fmt == FileFormat.RAW_IQ.value
        self._dtype_combo.setEnabled(is_raw)
        self._iq_combo.setEnabled(is_raw)
        self._offset_spin.setEnabled(is_raw)
        is_wav = fmt == FileFormat.WAV.value
        self._wav_interp_combo.setEnabled(is_wav)

    # ------------------------------------------------------------------
    # Build metadata
    # ------------------------------------------------------------------

    def _build_metadata(self) -> RecordingMetadata:
        return RecordingMetadata(
            source_path=self._file_path,
            source_format=FileFormat(self._format_combo.currentText()),
            sample_datatype=SampleDatatype(self._dtype_combo.currentText()),
            sample_rate_hz=self._sr_spin.value(),
            center_frequency_hz=self._cf_spin.value(),
            iq_order=IQOrder(self._iq_combo.currentText()),
            wav_interpretation=WavInterpretation(self._wav_interp_combo.currentText()),
            byte_offset=self._offset_spin.value(),
            num_channels=self._channels_spin.value(),
            notes=self._notes_edit.text(),
        )

    # ------------------------------------------------------------------
    # Browse
    # ------------------------------------------------------------------

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Recording File",
            "",
            "All Supported ("
            "*.wav *.iq *.cf32 *.cs16 *.cu8 "
            "*.sigmf-meta *.sigmf-data *.sigmf-archive);;"
            "WAV Files (*.wav);;"
            "Raw IQ (*.iq *.cf32 *.cs16 *.cu8);;"
            "SigMF (*.sigmf-meta *.sigmf-data *.sigmf-archive);;"
            "All Files (*)",
        )
        if path:
            self._file_edit.setText(path)
