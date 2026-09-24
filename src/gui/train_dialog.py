"""Dialog for training the learned modulation classifier from the GUI."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrainClassifierDialog(QDialog):
    """Collects training options; the caller runs the training in a worker."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Train Modulation Classifier")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)

        intro = QLabel(
            "Trains the learned classifier on synthetic signals of all supported "
            "modulations, plus (optionally) your own labelled recordings listed in a CSV "
            "manifest (columns: path, modulation, sample_rate, datatype, "
            "wav_interpretation). The model is saved for this user and used by the "
            "Hybrid and Learned-model classifier modes."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self.per_class = QSpinBox()
        self.per_class.setRange(20, 5000)
        self.per_class.setValue(600)
        self.per_class.setToolTip("Synthetic examples per modulation (600 ≈ 3–5 min on 8 cores)")
        form.addRow("Synthetic examples / class:", self.per_class)

        self.snr_lo = QDoubleSpinBox()
        self.snr_lo.setRange(-10, 40)
        self.snr_lo.setValue(0)
        self.snr_hi = QDoubleSpinBox()
        self.snr_hi.setRange(-10, 60)
        self.snr_hi.setValue(20)
        snr_row = QHBoxLayout()
        snr_row.addWidget(self.snr_lo)
        snr_row.addWidget(QLabel("to"))
        snr_row.addWidget(self.snr_hi)
        snr_row.addWidget(QLabel("dB"))
        form.addRow("SNR range:", snr_row)

        self.manifest = QLineEdit("")
        self.manifest.setPlaceholderText("optional: labels.csv for real recordings")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        man_row = QHBoxLayout()
        man_row.addWidget(self.manifest)
        man_row.addWidget(browse)
        form.addRow("Recordings manifest:", man_row)

        self.workers = QSpinBox()
        self.workers.setRange(1, 64)
        self.workers.setValue(max(1, (os.cpu_count() or 2) - 1))
        form.addRow("Worker processes:", self.workers)
        root.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Train")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Recordings manifest", "",
                                              "CSV (*.csv);;All (*)")
        if path:
            self.manifest.setText(path)

    def options(self) -> dict:
        text = self.manifest.text().strip()
        return {
            "per_class": self.per_class.value(),
            "snr_range": (self.snr_lo.value(), max(self.snr_hi.value(), self.snr_lo.value() + 1)),
            "manifest": Path(text) if text else None,
            "workers": self.workers.value(),
        }
