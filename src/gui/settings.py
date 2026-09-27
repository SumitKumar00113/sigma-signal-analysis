"""Persistent user settings, recent recordings, project files, and the
Settings dialog."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.models import RecordingMetadata
from src.gui.recording_overview import section_label

CLASSIFIER_MODES = {
    "hybrid": "Hybrid (rules + learned model)",
    "rules": "Rules only",
    "ml": "Learned model only",
}
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"]
MAX_RECENT = 8

PROJECT_FORMAT = "sigma-project"
PROJECT_VERSION = 1
PROJECT_SUFFIX = ".sigma"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Preferences:
    classifier_mode: str = "hybrid"
    max_samples_millions: int = 10       # samples read from a recording for display/analysis
    worker_threads: int = 4
    spectrum_fft: int = 4096
    waterfall_fft: int = 1024
    waterfall_colormap: str = "viridis"


class AppSettings:
    """Typed wrapper around :class:`QSettings`."""

    def __init__(self, qsettings: QSettings | None = None) -> None:
        self._s = qsettings or QSettings("Sigma", "Sigma Signal Analysis")

    def _int(self, key: str, default: int, lo: int, hi: int) -> int:
        try:
            v = int(self._s.value(key, default))
        except (TypeError, ValueError):
            v = default
        return min(hi, max(lo, v))

    def preferences(self) -> Preferences:
        d = Preferences()
        mode = str(self._s.value("classifier_mode", d.classifier_mode))
        cmap = str(self._s.value("waterfall_colormap", d.waterfall_colormap))
        return Preferences(
            classifier_mode=mode if mode in CLASSIFIER_MODES else d.classifier_mode,
            max_samples_millions=self._int("max_samples_millions", d.max_samples_millions, 1, 500),
            worker_threads=self._int("worker_threads", d.worker_threads, 1, 64),
            spectrum_fft=self._int("spectrum_fft", d.spectrum_fft, 128, 65536),
            waterfall_fft=self._int("waterfall_fft", d.waterfall_fft, 128, 8192),
            waterfall_colormap=cmap if cmap in COLORMAPS else d.waterfall_colormap,
        )

    def save_preferences(self, prefs: Preferences) -> None:
        for key, value in prefs.__dict__.items():
            self._s.setValue(key, value)
        self._s.sync()

    # ---- recent recordings -------------------------------------------

    def recent(self) -> list[tuple[str, RecordingMetadata]]:
        """Recent recordings that still exist, newest first."""
        try:
            items = json.loads(str(self._s.value("recent_recordings", "[]")))
        except json.JSONDecodeError:
            return []
        out = []
        for item in items:
            try:
                path = str(item["path"])
                meta = RecordingMetadata.model_validate(item["metadata"])
            except (KeyError, TypeError, ValueError):
                continue
            if Path(path).exists():
                out.append((path, meta))
        return out

    def add_recent(self, path: str, meta: RecordingMetadata) -> None:
        entries = [(p, m) for p, m in self.recent() if p != path]
        entries.insert(0, (path, meta))
        payload = [{"path": p, "metadata": m.model_dump(mode="json")}
                   for p, m in entries[:MAX_RECENT]]
        self._s.setValue("recent_recordings", json.dumps(payload))
        self._s.sync()

    def clear_recent(self) -> None:
        self._s.setValue("recent_recordings", "[]")
        self._s.sync()


# ---------------------------------------------------------------------------
# Project files
# ---------------------------------------------------------------------------


@dataclass
class Project:
    """What a saved project restores: the recordings (with the reader
    settings chosen in the input wizard), which one was active, and the
    analysis options."""

    recordings: list[tuple[str, RecordingMetadata]] = field(default_factory=list)
    active: int = 0
    classifier_mode: str = "hybrid"
    modulation_override: str | None = None
    symbol_rate_override: float | None = None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if not path.name.endswith(PROJECT_SUFFIX):
            path = path.with_name(path.name + PROJECT_SUFFIX)
        doc = {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "recordings": [{"path": p, "metadata": m.model_dump(mode="json")}
                           for p, m in self.recordings],
            "active": self.active,
            "classifier_mode": self.classifier_mode,
            "overrides": {"modulation": self.modulation_override,
                          "symbol_rate_hz": self.symbol_rate_override},
        }
        path.write_text(json.dumps(doc, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Project:
        doc = json.loads(Path(path).read_text())
        if doc.get("format") != PROJECT_FORMAT:
            raise ValueError("not a Sigma project file")
        recs = [(str(r["path"]), RecordingMetadata.model_validate(r["metadata"]))
                for r in doc.get("recordings", [])]
        ov = doc.get("overrides") or {}
        mode = doc.get("classifier_mode", "hybrid")
        return cls(
            recordings=recs,
            active=min(max(0, int(doc.get("active", 0))), max(0, len(recs) - 1)),
            classifier_mode=mode if mode in CLASSIFIER_MODES else "hybrid",
            modulation_override=ov.get("modulation"),
            symbol_rate_override=ov.get("symbol_rate_hz"),
        )


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class SettingsDialog(QDialog):
    """Edit :class:`Preferences`."""

    def __init__(self, prefs: Preferences, ml_available: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(10)

        title = QLabel("Settings")
        title.setProperty("role", "heading")
        root.addWidget(title)

        form = QFormLayout()
        form.setSpacing(8)
        form.addRow(section_label("Analysis"))
        self._mode = QComboBox()
        for key, label in CLASSIFIER_MODES.items():
            self._mode.addItem(label, key)
            if key != "rules" and not ml_available:
                idx = self._mode.count() - 1
                self._mode.model().item(idx).setEnabled(False)
        self._mode.setCurrentIndex(max(0, self._mode.findData(prefs.classifier_mode)))
        form.addRow("Classifier", self._mode)

        self._max_samples = QSpinBox()
        self._max_samples.setRange(1, 500)
        self._max_samples.setSuffix(" M samples")
        self._max_samples.setValue(prefs.max_samples_millions)
        self._max_samples.setToolTip("How much of a recording is read for display and analysis. "
                                     "Applies to the next recording opened.")
        form.addRow("Read up to", self._max_samples)

        self._threads = QSpinBox()
        self._threads.setRange(1, 64)
        self._threads.setValue(prefs.worker_threads)
        self._threads.setToolTip("Background threads for loading, analysis and decoding")
        form.addRow("Worker threads", self._threads)

        form.addRow(section_label("Display"))
        self._spec_fft = self._pow2_combo(prefs.spectrum_fft, 256, 65536)
        form.addRow("Spectrum FFT size", self._spec_fft)
        self._wf_fft = self._pow2_combo(prefs.waterfall_fft, 128, 8192)
        form.addRow("Waterfall FFT size", self._wf_fft)
        self._cmap = QComboBox()
        self._cmap.addItems(COLORMAPS)
        self._cmap.setCurrentText(prefs.waterfall_colormap)
        form.addRow("Waterfall colour map", self._cmap)
        root.addLayout(form)

        root.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _pow2_combo(value: int, lo: int, hi: int) -> QComboBox:
        combo = QComboBox()
        n = lo
        while n <= hi:
            combo.addItem(f"{n:,}", n)
            n *= 2
        idx = combo.findData(value)
        if idx < 0:
            combo.addItem(f"{value:,}", value)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)
        return combo

    def preferences(self) -> Preferences:
        return Preferences(
            classifier_mode=self._mode.currentData(),
            max_samples_millions=self._max_samples.value(),
            worker_threads=self._threads.value(),
            spectrum_fft=int(self._spec_fft.currentData()),
            waterfall_fft=int(self._wf_fft.currentData()),
            waterfall_colormap=self._cmap.currentText(),
        )
