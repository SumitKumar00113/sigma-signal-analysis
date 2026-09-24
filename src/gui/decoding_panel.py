"""Bit-stream decoding workbench.

A central tab that takes the demodulated bits and lets the analyst
run them through, in order:

1. **De-interleaving** – block, convolutional, diagonal, pseudo-random,
   each with its parameters, or blind auto-detection of type, size and
   alignment (:mod:`src.decoding.interleaver_id`).
2. **FEC decoding** – convolutional (Viterbi, standard or custom
   polynomials, optional puncturing), Reed-Solomon (n, k, field
   parameters), concatenated RS+conv, or LDPC (standard codes from
   .alist / .qc files, downloadable from a catalogue); or blind
   auto-detection of the code (:mod:`src.decoding.fec_id`).
3. **Correlation** – autocorrelation for frame-period detection, sync
   word search (binary or hex, with error tolerance, inverted-stream
   aware), and header/payload framing.

Every stage output is kept, so the view selector can show raw,
de-interleaved, or decoded bits and correlation can run on any of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.enums import FECType, InterleaverType
from src.decoding.correlation import (
    bit_autocorrelation,
    bits_to_hex,
    bits_to_string,
    detect_frame_period,
    find_sync_word,
    split_frames,
)
from src.decoding.fec_id import ConvHypothesis, FECIdentification, identify_fec
from src.decoding.interleaver_id import InterleaverIdentification, identify_interleaver
from src.decoding.interleaving import InterleaverSpec, deinterleave
from src.decoding.ldpc import (
    LDPCCode,
    ldpc_decode_stream,
    ldpc_from_H,
    load_ldpc,
    make_regular_ldpc,
)
from src.decoding.reed_solomon import ReedSolomon, rs_decode_stream
from src.decoding.viterbi import STANDARD_CODES, ConvCode, viterbi_decode
from src.gui.theme import (
    ACCENT_DANGER,
    ACCENT_PRIMARY,
    ACCENT_SUCCESS,
    ACCENT_WARNING,
    BG_DARKEST,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from src.gui.workers import Worker, WorkerPool

_MAX_DISPLAY_CHARS = 60_000
_STAGES = ("raw", "deinterleaved", "decoded")


def _small(label: QLabel, color: str = TEXT_SECONDARY) -> None:
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {color}; font-size: 11px; background: transparent;")


class DecodingPanel(QWidget):
    """Bit-stream de-interleaving, FEC decoding, and correlation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stages: dict[str, np.ndarray] = {}
        self._source_label = ""
        self._busy = False
        self._setup_ui()
        self._refresh_view_combo()

    # ==================================================================
    # UI
    # ==================================================================

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        self._status = QLabel("No bits loaded — run analysis on a recording first.")
        self._status.setProperty("role", "subtitle")
        header.addWidget(self._status)
        header.addStretch()
        header.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.currentIndexChanged.connect(self._render_bits)
        header.addWidget(self._view_combo)
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["Hex", "Binary"])
        self._fmt_combo.currentIndexChanged.connect(self._render_bits)
        header.addWidget(self._fmt_combo)
        save_btn = QPushButton("💾 Save bits…")
        save_btn.clicked.connect(self._save_bits)
        header.addWidget(save_btn)
        root.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.setSpacing(8)
        cl.addWidget(self._build_deinterleave_group())
        cl.addWidget(self._build_fec_group())
        cl.addWidget(self._build_correlation_group())
        cl.addStretch()
        controls_scroll.setWidget(controls)
        controls_scroll.setMinimumWidth(380)
        splitter.addWidget(controls_scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self._bits_view = QPlainTextEdit()
        self._bits_view.setReadOnly(True)
        self._bits_view.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 12px;")
        rl.addWidget(self._bits_view, stretch=3)

        pg.setConfigOptions(background=BG_DARKEST, foreground=TEXT_PRIMARY, antialias=True)
        self._ac_plot = pg.PlotWidget(title="Autocorrelation / Sync correlation")
        self._ac_plot.showGrid(x=True, y=True, alpha=0.15)
        self._ac_plot.setLabel("bottom", "Lag / position (bits)")
        self._ac_plot.setLabel("left", "Correlation")
        self._ac_curve = self._ac_plot.plot(pen=pg.mkPen(ACCENT_PRIMARY, width=1))
        rl.addWidget(self._ac_plot, stretch=2)

        self._frames_table = QTableWidget(0, 4)
        self._frames_table.setHorizontalHeaderLabels(
            ["Start (bit)", "Inverted", "Header (hex)", "Payload (hex, first 32 bytes)"]
        )
        self._frames_table.horizontalHeader().setStretchLastSection(True)
        self._frames_table.setEditTriggers(QTableWidget.NoEditTriggers)
        rl.addWidget(self._frames_table, stretch=2)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    # ---- 1 · De-interleave ---------------------------------------------

    def _build_deinterleave_group(self) -> QGroupBox:
        g = QGroupBox("1 · De-interleave")
        form = QFormLayout(g)
        form.setSpacing(6)

        self._il_type = QComboBox()
        for t in (InterleaverType.NONE, InterleaverType.BLOCK, InterleaverType.CONVOLUTIONAL,
                  InterleaverType.DIAGONAL, InterleaverType.PSEUDO_RANDOM):
            self._il_type.addItem(t.value, t)
        self._il_type.currentIndexChanged.connect(self._on_il_type_changed)
        form.addRow("Type:", self._il_type)

        def spin(lo: int, hi: int, val: int) -> QSpinBox:
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            return s

        self._il_rows = spin(1, 4096, 8)
        self._il_cols = spin(1, 4096, 8)
        self._il_branches = spin(1, 256, 4)
        self._il_delay = spin(1, 4096, 1)
        self._il_block = spin(2, 65536, 64)
        self._il_seed = spin(0, 2**31 - 1, 1)
        self._il_gen = QComboBox()
        self._il_gen.addItems(["lcg", "numpy"])

        self._il_widgets: dict[str, list[tuple[QLabel, QWidget]]] = {
            "grid": [(QLabel("Rows:"), self._il_rows), (QLabel("Columns:"), self._il_cols)],
            "conv": [(QLabel("Branches (N):"), self._il_branches),
                     (QLabel("Delay (M):"), self._il_delay)],
            "rnd": [(QLabel("Block size:"), self._il_block), (QLabel("Seed:"), self._il_seed),
                    (QLabel("Generator:"), self._il_gen)],
        }
        for pairs in self._il_widgets.values():
            for lbl, w in pairs:
                form.addRow(lbl, w)
        self._il_offset = spin(0, 10_000_000, 0)
        self._il_offset.setToolTip("Bits to drop before de-interleaving (block alignment)")
        form.addRow("Skip bits:", self._il_offset)

        btns = QHBoxLayout()
        btn = QPushButton("Apply de-interleave")
        btn.clicked.connect(self._apply_deinterleave)
        btns.addWidget(btn)
        self._il_auto_btn = QPushButton("🔍 Auto-detect")
        self._il_auto_btn.setToolTip(
            "Blind search for block, diagonal and convolutional interleavers (and pseudo-random "
            "seeds if block sizes are given). Needs a convolutional code in the stream; the code "
            "selected under FEC is tried first.")
        self._il_auto_btn.clicked.connect(self._auto_detect_interleaver)
        btns.addWidget(self._il_auto_btn)
        form.addRow(btns)
        self._il_pr_blocks = QLineEdit("")
        self._il_pr_blocks.setPlaceholderText("pseudo-random block sizes, e.g. 128,256 (optional)")
        form.addRow("Auto: PR blocks:", self._il_pr_blocks)
        self._il_pr_seeds = spin(1, 1_000_000, 256)
        self._il_pr_seeds.setToolTip("Pseudo-random auto-detect tries seeds 0 … N−1")
        form.addRow("Auto: PR seeds:", self._il_pr_seeds)
        self._il_result = QLabel("")
        _small(self._il_result)
        form.addRow(self._il_result)
        self._on_il_type_changed(0)
        return g

    def _on_il_type_changed(self, _i: int) -> None:
        t = InterleaverType(self._il_type.currentData())
        show = {
            "grid": t in (InterleaverType.BLOCK, InterleaverType.DIAGONAL),
            "conv": t == InterleaverType.CONVOLUTIONAL,
            "rnd": t == InterleaverType.PSEUDO_RANDOM,
        }
        for key, pairs in self._il_widgets.items():
            for lbl, w in pairs:
                lbl.setVisible(show[key])
                w.setVisible(show[key])

    # ---- 2 · FEC ------------------------------------------------------

    def _build_fec_group(self) -> QGroupBox:
        g = QGroupBox("2 · FEC Decode")
        form = QFormLayout(g)
        form.setSpacing(6)

        self._fec_type = QComboBox()
        self._fec_type.addItem("None", FECType.NONE)
        self._fec_type.addItem("Convolutional (Viterbi)", FECType.CONVOLUTIONAL)
        self._fec_type.addItem("Reed-Solomon", FECType.REED_SOLOMON)
        self._fec_type.addItem("Concatenated (RS outer + Conv inner)", FECType.CONCATENATED)
        self._fec_type.addItem("LDPC", FECType.LDPC)
        self._fec_type.currentIndexChanged.connect(self._on_fec_type_changed)
        form.addRow("Code:", self._fec_type)

        self._conv_preset = QComboBox()
        for name in STANDARD_CODES:
            self._conv_preset.addItem(name)
        self._conv_preset.addItem("Custom…")
        self._conv_k = QSpinBox()
        self._conv_k.setRange(2, 12)
        self._conv_k.setValue(7)
        self._conv_polys = QLineEdit("171,133")
        self._conv_polys.setPlaceholderText("octal polys, comma-separated")
        self._conv_punct = QLineEdit("")
        self._conv_punct.setPlaceholderText("e.g. 1,1,0,1,1,0 for rate 3/4 (blank = none)")

        self._rs_n = QSpinBox()
        self._rs_n.setRange(3, 255)
        self._rs_n.setValue(255)
        self._rs_k = QSpinBox()
        self._rs_k.setRange(1, 254)
        self._rs_k.setValue(223)
        self._rs_prim = QLineEdit("0x11d")
        self._rs_fcr = QSpinBox()
        self._rs_fcr.setRange(0, 254)
        self._rs_gen = QSpinBox()
        self._rs_gen.setRange(1, 254)
        self._rs_gen.setValue(1)
        self._rs_offset = QSpinBox()
        self._rs_offset.setRange(0, 10_000_000)
        self._rs_offset.setToolTip("Bits to drop at the RS decoder input (codeword alignment)")

        self._ldpc_codes: dict[str, LDPCCode] = {}
        self._ldpc_code = QComboBox()
        self._ldpc_code.setToolTip("Installed standard codes (~/.sigma/ldpc) and loaded files")
        self._ldpc_n = QSpinBox()
        self._ldpc_n.setRange(12, 4096)
        self._ldpc_n.setValue(96)
        self._ldpc_n.setToolTip("Length of the random regular (3,6) demo code")
        ldpc_btns = QWidget()
        lb = QHBoxLayout(ldpc_btns)
        lb.setContentsMargins(0, 0, 0, 0)
        load_btn = QPushButton("Load .alist/.qc…")
        load_btn.clicked.connect(self._load_ldpc_file)
        lb.addWidget(load_btn)
        dl_btn = QPushButton("⬇ Standard codes…")
        dl_btn.setToolTip("Download DVB-S2, Wi-Fi, WiMAX, 5G NR, CCSDS, 10GBASE-T matrices")
        dl_btn.clicked.connect(self._download_ldpc_codes)
        lb.addWidget(dl_btn)
        self._ldpc_iter = QSpinBox()
        self._ldpc_iter.setRange(1, 200)
        self._ldpc_iter.setValue(50)

        self._fec_widgets: dict[str, list[tuple[QLabel, QWidget]]] = {
            "conv": [(QLabel("Preset:"), self._conv_preset),
                     (QLabel("Constraint length K:"), self._conv_k),
                     (QLabel("Generators (octal):"), self._conv_polys),
                     (QLabel("Puncture pattern:"), self._conv_punct)],
            "rs": [(QLabel("RS n:"), self._rs_n), (QLabel("RS k:"), self._rs_k),
                   (QLabel("Primitive poly:"), self._rs_prim),
                   (QLabel("First root (fcr):"), self._rs_fcr),
                   (QLabel("Generator exp (prim):"), self._rs_gen),
                   (QLabel("RS skip bits:"), self._rs_offset)],
            "ldpc": [(QLabel("LDPC code:"), self._ldpc_code),
                     (QLabel("Demo code n:"), self._ldpc_n),
                     (QLabel(""), ldpc_btns),
                     (QLabel("Iterations:"), self._ldpc_iter)],
        }
        for pairs in self._fec_widgets.values():
            for lbl, w in pairs:
                form.addRow(lbl, w)

        self._fec_offset = QSpinBox()
        self._fec_offset.setRange(0, 10_000_000)
        self._fec_offset.setToolTip("Bits to drop before decoding (code phase / alignment)")
        form.addRow("Skip bits:", self._fec_offset)
        self._fec_invert = QCheckBox("Invert input bits")
        form.addRow(self._fec_invert)

        btns = QHBoxLayout()
        self._refresh_ldpc_codes()
        btn = QPushButton("Decode")
        btn.clicked.connect(self._apply_fec)
        btns.addWidget(btn)
        self._fec_auto_btn = QPushButton("🔍 Auto-detect")
        self._fec_auto_btn.setToolTip(
            "Blind identification: convolutional codes (library incl. punctured, and blind "
            "rate-1/n generator recovery), Reed-Solomon, concatenated RS+conv, and generic "
            "linear block structure.")
        self._fec_auto_btn.clicked.connect(self._auto_detect_fec)
        btns.addWidget(self._fec_auto_btn)
        form.addRow(btns)
        self._fec_result = QLabel("")
        _small(self._fec_result)
        form.addRow(self._fec_result)
        self._on_fec_type_changed(0)
        return g

    def _on_fec_type_changed(self, _i: int) -> None:
        t = FECType(self._fec_type.currentData())
        show = {
            "conv": t in (FECType.CONVOLUTIONAL, FECType.CONCATENATED),
            "rs": t in (FECType.REED_SOLOMON, FECType.CONCATENATED),
            "ldpc": t == FECType.LDPC,
        }
        for key, pairs in self._fec_widgets.items():
            for lbl, w in pairs:
                lbl.setVisible(show[key])
                w.setVisible(show[key])

    # ---- 3 · Correlation -----------------------------------------------

    def _build_correlation_group(self) -> QGroupBox:
        g = QGroupBox("3 · Bit-stream Correlation")
        form = QFormLayout(g)
        form.setSpacing(6)

        ac_btn = QPushButton("Autocorrelate (find frame period)")
        ac_btn.clicked.connect(self._run_autocorrelation)
        form.addRow(ac_btn)
        self._period_label = QLabel("")
        _small(self._period_label)
        form.addRow(self._period_label)

        self._sync_edit = QLineEdit("0x1ACFFC1D")
        self._sync_edit.setPlaceholderText("binary 1010… or hex 0x…")
        form.addRow("Sync word:", self._sync_edit)
        self._sync_errs = QSpinBox()
        self._sync_errs.setRange(0, 16)
        self._sync_errs.setValue(1)
        form.addRow("Max bit errors:", self._sync_errs)
        self._hdr_bits = QSpinBox()
        self._hdr_bits.setRange(0, 4096)
        form.addRow("Header bits after sync:", self._hdr_bits)
        self._frame_bits = QSpinBox()
        self._frame_bits.setRange(0, 1_000_000)
        self._frame_bits.setSpecialValueText("auto (to next sync)")
        form.addRow("Frame length (bits):", self._frame_bits)
        sync_btn = QPushButton("Search sync word & split frames")
        sync_btn.clicked.connect(self._run_sync_search)
        form.addRow(sync_btn)
        self._sync_result = QLabel("")
        _small(self._sync_result)
        form.addRow(self._sync_result)
        return g

    # ==================================================================
    # Data
    # ==================================================================

    def set_bits(self, bits: np.ndarray, source: str = "") -> None:
        """Load a fresh raw bit stream (clears derived stages)."""
        self._stages = {"raw": np.asarray(bits, dtype=np.uint8).reshape(-1)}
        self._source_label = source
        for lbl in (self._il_result, self._fec_result, self._sync_result, self._period_label):
            lbl.setText("")
        self._frames_table.setRowCount(0)
        self._ac_curve.setData([], [])
        self._refresh_view_combo()
        self._update_status()

    def clear(self) -> None:
        self._stages = {}
        self._refresh_view_combo()
        self._status.setText("No bits loaded — run analysis on a recording first.")

    def current_bits(self) -> np.ndarray:
        return self._stages.get(self._view_combo.currentText(), np.zeros(0, dtype=np.uint8))

    def _input_for_fec(self) -> np.ndarray:
        return self._stages.get("deinterleaved",
                                self._stages.get("raw", np.zeros(0, dtype=np.uint8)))

    def _refresh_view_combo(self) -> None:
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        for key in _STAGES:
            if key in self._stages:
                self._view_combo.addItem(key)
        # Show the most processed stage available
        for key in reversed(_STAGES):
            if key in self._stages:
                self._view_combo.setCurrentText(key)
                break
        self._view_combo.blockSignals(False)
        self._render_bits()

    def _update_status(self) -> None:
        parts = [f"{k}: {len(self._stages[k]):,} bits" for k in _STAGES if k in self._stages]
        src = f"  ·  from {self._source_label}" if self._source_label else ""
        self._status.setText("  |  ".join(parts) + src if parts else "No bits loaded.")

    def _render_bits(self) -> None:
        bits = self.current_bits()
        if len(bits) == 0:
            self._bits_view.setPlainText("")
            return
        if self._fmt_combo.currentText() == "Hex":
            text = bits_to_hex(bits[: (_MAX_DISPLAY_CHARS // 3) * 8])
        else:
            s = bits_to_string(bits[:_MAX_DISPLAY_CHARS])
            text = " ".join(s[i:i + 8] for i in range(0, len(s), 8))
        if len(bits) > _MAX_DISPLAY_CHARS:
            text += f"\n… ({len(bits):,} bits total; display truncated)"
        self._bits_view.setPlainText(text)

    def _set_result(self, label: QLabel, text: str, ok: bool | None = True) -> None:
        color = ACCENT_SUCCESS if ok else (ACCENT_WARNING if ok is None else ACCENT_DANGER)
        _small(label, color)
        label.setText(text)

    # ==================================================================
    # Actions
    # ==================================================================

    def _apply_deinterleave(self) -> None:
        raw = self._stages.get("raw")
        if raw is None or len(raw) == 0:
            self._set_result(self._il_result, "No bits loaded.", None)
            return
        spec = InterleaverSpec(
            kind=InterleaverType(self._il_type.currentData()),
            rows=self._il_rows.value(), cols=self._il_cols.value(),
            branches=self._il_branches.value(), delay=self._il_delay.value(),
            block=self._il_block.value(), seed=self._il_seed.value(),
            generator=self._il_gen.currentText(),
        )
        skip = self._il_offset.value()
        try:
            out = deinterleave(raw[skip:], spec)
        except Exception as exc:  # noqa: BLE001
            self._set_result(self._il_result, f"✗ {exc}", False)
            return
        if spec.kind == InterleaverType.NONE:
            self._stages.pop("deinterleaved", None)
            self._set_result(self._il_result, "No de-interleaving applied.")
        else:
            self._stages["deinterleaved"] = out.astype(np.uint8)
            self._set_result(self._il_result,
                             f"✓ {spec.kind.value}: {len(raw):,} → {len(out):,} bits")
        self._stages.pop("decoded", None)
        self._refresh_view_combo()
        self._update_status()

    def _conv_code(self) -> ConvCode:
        punct_text = self._conv_punct.text().strip()
        punct = (tuple(int(x) for x in punct_text.replace(" ", "").split(","))
                 if punct_text else None)
        preset = self._conv_preset.currentText()
        if preset in STANDARD_CODES:
            k, gens = STANDARD_CODES[preset]
            return ConvCode(k, gens, punct)
        gens = tuple(int(x, 8) for x in self._conv_polys.text().replace(" ", "").split(","))
        return ConvCode(self._conv_k.value(), gens, punct)

    def _rs_code(self) -> ReedSolomon:
        return ReedSolomon(self._rs_n.value(), self._rs_k.value(),
                           int(self._rs_prim.text(), 0), self._rs_fcr.value(),
                           self._rs_gen.value())

    def _apply_fec(self) -> None:
        bits = self._input_for_fec()
        if len(bits) == 0:
            self._set_result(self._fec_result, "No bits loaded.", None)
            return
        bits = bits[self._fec_offset.value():]
        if self._fec_invert.isChecked():
            bits = bits ^ 1
        rs_skip = self._rs_offset.value()
        t = FECType(self._fec_type.currentData())
        try:
            ok: bool | None = True
            if t == FECType.NONE:
                self._stages.pop("decoded", None)
                msg = "No FEC decoding applied."
            elif t == FECType.CONVOLUTIONAL:
                code = self._conv_code()
                r = viterbi_decode(bits, code, terminated=False)
                self._stages["decoded"] = r.bits
                msg = (f"✓ Viterbi K={code.constraint_length} rate {code.rate:.3g}: "
                       f"{len(bits):,} → {len(r.bits):,} bits; "
                       f"{r.estimated_errors:,} channel bit errors corrected "
                       f"({r.estimated_errors / max(1, len(bits)):.2%}).")
            elif t == FECType.REED_SOLOMON:
                rs = self._rs_code()
                out, results = rs_decode_stream(bits[rs_skip:], rs)
                self._stages["decoded"] = out
                good = sum(1 for x in results if x.success)
                corr = sum(x.corrected for x in results)
                msg = (f"RS({rs.n},{rs.k}) t={rs.t}: {len(results)} blocks, {good} OK, "
                       f"{len(results) - good} failed; {corr} symbol errors corrected.")
                ok = good == len(results) if results else None
            elif t == FECType.CONCATENATED:
                code = self._conv_code()
                inner = viterbi_decode(bits, code, terminated=False)
                rs = self._rs_code()
                out, results = rs_decode_stream(inner.bits[rs_skip:], rs)
                self._stages["decoded"] = out
                good = sum(1 for x in results if x.success)
                msg = (f"Concatenated: Viterbi corrected {inner.estimated_errors:,} bits → "
                       f"RS({rs.n},{rs.k}) {good}/{len(results)} blocks OK.")
                ok = good == len(results) if results else None
            elif t == FECType.LDPC:
                code = self._current_ldpc_code()
                r = ldpc_decode_stream(bits, code, max_iter=self._ldpc_iter.value())
                self._stages["decoded"] = r.info_bits
                msg = (f"{code.describe()}: {r.blocks} blocks, {r.converged} converged "
                       f"(mean {r.mean_iterations:.1f} iterations), {r.corrected_bits:,} "
                       "channel bits corrected.")
                ok = (r.converged == r.blocks) if r.blocks else None
            else:
                msg, ok = "Unsupported.", False
            self._set_result(self._fec_result, msg, ok)
        except Exception as exc:  # noqa: BLE001
            self._set_result(self._fec_result, f"✗ {type(exc).__name__}: {exc}", False)
        self._refresh_view_combo()
        self._update_status()

    # ---- Auto-detection (background workers) --------------------------

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._il_auto_btn.setEnabled(not busy)
        self._fec_auto_btn.setEnabled(not busy)

    def _selected_conv_hypothesis(self) -> ConvHypothesis | None:
        """The convolutional code currently configured under FEC, if any."""
        if FECType(self._fec_type.currentData()) not in (FECType.CONVOLUTIONAL,
                                                         FECType.CONCATENATED):
            return None
        try:
            code = self._conv_code()
        except ValueError:
            return None
        return ConvHypothesis("Selected code", code.constraint_length, code.generators,
                              code.puncture)

    def _auto_detect_interleaver(self) -> None:
        raw = self._stages.get("raw")
        if raw is None or len(raw) == 0:
            self._set_result(self._il_result, "No bits loaded.", None)
            return
        if self._busy:
            return
        try:
            text = self._il_pr_blocks.text().replace(" ", "")
            blocks = [int(v) for v in text.split(",") if v]
        except ValueError:
            self._set_result(self._il_result, "✗ PR block sizes must be integers.", False)
            return
        comb = None
        selected = self._selected_conv_hypothesis()
        if selected is not None:
            from src.decoding.fec_id import common_conv_hypotheses
            comb = [selected] + common_conv_hypotheses()
        worker = Worker(identify_interleaver, raw.copy(), comb_hypotheses=comb,
                        pseudo_random_blocks=blocks, seeds=range(self._il_pr_seeds.value()))
        worker.signals.progress.connect(self._on_il_progress)
        worker.signals.finished.connect(self._on_interleaver_detected)
        worker.signals.error.connect(self._on_il_error)
        self._set_busy(True)
        self._set_result(self._il_result, "Searching for an interleaver…", None)
        WorkerPool.instance().start(worker)

    def _on_il_progress(self, fraction: float, message: str) -> None:
        self._set_result(self._il_result, f"Searching… {fraction:.0%} · {message}", None)

    def _on_il_error(self, message: str) -> None:
        self._set_busy(False)
        self._set_result(self._il_result, f"✗ Auto-detect failed: {message}", False)

    def _on_interleaver_detected(self, result: InterleaverIdentification) -> None:
        self._set_busy(False)
        best = result.best
        if best is None:
            self._set_result(self._il_result, result.summary(), None)
            return
        spec = best.spec
        for i in range(self._il_type.count()):
            if self._il_type.itemData(i) == spec.kind:
                self._il_type.setCurrentIndex(i)
                break
        self._il_rows.setValue(spec.rows)
        self._il_cols.setValue(spec.cols)
        self._il_branches.setValue(spec.branches)
        self._il_delay.setValue(spec.delay)
        self._il_block.setValue(spec.block)
        self._il_seed.setValue(spec.seed)
        self._il_gen.setCurrentText(spec.generator)
        self._il_offset.setValue(best.offset)
        self._apply_deinterleave()
        applied = self._il_result.text()
        self._set_result(self._il_result, f"✓ Detected: {result.summary()}\n{applied}")

    def _auto_detect_fec(self) -> None:
        bits = self._input_for_fec()
        if len(bits) == 0:
            self._set_result(self._fec_result, "No bits loaded.", None)
            return
        if self._busy:
            return
        worker = Worker(identify_fec, bits.copy(), ldpc_codes=list(self._ldpc_codes.values()))
        worker.signals.progress.connect(self._on_fec_progress)
        worker.signals.finished.connect(self._on_fec_detected)
        worker.signals.error.connect(self._on_fec_error)
        self._set_busy(True)
        self._set_result(self._fec_result, "Identifying FEC…", None)
        WorkerPool.instance().start(worker)

    def _on_fec_progress(self, fraction: float, message: str) -> None:
        self._set_result(self._fec_result, f"Identifying… {fraction:.0%} · {message}", None)

    def _on_fec_error(self, message: str) -> None:
        self._set_busy(False)
        self._set_result(self._fec_result, f"✗ Auto-detect failed: {message}", False)

    def _select_fec_type(self, fec: FECType) -> None:
        for i in range(self._fec_type.count()):
            if self._fec_type.itemData(i) == fec:
                self._fec_type.setCurrentIndex(i)
                return

    def _on_fec_detected(self, result: FECIdentification) -> None:
        self._set_busy(False)
        conv, rs, ldpc = result.conv, result.rs, result.ldpc
        if conv is None and rs is None and ldpc is None:
            self._set_result(self._fec_result, result.summary(), None)
            return
        self._fec_offset.setValue(0)
        self._fec_invert.setChecked(False)
        self._rs_offset.setValue(0)
        if ldpc is not None:
            self._select_fec_type(FECType.LDPC)
            name = self._add_ldpc_code(ldpc.code)
            self._ldpc_code.setCurrentIndex(self._ldpc_code.findData(name))
            self._fec_offset.setValue(ldpc.offset)
            self._fec_invert.setChecked(ldpc.inverted)
            self._apply_fec()
            decoded = self._fec_result.text()
            self._set_result(self._fec_result, f"✓ Detected: {result.summary()}\n{decoded}")
            return
        if conv is not None:
            self._select_fec_type(FECType.CONCATENATED if rs else FECType.CONVOLUTIONAL)
            code = conv.code
            preset = next((name for name, (k, gens) in STANDARD_CODES.items()
                           if k == code.constraint_length and gens == code.generators), None)
            self._conv_preset.setCurrentText(preset or "Custom…")
            self._conv_k.setValue(code.constraint_length)
            self._conv_polys.setText(",".join(f"{g:o}" for g in code.generators))
            self._conv_punct.setText(conv.puncture_text)
            self._fec_offset.setValue(conv.phase)
            self._fec_invert.setChecked(conv.inverted)
        else:
            self._select_fec_type(FECType.REED_SOLOMON)
        if rs is not None:
            self._rs_n.setValue(rs.n)
            self._rs_k.setValue(rs.k)
            self._rs_prim.setText(f"0x{rs.prim_poly:x}")
            self._rs_fcr.setValue(rs.fcr)
            self._rs_gen.setValue(1)
            self._rs_offset.setValue(rs.bit_offset)
        self._apply_fec()
        decoded = self._fec_result.text()
        self._set_result(self._fec_result, f"✓ Detected: {result.summary()}\n{decoded}")

    # ---- LDPC code management ------------------------------------------

    def _add_ldpc_code(self, code: LDPCCode) -> str:
        name = code.name or f"LDPC ({code.n_transmitted})"
        if name not in self._ldpc_codes:
            self._ldpc_codes[name] = code
            self._ldpc_code.addItem(name, name)
        return name

    def _refresh_ldpc_codes(self) -> None:
        from src.decoding.ldpc_library import load_installed

        current = self._ldpc_code.currentData()
        self._ldpc_code.blockSignals(True)
        self._ldpc_code.clear()
        self._ldpc_code.addItem("Regular (3,6) demo code", "__demo__")
        for name in self._ldpc_codes:
            self._ldpc_code.addItem(name, name)
        for code in load_installed():
            self._add_ldpc_code(code)
        idx = self._ldpc_code.findData(current)
        self._ldpc_code.setCurrentIndex(max(idx, 0))
        self._ldpc_code.blockSignals(False)

    def _current_ldpc_code(self) -> LDPCCode:
        key = self._ldpc_code.currentData()
        if key in self._ldpc_codes:
            return self._ldpc_codes[key]
        n = self._ldpc_n.value() - (self._ldpc_n.value() % 2)
        return ldpc_from_H(make_regular_ldpc(n, 3, 6), name="Regular (3,6) demo")

    def _load_ldpc_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load LDPC parity-check matrix", "",
                                              "LDPC matrices (*.alist *.qc);;All (*)")
        if not path:
            return
        try:
            code = load_ldpc(path)
        except Exception as exc:  # noqa: BLE001
            self._set_result(self._fec_result, f"✗ Could not load {Path(path).name}: {exc}", False)
            return
        name = self._add_ldpc_code(code)
        self._ldpc_code.setCurrentIndex(self._ldpc_code.findData(name))
        self._set_result(self._fec_result, f"✓ Loaded {code.describe()}")

    def _download_ldpc_codes(self) -> None:
        from src.decoding.ldpc_library import CATALOGUE, SOURCE_BASE, fetch, installed

        have = {p.name for p in installed()}
        dlg = QDialog(self)
        dlg.setWindowTitle("Standard LDPC codes")
        lay = QVBoxLayout(dlg)
        note = QLabel("Parity-check matrices are downloaded on request from the AFF3CT "
                      f"project ({SOURCE_BASE.split('/dec/')[0]}) into ~/.sigma/ldpc.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lst = QListWidget()
        for e in CATALOGUE:
            item = QListWidgetItem(f"{e.name}   ({e.standard})")
            item.setData(Qt.UserRole, e.name)
            done = e.filename in have
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if done else Qt.Unchecked)
            if done:
                item.setText(item.text() + "  ✓ installed")
            lst.addItem(item)
        lay.addWidget(lst)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("Download selected")
        box.accepted.connect(dlg.accept)
        box.rejected.connect(dlg.reject)
        lay.addWidget(box)
        if dlg.exec() != QDialog.Accepted:
            return
        wanted = [e for i, e in enumerate(CATALOGUE)
                  if lst.item(i).checkState() == Qt.Checked and e.filename not in have]
        if not wanted:
            return

        def _fetch_all(progress_cb, cancel_check):
            failures = []
            for i, e in enumerate(wanted):
                progress_cb(i / len(wanted), f"Downloading {e.name}")
                try:
                    fetch(e)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{e.name}: {exc}")
            return failures

        worker = Worker(_fetch_all)
        worker.signals.progress.connect(self._on_fec_progress)
        worker.signals.finished.connect(self._on_ldpc_downloaded)
        worker.signals.error.connect(self._on_fec_error)
        self._set_result(self._fec_result, f"Downloading {len(wanted)} code(s)…", None)
        WorkerPool.instance().start(worker)

    def _on_ldpc_downloaded(self, failures: list) -> None:
        self._refresh_ldpc_codes()
        if failures:
            self._set_result(self._fec_result, "✗ " + "; ".join(failures), False)
        else:
            self._set_result(self._fec_result, "✓ Standard LDPC codes installed.")

    def _run_autocorrelation(self) -> None:
        bits = self.current_bits()
        if len(bits) < 64:
            self._set_result(self._period_label, "Need at least 64 bits.", None)
            return
        ac = bit_autocorrelation(bits)
        self._ac_curve.setData(np.arange(len(ac)), ac)
        self._ac_plot.setTitle("Bit autocorrelation")
        cands = detect_frame_period(bits)
        if not cands:
            self._set_result(self._period_label, "No periodic structure found.", None)
            return
        txt = "Frame period candidates: " + ", ".join(
            f"{c.period} bits (r={c.strength:.2f}, {c.harmonics} harmonics)" for c in cands[:4]
        )
        self._set_result(self._period_label, txt)
        if self._frame_bits.value() == 0:
            self._frame_bits.setValue(cands[0].period)

    def _run_sync_search(self) -> None:
        bits = self.current_bits()
        if len(bits) == 0:
            self._set_result(self._sync_result, "No bits loaded.", None)
            return
        try:
            res = find_sync_word(bits, self._sync_edit.text(), max_errors=self._sync_errs.value())
        except ValueError as exc:
            self._set_result(self._sync_result, f"✗ {exc}", False)
            return
        if len(res.correlation):
            self._ac_curve.setData(np.arange(len(res.correlation)), res.correlation)
            self._ac_plot.setTitle("Sync-word correlation (+1 match, −1 inverted match)")
        if not res.matches:
            self._set_result(self._sync_result, "No matches.", None)
            self._frames_table.setRowCount(0)
            return
        inv = sum(1 for m in res.matches if m.inverted)
        per = f", most common spacing {res.best_period} bits" if res.best_period else ""
        self._set_result(self._sync_result,
                         f"✓ {len(res.matches)} match(es) ({inv} inverted){per}. "
                         f"First at bit {res.matches[0].position}.")

        frame_len = self._frame_bits.value() or None
        frames = split_frames(bits, res, header_bits=self._hdr_bits.value(), frame_bits=frame_len)
        self._frames_table.setRowCount(min(len(frames), 500))
        for i, f in enumerate(frames[:500]):
            self._frames_table.setItem(i, 0, QTableWidgetItem(str(f.start)))
            self._frames_table.setItem(i, 1, QTableWidgetItem("yes" if f.inverted else ""))
            self._frames_table.setItem(i, 2, QTableWidgetItem(bits_to_hex(f.header)))
            self._frames_table.setItem(i, 3, QTableWidgetItem(bits_to_hex(f.payload[:256])))
        self._frames_table.resizeColumnsToContents()

    def _save_bits(self) -> None:
        bits = self.current_bits()
        if len(bits) == 0:
            return
        path, flt = QFileDialog.getSaveFileName(
            self, "Save bit stream", "", "Packed binary (*.bin);;Text 0/1 (*.txt);;Hex (*.hex)"
        )
        if not path:
            return
        p = Path(path)
        if flt.startswith("Text") or p.suffix == ".txt":
            p.write_text(bits_to_string(bits), encoding="utf-8")
        elif flt.startswith("Hex") or p.suffix == ".hex":
            p.write_text(bits_to_hex(bits), encoding="utf-8")
        else:
            n = (len(bits) // 8) * 8
            p.write_bytes(np.packbits(bits[:n]).tobytes())
