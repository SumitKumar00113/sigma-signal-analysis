"""GUI test: auto-detect interleaver + FEC in the decoding workbench."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from src.core.enums import FECType, InterleaverType  # noqa: E402
from src.decoding.interleaving import InterleaverSpec, interleave  # noqa: E402
from src.decoding.viterbi import ConvCode, conv_encode  # noqa: E402
from src.gui.decoding_panel import DecodingPanel  # noqa: E402


def test_auto_detect_chain(qtbot):
    rng = np.random.default_rng(4)
    data = rng.integers(0, 2, 12000).astype(np.uint8)
    coded = conv_encode(data, ConvCode(7, (0o171, 0o133), (1, 1, 0, 1)), terminate=False)
    x = interleave(coded, InterleaverSpec(InterleaverType.BLOCK, rows=12, cols=40))[57:]
    x ^= (rng.random(len(x)) < 0.005).astype(np.uint8)

    panel = DecodingPanel()
    qtbot.addWidget(panel)
    panel.set_bits(x, "test")

    panel._auto_detect_interleaver()
    qtbot.waitUntil(lambda: not panel._busy, timeout=120_000)
    assert panel._il_type.currentData() == InterleaverType.BLOCK
    assert (panel._il_rows.value(), panel._il_cols.value()) == (12, 40)
    assert panel._il_offset.value() == 480 - 57
    assert "deinterleaved" in panel._stages

    panel._auto_detect_fec()
    qtbot.waitUntil(lambda: not panel._busy, timeout=120_000)
    assert panel._fec_type.currentData() in (FECType.CONVOLUTIONAL, FECType.CONCATENATED)
    assert panel._conv_punct.text() == "1,1,0,1"
    decoded = panel._stages["decoded"]
    # Locate the decoded bits in the transmitted data and check they match
    key = decoded[300:364]
    pos = next(p for p in range(len(data) - 64) if np.array_equal(data[p:p + 64], key))
    start = pos - 300
    m = min(len(decoded), len(data) - start) - 300
    assert np.mean(decoded[300:m] != data[start + 300:start + m]) < 1e-3
    assert "Detected" in panel._fec_result.text()


def test_auto_detect_ldpc(qtbot):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_ldpc import ira_code

    from src.decoding.ldpc import ldpc_encode

    rng = np.random.default_rng(9)
    code = ira_code()
    infos = [rng.integers(0, 2, code.k).astype(np.uint8) for _ in range(6)]
    stream = np.concatenate([ldpc_encode(i, code) for i in infos])[1234:]
    stream ^= (rng.random(len(stream)) < 0.01).astype(np.uint8)

    panel = DecodingPanel()
    qtbot.addWidget(panel)
    panel._add_ldpc_code(code)
    panel.set_bits(stream, "ldpc test")
    panel._auto_detect_fec()
    qtbot.waitUntil(lambda: not panel._busy, timeout=180_000)
    assert panel._fec_type.currentData() == FECType.LDPC
    assert panel._fec_offset.value() == 2000 - 1234
    decoded = panel._stages["decoded"]
    assert np.array_equal(decoded, np.concatenate(infos[1:]))
