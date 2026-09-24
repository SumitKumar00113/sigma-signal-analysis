"""GUI test: the input wizard suggests a WAV interpretation."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import wave  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from src.core.enums import WavInterpretation  # noqa: E402
from src.gui.input_wizard import InputWizard  # noqa: E402


def _wav(path, data, fs=48000):
    data = np.atleast_2d(np.asarray(data, dtype=np.float64).T).T
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((data / np.max(np.abs(data)) * 26000).astype("<i2").tobytes())
    return path


@pytest.mark.parametrize("stereo,expected", [
    (False, WavInterpretation.REAL),
    (True, WavInterpretation.STEREO_IQ),
])
def test_wav_suggestion(qtbot, tmp_path, stereo, expected):
    n = np.arange(48000)
    z = np.exp(2j * np.pi * 3000 * n / 48000) * (1 + 0.3 * np.sin(2 * np.pi * 7 * n / 48000))
    data = np.column_stack([z.real, z.imag]) if stereo else z.real
    wiz = InputWizard()
    qtbot.addWidget(wiz)
    wiz._file_path = str(_wav(tmp_path / "x.wav", data))
    wiz._auto_detect()
    assert wiz._wav_interp_combo.currentText() == expected.value
    assert "Suggested" in wiz._wav_hint.text()
    assert wiz._sr_spin.value() == 48000
    wiz._wav_channel_spin.setValue(0)
    wiz._wav_swap_check.setChecked(True)
    meta = wiz._build_metadata()
    assert meta.wav_interpretation == expected and meta.wav_swap_iq
