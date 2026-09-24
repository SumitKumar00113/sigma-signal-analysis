"""Tests for the additional modulation types: classification, demodulation,
and the measurements that support them."""

import numpy as np
import pytest
from scipy import signal as sp_signal

from src.core.enums import ModulationType as M
from src.core.models import RecordingMetadata
from src.dsp.analog import analyse_sideband, demodulate_ssb
from src.dsp.classification import classify_modulation
from src.dsp.demod import demodulate, fit_levels
from src.dsp.measurements import (
    estimate_symbol_rate_candidates,
    estimate_symbol_rate_quadrature,
    estimate_symbol_rate_transitions,
    mpower_line_pair,
)
from src.dsp.pipeline import AnalysisPipeline
from src.reporting.exporter import export_audio_wav
from tests.signals import generators as sg
from tests.signals.metrics import ber_oqpsk, ber_with_ambiguity

FS, RS, F0 = 48000.0, 2400.0, 1500.0


def _digital(kind: str, snr: float = 15.0):
    """(samples, bits, bits_per_symbol) for a digital test signal."""
    np.random.seed(11)
    if kind == "ook":
        x, b = sg.generate_ask(3000, RS, FS, (0.0, 1.0), snr, F0)
        return x, b, 1
    if kind == "ask2":
        x, b = sg.generate_ask(3000, RS, FS, (0.4, 1.0), snr, F0)
        return x, b, 1
    if kind == "ask4":
        x, b = sg.generate_ask(3000, RS, FS, (0.25, 0.5, 0.75, 1.0), snr, F0)
        return x, b, 2
    if kind == "msk":
        x, b = sg.generate_msk(3000, RS, FS, None, snr, F0)
        return x, b, 1
    if kind == "gmsk":
        x, b = sg.generate_msk(3000, RS, FS, 0.3, snr, F0)
        return x, b, 1
    if kind == "oqpsk":
        x, b = sg.generate_oqpsk(3000, RS, FS, snr, F0)
        return x, b, 2
    if kind == "pi4":
        x, b = sg.generate_pi4_dqpsk(3000, RS, FS, snr, F0)
        return x, b, 2
    if kind == "dbpsk":
        x, b = sg.generate_dbpsk(3000, RS, FS, snr, F0)
        return x, b, 1
    raise ValueError(kind)


def _analog(kind: str, snr: float = 15.0):
    np.random.seed(5)
    n = int(1.5 * FS)
    if kind == "am":
        return sg.generate_am(n, FS, 0.6, None, snr, F0)
    if kind == "fm":
        return sg.generate_fm(n, FS, 3000.0, None, snr, F0)
    if kind == "usb":
        return sg.generate_ssb(n, FS, "usb", None, snr, F0)
    return sg.generate_ssb(n, FS, "lsb", None, snr, F0)


def _audio_corr(audio: np.ndarray, rate: float, msg: np.ndarray) -> float:
    ref = sp_signal.resample_poly(msg, int(rate), int(FS)) if rate != FS else msg
    ref = sp_signal.filtfilt(sp_signal.firwin(257, 3400 / (rate / 2)), 1, ref)
    n = min(len(ref), len(audio))
    a, r = audio[:n] - np.mean(audio[:n]), ref[:n] - np.mean(ref[:n])
    lag = int(np.argmax(np.abs(sp_signal.correlate(a, r, mode="full")))) - (n - 1)
    a2, r2 = (a[lag:], r[: n - lag]) if lag >= 0 else (a[: n + lag], r[-lag:])
    return float(abs(np.corrcoef(a2, r2)[0, 1]))


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


class TestRateEstimators:
    def test_transitions_find_ask_rate(self):
        x, _, _ = _digital("ask2", snr=8)
        rate, conf = estimate_symbol_rate_transitions(x, FS)[0]
        assert rate == pytest.approx(RS, rel=0.005) and conf > 0.3

    def test_quadrature_finds_oqpsk_rate(self):
        x, _, _ = _digital("oqpsk")
        assert estimate_symbol_rate_quadrature(x, FS)[0][0] == pytest.approx(RS, rel=0.005)

    def test_quadrature_ignores_carrier_and_constant_envelope(self):
        assert estimate_symbol_rate_quadrature(_digital("ask2")[0], FS) == []
        assert estimate_symbol_rate_quadrature(_digital("msk")[0], FS) == []

    @pytest.mark.parametrize("kind", ["ook", "ask2", "ask4", "msk", "gmsk", "pi4", "dbpsk"])
    def test_merged_candidates(self, kind):
        x, _, _ = _digital(kind, snr=8)
        assert estimate_symbol_rate_candidates(x, FS)[0][0] == pytest.approx(RS, rel=0.005)


class TestLinePair:
    def test_pi4_has_asymmetric_pair_and_exact_carrier(self):
        x, _, _ = _digital("pi4")
        carrier, strength, _ = mpower_line_pair(x, FS, RS, 4)
        assert strength > 0.7
        assert carrier == pytest.approx(F0, abs=2.0)

    def test_qpsk_has_no_pair(self):
        np.random.seed(3)
        x = sg._finish(sg.generate_qpsk(3000, RS, FS)[0], FS, 15, F0)
        assert mpower_line_pair(x, FS, RS, 4)[1] == 0.0


class TestLevelFit:
    def test_levels(self):
        rng = np.random.default_rng(0)
        two = rng.choice([0.2, 1.0], 2000) + 0.02 * rng.standard_normal(2000)
        four = rng.choice([0.25, 0.5, 0.75, 1.0], 2000) + 0.02 * rng.standard_normal(2000)
        cont = rng.standard_normal(2000)
        assert fit_levels(two).order == 2
        assert fit_levels(four).order == 4
        assert fit_levels(cont).order == 1


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    @pytest.mark.parametrize("kind,expected", [
        ("ook", M.OOK), ("ask2", M.ASK), ("ask4", M.ASK), ("msk", M.MSK),
        ("gmsk", M.GMSK), ("oqpsk", M.OQPSK), ("pi4", M.PI4_DQPSK), ("dbpsk", M.BPSK),
    ])
    def test_digital(self, kind, expected):
        x, _, _ = _digital(kind, snr=8)
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=FS))
        assert res.analysis.modulation == expected, res.classification.evidence
        assert res.classification.evidence

    @pytest.mark.parametrize("kind,expected", [
        ("am", M.AM), ("fm", M.FM), ("usb", M.SSB_USB), ("lsb", M.SSB_LSB),
    ])
    def test_analog(self, kind, expected):
        x, _ = _analog(kind, snr=8)
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=FS))
        assert res.analysis.modulation == expected, res.classification.evidence

    def test_unmodulated_carrier(self):
        t = np.arange(65536) / FS
        tone = np.exp(2j * np.pi * 1000 * t).astype(np.complex64)
        r = classify_modulation(tone, FS, 0.0, symbol_rate_confidence=0.0)
        assert r.modulation == M.UNKNOWN
        assert "Unmodulated carrier" in r.evidence[0]

    def test_pi4_refines_carrier(self):
        x, _, _ = _digital("pi4")
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=FS))
        assert res.frequency_offset_hz == pytest.approx(F0, abs=2.0)

    def test_oqpsk_rate_revised_from_bad_estimate(self):
        x, _, _ = _digital("oqpsk")
        r = classify_modulation(x, FS, 1560.0, symbol_rate_confidence=0.2,
                                rate_candidates=[(1560.0, 0.2), (2400.0, 0.1)])
        assert r.modulation == M.OQPSK
        assert r.symbol_rate_hz == pytest.approx(RS, rel=0.005)


# ---------------------------------------------------------------------------
# Demodulation
# ---------------------------------------------------------------------------


class TestDigitalDemod:
    @pytest.mark.parametrize("kind,mod", [
        ("ook", M.OOK), ("ask2", M.ASK), ("ask4", M.ASK), ("msk", M.MSK),
        ("gmsk", M.GMSK), ("pi4", M.PI4_DQPSK), ("dbpsk", M.DPSK),
    ])
    def test_ber(self, kind, mod):
        x, bits, k = _digital(kind)
        d = demodulate(x, FS, mod, RS)
        # 4-ASK levels are only 0.25 apart: a few errors remain at 15 dB
        assert ber_with_ambiguity(d.bits, bits, k)[0] < (1e-2 if kind == "ask4" else 1e-3)
        assert d.num_symbols > 2500

    def test_oqpsk(self):
        x, bits, _ = _digital("oqpsk")
        d = demodulate(x, FS, M.OQPSK, RS)
        assert ber_oqpsk(d.bits, bits) == 0.0
        assert d.evm_percent < 25

    def test_ask_order_detected(self):
        assert demodulate(_digital("ask4")[0], FS, M.ASK, RS).bits_per_symbol == 2
        assert demodulate(_digital("ask2")[0], FS, M.ASK, RS).bits_per_symbol == 1


class TestAnalogDemod:
    @pytest.mark.parametrize("kind,mod", [("am", M.AM), ("fm", M.FM)])
    def test_audio(self, kind, mod):
        x, msg = _analog(kind)
        d = demodulate(x, FS, mod, 0.0)
        assert d.audio is not None and d.audio_rate_hz == 8000
        assert len(d.audio) / d.audio_rate_hz == pytest.approx(1.5, abs=0.05)
        assert _audio_corr(d.audio, d.audio_rate_hz, msg) > 0.85
        assert d.bits.size == 0

    @pytest.mark.parametrize("side", ["usb", "lsb"])
    def test_ssb_exact_carrier(self, side):
        x, msg = _analog(side)
        d = demodulate_ssb(x, FS, side, carrier_hz=F0)
        assert _audio_corr(d.audio, d.audio_rate_hz, msg) > 0.95

    @pytest.mark.parametrize("side", ["usb", "lsb"])
    def test_ssb_blind_carrier(self, side):
        x, _ = _analog(side)
        sb = analyse_sideband(x, FS)
        assert sb.sideband == side
        assert sb.carrier_hz == pytest.approx(F0, abs=60)

    def test_audio_export(self, tmp_path):
        from scipy.io import wavfile

        x, _ = _analog("am")
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=FS))
        path = tmp_path / "am.wav"
        assert export_audio_wav(res, path) == pytest.approx(1.5, abs=0.05)
        rate, data = wavfile.read(path)
        assert rate == 8000 and data.dtype == np.int16 and len(data) > 10000

    def test_audio_export_requires_audio(self, tmp_path):
        x, _, _ = _digital("ook")
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=FS))
        with pytest.raises(ValueError):
            export_audio_wav(res, tmp_path / "x.wav")
