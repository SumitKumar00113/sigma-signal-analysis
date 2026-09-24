"""Classification + demodulation accuracy over all supported modulations.

Randomises seed, carrier offset and symbol rate (with a sample rate
giving 8–20 samples/symbol) and runs the full analysis pipeline, like the
GUI does.  Prints a per-type accuracy table and the demodulated BER (or
audio correlation for analog types).  Not a unit test; a development
harness.

Run:  python scripts/eval_classifier.py [trials_per_cell] [snr_db ...]

With a learned model installed (see ``python -m src.ml.train``; or point
``SIGMA_MODEL`` at a model file) the table also shows the accuracy of the
model alone and of the hybrid decision, measured on the same signals.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.enums import ModulationType as M  # noqa: E402
from src.core.models import RecordingMetadata  # noqa: E402
from src.dsp.demod import differential_decode_bits  # noqa: E402
from src.dsp.pipeline import AnalysisPipeline  # noqa: E402
from tests.signals import generators as sg  # noqa: E402
from tests.signals.metrics import ber_oqpsk, ber_with_ambiguity  # noqa: E402

# DBPSK is indistinguishable from BPSK on the air (the encoding is in the bits)
EQUIVALENT = {M.DPSK: {M.BPSK}, M.MSK: {M.FSK2}, M.GMSK: {M.GFSK}}


def _psk8(n, rs, fs, snr, f0):
    k = np.random.randint(0, 8, n)
    return sg._finish(sg._shape(np.exp(2j * np.pi * k / 8), int(fs / rs)), fs, snr, f0), None


def _qam64(n, rs, fs, snr, f0):
    lv = np.arange(-7, 8, 2)
    s = (lv[np.random.randint(0, 8, n)] + 1j * lv[np.random.randint(0, 8, n)]) / np.sqrt(42)
    return sg._finish(sg._shape(s, int(fs / rs)), fs, snr, f0), None


def _fsk4(n, rs, fs, snr, f0):
    lv = np.random.randint(0, 4, n)
    fr = np.repeat((2 * lv - 3) / 3.0 * 0.75 * rs, int(fs / rs))
    return sg._finish(np.exp(2j * np.pi * np.cumsum(fr) / fs), fs, snr, f0), None


def _fsk2(n, rs, fs, snr, f0):
    sig, bits = sg.generate_2fsk(n, rs, fs, rs * np.random.choice([0.4, 0.75, 1.5]))
    return sg._finish(sig, fs, snr, f0), bits


DIGITAL = {
    M.BPSK: (lambda n, rs, fs, snr, f0: sg.generate_bpsk(n, rs, fs, snr, f0), 1),
    M.QPSK: (lambda n, rs, fs, snr, f0: (sg._finish(sg.generate_qpsk(n, rs, fs)[0], fs, snr, f0),
                                         None), 2),
    M.PSK8: (_psk8, 3),
    M.QAM16: (lambda n, rs, fs, snr, f0: sg.generate_qam16(n, rs, fs, snr, f0), 4),
    M.QAM64: (_qam64, 6),
    M.FSK2: (_fsk2, 1),
    M.FSK4: (_fsk4, 2),
    M.MSK: (lambda n, rs, fs, snr, f0: sg.generate_msk(n, rs, fs, None, snr, f0), 1),
    M.GMSK: (lambda n, rs, fs, snr, f0: sg.generate_msk(n, rs, fs, 0.3, snr, f0), 1),
    M.OOK: (lambda n, rs, fs, snr, f0: sg.generate_ask(n, rs, fs, (0, 1), snr, f0), 1),
    M.ASK: (lambda n, rs, fs, snr, f0: sg.generate_ask(n, rs, fs, (0.25, 0.5, 0.75, 1.0),
                                                      snr, f0), 2),
    M.OQPSK: (lambda n, rs, fs, snr, f0: sg.generate_oqpsk(n, rs, fs, snr, f0), 2),
    M.PI4_DQPSK: (lambda n, rs, fs, snr, f0: sg.generate_pi4_dqpsk(n, rs, fs, snr, f0), 2),
    M.DPSK: (lambda n, rs, fs, snr, f0: sg.generate_dbpsk(n, rs, fs, snr, f0), 1),
}

ANALOG = {
    M.AM: lambda n, fs, snr, f0: sg.generate_am(n, fs, np.random.uniform(0.4, 0.9), None, snr, f0),
    M.FM: lambda n, fs, snr, f0: sg.generate_fm(n, fs, np.random.uniform(1500, 5000), None, snr,
                                               f0),
    M.SSB_USB: lambda n, fs, snr, f0: sg.generate_ssb(n, fs, "usb", None, snr, f0),
    M.SSB_LSB: lambda n, fs, snr, f0: sg.generate_ssb(n, fs, "lsb", None, snr, f0),
}


def _audio_corr(audio: np.ndarray, rate: float, msg: np.ndarray, fs: float) -> float:
    ref = sp_signal.resample_poly(msg, int(rate), int(fs)) if rate != fs else msg
    ref = sp_signal.filtfilt(sp_signal.firwin(257, 3400 / (rate / 2)), 1, ref)
    n = min(len(ref), len(audio))
    a, r = audio[:n] - np.mean(audio[:n]), ref[:n] - np.mean(ref[:n])
    lag = int(np.argmax(np.abs(sp_signal.correlate(a, r, mode="full")))) - (n - 1)
    a2, r2 = (a[lag:], r[: n - lag]) if lag >= 0 else (a[: n + lag], r[-lag:])
    return float(abs(np.corrcoef(a2, r2)[0, 1]))


def _match(got: M, truth: M) -> bool:
    return got == truth or got in EQUIVALENT.get(truth, set())


class _Tally:
    def __init__(self) -> None:
        self.n = 0
        self.rules = self.model = self.hybrid = 0
        self.has_model = False

    def add(self, res, truth: M) -> None:
        self.n += 1
        if res.classification is not None:
            self.rules += _match(res.classification.modulation, truth)
        if res.model_prediction is not None:
            self.has_model = True
            self.model += _match(res.model_prediction.modulation, truth)
        self.hybrid += _match(res.analysis.modulation, truth)

    def text(self) -> str:
        if not self.has_model:
            return ""
        return (f"  rules {self.rules / self.n:4.0%}  model {self.model / self.n:4.0%}  "
                f"hybrid {self.hybrid / self.n:4.0%}")


def run(trials: int, snrs: list[float]) -> None:
    rng = np.random.default_rng(0)
    for snr in snrs:
        print(f"\n=== SNR {snr:g} dB ({trials} trials each) ===")
        print(f"{'truth':11s} {'accuracy':>8s}  {'median BER / audio corr':>24s}  confusions")
        totals = _Tally()
        for truth, (gen, k) in DIGITAL.items():
            hits, quality, wrong, tally = 0, [], Counter(), _Tally()
            for _ in range(trials):
                np.random.seed(int(rng.integers(1 << 30)))
                rs = float(rng.choice([1200, 2400, 4800, 9600]))
                fs = rs * float(rng.choice([10, 16, 20]))
                f0 = float(rng.uniform(-0.1, 0.1) * fs)
                x, bits = gen(3000, rs, fs, snr, f0)
                res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=fs))
                tally.add(res, truth)
                totals.add(res, truth)
                got = res.analysis.modulation
                if _match(got, truth):
                    hits += 1
                    if bits is not None and res.demod is not None and len(res.demod.bits):
                        b = res.demod.bits
                        if truth == M.DPSK and got == M.BPSK:
                            # DBPSK read as BPSK: the data is in the phase changes
                            b = differential_decode_bits(b)
                        ber = (ber_oqpsk(b, bits) if truth == M.OQPSK
                               else ber_with_ambiguity(b, bits, k)[0])
                        quality.append(ber)
                else:
                    wrong[got.value] += 1
            q = f"{np.median(quality):.4f}" if quality else "—"
            print(f"{truth.value:11s} {hits / trials:8.0%}  {q:>24s}  {dict(wrong)}{tally.text()}")
        for truth, agen in ANALOG.items():
            hits, quality, wrong, tally = 0, [], Counter(), _Tally()
            for _ in range(trials):
                np.random.seed(int(rng.integers(1 << 30)))
                fs = float(rng.choice([24000, 48000]))
                f0 = float(rng.uniform(-0.1, 0.1) * fs)
                x, msg = agen(int(1.5 * fs), fs, snr, f0)
                res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=fs))
                tally.add(res, truth)
                totals.add(res, truth)
                got = res.analysis.modulation
                if got == truth:
                    hits += 1
                    d = res.demod
                    if d is not None and d.audio is not None and truth in (M.AM, M.FM):
                        quality.append(_audio_corr(d.audio, d.audio_rate_hz, msg, fs))
                else:
                    wrong[got.value] += 1
            q = f"corr {np.median(quality):.2f}" if quality else "—"
            print(f"{truth.value:11s} {hits / trials:8.0%}  {q:>24s}  {dict(wrong)}{tally.text()}")
        if totals.has_model:
            print(f"{'ALL':11s} {totals.hybrid / totals.n:8.1%}  {'':24s}  {totals.text()}")


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    levels = [float(v) for v in sys.argv[2:]] or [15.0, 8.0]
    run(n_trials, levels)
