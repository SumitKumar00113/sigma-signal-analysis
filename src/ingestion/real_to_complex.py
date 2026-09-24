"""Turning real-valued recordings into complex samples.

A mono WAV (receiver audio, an HF SSB/modem channel, a real IF) holds a
*real* signal, whose spectrum is Hermitian: every component at +f has a
mirror at −f.  Storing it as ``I = x, Q = 0`` keeps both images, so the
complex-baseband analysis downstream sees two copies of each signal,
a doubled occupied bandwidth, a centroid "carrier offset" of ~0 Hz, a
constellation that collapses onto the real axis, and a huge "IQ
imbalance".

The fix is the **analytic signal** ``x + j·H{x}`` (H = Hilbert
transform), which keeps only the positive frequencies: a tone at
1500 Hz in the audio becomes a single complex tone at +1500 Hz, and a
passband PSK/QAM/FSK modem signal becomes a proper complex signal at its
audio carrier, which the demodulators then treat like any carrier offset.

The Hilbert transform is a windowed-ideal FIR (type III, odd length) so
that chunked reads give *exactly* the same samples as converting the
whole file at once (the FFT method's ~1/n kernel leaks across chunks).

Also here: FM re-modulation for *discriminator* audio (where the audio is
the instantaneous frequency of the original signal) and a heuristic that
suggests how a WAV file's channels should be interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import signal as sp_signal

from src.core.enums import WavInterpretation

#: Default Hilbert FIR length: flat to ±0.1 dB and > 50 dB image rejection
#: between ~1.5 % and 48.5 % of the sample rate.
DEFAULT_HILBERT_TAPS = 255


@lru_cache(maxsize=8)
def hilbert_fir(num_taps: int = DEFAULT_HILBERT_TAPS, beta: float = 8.0) -> np.ndarray:
    """Kaiser-windowed ideal Hilbert transformer (odd *num_taps*)."""
    if num_taps % 2 == 0 or num_taps < 3:
        raise ValueError("num_taps must be odd and ≥ 3")
    m = np.arange(num_taps) - (num_taps - 1) // 2
    h = np.zeros(num_taps)
    odd = m % 2 != 0
    h[odd] = 2.0 / (np.pi * m[odd])
    h *= np.kaiser(num_taps, beta)
    h.setflags(write=False)
    return h


def hilbert_margin(num_taps: int = DEFAULT_HILBERT_TAPS) -> int:
    """Samples of context needed on each side of a chunk."""
    return (num_taps - 1) // 2


def analytic_from_padded(padded: np.ndarray, num_taps: int = DEFAULT_HILBERT_TAPS) -> np.ndarray:
    """Analytic signal of the centre of *padded*.

    *padded* must carry :func:`hilbert_margin` extra samples on both
    sides (zeros at file edges); the result has ``len(padded) − 2·margin``
    samples, aligned with the un-padded centre.
    """
    x = np.asarray(padded, dtype=np.float64).reshape(-1)
    d = hilbert_margin(num_taps)
    if len(x) <= 2 * d:
        return np.zeros(0, dtype=np.complex64)
    q = sp_signal.fftconvolve(x, hilbert_fir(num_taps), mode="valid")
    return (x[d: len(x) - d] + 1j * q).astype(np.complex64)


def analytic_signal(x: np.ndarray, num_taps: int = DEFAULT_HILBERT_TAPS) -> np.ndarray:
    """Analytic signal of a whole real array (zero context at both ends)."""
    d = hilbert_margin(num_taps)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return analytic_from_padded(np.pad(x, d), num_taps)


def padded_slice(x: np.ndarray, start: int, end: int, margin: int) -> np.ndarray:
    """``x[start−margin : end+margin]`` with zeros beyond the array ends."""
    lo, hi = start - margin, end + margin
    seg = x[max(lo, 0): min(hi, len(x))]
    return np.pad(seg, (max(0, -lo), max(0, hi - len(x))))


# ---------------------------------------------------------------------------
# Discriminator audio
# ---------------------------------------------------------------------------

#: Peak deviation used when re-modulating discriminator audio, as a
#: fraction of the sample rate.  The true deviation is unknown (it depends
#: on receiver gain), so only the *shape* of the frequency track matters.
DISCRIMINATOR_PEAK_DEVIATION = 0.125


def discriminator_phase(x: np.ndarray, peak_deviation: float = DISCRIMINATOR_PEAK_DEVIATION
                        ) -> np.ndarray:
    """Cumulative phase of an FM signal whose instantaneous frequency is *x*.

    The audio is DC-free by construction of most discriminators, so the
    mean is kept (it is a genuine frequency offset); it is scaled so its
    robust peak (99.9th percentile) maps to *peak_deviation*·fs.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    peak = float(np.percentile(np.abs(x), 99.9)) if len(x) else 0.0
    if peak <= 0:
        return np.zeros(len(x))
    return np.cumsum(2.0 * np.pi * peak_deviation * x / peak)


def fm_remodulate(x: np.ndarray, peak_deviation: float = DISCRIMINATOR_PEAK_DEVIATION
                  ) -> np.ndarray:
    return np.exp(1j * discriminator_phase(x, peak_deviation)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Interpretation heuristic
# ---------------------------------------------------------------------------


@dataclass
class InterpretationSuggestion:
    interpretation: WavInterpretation
    reason: str
    confidence: float           # 0..1


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / den) if den > 0 else 0.0


def suggest_wav_interpretation(data: np.ndarray, sample_rate: float = 1.0,
                               max_samples: int = 200_000) -> InterpretationSuggestion:
    """Guess how the channels of *data* (samples × channels, float) map to
    complex samples.

    * 1 channel → real signal (analytic conversion).
    * 2 identical channels → mono duplicated to stereo.
    * 2 channels whose Hilbert transforms line up (Q ≈ ±H{I}) and with
      matching power spectra → I/Q.
    * 2 strongly correlated but not quadrature channels → dual channel.
    * otherwise 2 channels default to I/Q (the common SDR convention),
      with low confidence.
    """
    d = np.asarray(data, dtype=np.float64)
    if d.ndim == 1:
        d = d[:, None]
    d = d[:max_samples]
    if d.shape[1] == 1:
        return InterpretationSuggestion(WavInterpretation.REAL,
                                        "Mono file: real signal, converted to analytic IQ.", 0.9)
    left, right = d[:, 0], d[:, 1]
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-12)
    if np.max(np.abs(left - right)) <= 1e-4 * scale:
        return InterpretationSuggestion(WavInterpretation.REAL,
                                        "Both channels identical: mono audio saved as stereo.",
                                        0.95)
    if float(np.std(right)) <= 1e-4 * scale:
        return InterpretationSuggestion(WavInterpretation.REAL,
                                        "Second channel silent: treating the first as mono.", 0.8)

    rho0 = _norm_corr(left, right)
    rho_q = _norm_corr(np.imag(analytic_signal(left)), right)

    nper = min(1024, len(left))
    _, p_l = sp_signal.welch(left, fs=sample_rate, nperseg=nper)
    _, p_r = sp_signal.welch(right, fs=sample_rate, nperseg=nper)
    shape = _norm_corr(np.log10(p_l + 1e-20), np.log10(p_r + 1e-20))
    power_db = 10 * np.log10((np.sum(p_l) + 1e-20) / (np.sum(p_r) + 1e-20))

    if abs(rho_q) > 0.7:
        return InterpretationSuggestion(
            WavInterpretation.STEREO_IQ,
            f"Channels are in quadrature (|corr(H{{L}}, R)| = {abs(rho_q):.2f}): I/Q.", 0.9)
    if abs(rho0) > 0.7:
        return InterpretationSuggestion(
            WavInterpretation.DUAL_CHANNEL,
            f"Channels strongly correlated in phase (corr = {rho0:.2f}): two real sensor "
            "channels; the first is analysed.", 0.7)
    if shape > 0.9 and abs(power_db) < 1.5:
        return InterpretationSuggestion(
            WavInterpretation.STEREO_IQ,
            "Channels have matching power spectra and are uncorrelated: I/Q.", 0.75)
    return InterpretationSuggestion(
        WavInterpretation.STEREO_IQ,
        "Two channels with no clear relation; assuming the usual I/Q convention.", 0.4)
