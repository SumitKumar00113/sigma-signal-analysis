"""Analog demodulation: AM, FM and SSB → audio.

* **AM** – envelope detector: ``|x| / mean(|x|) − 1`` is the message scaled
  by the modulation index, so the index is measured too.
* **FM** – frequency discriminator on the band-limited signal; the
  output is in Hz, so the peak deviation is measured too.
* **SSB** – product detector: the suppressed carrier is placed at DC and
  the real part taken.  The carrier is not transmitted, so it is inferred
  from the occupied band's edge nearest the carrier, assuming the usual
  voice channel that starts :data:`SSB_AUDIO_LOW_EDGE_HZ` above it.  An
  error there shifts the recovered audio in pitch but keeps it intelligible.

Recovered audio is low-pass filtered to at most :data:`MAX_AUDIO_BANDWIDTH_HZ`
and resampled to :data:`AUDIO_RATE_HZ` when the input rate is higher.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as sp_signal

from src.core.enums import ModulationType
from src.core.exceptions import DemodulationError, InsufficientSamplesError
from src.dsp.demod import DemodResult
from src.dsp.measurements import (
    bandlimit_to_signal,
    compute_instantaneous_frequency,
    estimate_frequency_offset,
    estimate_snr_inband,
)
from src.dsp.preprocessing import translate_frequency
from src.dsp.spectral import compute_psd, estimate_noise_floor

AUDIO_RATE_HZ = 8000.0
MAX_AUDIO_BANDWIDTH_HZ = 4000.0
#: Lowest audio frequency of a standard SSB voice channel above the carrier
SSB_AUDIO_LOW_EDGE_HZ = 300.0


def _to_audio(x: np.ndarray, sample_rate: float, bandwidth_hz: float) -> tuple[np.ndarray, float]:
    """Low-pass a real message and bring it to the audio rate."""
    x = np.asarray(x, dtype=np.float64)
    bw = float(np.clip(bandwidth_hz, 200.0, min(MAX_AUDIO_BANDWIDTH_HZ, 0.45 * sample_rate)))
    taps = sp_signal.firwin(257, bw / (sample_rate / 2.0))
    y = sp_signal.filtfilt(taps, 1.0, x) if len(x) > 3 * len(taps) else x
    rate = sample_rate
    if sample_rate > 2 * AUDIO_RATE_HZ:
        frac = Fraction(AUDIO_RATE_HZ / sample_rate).limit_denominator(1000)
        y = sp_signal.resample_poly(y, frac.numerator, frac.denominator)
        rate = sample_rate * frac.numerator / frac.denominator
    y = y - np.mean(y)
    return y.astype(np.float32), float(rate)


def _complex_lowpass(x: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase complex low-pass (no carrier-phase or delay error)."""
    cutoff = float(np.clip(cutoff_hz, 50.0, 0.49 * sample_rate))
    taps = sp_signal.firwin(257, cutoff / (sample_rate / 2.0))
    if len(x) <= 3 * len(taps):
        return np.asarray(x, dtype=np.complex128)
    return sp_signal.filtfilt(taps, 1.0, np.asarray(x, dtype=np.complex128))


def _check_length(samples: np.ndarray, sample_rate: float) -> None:
    if len(samples) < int(0.05 * sample_rate) or len(samples) < 1024:
        raise InsufficientSamplesError("Need at least 50 ms of signal for analog demodulation.")


def demodulate_am(samples: np.ndarray, sample_rate: float,
                  coarse_cfo_hz: float | None = None) -> DemodResult:
    _check_length(samples, sample_rate)
    if coarse_cfo_hz is None:
        coarse_cfo_hz = estimate_frequency_offset(samples, sample_rate)
    # Both sidebands of a voice channel fit in ±MAX_AUDIO_BANDWIDTH_HZ
    x = _complex_lowpass(translate_frequency(samples, sample_rate, -coarse_cfo_hz),
                         sample_rate, MAX_AUDIO_BANDWIDTH_HZ)
    env = np.abs(x).astype(np.float64)
    mean = float(np.mean(env))
    if mean <= 0:
        raise DemodulationError("Empty envelope.")
    audio, rate = _to_audio(env / mean - 1.0, sample_rate, MAX_AUDIO_BANDWIDTH_HZ)
    index = float(np.percentile(np.abs(audio), 99.9))
    return DemodResult(
        modulation=ModulationType.AM, symbols=np.zeros(0, dtype=np.complex64),
        bits=np.zeros(0, dtype=np.uint8), symbol_rate_hz=0.0, samples_per_symbol=0.0,
        bits_per_symbol=0, coarse_cfo_hz=float(coarse_cfo_hz), audio=audio,
        audio_rate_hz=rate, carrier_hz=float(coarse_cfo_hz),
        warnings=[f"Envelope detector; peak modulation index ≈ {index:.2f}."],
    )


def demodulate_fm(samples: np.ndarray, sample_rate: float,
                  coarse_cfo_hz: float | None = None) -> DemodResult:
    _check_length(samples, sample_rate)
    if coarse_cfo_hz is None:
        coarse_cfo_hz = estimate_frequency_offset(samples, sample_rate)
    x = bandlimit_to_signal(translate_frequency(samples, sample_rate, -coarse_cfo_hz),
                            sample_rate)
    inst = compute_instantaneous_frequency(x, sample_rate).astype(np.float64)
    _, occupied = estimate_snr_inband(x, sample_rate)
    audio_hz, rate = _to_audio(inst, sample_rate, occupied / 2.0)
    peak_dev = float(np.percentile(np.abs(audio_hz), 99.9))
    audio = audio_hz / (peak_dev + 1e-12)
    return DemodResult(
        modulation=ModulationType.FM, symbols=np.zeros(0, dtype=np.complex64),
        bits=np.zeros(0, dtype=np.uint8), symbol_rate_hz=0.0, samples_per_symbol=0.0,
        bits_per_symbol=0, coarse_cfo_hz=float(coarse_cfo_hz), audio=audio.astype(np.float32),
        audio_rate_hz=rate, carrier_hz=float(coarse_cfo_hz),
        warnings=[f"Frequency discriminator; peak deviation ≈ {peak_dev:,.0f} Hz."],
    )


# ---------------------------------------------------------------------------
# SSB
# ---------------------------------------------------------------------------


@dataclass
class SidebandAnalysis:
    low_edge_hz: float
    high_edge_hz: float
    position: float          # power centroid within [low, high]: 0 = low edge, 1 = high
    sideband: str            # "usb", "lsb" or "symmetric"

    @property
    def carrier_hz(self) -> float:
        if self.sideband == "lsb":
            return self.high_edge_hz + SSB_AUDIO_LOW_EDGE_HZ
        return self.low_edge_hz - SSB_AUDIO_LOW_EDGE_HZ


def analyse_sideband(samples: np.ndarray, sample_rate: float,
                     asymmetry_threshold: float = 0.08) -> SidebandAnalysis:
    """Occupied band edges and where the power sits within the band.

    Voice energy falls with audio frequency, so an upper sideband has its
    power near the *low* edge (the carrier side) and a lower sideband near
    the high edge.  Symmetric spectra (AM, DSB, PSK/QAM) sit near 0.5.
    """
    freqs, psd_db = compute_psd(samples, sample_rate, fft_size=min(4096, len(samples)))
    floor = estimate_noise_floor(psd_db, percentile=20.0)
    occ = np.flatnonzero(psd_db > floor + 10.0)
    if len(occ) < 3:
        return SidebandAnalysis(0.0, 0.0, 0.5, "symmetric")
    # Largest contiguous occupied segment
    breaks = np.flatnonzero(np.diff(occ) > 2)
    segs = np.split(occ, breaks + 1)
    seg = max(segs, key=lambda sg: float(np.sum(10 ** (psd_db[sg] / 10))))
    lo, hi = float(freqs[seg[0]]), float(freqs[seg[-1]])
    p = 10 ** (psd_db[seg] / 10) - 10 ** (floor / 10)
    p = np.clip(p, 0, None)
    centroid = float(np.sum(freqs[seg] * p) / (np.sum(p) + 1e-30))
    pos = (centroid - lo) / (hi - lo) if hi > lo else 0.5
    if pos < 0.5 - asymmetry_threshold:
        side = "usb"
    elif pos > 0.5 + asymmetry_threshold:
        side = "lsb"
    else:
        side = "symmetric"
    return SidebandAnalysis(lo, hi, float(pos), side)


def demodulate_ssb(samples: np.ndarray, sample_rate: float, sideband: str = "usb",
                   carrier_hz: float | None = None) -> DemodResult:
    _check_length(samples, sample_rate)
    sb = analyse_sideband(samples, sample_rate)
    if carrier_hz is None:
        if sb.sideband == "symmetric":
            sb.sideband = sideband
        carrier_hz = sb.carrier_hz if sb.sideband == sideband else (
            sb.low_edge_hz - SSB_AUDIO_LOW_EDGE_HZ if sideband == "usb"
            else sb.high_edge_hz + SSB_AUDIO_LOW_EDGE_HZ)
    width = float(np.clip(sb.high_edge_hz - sb.low_edge_hz + SSB_AUDIO_LOW_EDGE_HZ,
                          1000.0, MAX_AUDIO_BANDWIDTH_HZ))
    # Keep only the wanted sideband ([0, width] for USB, [−width, 0] for LSB)
    # with a zero-phase filter, so no phase error or opposite-sideband noise
    # reaches the product detector
    sign = 1.0 if sideband == "usb" else -1.0
    base = translate_frequency(samples, sample_rate, -(carrier_hz + sign * width / 2.0))
    base = _complex_lowpass(base, sample_rate, width / 2.0)
    base = translate_frequency(base, sample_rate, sign * width / 2.0)
    audio, rate = _to_audio(np.real(base), sample_rate, width)
    audio = audio / (np.percentile(np.abs(audio), 99.9) + 1e-12)
    mod = ModulationType.SSB_USB if sideband == "usb" else ModulationType.SSB_LSB
    return DemodResult(
        modulation=mod, symbols=np.zeros(0, dtype=np.complex64),
        bits=np.zeros(0, dtype=np.uint8), symbol_rate_hz=0.0, samples_per_symbol=0.0,
        bits_per_symbol=0, coarse_cfo_hz=float(carrier_hz), audio=audio.astype(np.float32),
        audio_rate_hz=rate, carrier_hz=float(carrier_hz),
        warnings=[f"Product detector, {sideband.upper()}; suppressed carrier assumed at "
                  f"{carrier_hz:,.0f} Hz ({SSB_AUDIO_LOW_EDGE_HZ:.0f} Hz beyond the band "
                  "edge). A tuning error shifts the audio pitch."],
    )


def demodulate_analog(samples: np.ndarray, sample_rate: float, modulation: ModulationType,
                      coarse_cfo_hz: float | None = None,
                      carrier_hz: float | None = None) -> DemodResult:
    if modulation == ModulationType.AM:
        return demodulate_am(samples, sample_rate, coarse_cfo_hz)
    if modulation == ModulationType.FM:
        return demodulate_fm(samples, sample_rate, coarse_cfo_hz)
    if modulation == ModulationType.SSB_USB:
        return demodulate_ssb(samples, sample_rate, "usb", carrier_hz)
    if modulation == ModulationType.SSB_LSB:
        return demodulate_ssb(samples, sample_rate, "lsb", carrier_hz)
    raise DemodulationError(f"{modulation.value} is not an analog modulation.")
