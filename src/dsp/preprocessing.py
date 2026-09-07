"""Signal preprocessing utilities.

Covers DC offset removal, gain normalization, and frequency translation
as defined in PRD §10.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# DC offset removal
# ---------------------------------------------------------------------------

def remove_dc_mean(samples: np.ndarray) -> np.ndarray:
    """Remove DC offset by subtracting the sample mean."""
    return (samples - np.mean(samples)).astype(np.complex64)


def remove_dc_highpass(
    samples: np.ndarray,
    sample_rate: float,
    cutoff_hz: float = 100.0,
) -> np.ndarray:
    """Remove DC with a first-order high-pass IIR filter.

    Parameters
    ----------
    cutoff_hz : 3-dB cutoff frequency of the HPF.
    """
    # Design a Butterworth HPF (1st order for minimal distortion)
    nyq = sample_rate / 2.0
    if cutoff_hz >= nyq:
        cutoff_hz = nyq * 0.01  # safety clamp
    b, a = sp_signal.butter(1, cutoff_hz / nyq, btype="high")
    # Filter I and Q independently to preserve complex structure
    out_i = sp_signal.lfilter(b, a, samples.real).astype(np.float32)
    out_q = sp_signal.lfilter(b, a, samples.imag).astype(np.float32)
    return (out_i + 1j * out_q).astype(np.complex64)


# ---------------------------------------------------------------------------
# Gain normalization
# ---------------------------------------------------------------------------

def normalize_rms(samples: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    """Scale samples so that the RMS amplitude equals *target_rms*."""
    rms = np.sqrt(np.mean(np.abs(samples) ** 2))
    if rms == 0:
        return samples
    return (samples * (target_rms / rms)).astype(np.complex64)


def normalize_peak(samples: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    """Scale samples so that the peak amplitude equals *target_peak*."""
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    return (samples * (target_peak / peak)).astype(np.complex64)


def normalize_percentile(
    samples: np.ndarray,
    percentile: float = 99.0,
    target: float = 1.0,
) -> np.ndarray:
    """Scale so that the given amplitude percentile maps to *target*."""
    p = np.percentile(np.abs(samples), percentile)
    if p == 0:
        return samples
    return (samples * (target / p)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Frequency translation
# ---------------------------------------------------------------------------

def translate_frequency(
    samples: np.ndarray,
    sample_rate: float,
    shift_hz: float,
) -> np.ndarray:
    """Shift the signal by *shift_hz* using a complex exponential mixer.

    A positive *shift_hz* moves the spectrum up; negative moves it down.
    """
    n = np.arange(len(samples), dtype=np.float64)
    mixer = np.exp(1j * 2.0 * np.pi * shift_hz / sample_rate * n)
    return (samples * mixer).astype(np.complex64)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def apply_lowpass(
    samples: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
    order: int = 64,
) -> np.ndarray:
    """Apply a FIR low-pass filter.

    Parameters
    ----------
    cutoff_hz : 3-dB cutoff in Hz.
    order : FIR filter order (number of taps - 1).
    """
    nyq = sample_rate / 2.0
    taps = sp_signal.firwin(order + 1, cutoff_hz / nyq)
    out_i = sp_signal.lfilter(taps, 1.0, samples.real).astype(np.float32)
    out_q = sp_signal.lfilter(taps, 1.0, samples.imag).astype(np.float32)
    return (out_i + 1j * out_q).astype(np.complex64)


def apply_bandpass(
    samples: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 128,
) -> np.ndarray:
    """Apply a FIR band-pass filter."""
    nyq = sample_rate / 2.0
    taps = sp_signal.firwin(order + 1, [low_hz / nyq, high_hz / nyq], pass_zero=False)
    out_i = sp_signal.lfilter(taps, 1.0, samples.real).astype(np.float32)
    out_q = sp_signal.lfilter(taps, 1.0, samples.imag).astype(np.float32)
    return (out_i + 1j * out_q).astype(np.complex64)
