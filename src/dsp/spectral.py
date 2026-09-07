"""Spectral analysis utilities.

Provides FFT, PSD (Welch), spectrogram computation, peak detection,
noise-floor estimation, and occupied-bandwidth measurement.  All functions
operate on 1-D complex64 sample arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# FFT / PSD
# ---------------------------------------------------------------------------

def compute_fft(
    samples: np.ndarray,
    fft_size: int = 4096,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a single windowed FFT and return (frequencies_norm, magnitude_dB).

    Parameters
    ----------
    samples : complex64 array (length >= *fft_size*).
    fft_size : FFT length in samples.
    window : SciPy window name.

    Returns
    -------
    freq_bins : 1-D float array of normalised frequencies in [-0.5, 0.5).
    mag_db    : 1-D float array of magnitudes in dBFS.
    """
    seg = samples[:fft_size]
    win = sp_signal.get_window(window, len(seg))
    windowed = seg * win
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    mag = np.abs(spectrum)
    mag[mag == 0] = 1e-30  # avoid log(0)
    mag_db = 20.0 * np.log10(mag / fft_size).astype(np.float32)
    freq_bins = np.fft.fftshift(np.fft.fftfreq(fft_size)).astype(np.float32)
    return freq_bins, mag_db


def compute_psd(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
    window: str = "hann",
    overlap_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density via Welch's method.

    Returns
    -------
    freqs : 1-D array of frequencies in Hz.
    psd   : 1-D array of PSD values in dB/Hz (relative to full-scale).
    """
    noverlap = int(fft_size * overlap_frac)
    freqs, pxx = sp_signal.welch(
        samples,
        fs=sample_rate,
        window=window,
        nperseg=fft_size,
        noverlap=noverlap,
        return_onesided=False,
        detrend=False,
    )
    # Shift so DC is in the centre
    freqs = np.fft.fftshift(freqs)
    pxx = np.fft.fftshift(pxx)

    pxx[pxx == 0] = 1e-30
    psd_db = 10.0 * np.log10(pxx).astype(np.float32)
    return freqs.astype(np.float32), psd_db


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 1024,
    window: str = "hann",
    overlap_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a time-frequency spectrogram.

    Returns
    -------
    times : 1-D array of time values (seconds).
    freqs : 1-D array of frequency values (Hz).
    sxx_db : 2-D array [freq × time] of power in dB.
    """
    noverlap = int(fft_size * overlap_frac)
    freqs, times, sxx = sp_signal.spectrogram(
        samples,
        fs=sample_rate,
        window=window,
        nperseg=fft_size,
        noverlap=noverlap,
        return_onesided=False,
        detrend=False,
        mode="psd",
    )
    freqs = np.fft.fftshift(freqs)
    sxx = np.fft.fftshift(sxx, axes=0)

    sxx[sxx == 0] = 1e-30
    sxx_db = 10.0 * np.log10(sxx).astype(np.float32)
    return times.astype(np.float32), freqs.astype(np.float32), sxx_db


# ---------------------------------------------------------------------------
# Noise floor & peaks
# ---------------------------------------------------------------------------

@dataclass
class SpectralPeak:
    """A detected peak in the spectrum."""
    frequency_hz: float = 0.0
    power_db: float = 0.0
    bandwidth_hz: float = 0.0
    snr_db: float = 0.0


@dataclass
class SpectralAnalysis:
    """Result of a full spectral analysis on a sample segment."""
    noise_floor_db: float = 0.0
    peaks: list[SpectralPeak] = field(default_factory=list)
    occupied_bandwidth_hz: float = 0.0
    total_power_db: float = 0.0
    dc_power_db: float = 0.0


def estimate_noise_floor(psd_db: np.ndarray, percentile: float = 25.0) -> float:
    """Estimate noise floor as a low percentile of the PSD."""
    return float(np.percentile(psd_db, percentile))


def detect_peaks(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    noise_floor_db: float,
    threshold_db: float = 6.0,
    min_distance_bins: int = 10,
) -> list[SpectralPeak]:
    """Find spectral peaks above the noise floor.

    Parameters
    ----------
    threshold_db : minimum dB above noise floor to qualify as a peak.
    min_distance_bins : minimum separation between peaks in FFT bins.
    """
    peaks: list[SpectralPeak] = []

    # Simple local-max search
    indices, _ = sp_signal.find_peaks(
        psd_db,
        height=noise_floor_db + threshold_db,
        distance=min_distance_bins,
    )

    freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    for idx in indices:
        # Estimate 3-dB bandwidth around peak
        peak_power = float(psd_db[idx])
        half_power = peak_power - 3.0
        left = idx
        while left > 0 and psd_db[left] > half_power:
            left -= 1
        right = idx
        while right < len(psd_db) - 1 and psd_db[right] > half_power:
            right += 1
        bw = (right - left) * abs(freq_res)

        peaks.append(SpectralPeak(
            frequency_hz=float(freqs[idx]),
            power_db=peak_power,
            bandwidth_hz=bw,
            snr_db=peak_power - noise_floor_db,
        ))

    # Sort by power descending
    peaks.sort(key=lambda p: p.power_db, reverse=True)
    return peaks


def estimate_occupied_bandwidth(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    fraction: float = 0.99,
) -> float:
    """Estimate occupied bandwidth containing *fraction* of total power.

    Uses the 99 % power containment method.
    """
    psd_linear = 10.0 ** (psd_db / 10.0)
    total = np.sum(psd_linear)
    if total == 0:
        return 0.0

    cumulative = np.cumsum(psd_linear)

    lower = 0
    upper = len(cumulative) - 1
    for i in range(len(cumulative)):
        if cumulative[i] >= (1 - fraction) / 2 * total:
            lower = i
            break
    for i in range(len(cumulative) - 1, -1, -1):
        if cumulative[i] <= (1 + fraction) / 2 * total:
            upper = i
            break

    return float(abs(freqs[upper] - freqs[lower]))


def analyze_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
    window: str = "hann",
) -> SpectralAnalysis:
    """Run a full spectral analysis on a sample segment.

    Computes PSD, noise floor, peaks, and occupied bandwidth.
    """
    freqs, psd_db = compute_psd(samples, sample_rate, fft_size, window)
    noise = estimate_noise_floor(psd_db)
    peaks = detect_peaks(freqs, psd_db, noise)
    obw = estimate_occupied_bandwidth(freqs, psd_db)

    psd_linear = 10.0 ** (psd_db / 10.0)
    total_power = 10.0 * np.log10(np.sum(psd_linear)) if np.any(psd_linear > 0) else -100.0

    return SpectralAnalysis(
        noise_floor_db=noise,
        peaks=peaks,
        occupied_bandwidth_hz=obw,
        total_power_db=float(total_power),
    )
