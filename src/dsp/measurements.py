"""Signal measurement utilities.

Measures specific properties of a detected signal like SNR,
frequency offset, and symbol rate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def estimate_snr(samples: np.ndarray) -> float:
    """Estimate SNR of a signal assuming stationary AWGN.
    
    This is a simplistic M2M4 (2nd & 4th moment) estimator.
    It works best for constant-envelope modulations (PSK, FSK).
    """
    if len(samples) == 0:
        return 0.0
        
    m2 = np.mean(np.abs(samples) ** 2)
    m4 = np.mean(np.abs(samples) ** 4)
    
    # Prevent math domain errors on pure noise or empty signals
    if m2 == 0 or m4 >= 2 * (m2 ** 2):
        return 0.0
        
    # M2M4 estimator for SNR
    # snr = sqrt(2*m2^2 - m4) / (m2 - sqrt(2*m2^2 - m4))
    sqrt_term = np.sqrt(max(0, 2 * m2**2 - m4))
    noise_power = m2 - sqrt_term
    
    if noise_power <= 0:
        return 100.0  # Cap at 100 dB for practically noiseless
        
    signal_power = sqrt_term
    snr_linear = signal_power / noise_power
    
    if snr_linear <= 0:
        return 0.0
        
    return float(10 * np.log10(snr_linear))


def estimate_frequency_offset(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096
) -> float:
    """Estimate the frequency offset from DC (0 Hz).
    
    Finds the center of mass (spectral centroid) of the dominant peak.
    """
    if len(samples) < fft_size:
        fft_size = len(samples)
        
    if fft_size < 16:
        return 0.0
        
    window = sp_signal.windows.hann(fft_size)
    # Average multiple FFTs if we have enough data
    num_blocks = len(samples) // fft_size
    if num_blocks == 0:
        num_blocks = 1
        
    psd_sum = np.zeros(fft_size)
    for i in range(num_blocks):
        chunk = samples[i*fft_size : (i+1)*fft_size]
        if len(chunk) < fft_size:
            chunk = np.pad(chunk, (0, fft_size - len(chunk)))
        fft_out = np.fft.fftshift(np.fft.fft(chunk * window))
        psd_sum += np.abs(fft_out) ** 2
        
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / sample_rate))
    
    # Find peak and calculate centroid around it
    peak_idx = np.argmax(psd_sum)
    
    # Calculate centroid using +/- 5% of bins around peak
    window_bins = max(3, int(fft_size * 0.05))
    start_idx = max(0, peak_idx - window_bins)
    end_idx = min(fft_size, peak_idx + window_bins)
    
    weights = psd_sum[start_idx:end_idx]
    f_window = freqs[start_idx:end_idx]
    
    if np.sum(weights) == 0:
        return float(freqs[peak_idx])
        
    centroid = np.sum(f_window * weights) / np.sum(weights)
    return float(centroid)


def estimate_symbol_rate_candidates(
    samples: np.ndarray,
    sample_rate: float
) -> list[tuple[float, float]]:
    """Estimate potential symbol rates.
    
    Uses AM delay-and-multiply followed by FFT to find symbol rate lines.
    Returns a list of (rate_hz, confidence) tuples.
    """
    if len(samples) < 1024:
        return []
        
    # Non-linear transformation to expose symbol rate features
    # |x(t)| applies to PSK/QAM due to pulse shaping
    # x(t)*conj(x(t-1)) applies to FSK/PSK
    
    # 1. Amplitude envelope method
    amp = np.abs(samples)
    amp_ac = amp - np.mean(amp)
    
    fft_size = min(65536, 2**int(np.log2(len(amp_ac))))
    window = sp_signal.windows.blackman(fft_size)
    
    # Take first block
    chunk = amp_ac[:fft_size]
    fft_mag = np.abs(np.fft.fft(chunk * window))
    
    # Only look at positive frequencies up to Nyquist
    half_size = fft_size // 2
    fft_mag = fft_mag[:half_size]
    freqs = np.fft.fftfreq(fft_size, 1 / sample_rate)[:half_size]
    
    # Ignore DC and very low frequencies
    min_idx = int(0.01 * half_size)
    fft_mag[:min_idx] = 0
    
    # Find peaks
    peaks, props = sp_signal.find_peaks(fft_mag, distance=fft_size//100, prominence=np.max(fft_mag)*0.1)
    
    candidates = []
    for idx in peaks:
        freq = float(freqs[idx])
        # Prominence relative to max as a rough confidence
        confidence = min(1.0, float(props['prominences'][np.where(peaks == idx)[0][0]] / np.max(fft_mag)))
        if freq > 100.0:  # Ignore very low symbol rates
            candidates.append((freq, confidence))
            
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:5]


def compute_instantaneous_frequency(
    samples: np.ndarray,
    sample_rate: float
) -> np.ndarray:
    """Compute the instantaneous frequency of the analytic signal."""
    phase = np.unwrap(np.angle(samples))
    # Derivative of phase
    inst_freq = np.diff(phase) / (2.0 * np.pi) * sample_rate
    # Pad to maintain original length
    return np.append(inst_freq, inst_freq[-1])


def compute_instantaneous_amplitude(samples: np.ndarray) -> np.ndarray:
    """Compute the instantaneous amplitude (envelope) of the signal."""
    return np.abs(samples)


def compute_instantaneous_phase(samples: np.ndarray) -> np.ndarray:
    """Compute the unwrapped instantaneous phase of the signal."""
    return np.unwrap(np.angle(samples))
