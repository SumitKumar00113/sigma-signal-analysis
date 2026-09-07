"""Channel impairment models."""

from __future__ import annotations

import numpy as np


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Additive White Gaussian Noise to a complex signal."""
    if len(signal) == 0:
        return signal
        
    # Calculate signal power
    sig_power = np.mean(np.abs(signal) ** 2)
    
    # Calculate noise power
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    # Generate complex noise
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    return (signal + noise).astype(np.complex64)


def add_frequency_offset(
    signal: np.ndarray,
    sample_rate: float,
    offset_hz: float
) -> np.ndarray:
    """Add a constant frequency offset (carrier error)."""
    t = np.arange(len(signal), dtype=np.float64) / sample_rate
    mixer = np.exp(1j * 2 * np.pi * offset_hz * t)
    return (signal * mixer).astype(np.complex64)


def add_iq_imbalance(
    signal: np.ndarray,
    amplitude_imbalance_db: float = 0.0,
    phase_imbalance_deg: float = 0.0
) -> np.ndarray:
    """Inject IQ amplitude and phase imbalance."""
    # Convert amplitude imbalance (dB) to linear scale difference
    # Let g_i = 1, then g_q = 10^(-imb/20) for negative imbalance
    amp_ratio = 10.0 ** (amplitude_imbalance_db / 20.0)
    
    # Apply phase imbalance
    theta = np.deg2rad(phase_imbalance_deg)
    
    i = signal.real
    q = signal.imag
    
    # New I remains I
    # New Q is scaled and rotated relative to I
    q_imbal = amp_ratio * (-i * np.sin(theta) + q * np.cos(theta))
    
    return (i + 1j * q_imbal).astype(np.complex64)


def add_phase_noise(signal: np.ndarray, std_dev_deg: float = 2.0) -> np.ndarray:
    """Add white phase noise to the signal."""
    phase_noise_rad = np.deg2rad(std_dev_deg) * np.random.randn(len(signal))
    phasor = np.exp(1j * phase_noise_rad)
    return (signal * phasor).astype(np.complex64)
