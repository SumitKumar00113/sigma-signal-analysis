"""Synthetic signal generators for testing."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def generate_bpsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a BPSK signal.
    
    Returns
    -------
    (samples, bits)
    """
    bits = np.random.randint(0, 2, num_symbols)
    symbols = 2 * bits - 1  # Map to -1, +1
    
    sps = int(sample_rate / symbol_rate)
    
    # Upsample
    upsampled = np.zeros(num_symbols * sps)
    upsampled[::sps] = symbols
    
    # RRC Pulse shaping
    num_taps = 6 * sps + 1
    t = np.arange(num_taps) - (num_taps - 1) // 2
    # Simple raised cosine approximation
    beta = 0.35
    ts = sps
    rc = np.sinc(t / ts) * np.cos(np.pi * beta * t / ts) / (1 - (2 * beta * t / ts)**2 + 1e-10)
    
    sig = np.convolve(upsampled, rc, mode='same').astype(np.complex64)
    
    # Apply frequency offset
    if freq_offset_hz != 0.0:
        t_sec = np.arange(len(sig)) / sample_rate
        mixer = np.exp(1j * 2 * np.pi * freq_offset_hz * t_sec)
        sig = (sig * mixer).astype(np.complex64)
        
    if snr_db is not None:
        from .channels import add_awgn
        sig = add_awgn(sig, snr_db)
        
    return sig, bits


def generate_2fsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    deviation_hz: float,
    snr_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a continuous-phase 2-FSK signal."""
    bits = np.random.randint(0, 2, num_symbols)
    symbols = 2 * bits - 1  # Map to -1, +1
    
    sps = int(sample_rate / symbol_rate)
    
    # NRZ shaping (rect pulse)
    upsampled = np.repeat(symbols, sps)
    
    # Integrate to phase (CPFSK)
    h = 2 * deviation_hz / symbol_rate  # modulation index
    phase_diff = upsampled * (np.pi * h / sps)
    phase = np.cumsum(phase_diff)
    
    sig = np.exp(1j * phase).astype(np.complex64)
    
    if snr_db is not None:
        from .channels import add_awgn
        sig = add_awgn(sig, snr_db)
        
    return sig, bits


def generate_qpsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    snr_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a QPSK signal."""
    bits = np.random.randint(0, 2, num_symbols * 2)
    # Map to complex symbols: 00 -> 1+j, 01 -> -1+j, 10 -> 1-j, 11 -> -1-j
    # Scale by 1/sqrt(2) for unit power
    i_syms = 2 * bits[0::2] - 1
    q_syms = 2 * bits[1::2] - 1
    symbols = (i_syms + 1j * q_syms) / np.sqrt(2)
    
    sps = int(sample_rate / symbol_rate)
    
    # Upsample
    upsampled = np.zeros(num_symbols * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    
    # RRC Pulse shaping
    num_taps = 6 * sps + 1
    t = np.arange(num_taps) - (num_taps - 1) // 2
    beta = 0.35
    rc = np.sinc(t / sps) * np.cos(np.pi * beta * t / sps) / (1 - (2 * beta * t / sps)**2 + 1e-10)
    
    sig = np.convolve(upsampled, rc, mode='same').astype(np.complex64)
    
    if snr_db is not None:
        from .channels import add_awgn
        sig = add_awgn(sig, snr_db)
        
    return sig, bits
